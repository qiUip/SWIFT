/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (C) 2026 SWIFT benchmarking.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

/* Config parameters. */
#include <config.h>

/* Some standard headers. */
#include <fenv.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Local headers. */
#include "argparse.h"
#include "barrier.h"
#include "likwid_wrapper.h"
#include "runner_doiact_grav.h"
#include "swift.h"

#define NODE_ID 0

/* Size of a runner's vectorised hydro cache. Mirrors engine_config.c, where it
 * is a file-local #define and so not reachable from here. */
#define CACHE_SIZE 512

/**
 * @brief Multi-threaded benchmark of a single task type in isolation.
 *
 * Builds a periodic grid of cells and runs one task type over it through the
 * real SWIFT scheduler: one task_type_self per cell plus one dedup'd
 * task_type_pair per unique neighbour pair. The kernel under test is chosen at
 * run time and is the only thing inside the LIKWID-marked "stencil_kernel"
 * region, so the measured arithmetic intensity is that of the task itself.
 *
 * The grid is periodic so that every cell has a full 27-cell neighbourhood and
 * owns exactly the same work; a non-periodic cluster would need ghost cells
 * that are neighbours but never own a task.
 */
enum bench_kernel {
  kernel_hydro_density,
  kernel_gravity_pp_full,
  kernel_gravity_pp_truncated,
  kernel_gravity_m2p,
};

/* All the gravity kernels go through the same entry points; the mesh and
 * allow_mpole decide which body is reached. */
static int is_gravity(const enum bench_kernel kernel) {
  return kernel == kernel_gravity_pp_full ||
         kernel == kernel_gravity_pp_truncated ||
         kernel == kernel_gravity_m2p;
}

/* Wrapped Chebyshev distance between two cells of a periodic grid. */
static int cell_distance(const int i, const int j, const int k, const int ii,
                         const int jj, const int kk, const int cdim[3]) {
  int d[3] = {abs(i - ii), abs(j - jj), abs(k - kk)};
  for (int a = 0; a < 3; ++a)
    if (cdim[a] - d[a] < d[a]) d[a] = cdim[a] - d[a];
  return max3(d[0], d[1], d[2]);
}

/* Which side of runner_iact_grav_pp_full()'s "r2 >= h2" test the pairs take.
 * The two pinned regimes are set by choosing epsilon against the separations
 * the stencil can produce; the mixed one places particles at random instead. */
enum softening_regime {
  softening_newtonian,
  softening_mixed,
  softening_softened,
};

static const char *softening_names[] = {"newtonian", "mixed", "softened"};

enum velocity_types {
  velocity_zero,
  velocity_random,
  velocity_divergent,
  velocity_rotating
};

/* Just a forward declaration... */
void runner_dopair1_branch_density(struct runner *r, struct cell *ci,
                                   struct cell *cj, int limit_h_min,
                                   int limit_h_max);
void runner_doself1_branch_density(struct runner *r, struct cell *c,
                                   int limit_h_min, int limit_h_max);

/*******************************************************************************
 * Cell construction and teardown
 ******************************************************************************/

/**
 * @brief Whether a cell owns tasks (i.e. whether its whole neighbourhood
 * exists). Halo cells for non-periodic kernels would be excluded.
 */
static int cell_owns_tasks(const int i, const int j, const int k,
                           const int cdim[3], const int halo) {
  return i >= halo && i < cdim[0] - halo && j >= halo && j < cdim[1] - halo &&
         k >= halo && k < cdim[2] - halo;
}

/**
 * @brief Fills in the geometry and scheduler fields of one cell of the
 * contiguous top-level array, i.e. those that do not depend on which particle
 * type it will hold. Mirrors space_regrid()'s per-cell setup.
 */
void make_cell_common(struct cell *cell, const double loc[3], double width) {

  cell->split = 0;
  cell->depth = 0;
  cell->width[0] = width;
  cell->width[1] = width;
  cell->width[2] = width;
  cell->dmin = width;
  cell->loc[0] = loc[0];
  cell->loc[1] = loc[1];
  cell->loc[2] = loc[2];
  cell->nodeID = NODE_ID;

  /* Owner must start at -1 so the "qid < 0 -> random queue" instead of pinning
   * every cell to queue 0. */
  cell->top = cell;
  cell->super = cell;
  cell->owner = -1;

  if (lock_init(&cell->hydro.lock) != 0) error("Failed to init hydro lock.");
  if (lock_init(&cell->grav.plock) != 0) error("Failed to init grav plock.");
  if (lock_init(&cell->grav.mlock) != 0) error("Failed to init grav mlock.");
}

/**
 * @brief Constructs a cell of parts in a valid state prior to a DOPAIR or
 * DOSELF density calculation.
 */
void make_cell_hydro(struct cell *cell, size_t n, const double offset[3],
                     double size, double h, double density, long long *partId,
                     double pert, enum velocity_types vel, double h_pert) {
  const size_t count = n * n * n;
  const double volume = size * size * size;
  float h_max = 0.f;
  make_cell_common(cell, offset, size);

  if (posix_memalign((void **)&cell->hydro.parts, part_align,
                     count * sizeof(struct part)) != 0) {
    error("couldn't allocate particles, no. of particles: %d", (int)count);
  }
  bzero(cell->hydro.parts, count * sizeof(struct part));

  /* Construct the parts */
  struct part *part = cell->hydro.parts;
  for (size_t x = 0; x < n; ++x) {
    for (size_t y = 0; y < n; ++y) {
      for (size_t z = 0; z < n; ++z) {
        part->x[0] =
            offset[0] +
            size * (x + 0.5 + random_uniform(-0.5, 0.5) * pert) / (float)n;
        part->x[1] =
            offset[1] +
            size * (y + 0.5 + random_uniform(-0.5, 0.5) * pert) / (float)n;
        part->x[2] =
            offset[2] +
            size * (z + 0.5 + random_uniform(-0.5, 0.5) * pert) / (float)n;
        switch (vel) {
          case velocity_zero:
            part->v[0] = 0.f;
            part->v[1] = 0.f;
            part->v[2] = 0.f;
            break;
          case velocity_random:
            part->v[0] = random_uniform(-0.05, 0.05);
            part->v[1] = random_uniform(-0.05, 0.05);
            part->v[2] = random_uniform(-0.05, 0.05);
            break;
          case velocity_divergent:
            part->v[0] = part->x[0] - 1.5 * size;
            part->v[1] = part->x[1] - 1.5 * size;
            part->v[2] = part->x[2] - 1.5 * size;
            break;
          case velocity_rotating:
            part->v[0] = part->x[1];
            part->v[1] = -part->x[0];
            part->v[2] = 0.f;
            break;
        }
        if (h_pert)
          part->h = size * h * random_uniform(1.f, h_pert) / (float)n;
        else
          part->h = size * h / (float)n;
        h_max = fmaxf(h_max, part->h);
        part->id = ++(*partId);
        part->depth_h = 0;

#if defined(GIZMO_MFV_SPH) || defined(GIZMO_MFM_SPH)
        part->conserved.mass = density * volume / count;
#else
        part->mass = density * volume / count;
#endif
#if defined(REMIX_SPH)
        part->rho_evol = density;
#endif
#if defined(HOPKINS_PE_SPH)
        part->entropy = 1.f;
        part->entropy_one_over_gamma = 1.f;
#endif

        part->time_bin = 1;

#ifdef SWIFT_DEBUG_CHECKS
        part->ti_drift = 8;
        part->ti_kick = 8;
#endif

        ++part;
      }
    }
  }

  /* Cell properties */
  cell->hydro.h_max = h_max;
  cell->hydro.h_max_active = h_max;
  cell->hydro.count = count;
  cell->hydro.dx_max_part = 0.;
  cell->hydro.dx_max_sort = 0.;
  cell->h_min_allowed = cell->dmin * 0.5 * (1. / kernel_gamma);
  cell->h_max_allowed = cell->dmin * (1. / kernel_gamma);

  cell->hydro.super = cell;
  cell->hydro.ti_old_part = 8;
  cell->hydro.ti_end_min = 8;

  shuffle_particles(cell->hydro.parts, cell->hydro.count);

  cell->hydro.sorted = 0;
  cell->hydro.sort = NULL;
}

/**
 * @brief Constructs a cell of gparts, on a perturbed Cartesian grid or, if
 * random_pos, uniformly at random within the cell.
 *
 * The grid bounds the closest approach of any pair, which is what lets the
 * Newtonian and softened regimes be pinned; random positions do not, and are
 * used for the mixed regime.
 */
void make_cell_gravity(struct cell *cell, struct gravity_tensors *multipole,
                       size_t n, const double offset[3], double size,
                       double pert, double epsilon, int random_pos,
                       long long *partId,
                       const struct gravity_props *grav_props) {
  const size_t count = n * n * n;
  const double cell_size = size / (double)n;
  make_cell_common(cell, offset, size);

  if (posix_memalign((void **)&cell->grav.parts, gpart_align,
                     count * sizeof(struct gpart)) != 0)
    error("Couldn't allocate gparts, no. of gparts: %d", (int)count);
  bzero(cell->grav.parts, count * sizeof(struct gpart));

  struct gpart *part = cell->grav.parts;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < n; ++k) {

        if (random_pos) {
          part->x[0] = offset[0] + size * random_uniform(0., 1.);
          part->x[1] = offset[1] + size * random_uniform(0., 1.);
          part->x[2] = offset[2] + size * random_uniform(0., 1.);
        } else {
          part->x[0] = offset[0] +
                       cell_size * (i + 0.5 + pert * random_uniform(-0.5, 0.5));
          part->x[1] = offset[1] +
                       cell_size * (j + 0.5 + pert * random_uniform(-0.5, 0.5));
          part->x[2] = offset[2] +
                       cell_size * (k + 0.5 + pert * random_uniform(-0.5, 0.5));
        }

        part->mass = 1.f / (float)count;
        part->epsilon = epsilon;
        part->type = swift_type_dark_matter;
        part->time_bin = 1;
        part->id_or_neg_offset = ++(*partId);

#ifdef SWIFT_DEBUG_CHECKS
        part->ti_drift = 8;
        part->initialised = 1;
#endif

        ++part;
      }
    }
  }

  /* Cell properties */
  cell->grav.count = count;
  cell->grav.count_total = count;
  cell->grav.super = cell;

  /* These MUST agree with engine.ti_current or cell_is_active_gravity()
   * returns 0 and the kernel silently early-outs. */
  cell->grav.ti_old_part = 8;
  cell->grav.ti_old_multipole = 8;
  cell->grav.ti_end_min = 8;

  /* The multipoles are dereferenced by the P-P kernel before any branch, even
   * with allow_mpole=0. */
  cell->grav.multipole = multipole;
  gravity_reset(cell->grav.multipole);
  gravity_P2M(cell->grav.multipole, cell->grav.parts, count, grav_props);
  gravity_multipole_compute_power(&cell->grav.multipole->m_pole);
}

/**
 * @brief Initialises all particle fields to be ready for a calculation.
 */
void zero_particle_fields(struct cell *c, enum bench_kernel kernel) {
  if (kernel == kernel_hydro_density) {
    struct hydro_space *hspointer = NULL;
    for (int pid = 0; pid < c->hydro.count; pid++) {
      hydro_init_part(&c->hydro.parts[pid], hspointer);
      adaptive_softening_init_part(&c->hydro.parts[pid]);
      mhd_init_part(&c->hydro.parts[pid]);
    }
  } else {
    for (int pid = 0; pid < c->grav.count; pid++)
      gravity_init_gpart(&c->grav.parts[pid]);
  }
}

void clean_up(struct cell *c, enum bench_kernel kernel) {
  if (kernel == kernel_hydro_density) {
    free(c->hydro.parts);
    free(c->hydro.sort);
  } else {
    free(c->grav.parts);
  }
}

/*******************************************************************************
 * Worker thread
 ******************************************************************************/

/**
 * @brief The #runner thread routine: waits to be released for a run, drains
 * the scheduler's queue, then reports back and waits for the next one. Mirrors
 * runner_main() over a persistent e->runners[]-style pool created once, rather
 * than a thread pthread_create'd per run.
 */
void *test_runner(void *data) {

  struct runner *r = (struct runner *)data;
  struct engine *e = r->e;
  struct scheduler *sched = &e->sched;

  while (1) {

    /* Report done with the previous run, then wait to be released for the
     * next one. */
    engine_barrier(e);

    if (e->step_props & engine_step_prop_done) break;

    swift_likwid_marker_start_region("stencil_full");

    struct task *t = NULL;
    struct task *prev = NULL;
    while (1) {
      if (t == NULL) {
        t = scheduler_gettask(sched, r->qid, prev);
        if (t == NULL) break;
      }

      swift_likwid_marker_start_region("stencil_kernel");

      switch (t->type) {

        case task_type_self:
          if (t->subtype == task_subtype_grav)
            runner_doself_grav_pp(r, t->ci);
          else
            runner_doself1_branch_density(r, t->ci, /*limit_h_min=*/0,
                                          /*limit_h_max=*/0);
          break;

        case task_type_pair:
          if (t->subtype == task_subtype_grav)
            runner_dopair_grav_pp(r, t->ci, t->cj, /*symmetric=*/1,
                                  e->gravity_properties->theta_crit > 0.);
          else
            runner_dopair1_branch_density(r, t->ci, t->cj, /*limit_h_min=*/0,
                                          /*limit_h_max=*/0);
          break;

        default:
          error("Unknown/invalid task type (%d).", t->type);
      }

      swift_likwid_marker_stop_region("stencil_kernel");

      prev = t;
      t = scheduler_done(sched, t);
    }

    swift_likwid_marker_stop_region("stencil_full");
  }

  return NULL;
}

/*******************************************************************************
 * main
 ******************************************************************************/

/* And go... */
int main(int argc, char *argv[]) {

#ifdef HAVE_SETAFFINITY
  engine_pin();
#endif

  int with_hydro_density = 0, with_gravity_pp_full = 0;
  int with_gravity_pp_truncated = 0, with_gravity_m2p = 0;
  int shell = 2;
  float theta = 0.7f;
  int particles = 0, runs = 0, threads = 1, vel = velocity_zero;
  int cdim[3] = {0, 0, 0};
  const char *cdim_str = NULL, *softening_str = NULL;
  float eta = 1.23485f, size = 1.f, rho = 1.f;
  float perturbation = -1.f, h_pert = 0.f, epsilon = -1.f;

  /* Initialize CPU frequency, this also starts time. */
  unsigned long long cpufreq = 0;
  clocks_set_cpufreq(cpufreq);

/* Choke on FP-exceptions */
#ifdef HAVE_FE_ENABLE_EXCEPT
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  /* Get some randomness going */
  srand(0);

  const char *const usages[] = {
      "testCellsMulti <kernel> -n PARTICLES_PER_AXIS -r RUNS [options]",
      NULL,
  };

  struct argparse_option options[] = {
      OPT_HELP(),
      OPT_GROUP("Kernel under test (exactly one is required)"),
      OPT_BOOLEAN(0, "hydro-density", &with_hydro_density,
                  "Benchmark runner_do{self,pair}1_branch_density.", NULL, 0,
                  0),
      OPT_BOOLEAN(0, "gravity-pp-full", &with_gravity_pp_full,
                  "Benchmark runner_do{self,pair}_grav_pp, full Newtonian "
                  "P-P with allow_mpole=0.",
                  NULL, 0, 0),
      OPT_BOOLEAN(0, "gravity-pp-truncated", &with_gravity_pp_truncated,
                  "Benchmark runner_do{self,pair}_grav_pp, truncated "
                  "long-range P-P with allow_mpole=0.",
                  NULL, 0, 0),
      OPT_BOOLEAN(0, "gravity-m2p", &with_gravity_m2p,
                  "Benchmark runner_dopair_grav_pm_truncated, the "
                  "particle-multipole interaction, with no P-P fallback.",
                  NULL, 0, 0),
      OPT_GROUP("Benchmark parameters"),
      OPT_INTEGER('n', "particles", &particles,
                  "Particles per axis in each cell.", NULL, 0, 0),
      OPT_INTEGER('r', "runs", &runs, "Number of runs.", NULL, 0, 0),
      OPT_INTEGER('t', "threads", &threads, "Number of worker threads.", NULL,
                  0, 0),
      OPT_STRING('c', "cdim", &cdim_str,
                 "Cells per axis of the periodic grid: CDIM or CDIM_X,CDIM_Y,"
                 "CDIM_Z. Default is the smallest cube >= 3 holding THREADS.",
                 NULL, 0, 0),
      OPT_FLOAT('s', "size", &size, "Physical size of a cell.", NULL, 0, 0),
      OPT_FLOAT('d', "perturbation", &perturbation,
                "Particle perturbation in [0,1[. Default 0 for hydro, 0.1 "
                "for gravity.",
                NULL, 0, 0),
      OPT_GROUP("Hydro only"),
      OPT_FLOAT(0, "eta", &eta,
                "Smoothing length in units of inter-particle separation.", NULL,
                0, 0),
      OPT_FLOAT('m', "rho", &rho, "Physical density in the cell.", NULL, 0, 0),
      OPT_FLOAT('p', "h-pert", &h_pert,
                "Random fractional change in h, h=h*random(1,p).", NULL, 0, 0),
      OPT_INTEGER('v', "velocity", &vel,
                  "Velocity field: 0 zero, 1 random, 2 divergent, 3 rotating.",
                  NULL, 0, 0),
      OPT_GROUP("Gravity only"),
      OPT_STRING(0, "softening", &softening_str,
                 "Which side of the kernel's r2 >= h2 test the pairs take: "
                 "newtonian (all), mixed (default, random positions, both) or "
                 "softened (none).",
                 NULL, 0, 0),
      OPT_INTEGER(0, "shell", &shell,
                  "--gravity-m2p only: interact cells 2..SHELL apart. Default "
                  "2.",
                  NULL, 0, 0),
      OPT_FLOAT(0, "theta", &theta,
                "--gravity-m2p only: geometric MAC opening angle. Default 0.7.",
                NULL, 0, 0),
      OPT_FLOAT('e', "epsilon", &epsilon,
                "Softening length. Default is chosen to pin the regime.", NULL,
                0, 0),
      OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usages, 0);
  argparse_describe(&argparse,
                    "\nBenchmarks a single SWIFT task type in isolation over a "
                    "periodic grid of cells, driven by the real task "
                    "scheduler on THREADS worker threads.",
                    NULL);
  argc = argparse_parse(&argparse, argc, (const char **)argv);

  /* There is no default kernel: the choice must always be explicit. */
  const int nr_kernels = with_hydro_density + with_gravity_pp_full +
                         with_gravity_pp_truncated + with_gravity_m2p;
  if (nr_kernels == 0)
    error(
        "No kernel selected. Specify exactly one of --hydro-density, "
        "--gravity-pp-full, --gravity-pp-truncated or --gravity-m2p.");
  if (nr_kernels > 1) error("The kernel options are mutually exclusive.");

  enum bench_kernel kernel = kernel_gravity_m2p;
  if (with_hydro_density)
    kernel = kernel_hydro_density;
  else if (with_gravity_pp_full)
    kernel = kernel_gravity_pp_full;
  else if (with_gravity_pp_truncated)
    kernel = kernel_gravity_pp_truncated;

  /* Gravity:r_cut_min defaults to 0, so a periodic run always takes the
   * truncated kernel; _full is reached only through the non-periodic branch.
   * Non-periodic needs a halo of cells that are neighbours but own no task,
   * one layer per cell of stencil reach. */
  const int periodic = (kernel != kernel_gravity_pp_full);

  /* How far, in cells, a task reaches. M2P skips the self and the direct
   * neighbours so that no particle can fall back to P-P. */
  const int reach = (kernel == kernel_gravity_m2p) ? shell : 1;
  const int near = (kernel == kernel_gravity_m2p) ? 2 : 1;
  const int halo = periodic ? 0 : reach;

  if (kernel == kernel_gravity_m2p) {
    if (shell < 2) error("--shell must be >= 2 for --gravity-m2p.");
    if (theta <= 0.f) error("--theta must be > 0.");
  }

  /* -d perturbs the Cartesian grid, used by hydro and by the two pinned
   * softening regimes; only --softening mixed places particles at random.
   * Default as the single-cell tests do, 0 for hydro and 0.1 for gravity. */
  if (perturbation < 0.f)
    perturbation = is_gravity(kernel) ? 0.1f : 0.f;
  else if (perturbation >= 1.f)
    error("-d PERTURBATION must be in [0,1[.");

  if (particles <= 0 || runs <= 0) {
    argparse_usage(&argparse);
    error("-n PARTICLES_PER_AXIS and -r RUNS must both be > 0.");
  }
  if (threads <= 0) error("-t THREADS must be >= 1.");

  enum softening_regime softening = softening_mixed;
  if (softening_str != NULL) {
    if (strcmp(softening_str, "newtonian") == 0)
      softening = softening_newtonian;
    else if (strcmp(softening_str, "mixed") == 0)
      softening = softening_mixed;
    else if (strcmp(softening_str, "softened") == 0)
      softening = softening_softened;
    else
      error("--softening must be one of newtonian, mixed or softened.");
  }

  if (epsilon == 0.f) error("-e EPSILON must be > 0.");

  /* Warn rather than silently ignore options the chosen kernel never reads. */
  if (kernel == kernel_hydro_density &&
      (epsilon > 0.f || softening_str != NULL))
    message(
        "WARNING: --epsilon and --softening are ignored by "
        "--hydro-density.");
  if (is_gravity(kernel) &&
      (eta != 1.23485f || rho != 1.f || h_pert != 0.f || vel != velocity_zero))
    message(
        "WARNING: --eta, --rho, --h-pert and --velocity are ignored by the "
        "gravity kernels.");

  /* Periodic needs 2*reach+1 cells per axis for the furthest neighbour not to
   * alias onto itself; non-periodic needs the reach as halo on each side of at
   * least three owning cells. */
  const int cdim_min = periodic ? 2 * reach + 1 : 2 * halo + 3;

  /* Default CDIM sizes the smallest cube with (CDIM-2*halo)^3 >= THREADS. An
   * explicit -c is used as given, subject to that minimum. */
  if (cdim_str != NULL) {
    int x = 0, y = 0, z = 0;
    const int n_dims = sscanf(cdim_str, "%d,%d,%d", &x, &y, &z);
    if (n_dims == 1) {
      cdim[0] = cdim[1] = cdim[2] = x;
    } else if (n_dims == 3) {
      cdim[0] = x;
      cdim[1] = y;
      cdim[2] = z;
    } else {
      error("-c must be CDIM or CDIM_X,CDIM_Y,CDIM_Z.");
    }
    if (cdim[0] < cdim_min || cdim[1] < cdim_min || cdim[2] < cdim_min)
      error("-c CDIM_X,CDIM_Y,CDIM_Z must each be >= %d for this kernel.",
            cdim_min);
  } else {
    int default_cdim = (int)ceil(cbrt((double)threads)) + 2 * halo;
    if (default_cdim < cdim_min) default_cdim = cdim_min;
    cdim[0] = cdim[1] = cdim[2] = default_cdim;
  }

  const int total_cells = cdim[0] * cdim[1] * cdim[2];
  const int count = particles * particles * particles;

  /* The extremes of pair separation the stencil can produce, which is what
   * the epsilon pins are checked against. Random positions have no lower
   * bound. */
  const double cell_size = size / (double)particles;
  int min_cdim = cdim[0];
  if (cdim[1] < min_cdim) min_cdim = cdim[1];
  if (cdim[2] < min_cdim) min_cdim = cdim[2];
  const double span = (reach + 1.) * size;
  const double max_sep =
      sqrt(3.) * (periodic ? fmin(span, 0.5 * (double)min_cdim * size) : span);
  const double min_sep = cell_size * (1. - perturbation);

  /* Unless -e was given, pick a softening that pins the requested regime. */
  if (epsilon < 0.f) {
    switch (softening) {
      case softening_newtonian:
        epsilon = 1e-4f;
        break;
      case softening_mixed:
        epsilon = (float)cell_size;
        break;
      case softening_softened:
        epsilon = (float)(1.1 * max_sep);
        break;
    }
  }

  if (is_gravity(kernel)) {
    if (softening == softening_newtonian && epsilon >= min_sep)
      error(
          "epsilon = %e does not pin the Newtonian branch: the closest pair "
          "is %e apart. Lower -e, raise -n, or lower -d.",
          epsilon, min_sep);
    if (softening == softening_softened && epsilon <= max_sep)
      error(
          "epsilon = %e does not pin the softened branch: pairs reach %e "
          "apart. Raise -e.",
          epsilon, max_sep);

    /* gravity_M2P_accept() also requires max_softening^2 < r2, which a
     * softened-regime epsilon fails everywhere. */
    if (kernel == kernel_gravity_m2p && softening == softening_softened)
      error("--gravity-m2p and --softening softened are incompatible.");
  }

  /* Gravity:r_cut_min defaults to 0 (src/gravity_properties.c), so in a
   * periodic run every P-P interaction is truncated: max_r > 0 always holds
   * for both entry points. r_s is set so the stencil spans the same reach a
   * production run gives the kernel, r_cut_max = 4.5 * r_s. */
  const double r_cut_min = 0.;
  const double r_s = max_sep / 4.5;

  /* engine_config() sizes the gravity caches on space_splitsize, which is safe
   * in production because cells are split until they hold at most that many
   * particles. Our cells are never split, so raise the parameter to match, as
   * a run with Scheduler:cell_split_size would; otherwise
   * gravity_cache_populate() would reallocate inside the measured region. */
  if (space_splitsize < count + VEC_SIZE) space_splitsize = count + VEC_SIZE;

#ifdef WITH_VECTORIZATION
  /* CACHE_SIZE has no such parameter: it is a compile-time constant that the
   * same splitting keeps cells below. Here it is a hard limit on -n. */
  if (kernel == kernel_hydro_density && count > CACHE_SIZE)
    error(
        "-n %d gives %d particles per cell, above CACHE_SIZE (%d): the "
        "vectorised density kernel would resize its cache inside the measured "
        "region.",
        particles, count, CACHE_SIZE);
#endif

  /* Help users... */
  if (kernel == kernel_hydro_density) {
    message("Kernel: runner_do{self,pair}1_branch_density");
#if defined(SWIFT_USE_NAIVE_INTERACTIONS)
    message("Branch: naive (SWIFT_USE_NAIVE_INTERACTIONS)");
#elif defined(WITH_VECTORIZATION) && defined(GADGET2_SPH)
    message(
        "Branch: vectorised, except the 8 corner sids which fall back to "
        "scalar (sort_is_corner)");
#else
    message("Branch: scalar, sorted");
#endif
    message("Hydro implementation: %s", SPH_IMPLEMENTATION);
    message("Kernel function: %s", kernel_name);
    message("Vector size: %d", VEC_SIZE);
    message("Adiabatic index: ga = %f", hydro_gamma);
    message("Smoothing length: h = %f", eta * size);
    message("Neighbour target: N = %f", pow_dimension(eta) * kernel_norm);
    message("Density target: rho = %f", rho);
    message("part size: %zu B", sizeof(struct part));
  } else {
    if (kernel == kernel_gravity_m2p)
      message(
          "Kernel: runner_dopair_grav_pm_truncated (symmetric=1, "
          "allow_mpole=1), no self tasks");
    else
      message(
          "Kernel: runner_do{self,pair}_grav_pp (symmetric=1, allow_mpole=0)");
    if (kernel == kernel_gravity_pp_full)
      message(
          "Branch: _full, taken by the non-periodic arm of the dispatcher");
    else
      message(
          "Branch: _truncated, r_cut_min = %e (r_s = %e, r/2r_s up to %.2f), "
          "periodic mesh",
          r_cut_min, r_s, 0.5 * max_sep / r_s);
    if (kernel == kernel_gravity_m2p)
      message("MAC: geometric, theta_crit = %f, cells %d to %d apart",
              (double)theta, near, reach);
    message("Multipole order: %d", SELF_GRAVITY_MULTIPOLE_ORDER);
    message("Softening regime: %s", softening_names[softening]);
    if (softening == softening_mixed)
      message(
          "Softening: epsilon = %e (random positions, separations up to %e)",
          epsilon, max_sep);
    else
      message("Softening: epsilon = %e (separations %e to %e)", epsilon,
              min_sep, max_sep);
    message("gpart size: %zu B", sizeof(struct gpart));
  }
  message("Worker threads:  t = %d", threads);
  message("Boundary: %s", periodic ? "periodic" : "non-periodic");
  message("Grid: cdim = %d x %d x %d (%d cells, %d particles per cell)",
          cdim[0], cdim[1], cdim[2], total_cells, count);
  if (halo > 0)
    message("Owning cells: %d of %d (%d halo layer(s), neighbours only)",
            (cdim[0] - 2 * halo) * (cdim[1] - 2 * halo) *
                (cdim[2] - 2 * halo),
            total_cells, halo);
  printf("\n");

  /* Build the infrastructure */
  const double dim[3] = {(double)cdim[0] * size, (double)cdim[1] * size,
                         (double)cdim[2] * size};

  struct space space;
  bzero(&space, sizeof(struct space));
  space.periodic = periodic;
  space.dim[0] = dim[0];
  space.dim[1] = dim[1];
  space.dim[2] = dim[2];

  struct hydro_props hp;
  hydro_props_init_no_hydro(&hp);
  hp.eta_neighbours = eta;
  hp.h_tolerance = 1e0;
  hp.h_max = FLT_MAX;
  hp.max_smoothing_iterations = 1;
  hp.CFL_condition = 0.1;

  struct gravity_props gravity_props;
  bzero(&gravity_props, sizeof(struct gravity_props));
  gravity_props.use_advanced_MAC = 0;
  gravity_props.use_adaptive_tolerance = 0;
  gravity_props.theta_crit = (kernel == kernel_gravity_m2p) ? theta : 0.;
  gravity_props.G_Newton = 1.;
  gravity_props.epsilon_DM_cur = epsilon;
  gravity_props.epsilon_baryon_cur = epsilon;

  /* pm_mesh_init_no_mesh() leaves r_cut_min = FLT_MAX, which no separation can
   * exceed, so the P-P kernels take the _full branch whatever CDIM is; the
   * truncated kernel lowers it instead. periodic = 1 gives the correct wrapped
   * separations, applied per interaction inside the kernel. Nothing here may
   * touch the FFT machinery. */
  struct pm_mesh mesh;
  pm_mesh_init_no_mesh(&mesh, (double *)dim);
  mesh.periodic = periodic;
  if (periodic && is_gravity(kernel)) {
    mesh.r_s = r_s;
    mesh.r_s_inv = 1. / r_s;
    mesh.r_cut_min = r_cut_min;
  }

  struct engine engine;
  bzero(&engine, sizeof(struct engine));
  engine.s = &space;
  engine.time = 0.1f;
  engine.ti_current = 8;
  engine.max_active_bin = num_time_bins;
  engine.hydro_properties = &hp;
  engine.gravity_properties = &gravity_props;
  engine.mesh = &mesh;
  engine.nodeID = NODE_ID;

  struct phys_const prog_const;
  prog_const.const_vacuum_permeability = 1.0;
  engine.physical_constants = &prog_const;

  struct cosmology cosmo;
  cosmology_init_no_cosmo(&cosmo);
  engine.cosmology = &cosmo;

  struct sink_props sink_props;
  bzero(&sink_props, sizeof(struct sink_props));
  engine.sink_properties = &sink_props;

  struct lightcone_array_props lightcone_array_properties;
  lightcone_array_properties.nr_lightcones = 0;
  engine.lightcone_array_properties = &lightcone_array_properties;

  struct pressure_floor_props pressure_floor;
  engine.pressure_floor_props = &pressure_floor;

  space.e = &engine;

  struct runner runner;
  bzero(&runner, sizeof(struct runner));
  runner.e = &engine;

  /* One contiguous array of top-level cells, and one of multipoles, exactly as
   * space_regrid() lays them out: the cell metadata a task walks is then the
   * same working set it would be in a real run. */
  struct cell *cells = NULL;
  if (swift_memalign("cells_top", (void **)&cells, cell_align,
                     total_cells * sizeof(struct cell)) != 0)
    error("Failed to allocate top-level cells.");
  bzero(cells, total_cells * sizeof(struct cell));

  struct gravity_tensors *multipoles = NULL;
  if (is_gravity(kernel)) {
    if (swift_memalign("multipoles_top", (void **)&multipoles, multipole_align,
                       total_cells * sizeof(struct gravity_tensors)) != 0)
      error("Failed to allocate top-level multipoles.");
    bzero(multipoles, total_cells * sizeof(struct gravity_tensors));
  }

  space.cells_top = cells;
  space.nr_cells = total_cells;
  space.tot_cells = total_cells;
  for (int l = 0; l < 3; ++l) {
    space.cdim[l] = cdim[l];
    space.width[l] = size;
    space.iwidth[l] = 1. / size;
  }

  /* Construct the periodic grid of cells */
  static long long partId = 0;
  for (int i = 0; i < cdim[0]; ++i) {
    for (int j = 0; j < cdim[1]; ++j) {
      for (int k = 0; k < cdim[2]; ++k) {
        const double offset[3] = {i * size, j * size, k * size};
        const int cid = cell_getid(cdim, i, j, k);

        if (kernel == kernel_hydro_density) {
          make_cell_hydro(&cells[cid], particles, offset, size, eta, rho,
                          &partId, perturbation, vel, h_pert);
          runner_do_drift_part(&runner, &cells[cid], 0);
          runner_do_hydro_sort(&runner, &cells[cid], 0x1FFF, 0, 0, 0, 0);
        } else {
          make_cell_gravity(&cells[cid], &multipoles[cid], particles, offset,
                            size, perturbation, epsilon,
                            softening == softening_mixed, &partId,
                            &gravity_props);
        }
      }
    }
  }

  /* Every particle must accept the MAC: a single fall-back to P-P brings
   * O(n^2) work into an O(n) measurement. Check the real r_max against the
   * closest point of each cell, which no particle can beat. */
  if (kernel == kernel_gravity_m2p) {

    double worst = 0.;
    for (int i = 0; i < cdim[0]; ++i) {
      for (int j = 0; j < cdim[1]; ++j) {
        for (int k = 0; k < cdim[2]; ++k) {

          const double centre[3] = {(i + 0.5) * size, (j + 0.5) * size,
                                    (k + 0.5) * size};

          for (int ii = -reach; ii <= reach; ++ii) {
            const int iii = (i + ii + cdim[0]) % cdim[0];
            for (int jj = -reach; jj <= reach; ++jj) {
              const int jjj = (j + jj + cdim[1]) % cdim[1];
              for (int kk = -reach; kk <= reach; ++kk) {
                const int kkk = (k + kk + cdim[2]) % cdim[2];
                if (cell_distance(i, j, k, iii, jjj, kkk, cdim) < near)
                  continue;

                const struct gravity_tensors *m =
                    cells[cell_getid(cdim, iii, jjj, kkk)].grav.multipole;

                /* Closest approach of this cell's box to that CoM. */
                double r2 = 0.;
                for (int a = 0; a < 3; ++a) {
                  double d = nearest(m->CoM[a] - centre[a], dim[a]);
                  d = fabs(d) - 0.5 * size;
                  if (d > 0.) r2 += d * d;
                }
                if (r2 <= 0.) error("Cell overlaps a source centre of mass.");

                const double ratio = m->r_max / sqrt(r2);
                if (ratio > worst) worst = ratio;
              }
            }
          }
        }
      }
    }

    if (worst >= theta)
      error(
          "--theta %f cannot accept every particle: the worst cell needs "
          "theta > %f. Raise --theta, or --shell to start further out.",
          (double)theta, worst);
  }

  /* Build the task graph once; each run just reactivates it. One
   * task_type_self per cell plus one dedup'd task_type_pair per unique
   * neighbour pair (cid < cjd), mirroring
   * engine_make_hydroloop_tasks_mapper's rule in src/engine_maketasks.c. */
  const enum task_subtypes subtype = (kernel == kernel_hydro_density)
                                         ? task_subtype_density
                                         : task_subtype_grav;
  const int span_cells = 2 * reach + 1;
  const int nr_tasks =
      total_cells * (span_cells * span_cells * span_cells / 2 + 1);
  struct task **tasks = malloc(sizeof(struct task *) * nr_tasks);
  int nr_tasks_built = 0, nr_self_tasks = 0;

  threadpool_init(&engine.threadpool, threads);
  scheduler_init(&engine.sched, &space, nr_tasks, threads, scheduler_flag_steal,
                 NODE_ID, &engine.threadpool);

  for (int i = 0; i < cdim[0]; ++i) {
    for (int j = 0; j < cdim[1]; ++j) {
      for (int k = 0; k < cdim[2]; ++k) {
        const int cid = cell_getid(cdim, i, j, k);
        const int ci_owns = cell_owns_tasks(i, j, k, cdim, halo);

        if (ci_owns && kernel != kernel_gravity_m2p) {
          tasks[nr_tasks_built++] = scheduler_addtask(
              &engine.sched, task_type_self, subtype, 0, 0, &cells[cid], NULL);
          ++nr_self_tasks;
        }

        for (int ii = -reach; ii <= reach; ++ii) {
          const int iii = periodic ? (i + ii + cdim[0]) % cdim[0] : i + ii;
          if (iii < 0 || iii >= cdim[0]) continue;
          for (int jj = -reach; jj <= reach; ++jj) {
            const int jjj = periodic ? (j + jj + cdim[1]) % cdim[1] : j + jj;
            if (jjj < 0 || jjj >= cdim[1]) continue;
            for (int kk = -reach; kk <= reach; ++kk) {
              const int kkk = periodic ? (k + kk + cdim[2]) % cdim[2] : k + kk;
              if (kkk < 0 || kkk >= cdim[2]) continue;

              /* M2P interacts only cells far enough apart that every particle
               * accepts the MAC; anything nearer would fall back to P-P. */
              if (cell_distance(i, j, k, iii, jjj, kkk, cdim) < near) continue;

              const int cjd = cell_getid(cdim, iii, jjj, kkk);
              if (cid >= cjd) continue;

              /* A pair of two halo cells updates nothing anyone reads. */
              if (!ci_owns && !cell_owns_tasks(iii, jjj, kkk, cdim, halo))
                continue;

              tasks[nr_tasks_built++] =
                  scheduler_addtask(&engine.sched, task_type_pair, subtype, 0,
                                    0, &cells[cid], &cells[cjd]);
            }
          }
        }
      }
    }
  }

  message("Tasks: %d (%d self, %d pair)", nr_tasks_built, nr_self_tasks,
          nr_tasks_built - nr_self_tasks);

  ticks time = 0;

  swift_likwid_marker_init();
  swift_likwid_marker_register("stencil_full");
  swift_likwid_marker_register("stencil_kernel");

  /* Persistent worker pool, mirroring engine_config()'s e->runners[]: each
   * thread is created once and woken per run via the engine's barriers. */
  if (swift_barrier_init(&engine.wait_barrier, NULL, threads + 1) != 0 ||
      swift_barrier_init(&engine.run_barrier, NULL, threads + 1) != 0)
    error("Failed to initialize barriers.");

  engine.step_props = engine_step_prop_none;

  struct runner *runners = NULL;
  if (swift_memalign("runners", (void **)&runners, SWIFT_CACHE_ALIGNMENT,
                     threads * sizeof(struct runner)) != 0)
    error("Failed to allocate runners.");
  bzero(runners, threads * sizeof(struct runner));

  /* The cores we may use, in the order engine_config() walks them. The NUMA
   * reordering it does under libnuma is not reproduced here. */
#ifdef HAVE_SETAFFINITY
  cpu_set_t *entry_affinity = engine_entry_affinity();
  const int nr_affinity_cores = CPU_COUNT(entry_affinity);
  int *cpuid = (int *)malloc(nr_affinity_cores * sizeof(int));
  if (cpuid == NULL) error("Failed to allocate the cpuid array.");
  int skip = 0;
  for (int k = 0; k < nr_affinity_cores; k++) {
    int c;
    for (c = skip; c < CPU_SETSIZE && !CPU_ISSET(c, entry_affinity); ++c) {
      /* Nothing to do here */
    }
    cpuid[k] = c;
    skip = c + 1;
  }

  /* The cells are allocated and first-touched by now, so release this thread
   * onto the full mask before it spawns anything, as engine_config() does. */
  engine_unpin();
#endif

  for (int k = 0; k < threads; k++) {
    runners[k].id = k;
    runners[k].e = &engine;
    runners[k].cpuid = k;
    runners[k].qid = k;

    if (pthread_create(&runners[k].thread, NULL, &test_runner, &runners[k]) !=
        0)
      error("Failed to create worker thread.");

      /* Pin the worker to one core from the parent thread, as engine_config()
       * does for '-a'. */
#ifdef HAVE_SETAFFINITY
    const int coreid = k % nr_affinity_cores;
    runners[k].cpuid = cpuid[coreid];

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuid[coreid], &cpuset);
    if (pthread_setaffinity_np(runners[k].thread, sizeof(cpu_set_t), &cpuset) !=
        0)
      error("Failed to set worker thread affinity.");
#endif

    /* Allocate particle caches. */
    runners[k].ci_gravity_cache.count = 0;
    runners[k].cj_gravity_cache.count = 0;
    gravity_cache_init(&runners[k].ci_gravity_cache, space_splitsize);
    gravity_cache_init(&runners[k].cj_gravity_cache, space_splitsize);
#ifdef WITH_VECTORIZATION
    runners[k].ci_cache.count = 0;
    runners[k].cj_cache.count = 0;
    cache_init(&runners[k].ci_cache, CACHE_SIZE);
    cache_init(&runners[k].cj_cache, CACHE_SIZE);
#endif
  }

#ifdef HAVE_SETAFFINITY
  free(cpuid);
#endif

  /* Startup rendezvous: each worker's first action is to wait at wait_barrier,
   * so join it once here to confirm they are all up and release them onto
   * run_barrier, where the first real run will find them. */
  swift_barrier_wait(&engine.wait_barrier);

  for (int i = 0; i < runs; ++i) {

    for (int j = 0; j < total_cells; ++j)
      zero_particle_fields(&cells[j], kernel);

    scheduler_clear_active(&engine.sched);
    for (int j = 0; j < nr_tasks_built; ++j)
      scheduler_activate(&engine.sched, tasks[j]);

    const ticks tic = getticks();

    /* Mirrors engine_launch()'s atomic_inc/dec. */
    /* Prepare the scheduler. */
    atomic_inc(&engine.sched.waiting);

    /* Cry havoc and let loose the dogs of war. */
    swift_barrier_wait(&engine.run_barrier);

    /* Load the tasks. */
    scheduler_start(&engine.sched);

    /* Remove the safeguard. */
    pthread_mutex_lock(&engine.sched.sleep_mutex);
    atomic_dec(&engine.sched.waiting);
    pthread_cond_broadcast(&engine.sched.sleep_cond);
    pthread_mutex_unlock(&engine.sched.sleep_mutex);

    /* Sit back and wait for the runners to come home. */
    swift_barrier_wait(&engine.wait_barrier);

    const ticks toc = getticks();
    time += toc - tic;
  }

  swift_likwid_marker_close();

  /* Shut the persistent workers down. */
  engine.step_props |= engine_step_prop_done;
  swift_barrier_wait(&engine.run_barrier);
  for (int k = 0; k < threads; k++) pthread_join(runners[k].thread, NULL);
  swift_barrier_destroy(&engine.run_barrier);
  swift_barrier_destroy(&engine.wait_barrier);

  for (int k = 0; k < threads; k++) {
    gravity_cache_clean(&runners[k].ci_gravity_cache);
    gravity_cache_clean(&runners[k].cj_gravity_cache);
#ifdef WITH_VECTORIZATION
    cache_clean(&runners[k].ci_cache);
    cache_clean(&runners[k].cj_cache);
#endif
  }

  message("SWIFT calculation took: %.3f %s per run (all %d cells).",
          clocks_from_ticks(time / runs), clocks_getunit(), total_cells);

  /* Clean things to make the sanitizer happy ... */
  for (int i = 0; i < total_cells; ++i) clean_up(&cells[i], kernel);
  swift_free("cells_top", cells);
  if (multipoles != NULL) swift_free("multipoles_top", multipoles);
  free(tasks);
  swift_free("runners", runners);
  scheduler_clean(&engine.sched);
  threadpool_clean(&engine.threadpool);

  return 0;
}
