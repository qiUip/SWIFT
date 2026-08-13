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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Local headers. */
#include "likwid_wrapper.h"
#include "runner_doiact_grav.h"
#include "swift.h"

/**
 * @brief Gravity P-P analogue of test27cells.
 *
 * Builds a 3x3x3 stencil of cells and runs one gravity task type in
 * isolation: 26 x runner_dopair_grav_pp() against the central cell, plus
 * one runner_doself_grav_pp() on it. allow_mpole=0 forces true P-P, so no
 * multipole approximation is taken.
 *
 * Non-periodic (mesh.periodic = 0) selects the _full kernel rather than
 * _truncated. Softening is not applied as the spacing is set to that
 * every pair is in the unsoftened Newtonian regime.
 */

/**
 * @brief Fill a cell with gparts on a perturbed Cartesian grid.
 *
 * Using a grid rather than random_uniform() to ensure particles are not too
 * close together and can run without adaptive_softening.
 */
struct cell *make_cell(int n, const double loc[3], double width,
                       double perturbation, double epsilon, long long id_base,
                       const struct gravity_props *grav_props) {
  const int count = n * n * n;
  const double cell_size = width / (double)n;
  struct cell *cell = NULL;
  if (posix_memalign((void **)&cell, cell_align, sizeof(struct cell)) != 0) {
    error("Couldn't allocate the cell");
  }
  bzero(cell, sizeof(struct cell));

  if (posix_memalign((void **)&cell->grav.parts, gpart_align,
                     count * sizeof(struct gpart)) != 0)
    error("Couldn't allocate gparts, no. of gparts: %d", count);
  bzero(cell->grav.parts, count * sizeof(struct gpart));

  struct gpart *part = cell->grav.parts;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {

        part->x[0] =
            loc[0] +
            cell_size * (i + 0.5 + perturbation * random_uniform(-0.5, 0.5));
        part->x[1] =
            loc[1] +
            cell_size * (j + 0.5 + perturbation * random_uniform(-0.5, 0.5));
        part->x[2] =
            loc[2] +
            cell_size * (k + 0.5 + perturbation * random_uniform(-0.5, 0.5));

        part->mass = 1.f / (float)count;
        part->epsilon = epsilon;
        part->type = swift_type_dark_matter;
        part->time_bin = 1;
        part->id_or_neg_offset = id_base + (part - cell->grav.parts);

        ++part;
      }
    }
  }

  /* Cell properties */
  cell->loc[0] = loc[0];
  cell->loc[1] = loc[1];
  cell->loc[2] = loc[2];
  cell->width[0] = width;
  cell->width[1] = width;
  cell->width[2] = width;
  cell->grav.count = count;
  cell->grav.count_total = count;

  /* Initialise the locks */
  lock_init(&cell->grav.plock);
  lock_init(&cell->grav.mlock);

  /* Time bins. These MUST agree with engine.ti_current or
   * cell_is_active_gravity() returns 0 and the kernel silently early-outs
   * (src/active.h:237), unlike in testGravitySpeed. */
  cell->grav.ti_old_part = 8;
  cell->grav.ti_old_multipole = 8;
  cell->grav.ti_end_min = 8;

  /* Create the multipoles. allow_mpole=0 so that they are not used for the
   * interaction. */
  if (posix_memalign((void **)&cell->grav.multipole, multipole_align,
                     sizeof(struct gravity_tensors)) != 0)
    error("Couldn't allocate multipole.");
  gravity_reset(cell->grav.multipole);
  gravity_P2M(cell->grav.multipole, cell->grav.parts, count, grav_props);
  gravity_multipole_compute_power(&cell->grav.multipole->m_pole);

  return cell;
}

void clean_up(struct cell *c) {
  free(c->grav.parts);
  free(c->grav.multipole);
  free(c);
}

/**
 * @brief Direct unsoftened Newtonian summation onto the main cell.
 *
 * Valid only while every pair separation exceeds the softening, which the
 * grid placement guarantees for the default epsilon. Used to prove the
 * kernel is doing real work and getting the right answer.
 */
void brute_force_stencil(struct cell **cells, struct cell *main_cell,
                         double G_Newton, double *acc) {

  for (int i = 0; i < main_cell->grav.count; ++i) {

    const struct gpart *part_i = &main_cell->grav.parts[i];
    double a[3] = {0., 0., 0.};

    for (int c = 0; c < 27; ++c) {
      for (int j = 0; j < cells[c]->grav.count; ++j) {

        const struct gpart *part_j = &cells[c]->grav.parts[j];
        if (part_i == part_j) continue;

        const double dx[3] = {part_j->x[0] - part_i->x[0],
                              part_j->x[1] - part_i->x[1],
                              part_j->x[2] - part_i->x[2]};
        const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
        const double r = sqrt(r2);
        const double f = G_Newton * part_j->mass / (r2 * r);

        a[0] += f * dx[0];
        a[1] += f * dx[1];
        a[2] += f * dx[2];
      }
    }

    acc[3 * i + 0] = a[0];
    acc[3 * i + 1] = a[1];
    acc[3 * i + 2] = a[2];
  }
}

int main(int argc, char *argv[]) {

#ifdef HAVE_SETAFFINITY
  engine_pin();
#endif

  size_t runs = 0;
  int particles = 0;
  double size = 1.;
  double perturbation = 0.1;
  double epsilon = 1e-4;
  int skip_brute_force = 0;

  unsigned long long cpufreq = 0;
  clocks_set_cpufreq(cpufreq);

#ifdef HAVE_FE_ENABLE_EXCEPT
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  srand(0);

  int c;
  while ((c = getopt(argc, argv, "n:r:s:d:e:b")) != -1) {
    switch (c) {
      case 'n':
        sscanf(optarg, "%d", &particles);
        break;
      case 'r':
        sscanf(optarg, "%zu", &runs);
        break;
      case 's':
        sscanf(optarg, "%lf", &size);
        break;
      case 'd':
        sscanf(optarg, "%lf", &perturbation);
        break;
      case 'e':
        sscanf(optarg, "%lf", &epsilon);
        break;
      case 'b':
        skip_brute_force = 1;
        break;
      case '?':
        error("Unknown option.");
        break;
    }
  }

  if (particles <= 0 || runs == 0) {
    printf(
        "\nUsage: %s -n PARTICLES_PER_AXIS -r NUMBER_OF_RUNS [OPTIONS...]\n"
        "\nGenerates a 3x3x3 stencil of cells filled with gparts, then runs"
        "\n26 x runner_dopair_grav_pp() plus 1 x runner_doself_grav_pp() on"
        "\nthe central cell. allow_mpole=0, symmetric=1, non-periodic."
        "\n\nOptions:"
        "\n-s size            - Physical size of a cell"
        "\n-d pert            - Grid perturbation in [0,1["
        "\n-e epsilon         - Softening length (default 1e-4, unsoftened)"
        "\n-b                 - Skip the brute-force accuracy check\n",
        argv[0]);
    exit(1);
  }

  const int count = particles * particles * particles;

  message("Stencil: 27 cells of %d gparts (%d total)", count, 27 * count);
  message("Cell size: %f, perturbation: %f", size, perturbation);
  message("Softening: epsilon = %e", epsilon);
  message("Kernel: runner_dopair_grav_pp(symmetric=1, allow_mpole=0), _full");
  message("Multipole order: %d", SELF_GRAVITY_MULTIPOLE_ORDER);
  message("gpart size: %zu B, stencil footprint: %.2f MB", sizeof(struct gpart),
          27. * count * sizeof(struct gpart) / (1024. * 1024.));
  printf("\n");

  /* Build the infrastructure */
  struct gravity_props grav_props;
  bzero(&grav_props, sizeof(struct gravity_props));
  grav_props.use_advanced_MAC = 0;
  grav_props.use_adaptive_tolerance = 0;
  grav_props.theta_crit = 0.;
  grav_props.G_Newton = 1.;
  grav_props.epsilon_DM_cur = epsilon;
  grav_props.epsilon_baryon_cur = epsilon;

  const double box_size = 3. * size;
  double dim[3] = {box_size, box_size, box_size};
  struct pm_mesh mesh;
  pm_mesh_init_no_mesh(&mesh, dim);

  struct space space;
  bzero(&space, sizeof(struct space));
  space.periodic = 0;
  space.dim[0] = dim[0];
  space.dim[1] = dim[1];
  space.dim[2] = dim[2];

  struct engine engine;
  bzero(&engine, sizeof(struct engine));
  engine.s = &space;
  engine.mesh = &mesh;
  engine.gravity_properties = &grav_props;
  engine.max_active_bin = num_time_bins;
  engine.ti_current = 8;
  engine.time = 0.1;
  engine.nodeID = 0;

  struct cosmology cosmo;
  cosmology_init_no_cosmo(&cosmo);
  engine.cosmology = &cosmo;

  /* Runner, with its per-runner gravity caches. Initialised once, outside
   * the measured region. */
  struct runner runner;
  bzero(&runner, sizeof(struct runner));
  runner.e = &engine;

  /* Init the cache for gravity interaction */
  gravity_cache_init(&runner.ci_gravity_cache, count + VEC_SIZE);
  gravity_cache_init(&runner.cj_gravity_cache, count + VEC_SIZE);

  /* Construct some cells */
  struct cell *cells[27];
  struct cell *main_cell;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {

        const int idx = i * 9 + j * 3 + k;
        const double loc[3] = {i * size, j * size, k * size};

        cells[idx] = make_cell(particles, loc, size, perturbation, epsilon,
                               (long long)idx * count, &grav_props);
      }
    }
  }

  /* Store the main cell for future use */
  main_cell = cells[13];

  ticks time = 0;

  swift_likwid_marker_init();
  swift_likwid_marker_register("grav_pp_stencil");

  for (size_t run = 0; run < runs; ++run) {

    /* Zero the fields */
    for (int i = 0; i < 27; ++i)
      for (int pid = 0; pid < cells[i]->grav.count; pid++)
        gravity_init_gpart(&cells[i]->grav.parts[pid]);

    const ticks tic = getticks();
    swift_likwid_marker_start_region("grav_pp_stencil");

    for (int i = 0; i < 27; ++i) {
      if (cells[i] != main_cell)
        runner_dopair_grav_pp(&runner, main_cell, cells[i], /*symmetric=*/1,
                              /*allow_mpole=*/0);
    }
    runner_doself_grav_pp(&runner, main_cell);

    swift_likwid_marker_stop_region("grav_pp_stencil");
    const ticks toc = getticks();
    time += toc - tic;
  }

  swift_likwid_marker_close();

  message("SWIFT calculation took: %.3f %s.", clocks_from_ticks(time / runs),
          clocks_getunit());

  /* Finish the gravity calculation. The kernels accumulate a_grav without
   * Newton's constant; gravity_end_force() applies it. Without this the
   * brute-force comparison would only agree for G_Newton = 1. Outside the
   * timed region, and only on the cell we check. */
  for (int i = 0; i < count; ++i)
    gravity_end_force(&main_cell->grav.parts[i], grav_props.G_Newton,
                      /*potential_normalisation=*/0.f, /*periodic=*/0,
                      /*with_self_gravity=*/1);

  /* Accuracy check against direct summation. */
  if (!skip_brute_force) {

    double *acc = (double *)malloc(3 * count * sizeof(double));
    if (acc == NULL) error("Couldn't allocate brute-force array.");

    const ticks tic = getticks();
    brute_force_stencil(cells, main_cell, grav_props.G_Newton, acc);
    const ticks toc = getticks();

    double max_rel = 0., max_rel_vec = 0.;
    int n_bad = 0, n_bad_vec = 0;
    for (int i = 0; i < count; ++i) {

      double da2 = 0., truth2 = 0.;

      for (int k = 0; k < 3; ++k) {

        const double truth = acc[3 * i + k];
        const double got = main_cell->grav.parts[i].a_grav[k];
        const double scale = fabs(truth) + fabs(got);

        da2 += (truth - got) * (truth - got);
        truth2 += truth * truth;

        if (scale < 1e-30) continue;

        const double rel = fabs(truth - got) / scale;
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-4) ++n_bad;
      }

      /* Vector error, |a_truth - a_got| / |a_truth| */
      if (truth2 < 1e-60) continue;

      const double rel_vec = sqrt(da2 / truth2);
      if (rel_vec > max_rel_vec) max_rel_vec = rel_vec;
      if (rel_vec > 1e-4) ++n_bad_vec;
    }

    message("Brute force calculation took: %.3f %s.",
            clocks_from_ticks(toc - tic), clocks_getunit());
    message("Max relative error: %e (%d components above 1e-4)", max_rel,
            n_bad);
    message("Max vector error:   %e (%d particles above 1e-4)", max_rel_vec,
            n_bad_vec);

    if (n_bad > 0 || n_bad_vec > 0)
      message("WARNING: accuracy check FAILED - results are not trustworthy.");
    else
      message("Accuracy check passed.");

    free(acc);
  }

  for (int i = 0; i < 27; ++i) clean_up(cells[i]);
  gravity_cache_clean(&runner.ci_gravity_cache);
  gravity_cache_clean(&runner.cj_gravity_cache);

  return 0;
}
