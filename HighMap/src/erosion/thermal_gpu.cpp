/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <vector> // for allocator, vector

#include "cl_wrapper/run.hpp" // for Run

#include "highmap/array.hpp"      // for Array
#include "highmap/boundary.hpp"   // for extrapolate_borders
#include "highmap/math/array.hpp" // for abs
#include "highmap/operator.hpp"   // for apply_with_mask
#include "highmap/range.hpp"      // for maximum

namespace hmap::gpu
{

void thermal(Array       &z,
             const Array &talus,
             int          iterations,
             Array       *p_bedrock,
             Array       *p_deposition_map)
{
  Array z_bckp = Array();
  if (p_deposition_map != nullptr) z_bckp = z;

  if (p_bedrock)
  {
    auto run = clwrapper::Run("thermal_with_bedrock");

    run.bind_buffer<float>("z", z.vector);
    run.bind_buffer<float>("talus",
                           const_cast<std::vector<float> &>(talus.vector));
    run.bind_buffer<float>("bedrock", p_bedrock->vector);
    run.bind_arguments(z.shape.x, z.shape.y, 0);

    run.write_buffer("z");
    run.write_buffer("talus");
    run.write_buffer("bedrock");

    for (int it = 0; it < iterations; it++)
    {
      run.set_argument(5, it);
      run.execute({z.shape.x, z.shape.y});
    }

    run.read_buffer("z");
  }
  else
  {
    auto run = clwrapper::Run("thermal");

    run.bind_buffer<float>("z", z.vector);
    run.bind_buffer<float>("talus",
                           const_cast<std::vector<float> &>(talus.vector));
    run.bind_arguments(z.shape.x, z.shape.y, 0);

    run.write_buffer("z");
    run.write_buffer("talus");

    for (int it = 0; it < iterations; it++)
    {
      run.set_argument(4, it);
      run.execute({z.shape.x, z.shape.y});
    }

    run.read_buffer("z");
  }

  extrapolate_borders(z);

  if (p_deposition_map) *p_deposition_map = maximum(z - z_bckp, 0.f);
}

void thermal(Array       &z,
             const Array *p_mask,
             const Array &talus,
             int          iterations,
             Array       *p_bedrock,
             Array       *p_deposition_map)
{
  apply_with_mask(
      z,
      p_mask,
      [&](Array &a)
      { gpu::thermal(a, talus, iterations, p_bedrock, p_deposition_map); });
}

void thermal(Array &z,
             float  talus,
             int    iterations,
             Array *p_bedrock,
             Array *p_deposition_map)
{
  Array talus_map(z.shape, talus);
  gpu::thermal(z, talus_map, iterations, p_bedrock, p_deposition_map);
}

void thermal_auto_bedrock(Array       &z,
                          const Array &talus,
                          int          iterations,
                          Array       *p_deposition_map)
{
  Array z_bckp = z;
  Array bedrock(z.shape);

  auto run = clwrapper::Run("thermal_auto_bedrock");

  run.bind_buffer<float>("z", z.vector);
  run.bind_buffer<float>("talus",
                         const_cast<std::vector<float> &>(talus.vector));
  run.bind_buffer<float>("bedrock", bedrock.vector);
  run.bind_buffer<float>("z0", z_bckp.vector);
  run.bind_arguments(z.shape.x, z.shape.y, 0);

  run.write_buffer("z");
  run.write_buffer("talus");
  run.write_buffer("bedrock");
  run.write_buffer("z0");

  for (int it = 0; it < iterations; it++)
  {
    run.set_argument(6, it);
    run.execute({z.shape.x, z.shape.y});
  }

  run.read_buffer("z");
  extrapolate_borders(z);

  if (p_deposition_map) *p_deposition_map = maximum(z - z_bckp, 0.f);
}

void thermal_auto_bedrock(Array &z,
                          float  talus,
                          int    iterations,
                          Array *p_deposition_map)
{
  Array talus_map(z.shape, talus);
  gpu::thermal_auto_bedrock(z, talus_map, iterations, p_deposition_map);
}

void thermal_auto_bedrock(Array       &z,
                          const Array *p_mask,
                          const Array &talus,
                          int          iterations,
                          Array       *p_deposition_map)
{
  apply_with_mask(
      z,
      p_mask,
      [&](Array &a)
      { gpu::thermal_auto_bedrock(a, talus, iterations, p_deposition_map); });
}

void thermal_flatten(Array       &z,
                     const Array &talus,
                     int          iterations,
                     float        sigma_inf,
                     float        sigma_sup)
{
  const glm::ivec2 &shape = z.shape;

  auto run = clwrapper::Run("thermal_flatten");

  run.bind_buffer<float>("z", z.vector);
  run.bind_buffer<float>("talus", talus.vector);
  run.bind_arguments(shape.x, shape.y, sigma_inf, sigma_sup);

  run.write_buffer("z");
  run.write_buffer("talus");

  for (int it = 0; it < iterations; it++)
    run.execute({shape.x, shape.y});

  run.read_buffer("z");
  extrapolate_borders(z);
}

void thermal_flatten(Array       &z,
                     const Array *p_mask,
                     const Array &talus,
                     int          iterations,
                     float        sigma_inf,
                     float        sigma_sup)
{
  apply_with_mask(
      z,
      p_mask,
      [&](Array &a)
      { gpu::thermal_flatten(a, talus, iterations, sigma_inf, sigma_sup); });
}

void thermal_inflate(Array &z, const Array &talus, int iterations)
{
  auto run = clwrapper::Run("thermal_inflate");

  run.bind_buffer<float>("z", z.vector);
  run.bind_buffer<float>("talus", talus.vector);
  run.bind_arguments(z.shape.x, z.shape.y);

  run.write_buffer("z");
  run.write_buffer("talus");

  for (int it = 0; it < iterations; it++)
    run.execute({z.shape.x, z.shape.y});

  run.read_buffer("z");
  extrapolate_borders(z);
}

void thermal_inflate(Array       &z,
                     const Array *p_mask,
                     const Array &talus,
                     int          iterations)
{
  apply_with_mask(z,
                  p_mask,
                  [&](Array &a)
                  { gpu::thermal_inflate(a, talus, iterations); });
}

void thermal_olsen(Array &z, const Array &talus, int iterations)
{
  auto run = clwrapper::Run("thermal_olsen");

  run.bind_buffer<float>("z", z.vector);
  run.bind_buffer<float>("talus", talus.vector);
  run.bind_arguments(z.shape.x, z.shape.y);

  run.write_buffer("z");
  run.write_buffer("talus");

  for (int it = 0; it < iterations; it++)
    run.execute({z.shape.x, z.shape.y});

  run.read_buffer("z");
  extrapolate_borders(z);
}

void thermal_olsen(Array       &z,
                   const Array *p_mask,
                   const Array &talus,
                   int          iterations)
{
  apply_with_mask(z,
                  p_mask,
                  [&](Array &a) { gpu::thermal_olsen(a, talus, iterations); });
}

void thermal_rib(Array &z, int iterations)
{
  auto run = clwrapper::Run("thermal_rib");

  run.bind_buffer<float>("z", z.vector);
  run.bind_arguments(z.shape.x, z.shape.y);

  run.write_buffer("z");

  for (int it = 0; it < iterations; it++)
  {
    run.execute({z.shape.x, z.shape.y});
  }

  run.read_buffer("z");
  extrapolate_borders(z, 3);
}

void thermal_rib(Array &z, const Array *p_mask, int iterations)
{
  apply_with_mask(z,
                  p_mask,
                  [&](Array &a) { gpu::thermal_rib(a, iterations); });
}

void thermal_ridge(Array       &z,
                   const Array &talus,
                   int          iterations,
                   Array       *p_deposition_map)
{
  Array z_bckp = Array();
  if (p_deposition_map != nullptr) z_bckp = z;

  auto run = clwrapper::Run("thermal_ridge");

  run.bind_buffer<float>("z", z.vector);
  run.bind_buffer<float>("talus", talus.vector);
  run.bind_arguments(z.shape.x, z.shape.y);

  run.write_buffer("z");
  run.write_buffer("talus");

  for (int it = 0; it < iterations; it++)
    run.execute({z.shape.x, z.shape.y});

  run.read_buffer("z");
  extrapolate_borders(z);

  if (p_deposition_map) *p_deposition_map = abs(z - z_bckp);
}

void thermal_ridge(Array       &z,
                   const Array *p_mask,
                   const Array &talus,
                   int          iterations,
                   Array       *p_deposition_map)
{
  apply_with_mask(z,
                  p_mask,
                  [&](Array &a) {
                    gpu::thermal_ridge(a, talus, iterations, p_deposition_map);
                  });
}

void thermal_schott(Array       &z,
                    const Array &talus,
                    int          iterations,
                    float        intensity,
                    Array       *p_deposition_map)
{
  Array z_bckp = Array();
  if (p_deposition_map != nullptr) z_bckp = z;

  auto run = clwrapper::Run("thermal_schott");

  run.bind_buffer<float>("z", z.vector);
  run.bind_buffer<float>("talus", talus.vector);
  run.bind_arguments(z.shape.x, z.shape.y, intensity);

  run.write_buffer("z");
  run.write_buffer("talus");

  for (int it = 0; it < iterations; it++)
    run.execute({z.shape.x, z.shape.y});

  run.read_buffer("z");
  extrapolate_borders(z);

  if (p_deposition_map) *p_deposition_map = abs(z - z_bckp);
}

void thermal_schott(Array       &z,
                    const Array *p_mask,
                    const Array &talus,
                    int          iterations,
                    float        intensity,
                    Array       *p_deposition_map)
{
  apply_with_mask(
      z,
      p_mask,
      [&](Array &a) {
        gpu::thermal_schott(a, talus, iterations, intensity, p_deposition_map);
      });
}

void thermal_scree(Array       &z,
                   const Array &talus,
                   const Array &zmax,
                   int          iterations,
                   Array       *p_deposition_map)
{
  Array z_bckp = Array();
  if (p_deposition_map != nullptr) z_bckp = z;

  auto run = clwrapper::Run("thermal_scree");

  run.bind_buffer<float>("z", z.vector);
  run.bind_buffer<float>("talus", talus.vector);
  run.bind_buffer<float>("zmax", zmax.vector);
  run.bind_arguments(z.shape.x, z.shape.y);

  run.write_buffer("z");
  run.write_buffer("talus");
  run.write_buffer("zmax");

  for (int it = 0; it < iterations; it++)
    run.execute({z.shape.x, z.shape.y});

  run.read_buffer("z");
  extrapolate_borders(z);

  if (p_deposition_map) *p_deposition_map = maximum(z - z_bckp, 0.f);
}

void thermal_scree(Array       &z,
                   const Array *p_mask,
                   const Array &talus,
                   const Array &zmax,
                   int          iterations,
                   Array       *p_deposition_map)
{
  apply_with_mask(
      z,
      p_mask,
      [&](Array &a)
      { gpu::thermal_scree(a, talus, zmax, iterations, p_deposition_map); });
}

} // namespace hmap::gpu
