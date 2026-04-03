/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <limits>

#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/erosion.hpp"
#include "highmap/features.hpp"
#include "highmap/filters.hpp"
#include "highmap/hydrology/hydrology.hpp"
#include "highmap/interpolate2d.hpp"
#include "highmap/kernels.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/range.hpp"

namespace hmap
{

Array merge_water_depths(const Array &depth1,
                         const Array &depth2,
                         float        k_smooth)
{
  Array water_depth(depth1.shape);

  if (k_smooth == 0.f)
    water_depth = maximum(depth1, depth2);
  else
    water_depth = maximum_smooth(depth1, depth2, k_smooth) - k_smooth / 6.f;

  water_depth.infos();

  return water_depth;
}

void water_depth_dry_out(Array       &water_depth,
                         float        dry_out_ratio,
                         const Array *p_mask,
                         float        depth_max)
{
  if (depth_max == std::numeric_limits<float>::max())
    depth_max = water_depth.max();

  if (p_mask)
  {
    for (int j = 0; j < water_depth.shape.y; j++)
      for (int i = 0; i < water_depth.shape.x; i++)
      {
        water_depth(i, j) -= dry_out_ratio * depth_max * (*p_mask)(i, j);
        water_depth(i, j) = std::max(0.f, water_depth(i, j));
      }
  }
  else
  {
    for (int j = 0; j < water_depth.shape.y; j++)
      for (int i = 0; i < water_depth.shape.x; i++)
      {
        water_depth(i, j) -= dry_out_ratio * depth_max;
        water_depth(i, j) = std::max(0.f, water_depth(i, j));
      }
  }
}

Array water_depth_from_mask(const Array &z,
                            const Array &mask,
                            float        mask_threshold,
                            int          iterations_max,
                            float        tolerance,
                            float        omega)
{
  Array water_depth(z.shape);

  // transform to binary 0|1 mask
  Array mask_t = mask;
  make_binary(mask_t, mask_threshold);
  mask_t = 1.f - mask_t; // fixed values

  water_depth = harmonic_interpolation(z,
                                       mask_t,
                                       iterations_max,
                                       tolerance,
                                       omega) -
                z;

  return water_depth;
}

Array water_depth_increase(const Array &water_depth,
                           const Array &z,
                           float        additional_depth)
{
  const glm::ivec2 shape = water_depth.shape;
  Array            water_depth_extended(shape);

  const std::array<glm::ivec2, 8> neighbors = {
      {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}}};

  auto in_bounds = [&](const glm::ivec2 &p)
  { return p.x >= 0 && p.x < shape.x && p.y >= 0 && p.y < shape.y; };

  std::deque<glm::ivec2> queue;
  const size_t           max_it = 2 * shape.x * shape.y;

  // --- Seed water cells and enqueue border cells

  for (int y = 0; y < shape.y; ++y)
  {
    for (int x = 0; x < shape.x; ++x)
    {
      if (water_depth(x, y) <= 0.f) continue;

      water_depth_extended(x, y) = water_depth(x, y) + additional_depth;
      const glm::ivec2 p{x, y};

      for (const auto &d : neighbors)
      {
        glm::ivec2 n = p + d;
        if (in_bounds(n) && water_depth(n.x, n.y) == 0.f)
        {
          queue.push_back(p);
          break;
        }
      }
    }
  }

  // --- Upward flood propagation

  for (size_t it = 0; !queue.empty() && it < max_it; ++it)
  {
    glm::ivec2 p = queue.front();
    queue.pop_front();

    const float base_depth = water_depth_extended(p.x, p.y);
    const float base_z = z(p.x, p.y);

    for (const auto &d : neighbors)
    {
      glm::ivec2 n = p + d;
      if (!in_bounds(n)) continue;

      const float dz = z(n.x, n.y) - base_z;
      if (dz <= 0.f) continue;

      const float propagated = base_depth - dz;
      if (propagated > water_depth_extended(n.x, n.y))
      {
        water_depth_extended(n.x, n.y) = propagated;
        queue.push_back(n);
      }
    }
  }

  // --- Hole filling (downward propagation)

  queue.clear();

  for (int y = 0; y < shape.y; ++y)
  {
    for (int x = 0; x < shape.x; ++x)
    {
      if (water_depth_extended(x, y) <= 0.f) continue;

      const glm::ivec2 p{x, y};

      for (const auto &d : neighbors)
      {
        glm::ivec2 n = p + d;
        if (in_bounds(n) && water_depth_extended(n.x, n.y) == 0.f)
        {
          queue.push_back(p);
          break;
        }
      }
    }
  }

  for (size_t it = 0; !queue.empty() && it < max_it; ++it)
  {
    glm::ivec2 p = queue.front();
    queue.pop_front();

    const float base_depth = water_depth_extended(p.x, p.y);
    const float base_z = z(p.x, p.y);

    for (const auto &d : neighbors)
    {
      glm::ivec2 n = p + d;
      if (!in_bounds(n)) continue;

      const float dz = z(n.x, n.y) - base_z;
      if (dz >= 0.f) continue;

      const float propagated = base_depth + dz;
      if (propagated > water_depth_extended(n.x, n.y))
      {
        water_depth_extended(n.x, n.y) = propagated;
        queue.push_back(n);
      }
    }
  }

  return water_depth_extended;
}

Array water_mask(const Array &water_depth)
{
  Array mask = water_depth;
  make_binary(mask);
  return mask;
}

Array water_mask(const Array &water_depth,
                 const Array &z,
                 float        additional_depth)
{
  Array mask(water_depth.shape);
  Array water_depth_extended = water_depth_increase(water_depth,
                                                    z,
                                                    additional_depth);

  mask = water_depth_extended - water_depth;
  mask /= additional_depth;
  mask = smoothstep3(mask);

  return mask;
}

} // namespace hmap

namespace hmap::gpu
{

void water_depth_filter(Array &depth, const Array &z, int ir)
{
  const glm::ivec2 shape = depth.shape;
  Array            zt = z + depth;

  auto run = clwrapper::Run("water_depth_filter");

  run.bind_imagef("depth", depth.vector, shape.x, shape.y);
  run.bind_imagef("zt", zt.vector, shape.x, shape.y);
  run.bind_imagef("zt_out", zt.vector, shape.x, shape.y, true);

  run.bind_arguments(shape.x, shape.y, ir);

  run.execute({shape.x, shape.y});
  run.read_imagef("zt_out");

  // retrieve depth
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
      if (depth(i, j) != 0.f) depth(i, j) = std::max(0.f, zt(i, j) - z(i, j));
}

} // namespace hmap::gpu
