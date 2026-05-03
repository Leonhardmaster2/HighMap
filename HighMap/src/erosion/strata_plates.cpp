/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/erosion.hpp"
#include "highmap/filters.hpp"
#include "highmap/internal/vector_utils.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/range.hpp"
#include "highmap/transform.hpp"

namespace hmap::gpu
{

void strata_plates(Array       &z,
                   const Array &talus,
                   int          direction_offset,
                   int          direction_count,
                   bool         random_directions,
                   uint         seed,
                   float        mix_ratio,
                   const Array *p_mask,
                   const Array *p_dx,
                   const Array *p_dy)
{
  // --- List of directions to project the talus along

  direction_count = std::min(8, direction_count);

  std::vector<int> directions(direction_count);

  if (random_directions)
  {
    directions = generate_unique_random_vector(direction_count, 0, 7, seed);
  }
  else
  {
    for (int k = 0; k < direction_count; ++k)
      directions[k] = (k + direction_offset) % 8;
  }

  // --- Apply...

  for (int k = 0; k < direction_count; ++k)
  {
    Array zp = project_talus_along_direction(z, talus, p_mask, directions[k]);
    z = lerp(z, zp, mix_ratio);
  }

  // --- Warp

  gpu::warp(z, p_dx, p_dy);
}

} // namespace hmap::gpu
