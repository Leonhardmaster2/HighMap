/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm> // for fill_n, min
#include <cstdint>   // for uint32_t
#include <vector>    // for vector

#include "highmap/array.hpp"   // for Array, operator*
#include "highmap/erosion.hpp" // for strata_plates
#include "highmap/filters.hpp" // for project_talus_along_direction
#include "highmap/internal/vector_utils.hpp"
#include "highmap/math/array.hpp" // for lerp
#include "highmap/math/core.hpp"  // for triangle
#include "highmap/transform.hpp"  // for warp

namespace hmap::gpu
{

void strata_plates(Array        &z,
                   const Array  &talus,
                   int           direction_offset,
                   int           direction_count,
                   bool          random_directions,
                   std::uint32_t seed,
                   float         vmin,
                   float         skew,
                   float         mix_ratio,
                   const Array  *p_mask,
                   const Array  *p_dx,
                   const Array  *p_dy)
{
  // --- List of directions to project the talus along

  direction_count = std::min(8, direction_count);

  std::vector<int>   directions(direction_count);
  std::vector<float> skew_factor(direction_count);

  if (random_directions)
  {
    directions = generate_unique_random_vector(direction_count, 0, 7, seed);
  }
  else
  {
    for (int k = 0; k < direction_count; ++k)
      directions[k] = (k + direction_offset) % 8;
  }

  float shift = direction_count % 2 ? 1.f : 0.f;

  for (int k = 0; k < direction_count; ++k)
  {
    float t = 1.f - triangle(float(k), 0.f, direction_count - shift);
    skew_factor[k] = 1.f + skew * t;
  }

  // --- Apply...

  for (int k = 0; k < direction_count; ++k)
  {
    Array zp = project_talus_along_direction(z,
                                             skew_factor[k] * talus,
                                             p_mask,
                                             directions[k],
                                             vmin);
    z = lerp(z, zp, mix_ratio);
  }

  // --- Warp

  gpu::warp(z, p_dx, p_dy);
}

} // namespace hmap::gpu
