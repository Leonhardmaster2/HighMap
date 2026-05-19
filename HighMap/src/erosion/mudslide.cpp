/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <utility> // for move

#include "highmap/array.hpp"               // for Array, operator*
#include "highmap/erosion.hpp"             // for mudslide
#include "highmap/filters.hpp"             // for laplace, make_binary
#include "highmap/gradient.hpp"            // for gradient_norm
#include "highmap/hydrology/hydrology.hpp" // for flow_simulation_viscous
#include "highmap/math/array.hpp"          // for is_zero, pow
#include "highmap/morphology.hpp"          // for distance_transform
#include "highmap/range.hpp"               // for remap

namespace hmap::gpu
{

void mudslide(Array       &z,
              const Array &landslide_mask,
              float        depth,
              int          iterations,
              float        depth_map_exponent,
              float        viscosity_law_power,
              Array       *p_depth_end,
              Array       *p_depth_init)
{
  Array depth_map = distance_transform(is_zero(landslide_mask));
  hmap::laplace(depth_map);
  remap(depth_map);
  depth_map = pow(depth_map, depth_map_exponent);

  {
    Array scaled_depth = depth * depth_map;
    z -= scaled_depth;

    if (p_depth_init) *p_depth_init = std::move(scaled_depth);
  }

  Array new_depth = flow_simulation_viscous(z,
                                            depth,
                                            depth_map,
                                            iterations,
                                            /* dt */ 0.5,
                                            /* dry_out_ratio */ 0.f,
                                            /* viscosity */ 1.f,
                                            viscosity_law_power);

  z += new_depth;

  if (p_depth_end) *p_depth_end = std::move(new_depth);
}

void mudslide(Array &z,
              float  talus_limit,
              float  depth,
              int    iterations,
              float  depth_map_exponent,
              float  viscosity_law_power,
              Array *p_depth_end,
              Array *p_depth_init)
{
  Array mask = hmap::gradient_norm(z);
  make_binary(mask, talus_limit);

  mudslide(z,
           mask,
           depth,
           iterations,
           depth_map_exponent,
           viscosity_law_power,
           p_depth_end,
           p_depth_init);
}

} // namespace hmap::gpu
