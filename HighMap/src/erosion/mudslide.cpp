/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/filters.hpp"
#include "highmap/hydrology.hpp"
#include "highmap/math.hpp"
#include "highmap/morphology.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/range.hpp"

namespace hmap::gpu
{

void mudslide(Array       &z,
              const Array &landslide_mask,
              float        depth,
              int          iterations,
              float        depth_map_exponent,
              float        viscosity_law_power,
              Array       *p_depth)
{
  Array depth_map = distance_transform(is_zero(landslide_mask));
  hmap::laplace(depth_map);
  remap(depth_map);
  depth_map = pow(depth_map, depth_map_exponent);

  z -= depth * depth_map;

  Array new_depth = flow_simulation_viscous(z,
                                            depth,
                                            depth_map,
                                            iterations,
                                            /* dt */ 0.5,
                                            /* dry_out_ratio */ 0.f,
                                            /* viscosity */ 1.f,
                                            viscosity_law_power);

  z += new_depth;

  if (p_depth) *p_depth = std::move(new_depth);
}

} // namespace hmap::gpu
