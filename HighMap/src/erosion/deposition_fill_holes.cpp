/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/blending.hpp"
#include "highmap/filters.hpp"
#include "highmap/math.hpp"
#include "highmap/opencl/gpu_opencl.hpp"

namespace hmap::gpu
{

void deposition_fill_holes(Array &z,
                           int    deposition_ir,
                           float  deposition_strength,
                           int    iterations)
{
  Array zd;

  for (int it = 0; it < iterations; ++it)
  {
    zd = z;
    gpu::smooth_fill_holes(zd, deposition_ir);
    zd = gpu::blend_gradients(zd, z, deposition_ir);
    z = lerp(z, zd, deposition_strength);
  }
}

} // namespace hmap::gpu
