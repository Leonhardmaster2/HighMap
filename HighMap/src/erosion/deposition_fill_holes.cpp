/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/array.hpp"      // for Array
#include "highmap/blending.hpp"   // for blend_gradients
#include "highmap/erosion.hpp"    // for deposition_fill_holes
#include "highmap/filters.hpp"    // for smooth_fill_holes
#include "highmap/math/array.hpp" // for lerp
#include "highmap/operator.hpp"   // for apply_with_mask

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

void deposition_fill_holes(Array       &z,
                           int          deposition_ir,
                           float        deposition_strength,
                           const Array *p_mask,
                           int          iterations)
{
  apply_with_mask(z,
                  p_mask,
                  [&](Array &a) {
                    deposition_fill_holes(a,
                                          deposition_ir,
                                          deposition_strength,
                                          iterations);
                  });
}

} // namespace hmap::gpu
