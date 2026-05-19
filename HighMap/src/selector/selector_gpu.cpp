/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/array.hpp"      // for Array
#include "highmap/curvature.hpp"  // for CurvatureType, curvature_quadric
#include "highmap/filters.hpp"    // for smooth_cpulse
#include "highmap/morphology.hpp" // for morphological_black_hat, morpholog...
#include "highmap/range.hpp"      // for clamp_min

namespace hmap::gpu
{

Array select_cavities(const Array &array, int ir, bool concave)
{
  Array array_smooth = array;
  gpu::smooth_cpulse(array_smooth, ir);
  Array c = -gpu::curvature_quadric(array_smooth, 0, CurvatureType::CT_MEAN);

  if (!concave) c *= -1.f;

  clamp_min(c, 0.f);
  return c;
}

Array select_valley(const Array &z, int ir, bool ridge_select)
{
  if (ridge_select)
    return gpu::morphological_top_hat(z, ir);
  else
    return gpu::morphological_black_hat(z, ir);
}

} // namespace hmap::gpu
