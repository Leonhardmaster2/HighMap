/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/boundary.hpp"
#include "highmap/curvature.hpp"
#include "highmap/filters.hpp"
#include "highmap/gradient.hpp"
#include "highmap/range.hpp"

namespace hmap::gpu
{

Array level_set_curvature(const Array &array, int prefilter_ir)
{
  Array array_f = array;
  if (prefilter_ir) gpu::smooth_cpulse(array_f, prefilter_ir);

  // compute divergence of the normalized gradient
  Array gx = gradient_x(array_f);
  Array gy = gradient_y(array_f);
  Array gn = hmap::gradient_norm(array_f) + 1e-12f;

  Array dgx = gradient_x(gx / gn);
  Array dgy = gradient_y(gy / gn);

  gn = dgx + dgy;

  return gn;
}

} // namespace hmap::gpu
