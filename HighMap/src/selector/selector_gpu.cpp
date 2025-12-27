/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/curvature.hpp"
#include "highmap/filters.hpp"
#include "highmap/morphology.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/range.hpp"

namespace hmap::gpu
{

Array select_cavities(const Array &array, int ir, bool concave)
{
  Array array_smooth = array;
  gpu::smooth_cpulse(array_smooth, ir);
  Array c = curvature_mean(array_smooth);

  if (!concave) c *= -1.f;

  clamp_min(c, 0.f);
  return c;
}

Array select_valley(const Array &z,
                    int          ir,
                    bool         zero_at_borders,
                    bool         ridge_select)
{
  Array w = z;
  gpu::smooth_cpulse(w, std::max(1, ir));

  if (not(ridge_select)) w *= -1.f;

  w = curvature_mean(w);
  make_binary(w);
  w = gpu::relative_distance_from_skeleton(w, ir, zero_at_borders);

  return w;
}

} // namespace hmap::gpu
