/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/filters.hpp"
#include "highmap/gradient.hpp"
#include "highmap/morphology.hpp"

namespace hmap
{

Array signed_curvature_from_distance(const Array &array, int prefilter_ir)
{
  Array dist = distance_transform(array);

  if (prefilter_ir) smooth_cpulse(dist, prefilter_ir);

  // compute divergence of the normalized gradient
  Array gx = gradient_x(dist);
  Array gy = gradient_y(dist);
  Array gn = gradient_norm(dist) + 1e-12f;

  Array dgx = gradient_x(gx / gn);
  Array dgy = gradient_y(gy / gn);

  gn = dgx + dgy;

  return gn;
}

Array signed_distance_transform(const Array &array, int prefilter_ir)
{
  Array dist0 = distance_transform(array);
  Array dist = dist0;

  // sign based on curvature
  if (prefilter_ir) smooth_cpulse(dist, prefilter_ir);

  // compute divergence of the normalized gradient
  Array gx = gradient_x(dist);
  Array gy = gradient_y(dist);
  Array gn = gradient_norm(dist) + 1e-12f;

  Array dgx = gradient_x(gx / gn);
  Array dgy = gradient_y(gy / gn);

  gn = dgx + dgy;

  for (int j = 0; j < array.shape.y; ++j)
    for (int i = 0; i < array.shape.x; ++i)
      dist0(i, j) *= std::copysign(1.f, gn(i, j));

  return dist0;
}

} // namespace hmap

namespace hmap::gpu
{

Array signed_curvature_from_distance(const Array &array, int prefilter_ir)
{
  Array dist = distance_transform(array);

  if (prefilter_ir) gpu::smooth_cpulse(dist, prefilter_ir);

  // compute divergence of the normalized gradient
  Array gx = gradient_x(dist);
  Array gy = gradient_y(dist);
  Array gn = hmap::gradient_norm(dist) + 1e-12f;

  Array dgx = gradient_x(gx / gn);
  Array dgy = gradient_y(gy / gn);

  gn = dgx + dgy;

  return gn;
}

Array signed_distance_transform(const Array &array, int prefilter_ir)
{
  Array dist0 = distance_transform(array);
  Array dist = dist0;

  // sign based on curvature
  if (prefilter_ir) gpu::smooth_cpulse(dist, prefilter_ir);

  // compute divergence of the normalized gradient
  Array gx = gradient_x(dist);
  Array gy = gradient_y(dist);
  Array gn = hmap::gradient_norm(dist) + 1e-12f;

  Array dgx = gradient_x(gx / gn);
  Array dgy = gradient_y(gy / gn);

  gn = dgx + dgy;

  for (int j = 0; j < array.shape.y; ++j)
    for (int i = 0; i < array.shape.x; ++i)
      dist0(i, j) *= std::copysign(1.f, gn(i, j));

  return dist0;
}

} // namespace hmap::gpu
