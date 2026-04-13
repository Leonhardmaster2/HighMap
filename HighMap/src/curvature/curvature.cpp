/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/curvature.hpp"
#include "highmap/boundary.hpp"
#include "highmap/filters.hpp"
#include "highmap/gradient.hpp"
#include "highmap/range.hpp"

namespace hmap
{

Array curvature_mean(const Array &z)
{
  Array p, q, r, s, t;
  compute_curvature_gradients(z, p, q, r, s, t);
  return -compute_curvature_h(r, t);
}

Array level_set_curvature(const Array &array, int prefilter_ir)
{
  Array array_f = array;
  if (prefilter_ir) smooth_cpulse(array_f, prefilter_ir);

  // compute divergence of the normalized gradient
  Array gx = gradient_x(array_f);
  Array gy = gradient_y(array_f);
  Array gn = hmap::gradient_norm(array_f) + 1e-12f;

  Array dgx = gradient_x(gx / gn);
  Array dgy = gradient_y(gy / gn);

  gn = dgx + dgy;

  return gn;
}

// --- helpers

void compute_curvature_gradients(const Array &z,
                                 Array       &p,
                                 Array       &q,
                                 Array       &r,
                                 Array       &s,
                                 Array       &t)
{
  p = Array(z.shape);
  q = Array(z.shape);
  r = Array(z.shape);
  s = Array(z.shape);
  t = Array(z.shape);

  for (int j = 1; j < z.shape.y - 1; ++j)
    for (int i = 1; i < z.shape.x - 1; ++i)
    {
      p(i, j) = 0.5f * (z(i + 1, j) - z(i - 1, j));        // dz/dx
      q(i, j) = 0.5f * (z(i, j + 1) - z(i, j - 1));        // dz/dy
      r(i, j) = z(i + 1, j) - 2.f * z(i, j) + z(i - 1, j); // d2z/dx2
      s(i, j) = 0.25f * (z(i - 1, j - 1) - z(i - 1, j + 1) - z(i + 1, j - 1) +
                         z(i + 1, j + 1));                 // d2z/dxdy
      t(i, j) = z(i, j + 1) - 2.f * z(i, j) + z(i, j - 1); // d2z/dy2
    }
}

Array compute_curvature_h(const Array &r, const Array &t)
{
  return -0.5f * (r + t);
}

Array compute_curvature_k(const Array &p,
                          const Array &q,
                          const Array &r,
                          const Array &s,
                          const Array &t)
{
  return (r * t - s * s) / pow(1.f + p * p + q * q, 2.f);
}

} // namespace hmap
