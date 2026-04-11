/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/array.hpp"
#include "highmap/convolve.hpp"
#include "highmap/curvature.hpp"
#include "highmap/filters.hpp"
#include "highmap/math.hpp"
#include "highmap/morphology.hpp"
#include "highmap/range.hpp"

namespace hmap
{

Array local_max(const Array &array, int ir)
{
  Array array_out = Array(array.shape);
  Array array_tmp = Array(array.shape);

  // row
  for (int i = 0; i < array.shape.x; i++)
  {
    int i1 = std::max(0, i - ir);
    int i2 = std::min(array.shape.x, i + ir + 1);

    for (int j = 0; j < array.shape.y; j++)
    {
      float max = array(i, j);
      for (int u = i1; u < i2; u++)
        if (array(u, j) > max) max = array(u, j);
      array_tmp(i, j) = max;
    }
  }

  // column
  for (int j = 0; j < array.shape.y; j++)
  {
    int j1 = std::max(0, j - ir);
    int j2 = std::min(array.shape.y, j + ir + 1);
    for (int i = 0; i < array.shape.x; i++)
    {
      float max = array_tmp(i, j);
      for (int v = j1; v < j2; v++)
        if (array_tmp(i, v) > max) max = array_tmp(i, v);
      array_out(i, j) = max;
    }
  }

  return array_out;
}

Array local_mean(const Array &array, int ir)
{
  Array array_out = Array(array.shape);

  std::vector<float> k1d(2 * ir + 1);
  for (auto &v : k1d)
    v = 1.f / (float)(2 * ir + 1);

  array_out = convolve1d_i(array, k1d);
  array_out = convolve1d_j(array_out, k1d);

  return array_out;
}

Array local_min(const Array &array, int ir)
{
  return -local_max(-array, ir);
}

Array local_median_deviation(const Array &array, int ir)
{
  Array mean = local_mean(array, ir);
  Array med = median_pseudo(array, ir); // TODO exact
  return abs(mean - med);
}

Array relative_elevation(const Array &array, int ir)
{
  Array amin = local_min(array, ir);
  Array amax = local_max(array, ir);

  smooth_cpulse(amin, ir);
  smooth_cpulse(amax, ir);

  return (array - amin) / (amax - amin + std::numeric_limits<float>::min());
}

Array ruggedness(const Array &array, int ir)
{
  Array rg(array.shape);

  for (int j = 0; j < array.shape.y; j++)
    for (int i = 0; i < array.shape.x; i++)
    {
      int i1 = std::max(i - ir, 0);
      int i2 = std::min(i + ir + 1, array.shape.x);
      int j1 = std::max(j - ir, 0);
      int j2 = std::min(j + ir + 1, array.shape.y);

      for (int p = i1; p < i2; p++)
        for (int q = j1; q < j2; q++)
        {
          float delta = array(i, j) - array(p, q);
          rg(i, j) += delta * delta;
        }
    }

  for (int j = 0; j < array.shape.y; j++)
    for (int i = 0; i < array.shape.x; i++)
      rg(i, j) = std::sqrt(rg(i, j));

  return rg;
}

Array rugosity(const Array &z, int ir, bool convex)
{
  hmap::Array z_avg = Array(z.shape);
  hmap::Array z_std = Array(z.shape);
  hmap::Array z_skw = Array(z.shape);

  // pre high-pass filter to remove low wavenumbers
  Array zf = z;
  smooth_cpulse(zf, 2 * ir);
  zf = z - zf;

  // use Gaussian windowing instead of a real arithmetic averaging to
  // limit boundary artifacts
  z_avg = zf;
  smooth_cpulse(z_avg, ir);

  z_std = (zf - z_avg) * (zf - z_avg);
  smooth_cpulse(z_std, ir);

  // Fisher-Pearson coefficient of skewness
  z_skw = (zf - z_avg) * (zf - z_avg) * (zf - z_avg);

  // do not filter, surprisingly yields much better results
  // smooth_cpulse(z_skw, ir);

  float tol = 1e-30f * z.ptp();

  for (int j = 0; j < z.shape.y; j++)
    for (int i = 0; i < z.shape.x; i++)
      if (z_std(i, j) > tol)
        z_skw(i, j) /= std::pow(z_std(i, j), 1.5f);
      else
        z_skw(i, j) = 0.f;

  // keep only "bumpy" rugosities
  if (convex)
    clamp_min(z_skw, 0.f);
  else
    clamp_max(z_skw, 0.f);

  return z_skw;
}

} // namespace hmap
