/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>
#include <random>

#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/erosion.hpp"
#include "highmap/interpolate1d.hpp"
#include "highmap/kernels.hpp"
#include "highmap/math.hpp"
#include "highmap/operator.hpp"
#include "highmap/primitives.hpp"
#include "highmap/range.hpp"

namespace hmap
{

void recurve(Array                    &array,
             const std::vector<float> &t,
             const std::vector<float> &v)
{
  Interpolator1D interp = Interpolator1D(t, v, InterpolationMethod1D::CUBIC);

  auto lambda = [&interp](float a) { return interp(a); };

  std::transform(array.vector.begin(),
                 array.vector.end(),
                 array.vector.begin(),
                 lambda);
}

void recurve(Array                    &array,
             const std::vector<float> &t,
             const std::vector<float> &v,
             const Array              *p_mask)
{
  apply_with_mask(array, p_mask, [&](Array &a) { recurve(a, t, v); });
}

void recurve_bexp(Array &array, float tau)
{
  float c = -1.f / tau;
  auto  lambda = [&c](float a) { return 1.f - std::exp(c * a); };

  std::transform(array.vector.begin(),
                 array.vector.end(),
                 array.vector.begin(),
                 lambda);
}

void recurve_bexp(Array &array, float tau, const Array *p_mask)
{
  apply_with_mask(array, p_mask, [&](Array &a) { recurve_bexp(a, tau); });
}

void recurve_exp(Array &array, float tau)
{
  float c = -1.f / tau;
  auto  lambda = [&c](float a) { return std::exp(c * (1.f - a)); };

  std::transform(array.vector.begin(),
                 array.vector.end(),
                 array.vector.begin(),
                 lambda);
}

void recurve_exp(Array &array, float tau, const Array *p_mask)
{
  apply_with_mask(array, p_mask, [&](Array &a) { recurve_exp(a, tau); });
}

void recurve_kura(Array &array, float a, float b)
{
  auto lambda = [&a, &b](float v)
  { return 1.f - std::pow(1.f - std::pow(v, a), b); };

  std::transform(array.vector.begin(),
                 array.vector.end(),
                 array.vector.begin(),
                 lambda);
}

void recurve_kura(Array &array, float a, float b, const Array *p_mask)
{
  apply_with_mask(array, p_mask, [&](Array &ar) { recurve_kura(ar, a, b); });
}

void recurve_s(Array &array)
{
  auto lambda = [](float a) { return a * a * (3.f - 2.f * a); };

  std::transform(array.vector.begin(),
                 array.vector.end(),
                 array.vector.begin(),
                 lambda);
}

void recurve_s(Array &array, const Array *p_mask)
{
  apply_with_mask(array, p_mask, [&](Array &a) { recurve_s(a); });
}

void recurve_smoothstep_rational(Array &array, float n)
{
  auto lambda = [&n](float a)
  {
    float an = std::pow(a, n);
    return an / (an + std::pow(1.f - a, n));
  };

  std::transform(array.vector.begin(),
                 array.vector.end(),
                 array.vector.begin(),
                 lambda);
}

void recurve_smoothstep_rational(Array &array, float n, const Array *p_mask)
{
  apply_with_mask(array,
                  p_mask,
                  [&](Array &a) { recurve_smoothstep_rational(a, n); });
}

void saturate(Array &array,
              float  vmin,
              float  vmax,
              float  from_min,
              float  from_max,
              float  k)
{
  if (k > 0.f)
    clamp_smooth(array, vmin, vmax, k);
  else
    clamp(array, vmin, vmax);

  remap(array, from_min, from_max, vmin, vmax);
}

void saturate(Array &array, float vmin, float vmax, float k)
{
  float min_bckp = array.min();
  float max_bckp = array.max();

  if (k > 0.f)
    clamp_smooth(array, vmin, vmax, k);
  else
    clamp(array, vmin, vmax);

  remap(array, min_bckp, max_bckp);
}

void saturate_percentile(Array &array,
                         float  percentile_low,
                         float  percentile_high,
                         float  k)
{
  glm::vec2 range = array.range_percentile(percentile_low, percentile_high);
  saturate(array, range.x, range.y, k);
}

} // namespace hmap
