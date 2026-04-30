/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cmath>

#include "highmap/array.hpp"
#include "highmap/statistics.hpp"

namespace hmap
{

float autocorr_length_scale(const Array &array, float max_lag_fraction)
{
  const int ni = array.shape.x;
  const int nj = array.shape.y;

  // --- Global mean and variance

  float mean = array.mean();
  float var = variance(array, &mean);

  // flat array
  if (var == 0.f) return static_cast<float>(std::max(ni, nj));

  // --- Autocorrelation

  const float target = var / 2.7182818284590452354f;
  const int   max_lag = static_cast<int>(std::min(ni, nj) * max_lag_fraction);

  // evaluate unnormalized autocorrelation at a given lag == average
  // of horizontal (i-direction) and vertical (j-direction)
  auto R = [&](int lag) -> float
  {
    float ri = 0.f;
    for (int j = 0; j < nj; ++j)
      for (int i = lag; i < ni; ++i)
        ri += (array(i, j) - mean) * (array(i - lag, j) - mean);
    ri /= ((ni - lag) * nj);

    float rj = 0.0;
    for (int j = lag; j < nj; ++j)
      for (int i = 0; i < ni; ++i)
        rj += (array(i, j) - mean) * (array(i, j - lag) - mean);
    rj /= (ni * (nj - lag));

    return 0.5f * (ri + rj);
  };

  // binary search for lag where R drops below var/e
  int lo = 1, hi = max_lag;
  while (hi - lo > 1)
  {
    int mid = (lo + hi) / 2;
    if (R(mid) > target)
      lo = mid;
    else
      hi = mid;
  }

  // length scale in grid cells
  return static_cast<float>(hi);
}

glm::vec2 autocorr_length_scale_axial(const Array &array,
                                      float        max_lag_fraction)
{
  const int ni = array.shape.x;
  const int nj = array.shape.y;

  // --- Global mean and variance

  float mean = array.mean();
  float var = variance(array, &mean);

  // flat array
  if (var == 0.f) return {static_cast<float>(ni), static_cast<float>(nj)};

  // --- Autocorrelation

  const float target = var / 2.7182818284590452354f;

  // along rows (i-direction)
  auto Ri = [&](int lag)
  {
    float r = 0.0;
    for (int j = 0; j < nj; ++j)
      for (int i = lag; i < ni; ++i)
        r += (array(i, j) - mean) * (array(i - lag, j) - mean);
    return r / ((ni - lag) * nj);
  };

  // along cols (j-direction)
  auto Rj = [&](int lag)
  {
    float r = 0.0;
    for (int j = lag; j < nj; ++j)
      for (int i = 0; i < ni; ++i)
        r += (array(i, j) - mean) * (array(i, j - lag) - mean);
    return r / (ni * (nj - lag));
  };

  auto binary_search_scale = [&](auto &Rfunc, int max_l) -> float
  {
    int lo = 1, hi = max_l;
    while (hi - lo > 1)
    {
      int mid = (lo + hi) / 2;
      if (Rfunc(mid) > target)
        lo = mid;
      else
        hi = mid;
    }
    return static_cast<float>(hi);
  };

  const int max_lag_i = static_cast<int>(ni * max_lag_fraction);
  const int max_lag_j = static_cast<int>(nj * max_lag_fraction);

  return {binary_search_scale(Ri, max_lag_i),
          binary_search_scale(Rj, max_lag_j)};
}

} // namespace hmap
