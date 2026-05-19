/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm> // for max
#include <cmath>     // for sqrt
#include <stdexcept> // for invalid_argument

#include "highmap/array.hpp"      // for Array
#include "highmap/range.hpp"      // for remap
#include "highmap/statistics.hpp" // for NormalizationMethod, variance

namespace hmap
{

void normalize(Array &array, NormalizationMethod method)
{
  switch (method)
  {
  case NormalizationMethod::NM_MIN_MAX:
  {
    remap(array);
    return;
  }
  //
  case NormalizationMethod::NM_STANDARDIZE:
  {
    float mean = array.mean();
    float sigma = std::sqrt(variance(array, &mean));
    array = (array - mean) / sigma;
    return;
  }
  //
  case NormalizationMethod::NM_ROBUST:
  {
    float     median = array.median();
    glm::vec2 range = array.range_percentile(0.25f, 0.75f);
    float     denom = std::max(range.y - range.x, 1e-8f);
    array = (array - median) / denom;
    return;
  }
  //
  default:
    throw std::invalid_argument(
        "hmap::normalized: invalid normalization method.");
  }
}

Array normalized(const Array &array, NormalizationMethod method)
{
  Array out = array;
  normalize(out, method);
  return out;
}

} // namespace hmap
