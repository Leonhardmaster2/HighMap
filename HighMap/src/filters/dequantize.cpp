/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>
#include <cstdint>

#include "highmap/array.hpp"
#include "highmap/filters.hpp"
#include "highmap/primitives/random.hpp"

namespace hmap
{

Array dequantize(const Array  &array,
                 std::uint32_t seed,
                 float         dither_amplitude,
                 int           filtering_iterations)
{
  const glm::ivec2 &shape = array.shape;
  Array             out(shape);

  out = array + white(shape, 0.f, dither_amplitude, seed);

  float sigma = 0.125f;
  laplace(out, sigma, filtering_iterations);

  return out;
}

Array quantize(const Array &array, int nlevels, float vmin, float vmax)
{
  const glm::ivec2 &shape = array.shape;
  Array             out(shape);

  if (nlevels <= 1) return out;

  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
    {
      float v = array(i, j);
      v = glm::clamp(v, vmin, vmax);
      float q = std::round(v * float(nlevels - 1)) / float(nlevels - 1);
      out(i, j) = q;
    }

  return out;
}

Array quantize(const Array &array, int nlevels)
{
  return quantize(array, nlevels, array.min(), array.max());
}

} // namespace hmap
