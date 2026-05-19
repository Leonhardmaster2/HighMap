/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

#include <algorithm> // for clamp
#include <cmath>     // for ceil
#include <vector>    // for vector

#include "highmap/array.hpp"    // for Array
#include "highmap/boundary.hpp" // for generate_buffered_array
#include "highmap/convolve.hpp" // for convolve1d_i, convolve1d_j, conv...

namespace hmap
{

Array convolve1d_i(const Array &array, const std::vector<float> &kernel)
{
  Array     array_out = Array(array.shape);
  const int nk = (int)kernel.size();
  const int i1 = nk / 2;

  for (int p = 0; p < nk; p++)
  {
    for (int i = 0; i < array.shape.x; i++)
    {
      const int ii = std::clamp(i + p - i1, 0, array.shape.x - 1);
      for (int j = 0; j < array.shape.y; j++)
        array_out(i, j) += array(ii, j) * kernel[p];
    }
  }
  return array_out;
}

Array convolve1d_j(const Array &array, const std::vector<float> &kernel)
{
  Array     array_out = Array(array.shape);
  const int nk = (int)kernel.size();
  const int j1 = nk / 2;

  for (int p = 0; p < nk; p++)
  {
    for (int i = 0; i < array.shape.x; i++)
    {
      for (int j = 0; j < array.shape.y; j++)
      {
        const int jj = std::clamp(j + p - j1, 0, array.shape.y - 1);
        array_out(i, j) += array(i, jj) * kernel[p];
      }
    }
  }
  return array_out;
}

Array convolve2d(const Array &array, const Array &kernel)
{
  const int i1 = (int)ceil(0.5f * (float)kernel.shape.x);
  const int i2 = kernel.shape.x - i1;
  const int j1 = (int)ceil(0.5f * (float)kernel.shape.y);
  const int j2 = kernel.shape.y - j1;

  Array array_buffered = generate_buffered_array(array, {i1, i2, j1, j2});
  Array array_out = convolve2d_truncated(array_buffered, kernel);

  return array_out;
}

Array convolve2d_truncated(const Array &array, const Array &kernel)
{
  Array array_out = Array(glm::ivec2(array.shape.x - kernel.shape.x,
                                     array.shape.y - kernel.shape.y));

  for (int j = 0; j < array_out.shape.y; j++)
    for (int i = 0; i < array_out.shape.x; i++)
      for (int q = 0; q < kernel.shape.y; q++)
        for (int p = 0; p < kernel.shape.x; p++)
          array_out(i, j) += array(i + p, j + q) * kernel(p, q);

  return array_out;
}

} // namespace hmap
