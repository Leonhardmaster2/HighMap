/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <stddef.h> // for size_t

#include <vector> // for vector

#include "highmap/array.hpp"          // for Array
#include "highmap/boundary.hpp"       // for extrapolate_borders
#include "highmap/geometry/grids.hpp" // for grid_xy_vector
#include "highmap/interpolate2d.hpp"  // for NaturalNeighborInterpolator

namespace hmap
{

Array interpolate2d_nni(glm::ivec2                shape,
                        const std::vector<float> &x,
                        const std::vector<float> &y,
                        const std::vector<float> &values,
                        const Array              *p_noise_x,
                        const Array              *p_noise_y,
                        glm::vec4                 bbox)
{
  std::vector<float> xg, yg;
  grid_xy_vector(xg, yg, shape, bbox, /* endpoint */ false);

  std::vector<float> xout, yout;
  xout.reserve(shape.x * shape.y);
  yout.reserve(shape.x * shape.y);

  if (p_noise_x || p_noise_y)
  {
    for (int j = 0; j < shape.y; j++)
      for (int i = 0; i < shape.x; i++)
      {
        float dx = p_noise_x ? (*p_noise_x)(i, j) : 0.f;
        float dy = p_noise_y ? (*p_noise_y)(i, j) : 0.f;

        xout.push_back(xg[i] + dx);
        yout.push_back(yg[j] + dy);
      }
  }
  else
  {
    for (int j = 0; j < shape.y; j++)
      for (int i = 0; i < shape.x; i++)
      {
        xout.push_back(xg[i]);
        yout.push_back(yg[j]);
      }
  }

  NaturalNeighborInterpolator nn;
  nn.setup_output_points(xout, yout);
  nn.build(x, y);

  std::vector<float> result;
  nn.interpolate(values, result);

  Array array_out = Array(shape);

  size_t k = 0;
  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
    {
      array_out(i, j) = result[k];
      k++;
    }

  extrapolate_borders(array_out);

  return array_out;
}

} // namespace hmap
