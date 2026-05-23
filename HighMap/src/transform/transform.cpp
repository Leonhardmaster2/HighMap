/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "highmap/array.hpp"
#include "highmap/boundary.hpp"
#include "highmap/functions.hpp"
#include "highmap/geometry/grids.hpp"
#include "highmap/operator.hpp"
#include "highmap/primitives/functions.hpp"
#include "highmap/transform.hpp"

namespace hmap
{

void flip_lr(Array &array)
{
  for (int j = 0; j < array.shape.y; j++)
    for (int i = 0; i < (int)(0.5f * array.shape.y); i++)
      std::swap(array(i, j), array(array.shape.x - i - 1, j));
}

void flip_ud(Array &array)
{
  for (int j = 0; j < (int)(0.5f * array.shape.y); j++)
    for (int i = 0; i < array.shape.x; i++)
      std::swap(array(i, j), array(i, array.shape.y - j - 1));
}

void radial_displacement_to_xy(const Array &dr,
                               Array       &dx,
                               Array       &dy,
                               float        smoothing,
                               glm::vec2    center,
                               glm::vec4    bbox)
{
  glm::ivec2 shape = dr.shape;
  dx = Array(shape);
  dy = Array(shape);

  std::vector<float> x, y;
  grid_xy_vector(x, y, shape, bbox, false); // no endpoint

  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
    {
      float xr = x[i] - center.x;
      float yr = y[j] - center.y;
      float r2 = smoothing * std::hypot(xr, yr);
      float factor = r2 / (1.f + r2);
      float theta = std::atan2(yr, xr);
      dx(i, j) = factor * dr(i, j) * std::cos(theta);
      dy(i, j) = factor * dr(i, j) * std::sin(theta);
    }
}

void rot180(Array &array)
{
  flip_lr(array);
  flip_ud(array);
}

void rot270(Array &array)
{
  array = transpose(array);
  flip_lr(array);
}

void rot90(Array &array)
{
  array = transpose(array);
  flip_ud(array);
}

void rotate(Array &array, float angle, bool zoom_in, bool zero_padding)
{
  float ca = std::cos(-angle / 180.f * M_PI);
  float sa = std::sin(-angle / 180.f * M_PI);

  // create a larger array filled using symmetry to have a domain
  // large enough to avoid 'holes' while interpolating
  int nbuffer = std::max((int)(0.25f * array.shape.x),
                         (int)(0.25f * array.shape.y));
  nbuffer = std::max(1, nbuffer);

  Array array_bf = generate_buffered_array(
      array,
      glm::ivec4(nbuffer, nbuffer, nbuffer, nbuffer),
      zero_padding);

  float xc = 0.5f * array.shape.x;
  float yc = 0.5f * array.shape.y;

  float zoom = zoom_in ? 1.0f / (fabs(ca) + fabs(sa)) : 1.f;

  for (int j = 0; j < array.shape.y; j++)
    for (int i = 0; i < array.shape.x; i++)
    {
      float x = xc + zoom * (ca * (i - xc) - sa * (j - yc));
      float y = yc + zoom * (sa * (i - xc) + ca * (j - yc));

      // corresponding nearest cells in buffered array (and bilinear
      // interpolation parameters)
      int ix = std::clamp((int)x, 0, array.shape.x - 1);
      int jy = std::clamp((int)y, 0, array.shape.y - 1);

      float u = std::clamp(x - (float)ix, 0.f, 1.f);
      float v = std::clamp(y - (float)jy, 0.f, 1.f);

      int ib = nbuffer + ix;
      int jb = nbuffer + jy;

      array(i, j) = array_bf.get_value_bilinear_at(ib, jb, u, v);
    }
}

void rotate_displacement(const Array &delta, float angle, Array &dx, Array &dy)
{
  const float alpha = angle / 180.f * M_PI;
  dx = delta * std::cos(alpha);
  dy = delta * std::sin(alpha);
}

Array translate(const Array &array,
                float        dx,
                float        dy,
                bool         periodic,
                const Array *p_noise_x,
                const Array *p_noise_y,
                glm::vec4    bbox)
{
  hmap::ArrayFunction f = hmap::ArrayFunction(array,
                                              glm::vec2(1.f, 1.f),
                                              periodic);

  Array dx_array = constant(array.shape, -dx);
  Array dy_array = constant(array.shape, -dy);

  if (p_noise_x) dx_array += *p_noise_x;

  if (p_noise_y) dy_array += *p_noise_y;

  Array array_out = Array(array.shape);

  fill_array_using_xy_function(array_out,
                               bbox,
                               nullptr,
                               &dx_array,
                               &dy_array,
                               nullptr,
                               f.get_delegate());

  return array_out;
}

Array transpose(const Array &array)
{
  Array array_out = Array(glm::ivec2(array.shape.y, array.shape.x));

  for (int j = 0; j < array.shape.y; j++)
    for (int i = 0; i < array.shape.x; i++)
      array_out(j, i) = array(i, j);

  return array_out;
}

Array zoom(const Array &array,
           float        zoom_factor,
           bool         periodic,
           glm::vec2    center,
           const Array *p_noise_x,
           const Array *p_noise_y,
           glm::vec4    bbox)
{

  hmap::ArrayFunction f = hmap::ArrayFunction(
      array,
      glm::vec2(1.f / zoom_factor, 1.f / zoom_factor),
      periodic);

  Array array_out = Array(array.shape);

  glm::vec4 bbox2 = {bbox.x + center.x,
                     bbox.y + center.x,
                     bbox.z + center.y,
                     bbox.w + center.y};

  fill_array_using_xy_function(array_out,
                               bbox2,
                               nullptr,
                               p_noise_x,
                               p_noise_y,
                               nullptr,
                               f.get_delegate());

  return array_out;
}

} // namespace hmap
