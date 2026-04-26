/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "macrologger.h"

#include "highmap/algebra.hpp"
#include "highmap/array.hpp"
#include "highmap/interpolate_array.hpp"
#include "highmap/operator.hpp"
#include "highmap/range.hpp"
#include "highmap/transform.hpp"

#include "highmap/internal/vector_utils.hpp"

namespace hmap
{

void Array::argmax(float &max, int &im, int &jm) const
{
  max = -std::numeric_limits<float>::infinity();
  im = -1;
  jm = -1;

  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
    {
      float v = (*this)(i, j);
      if (v > max)
      {
        max = v;
        im = i;
        jm = j;
      }
    }
}

std::vector<float> Array::col_to_vector(int j)
{
  std::vector<float> vec(this->shape.x);
  for (int i = 0; i < this->shape.x; i++)
    vec[i] = (*this)(i, j);
  return vec;
}

void Array::depose_amount_bilinear_at(int   i,
                                      int   j,
                                      float u,
                                      float v,
                                      float amount)
{
  (*this)(i, j) += amount * (1 - u) * (1 - v);
  (*this)(i + 1, j) += amount * u * (1 - v);
  (*this)(i, j + 1) += amount * (1 - u) * v;
  (*this)(i + 1, j + 1) += amount * u * v;
}

void Array::depose_amount_kernel_bilinear_at(int   i,
                                             int   j,
                                             float u,
                                             float v,
                                             int   ir,
                                             float amount)
{
  Array kernel = Array(glm::ivec2(2 * ir + 1, 2 * ir + 1));

  // compute kernel first
  for (int p = -ir; p < ir + 1; p++)
  {
    for (int q = -ir; q < ir + 1; q++)
    {
      float x = (float)p - u;
      float y = (float)q - v;
      float r = std::max(0.f, 1.f - std::hypot(x, y));
      kernel(p + ir, q + ir) = r;
    }
  }
  kernel.normalize();

  // perform deposition
  this->depose_amount_kernel_at(i, j, kernel, amount);
}

void Array::depose_amount_kernel_at(int          i,
                                    int          j,
                                    const Array &kernel,
                                    float        amount)
{
  const int ir = (kernel.shape.x - 1) / 2;
  const int jr = (kernel.shape.y - 1) / 2;

  for (int p = 0; p < kernel.shape.x; p++)
  {
    for (int q = 0; q < kernel.shape.y; q++)
    {
      (*this)(i + p - ir, j + q - jr) += amount * kernel(p, q);
    }
  }
}

void Array::dump(const std::string &fname) const
{
  this->infos(fname);
  this->to_png_grayscale(fname, CV_16U);
}

void Array::dump_histogram(const std::string &msg) const
{
  std::cout << "Array: " << msg << "\n";
  std::cout << make_histogram(this->vector, 32, 6);
}

Array Array::extract_slice(glm::ivec4 idx) const
{
  Array array_out = Array(glm::ivec2(idx.y - idx.x, idx.w - idx.z));

  for (int j = idx.z; j < idx.w; j++)
    for (int i = idx.x; i < idx.y; i++)
      array_out(i - idx.x, j - idx.z) = (*this)(i, j);

  return array_out;
}

Array Array::extract_slice(int i1, int i2, int j1, int j2) const
{
  glm::ivec4 idx(i1, i2, j1, j2);
  return this->extract_slice(idx);
}

float Array::get_gradient_x_at(int i, int j) const
{
  return 0.5f * ((*this)(i + 1, j) - (*this)(i - 1, j));
}

float Array::get_gradient_y_at(int i, int j) const
{
  return 0.5f * ((*this)(i, j + 1) - (*this)(i, j - 1));
}

float Array::get_gradient_x_bilinear_at(int i, int j, float u, float v) const
{
  float f00 = (*this)(i, j) - (*this)(i - 1, j);
  float f10 = (*this)(i + 1, j) - (*this)(i, j);
  float f01 = (*this)(i, j + 1) - (*this)(i - 1, j + 1);
  float f11 = (*this)(i + 1, j + 1) - (*this)(i, j + 1);

  float a10 = f10 - f00;
  float a01 = f01 - f00;
  float a11 = f11 - f10 - f01 + f00;

  return f00 + a10 * u + a01 * v + a11 * u * v;
}

float Array::get_gradient_y_bilinear_at(int i, int j, float u, float v) const
{
  float f00 = (*this)(i, j) - (*this)(i, j - 1);
  float f10 = (*this)(i + 1, j) - (*this)(i + 1, j - 1);
  float f01 = (*this)(i, j + 1) - (*this)(i, j);
  float f11 = (*this)(i + 1, j + 1) - (*this)(i + 1, j);

  float a10 = f10 - f00;
  float a01 = f01 - f00;
  float a11 = f11 - f10 - f01 + f00;

  return f00 + a10 * u + a01 * v + a11 * u * v;
}

glm::vec3 Array::get_normal_at(int i, int j) const
{
  glm::vec3 normal;

  normal.x = -this->get_gradient_x_at(i, j);
  normal.y = -this->get_gradient_y_at(i, j);
  normal.z = 1.f;

  float norm = std::hypot(normal.x, normal.y, normal.z);
  normal /= norm;

  return normal;
}

size_t Array::get_sizeof() const
{
  return sizeof(float) * this->vector.size();
}

glm::vec2 Array::normalization_coeff(float vmin, float vmax) const
{
  float a = 0.f;
  float b = 0.f;
  if (vmin != vmax)
  {
    a = 1.f / (vmax - vmin);
    b = -vmin / (vmax - vmin);
  }
  return glm::vec2(a, b);
}

float Array::get_value_bicubic_at(int i, int j, float u, float v) const
{
  float arr[4][4];

  // Get the 4x4 surrounding grid points
  for (int n = -1; n <= 2; ++n)
    for (int m = -1; m <= 2; ++m)
    {
      int ip = std::clamp(i + m, 0, this->shape.x - 1);
      int jp = std::clamp(j + n, 0, this->shape.y - 1);
      arr[m + 1][n + 1] = (*this)(ip, jp);
    }

  // interpolate in the x direction
  float col_results[4];
  for (int i = 0; i < 4; ++i)
    col_results[i] = cubic_interpolate(arr[i], v);

  // interpolate in the y direction
  return cubic_interpolate(col_results, u);
}

float Array::get_value_bilinear_at(int i, int j, float u, float v) const
{
  if (i == this->shape.x - 1)
  {
    i = this->shape.x - 2;
    u = 1.f;
  }

  if (j == this->shape.y - 1)
  {
    j = this->shape.y - 2;
    v = 1.f;
  }

  float a10 = (*this)(i + 1, j) - (*this)(i, j);
  float a01 = (*this)(i, j + 1) - (*this)(i, j);
  float a11 = (*this)(i + 1, j + 1) - (*this)(i + 1, j) - (*this)(i, j + 1) +
              (*this)(i, j);

  return (*this)(i, j) + a10 * u + a01 * v + a11 * u * v;
}

float Array::get_value_nearest(float x, float y, glm::vec4 bbox)
{
  int i = (int)(std::clamp((x - bbox.x) / (bbox.y - bbox.x), 0.f, 1.f) *
                (this->shape.x - 1));
  int j = (int)(std::clamp((y - bbox.z) / (bbox.w - bbox.z), 0.f, 1.f) *
                (this->shape.y - 1));
  return (*this)(i, j);
}

int Array::linear_index(int i, int j) const
{
  return j * this->shape.x + i;
}

glm::ivec2 Array::linear_index_reverse(int k) const
{
  glm::ivec2 ij;
  ij.y = std::floor(k / shape.x);
  ij.x = k - ij.y * shape.x;
  return ij;
}

float Array::max() const
{
  return *std::max_element(this->vector.begin(), this->vector.end());
}

float Array::mean() const
{
  return this->sum() / (float)this->size();
};

float Array::median() const
{
  return compute_median(this->vector);
}

float Array::min() const
{
  return *std::min_element(this->vector.begin(), this->vector.end());
};

void Array::normalize()
{
  float sum = this->sum();

  std::transform(this->vector.begin(),
                 this->vector.end(),
                 this->vector.begin(),
                 [&sum](float v) { return v / sum; });
}

float Array::ptp() const
{
  return this->max() - this->min();
}

glm::vec2 Array::range() const
{
  auto [min_it, max_it] = std::minmax_element(this->vector.begin(),
                                              this->vector.end());
  return glm::vec2(*min_it, *max_it);
}

glm::vec2 Array::range_percentile(float p_low, float p_high, size_t bins) const
{
  const std::vector<float> &data = this->vector;

  const glm::vec2 range = this->range();
  float           min_val = range.x;
  float           max_val = range.y;

  std::vector<size_t> hist(bins, 0);

  // fill histogram
  for (float v : data)
  {
    size_t idx = std::min<size_t>(
        bins - 1,
        size_t((v - min_val) / (max_val - min_val) * bins));
    hist[idx]++;
  }

  // accumulate
  size_t total = data.size();
  size_t low_target = size_t(p_low * total);
  size_t high_target = size_t(p_high * total);

  size_t acc = 0;
  float  low = min_val;
  float  high = max_val;

  for (size_t i = 0; i < bins; ++i)
  {
    acc += hist[i];

    if (acc >= low_target && low == min_val)
      low = min_val + (max_val - min_val) * (float(i) / bins);

    if (acc >= high_target)
    {
      high = min_val + (max_val - min_val) * (float(i) / bins);
      break;
    }
  }

  return {low, high};
}

Array Array::remapped() const
{
  Array out = *this;
  remap(out);
  return out;
}

Array Array::resample_to_shape(glm::ivec2 new_shape) const
{
  return this->resample_to_shape_bilinear(new_shape);
}

Array Array::resample_to_shape_bicubic(glm::ivec2 new_shape) const
{
  Array array_out = Array(new_shape);
  interpolate_array_bicubic(*this, array_out);

  return array_out;
}

Array Array::resample_to_shape_bilinear(glm::ivec2 new_shape) const
{
  Array array_out = Array(new_shape);
  interpolate_array_bilinear(*this, array_out);

  return array_out;
}

Array Array::resample_to_shape_nearest(glm::ivec2 new_shape) const
{
  Array array_out = Array(new_shape);
  interpolate_array_nearest(*this, array_out);

  return array_out;
}

std::vector<float> Array::row_to_vector(int i)
{
  std::vector<float> vec(this->shape.y);
  for (int j = 0; j < this->shape.y; j++)
    vec[j] = (*this)(i, j);
  return vec;
}

void Array::set_slice(glm::ivec4 idx, float value)
{
  for (int i = idx.x; i < idx.y; i++)
    for (int j = idx.z; j < idx.w; j++)
      (*this)(i, j) = value;
}

void Array::set_slice(glm::ivec4 idx, const Array &array)
{
  for (int i = idx.x; i < idx.y; i++)
    for (int j = idx.z; j < idx.w; j++)
      (*this)(i, j) = array(i - idx.x, j - idx.z);
}

int Array::size() const
{
  return this->shape.x * this->shape.y;
}

float Array::std() const
{
  float mean = this->mean();
  Array a2 = (*this) - mean;
  a2 *= a2;
  return a2.mean();
};

float Array::sum() const
{
  return std::accumulate(this->vector.begin(), this->vector.end(), 0.f);
}

std::vector<float> Array::unique_values() const
{
  std::vector<float> v = this->vector;
  vector_unique_values(v);
  return v;
}

} // namespace hmap
