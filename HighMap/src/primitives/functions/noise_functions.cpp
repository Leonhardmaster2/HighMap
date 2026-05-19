/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <bits/std_abs.h> // for abs
#include <stddef.h>       // for size_t

#include <algorithm>  // for copy, fill_n, max
#include <array>      // for array
#include <cstdint>    // for uint32_t
#include <functional> // for function
#include <memory>     // for unique_ptr
#include <stdexcept>  // for invalid_argument
#include <vector>     // for vector

#include "FastNoiseLite.h"    // for FastNoiseLite
#include "delaunator-cpp.hpp" // for Delaunator

#include "highmap/array.hpp"                   // for Array, operator*
#include "highmap/functions.hpp"               // for NoiseFunction, NoiseType
#include "highmap/geometry/point_sampling.hpp" // for PointSamplingMethod
#include "highmap/primitives.hpp"              // for white
#include "highmap/range.hpp"                   // for clamp_min_smooth, max...

namespace hmap
{

//----------------------------------------------------------------------
// derived from NoiseFunction class
//----------------------------------------------------------------------

PerlinFunction::PerlinFunction(glm::vec2 kw, std::uint32_t seed)
    : NoiseFunction(kw, seed)
{
  this->set_seed(seed);
  this->noise.SetFrequency(1.f);
  this->noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);

  this->set_delegate(
      [this](float x, float y, float)
      { return this->noise.GetNoise(this->kw.x * x, this->kw.y * y); });
}

PerlinBillowFunction::PerlinBillowFunction(glm::vec2 kw, std::uint32_t seed)
    : NoiseFunction(kw, seed)
{
  this->set_seed(seed);
  this->noise.SetFrequency(1.f);
  this->noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);

  this->set_delegate(
      [this](float x, float y, float)
      {
        float value = this->noise.GetNoise(this->kw.x * x, this->kw.y * y);
        return 2.f * std::abs(value) - 1.f;
      });
}

PerlinHalfFunction::PerlinHalfFunction(glm::vec2     kw,
                                       std::uint32_t seed,
                                       float         k)
    : NoiseFunction(kw, seed), k(k)
{
  this->set_seed(seed);
  this->noise.SetFrequency(1.f);
  this->noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);

  this->set_delegate(
      [this](float x, float y, float)
      {
        float value = this->noise.GetNoise(this->kw.x * x, this->kw.y * y);
        return clamp_min_smooth(value, 0.f, this->k);
      });
}

PerlinMixFunction::PerlinMixFunction(glm::vec2 kw, std::uint32_t seed)
    : NoiseFunction(kw, seed)
{
  this->set_seed(seed);
  this->noise.SetFrequency(1.f);
  this->noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);

  this->set_delegate(
      [this](float x, float y, float)
      {
        float value = this->noise.GetNoise(this->kw.x * x, this->kw.y * y);
        return 0.5f * value + std::abs(value) - 0.5f;
      });
}

Simplex2Function::Simplex2Function(glm::vec2 kw, std::uint32_t seed)
    : NoiseFunction(kw, seed)
{
  this->set_seed(seed);
  this->noise.SetFrequency(0.5f);
  this->noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);

  this->set_delegate(
      [this](float x, float y, float)
      { return this->noise.GetNoise(this->kw.x * x, this->kw.y * y); });
}

Simplex2SFunction::Simplex2SFunction(glm::vec2 kw, std::uint32_t seed)
    : NoiseFunction(kw, seed)
{
  this->set_seed(seed);
  this->noise.SetFrequency(0.5f);
  this->noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2S);

  this->set_delegate(
      [this](float x, float y, float)
      { return this->noise.GetNoise(this->kw.x * x, this->kw.y * y); });
}

ValueNoiseFunction::ValueNoiseFunction(glm::vec2 kw, std::uint32_t seed)
    : NoiseFunction(kw, seed)
{
  this->set_seed(seed);
  this->noise.SetFrequency(1.f);
  this->noise.SetNoiseType(FastNoiseLite::NoiseType_Value);

  this->set_delegate(
      [this](float x, float y, float)
      { return this->noise.GetNoise(this->kw.x * x, this->kw.y * y); });
}

ValueCubicNoiseFunction::ValueCubicNoiseFunction(glm::vec2     kw,
                                                 std::uint32_t seed)
    : NoiseFunction(kw, seed)
{
  this->set_seed(seed);
  this->noise.SetFrequency(1.f);
  this->noise.SetNoiseType(FastNoiseLite::NoiseType_ValueCubic);

  this->set_delegate(
      [this](float x, float y, float)
      { return 1.43f * this->noise.GetNoise(this->kw.x * x, this->kw.y * y); });
}

ValueDelaunayNoiseFunction::ValueDelaunayNoiseFunction(glm::vec2     kw,
                                                       std::uint32_t seed)
    : NoiseFunction(kw, seed)
{
  this->set_kw(kw);
  this->set_seed(seed);
  this->update_interpolation_function();
}

void ValueDelaunayNoiseFunction::update_interpolation_function()
{
  // --- generate 'n' random grid points
  int n = (int)(this->kw.x * this->kw.y);

  std::vector<float> x(n);
  std::vector<float> y(n);
  std::vector<float> value(n);

  auto xy = random_points(n,
                          this->seed,
                          PointSamplingMethod::RND_LHS,
                          {0.f, 1.f, 0.f, 1.f});
  x = xy[0];
  y = xy[1];

  expand_points_domain(x, y, value, {0.f, 1.f, 0.f, 1.f});

  // --- interpolation function

  // triangulate
  std::vector<float> coords(2 * x.size());

  for (size_t k = 0; k < x.size(); k++)
  {
    coords[2 * k] = x[k];
    coords[2 * k + 1] = y[k];
  }

  delaunator::Delaunator<float> d(coords);

  // compute and store triangles area
  std::vector<float> area(d.triangles.size());

  for (size_t k = 0; k < d.triangles.size(); k += 3)
  {
    int p0 = d.triangles[k];
    int p1 = d.triangles[k + 1];
    int p2 = d.triangles[k + 2];

    // true area
    area[k] = 0.5f * (-y[p1] * x[p2] + y[p0] * (-x[p1] + x[p2]) +
                      x[p0] * (y[p1] - y[p2]) + x[p1] * y[p2]);

    // but stored like this to avoid doing it at each evaluation while
    // interpolating
    area[k] = 1.f / (2.f * area[k]);
  }

  auto itp_fct = [x, y, value, d, area](float x_, float y_, float)
  {
    // https://stackoverflow.com/questions/2049582

    // compute barycentric coordinates to find in which triangle the
    // point (x_, y_) is inside
    for (size_t k = 0; k < d.triangles.size(); k += 3)
    {
      int p0 = d.triangles[k];
      int p1 = d.triangles[k + 1];
      int p2 = d.triangles[k + 2];

      float s = area[k] * (y[p0] * x[p2] - x[p0] * y[p2] +
                           (y[p2] - y[p0]) * x_ + (x[p0] - x[p2]) * y_);
      float t = area[k] * (x[p0] * y[p1] - y[p0] * x[p1] +
                           (y[p0] - y[p1]) * x_ + (x[p1] - x[p0]) * y_);

      if (s >= 0.f && t >= 0.f && s + t <= 1.f)
        return value[p0] + s * (value[p1] - value[p0]) +
               t * (value[p2] - value[p0]);
    }

    return 1.f;
  };

  this->set_delegate(itp_fct);
}

ValueLinearNoiseFunction::ValueLinearNoiseFunction(glm::vec2     kw,
                                                   std::uint32_t seed)
    : NoiseFunction(kw, seed)
{
  this->set_kw(kw);
  this->set_seed(seed);
  this->update_interpolation_function();
}

void ValueLinearNoiseFunction::update_interpolation_function()
{
  // generate random values on a regular coarse grid (adjust extent
  // according to the input noise in order to avoid "holes" in the
  // data for large noise displacement)
  glm::vec4 bbox = {-1.f, 2.f, -1.f, 2.f}; // bounding box

  float lx = bbox.y - bbox.x;
  float ly = bbox.w - bbox.z;

  glm::ivec2 shape_base = glm::ivec2((int)(this->kw.x * lx) + 1,
                                     (int)(this->kw.y * ly) + 1);

  Array value = 2.f * white(shape_base, 0.f, 1.f, seed) - 1.f;

  // corresponding grids
  std::vector<float> xv(shape_base.x);
  std::vector<float> yv(shape_base.y);

  for (int i = 0; i < shape_base.x; i++)
    xv[i] = bbox.x + lx * (float)i / (float)(shape_base.x - 1);

  for (int j = 0; j < shape_base.y; j++)
    yv[j] = bbox.z + ly * (float)j / (float)(shape_base.y - 1);

  auto itp_fct = [xv, yv, value, bbox](float x_, float y_, float)
  {
    float xn = (x_ - bbox.x) / (bbox.y - bbox.x) * (float)(value.shape.x - 1);
    float yn = (y_ - bbox.z) / (bbox.w - bbox.z) * (float)(value.shape.y - 1);

    int in = (int)xn;
    int jn = (int)yn;

    float u = xn - (float)in;
    float v = yn - (float)jn;

    if (in == value.shape.x - 1)
    {
      in = value.shape.x - 2;
      u = 1.f;
    }

    if (jn == value.shape.y - 1)
    {
      jn = value.shape.y - 2;
      v = 1.f;
    }

    return value.get_value_bilinear_at(in, jn, u, v);
  };

  this->set_delegate(itp_fct);
}

WorleyFunction::WorleyFunction(glm::vec2     kw,
                               std::uint32_t seed,
                               bool          return_cell_value)
    : NoiseFunction(kw, seed)
{
  this->set_seed(seed);
  this->noise.SetFrequency(1.f);
  this->noise.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
  this->noise.SetCellularJitter(1.f);

  if (return_cell_value)
    this->noise.SetCellularReturnType(
        FastNoiseLite::CellularReturnType_CellValue);
  else
    this->noise.SetCellularReturnType(
        FastNoiseLite::CellularReturnType_Distance);

  this->set_delegate(
      [this](float x, float y, float)
      {
        return 1.66f *
               (0.4f + this->noise.GetNoise(this->kw.x * x, this->kw.y * y));
      });
}

WorleyDoubleFunction::WorleyDoubleFunction(glm::vec2     kw,
                                           std::uint32_t seed,
                                           float         ratio,
                                           float         k)
    : NoiseFunction(kw, seed), ratio(ratio), k(k)
{
  this->set_seed(seed);

  this->noise1.SetFrequency(1.0f);
  this->noise2.SetFrequency(1.0f);

  this->noise1.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
  this->noise2.SetNoiseType(FastNoiseLite::NoiseType_Cellular);

  this->set_delegate(
      [this](float x, float y, float ctrl_param)
      {
        float local_ratio = ctrl_param * this->ratio;

        float w1 = this->noise1.GetNoise(this->kw.x * x, this->kw.y * y);
        float w2 = this->noise2.GetNoise(this->kw.x * x, this->kw.y * y);
        if (this->k)
          return maximum_smooth(local_ratio * w1,
                                (1.f - local_ratio) * w2,
                                this->k);
        else
          return std::max(local_ratio * w1, (1.f - local_ratio) * w2);
      });
}

// --- helper

std::unique_ptr<NoiseFunction> create_noise_function_from_type(
    NoiseType     noise_type,
    glm::vec2     kw,
    std::uint32_t seed)
{
  switch (noise_type)
  {
    // clang-format off
  case (NoiseType::PARBERRY):
    throw std::invalid_argument("create_noise_function_from_type: PARBERRY noise function not available");
  case (NoiseType::PERLIN):
    return std::unique_ptr<NoiseFunction>(new PerlinFunction(kw, seed));
  case (NoiseType::PERLIN_BILLOW):
    return std::unique_ptr<NoiseFunction>(new PerlinBillowFunction(kw, seed));
  case (NoiseType::PERLIN_HALF):
  {
    float k = 0.5f;
    return std::unique_ptr<NoiseFunction>(new PerlinHalfFunction(kw, seed, k));
  }
  case (NoiseType::SIMPLEX2):
    return std::unique_ptr<NoiseFunction>(new Simplex2Function(kw, seed));
  case (NoiseType::SIMPLEX2S):
    return std::unique_ptr<NoiseFunction>(new Simplex2SFunction(kw, seed));
  case (NoiseType::VALUE):
    return std::unique_ptr<NoiseFunction>(new ValueNoiseFunction(kw, seed));
  case (NoiseType::VALUE_CUBIC):
    return std::unique_ptr<NoiseFunction>(new ValueCubicNoiseFunction(kw, seed));
  case (NoiseType::VALUE_DELAUNAY):
    return std::unique_ptr<NoiseFunction>(new ValueDelaunayNoiseFunction(kw, seed));
  case (NoiseType::VALUE_LINEAR):
    return std::unique_ptr<NoiseFunction>(new ValueLinearNoiseFunction(kw, seed));
  case (NoiseType::WORLEY):
    return std::unique_ptr<NoiseFunction>(new WorleyFunction(kw, seed, false));
  case (NoiseType::WORLEY_DOUBLE):
  {
    float ratio = 0.5f;
    float k = 0.5f;
    return std::unique_ptr<NoiseFunction>(new WorleyDoubleFunction(kw, seed, ratio, k));
  }
  case (NoiseType::WORLEY_VALUE):
    return std::unique_ptr<NoiseFunction>(new WorleyFunction(kw, seed, true));
    // clang-format on
  }

  return nullptr;
}

} // namespace hmap
