/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>

#include "FastNoiseLite.h"
#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/operator.hpp"
#include "highmap/primitives.hpp"

namespace hmap
{

Array caldera(glm::ivec2   shape,
              float        radius,
              float        sigma_inner,
              float        sigma_outer,
              float        z_bottom,
              const Array *p_noise,
              float        noise_r_amp,
              float        noise_z_ratio,
              glm::vec2    center,
              glm::vec4    bbox)
{
  Array z = Array(shape);

  glm::vec2 shift = {bbox.x, bbox.z};
  glm::vec2 scale = {bbox.y - bbox.x, bbox.w - bbox.z};

  int ic = (int)((center.x - shift.x) / scale.x * z.shape.x);
  int jc = (int)((center.y - shift.y) / scale.y * z.shape.y);

  float si2 = sigma_inner * sigma_inner;
  float so2 = sigma_outer * sigma_outer;

  if (p_noise)
  {
    for (int j = 0; j < z.shape.y; j++)
      for (int i = 0; i < z.shape.x; i++)
      {
        float r = std::hypot((float)(i - ic), (float)(j - jc)) - radius;

        r += noise_r_amp * (2 * (*p_noise)(i, j) - 1);

        if (r < 0.f)
          z(i, j) = z_bottom + std::exp(-0.5f * r * r / si2) * (1 - z_bottom);
        else
          z(i, j) = 1 / (1 + r * r / so2);

        z(i, j) *= 1.f + noise_z_ratio * (2.f * (*p_noise)(i, j) - 1.f);
      }
  }
  else
  {
    for (int j = 0; j < z.shape.y; j++)
      for (int i = 0; i < z.shape.x; i++)
      {
        float r = std::hypot((float)(i - ic), (float)(j - jc)) - radius;

        if (r < 0.f)
          z(i, j) = z_bottom + std::exp(-0.5f * r * r / si2) * (1 - z_bottom);
        else
          z(i, j) = 1 / (1 + r * r / so2);
      }
  }

  return z;
}

Array caldera(glm::ivec2 shape,
              float      radius,
              float      sigma_inner,
              float      sigma_outer,
              float      z_bottom,
              glm::vec2  center,
              glm::vec4  bbox)
{
  Array noise = constant(shape, 0.f);
  Array z = caldera(shape,
                    radius,
                    sigma_inner,
                    sigma_outer,
                    z_bottom,
                    nullptr,
                    0.f,
                    0.f,
                    center,
                    bbox);
  return z;
}

Array peak(glm::ivec2   shape,
           float        radius,
           const Array *p_noise,
           float        noise_r_amp,
           float        noise_z_ratio,
           glm::vec4    bbox)
{
  glm::vec2 shift = {bbox.x, bbox.z};
  glm::vec2 scale = {bbox.y - bbox.x, bbox.w - bbox.z};

  Array z = Array(shape);
  int   ic = (int)((0.5f - shift.x) / scale.x * z.shape.x);
  int   jc = (int)((0.5f - shift.y) / scale.y * z.shape.y);

  if (!p_noise)
  {
    for (int j = 0; j < z.shape.y; j++)
      for (int i = 0; i < z.shape.x; i++)
      {
        float r = std::hypot((float)(i - ic), (float)(j - jc)) / radius;

        if (r < 1.f) z(i, j) = 1.f - r * r * (3.f - 2.f * r);
      }
  }
  else
  {
    for (int j = 0; j < z.shape.y; j++)
      for (int i = 0; i < z.shape.x; i++)
      {
        float r = std::hypot((float)(i - ic), (float)(j - jc)) / radius;
        r += r * noise_r_amp / radius * (2 * (*p_noise)(i, j) - 1);

        if (r < 1.f) z(i, j) = 1.f - r * r * (3.f - 2.f * r);

        z(i, j) *= 1.f + noise_z_ratio * (2.f * (*p_noise)(i, j) - 1.f);
      }
  }

  return z;
}

} // namespace hmap
