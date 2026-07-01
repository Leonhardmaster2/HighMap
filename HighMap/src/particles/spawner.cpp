/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cmath>
#include <random>

#include "highmap/particles.hpp"

namespace hmap
{

glm::ivec2 spawn_borders(std::mt19937     &rng,
                         int               nbuffer,
                         const glm::ivec2 &shape)
{
  nbuffer = std::max(1, nbuffer);

  const int nx = shape.x;
  const int ny = shape.y;

  const int top_area = nx * nbuffer;
  const int bottom_area = nx * nbuffer;
  const int left_area = (ny - 2 * nbuffer) * nbuffer;
  const int right_area = (ny - 2 * nbuffer) * nbuffer;

  const int total_area = top_area + bottom_area + left_area + right_area;

  std::uniform_int_distribution<int> dist_area(0, total_area - 1);

  int r = dist_area(rng);

  // top
  if (r < top_area)
  {
    std::uniform_int_distribution<int> dist_x(0, nx - 1);
    std::uniform_int_distribution<int> dist_y(0, nbuffer - 1);

    return {dist_x(rng), dist_y(rng)};
  }

  r -= top_area;

  // bottom
  if (r < bottom_area)
  {
    std::uniform_int_distribution<int> dist_x(0, nx - 1);
    std::uniform_int_distribution<int> dist_y(ny - nbuffer, ny - 1);

    return {dist_x(rng), dist_y(rng)};
  }

  r -= bottom_area;

  // left
  if (r < left_area)
  {
    std::uniform_int_distribution<int> dist_x(0, nbuffer - 1);
    std::uniform_int_distribution<int> dist_y(nbuffer, ny - nbuffer - 1);

    return {dist_x(rng), dist_y(rng)};
  }

  // right
  std::uniform_int_distribution<int> dist_x(nx - nbuffer, nx - 1);
  std::uniform_int_distribution<int> dist_y(nbuffer, ny - nbuffer - 1);

  return {dist_x(rng), dist_y(rng)};
}

glm::ivec2 spawn_disc(std::mt19937     &rng,
                      const glm::ivec2 &center,
                      float             radius,
                      const glm::ivec2 &shape)
{
  std::uniform_real_distribution<float> dist_angle(0.f, 2.f * M_PI);
  std::uniform_real_distribution<float> dist01(0.f, 1.f);

  float theta = dist_angle(rng);

  // sqrt() ensures uniform density over the disc area
  float r = radius * std::sqrt(dist01(rng));

  glm::ivec2 p{static_cast<int>(std::round(center.x + r * std::cos(theta))),
               static_cast<int>(std::round(center.y + r * std::sin(theta)))};

  p.x = std::clamp(p.x, 0, shape.x - 1);
  p.y = std::clamp(p.y, 0, shape.y - 1);

  return p;
}

glm::ivec2 spawn_gaussian(std::mt19937     &rng,
                          const glm::vec2  &center,
                          float             sigma,
                          const glm::ivec2 &shape)
{
  std::normal_distribution<float> dist_x(center.x, sigma);
  std::normal_distribution<float> dist_y(center.y, sigma);

  glm::ivec2 p{static_cast<int>(std::round(dist_x(rng))),
               static_cast<int>(std::round(dist_y(rng)))};

  p.x = std::clamp(p.x, 0, shape.x - 1);
  p.y = std::clamp(p.y, 0, shape.y - 1);

  return p;
}

glm::ivec2 spawn_ring(std::mt19937     &rng,
                      const glm::ivec2 &center,
                      float             radius_min,
                      float             radius_max,
                      const glm::ivec2 &shape)
{
  std::uniform_real_distribution<float> dist_angle(0.f, 2.f * M_PI);
  std::uniform_real_distribution<float> dist01(0.f, 1.f);

  float theta = dist_angle(rng);

  // Uniform area density in annulus
  float r2_min = radius_min * radius_min;
  float r2_max = radius_max * radius_max;

  float r = std::sqrt(r2_min + dist01(rng) * (r2_max - r2_min));

  glm::ivec2 p{static_cast<int>(std::round(center.x + r * std::cos(theta))),
               static_cast<int>(std::round(center.y + r * std::sin(theta)))};

  p.x = std::clamp(p.x, 0, shape.x - 1);
  p.y = std::clamp(p.y, 0, shape.y - 1);

  return p;
}

glm::ivec2 spawn_uniform(std::mt19937 &rng, const glm::ivec2 &shape)
{
  std::uniform_int_distribution<int> dist_x(0, shape.x - 1);
  std::uniform_int_distribution<int> dist_y(0, shape.y - 1);

  return {dist_x(rng), dist_y(rng)};
}

} // namespace hmap
