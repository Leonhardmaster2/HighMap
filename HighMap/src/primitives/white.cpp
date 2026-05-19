/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstdint>
#include <random> // for uniform_real_distribution
#include <vector> // for vector

#include "highmap/array.hpp" // for Array

namespace hmap
{

Array white(glm::ivec2 shape, float a, float b, std::uint32_t seed)
{
  Array                                 array = Array(shape);
  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dis(a, b);
  for (auto &v : array.vector)
  {
    v = dis(gen);
  }
  return array;
}

Array white_density_map(const Array &density_map, std::uint32_t seed)
{
  Array                                 array = Array(density_map.shape);
  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dis(0.f, 1.f);

  for (int j = 0; j < density_map.shape.y; j++)
    for (int i = 0; i < density_map.shape.x; i++)
    {
      float r = dis(gen);
      if (r < density_map(i, j)) array(i, j) = r / density_map(i, j);
    }
  return array;
}

Array white_sparse(glm::ivec2    shape,
                   float         a,
                   float         b,
                   float         density,
                   std::uint32_t seed)
{
  Array                                 array = Array(shape);
  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dis1(0.f, 1.f);
  std::uniform_real_distribution<float> dis2(a, b);

  for (auto &v : array.vector)
  {
    if (dis1(gen) < density) v = dis2(gen);
  }
  return array;
}

Array white_sparse_binary(glm::ivec2 shape, float density, std::uint32_t seed)
{
  Array                                 array = Array(shape);
  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dis(0.f, 1.f);

  for (auto &v : array.vector)
  {
    float r = dis(gen);
    if (r < density) v = 1.f;
  }
  return array;
}

} // namespace hmap
