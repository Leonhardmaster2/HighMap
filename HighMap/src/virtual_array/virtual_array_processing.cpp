/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <filesystem>
#include <random>

#include "macrologger.h"

#include "highmap/internal/vector_utils.hpp"
#include "highmap/math.hpp"
#include "highmap/virtual_array/virtual_array.hpp"

namespace hmap
{

float VirtualArray::max(ForEachMode mode) const
{
  glm::ivec2 nt = this->get_max_tiles();

  if (nt.x * nt.y == 0) return 0.f;

  std::vector<float> max_per_tile;
  max_per_tile.reserve(nt.x * nt.y);

  auto lambda = [&max_per_tile](const Array &tile, const TileRegion &)
  { max_per_tile.push_back(tile.max()); };

  for_each_tile(*this, lambda, mode);

  return *std::max_element(max_per_tile.begin(), max_per_tile.end());
}

float VirtualArray::mean(ForEachMode mode) const
{
  float mean = this->sum(mode) / (float)(this->shape.x * this->shape.y);
  return mean;
}

float VirtualArray::min(ForEachMode mode) const
{
  glm::ivec2 nt = this->get_max_tiles();

  if (nt.x * nt.y == 0) return 0.f;

  std::vector<float> min_per_tile;
  min_per_tile.reserve(nt.x * nt.y);

  auto lambda = [&min_per_tile](const Array &tile, const TileRegion &)
  { min_per_tile.push_back(tile.min()); };

  for_each_tile(*this, lambda, mode);

  return *std::min_element(min_per_tile.begin(), min_per_tile.end());
}

float VirtualArray::sum(ForEachMode mode) const
{
  glm::ivec2 nt = this->get_max_tiles();

  if (nt.x * nt.y == 0) return 0.f;

  std::vector<float> sum_per_tile;
  sum_per_tile.reserve(nt.x * nt.y);

  auto lambda = [&sum_per_tile](const Array &tile, const TileRegion &)
  { sum_per_tile.push_back(tile.sum()); };

  for_each_tile(*this, lambda, mode);

  float sum = 0.f;
  for (auto &v : sum_per_tile)
    sum += v;

  return sum;
}

std::vector<float> VirtualArray::unique_values(ForEachMode mode) const
{
  glm::ivec2 nt = this->get_max_tiles();

  if (nt.x * nt.y == 0) return {0.f};

  std::vector<std::vector<float>> unique_per_tile;
  unique_per_tile.reserve(nt.x * nt.y);

  auto lambda = [&unique_per_tile](const Array &tile, const TileRegion &)
  { unique_per_tile.push_back(tile.unique_values()); };

  for_each_tile(*this, lambda, mode);

  // flatten
  std::vector<float> unique_values = {};
  for (const auto &vec : unique_per_tile)
    for (const auto &v : vec)
      unique_values.push_back(v);

  vector_unique_values(unique_values);
  return unique_values;
}

} // namespace hmap
