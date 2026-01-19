/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <filesystem>
#include <random>

#include "macrologger.h"

#include "highmap/internal/vector_utils.hpp"
#include "highmap/math.hpp"
#include "highmap/range.hpp"
#include "highmap/virtual_array/virtual_array.hpp"

namespace hmap
{

void VirtualArray::inverse(const ComputeMode &cm)
{
  float hmax = this->max(cm);

  for_each_tile(
      {this},
      [hmax](std::vector<hmap::Array *> p_arrays, const TileRegion &)
      {
        hmap::Array *pa_out = p_arrays[0];
        *pa_out *= -1.f;
        *pa_out += hmax;
      },
      cm);
}

float VirtualArray::max(const ComputeMode &cm) const
{
  glm::ivec2 nt = this->get_max_tiles();

  if (nt.x * nt.y == 0) return 0.f;

  std::vector<float> max_per_tile;
  max_per_tile.reserve(nt.x * nt.y);

  auto lambda = [&max_per_tile](const Array &tile, const TileRegion &)
  { max_per_tile.push_back(tile.max()); };

  for_each_tile(*this, lambda, cm);

  return *std::max_element(max_per_tile.begin(), max_per_tile.end());
}

float VirtualArray::mean(const ComputeMode &cm) const
{
  float mean = this->sum(cm) / (float)(this->shape.x * this->shape.y);
  return mean;
}

float VirtualArray::min(const ComputeMode &cm) const
{
  glm::ivec2 nt = this->get_max_tiles();

  if (nt.x * nt.y == 0) return 0.f;

  std::vector<float> min_per_tile;
  min_per_tile.reserve(nt.x * nt.y);

  auto lambda = [&min_per_tile](const Array &tile, const TileRegion &)
  { min_per_tile.push_back(tile.min()); };

  for_each_tile(*this, lambda, cm);

  return *std::min_element(min_per_tile.begin(), min_per_tile.end());
}

void VirtualArray::remap(float vmin, float vmax, const ComputeMode &cm)
{
  float global_min = this->min(cm);
  float global_max = this->max(cm);

  this->remap(vmin, vmax, global_min, global_max, cm);
}

void VirtualArray::remap(float              vmin,
                         float              vmax,
                         float              from_min,
                         float              from_max,
                         const ComputeMode &cm)
{
  auto lambda =
      [vmin, vmax, from_min, from_max](Array &tile, const TileRegion &)
  { hmap::remap(tile, vmin, vmax, from_min, from_max); };

  for_each_tile(*this, lambda, cm);
}

float VirtualArray::sum(const ComputeMode &cm) const
{
  glm::ivec2 nt = this->get_max_tiles();

  if (nt.x * nt.y == 0) return 0.f;

  std::vector<float> sum_per_tile;
  sum_per_tile.reserve(nt.x * nt.y);

  auto lambda = [&sum_per_tile](const Array &tile, const TileRegion &region)
  {
    // sum over the inner cells only to avoid double count
    const glm::ivec4 &h = region.halo;
    float             sum = 0.f;

    for (int j = h.z; j < region.shape.y - h.w; ++j)
      for (int i = h.x; i < region.shape.x - h.y; ++i)
        sum += tile(i, j);

    sum_per_tile.push_back(sum);
  };

  for_each_tile(*this, lambda, cm);

  float sum = 0.f;
  for (auto &v : sum_per_tile)
    sum += v;

  return sum;
}

std::vector<float> VirtualArray::unique_values(const ComputeMode &cm) const
{
  glm::ivec2 nt = this->get_max_tiles();

  if (nt.x * nt.y == 0) return {0.f};

  std::vector<std::vector<float>> unique_per_tile;
  unique_per_tile.reserve(nt.x * nt.y);

  auto lambda = [&unique_per_tile](const Array &tile, const TileRegion &)
  { unique_per_tile.push_back(tile.unique_values()); };

  for_each_tile(*this, lambda, cm);

  // flatten
  std::vector<float> unique_values = {};
  for (const auto &vec : unique_per_tile)
    for (const auto &v : vec)
      unique_values.push_back(v);

  vector_unique_values(unique_values);
  return unique_values;
}

} // namespace hmap
