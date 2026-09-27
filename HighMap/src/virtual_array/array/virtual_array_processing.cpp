/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cstddef>
#include <vector>

#include "highmap/array.hpp"
#include "highmap/range.hpp"
#include "highmap/vectors.hpp"
#include "highmap/virtual_array/tile_region.hpp"
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

  std::vector<float> max_per_tile(nt.x * nt.y);

  auto lambda = [&max_per_tile, nt](const Array &tile, const TileRegion &region)
  {
    int tile_idx = region.key.ty * nt.x + region.key.tx;
    max_per_tile[tile_idx] = tile.max();
  };

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

  std::vector<float> min_per_tile(nt.x * nt.y);

  auto lambda = [&min_per_tile, nt](const Array &tile, const TileRegion &region)
  {
    int tile_idx = region.key.ty * nt.x + region.key.tx;
    min_per_tile[tile_idx] = tile.min();
  };

  for_each_tile(*this, lambda, cm);

  return *std::min_element(min_per_tile.begin(), min_per_tile.end());
}

glm::vec2 VirtualArray::range(const ComputeMode &cm) const
{
  glm::ivec2 nt = this->get_max_tiles();

  if (nt.x * nt.y == 0) return glm::vec2(0.f, 0.f);

  std::vector<float> max_per_tile(nt.x * nt.y);
  std::vector<float> min_per_tile(nt.x * nt.y);

  auto lambda = [&max_per_tile, &min_per_tile, nt](const Array      &tile,
                                                   const TileRegion &region)
  {
    int tile_idx = region.key.ty * nt.x + region.key.tx;
    max_per_tile[tile_idx] = tile.max();
    min_per_tile[tile_idx] = tile.min();
  };

  for_each_tile(*this, lambda, cm);

  float gmin = *std::min_element(min_per_tile.begin(), min_per_tile.end());
  float gmax = *std::max_element(max_per_tile.begin(), max_per_tile.end());

  return glm::vec2(gmin, gmax);
}

glm::vec2 VirtualArray::range_percentile(float              p_low,
                                         float              p_high,
                                         const ComputeMode &cm,
                                         size_t             bins) const
{
  glm::ivec2 nt = this->get_max_tiles();

  if (nt.x * nt.y == 0) return glm::vec2(0.f, 0.f);

  // global range
  glm::vec2 range = this->range(cm);

  if (range.x == range.y) return range;

  // global and per-tile histograms
  std::vector<size_t>              global_hist(bins, 0);
  std::vector<std::vector<size_t>> hist_per_tile(nt.x * nt.y,
                                                 std::vector<size_t>(bins, 0));

  auto lambda = [&](const Array &tile, const TileRegion &region)
  {
    int   tile_idx = region.key.ty * nt.x + region.key.tx;
    auto &hist = hist_per_tile[tile_idx];

    // sum over the inner cells only to avoid double count
    const glm::ivec4 &h = region.halo;

    for (int j = h.z; j < region.shape.y - h.w; ++j)
      for (int i = h.x; i < region.shape.x - h.y; ++i)
      {
        float v = tile(i, j);

        size_t idx = std::min<size_t>(
            bins - 1,
            size_t((v - range.x) / (range.y - range.x) * bins));
        hist[idx]++;
      }
  };

  for_each_tile(*this, lambda, cm);

  // reduce
  for (const auto &hist : hist_per_tile)
  {
    for (size_t i = 0; i < bins; ++i)
      global_hist[i] += hist[i];
  }

  // percentile extraction
  size_t total = 0;
  for (size_t v : global_hist)
    total += v;

  size_t low_target = size_t(p_low * total);
  size_t high_target = size_t(p_high * total);

  size_t acc = 0;
  float  low = range.x;
  float  high = range.y;

  for (size_t i = 0; i < bins; ++i)
  {
    acc += global_hist[i];

    if (acc >= low_target && low == range.x)
      low = range.x + (range.y - range.x) * (float(i) / bins);

    if (acc >= high_target)
    {
      high = range.x + (range.y - range.x) * (float(i) / bins);
      break;
    }
  }

  return {low, high};
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

  std::vector<float> sum_per_tile(nt.x * nt.y);

  auto lambda = [&sum_per_tile, nt](const Array &tile, const TileRegion &region)
  {
    // sum over the inner cells only to avoid double count
    const glm::ivec4 &h = region.halo;
    float             sum = 0.f;
    int               tile_idx = region.key.ty * nt.x + region.key.tx;

    for (int j = h.z; j < region.shape.y - h.w; ++j)
      for (int i = h.x; i < region.shape.x - h.y; ++i)
        sum += tile(i, j);

    sum_per_tile[tile_idx] = sum;
  };

  for_each_tile(*this, lambda, cm);

  float sum = 0.f;
  for (auto &v : sum_per_tile)
    sum += v;

  return sum;
}

void VirtualArray::smooth_overlap_buffers()
{
  if (this->storage->max_live_tiles() < 2)
  {
    hmap::log::error(
        "smooth_overlap_buffers requires at least 2 tiles in memory, skipping");
    return;
  }

  int nx = ceil_div(this->shape.x, this->tile_shape.x);
  int ny = ceil_div(this->shape.y, this->tile_shape.y);

  // --- x-direction

  for (int ty = 0; ty < ny; ++ty)
    for (int tx = 0; tx < nx - 1; ++tx)
    {
      // load
      TileRegion region0 = this->tile_region_from_tile_coords(tx, ty);
      TileRegion region1 = this->tile_region_from_tile_coords(tx + 1, ty);
      Array     &tile0 = this->storage->get_tile(region0);
      Array     &tile1 = this->storage->get_tile(region1);

      // average overlap
      for (int p = 0; p < this->halo; p++)
        for (int q = 0; q < tile0.shape.y; q++)
        {
          float r = float(p) / float(this->halo - 1);
          r = smoothstep5(r);

          int pbuf = tile0.shape.x - 2 * this->halo + p;
          tile1(p, q) = lerp(tile0(pbuf, q), tile1(p, q), r);
          tile0(pbuf, q) = tile1(p, q);
        }

      // release
      this->storage->release_tile(region0);
      this->storage->release_tile(region1);
    }

  // --- y-direction

  for (int ty = 0; ty < ny - 1; ++ty)
    for (int tx = 0; tx < nx; ++tx)
    {
      // load
      TileRegion region0 = this->tile_region_from_tile_coords(tx, ty);
      TileRegion region1 = this->tile_region_from_tile_coords(tx, ty + 1);
      Array     &tile0 = this->storage->get_tile(region0);
      Array     &tile1 = this->storage->get_tile(region1);

      // average overlap
      for (int p = 0; p < tile0.shape.x; p++)
        for (int q = 0; q < this->halo; q++)
        {
          float r = float(q) / float(this->halo - 1);
          r = smoothstep5(r);

          int qbuf = tile0.shape.y - 2 * this->halo + q;
          tile1(p, q) = lerp(tile0(p, qbuf), tile1(p, q), r);
          tile0(p, qbuf) = tile1(p, q);
        }

      // release
      this->storage->release_tile(region0);
      this->storage->release_tile(region1);
    }
}

std::vector<float> VirtualArray::unique_values(const ComputeMode &cm) const
{
  glm::ivec2 nt = this->get_max_tiles();

  if (nt.x * nt.y == 0) return {0.f};

  std::vector<std::vector<float>> unique_per_tile(nt.x * nt.y);

  auto lambda =
      [&unique_per_tile, nt](const Array &tile, const TileRegion &region)
  {
    int tile_idx = region.key.ty * nt.x + region.key.tx;
    unique_per_tile[tile_idx] = tile.unique_values();
  };

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
