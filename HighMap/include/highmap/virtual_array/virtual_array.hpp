/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file virtual_array.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <future>
#include <thread>

#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/virtual_array/tile_region.hpp"
#include "highmap/virtual_array/tile_storage.hpp"

namespace hmap
{

enum ForEachMode : int
{
  VA_SEQUENTIAL,  // tile-by-tile, single thread
  VA_DISTRIBUTED, // tile-by-tile, multi-threaded
  VA_SINGLE_ARRAY // full array materialized at once
};

enum StorageMode : int
{
  VA_RAM,
  VA_DISK_LRU,
  VA_DISK_LRU_MIN,   // 2 live tiles, min mem footprint
  VA_DISK_SEQUENTIAL // sequential for/each, no smooth_overlap
};

struct VirtualArray
{
  VirtualArray(glm::ivec2                   shape,
               glm::vec4                    bbox, // (x1, x2, y1, y2)
               glm::ivec2                   tile_shape,
               int                          halo,
               std::unique_ptr<TileStorage> storage);

  VirtualArray(glm::ivec2  shape,
               glm::vec4   bbox,
               glm::ivec2  tile_shape,
               int         halo,
               StorageMode storage_mode = StorageMode::VA_RAM);

  // --- access individual cells (slower)
  float      get(int global_i, int global_j) const;
  void       set(int global_i, int global_j, float v);
  glm::ivec2 get_max_tiles() const;

  // clang-format off
  void  from_array(const Array &array, ForEachMode  mode = ForEachMode::VA_DISTRIBUTED);
  Array to_array(const glm::ivec2 array_shape, ForEachMode mode = ForEachMode::VA_DISTRIBUTED) const;
  Array to_array(ForEachMode mode = ForEachMode::VA_DISTRIBUTED) const;
  Array to_array_dbg() const;
  // clang-format on

  // --- processing methods

  // clang-format off
  float max(ForEachMode mode = ForEachMode::VA_DISTRIBUTED) const;
  float mean(ForEachMode mode = ForEachMode::VA_DISTRIBUTED) const;
  float min(ForEachMode mode = ForEachMode::VA_DISTRIBUTED) const;
  void  remap(float vmin = 0.f, float vmax = 1.f, ForEachMode mode = ForEachMode::VA_DISTRIBUTED);
  void  remap(float vmin, float vmax, float from_min, float from_max, ForEachMode mode = ForEachMode::VA_DISTRIBUTED);
  float sum(ForEachMode mode = ForEachMode::VA_DISTRIBUTED) const;
  // clang-format on

  std::vector<float> unique_values(
      ForEachMode mode = ForEachMode::VA_DISTRIBUTED) const;

  void smooth_overlap_buffers();

  // Find which tile covers a given index
  TileRegion tile_region_from_global_index(int global_i, int global_j) const;
  TileRegion tile_region_from_tile_coords(int tile_x, int tile_y) const;
  glm::ivec2 local_indices(const TileRegion &region,
                           int               global_i,
                           int               global_j) const;

  // --- Members
  glm::ivec2                   shape;
  glm::vec4                    bbox; // (x1, x2, y1, y2)
  glm::ivec2                   tile_shape;
  int                          halo;
  std::unique_ptr<TileStorage> storage;
};

// actual implementation
#include "highmap/virtual_array/virtual_array.inl"

} // namespace hmap
