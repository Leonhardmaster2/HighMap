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
#include "highmap/math.hpp"
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

struct ComputeMode
{
  ForEachMode mode;
  bool        trim_storage = false;
};

struct VirtualArray
{
  VirtualArray() = default;

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

  VirtualArray(glm::ivec2  shape,
               glm::ivec2  tile_shape,
               int         halo,
               StorageMode storage_mode = StorageMode::VA_RAM);

  // --- access individual cells (slower)
  float      get(int global_i, int global_j) const;
  float      get_bilinear(float x, float y) const;
  float      get_nearest(float x, float y) const;
  void       set(int global_i, int global_j, float v);
  glm::ivec2 get_max_tiles() const;

  void  from_array(const Array &array, const ComputeMode &cm);
  Array to_array(const glm::ivec2 array_shape, const ComputeMode &cm) const;
  Array to_array(const ComputeMode &cm) const;
  Array to_array_dbg() const;

  // --- processing methods

  float max(const ComputeMode &cm) const;
  float mean(const ComputeMode &cm) const;
  float min(const ComputeMode &cm) const;
  float sum(const ComputeMode &cm) const;

  void inverse(const ComputeMode &cm);
  void remap(float vmin, float vmax, const ComputeMode &cm);
  void remap(float              vmin,
             float              vmax,
             float              from_min,
             float              from_max,
             const ComputeMode &cm);

  std::vector<float> unique_values(const ComputeMode &cm) const;

  void smooth_overlap_buffers();

  // Find which tile covers a given index
  TileRegion tile_region_from_global_index(int global_i, int global_j) const;
  TileRegion tile_region_from_tile_coords(int tile_x, int tile_y) const;
  glm::ivec2 local_indices(const TileRegion &region,
                           int               global_i,
                           int               global_j) const;

  void trim_storage();

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
