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

struct VirtualArray
{
  VirtualArray(glm::ivec2                   shape,
               glm::vec4                    bbox, // (x1, x2, y1, y2)
               glm::ivec2                   tile_shape,
               int                          halo,
               std::unique_ptr<TileStorage> storage);

  // Access individual cells (slower)
  float get(int global_i, int global_j) const;
  void  set(int global_i, int global_j, float v);

  void  smooth_overlap_buffers();
  Array to_array() const;

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

template <typename Func>
void for_each_tile_sequential(VirtualArray &va, Func &&func);
template <typename Func>
void for_each_tile_distributed(VirtualArray &va, Func &&func);

// actual implementation
#include "highmap/virtual_array/virtual_array.inl"

} // namespace hmap