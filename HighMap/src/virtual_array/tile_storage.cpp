/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/math.hpp"
#include "highmap/virtual_array/tile_storage.hpp"

namespace hmap
{

std::unique_ptr<TileStorage> make_storage(glm::ivec2  shape,
                                          glm::ivec2  tile_shape,
                                          StorageMode storage_mode)
{
  switch (storage_mode)
  {
  case StorageMode::VA_RAM: return std::make_unique<RamTileStorage>();

  case StorageMode::VA_DISK_LRU:
  {
    int nx = ceil_div(shape.x, tile_shape.x);
    int ny = ceil_div(shape.y, tile_shape.y);
    return std::make_unique<DiskLruTileStorage>(nx * ny);
  }

  case StorageMode::VA_DISK_LRU_MIN:
    return std::make_unique<DiskLruTileStorage>(2);

  case StorageMode::VA_DISK_SEQUENTIAL:
    return std::make_unique<DiskSequentialTileStorage>();
  }

  // fallback
  LOG_ERROR("Unknown StorageMode, defaulting to RAM");
  return std::make_unique<RamTileStorage>();
}

} // namespace hmap
