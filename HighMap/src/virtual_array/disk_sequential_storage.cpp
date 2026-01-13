/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/internal/string_utils.hpp"
#include "highmap/virtual_array/tile_storage.hpp"

namespace hmap
{

DiskSequentialTileStorage::DiskSequentialTileStorage()
{
  root_dir = make_unique_temp_dir("va_seq");
}

DiskSequentialTileStorage::~DiskSequentialTileStorage()
{
  std::error_code ec;
  std::filesystem::remove_all(root_dir, ec);
}

Array &DiskSequentialTileStorage::get_tile(const TileRegion &region)
{
  if (current_tile)
  {
    LOG_ERROR("DiskSequentialTileStorage: tile already loaded");
    throw std::logic_error("DiskSequentialTileStorage misuse");
  }

  current_key = region.key;
  current_tile = load_or_create(region);
  return *current_tile;
}

Array DiskSequentialTileStorage::load_or_create(const TileRegion &region)
{
  auto path = tile_path(region.key);

  Array tile(region.shape);

  if (std::filesystem::exists(path))
  {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("DiskSequential: cannot read tile");

    in.read(reinterpret_cast<char *>(tile.vector.data()),
            tile.size() * sizeof(float));
  }

  return tile;
}

size_t DiskSequentialTileStorage::max_live_tiles() const
{
  return 1;
}

void DiskSequentialTileStorage::release_tile(const TileRegion &region)
{
  if (!current_tile)
  {
    LOG_ERROR("DiskSequentialTileStorage: no tile loaded on release");
    return;
  }

  if (region.key != current_key)
    LOG_ERROR("DiskSequentialTileStorage: releasing wrong tile");

  save_tile(region.key, *current_tile);
  current_tile.reset();
}

void DiskSequentialTileStorage::save_tile(const TileKey &key, const Array &tile)
{
  auto path = tile_path(key);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("DiskSequential: cannot write tile");

  out.write(reinterpret_cast<const char *>(tile.vector.data()),
            tile.size() * sizeof(float));
}

std::filesystem::path DiskSequentialTileStorage::tile_path(
    const TileKey &key) const
{
  return root_dir / ("tile_" + std::to_string(key.tx) + "_" +
                     std::to_string(key.ty) + ".bin");
}

} // namespace hmap
