/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <limits>

#include "macrologger.h"

#include "highmap/virtual_array/tile_storage.hpp"

namespace hmap
{

std::unique_ptr<TileStorage> RamTileStorage::clone() const
{
  return std::make_unique<RamTileStorage>(*this);
}

Array &RamTileStorage::get_tile(const TileRegion &region)
{
  auto it = tiles.find(region.key);
  if (it != tiles.end()) return it->second;

  glm::ivec2 total = region.shape; // include halo
  Array      tile(total);
  auto [inserted_it, _] = tiles.emplace(region.key, std::move(tile));
  return inserted_it->second;
}

size_t RamTileStorage::max_live_tiles() const
{
  return std::numeric_limits<size_t>::max();
}

void RamTileStorage::release_tile(const TileRegion & /* region */)
{
  // nothing for RAM
}

} // namespace hmap
