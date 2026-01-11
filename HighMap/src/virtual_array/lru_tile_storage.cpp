/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/virtual_array/tile_storage.hpp"

namespace hmap
{

LruTileStorage::LruTileStorage(size_t max_tiles) : max_tiles(max_tiles)
{
}

Array &LruTileStorage::get_tile(const TileRegion &region)
{
  const TileKey &key = region.key;

  // ---- hit

  auto it = tiles.find(key);
  if (it != tiles.end())
  {
    this->lru.splice(this->lru.begin(), lru, it->second.lru_it);
    return it->second.value;
  }

  // ---- miss → allocate

  glm::ivec2 total = region.total_shape();
  Array      tile(total);

  // ---- eviction if needed

  if (this->tiles.size() >= this->max_tiles)
  {
    const TileKey &evict_key = this->lru.back();
    auto           evict_it = this->tiles.find(evict_key);

    this->on_evict(evict_key, evict_it->second.value);

    this->tiles.erase(evict_it);
    this->lru.pop_back();
  }

  // ---- insert new tile

  this->lru.push_front(key);
  auto [inserted_it, _] = this->tiles.emplace(
      key,
      LruTileEntry{std::move(tile), this->lru.begin()});

  return inserted_it->second.value;
}

void LruTileStorage::release_tile(const TileRegion & /* region */)
{
  // no-op
}

void LruTileStorage::on_evict(const TileKey & /* key */, Array & /* tile */)
{
  // default: do nothing
}

} // namespace hmap
