/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/virtual_array/virtual_texture_storage.hpp" // for Virtual...

#include <memory>  // for unique_ptr
#include <utility> // for move
#include <vector>  // for vector

#include "highmap/internal/string_utils.hpp"
#include "highmap/virtual_array/tile_storage.hpp" // for TileSto...

namespace hmap
{

VirtualTextureStorage::VirtualTextureStorage(
    int                          nchannels,
    std::unique_ptr<TileStorage> storage_proto)
{
  this->channel_storages.reserve(nchannels);

  for (int i = 0; i < nchannels; ++i)
  {
    auto storage = storage_proto->clone();
    this->channel_storages.push_back(std::move(storage));
  }
}

TileStorage &VirtualTextureStorage::channel(int idx)
{
  return *this->channel_storages[idx];
}

const TileStorage &VirtualTextureStorage::channel(int idx) const
{
  return *this->channel_storages[idx];
}

int VirtualTextureStorage::channel_count() const
{
  return int(this->channel_storages.size());
}

void VirtualTextureStorage::trim_storage()
{
  for (auto &sp_tile_storage : this->channel_storages)
    sp_tile_storage->trim();
}

} // namespace hmap
