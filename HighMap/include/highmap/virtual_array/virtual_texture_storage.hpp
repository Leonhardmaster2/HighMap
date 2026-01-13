/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file virtual_array.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @copyright Copyright (c) 2025
 */
#pragma once
#include "macrologger.h"

#include "highmap/virtual_array/tile_storage.hpp"

namespace hmap
{

/**
 * @brief Storage manager for VirtualTexture
 *
 * A VirtualTexture is composed of multiple VirtualArrays (channels). This
 * storage wrapper ensures that each channel has a proper TileStorage (RAM,
 * DiskLRU, Sequential) and can manage shared directories, cleanup, etc.
 */
struct VirtualTextureStorage
{
  std::vector<std::unique_ptr<TileStorage>> channel_storages;

  // --- Constructor: creates N channels

  VirtualTextureStorage(int                          nchannels,
                        std::unique_ptr<TileStorage> storage_proto);

  // --- Access per channel

  TileStorage       &channel(int idx);
  const TileStorage &channel(int idx) const;
  int                channel_count() const;

  // --- Free memory: release all tiles

  void trim_storage();
};

} // namespace hmap