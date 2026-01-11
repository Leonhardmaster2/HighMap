/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file tile_storage.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <filesystem>
#include <fstream>
#include <list>
#include <unordered_map>

#include "highmap/array.hpp"
#include "highmap/virtual_array/tile_region.hpp"

namespace hmap
{

struct TileKeyHash
{
  size_t operator()(const TileKey &k) const
  {
    return (static_cast<size_t>(k.tx) << 32) ^ static_cast<size_t>(k.ty);
  }
};

// =====================================
// Abstract class
// =====================================

class TileStorage
{
public:
  virtual ~TileStorage() = default;
  virtual Array &get_tile(const TileRegion &region) = 0;
  virtual void   release_tile(const TileRegion &region) = 0;
};

// =====================================
// RAM storage
// =====================================

class RamTileStorage : public TileStorage
{
public:
  Array &get_tile(const TileRegion &region) override;
  void   release_tile(const TileRegion &region) override;

private:
  std::unordered_map<TileKey, Array, TileKeyHash> tiles;
};

// =====================================
// LRU storage
// =====================================

struct LruTileEntry
{
  Array                        value;
  std::list<TileKey>::iterator lru_it;
};

class LruTileStorage : public TileStorage
{
public:
  explicit LruTileStorage(size_t max_tiles);

  Array &get_tile(const TileRegion &region) override;
  void   release_tile(const TileRegion &region) override;

protected:
  size_t                                                 max_tiles;
  std::list<TileKey>                                     lru;
  std::unordered_map<TileKey, LruTileEntry, TileKeyHash> tiles;

  virtual void on_evict(const TileKey &key, Array &tile);
};

// --- Disk-LRU

class DiskLruTileStorage : public LruTileStorage
{
public:
  DiskLruTileStorage(size_t max_tiles, const std::filesystem::path &directory);

  Array &get_tile(const TileRegion &region) override;

protected:
  void on_evict(const TileKey &key, Array &tile) override;

private:
  std::filesystem::path root_dir;

  std::filesystem::path tile_path(const TileKey &key) const;
  Array                 load_tile_from_disk(const TileRegion &region);
};

} // namespace hmap