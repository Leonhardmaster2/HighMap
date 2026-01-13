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
#include <mutex>
#include <optional>
#include <unordered_map>

#include "highmap/array.hpp"
#include "highmap/virtual_array/tile_region.hpp"

namespace hmap
{

enum StorageMode : int
{
  VA_RAM,
  VA_DISK_LRU,
  VA_DISK_LRU_MIN,   // 2 live tiles, min mem footprint
  VA_DISK_SEQUENTIAL // sequential for/each, no smooth_overlap
};

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
  virtual size_t max_live_tiles() const = 0;
  virtual std::unique_ptr<TileStorage> clone() const = 0;

  // Opportunistically free memory while keeping data persistent.
  virtual void trim()
  {
  }
};

// =====================================
// RAM storage
// =====================================

class RamTileStorage : public TileStorage
{
public:
  std::unique_ptr<TileStorage> clone() const override;

  Array &get_tile(const TileRegion &region) override;
  void   release_tile(const TileRegion &region) override;
  size_t max_live_tiles() const override;

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

  std::unique_ptr<TileStorage> clone() const override;

  Array &get_tile(const TileRegion &region) override;
  void   release_tile(const TileRegion &region) override;
  size_t max_live_tiles() const override;

protected:
  size_t                                                 max_tiles;
  std::list<TileKey>                                     lru;
  std::unordered_map<TileKey, LruTileEntry, TileKeyHash> tiles;
  std::mutex                                             mutex;

  Array       &get_tile_no_mutex_lock(const TileRegion &region);
  virtual void on_evict(const TileKey &key, Array &tile);
};

// =====================================
// DISK-LRU storage
// =====================================

class DiskLruTileStorage : public LruTileStorage
{
public:
  DiskLruTileStorage(size_t max_tiles);
  ~DiskLruTileStorage();

  std::unique_ptr<TileStorage> clone() const override;

  Array &get_tile(const TileRegion &region) override;
  size_t max_live_tiles() const override;
  void   trim() override;

protected:
  void on_evict(const TileKey &key, Array &tile) override;

private:
  std::filesystem::path root_dir;

  std::filesystem::path tile_path(const TileKey &key) const;
  Array                 load_tile_from_disk(const TileRegion &region);
};

// =====================================
// DISK-sequential storage
// =====================================

class DiskSequentialTileStorage : public TileStorage
{
public:
  DiskSequentialTileStorage();
  ~DiskSequentialTileStorage() override;

  std::unique_ptr<TileStorage> clone() const override;

  Array &get_tile(const TileRegion &region) override;
  void   release_tile(const TileRegion &region) override;
  size_t max_live_tiles() const override;
  void   trim() override;

private:
  std::filesystem::path root_dir;
  TileKey               current_key{};
  std::optional<Array>  current_tile;

  Array                 load_or_create(const TileRegion &region);
  void                  save_tile(const TileKey &key, const Array &tile);
  std::filesystem::path tile_path(const TileKey &key) const;
};

// =====================================
// functions
// =====================================

std::unique_ptr<TileStorage> make_storage(glm::ivec2  shape,
                                          glm::ivec2  tile_shape,
                                          StorageMode storage_mode);

} // namespace hmap