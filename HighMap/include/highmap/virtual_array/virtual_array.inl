/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once

template <typename Func>
void for_each_tile_sequential(VirtualArray &va, Func &&func)
{
  int nx = ceil_div(va.shape.x, va.tile_shape.x);
  int ny = ceil_div(va.shape.y, va.tile_shape.y);

  for (int ty = 0; ty < ny; ++ty)
    for (int tx = 0; tx < nx; ++tx)
    {
      TileRegion region = va.tile_region_from_tile_coords(tx, ty);
      Array     &tile = va.storage->get_tile(region);
      func(tile, region.total_shape(), region.bbox);
      va.storage->release_tile(region);
    }
}

template <typename Func>
void for_each_tile_distributed(VirtualArray &va, Func &&func)
{
  int nx = ceil_div(va.shape.x, va.tile_shape.x);
  int ny = ceil_div(va.shape.y, va.tile_shape.y);

  size_t                         nthreads = size_t(nx * ny);
  std::vector<std::future<void>> futures;
  futures.reserve(nthreads);

  auto task = [&](int tx, int ty)
  {
    TileRegion region = va.tile_region_from_tile_coords(tx, ty);

    Array &tile = va.storage->get_tile(region);
    func(tile, region.total_shape(), region.bbox);
    va.storage->release_tile(region);
  };

  for (int ty = 0; ty < ny; ++ty)
    for (int tx = 0; tx < nx; ++tx)
    {
      futures.emplace_back(std::async(std::launch::async, task, tx, ty));
    }

  for (auto &f : futures)
    f.get();
}

template <typename Func>
void for_each_tile_distributed(const std::vector<VirtualArray *> &p_vas,
                               Func                             &&func,
                               int                                nthreads = 0)
{
  if (p_vas.empty())
  {
    LOG_ERROR("Empty VirtualArray list");
    return;
  }

  if (!p_vas.front()) return;

  const VirtualArray &va = *p_vas.front();

  // --- validate compatibility
  for (auto p_va : p_vas)
  {
    if (!p_va) continue;
    if (p_va->shape != va.shape || p_va->tile_shape != va.tile_shape)
    {
      LOG_ERROR("Incompatible VirtualArray configuration");
      return;
    }
  }

  const int nx = ceil_div(va.shape.x, va.tile_shape.x);
  const int ny = ceil_div(va.shape.y, va.tile_shape.y);
  const int ntasks = nx * ny;

  if (nthreads <= 0) nthreads = std::thread::hardware_concurrency();

  nthreads = std::min(nthreads, ntasks);

  std::vector<std::future<void>> futures;
  futures.reserve(nthreads);

  auto worker = [&](int thread_id)
  {
    std::vector<Array *> p_arrays;
    p_arrays.reserve(p_vas.size());

    for (int k = thread_id; k < ntasks; k += nthreads)
    {
      const int ty = k / nx;
      const int tx = k % nx;

      LOG_DEBUG("%d %d", tx, ty);
      
      TileRegion region = va.tile_region_from_tile_coords(tx, ty);

      p_arrays.clear();

      for (auto p_va : p_vas)
      {
        if (!p_va)
        {
          p_arrays.push_back(nullptr);
          continue;
        }

        Array &tile = p_va->storage->get_tile(region);
        p_arrays.push_back(&tile);
      }

      func(p_arrays, region.total_shape(), region.bbox);

      for (auto p_va : p_vas)
        if (p_va) p_va->storage->release_tile(region);
    }
  };

  for (int t = 0; t < nthreads; ++t)
  {
    futures.emplace_back(std::async(std::launch::async, worker, t));
  }

  for (auto &f : futures)
    f.get();
}
