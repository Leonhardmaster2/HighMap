/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once

template <typename Func>
void for_each_tile_distributed(const std::vector<VirtualArray *> &p_vas,
                               Func                             &&func,
                               int                                nthreads = 0)
{
  // --- Failsafe

  if (p_vas.empty())
  {
    LOG_ERROR("Empty VirtualArray list");
    return;
  }

  if (!p_vas.front()) return;

  const VirtualArray &va = *p_vas.front();

  // --- Validate compatibility

  for (auto p_va : p_vas)
  {
    if (!p_va) continue;
    if (p_va->shape != va.shape || p_va->tile_shape != va.tile_shape)
    {
      LOG_ERROR("Incompatible VirtualArray configuration");
      return;
    }
  }

  // --- Compute

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

template <typename Func>
void for_each_tile_distributed(VirtualArray &va, Func &&func, int nthreads = 0)
{
  for_each_tile_distributed(
      std::vector<VirtualArray *>{&va},
      [&](std::vector<Array *> &p_arrays,
          const glm::ivec2     &shape,
          const glm::vec4      &bbox) { func(*p_arrays[0], shape, bbox); },
      nthreads);
}

template <typename Func>
void for_each_tile_sequential(const std::vector<VirtualArray *> &p_vas,
                              Func                             &&func)
{
  // --- Failsafe

  if (p_vas.empty())
  {
    LOG_ERROR("Empty VirtualArray list");
    return;
  }

  if (!p_vas.front()) return;

  const VirtualArray &va = *p_vas.front();

  // --- Validate compatibility

  for (auto p_va : p_vas)
  {
    if (!p_va) continue;
    if (p_va->shape != va.shape || p_va->tile_shape != va.tile_shape)
    {
      LOG_ERROR("Incompatible VirtualArray configuration");
      return;
    }
  }

  // --- Compute

  const int nx = ceil_div(va.shape.x, va.tile_shape.x);
  const int ny = ceil_div(va.shape.y, va.tile_shape.y);

  std::vector<Array *> p_arrays;
  p_arrays.reserve(p_vas.size());

  for (int ty = 0; ty < ny; ++ty)
    for (int tx = 0; tx < nx; ++tx)
    {
      TileRegion region = va.tile_region_from_tile_coords(tx, ty);

      p_arrays.clear();

      // acquire tiles
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

      // user computation
      func(p_arrays, region.total_shape(), region.bbox);

      // release tiles
      for (auto p_va : p_vas)
        if (p_va) p_va->storage->release_tile(region);
    }
}

template <typename Func>
void for_each_tile_sequential(VirtualArray &va, Func &&func)
{
  for_each_tile_sequential(std::vector<VirtualArray *>{&va},
                           [&](std::vector<Array *> &p_arrays,
                               const glm::ivec2     &shape,
                               const glm::vec4      &bbox)
                           { func(*p_arrays[0], shape, bbox); });
}

template <typename Func>
void for_each_tile_single_array(const std::vector<VirtualArray *> &p_vas,
                                Func                             &&func)
{
  // --- Failsafe

  if (p_vas.empty())
  {
    LOG_ERROR("Empty VirtualArray list");
    return;
  }

  if (!p_vas.front()) return;

  const VirtualArray &va = *p_vas.front();

  // --- Validate compatibility

  for (auto p_va : p_vas)
  {
    if (!p_va) continue;
    if (p_va->shape != va.shape || p_va->tile_shape != va.tile_shape)
    {
      LOG_ERROR("Incompatible VirtualArray configuration");
      return;
    }
  }

  // --- Compute

  // create temporary arrays
  std::vector<Array> arrays;
  arrays.reserve(p_vas.size());

  for (auto p_va : p_vas)
    arrays.push_back(p_va ? p_va->to_array() : Array());

  // create pointer vector for user func
  std::vector<Array *> p_arrays;
  p_arrays.reserve(p_vas.size());

  for (size_t i = 0; i < p_vas.size(); ++i)
    p_arrays.push_back(p_vas[i] ? &arrays[i] : nullptr);

  // call user operation
  func(p_arrays, va.shape, va.bbox);

  // copy back to VirtualArrays
  for (size_t i = 0; i < p_vas.size(); ++i)
    if (p_vas[i]) p_vas[i]->from_array(arrays[i]);
}

template <typename Func>
void for_each_tile_single_array(VirtualArray &va, Func &&func)
{
  for_each_tile_single_array(std::vector<VirtualArray *>{&va},
                             [&](std::vector<Array *> &p_arrays,
                                 const glm::ivec2     &shape,
                                 const glm::vec4      &bbox)
                             { func(*p_arrays[0], shape, bbox); });
}

template <typename Func>
void for_each_tile(const std::vector<VirtualArray *> &p_vas,
                   Func                             &&func,
                   ForEachMode mode = ForEachMode::VA_DISTRIBUTED)
{
  switch (mode)
  {
  case ForEachMode::VA_SEQUENTIAL: for_each_tile_sequential(p_vas, func); break;
  case ForEachMode::VA_DISTRIBUTED:
    for_each_tile_distributed(p_vas, func);
    break;
  case ForEachMode::VA_SINGLE_ARRAY:
    for_each_tile_single_array(p_vas, func);
    break;
  }
}

template <typename Func>
void for_each_tile(VirtualArray &va,
                   Func        &&func,
                   ForEachMode   mode = ForEachMode::VA_DISTRIBUTED)
{
  switch (mode)
  {
  case ForEachMode::VA_SEQUENTIAL:
    for_each_tile_sequential(std::vector<VirtualArray *>{&va},
                             [&](std::vector<Array *> &p_arrays,
                                 const glm::ivec2     &shape,
                                 const glm::vec4      &bbox)
                             { func(*p_arrays[0], shape, bbox); });
    break;
    //
  case ForEachMode::VA_DISTRIBUTED:
    for_each_tile_distributed(std::vector<VirtualArray *>{&va},
                              [&](std::vector<Array *> &p_arrays,
                                  const glm::ivec2     &shape,
                                  const glm::vec4      &bbox)
                              { func(*p_arrays[0], shape, bbox); });
    break;
    //
  case ForEachMode::VA_SINGLE_ARRAY:
    for_each_tile_single_array(std::vector<VirtualArray *>{&va},
                               [&](std::vector<Array *> &p_arrays,
                                   const glm::ivec2     &shape,
                                   const glm::vec4      &bbox)
                               { func(*p_arrays[0], shape, bbox); });

    break;
  }
}
