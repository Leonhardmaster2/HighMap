#include <iostream>

#include "highmap.hpp"

int main(void)
{
  glm::vec4  bbox(0.f, 1.0f, 0.f, 1.f); // x1,x2,y1,y2
  glm::ivec2 shape(1024, 1024);
  glm::ivec2 tile_shape(256, 256);
  size_t     halo = 32;

  // shape = {1000, 581};
  // tile_shape = {220, 312};

  // shape = {10000, 10000};
  // tile_shape = {4096, 4096};

  // auto mode = hmap::ForEachMode::VA_SEQUENTIAL;
  auto mode = hmap::ForEachMode::VA_DISTRIBUTED;
  // auto mode = hmap::ForEachMode::VA_SINGLE_ARRAY;

  // auto storage_mode = hmap::StorageMode::VA_RAM;
  auto storage_mode = hmap::StorageMode::VA_DISK_LRU;
  // auto storage_mode = hmap::StorageMode::VA_DISK_LRU_MIN;
  // auto storage_mode = hmap::StorageMode::VA_DISK_SEQUENTIAL;

  // auto storage = std::make_unique<hmap::RamTileStorage>();

  hmap::VirtualArray varray(shape, bbox, tile_shape, halo, storage_mode);
  hmap::VirtualArray varray2(shape, bbox, tile_shape, halo, storage_mode);

  auto lambda = [](hmap::Array &tile, const hmap::TileRegion &)
  { hmap::gamma_correction(tile, 2.f); };

  auto lambda_s = [](hmap::Array &tile, const hmap::TileRegion &)
  { hmap::smooth_cpulse(tile, 64); };

  auto lambda_noise = [](hmap::Array &tile, const hmap::TileRegion &region)
  {
    tile = hmap::noise(hmap::NoiseType::PERLIN,
                       region.shape,
                       {2.f, 2.f},
                       0,
                       nullptr,
                       nullptr,
                       nullptr,
                       region.bbox);
  };

  hmap::for_each_tile(varray, lambda_noise, mode);

  hmap::for_each_tile(varray, lambda, mode);
  // hmap::for_each_tile(varray, lambda_s, mode);
  // hmap::for_each_tile(varray, lambda_s, mode);
  hmap::for_each_tile(varray, lambda_s, mode);

  auto lambda_f =
      [](std::vector<hmap::Array *> p_arrays, const hmap::TileRegion &)
  {
    hmap::Array *pa_0 = p_arrays[0];
    hmap::Array *pa_1 = p_arrays[1];

    *pa_1 = *pa_0;
    hmap::make_binary(*pa_1, 0.5f);
  };

  hmap::for_each_tile({&varray, &varray2}, lambda_f, mode);
  // hmap::for_each_tile_single_array({&varray, &varray2}, lambda_f);

  // --- Computing methods

  varray.smooth_overlap_buffers();

  varray.remap();

  std::cout << "min: " << varray.min(mode) << "\n";
  std::cout << "max: " << varray.max(mode) << "\n";
  std::cout << "sum: " << varray.sum(mode) << "\n";
  std::cout << "mean: " << varray.mean(mode) << "\n";

  for (const auto &v : varray2.unique_values())
    std::cout << "unique: " << v << "\n";

  auto a = varray.to_array(mode);
  a.to_png("out0.png", hmap::Cmap::JET);

  auto b = varray.to_array({256, 256}, mode);
  b.to_png("out2.png", hmap::Cmap::JET);

  auto a2 = varray2.to_array(mode);
  a2.to_png("out1.png", hmap::Cmap::JET);

  // auto gn = hmap::gradient_norm(a);
  // gn.to_png("out1.png", hmap::Cmap::JET);
}
