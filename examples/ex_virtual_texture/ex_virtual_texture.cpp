#include <iostream>

#include "highmap.hpp"

int main(void)
{
  glm::vec4  bbox(0.f, 1.0f, 0.f, 1.f); // x1,x2,y1,y2
  glm::ivec2 shape(1024, 1024);
  glm::ivec2 tile_shape(256, 256);
  size_t     halo = 32;

  shape = {1000, 581};
  tile_shape = {220, 312};

  // shape = {10000, 10000};
  // tile_shape = {4096, 4096};

  // auto mode = hmap::ForEachMode::VA_SEQUENTIAL;
  auto mode = hmap::ForEachMode::VA_DISTRIBUTED;
  // auto mode = hmap::ForEachMode::VA_SINGLE_ARRAY;

  // bool trim_storage = true;
  bool trim_storage = false;

  // auto storage_mode = hmap::StorageMode::VA_RAM;
  auto storage_mode = hmap::StorageMode::VA_DISK_LRU;
  // auto storage_mode = hmap::StorageMode::VA_DISK_LRU_MIN;
  // auto storage_mode = hmap::StorageMode::VA_DISK_SEQUENTIAL;

  hmap::ComputeMode cm = {.mode = mode, .trim_storage = trim_storage};

  auto tex = hmap::VirtualTexture(shape,
                                  bbox,
                                  tile_shape,
                                  halo,
                                  3,
                                  storage_mode);

  auto lambda0 =
      [](std::vector<hmap::Array *> &tiles, const hmap::TileRegion &region)
  {
    hmap::Array &R = *tiles[0];
    hmap::Array &G = *tiles[1];
    hmap::Array &B = *tiles[2];

    for (int j = 0; j < region.shape.y; ++j)
      for (int i = 0; i < region.shape.x; ++i)
      {
        glm::vec2 pos = region.cell_center(i, j);

        R(i, j) = pos.x;
        G(i, j) = 0.f;
        B(i, j) = pos.y;
      }
  };

  hmap::for_each_tile(tex.channels_ptr(), lambda0, cm);

  tex.to_png_dbg("out0.png", cm);

  // --- per pixel

  auto tex2 = hmap::VirtualTexture(shape,
                                   bbox,
                                   tile_shape,
                                   halo,
                                   3,
                                   storage_mode);

  auto lambda1 =
      [](std::vector<float> &px, int i, int j, const hmap::TileRegion &region)
  {
    glm::vec2 pos = region.cell_center(i, j);
    px[0] = pos.x;
    px[1] = 0.f;
    px[2] = pos.y;
  };

  hmap::for_each_pixel(tex2, lambda1, cm);

  tex2.to_png_dbg("out1.png", cm);
}
