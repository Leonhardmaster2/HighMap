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

  // --- base texture

  auto tex = hmap::VirtualTexture(shape,
                                  bbox,
                                  tile_shape,
                                  halo,
                                  4,
                                  storage_mode);

  auto lambda0 =
      [](std::vector<hmap::Array *> &tiles, const hmap::TileRegion &region)
  {
    hmap::Array &R = *tiles[0];
    hmap::Array &G = *tiles[1];
    hmap::Array &B = *tiles[2];
    hmap::Array &A = *tiles[3];

    for (int j = 0; j < region.shape.y; ++j)
      for (int i = 0; i < region.shape.x; ++i)
      {
        glm::vec2 pos = region.cell_center(i, j);

        R(i, j) = pos.x;
        G(i, j) = 0.f;
        B(i, j) = pos.y;
        A(i, j) = 1.f; // pos.x;
      }
  };

  hmap::for_each_tile(tex.channels_ptr(), lambda0, cm);

  tex.to_png("ex_color_adjust0.png", cm);

  // --- modify

  auto tex_adjust = hmap::VirtualTexture();
  tex_adjust.copy_from(tex, cm);

  hmap::ColorAdjust param;
  // param.gamma = 2.1f;
  param.aces_tonemap = true;

  hmap::color_adjust(tex, param, cm);

  tex.to_png("ex_color_adjust1.png", cm);
}
