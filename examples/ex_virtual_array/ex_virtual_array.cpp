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

  // auto storage = std::make_unique<hmap::RamTileStorage>();

  size_t max_tiles = 32;
  auto   storage = std::make_unique<hmap::DiskLruTileStorage>(max_tiles,
                                                            "tiles/");
  hmap::VirtualArray varray(shape, bbox, tile_shape, halo, std::move(storage));

  auto               storage2 = std::make_unique<hmap::RamTileStorage>();
  hmap::VirtualArray varray2(shape,
                             bbox,
                             tile_shape,
                             halo,
                             std::move(storage2));

  auto lambda = [](hmap::Array &tile, const glm::ivec2 &, const glm::vec4 &)
  { hmap::gamma_correction(tile, 2.f); };

  auto lambda_s = [](hmap::Array &tile, const glm::ivec2 &, const glm::vec4 &)
  { hmap::smooth_cpulse(tile, 64); };

  auto lambda_noise =
      [](hmap::Array &tile, const glm::ivec2 &shape, const glm::vec4 &bbox)
  {
    tile = hmap::noise(hmap::NoiseType::PERLIN,
                       shape,
                       {2.f, 2.f},
                       0,
                       nullptr,
                       nullptr,
                       nullptr,
                       bbox);
  };

  hmap::for_each_tile_distributed(varray, lambda_noise);

  hmap::for_each_tile_sequential(varray, lambda);
  // hmap::for_each_tile_sequential(varray, lambda_s);
  // hmap::for_each_tile_sequential(varray, lambda_s);
  hmap::for_each_tile_single_array(varray, lambda_s);

  auto lambda_f = [](std::vector<hmap::Array *> p_arrays,
                     const glm::ivec2 &,
                     const glm::vec4 &)
  {
    hmap::Array *pa_0 = p_arrays[0];
    hmap::Array *pa_1 = p_arrays[1];

    *pa_1 = *pa_0;
    hmap::make_binary(*pa_1, 0.5f);
  };

  hmap::for_each_tile_distributed({&varray, &varray2}, lambda_f);
  // hmap::for_each_tile_single_array({&varray, &varray2}, lambda_f);

  varray.smooth_overlap_buffers();

  auto a = varray.to_array();
  a.to_png("out0.png", hmap::Cmap::JET);

  auto a2 = varray2.to_array();
  a2.to_png("out1.png", hmap::Cmap::JET);

  // auto gn = hmap::gradient_norm(a);
  // gn.to_png("out1.png", hmap::Cmap::JET);
}
