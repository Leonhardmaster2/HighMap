#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  hmap::Vec2<int> shape = {256, 256};
  shape = {1024, 1024};
  hmap::Vec2<float> kw = {4.f, 4.f};
  uint              seed = 0;

  hmap::Array z1 = hmap::gpu::plates(shape, kw, seed, 2.f / shape.x);

  z1.dump();

  hmap::export_banner_png("ex_plates.png", {z1}, hmap::Cmap::TERRAIN, true);
}
