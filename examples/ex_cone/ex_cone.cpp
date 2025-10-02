#include "highmap.hpp"

int main(void)
{
  hmap::Vec2<int> shape = {256, 256};

  hmap::Array z = hmap::cone(shape, 4.f / shape.x);

  z.to_png("ex_cone.png", hmap::Cmap::MAGMA);
}
