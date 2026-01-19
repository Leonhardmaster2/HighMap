#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  int        seed = 1;

  hmap::Array z = hmap::white(shape, 0.f, 1.f, seed);
  z.to_png("ex_white.png", hmap::Cmap::VIRIDIS);
}
