#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 3.f};

  hmap::Array z = hmap::checkerboard(shape, kw);

  z.to_png("ex_checkerboard.png", hmap::Cmap::GRAY);
}
