#include "highmap.hpp"

int main(void)
{
  glm::ivec2    shape = {256, 256};
  std::uint32_t seed = 0;

  std::mt19937 rng(seed);

  int nparticles = 1000;

  hmap::Array au(shape);
  hmap::Array ad(shape);
  hmap::Array ag(shape);
  hmap::Array ar(shape);
  hmap::Array ab(shape);

  glm::ivec2 center = {shape.x / 2, shape.y / 2};
  int        radius = shape.x / 4;

  for (int n = 0; n < nparticles; ++n)
  {
    {
      glm::ivec2 p = hmap::spawn_uniform(rng, shape);
      au(p) = 1.f;
    }

    {
      glm::ivec2 p = hmap::spawn_disc(rng, center, radius, shape);
      ad(p) = 1.f;
    }

    {
      glm::ivec2 p = hmap::spawn_gaussian(rng, center, radius / 4, shape);
      ag(p) = 1.f;
    }

    {
      glm::ivec2 p = hmap::spawn_ring(rng, center, radius / 4, radius, shape);
      ar(p) = 1.f;
    }

    {
      glm::ivec2 p = hmap::spawn_borders(rng, radius, shape);
      ab(p) = 1.f;
    }
  }

  hmap::export_banner_png("ex_spawner.png",
                          {au, ad, ag, ar, ab},
                          hmap::Cmap::GRAY);
}
