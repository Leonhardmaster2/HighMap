#include <iostream>

#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  auto facc_d8 = hmap::flow_accumulation_d8(z);
  auto facc_st = hmap::flow_accumulation_stochastic(z, 1 << 18, 1);

  std::cout << "stochastic flow acc. min/max: " << facc_st.min() << " "
            << facc_st.max() << "\n";

  z.to_png("ex_flow_accumulation_stochastic0.png", hmap::Cmap::TERRAIN, true);
  facc_d8.to_png("ex_flow_accumulation_stochastic1.png", hmap::Cmap::HOT);
  facc_st.to_png("ex_flow_accumulation_stochastic2.png", hmap::Cmap::HOT);
}
