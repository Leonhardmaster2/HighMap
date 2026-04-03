#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {1.f, 1.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::Array talus = hmap::gradient_talus(z);

  auto facc = hmap::flow_accumulation_dinf(z, talus.max());

  // very high values are less relevant
  hmap::clamp_max(facc, 200.f);

  int         nsamples = 20;
  glm::vec2   kw = {4.f, 4.f};
  float       amp = 0.1f;
  hmap::Array scaling = hmap::cubic_pulse(shape, nullptr, nullptr);

  auto facc_p = hmap::flow_accumulation_dinf_perturbed(z,
                                                       talus.max(),
                                                       nsamples,
                                                       kw,
                                                       seed,
                                                       amp,
                                                       &scaling);

  hmap::clamp_max(facc_p, 200.f);

  z.to_png("ex_flow_accumulation_dinf0.png", hmap::Cmap::TERRAIN, true);
  facc.to_png("ex_flow_accumulation_dinf1.png", hmap::Cmap::HOT);
  facc_p.to_png("ex_flow_accumulation_dinf2.png", hmap::Cmap::HOT);
}
