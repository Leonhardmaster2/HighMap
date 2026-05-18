#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};

  // --- Base

  float       radius = 0.3f;
  hmap::Array z1 = hmap::crater(shape, radius);

  // --- With noise

  float     noise_amp = 0.2f;
  glm::vec2 kw = {4.f, 4.f};
  int       seed = 1;

  auto noise = noise_amp *
               hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::Array crater_mask;

  radius = 0.2f;
  glm::vec2 center = {0.8f, 0.6f};
  float     angle = -15.f;
  float     inner_depth = 0.1f;
  float     inner_exp = 3.f;
  float     lip_height = 0.02f;
  float     lip_extent = 0.25f;
  float     lip_exp = 2.f;
  float     asym_ratio = 0.8f;
  float     central_peak_height = 0.01f;
  float     central_peak_extent = 0.4f;
  int       n_terraces = 0;
  float     terrace_extent = 0.1f;
  float     terrace_exp = 2.f;
  float     terrace_persistence = 0.5f;
  glm::vec4 bbox = {0.f, 1.f, 0.f, 1.f};

  hmap::Array z2 = hmap::crater(shape,
                                radius,
                                center,
                                angle,
                                inner_depth,
                                inner_exp,
                                lip_height,
                                lip_extent,
                                lip_exp,
                                asym_ratio,
                                central_peak_height,
                                central_peak_extent,
                                n_terraces,
                                terrace_extent,
                                terrace_exp,
                                terrace_persistence,
                                &noise,
                                bbox,
                                &crater_mask,
                                /* p_inner_crater_mask */ nullptr);

  // --- Crater merging - mask-based

  // 2nd crater hits after the 1st one and replaces the ground
  hmap::Array z3 = lerp(z1, z2, crater_mask);

  hmap::export_banner_png("ex_crater.png",
                          {z1, z2, z3},
                          hmap::Cmap::TERRAIN,
                          true);
}
