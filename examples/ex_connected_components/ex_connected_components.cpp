#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  uint       seed = 5;

  hmap::Array z = hmap::noise(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::clamp_min(z, 0.f);

  // --- Basic usage

  hmap::Array labels = hmap::connected_components(z);

  // unique labels
  LOG_DEBUG("Labels");
  for (const auto &v : labels.unique_values())
    LOG_DEBUG(" - %f", v);

  // --- Secondary outputs

  std::map<float, float>                surfaces;
  std::map<float, std::array<float, 2>> centroids;
  auto dummy = hmap::connected_components(z, 0.f, 0.f, &surfaces, &centroids);

  // centroids of each components in index (i, j) scale
  LOG_DEBUG("Centroids");
  for (auto &[k, v] : centroids)
    LOG_DEBUG(" - %f %f %f", k, v[0], v[1]);

  // --- Plots

  z.to_png("ex_connected_components0.png", hmap::Cmap::INFERNO);
  labels.to_png("ex_connected_components1.png", hmap::Cmap::NIPY_SPECTRAL);
}
