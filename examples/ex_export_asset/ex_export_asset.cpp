#include <iostream>

#include "highmap.hpp"

int main(void)
{

  // --- heightmaps

  glm::ivec2 shape = {256, 256};
  glm::vec2  kv = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise(hmap::NoiseType::SIMPLEX2, shape, kv, seed);
  hmap::clamp_min_smooth(z, 0.f, 0.2f);
  hmap::remap(z);

  z.to_png("hmap.png", hmap::Cmap::TERRAIN);
  hmap::export_normal_map_png("name.png", z);

  hmap::Array mask = hmap::noise(hmap::NoiseType::SIMPLEX2,
                                 shape,
                                 4.f * kv,
                                 ++seed);
  hmap::clamp_min(mask, 0.f);

  mask.dump("mask.png");

  for (auto &[export_id, export_infos] : hmap::asset_export_format_as_string)
  {
    std::cout << "exporting format: " << export_infos[0] << "\n";

    float error_tolerance = 1e-2f; // for tri_optimized only

    hmap::export_asset("hmap.dummy_extension",
                       z,
                       hmap::MeshType::TRI,
                       export_id,
                       0.2f,
                       "hmap.png",
                       "nmap.png", // normal map
                       error_tolerance);

    hmap::export_asset("hmap.dummy_extension_opt",
                       z,
                       hmap::MeshType::TRI_OPTIMIZED,
                       export_id,
                       0.2f,
                       "hmap.png",
                       "nmap.png", // normal map
                       error_tolerance);

    hmap::export_asset("hmap_masked",
                       z,
                       mask,
                       export_id,
                       0.2f,
                       "hmap.png",
                       "nmap.png");
  }

  return 0;
}
