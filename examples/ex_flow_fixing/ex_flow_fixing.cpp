#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z0);

  float riverbed_talus = 0.01f / shape.x;
  auto  z1 = hmap::flow_fixing(z0, riverbed_talus);
  auto  z2 = hmap::flow_fixing_drainage_basin(z0,
                                             hmap::FlowDirectionMethod::FDM_D8,
                                             riverbed_talus,
                                             50,
                                             true);
  auto  z3 = hmap::flow_fixing_mst(z0,
                                  riverbed_talus,
                                  0.95f,  // elevation_ratio
                                  2.f,    // distance_exponent
                                  50.f,   // upward_penalization
                                  0.3f,   // valley_affinity
                                  0.3f,   // path_sinuosity
                                  1,      // prefilter_ir
                                  true);  // carve_riverbed

  hmap::export_banner_png("ex_flow_fixing.png",
                          {z0, z1, z2, z3},
                          hmap::Cmap::TERRAIN,
                          true);
}
