#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  shape = {1024, 1024};
  glm::vec2 kw = {4.f, 4.f};
  int       seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                   shape,
                                   kw,
                                   seed,
                                   8,
                                   0.f);
  z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CONE_SMOOTH, 1.f);
  hmap::remap(z0);

  int nparticles = int(0.5f * shape.x * shape.y);

  auto z1 = z0;
  hmap::gpu::hydraulic_particle(z1,
                                nparticles,
                                seed,
                                /* p_bedrock */ nullptr,
                                /* p_moisture_map */ nullptr,
                                /* p_elevation_shift */ nullptr,
                                /* p_erosion_map */ nullptr,
                                /* p_deposition_map */ nullptr,
                                /* c_capacity */ 10.f,
                                /* c_erosion */ 0.05f,
                                /* c_deposition */ 0.01f,
                                /* c_inertia */ 0.f,
                                /* c_gravity */ 1.f,
                                /* drag_rate */ 0.001f,
                                /* evap_rate */ 0.001f,
                                /* enable_directional_bias */ false,
                                /* angle_bias */ 30.f,
                                /* iterations */ 16);

  auto z2 = z0;
  auto zb2 = hmap::generate_bedrock(z2,
                                    /* elevation_strength */ 0.05f,
                                    /* slope_strength */ 0.f,
                                    /* slope_limit */ 0.f);
  hmap::gpu::hydraulic_particle(z2, nparticles, seed, &zb2);

  auto z3 = z0;
  auto zb3 = hmap::generate_bedrock(z3,
                                    /* elevation_strength */ 0.f,
                                    /* slope_strength */ 0.1f,
                                    /* slope_limit */ 2.f / shape.x);
  hmap::gpu::hydraulic_particle(z3, nparticles, seed, &zb3);

  // --- output

  z0.dump("out0.png");
  z1.dump("out1.png");
  z2.dump("out2.png");
  z3.dump("out3.png");

  hmap::export_banner_png("ex_hydraulic_particle.png",
                          {z0, z1, z2, z3},
                          hmap::Cmap::TERRAIN,
                          true);
}
