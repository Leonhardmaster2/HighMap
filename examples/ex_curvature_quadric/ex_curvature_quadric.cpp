#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  int ir = 16;

  std::vector<hmap::Array> arrays = {z};

  std::vector<hmap::gpu::CurvatureType> types = {
      hmap::gpu::CurvatureType::CT_MIN,
      hmap::gpu::CurvatureType::CT_MAX,
      hmap::gpu::CurvatureType::CT_MEAN,
      hmap::gpu::CurvatureType::CT_GAUSSIAN,
      hmap::gpu::CurvatureType::CT_PROFILE,
      hmap::gpu::CurvatureType::CT_CONTOUR,
      hmap::gpu::CurvatureType::CT_TANGENTIAL,
      hmap::gpu::CurvatureType::CT_ACCUMULATION,
      hmap::gpu::CurvatureType::CT_SHAPE_INDEX,
      hmap::gpu::CurvatureType::CT_UNSPHERICITY,
      hmap::gpu::CurvatureType::CT_RING,
      hmap::gpu::CurvatureType::CT_ROTOR,
  };

  for (const auto type : types)
  {
    hmap::Array out = hmap::gpu::curvature_quadric(z, ir, type);
    arrays.push_back(out);
  }

  hmap::export_banner_png("ex_curvature_quadric.png",
                          arrays,
                          hmap::Cmap::JET,
                          false,
                          true);
}
