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

  std::vector<hmap::CurvatureType> types = {
      hmap::CurvatureType::CT_MIN,
      hmap::CurvatureType::CT_MAX,
      hmap::CurvatureType::CT_MEAN,
      hmap::CurvatureType::CT_GAUSSIAN,
      hmap::CurvatureType::CT_PROFILE,
      hmap::CurvatureType::CT_CONTOUR,
      hmap::CurvatureType::CT_TANGENTIAL,
      hmap::CurvatureType::CT_ACCUMULATION,
      hmap::CurvatureType::CT_SHAPE_INDEX,
      hmap::CurvatureType::CT_UNSPHERICITY,
      hmap::CurvatureType::CT_RING,
      hmap::CurvatureType::CT_ROTOR,
  };

  for (const auto type : types)
  {
    {
      hmap::Array out = hmap::curvature_quadric(z, ir, type);
      arrays.push_back(out);
    }

    {
      hmap::Array out = hmap::gpu::curvature_quadric(z, ir, type);
      arrays.push_back(out);
    }
  }

  hmap::export_banner_png("ex_curvature_quadric.png",
                          arrays,
                          hmap::Cmap::JET,
                          false,
                          true);
}
