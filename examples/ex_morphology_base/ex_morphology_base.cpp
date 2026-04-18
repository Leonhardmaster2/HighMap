#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::clamp_min(z, 0.f);
  hmap::remap(z);

  // radius = 1 ==> 3x3 square kernel
  int ir = 3;

  hmap::Array zr = hmap::border(z, ir);
  hmap::Array zd = hmap::dilation(z, ir);
  hmap::Array ze = hmap::erosion(z, ir);
  hmap::Array zc = hmap::closing(z, ir);
  hmap::Array zo = hmap::opening(z, ir);
  hmap::Array zg = hmap::morphological_gradient(z, ir);
  hmap::Array zt = hmap::morphological_top_hat(z, ir);
  hmap::Array zh = hmap::morphological_black_hat(z, ir);
  hmap::Array zl = hmap::morphological_laplacian(z, ir);

  // hmap::make_binary(z);
  // hmap::Array zr = hmap::border(z, ir);

  hmap::export_banner_png("ex_morphology_base.png",
                          {z, zr, zd, ze, zc, zo, zg, zt, zh, zl},
                          hmap::Cmap::GRAY,
                          false,
                          true);

  // --- Wrapper

  std::vector<hmap::MorphologyOperation> ops = {
      hmap::MorphologyOperation::MO_BORDER,
      hmap::MorphologyOperation::MO_CLOSING,
      hmap::MorphologyOperation::MO_DILATION,
      hmap::MorphologyOperation::MO_EROSION,
      hmap::MorphologyOperation::MO_OPENING,
      hmap::MorphologyOperation::MO_GRADIENT,
      hmap::MorphologyOperation::MO_TOP_HAT,
      hmap::MorphologyOperation::MO_BLACK_HAT,
      hmap::MorphologyOperation::MO_LAPLACIAN,
  };

  // CPU
  std::vector<hmap::Array> arrays = {z};

  for (const auto op : ops)
  {
    hmap::Array out = hmap::morphological_operators(z, ir, op);
    arrays.push_back(out);
  }

  hmap::export_banner_png("ex_morphology_base0.png",
                          arrays,
                          hmap::Cmap::INFERNO,
                          false,
                          true);

  // GPU
  hmap::gpu::init_opencl();

  arrays = {z};

  for (const auto op : ops)
  {
    hmap::Array out = hmap::gpu::morphological_operators(z, ir, op);
    arrays.push_back(out);
  }

  hmap::export_banner_png("ex_morphology_base1.png",
                          arrays,
                          hmap::Cmap::INFERNO,
                          false,
                          true);
}
