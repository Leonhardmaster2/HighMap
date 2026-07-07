#include <cmath>
#include <iostream>

#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  auto facc_d8 = hmap::flow_accumulation_d8(z);
  auto facc_dinf = hmap::flow_accumulation_dinf(z,
                                                1.f / shape.x); // for reference
  auto facc_st = hmap::flow_accumulation_stochastic(z, 1 << 18, 1);

  hmap::gpu::init_opencl();
  auto facc_gpu = hmap::gpu::flow_accumulation_stochastic(z, 1 << 18, 1);

  // CPU/GPU agreement: Pearson correlation on log(1 + flux)
  double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  int    ncells = shape.x * shape.y;
  for (int k = 0; k < ncells; ++k)
  {
    double a = std::log1p((double)facc_st.vector[k]);
    double b = std::log1p((double)facc_gpu.vector[k]);
    sx += a;
    sy += b;
    sxx += a * a;
    syy += b * b;
    sxy += a * b;
  }
  double corr = (ncells * sxy - sx * sy) / (std::sqrt(ncells * sxx - sx * sx) *
                                            std::sqrt(ncells * syy - sy * sy));

  std::cout << "stochastic flow acc. min/max: " << facc_st.min() << " "
            << facc_st.max() << "\n";
  std::cout << "CPU/GPU log-flux correlation: " << corr << "\n";

  // flux spans many orders of magnitude, so log-scale before rendering to
  // PNG (linear scaling makes the image mostly black with a few blown-out
  // pixels); apply the same treatment to the D8 output so both images stay
  // visually comparable
  auto facc_d8_log = hmap::log10(1.f + facc_d8);
  auto facc_st_log = hmap::log10(1.f + facc_st);
  auto facc_dinf_log = hmap::log10(1.f + facc_dinf);

  z.to_png("ex_flow_accumulation_stochastic0.png", hmap::Cmap::TERRAIN, true);
  facc_d8_log.to_png("ex_flow_accumulation_stochastic1.png", hmap::Cmap::HOT);
  facc_st_log.to_png("ex_flow_accumulation_stochastic2.png", hmap::Cmap::HOT);
  facc_dinf_log.to_png("ex_flow_accumulation_stochastic3.png", hmap::Cmap::HOT);
}
