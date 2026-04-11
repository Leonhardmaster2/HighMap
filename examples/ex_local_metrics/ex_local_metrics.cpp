#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);

  int ir = 8;

  hmap::gpu::init_opencl();

  std::vector<hmap::Array> arrays = {z};

  std::vector<hmap::gpu::LocalMetrics> metrics = {
      hmap::gpu::LocalMetrics::LM_LOCAL_ASPECT_VARIANCE,
      hmap::gpu::LocalMetrics::LM_LOCAL_MAX,
      hmap::gpu::LocalMetrics::LM_LOCAL_MEDIAN_DEVIATION,
      hmap::gpu::LocalMetrics::LM_LOCAL_MIN,
      hmap::gpu::LocalMetrics::LM_LOCAL_RELIEF,
      hmap::gpu::LocalMetrics::LM_LOCAL_VARIANCE,
      hmap::gpu::LocalMetrics::LM_LOCAL_MEAN,
      hmap::gpu::LocalMetrics::LM_LOCAL_Z_SCORE,
      hmap::gpu::LocalMetrics::LM_TOPOGRAPHIC_POSITION_INDEX,
      hmap::gpu::LocalMetrics::LM_RELATIVE_ELEVATION,
      hmap::gpu::LocalMetrics::LM_RUGGEDNESS,
      hmap::gpu::LocalMetrics::LM_RUGOSITY_CONCAVE,
      hmap::gpu::LocalMetrics::LM_RUGOSITY_CONVEX,
  };

  for (const auto metric : metrics)
  {
    hmap::Array out = hmap::gpu::local_metrics(z, ir, metric);
    arrays.push_back(out);
  }

  hmap::export_banner_png("ex_local_metrics.png",
                          arrays,
                          hmap::Cmap::JET,
                          false,
                          true);
}
