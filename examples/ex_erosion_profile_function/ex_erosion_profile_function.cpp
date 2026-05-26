#include "highmap.hpp"

int main(void)
{
  glm::ivec2               shape = {256, 256};
  std::vector<hmap::Array> arrays;

  float delta = 3.f;

  std::vector<hmap::ErosionProfile> profiles = {
      hmap::ErosionProfile::EP_COSINE,
      hmap::ErosionProfile::EP_COSINE_BULK,
      hmap::ErosionProfile::EP_COSINE_PEAK,
      hmap::ErosionProfile::EP_PARABOL,
      hmap::ErosionProfile::EP_SAW_SHARP,
      hmap::ErosionProfile::EP_SAW_SMOOTH,
      hmap::ErosionProfile::EP_SHARP_VALLEYS,
      hmap::ErosionProfile::EP_SQRT,
      hmap::ErosionProfile::EP_TRIANGLE_GRENIER,
      hmap::ErosionProfile::EP_TRIANGLE_SHARP,
      hmap::ErosionProfile::EP_TRIANGLE_SMOOTH};

  for (const auto profile : profiles)
  {
    float profile_avg; // not used here
    auto  fct = hmap::get_erosion_profile_function(profile, delta, profile_avg);

    // plot to an array
    hmap::Array mask(shape);

    for (int i = 0; i < shape.x; ++i)
    {
      float r = float(i) / float(shape.x - 1); // in [0, 1]
      r = M_PI * (2.f * r - 1.f);              // in [-pi, pi]
      float f = fct(r);                        // in [0, 1]

      int j0 = int(f * (shape.y - 1));

      for (int j = 0; j < j0; ++j)
        mask(i, j) = 1.f;
    }

    arrays.push_back(mask);
  }

  hmap::export_banner_png("ex_erosion_profile_function.png",
                          arrays,
                          hmap::Cmap::GRAY);
}
