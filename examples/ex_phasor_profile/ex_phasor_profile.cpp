#include "highmap.hpp"

int main(void)
{
  glm::ivec2               shape = {256, 256};
  std::vector<hmap::Array> arrays;

  float delta = 3.f;

  std::vector<hmap::PhasorProfile> profiles = {
      hmap::PhasorProfile::PP_COSINE_BULKY,
      hmap::PhasorProfile::PP_COSINE_PEAKY,
      hmap::PhasorProfile::PP_COSINE_SQUARE,
      hmap::PhasorProfile::PP_COSINE_STD,
      hmap::PhasorProfile::PP_TRIANGLE};

  for (const auto profile : profiles)
  {
    auto fct = hmap::get_phasor_profile_function(profile, delta);

    // plot to an array
    hmap::Array mask(shape);

    for (int i = 0; i < shape.x; ++i)
    {
      float r = float(i) / float(shape.x - 1);
      r = M_PI * (2.f * r - 1.f); // in [0..1]

      // fct output in [-1..1] to [0..1]
      float f = 0.5f * (fct(r) + 1.f);

      int j0 = int(f * (shape.y - 1));

      for (int j = 0; j < j0; ++j)
        mask(i, j) = 1.f;
    }

    arrays.push_back(mask);
  }

  hmap::export_banner_png("ex_phasor_profile.png", arrays, hmap::Cmap::GRAY);
}
