#include "highmap.hpp"

int main(void)
{
  glm::ivec2               shape = {256, 256};
  std::vector<hmap::Array> arrays;

  float delta = 3.f;

  std::vector<hmap::RadialProfile> profiles = {
      hmap::RadialProfile::RP_GAIN,
      hmap::RadialProfile::RP_GAIN,
      hmap::RadialProfile::RP_LINEAR,
      hmap::RadialProfile::RP_POW,
      hmap::RadialProfile::RP_SMOOTHSTEP,
      hmap::RadialProfile::RP_SMOOTHSTEP_UPPER,
      hmap::RadialProfile::RP_FLAT_BOTTOM,
      hmap::RadialProfile::RP_SQRT};

  for (const auto radial_profile : profiles)
  {
    auto fct = hmap::get_radial_profile_function(radial_profile, delta);

    // plot to an array
    hmap::Array mask(shape);

    for (int i = 0; i < shape.x; ++i)
    {
      float r = float(i) / float(shape.x - 1);
      float f = fct(r);
      int   j0 = int(f * (shape.y - 1));

      for (int j = 0; j < j0; ++j)
        mask(i, j) = 1.f;
    }

    arrays.push_back(mask);
  }

  hmap::export_banner_png("ex_radial_profile_function.png",
                          arrays,
                          hmap::Cmap::GRAY);
}
