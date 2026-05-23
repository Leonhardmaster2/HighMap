#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  shape = {1024, 1024};
  std::uint32_t seed = 0;
  glm::vec4     bbox = hmap::unit_square_bbox();
  glm::vec2     center = {0.f, 0.f};

  // std::unique_ptr<hmap::Function> p = std::unique_ptr<hmap::Function>(
  //     new hmap::BumpFunction(gain, center));

  std::unique_ptr<hmap::Function> p = std::unique_ptr<hmap::Function>(
      new hmap::CraterFunction(0.2f, 0.12f, 0.5f, 0.5f, center));

  // std::unique_ptr<hmap::NoiseFunction> f =
  // std::unique_ptr<hmap::NoiseFunction>(
  //     new hmap::PerlinFunction({1.f, 1.f}, seed));
  // std::unique_ptr<hmap::Function> p = std::unique_ptr<hmap::Function>(
  //     new hmap::FbmFunction(std::move(f), 8, 0.7f, 0.5f, 2.f));

  hmap::Cloud cloud = hmap::Cloud(15, seed, bbox);
  cloud.remap_values(1.f, 4.f);

  hmap::FieldFunction field_fct = hmap::FieldFunction(std::move(p),
                                                      cloud.get_x(),
                                                      cloud.get_y(),
                                                      cloud.get_values());

  //

  hmap::Array z = hmap::Array(shape);

  glm::vec2 kw = {4.f, 4.f};
  auto      n = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(n);

  fill_array_using_xy_function(z,
                               glm::vec4(0.f, 1.f, 0.f, 1.f),
                               &n,
                               nullptr,
                               nullptr,
                               nullptr,
                               field_fct.get_delegate());

  z.infos("z");

  z.to_png("ex_field_function.png", hmap::Cmap::GRAY, true);
}
