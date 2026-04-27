#include <iostream>

#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {1024, 1024};
  int        seed = 1;

  // isotropic
  {
    glm::vec2   kw = {32.f, 32.f};
    hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);

    float scale = hmap::autocorr_length_scale(z);
    float scale_ref_x = 0.5f * shape.x / kw.x;
    float scale_ref_y = 0.5f * shape.y / kw.y;

    std::cout << "--- Scales ---\n";
    std::cout << "  scale: " << scale << "\n";
    std::cout << "  scale_ref_x: " << scale_ref_x << "\n";
    std::cout << "  scale_ref_y: " << scale_ref_y << "\n";
  }

  // per grid direction length-scales
  {
    glm::vec2   kw = {32.f, 8.f};
    hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);

    auto  scale = hmap::autocorr_length_scale_axial(z);
    float scale_ref_x = 0.5f * shape.x / kw.x;
    float scale_ref_y = 0.5f * shape.y / kw.y;

    std::cout << "--- Scales ---\n";
    std::cout << "  scale.x: " << scale.x << "\n";
    std::cout << "  scale.y: " << scale.y << "\n";
    std::cout << "  scale_ref_x: " << scale_ref_x << "\n";
    std::cout << "  scale_ref_y: " << scale_ref_y << "\n";
  }
}
