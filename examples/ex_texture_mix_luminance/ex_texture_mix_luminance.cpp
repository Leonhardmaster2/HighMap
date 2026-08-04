#include "highmap.hpp"

int main(void)
{
  const glm::ivec2 shape = {256, 256};
  const glm::vec2  res = {4.f, 4.f};
  int              seed = 42;

  // Generate some base noise layers
  hmap::Array base = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(base);

  // Colorize with custom position and colormap colors
  std::vector<float>     positions = {0.0f, 0.5f, 1.0f};
  std::vector<glm::vec3> colors = {
      glm::vec3(0.1f, 0.05f, 0.05f), // dark brown
      glm::vec3(0.8f, 0.7f, 0.5f),   // sand
      glm::vec3(0.9f, 0.95f, 1.0f)   // snow-white
  };
  hmap::Texture tex1 = hmap::colorize(base, 0.0f, 1.0f, positions, colors);

  // Create another texture (e.g. green terrain tint)
  hmap::Texture tex2 = hmap::colorize(
      base,
      0.0f,
      1.0f,
      {0.0f, 1.0f},
      {glm::vec3(0.0f, 0.2f, 0.0f), glm::vec3(0.4f, 0.8f, 0.3f)});

  // Convert to RGBA for blending (mix demands 4 channels)
  hmap::Texture rgba1(tex1[0],
                      tex1[1],
                      tex1[2],
                      hmap::Array(shape, 0.7f)); // 70% opacity
  hmap::Texture rgba2(tex2[0],
                      tex2[1],
                      tex2[2],
                      hmap::Array(shape, 0.3f)); // 30% opacity

  // Mix the two textures together
  hmap::Texture blended = hmap::mix(rgba1, rgba2, hmap::MixMethod::MM_LINEAR);

  // Save the mixed texture
  blended.to_png("ex_texture_mixed.png");

  // Compute luminance
  hmap::Array lum = hmap::luminance(blended);
  lum.to_png_grayscale("ex_texture_luminance.png");
}
