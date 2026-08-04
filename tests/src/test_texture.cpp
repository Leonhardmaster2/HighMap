#include "highmap/colorize.hpp"
#include "highmap/colormaps.hpp"
#include "highmap/dbg/assert.hpp"
#include "highmap/primitives.hpp"
#include "highmap/texture.hpp"

#include <gtest/gtest.h>

using namespace hmap;

TEST(TextureIO, PngRoundTripOrientationAndData)
{
  glm::ivec2 shape = {16, 32};

  // create a 3-channel texture with a distinct coordinate/color pattern to
  // check orientation
  Texture tex(shape, 3);

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      // Red channel depends on X, Green on Y, Blue on diagonal
      float r = float(i) / float(shape.x - 1);
      float g = float(j) / float(shape.y - 1);
      float b = float(i + j) / float(shape.x + shape.y - 2);
      tex.set_pixel(i, j, glm::vec3(r, g, b));
    }

  const std::string fname = "test_texture_roundtrip.png";

  // to_png flips vertically (cv::flip(mat, mat, 0)) to match typical graphics
  // conventions
  tex.to_png(fname, CV_8U);

  // since to_png flips y, we construct the loaded texture with flip_j = true to
  // invert it back
  Texture loaded(fname, true);

  EXPECT_EQ(loaded.shape.x, tex.shape.x);
  EXPECT_EQ(loaded.shape.y, tex.shape.y);

  // file load constructor creates 4 channels (RGBA)
  EXPECT_EQ(loaded.num_channels(), 4);

  // verify that the data and orientation match within CV_8U quantization error
  // (1.0 / 255.0)
  float tolerance = 1.5f / 255.f;

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      glm::vec3 orig = tex.get_pixel3(i, j);
      glm::vec4 read_val = loaded.get_pixel4(i, j);

      EXPECT_NEAR(orig.x, read_val.x, tolerance);
      EXPECT_NEAR(orig.y, read_val.y, tolerance);
      EXPECT_NEAR(orig.z, read_val.z, tolerance);
      EXPECT_NEAR(read_val.w,
                  1.0f,
                  tolerance); // Alpha should be filled with 1.f
    }
}

TEST(TextureColorize, CustomColormapAndOperations)
{
  glm::ivec2 shape = {8, 8};
  Array      level = white(shape, 0.f, 1.f, 0);

  // Red, Green, Blue at positions 0.0, 0.5, 1.0
  std::vector<float>     positions = {0.f, 0.5f, 1.f};
  std::vector<glm::vec3> colors = {glm::vec3(1.f, 0.f, 0.f),
                                   glm::vec3(0.f, 1.f, 0.f),
                                   glm::vec3(0.f, 0.f, 1.f)};

  Texture col = colorize(level, 0.f, 1.f, positions, colors, false, nullptr);
  EXPECT_EQ(col.num_channels(), 3);
  EXPECT_EQ(col.shape, shape);

  // Check luminance
  Array lum = luminance(col);
  EXPECT_EQ(lum.shape, shape);
  for (int j = 0; j < shape.y; ++j)
  {
    for (int i = 0; i < shape.x; ++i)
    {
      glm::vec3 rgb = col.get_pixel3(i, j);
      float     expected_lum = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
      EXPECT_NEAR(lum(i, j), expected_lum, 1e-5f);
    }
  }

  // Check mixing
  Texture t1(shape, 4, 0.f);
  Texture t2(shape, 4, 1.f);
  // Set alpha channels
  t1[3] = Array(shape, 1.f);
  t2[3] = Array(shape, 1.f);

  Texture t_mixed = mix(t1, t2, MixMethod::MM_LINEAR);
  EXPECT_EQ(t_mixed.num_channels(), 4);
  // Mixing equal alphas (1.f, 1.f) with mixing factor t = 1.0 / (1.0 + 1.0 *
  // 0.0) = 1.0 Result should equal t2
  for (int c = 0; c < 3; ++c)
  {
    for (int idx = 0; idx < shape.x * shape.y; ++idx)
    {
      EXPECT_NEAR(t_mixed[c].vector[idx], 1.f, 1e-5f);
    }
  }

  // Also test Mixbox mixing compiles and runs
  Texture t_mixbox = mix(t1, t2, MixMethod::MM_MIXBOX);
  EXPECT_EQ(t_mixbox.num_channels(), 4);

  // Check mix_normal_map basic execution
  Texture n1(shape, 4, 0.5f); // flat normals
  Texture n2(shape, 4, 0.5f);
  n1[2] = Array(shape, 1.f);
  n2[2] = Array(shape, 1.f);

  Texture blended = mix_normal_map(n1,
                                   n2,
                                   1.0f,
                                   NormalMapBlendingMethod::NMAP_LINEAR);
  EXPECT_EQ(blended.num_channels(), 4);
}
