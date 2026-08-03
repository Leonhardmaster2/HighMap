#include "highmap/dbg/assert.hpp"
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
