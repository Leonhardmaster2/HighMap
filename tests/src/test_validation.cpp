#include <cmath>
#include <limits>
#include <sstream>

#include "highmap/array.hpp"
#include "highmap/internal/validation.hpp"
#include "highmap/logger.hpp"

#include <gtest/gtest.h>

TEST(ValidationTest, ValidateShape)
{
  EXPECT_TRUE(hmap::validate_shape(glm::ivec2(10, 10)));
  EXPECT_TRUE(hmap::validate_shape(glm::ivec2(1, 1)));

  EXPECT_FALSE(hmap::validate_shape(glm::ivec2(0, 10)));
  EXPECT_FALSE(hmap::validate_shape(glm::ivec2(10, 0)));
  EXPECT_FALSE(hmap::validate_shape(glm::ivec2(-5, 10)));
  EXPECT_FALSE(hmap::validate_shape(glm::ivec2(10, -5)));
}

TEST(ValidationTest, ValidateNonEmpty)
{
  hmap::Array empty_arr;
  EXPECT_FALSE(hmap::validate_non_empty(empty_arr));

  hmap::Array valid_arr(glm::ivec2(4, 4), 1.0f);
  EXPECT_TRUE(hmap::validate_non_empty(valid_arr));

  hmap::Array bad_buffer_arr = valid_arr;
  bad_buffer_arr.vector.pop_back();
  EXPECT_FALSE(hmap::validate_non_empty(bad_buffer_arr));
}

TEST(ValidationTest, ValidateSameShape)
{
  hmap::Array a(glm::ivec2(4, 4), 1.0f);
  hmap::Array b(glm::ivec2(4, 4), 2.0f);
  hmap::Array c(glm::ivec2(4, 5), 1.0f);

  EXPECT_TRUE(hmap::validate_same_shape(a, b));
  EXPECT_FALSE(hmap::validate_same_shape(a, c));
}

TEST(ValidationTest, ValidateSlice)
{
  hmap::Array arr(glm::ivec2(10, 10));

  EXPECT_TRUE(hmap::validate_slice(arr, glm::ivec4(0, 5, 0, 5)));
  EXPECT_TRUE(hmap::validate_slice(arr, glm::ivec4(0, 10, 0, 10)));

  EXPECT_FALSE(hmap::validate_slice(arr, glm::ivec4(-1, 5, 0, 5)));
  EXPECT_FALSE(hmap::validate_slice(arr, glm::ivec4(5, 5, 0, 5)));
  EXPECT_FALSE(hmap::validate_slice(arr, glm::ivec4(6, 5, 0, 5)));
  EXPECT_FALSE(hmap::validate_slice(arr, glm::ivec4(0, 11, 0, 5)));
  EXPECT_FALSE(hmap::validate_slice(arr, glm::ivec4(0, 5, 0, 11)));
}

TEST(ValidationTest, ValidateSliceForArray)
{
  hmap::Array dest(glm::ivec2(10, 10));
  hmap::Array src_match(glm::ivec2(3, 4));
  hmap::Array src_mismatch(glm::ivec2(2, 2));

  EXPECT_TRUE(
      hmap::validate_slice_for_array(dest, glm::ivec4(1, 4, 2, 6), src_match));
  EXPECT_FALSE(hmap::validate_slice_for_array(dest,
                                              glm::ivec4(1, 4, 2, 6),
                                              src_mismatch));
}

TEST(ValidationTest, ValidateNotZero)
{
  EXPECT_TRUE(hmap::validate_not_zero(1.0f));
  EXPECT_TRUE(hmap::validate_not_zero(-0.5f));
  EXPECT_FALSE(hmap::validate_not_zero(0.0f));
}

TEST(ValidationTest, ValidateParameterRange)
{
  EXPECT_TRUE(hmap::validate_parameter_range(glm::vec2(0.2f, 0.8f),
                                             0.0f,
                                             1.0f,
                                             "Vec2Param"));
  EXPECT_TRUE(
      hmap::validate_parameter_range(glm::vec2(0.5f, 0.5f), 0.0f, 1.0f));
  EXPECT_FALSE(
      hmap::validate_parameter_range(glm::vec2(-0.1f, 0.8f), 0.0f, 1.0f));
  EXPECT_FALSE(
      hmap::validate_parameter_range(glm::vec2(0.2f, 1.1f), 0.0f, 1.0f));
  EXPECT_FALSE(
      hmap::validate_parameter_range(glm::vec2(0.8f, 0.2f), 0.0f, 1.0f));
}

TEST(ValidationTest, ValidateFinite)
{
  hmap::Array arr(glm::ivec2(3, 3), 1.0f);
  EXPECT_TRUE(hmap::validate_finite(arr));

  arr(1, 1) = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(hmap::validate_finite(arr));

  arr(1, 1) = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(hmap::validate_finite(arr));
}

TEST(ValidationTest, SourceLocationLoggingOutput)
{
  std::stringstream buffer;
  std::streambuf   *old_cerr = std::cerr.rdbuf(buffer.rdbuf());

  // Trigger a validation failure at this line
  EXPECT_FALSE(hmap::validate_shape(glm::ivec2(0, 0)));

  std::cerr.rdbuf(old_cerr);

#if HIGHMAP_ENABLE_LOGS
  std::string output = buffer.str();
  EXPECT_NE(output.find("[warn ]"), std::string::npos);
  EXPECT_NE(output.find("Invalid array shape"), std::string::npos);
  EXPECT_NE(output.find("test_validation.cpp"), std::string::npos);
#endif
}

TEST(ValidationTest, ArrayOperationsGracefulFailure)
{
  // Operator on mismatched shapes should log warning and not crash
  hmap::Array a(glm::ivec2(3, 3), 1.0f);
  hmap::Array b(glm::ivec2(4, 4), 2.0f);

  hmap::Array sum = a + b;
  EXPECT_EQ(sum.shape, glm::ivec2(0, 0));

  a += b; // in-place += should be no-op on mismatch
  EXPECT_EQ(a.shape, glm::ivec2(3, 3));
  EXPECT_EQ(a(0, 0), 1.0f);

  // Method on empty array
  hmap::Array empty;
  float       max_val = 0.f;
  int         im = 0, jm = 0;
  empty.argmax(max_val, im, jm);
  EXPECT_EQ(im, -1);
  EXPECT_EQ(jm, -1);
}

TEST(ValidationTest, ValidateTexture)
{
  hmap::Texture empty_tex;
  EXPECT_FALSE(hmap::validate_non_empty(empty_tex));

  hmap::Texture valid_tex(glm::ivec2(4, 4), 3, 1.0f);
  EXPECT_TRUE(hmap::validate_non_empty(valid_tex));
  EXPECT_TRUE(hmap::validate_non_empty(valid_tex, 3));
  EXPECT_FALSE(hmap::validate_non_empty(valid_tex, 4));

  EXPECT_TRUE(hmap::validate_channels(valid_tex, 3));
  EXPECT_FALSE(hmap::validate_channels(valid_tex, 4));

  hmap::Texture diff_shape_tex(glm::ivec2(4, 5), 3, 1.0f);
  EXPECT_FALSE(hmap::validate_same_shape(valid_tex, diff_shape_tex));

  hmap::Array arr(glm::ivec2(4, 4), 1.0f);
  EXPECT_TRUE(hmap::validate_same_shape(valid_tex, arr));

  hmap::Array arr_diff(glm::ivec2(4, 5), 1.0f);
  EXPECT_FALSE(hmap::validate_same_shape(valid_tex, arr_diff));
}

TEST(ValidationTest, ValidateCmap)
{
  EXPECT_TRUE(hmap::validate_cmap(hmap::Cmap::VIRIDIS));
  EXPECT_TRUE(hmap::validate_cmap(hmap::Cmap::BONE));
  EXPECT_TRUE(hmap::validate_cmap(hmap::Cmap::WHITE_UNIFORM));
  EXPECT_FALSE(hmap::validate_cmap(-1));
  EXPECT_FALSE(hmap::validate_cmap(999));

  // get_colormap_data graceful fallback on invalid cmap ID
  auto colors_invalid = hmap::get_colormap_data(999);
  auto colors_default = hmap::get_colormap_data(hmap::Cmap::WHITE_UNIFORM);
  EXPECT_EQ(colors_invalid.size(), colors_default.size());
}
