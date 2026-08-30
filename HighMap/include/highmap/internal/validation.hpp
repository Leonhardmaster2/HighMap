/**
 * @file validation.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @brief Array validation functions with source location logging.
 *
 * @copyright Copyright (c) 2023 Otto Link. Distributed under the terms of the
 * GNU General Public License. The full license is in the file LICENSE,
 * distributed with this software.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "highmap/array.hpp"
#include "highmap/colormaps.hpp"
#include "highmap/logger.hpp"
#include "highmap/texture.hpp"

#include <source_location>

namespace hmap
{

/**
 * @brief Validates that array dimensions are strictly positive.
 *
 * @param  shape Dimensions {x, y} to check.
 * @param  loc   Source location of the caller.
 * @return       true if shape.x > 0 and shape.y > 0, false otherwise.
 */
[[nodiscard]] inline bool validate_shape(
    glm::ivec2                  shape,
    const std::source_location &loc = std::source_location::current())
{
  if (shape.x <= 0 || shape.y <= 0)
  {
    hmap::log::warn(loc,
                    "Invalid array shape ({}, {}): dimensions must be positive",
                    shape.x,
                    shape.y);
    return false;
  }
  return true;
}

/**
 * @brief Validates that an array is initialized, non-empty, and has a
 * consistent buffer size.
 *
 * @param  array Array to check.
 * @param  loc   Source location of the caller.
 * @return       true if the array is non-empty and buffer size matches shape
 *               dimensions, false otherwise.
 */
[[nodiscard]] inline bool validate_non_empty(
    const Array                &array,
    const std::source_location &loc = std::source_location::current())
{
  if (array.shape.x <= 0 || array.shape.y <= 0 || array.vector.empty())
  {
    hmap::log::warn(
        loc,
        "Array is empty or uninitialized (shape: {}x{}, buffer size: {})",
        array.shape.x,
        array.shape.y,
        array.vector.size());
    return false;
  }
  if (array.vector.size() != static_cast<size_t>(array.shape.x * array.shape.y))
  {
    hmap::log::warn(
        loc,
        "Array buffer size ({}) does not match shape dimensions ({}x{} = {})",
        array.vector.size(),
        array.shape.x,
        array.shape.y,
        array.shape.x * array.shape.y);
    return false;
  }
  return true;
}

/**
 * @brief Validates that a container or object with an empty() method is not
 * empty.
 *
 * @param  container Container to check.
 * @param  name      Name or description of the container for logging.
 * @param  loc       Source location of the caller.
 * @return           true if container is not empty, false otherwise.
 */
template <typename T>
  requires requires(const T &t) {
    { t.empty() } -> std::convertible_to<bool>;
  }
[[nodiscard]] inline bool validate_non_empty(
    const T                    &container,
    std::string_view            name = "Container",
    const std::source_location &loc = std::source_location::current())
{
  if (container.empty())
  {
    hmap::log::warn(loc, "{} is empty", name);
    return false;
  }
  return true;
}

/**
 * @brief Validates that two arrays have identical shapes and buffer sizes.
 *
 * @param  a   First array.
 * @param  b   Second array.
 * @param  loc Source location of the caller.
 * @return     true if shapes and buffer sizes match, false otherwise.
 */
[[nodiscard]] inline bool validate_same_shape(
    const Array                &a,
    const Array                &b,
    const std::source_location &loc = std::source_location::current())
{
  if (a.shape != b.shape || a.vector.size() != b.vector.size())
  {
    hmap::log::warn(
        loc,
        "Array shape mismatch: lhs is ({}, {}) [size: {}], rhs is ({}, {}) "
        "[size: {}]",
        a.shape.x,
        a.shape.y,
        a.vector.size(),
        b.shape.x,
        b.shape.y,
        b.vector.size());
    return false;
  }
  return true;
}

/**
 * @brief Validates that an array has the expected 2D shape.
 *
 * @param  shape Expected shape {x, y}.
 * @param  b     Array to check.
 * @param  loc   Source location of the caller.
 * @return       true if b.shape matches shape and b is non-empty, false otherwise.
 */
[[nodiscard]] inline bool validate_same_shape(
    glm::ivec2                  shape,
    const Array                &b,
    const std::source_location &loc = std::source_location::current())
{
  if (shape != b.shape ||
      static_cast<size_t>(shape.x * shape.y) != b.vector.size())
  {
    hmap::log::warn(
        loc,
        "Array shape mismatch: expected shape ({}, {}), actual is ({}, {}) "
        "[size: {}]",
        shape.x,
        shape.y,
        b.shape.x,
        b.shape.y,
        b.vector.size());
    return false;
  }
  return true;
}

/**
 * @brief Validates that a Texture is initialized, non-empty, and has valid
 * channel buffers.
 *
 * @param  tex          Texture to check.
 * @param  min_channels Minimum required number of channels.
 * @param  loc          Source location of the caller.
 * @return              true if the Texture is non-empty and channels are valid,
 * false otherwise.
 */
[[nodiscard]] inline bool validate_non_empty(
    const Texture              &tex,
    int                         min_channels = 1,
    const std::source_location &loc = std::source_location::current())
{
  if (tex.shape.x <= 0 || tex.shape.y <= 0 || tex.channels.empty())
  {
    hmap::log::warn(
        loc,
        "Texture is empty or uninitialized (shape: {}x{}, channels: {})",
        tex.shape.x,
        tex.shape.y,
        tex.channels.size());
    return false;
  }
  if (static_cast<int>(tex.channels.size()) < min_channels)
  {
    hmap::log::warn(loc,
                    "Texture has {} channels, but at least {} are required",
                    tex.channels.size(),
                    min_channels);
    return false;
  }
  for (size_t k = 0; k < tex.channels.size(); ++k)
  {
    if (!validate_non_empty(tex.channels[k], loc)) return false;
    if (tex.channels[k].shape != tex.shape)
    {
      hmap::log::warn(loc,
                      "Texture channel {} shape mismatch: channel is ({}, {}), "
                      "texture is ({}, {})",
                      k,
                      tex.channels[k].shape.x,
                      tex.channels[k].shape.y,
                      tex.shape.x,
                      tex.shape.y);
      return false;
    }
  }
  return true;
}

/**
 * @brief Validates exact channel count of a Texture.
 *
 * @param  tex               Texture to check.
 * @param  expected_channels Exact number of expected channels.
 * @param  loc               Source location of the caller.
 * @return                   true if channel count matches, false otherwise.
 */
[[nodiscard]] inline bool validate_channels(
    const Texture              &tex,
    int                         expected_channels,
    const std::source_location &loc = std::source_location::current())
{
  if (static_cast<int>(tex.channels.size()) != expected_channels)
  {
    hmap::log::warn(loc,
                    "Texture has {} channels, but exactly {} are required",
                    tex.channels.size(),
                    expected_channels);
    return false;
  }
  return true;
}

/**
 * @brief Validates that two Textures have matching 2D shapes.
 *
 * @param  a   First texture.
 * @param  b   Second texture.
 * @param  loc Source location of the caller.
 * @return     true if shapes match, false otherwise.
 */
[[nodiscard]] inline bool validate_same_shape(
    const Texture              &a,
    const Texture              &b,
    const std::source_location &loc = std::source_location::current())
{
  if (a.shape != b.shape)
  {
    hmap::log::warn(loc,
                    "Texture shape mismatch: lhs is ({}, {}), rhs is ({}, {})",
                    a.shape.x,
                    a.shape.y,
                    b.shape.x,
                    b.shape.y);
    return false;
  }
  return true;
}

/**
 * @brief Validates that a Texture and an Array have matching 2D shapes.
 *
 * @param  tex Texture to check.
 * @param  arr Array to check.
 * @param  loc Source location of the caller.
 * @return     true if shapes match, false otherwise.
 */
[[nodiscard]] inline bool validate_same_shape(
    const Texture              &tex,
    const Array                &arr,
    const std::source_location &loc = std::source_location::current())
{
  if (tex.shape != arr.shape)
  {
    hmap::log::warn(
        loc,
        "Texture/Array shape mismatch: texture is ({}, {}), array is ({}, {})",
        tex.shape.x,
        tex.shape.y,
        arr.shape.x,
        arr.shape.y);
    return false;
  }
  return true;
}

/**
 * @brief Validates that a slice extent {i1, i2, j1, j2} is valid and within
 * array boundaries.
 *
 * @param  array Target array.
 * @param  idx   Slice extents: {i1, i2, j1, j2}.
 * @param  loc   Source location of the caller.
 * @return       true if slice is valid and non-empty, false otherwise.
 */
[[nodiscard]] inline bool validate_slice(
    const Array                &array,
    glm::ivec4                  idx,
    const std::source_location &loc = std::source_location::current())
{
  if (idx.x < 0 || idx.y > array.shape.x || idx.x >= idx.y || idx.z < 0 ||
      idx.w > array.shape.y || idx.z >= idx.w)
  {
    hmap::log::warn(loc,
                    "Invalid slice [{}, {}, {}, {}] for array shape ({}, {})",
                    idx.x,
                    idx.y,
                    idx.z,
                    idx.w,
                    array.shape.x,
                    array.shape.y);
    return false;
  }
  return true;
}

/**
 * @brief Validates that a slice extent matches the dimensions of a source
 * array.
 *
 * @param  dest Destination array.
 * @param  idx  Slice extents {i1, i2, j1, j2}.
 * @param  src  Source array to assign into the slice.
 * @param  loc  Source location of the caller.
 * @return      true if slice is valid and matches src.shape, false otherwise.
 */
[[nodiscard]] inline bool validate_slice_for_array(
    const Array                &dest,
    glm::ivec4                  idx,
    const Array                &src,
    const std::source_location &loc = std::source_location::current())
{
  if (!validate_slice(dest, idx, loc)) return false;

  const int slice_w = idx.y - idx.x;
  const int slice_h = idx.w - idx.z;
  if (src.shape.x != slice_w || src.shape.y != slice_h)
  {
    hmap::log::warn(
        loc,
        "Slice dimensions ({}, {}) do not match source array shape ({}, {})",
        slice_w,
        slice_h,
        src.shape.x,
        src.shape.y);
    return false;
  }
  return true;
}

/**
 * @brief Validates that a divisor value is not zero.
 *
 * @param  value Divisor value.
 * @param  loc   Source location of the caller.
 * @return       true if value != 0, false otherwise.
 */
[[nodiscard]] inline bool validate_not_zero(
    float                       value,
    const std::source_location &loc = std::source_location::current())
{
  if (value == 0.f)
  {
    hmap::log::warn(loc, "Division by zero encountered");
    return false;
  }
  return true;
}

/**
 * @brief Validates that a 2D range vector {low, high} lies within [min_val,
 * max_val] with low <= high.
 *
 * @param  range      Range vector {low, high}.
 * @param  min_val    Minimum allowed value.
 * @param  max_val    Maximum allowed value.
 * @param  param_name Name or description of the range parameter for logging.
 * @param  loc        Source location of the caller.
 * @return            true if min_val <= range.x <= range.y <= max_val, false
 *                    otherwise.
 */
[[nodiscard]] inline bool validate_parameter_range(
    glm::vec2                   range,
    float                       min_val,
    float                       max_val,
    std::string_view            param_name = "Range",
    const std::source_location &loc = std::source_location::current())
{
  if (range.x < min_val || range.y > max_val || range.x > range.y)
  {
    hmap::log::warn(loc,
                    "{} [{}, {}] is invalid: expected values within [{}, {}] "
                    "with low <= high",
                    param_name,
                    range.x,
                    range.y,
                    min_val,
                    max_val);
    return false;
  }
  return true;
}

/**
 * @brief Validates that all elements of an array are finite (not NaN or Inf).
 *
 * @param  array Array to check.
 * @param  loc   Source location of the caller.
 * @return       true if all values are finite, false otherwise.
 */
[[nodiscard]] inline bool validate_finite(
    const Array                &array,
    const std::source_location &loc = std::source_location::current())
{
  if (!validate_non_empty(array, loc)) return false;

  for (size_t k = 0; k < array.vector.size(); ++k)
  {
    float    val = array.vector[k];
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    if ((bits & 0x7F800000u) == 0x7F800000u)
    {
      hmap::log::warn(loc,
                      "Non-finite value ({}) encountered at linear index {}",
                      val,
                      k);
      return false;
    }
  }
  return true;
}

/**
 * @brief Validates that a colormap ID corresponds to a valid Cmap enum value.
 *
 * @param  cmap Colormap ID.
 * @param  loc  Source location of the caller.
 * @return      true if cmap is a valid Cmap enum value, false otherwise.
 */
[[nodiscard]] inline bool validate_cmap(
    int                         cmap,
    const std::source_location &loc = std::source_location::current())
{
  if (cmap < static_cast<int>(Cmap::BONE) ||
      cmap > static_cast<int>(Cmap::WHITE_UNIFORM))
  {
    hmap::log::warn(loc, "Invalid colormap ID ({})", cmap);
    return false;
  }
  return true;
}

} // namespace hmap
