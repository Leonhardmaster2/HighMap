/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file algebra.hpp
 * @author  Otto Link (otto.link.bv@gmail.com)
 * @brief Header file defining basic vector and matrix manipulation classes.
 * @version 0.1
 * @date 2023-08-01
 *
 * @copyright Copyright (c) 2023
 */
#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace hmap
{

void to_csv(const std::vector<glm::vec3> &xyz, const std::string &fname);

// --- For unordered_map

// for glm::ivec2 map
struct IVec2Hash
{
  size_t operator()(const glm::ivec2 &v) const noexcept
  {
    return std::hash<int>()(v.x) ^ (std::hash<int>()(v.y) << 1);
  }
};

struct IVec2Eq
{
  bool operator()(const glm::ivec2 &a, const glm::ivec2 &b) const noexcept
  {
    return a == b;
  }
};

// for glm::ivec4 map
struct IVec4Hash
{
  std::size_t operator()(const glm::ivec4 &v) const noexcept
  {
    std::size_t h = 0;
    h ^= std::hash<int>{}(v.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(v.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(v.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(v.w) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct IVec4Eq
{
  bool operator()(const glm::ivec4 &a, const glm::ivec4 &b) const noexcept
  {
    return a == b;
  }
};

inline glm::vec4 adjust(const glm::vec4 &v,
                        float            dx,
                        float            dy,
                        float            dz,
                        float            dw)
{
  return glm::vec4{v.x + dx, v.y + dy, v.z + dz, v.w + dw};
}

inline glm::vec4 adjust(const glm::vec4 &v, float dr)
{
  return glm::vec4{v.x - dr, v.y + dr, v.z - dr, v.w + dr};
}

/**
 * @brief Mat class for basic manipulation of 2D matrices.
 *
 * This class provides basic operations for 2D matrices, such as element access
 * and initialization. It stores the matrix elements in a 1D vector and provides
 * a convenient interface for accessing elements using 2D indices.
 *
 * @tparam T Data type for the matrix elements (e.g., int, float, double).
 */
template <typename T> struct Mat
{
  std::vector<T> vector; /**< @brief 1D vector storing matrix elements in
                            row-major order. */
  glm::ivec2 shape;      /**< @brief Dimensions of the matrix (rows x columns).
                          */

  Mat() = default;

  /**
   * @brief Constructor to initialize a matrix with a given shape.
   *
   * Allocates memory for the matrix elements based on the specified
   * shape. The matrix is initialized with the default value of the type
   * T.
   *
   * @param shape A glm::ivec2 representing the number of rows and columns
   *              in the matrix.
   */
  Mat(glm::ivec2 shape) : shape(shape)
  {
    this->vector.resize(shape.x * shape.y);
  }

  /**
   * @brief Constructor to initialize a matrix with a given shape and
   * value.
   *
   * Allocates memory for the matrix elements and initializes all elements
   * with the provided value.
   *
   * @param shape A glm::ivec2 representing the number of rows and columns
   *              in the matrix.
   * @param value Initial value assigned to all matrix elements.
   */
  Mat(glm::ivec2 shape, T value) : shape(shape)
  {
    this->vector.resize(shape.x * shape.y);
    std::fill(this->vector.begin(), this->vector.end(), value);
  }

  /**
   * @brief Access operator to get a reference to the element at (i, j).
   *
   * Provides non-const access to the matrix element at the specified row
   * and column.
   *
   * @param  i Row index (0-based).
   * @param  j Column index (0-based).
   * @return   A reference to the element at the specified position.
   */
  T &operator()(int i, int j)
  {
    return this->vector[j * this->shape.x + i];
  }

  /**
   * @brief Const access operator to get the value of the element at (i,
   * j).
   *
   * Provides const access to the matrix element at the specified row and
   * column.
   *
   * @param  i Row index (0-based).
   * @param  j Column index (0-based).
   * @return   A const reference to the element at the specified position.
   */
  const T &operator()(int i, int j) const
  {
    return this->vector[j * this->shape.x + i];
  }

  T &operator()(glm::ivec2 ij)
  {
    return this->vector[ij.y * this->shape.x + ij.x];
  }

  const T &operator()(glm::ivec2 ij) const
  {
    return this->vector[ij.y * this->shape.x + ij.x];
  }
};

} // namespace hmap
