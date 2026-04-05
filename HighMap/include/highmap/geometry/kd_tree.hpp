/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file kd_tree.hpp
 * @brief KD-tree utilities for 2D point queries using nanoflann.
 */
#pragma once
#include <vector>

#include <glm/glm.hpp>

#include <nanoflann.hpp>

namespace hmap
{

/**
 * @brief Adaptor exposing a 2D point cloud to nanoflann.
 *
 * This class provides the minimal interface required by nanoflann to access a
 * set of 2D points stored as separate x and y coordinate arrays.
 *
 * The data is not owned by the adaptor; it references external vectors.
 */
struct NanoflannPointCloudAdaptor
{
  /** Reference to x coordinates */
  const std::vector<float> &x;

  /** Reference to y coordinates */
  const std::vector<float> &y;

  /**
   * @brief Construct the adaptor from coordinate arrays.
   * @param x_ Vector of x coordinates
   * @param y_ Vector of y coordinates
   */
  NanoflannPointCloudAdaptor(const std::vector<float> &x_,
                             const std::vector<float> &y_);

  /**
   * @brief Get the number of points in the dataset.
   * @return Number of points
   */
  inline size_t kdtree_get_point_count() const;

  /**
   * @brief Get a coordinate component of a point.
   * @param  idx Point index
   * @param  dim Dimension (0 = x, 1 = y)
   * @return     Coordinate value
   */
  inline float kdtree_get_pt(const size_t idx, int dim) const;

  /**
   * @brief Optional bounding-box computation (unused).
   * @return Always false (no bounding box provided)
   */
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const
  {
    return false;
  }
};

/**
 * @brief Alias for a 2D KD-tree using L2 (Euclidean) distance.
 */
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, NanoflannPointCloudAdaptor>,
    NanoflannPointCloudAdaptor,
    2>;

/**
 * @brief Context wrapper for KD-tree operations.
 *
 * This class encapsulates:
 * - input point data
 * - nanoflann adaptor
 * - KD-tree index
 *
 * It provides convenient methods for nearest-neighbor and radius-based queries.
 *
 * @note The input vectors must remain valid for the lifetime of this object.
 */
struct KDTreeContext
{
  /** Reference to x coordinates */
  const std::vector<float> &x;

  /** Reference to y coordinates */
  const std::vector<float> &y;

  /** Nanoflann adaptor */
  NanoflannPointCloudAdaptor adaptor;

  /** KD-tree index */
  KDTree index;

  /**
   * @brief Construct the KD-tree context and build the index.
   * @param x_ Vector of x coordinates
   * @param y_ Vector of y coordinates
   */
  KDTreeContext(const std::vector<float> &x_, const std::vector<float> &y_);

  /**
   * @brief Estimate the range of neighbor distances.
   *
   * Computes the minimum and maximum distance to the k-th nearest
   * neighbor across all points in the dataset.
   *
   * @param  k_neighbors Number of neighbors considered
   * @return             vec2(min_distance, max_distance)
   */
  glm::vec2 compte_neighbor_distance_range(size_t k_neighbors) const;

  /**
   * @brief Perform a k-nearest neighbor search.
   *
   * @param      x_query     Query x coordinate
   * @param      y_query     Query y coordinate
   * @param      k_neighbors Number of neighbors to retrieve
   * @param[out] indices     Output indices of neighbors
   * @param[out] distances   Output squared distances to neighbors
   */
  void neighbor_search(float                x_query,
                       float                y_query,
                       size_t               k_neighbors,
                       std::vector<size_t> &indices,
                       std::vector<float>  &distances) const;

  /**
   * @brief Perform a radius-based neighbor search.
   *
   * @param  x_query Query x coordinate
   * @param  y_query Query y coordinate
   * @param  radius  Search radius (Euclidean distance)
   * @return         Vector of (index, squared distance) pairs
   *
   * @note Returned distances are squared distances.
   */
  std::vector<nanoflann::ResultItem<unsigned int, float>> radius_search(
      float x_query,
      float y_query,
      float radius) const;
};

} // namespace hmap