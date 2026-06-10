/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file distance_functions.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @copyright Copyright (c) 2023
 */
#pragma once
#include <functional>

namespace hmap
{

/**
 * @brief Distance function type.
 */
enum DistanceFunction : int
{
  CHEBYSHEV, ///< Chebyshev
  EUCLIDIAN, ///< Euclidian
  EUCLISHEV, ///< Euclidian and Chebyshev mix
  MANHATTAN, ///< Manhattan
};

/**
 * @brief Axis used for the distance function.
 */
enum DistanceFunctionAxis : int
{
  DFA_X,  ///< X-only
  DFA_Y,  ///< Y-only
  DFA_XY, ///< X and Y
};

/**
 * @brief Return the requested distance function.
 * @param  dist_fct Distance function type.
 * @return          Distance function.
 */
std::function<float(float, float)> get_distance_function(
    DistanceFunction     dist_fct,
    DistanceFunctionAxis dfa = DistanceFunctionAxis::DFA_XY);

} // namespace hmap