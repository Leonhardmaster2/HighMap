/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file statistics.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @copyright Copyright (c) 2026
 */
#pragma once

#include "highmap/array.hpp"

namespace hmap
{

/**
 * @brief Estimate the isotropic dominant length scale of a 2D heightmap.
 *
 * Computes the e-folding lag of the autocorrelation function (averaged over i
 * and j directions) via binary search. Returns `max(ni, nj)` for flat arrays.
 *
 * @param  array            Input 2D heightmap, accessed via array(i, j).
 * @param  max_lag_fraction Search range as a fraction of the shorter dimension.
 * @return                  Length scale in grid cells.
 *
 * **Example**
 * @include ex_cautocorr_length_scale.cpp
 */
float autocorr_length_scale(const Array &array, float max_lag_fraction = 0.4f);

/**
 * @brief Estimate axis-aligned dominant length scales of a 2D heightmap.
 *
 * Independently computes the e-folding autocorrelation lag along the i and j
 * grid axes. Use when the field is anisotropic but aligned with the grid.
 * Returns `{ni, nj}` for flat arrays.
 *
 * @param  array            Input 2D heightmap, accessed via array(i, j).
 * @param  max_lag_fraction Search range as a fraction of each axis length.
 * @return                  {scale_i, scale_j} length scales in grid cells.
 */
glm::vec2 autocorr_length_scale_axial(const Array &array,
                                      float        max_lag_fraction = 0.4f);

/**
 * @brief Compute the population variance of a 2D array.
 *
 * @param  array  Input 2D array, accessed via array(i, j).
 * @param  p_mean Pointer to a precomputed mean (avoids recomputation); if null,
 *                the mean is computed internally.
 * @return        Population variance.
 */
float variance(const Array &array, float *p_mean = nullptr);

} // namespace hmap