/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file local_metrics.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 */
#pragma once

#include "highmap/array.hpp"

namespace hmap
{

/**
 * @brief Return the local maxima based on a maximum filter with a square
 * kernel.
 *
 * This function identifies the local maxima in the input array using a maximum
 * filter with a square kernel. The local maxima are determined based on the
 * footprint radius specified by `ir`. The result is an array where each value
 * represents the local maximum within the defined kernel size.
 *
 * @param  array Input array from which local maxima are to be extracted.
 * @param  ir    Square kernel footprint radius. The size of the kernel used to
 *               determine the local maxima.
 * @return       Array Resulting array containing the local maxima.
 *
 * **Example**
 * @include ex_local_max.cpp
 *
 * **Result**
 * @image html ex_local_max.png
 *
 * @see          {@link local_max_disk}, {@link local_min}
 */
Array local_max(const Array &array, int ir);

/**
 * @brief Computes the local median deviation of a 2D array.
 *
 * This function calculates the absolute difference between the local mean and a
 * pseudo-local median of each element in the input array. The local
 * neighborhood is defined by a square window with radius `ir`.
 *
 * @param  array The input 2D array of values (e.g., a heightmap).
 * @param  ir    The radius of the square neighborhood window used for computing
 *               local statistics. The window size is (2*ir + 1) x (2*ir + 1).
 *
 * @return       A new array of the same size as `array`, where each element
 *               represents the absolute deviation between the local mean and
 *               pseudo-local median in its neighborhood.
 *
 * @note The function uses a pseudo-median approximation. For exact median
 * computation, replace the `median_pseudo()` call with a proper median filter
 * implementation.
 *
 * **Example**
 * @include ex_local_median_deviation.cpp
 *
 * **Result**
 * @image html ex_local_median_deviation.png
 *
 * @see          local_mean(), median_pseudo()
 */
Array local_median_deviation(const Array &array, int ir);

/**
 * @brief Return the local minima based on a maximum filter with a square
 * kernel.
 *
 * This function identifies the local minima in the input array using a maximum
 * filter with a square kernel. The local minima are determined based on the
 * footprint radius specified by `ir`. The result is an array where each value
 * represents the local minimum within the defined kernel size.
 *
 * @param  array Input array from which local minima are to be extracted.
 * @param  ir    Square kernel footprint radius. The size of the kernel used to
 *               determine the local minima.
 * @return       Array Resulting array containing the local minima.
 *
 * **Example**
 * @include ex_local_max.cpp
 *
 * **Result**
 * @image html ex_local_max.png
 *
 * @see          {@link local_min}
 */
Array local_min(const Array &array, int ir);

/**
 * @brief Return the local mean based on a mean filter with a square kernel.
 *
 * This function calculates the local mean of the input array using a mean
 * filter with a square kernel. The local mean is determined by averaging values
 * within a square neighborhood defined by the footprint radius `ir`. The result
 * is an array where each value represents the mean of the surrounding values
 * within the kernel size.
 *
 * @param  array Input array from which the local mean is to be calculated.
 * @param  ir    Square kernel footprint radius. The size of the kernel used to
 *               compute the local mean.
 * @return       Array Resulting array containing the local means.
 *
 * **Example**
 * @include ex_local_mean.cpp
 *
 * **Result**
 * @image html ex_local_mean0.png
 * @image html ex_local_mean1.png
 *
 * @see          {@link local_max}, {@link local_min}
 */
Array local_mean(const Array &array, int ir);

/**
 * @brief Calculates the relative elevation within a specified radius, helping
 * to identify local highs and lows.
 *
 * Relative elevation analysis determines how high or low a point is relative to
 * its surroundings, which can be critical in hydrological modeling and
 * geomorphology.
 *
 * **Usage**:
 * - Use this function to detect local depressions or peaks in the terrain,
 * which could indicate potential water collection points or hilltops.
 * - Useful in flood risk assessment and landscape classification.
 *
 * @param  array The input array representing the terrain elevation data.
 * @param  ir    The radius (in pixels) within which to calculate the relative
 *               elevation for each point.
 * @return       Array An output array containing the relative elevation values,
 *               normalized between 0 and 1.
 *
 * **Example**
 * @include ex_relative_elevation.cpp
 *
 * **Result**
 * @image html ex_relative_elevation.png
 */
Array relative_elevation(const Array &array, int ir);

/**
 * @brief Computes the ruggedness of each element in the input array.
 *
 * The ruggedness is calculated as the square root of the sum of squared
 * differences between each element and its neighbors within a specified radius.
 *
 * @param  array The input array for which ruggedness is to be computed.
 * @param  ir    The radius within which neighbors are considered for ruggedness
 *               calculation.
 * @return       An array containing the ruggedness values for each element in
 *               the input array.
 *
 * @note https://xdem.readthedocs.io/en/latest/terrain.html
 *
 * **Example**
 * @include ex_ruggedness.cpp
 *
 * **Result**
 * @image html ex_ruggedness0.png
 * @image html ex_ruggedness1.png
 */
Array ruggedness(const Array &array, int ir);

/**
 * @brief Estimates the rugosity of a surface by analyzing the skewness of the
 * elevation data, which reflects surface roughness.
 *
 * Rugosity is a measure of terrain roughness, often used in ecological studies
 * and habitat mapping. Higher rugosity values indicate more rugged terrain,
 * which can affect species distribution and water flow.
 *
 * @param  z      The input array representing the heightmap data (elevation
 *                values).
 * @param  ir     The radius of the square kernel used for calculations,
 *                determining the scale of the analysis.
 * @param  convex Return the convex rugosity if true, and the concave ones if
 *                not.
 * @return        Array An output array containing the rugosity estimates, where
 *                higher values indicate rougher terrain.
 *
 * **Example**
 * @include ex_rugosity.cpp
 *
 * **Result**
 * @image html ex_rugosity0.png
 * @image html ex_rugosity1.png
 */
Array rugosity(const Array &z, int ir, bool convex = true);

} // namespace hmap

namespace hmap::gpu
{

/**
 * @brief Compute the average local aspect variance of an array.
 *
 * @param  array Input array.
 * @param  ir    Radius of the local window.
 * @return       Array containing the averaged local variance.
 *
 * **Example**
 * @include ex_local_metrics.cpp
 *
 * **Result**
 * @image html ex_local_metrics.png
 */
Array local_aspect_variance(const Array &array, int ir);

/**
 * @brief Compute the local maximum using a disk kernel.
 *
 * For each cell, returns the maximum value within a neighborhood of radius @p
 * ir defined by a disk-shaped kernel.
 *
 * @param  array Input array.
 * @param  ir    Radius of the disk kernel.
 * @return       Array of local maximum values.
 *
 *  **Example**
 * @include ex_local_metrics.cpp
 *
 * **Result**
 * @image html ex_local_metrics.png
 */
Array local_max(const Array &array, int ir);

/*! @brief See hmap::local_median_deviation */
Array local_median_deviation(const Array &array, int ir);

/**
 * @brief Compute the local minimum using a disk kernel.
 *
 * For each cell, returns the minimum value within a neighborhood of radius @p
 * ir defined by a disk-shaped kernel.
 *
 * @param  array Input array.
 * @param  ir    Radius of the disk kernel.
 * @return       Array of local minimum values.
 *
 *  **Example**
 * @include ex_local_metrics.cpp
 *
 * **Result**
 * @image html ex_local_metrics.png
 */
Array local_min(const Array &array, int ir);

/**
 * @brief Compute the local relief of an array.
 *
 * The local relief is defined as the difference between the maximum and minimum
 * values within a neighborhood of radius @p ir around each element. This
 * provides a measure of local variation in the array (e.g., terrain
 * ruggedness).
 *
 * @param  array Input array representing scalar values (e.g., elevation map).
 * @param  ir    Radius of the neighborhood (in pixels) used to compute local
 *               extrema.
 *
 * @return       Array An array of the same size as @p array, where each element
 *               contains the difference between the local maximum and minimum
 *               within the specified neighborhood.
 *
 * **Example**
 * @include ex_local_relief.cpp
 *
 * **Result**
 * @image html ex_local_relief.png
 */
Array local_relief(const Array &array, int ir);

/**
 * @brief Compute the average local variance of an array.
 *
 * Local variance is defined as the standard deviation within a window of radius
 * @p ir centered on each cell. The function computes this value for every cell,
 * then returns the average over the entire domain.
 *
 * @param  array Input array.
 * @param  ir    Radius of the local window.
 * @return       Array containing the averaged local variance.
 *
 * **Example**
 * @include ex_local_variance.cpp
 *
 * **Result**
 * @image html ex_local_variance.png
 */
Array local_variance(const Array &array, int ir);

/*! @brief See hmap::local_mean */
Array local_mean(const Array &array, int ir);

Array local_skewness(const Array &array, int ir);

/**
 * @brief Compute the local z-score of an array.
 *
 * For each cell, computes the normalized difference between its value and the
 * mean of a neighborhood of radius @p ir.
 *
 * @param  array Input array.
 * @param  ir    Radius of the local window.
 * @return       Array of local z-score values.
 *
 * **Example**
 * @include ex_local_z_score.cpp
 *
 * **Result**
 * @image html ex_local_z_score.png
 */
Array local_z_score(const Array &array, int ir);

/**
 * @brief Compute the Topographic Position Index (TPI).
 *
 * TPI measures the elevation difference between each cell and the mean
 * elevation within a neighborhood of radius @p ir.
 *
 * @param  array Input elevation array.
 * @param  ir    Radius of the neighborhood.
 * @return       Array of TPI values.
 *
 * **Example**
 * @include ex_topographic_position_index.cpp
 *
 * **Result**
 * @image html ex_topographic_position_index.png
 */
Array topographic_position_index(const Array &array, int ir);

/*! @brief See hmap::relative_elevation */
Array relative_elevation(const Array &array, int ir);

/*! @brief See hmap::ruggedness */
Array ruggedness(const Array &array, int ir);

/*! @brief See hmap::rugosity */
Array rugosity(const Array &z, int ir, bool convex = true);

} // namespace hmap::gpu

// --- WRAPPER

namespace hmap::gpu
{

// clang-format off
enum LocalMetrics : int
{
   LM_LOCAL_ASPECT_VARIANCE,
   LM_LOCAL_MAX,
   LM_LOCAL_MEDIAN_DEVIATION,
   LM_LOCAL_MIN,
   LM_LOCAL_RELIEF,
   LM_LOCAL_VARIANCE,
   LM_LOCAL_MEAN,
   LM_LOCAL_SKEWNESS,
   LM_LOCAL_Z_SCORE,
   LM_TOPOGRAPHIC_POSITION_INDEX,
   LM_RELATIVE_ELEVATION,
   LM_RUGGEDNESS,
   LM_RUGOSITY_CONCAVE,
   LM_RUGOSITY_CONVEX,
};
// clang-format on

Array local_metrics(const Array &array, int ir, LocalMetrics metric);

} // namespace hmap::gpu