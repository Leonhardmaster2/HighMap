/**
 * @file vector_utils.hpp
 * @copyright Copyright (c) 2023 Otto Link. Distributed under the terms of the
 * GNU General Public License. The full license is in the file LICENSE,
 * distributed with this software.
 */
#pragma once
#include <algorithm>
#include <random>
#include <string>
#include <vector>

typedef unsigned int uint;

namespace hmap
{

// ============================================================
// Indexing / Sorting
// ============================================================

/**
 * @brief Returns the indices that would sort the vector.
 *
 * @param  v Input vector.
 * @return   Indices representing the sorted order of v.
 *
 * @note The original vector is not modified.
 */
std::vector<size_t> argsort(const std::vector<float> &v);

/**
 * @brief Reorders a vector using a given index mapping.
 *
 * @tparam T Type of the vector elements.
 * @param v   Vector to reorder.
 * @param idx Index mapping (typically from argsort).
 *
 * @note idx[k] indicates the source index of the k-th element in the result.
 */
template <typename T>
void reindex_vector(std::vector<T> &v, std::vector<size_t> &idx)
{
  std::vector<T> v_new(v.size());
  for (uint k = 0; k < v.size(); k++)
    v_new[k] = v[idx[k]];
  v = v_new;
}

/**
 * @brief Returns the index of the first element greater than a given value.
 *
 * Equivalent to std::upper_bound but returns the index instead of an iterator.
 *
 * @param  v     Sorted input vector.
 * @param  value Value to compare.
 * @return       Index of the first element greater than value.
 */
size_t upperbound_right(const std::vector<float> &v, float value);

// ============================================================
// Random / Shuffling
// ============================================================

/**
 * @brief Generate a vector of random floating-point values within a given
 * range.
 *
 * Uses a deterministic random number generator initialized with the provided
 * seed. The same seed will always produce the same sequence of values.
 *
 * @param  size      Number of elements in the vector.
 * @param  min_value Lower bound of the range (inclusive).
 * @param  max_value Upper bound of the range (inclusive).
 * @param  seed      Seed used to initialize the random number generator.
 * @return           std::vector<float> Vector filled with random values in
 *                   [min_value, max_value].
 */
std::vector<float> generate_random_vector(size_t   size,
                                          float    min_value,
                                          float    max_value,
                                          uint32_t seed);

/**
 * @brief Generate a vector of random integers within a given range.
 *
 * Uses a deterministic random number generator initialized with the provided
 * seed. The same seed will always produce the same sequence of values.
 *
 * @param  size      Number of elements in the vector.
 * @param  min_value Lower bound of the range (inclusive).
 * @param  max_value Upper bound of the range (inclusive).
 * @param  seed      Seed used to initialize the random number generator.
 * @return           std::vector<int> Vector filled with random integers in
 *                   [min_value, max_value].
 */
std::vector<int> generate_random_vector(size_t   size,
                                        int      min_value,
                                        int      max_value,
                                        uint32_t seed);

/**
 * @brief Generate a vector of unique random integers within a given range.
 *
 * Values are sampled without replacement using a deterministic shuffle based on
 * the provided seed.
 *
 * @param  size      Number of elements to generate.
 * @param  min_value Lower bound (inclusive).
 * @param  max_value Upper bound (inclusive).
 * @param  seed      Seed for reproducible randomness.
 * @return           std::vector<int> Vector of unique integers in [min_value,
 *                   max_value].
 *
 * @throws std::invalid_argumentIfsizeexceedsavailableuniquevalues.
 */
std::vector<int> generate_unique_random_vector(size_t   size,
                                               int      min_value,
                                               int      max_value,
                                               uint32_t seed);

/**
 * @brief Shuffles a vector in-place using a deterministic seed.
 *
 * @tparam T Type of the vector elements.
 * @param values Vector to shuffle.
 * @param seed   Seed for the random generator.
 */
template <typename T>
void shuffle_vector(std::vector<T> &values, std::uint32_t seed)
{
  std::mt19937 rng(seed);
  std::shuffle(values.begin(), values.end(), rng);
}

/**
 * @brief Returns a shuffled copy of a vector.
 *
 * @tparam T Type of the vector elements.
 * @param  values Input vector.
 * @param  seed   Seed for the random generator.
 * @return        Shuffled copy of the input vector.
 */
template <typename T>
std::vector<T> shuffled_vector(const std::vector<T> &values, std::uint32_t seed)
{
  std::vector<T> result = values;
  shuffle_vector(result, seed);
  return result;
}

// ============================================================
// Value Processing
// ============================================================

/**
 * @brief Returns indices where a sign change occurs in the input vector.
 *
 * A sign change is detected between i-1 and i when values have opposite signs.
 *
 * @param  data Input vector.
 * @return      Indices i where a sign change occurs between data[i-1] and
 *              data[i].
 */
std::vector<size_t> find_sign_changes(const std::vector<float> &data);

/**
 * @brief Computes the median value of a set of floats.
 * @param  values Input values (copied and partially reordered).
 * @return        Median of the input values.
 */
float compute_median(std::vector<float> values);

/**
 * @brief Smooths a vector using a centered moving average.
 *
 * Each value is replaced by the average of its neighbors within a given radius.
 *
 * @param  input  Input vector.
 * @param  radius Number of elements on each side to include in the average.
 * @return        Smoothed vector of the same size.
 *
 * @note Borders are handled by clamping (no wrapping).
 */
std::vector<float> moving_average(const std::vector<float> &input, int radius);

/**
 * @brief Removes duplicate values from a vector.
 *
 * @param v Input vector.
 *
 * @note The vector is modified in-place.
 * @note The resulting order is not guaranteed to be preserved.
 */
void vector_unique_values(std::vector<float> &v);

/**
 * @brief Remaps values of a vector to a given range [new_min, new_max].
 *
 * @param  data    Input vector.
 * @param  new_min Target minimum value.
 * @param  new_max Target maximum value.
 * @return         Remapped vector.
 *
 * @note If input values are constant, all outputs are set to new_min.
 */
std::vector<float> remap(const std::vector<float> &data,
                         float                     new_min = 0.f,
                         float                     new_max = 1.f);

// ============================================================
// Analysis / Visualization
// ============================================================

/**
 * @brief Generates an ASCII histogram representation of values.
 *
 * @param  values      Input data.
 * @param  bin_count   Number of bins.
 * @param  hist_height Height of the histogram in characters.
 * @return             String representing the histogram.
 */
std::string make_histogram(const std::vector<float> &values,
                           int                       bin_count,
                           int                       hist_height);

} // namespace hmap