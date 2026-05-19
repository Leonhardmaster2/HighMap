/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t

#include <algorithm> // for fill_n, max, count, minmax_element, nth_element
#include <numeric>   // for iota
#include <random>    // for mt19937, uniform_int_distribution, uniform_real...
#include <sstream>   // for basic_ostream, operator<<, basic_ostringstream
#include <stdexcept> // for invalid_argument
#include <string>    // for char_traits, allocator, operator<<, string
#include <vector>    // for vector

namespace hmap
{

std::vector<size_t> argsort(const std::vector<float> &v)
{
  // https://stackoverflow.com/questions/1577475
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(),
                   idx.end(),
                   [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return idx;
}

float compute_median(std::vector<float> values)
{
  size_t n = values.size();
  if (n == 0) return 0.f; // or handle error

  size_t mid = n / 2;

  std::nth_element(values.begin(), values.begin() + mid, values.end());
  float median = values[mid];

  if (n % 2 == 0)
  {
    // Need the lower middle value as well
    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    median = 0.5f * (median + values[mid - 1]);
  }

  return median;
}

std::vector<size_t> find_sign_changes(const std::vector<float> &data)
{
  std::vector<size_t> indices;

  if (data.size() < 2) return indices;

  auto sign = [](float v) -> int
  {
    if (v > 0.f) return 1;
    if (v < 0.f) return -1;
    return 0;
  };

  int prev_sign = 0;

  for (size_t i = 0; i < data.size(); ++i)
  {
    int s = sign(data[i]);

    if (s == 0) continue; // skip zeros

    if (prev_sign != 0 && s != prev_sign) indices.push_back(i);

    prev_sign = s;
  }

  return indices;
}

std::vector<float> generate_random_vector(size_t   size,
                                          float    min_value,
                                          float    max_value,
                                          uint32_t seed)
{
  std::vector<float> result(size);

  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> dist(min_value, max_value);

  for (auto &value : result)
  {
    value = dist(rng);
  }

  return result;
}

std::vector<int> generate_random_vector(size_t   size,
                                        int      min_value,
                                        int      max_value,
                                        uint32_t seed)
{
  std::vector<int> result(size);

  std::mt19937                       rng(seed); // deterministic engine
  std::uniform_int_distribution<int> dist(min_value, max_value);

  for (auto &value : result)
  {
    value = dist(rng);
  }

  return result;
}

std::vector<int> generate_unique_random_vector(size_t   size,
                                               int      min_value,
                                               int      max_value,
                                               uint32_t seed)
{
  if (min_value > max_value)
    throw std::invalid_argument("min_value must be <= max_value");

  size_t range_size = static_cast<size_t>(max_value - min_value + 1);

  if (size > range_size)
    throw std::invalid_argument(
        "size exceeds number of unique values in range");

  // fill full range
  std::vector<int> values(range_size);
  for (size_t i = 0; i < range_size; ++i)
  {
    values[i] = min_value + static_cast<int>(i);
  }

  // shuffle
  std::mt19937 rng(seed);
  std::shuffle(values.begin(), values.end(), rng);

  // keep only what we need
  values.resize(size);

  return values;
}

std::string make_histogram(const std::vector<float> &values,
                           int                       bin_count,
                           int                       hist_height)
{
  std::ostringstream out;

  if (values.empty() || bin_count <= 0 || hist_height <= 0)
  {
    return "Invalid input.\n";
  }

  // Compute min and max
  auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
  float min_val = *min_it;
  float max_val = *max_it;

  // Count occurrences of exact min and max
  int count_min = std::count(values.begin(), values.end(), min_val);
  int count_max = std::count(values.begin(), values.end(), max_val);

  // Edge case: all values equal
  if (min_val == max_val)
  {
    out << "All values are equal to " << min_val << ".\n";
    out << "Count = " << values.size() << "\n\n";
    return out.str();
  }

  // Prepare bins
  std::vector<int> bins(bin_count, 0);
  float            range = max_val - min_val;
  float            inv_range = 1.0f / range;

  // Fill bins
  for (float v : values)
  {
    int idx = int((v - min_val) * inv_range * bin_count);
    if (idx == bin_count) idx = bin_count - 1; // clamp
    bins[idx]++;
  }

  // Find maximum bin height for scaling
  int   max_bin = *std::max_element(bins.begin(), bins.end());
  float scale = float(hist_height) / float(max_bin);

  // Build histogram (top to bottom)
  for (int row = hist_height; row > 0; --row)
  {
    for (int b = 0; b < bin_count; ++b)
    {
      float scaled_height = bins[b] * scale;
      out << (scaled_height >= row ? "█" : " ");
    }
    out << "\n";
  }

  // Add horizontal axis
  out << std::string(bin_count, '-') << "\n";

  // Add stats
  out << "Min value:  " << min_val << " (count = " << count_min << ")\n";
  out << "Max value:  " << max_val << " (count = " << count_max << ")\n";

  return out.str();
}

std::vector<float> moving_average(const std::vector<float> &input, int radius)
{
  if (input.empty() || radius <= 0) return input;

  std::vector<float> output(input.size(), 0.0f);

  for (size_t i = 0; i < input.size(); ++i)
  {
    int start = std::max<int>(0, i - radius);
    int end = std::min<int>(input.size() - 1, i + radius);

    float sum = 0.0f;
    int   count = 0;

    for (int j = start; j <= end; ++j)
    {
      sum += input[j];
      count++;
    }

    output[i] = sum / static_cast<float>(count);
  }

  return output;
}

size_t upperbound_right(const std::vector<float> &v, float value)
{
  size_t idx = 0;
  for (size_t k = v.size() - 1; k > 0; k--)
    if (value > v[k])
    {
      idx = k;
      break;
    }
  return idx;
}

void reindex_vector(std::vector<int> &v, std::vector<size_t> &idx);
void reindex_vector(std::vector<float> &v, std::vector<size_t> &idx);

std::vector<float> remap(const std::vector<float> &data,
                         float                     new_min,
                         float                     new_max)
{
  if (data.empty()) return {};

  auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
  float old_min = *min_it;
  float old_max = *max_it;

  std::vector<float> result(data.size());

  // Avoid division by zero
  if (old_max == old_min)
  {
    std::fill(result.begin(), result.end(), new_min);
    return result;
  }

  float scale = (new_max - new_min) / (old_max - old_min);

  for (size_t i = 0; i < data.size(); ++i)
  {
    result[i] = new_min + (data[i] - old_min) * scale;
  }

  return result;
}

void vector_unique_values(std::vector<float> &v)
{
  auto last = std::unique(v.begin(), v.end());
  v.erase(last, v.end());
  std::sort(v.begin(), v.end());
  last = std::unique(v.begin(), v.end());
  v.erase(last, v.end());
}

} // namespace hmap
