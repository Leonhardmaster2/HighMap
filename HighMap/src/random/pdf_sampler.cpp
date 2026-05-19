/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <stddef.h> // for size_t

#include <algorithm> // for lower_bound, max, fill_n
#include <cstdint>   // for uint32_t
#include <iterator>  // for distance
#include <numeric>   // for accumulate
#include <random>    // for uniform_real_distribution
#include <vector>    // for vector

#include "highmap/random.hpp" // for PdfSampler

namespace hmap
{

PdfSampler::PdfSampler(const std::vector<float> &pdf, uint32_t seed)
    : generator(seed)
{
  float sum = std::accumulate(pdf.begin(), pdf.end(), 0.f);

  cdf.resize(pdf.size());

  float cumulative = 0.f;

  for (size_t i = 0; i < pdf.size(); ++i)
  {
    cumulative += pdf[i] / sum;
    cdf[i] = cumulative;
  }
}

float PdfSampler::sample()
{
  float u = distribution(generator);

  auto it = std::lower_bound(cdf.begin(), cdf.end(), u);

  size_t index = std::distance(cdf.begin(), it);

  // Uniform random offset inside the bin
  float local = distribution(generator);

  return static_cast<float>((static_cast<float>(index) + local) /
                            static_cast<float>(cdf.size()));
}

std::vector<float> PdfSampler::sample(size_t nb_samples)
{
  std::vector<float> samples;

  samples.reserve(nb_samples);

  for (size_t i = 0; i < nb_samples; ++i)
    samples.push_back(sample());

  return samples;
}

} // namespace hmap
