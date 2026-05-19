/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <cstddef>
#include <cstdint>

namespace hmap
{
/**
 * @brief Generates a fast deterministic uniform float in [0,1) using a 32-bit
 * hash.
 *
 * @param  seed Base seed value.
 * @param  k    Sample index.
 * @return      Pseudo-random float in [0,1).
 * @note Faster but lower quality than splitmix64_to_unit_float().
 */
float fast_hash32_to_unit_float(unsigned int seed, size_t k);

/**
 * @brief Generates a deterministic uniform float in [0,1) using a
 * SplitMix64-style hash.
 *
 * @param  seed Base seed value.
 * @param  k    Sample index.
 * @return      Pseudo-random float in [0,1).
 */
float splitmix64_to_unit_float(unsigned int seed, size_t k);

// === PdfSampler class ===

/**
 * @brief Samples indices from a discrete probability distribution.
 */
class PdfSampler
{
public:
  /**
   * @brief Builds the sampler from a PDF and seed.
   * @param pdf  Probability weights.
   * @param seed Random generator seed.
   */
  PdfSampler(const std::vector<float> &pdf, uint32_t seed);

  /**
   * @brief Samples a float value in [0, 1[.
   * @return Sampled float.
   */
  float sample();

  /**
   * @brief Samples multiple float values.
   * @param  nb_samples Number of samples.
   * @return            Vector of sampled values.
   */
  std::vector<float> sample(size_t nb_samples);

private:
  std::vector<float>                    cdf;
  std::mt19937                          generator;
  std::uniform_real_distribution<float> distribution{0.f, 1.f};
};

} // namespace hmap