/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "highmap/array.hpp"
#include "highmap/range.hpp"

using namespace hmap;

// With FMA contraction (Apple Silicon) x * scale + offset left a tiny negative
// at the minimum; pow(remap(a), p) then produced NaN, which spread through
// hydraulic_stream_log into whole terrain regions.
TEST(Remap, StaysWithinTargetRange)
{
  std::mt19937                          gen(3);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);

  for (int rep = 0; rep < 200; ++rep)
  {
    Array       a(glm::ivec2(64, 48));
    const float shift = 1000.f * dis(gen);
    const float scale = std::pow(10.f, 3.f * dis(gen));
    for (float &v : a.vector) v = shift + scale * dis(gen);

    remap(a, 0.f, 1.f);
    EXPECT_EQ(a.min(), 0.f);
    EXPECT_LE(a.max(), 1.f);
    for (float v : a.vector) ASSERT_TRUE(std::isfinite(std::pow(v, 0.8f)));
  }
}
