/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

#include <cmath>
#include <future>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "highmap/interpolate/interpolate2d.hpp"

using namespace hmap;

// VirtualArray tile workers call natural-neighbour interpolation
// concurrently; nn-c/triangle.c keeps global state, so without serialization
// overlapping calls returned NaN tiles (seen in va::hydraulic_saleve).
TEST(Interpolate2d, NaturalNeighborIsThreadSafe)
{
  std::mt19937                          gen(7);
  std::uniform_real_distribution<float> dis(0.f, 1.f);
  std::vector<float>                    x(600), y(600), v(600);
  for (size_t k = 0; k < x.size(); ++k)
  {
    x[k] = dis(gen);
    y[k] = dis(gen);
    v[k] = std::sin(6.f * x[k]) * std::cos(4.f * y[k]);
  }

  const glm::ivec2 shape{96, 96};
  auto             tile_bbox = [](int t)
  {
    const float x0 = 0.25f * float(t % 4);
    const float y0 = 0.25f * float(t / 4);
    return glm::vec4(x0, x0 + 0.25f, y0, y0 + 0.25f);
  };

  std::vector<Array> expected;
  for (int t = 0; t < 16; ++t)
    expected.push_back(interpolate2d(shape,
                                     x,
                                     y,
                                     v,
                                     InterpolationMethod2D::ITP2D_NNI,
                                     nullptr,
                                     nullptr,
                                     tile_bbox(t)));

  for (int rep = 0; rep < 5; ++rep)
  {
    std::vector<std::future<Array>> jobs;
    for (int t = 0; t < 16; ++t)
      jobs.push_back(std::async(std::launch::async,
                                [&, t]
                                {
                                  return interpolate2d(
                                      shape,
                                      x,
                                      y,
                                      v,
                                      InterpolationMethod2D::ITP2D_NNI,
                                      nullptr,
                                      nullptr,
                                      tile_bbox(t));
                                }));
    for (int t = 0; t < 16; ++t)
    {
      const Array result = jobs[t].get();
      ASSERT_EQ(result.vector.size(), expected[t].vector.size());
      for (size_t k = 0; k < result.vector.size(); ++k)
      {
        ASSERT_TRUE(std::isfinite(result.vector[k])) << "tile " << t;
        ASSERT_EQ(result.vector[k], expected[t].vector[k]) << "tile " << t;
      }
    }
  }
}
