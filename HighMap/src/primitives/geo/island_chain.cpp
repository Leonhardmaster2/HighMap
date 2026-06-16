/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "highmap/array.hpp"
#include "highmap/functions.hpp"
#include "highmap/geometry/path.hpp"
#include "highmap/math/core.hpp"
#include "highmap/primitives/coherent_noise.hpp"
#include "highmap/primitives/geo.hpp"

namespace hmap
{

Array island_chain_land_mask(glm::ivec2    shape,
                             const Path   &path,
                             std::uint32_t seed,
                             int           island_count,
                             float         island_radius,
                             float         size_falloff,
                             float         size_jitter,
                             float         scatter,
                             float         displacement,
                             NoiseType     noise_type,
                             float         kw,
                             int           octaves,
                             float         weight,
                             float         persistence,
                             float         lacunarity,
                             glm::vec4     bbox)
{
  Array mask(shape);

  if (path.size() < 2 || island_count < 1) return mask;

  const std::vector<Point> &pts = path.points;
  const std::vector<float>  arc = path.get_cumulative_distance();

  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dis(-1.f, 1.f);

  // position and unit tangent at a normalized arc position t
  auto sample_at = [&pts, &arc](float t, glm::vec2 &pos, glm::vec2 &tg)
  {
    t = std::clamp(t, 0.f, 1.f);

    size_t k = 1;
    while (k < arc.size() - 1 && arc[k] < t)
      k++;

    float span = std::max(arc[k] - arc[k - 1], 1e-9f);
    float r = (t - arc[k - 1]) / span;

    const Point &p0 = pts[k - 1];
    const Point &p1 = pts[k];

    pos = {(1.f - r) * p0.x + r * p1.x, (1.f - r) * p0.y + r * p1.y};

    float ddx = p1.x - p0.x;
    float ddy = p1.y - p0.y;
    float dn = std::hypot(ddx, ddy);
    tg = dn > 0.f ? glm::vec2(ddx / dn, ddy / dn) : glm::vec2(1.f, 0.f);
  };

  for (int k = 0; k < island_count; k++)
  {
    // deterministic draw order: always 3 draws per island
    float jt = dis(gen);
    float jn = dis(gen);
    float jr = dis(gen);

    float t = island_count == 1 ? 0.5f : float(k) / float(island_count - 1);
    t += scatter * jt;

    glm::vec2 pos, tg;
    sample_at(t, pos, tg);

    glm::vec2 normal = {-tg.y, tg.x};
    pos += scatter * jn * normal;

    // signed size falloff: +1 shrinks toward the path end, -1 toward start
    float fall = size_falloff >= 0.f ? 1.f - size_falloff * t
                                     : 1.f + size_falloff * (1.f - t);
    float radius_factor = std::max(0.f, fall) * (1.f + size_jitter * jr);
    float rk = island_radius * radius_factor;

    if (rk <= 0.f) continue;

    std::unique_ptr<NoiseFunction> p = create_noise_function_from_type(
        noise_type,
        glm::vec2(0.5f * kw, 0.5f * kw),
        seed + std::uint32_t(k));

    FbmFunction f = FbmFunction(std::move(p),
                                octaves,
                                weight,
                                persistence,
                                lacunarity);

    // pixel window around the island (same x/y mapping as island_land_mask)
    float lx = bbox.y - bbox.x;
    float ly = bbox.w - bbox.z;
    float margin = rk + displacement;

    int i1 = std::clamp(
        int((pos.x - margin - bbox.x) / lx * float(shape.x - 1)),
        0,
        shape.x - 1);
    int i2 = std::clamp(
        int((pos.x + margin - bbox.x) / lx * float(shape.x - 1)) + 1,
        0,
        shape.x - 1);
    int j1 = std::clamp(
        int((pos.y - margin - bbox.z) / ly * float(shape.y - 1)),
        0,
        shape.y - 1);
    int j2 = std::clamp(
        int((pos.y + margin - bbox.z) / ly * float(shape.y - 1)) + 1,
        0,
        shape.y - 1);

    for (int j = j1; j <= j2; j++)
      for (int i = i1; i <= i2; i++)
      {
        float x = lx * float(i) / float(shape.x - 1) + bbox.x - pos.x;
        float y = ly * float(j) / float(shape.y - 1) + bbox.z - pos.y;

        float r = std::hypot(x, y);
        float theta = std::atan2(y, x);

        float dr = radius_factor * displacement *
                   f.get_delegate()(std::cos(theta), std::sin(theta), 0.f);

        if (r < rk + dr) mask(i, j) = 1.f;
      }
  }

  return mask;
}

} // namespace hmap
