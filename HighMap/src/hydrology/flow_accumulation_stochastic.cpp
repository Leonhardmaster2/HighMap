/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/* Port of the stochastic transport estimator from erosiv/geotransport
 * (MIT license) — "Stochastic Geomorphological Transport for Terrain
 * Erosion Simulation", N. McDonald & G. Cordonnier. Adapted to HighMap
 * conventions: unit cell scale, hash-based deterministic RNG, clamped
 * normalization (reference yields inf on zero-velocity cells). */
#include <cmath>
#include <cstdint>

#include "highmap/array.hpp"
#include "highmap/gradient.hpp"
#include "highmap/hydrology/hydrology.hpp"
#include "highmap/random.hpp"

namespace hmap
{

namespace
{

// distance to the next cell-boundary intersection midpoint along d
// (port of geotransport __stepsize; d must be normalized)
float voxel_stepsize(float px, float py, float dx, float dy)
{
  const float tmax = 1.41421356f;

  float x_neg = std::floor(px), x_pos = 1.f + x_neg;
  float y_neg = std::floor(py), y_pos = 1.f + y_neg;

  float tx = std::min(std::max((x_neg - px) / dx, (x_pos - px) / dx), tmax);
  float ty = std::min(std::max((y_neg - py) / dy, (y_pos - py) / dy), tmax);

  return 0.5f * (tx + ty);
}

} // namespace

Array flow_accumulation_stochastic(const Array  &z,
                                   int           n_samples,
                                   std::uint32_t seed,
                                   const Array  *p_source,
                                   const Array  *p_decay)
{
  int nx = z.shape.x;
  int ny = z.shape.y;

  Array vx = -gradient_x(z);
  Array vy = -gradient_y(z);

  Array flux(z.shape);

  const float eps = 1e-16f;
  const int   max_step = nx + ny;            // Manhattan bound
  const float p_inv = (float)nx * (float)ny; // 1/P with unit cell area

#pragma omp parallel for schedule(dynamic, 256)
  for (int n = 0; n < n_samples; ++n)
  {
    uint64_t h = splitmix64(((uint64_t)seed << 32) ^ (uint64_t)n);
    float    px = uniform01(h) * (float)nx;
    h = splitmix64(h);
    float py = uniform01(h) * (float)ny;

    int i0 = (int)px, j0 = (int)py;
    if (i0 >= nx || j0 >= ny) continue;

    float s = (p_source ? (*p_source)(i0, j0) : 1.f) * p_inv;
    if (std::abs(s) < eps) continue;

    float att = 1.f;
    int   ind = j0 * nx + i0;

    for (int step = 0; step < max_step; ++step)
    {
      int ci = (int)px, cj = (int)py;
      if (ci < 0 || ci >= nx || cj < 0 || cj >= ny) break;

      // deposit with current attenuation upon entering a new cell
      int cind = cj * nx + ci;
      if (cind != ind)
      {
        ind = cind;
        float dep = s * att;
#pragma omp atomic
        flux.vector[cind] += dep;
      }

      // bilinear interpolation params
      int   ip = (int)px;
      int   jp = (int)py;
      float up = px - ip;
      float vp = py - jp;

      float v_x = vx.get_value_bilinear_at(ip, jp, up, vp);
      float v_y = vy.get_value_bilinear_at(ip, jp, up, vp);
      float v_len = std::hypot(v_x, v_y);
      if (v_len < eps) break;

      float ds = voxel_stepsize(px, py, v_x / v_len, v_y / v_len);
      px += ds * v_x / v_len;
      py += ds * v_y / v_len;

      float dlambda = ds * 1.41421356f / v_len; // L = |(1,1)| in cell units
      att *= std::exp(-dlambda * (p_decay ? (*p_decay)(ci, cj) : 0.f));
      if (std::abs(att) < eps) break;
    }
  }

  // analytic normalization; clamped so zero-velocity cells stay finite
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    {
      float src = p_source ? (*p_source)(i, j) : 1.f;
      float norm = std::abs(vx(i, j)) + std::abs(vy(i, j));
      norm = std::max(norm, 1e-9f);
      flux(i, j) = (src + flux(i, j) / (float)n_samples) / norm;
    }

  return flux;
}

} // namespace hmap
