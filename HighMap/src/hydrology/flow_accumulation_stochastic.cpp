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
#include "highmap/hydrology/hydrology.hpp"

namespace hmap
{

// Downhill velocity field: central difference with one-sided fallback at
// the boundaries (matches geotransport gradient.cu as shipped), negated so
// the field points downhill. Shared with the GPU wrapper.
void downhill_velocity(const Array &z, Array &vx, Array &vy)
{
  int nx = z.shape.x;
  int ny = z.shape.y;

  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    {
      float gx, gy;

      if (i > 0 && i < nx - 1)
        gx = 0.5f * (z(i + 1, j) - z(i - 1, j));
      else if (i > 0)
        gx = z(i, j) - z(i - 1, j);
      else if (i < nx - 1)
        gx = z(i + 1, j) - z(i, j);
      else
        gx = 0.f;

      if (j > 0 && j < ny - 1)
        gy = 0.5f * (z(i, j + 1) - z(i, j - 1));
      else if (j > 0)
        gy = z(i, j) - z(i, j - 1);
      else if (j < ny - 1)
        gy = z(i, j + 1) - z(i, j);
      else
        gy = 0.f;

      vx(i, j) = -gx;
      vy(i, j) = -gy;
    }
}

namespace
{

// SplitMix64 — deterministic per-sample uniforms
uint64_t splitmix64(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

float uniform01(uint64_t h)
{
  return (float)(h >> 40) / 16777216.f; // 24-bit mantissa in [0, 1)
}

float bilinear(const Array &a, float x, float y)
{
  int nx = a.shape.x, ny = a.shape.y;
  int i = (int)x, j = (int)y;
  if (i > nx - 2) i = nx - 2;
  if (j > ny - 2) j = ny - 2;
  float u = x - (float)i, v = y - (float)j;
  return (1.f - u) * (1.f - v) * a(i, j) + u * (1.f - v) * a(i + 1, j) +
         (1.f - u) * v * a(i, j + 1) + u * v * a(i + 1, j + 1);
}

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

Array flow_accumulation_stochastic(const Array   &z,
                                   int            n_samples,
                                   std::uint32_t  seed,
                                   const Array   *p_source,
                                   const Array   *p_decay)
{
  int nx = z.shape.x;
  int ny = z.shape.y;

  Array vx(z.shape), vy(z.shape);
  downhill_velocity(z, vx, vy);

  Array flux(z.shape);

  const float eps = 1e-16f;
  const int   max_step = nx + ny;              // Manhattan bound
  const float p_inv = (float)nx * (float)ny;   // 1/P with unit cell area

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

      float v_x = bilinear(vx, px, py);
      float v_y = bilinear(vy, px, py);
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
