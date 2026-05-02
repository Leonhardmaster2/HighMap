/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>
#include <list>

#include "highmap/array.hpp"
#include "highmap/boundary.hpp"
#include "highmap/filters.hpp"
#include "highmap/gradient.hpp"
#include "highmap/hydrology/hydrology.hpp"
#include "highmap/math.hpp"
#include "highmap/primitives.hpp"
#include "highmap/range.hpp"

#include "highmap/dbg/timer.hpp"

// neighbor pattern search based on D8 flow direction neighborhood
// coding

// 5 1 7
// 0 . 3
// 4 2 6
// clang-format off
#define HMAP_DINF_DI {-1, 0, 0, 1, -1, -1, 1, 1}
#define HMAP_DINF_DJ {0, 1, -1, 0, -1, 1, -1, 1}
#define HMAP_DINF_C  {1.f, 1.f, 1.f, 1.f, M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, M_SQRT1_2}
  
// the "effective contour length" of pixel i. The value of L i is 0.5
// for pixels in cardinal directions and 0.354 for pixels in diagonal
// directions (Quinn et al., 1991)
#define HMAP_DINF_ECL {0.5, 0.5, 0.5, 0.5 , 0.354, 0.354, 0.354, 0.354}
// clang-format on

namespace hmap
{

Array flow_accumulation_dinf(const Array &z, float talus_ref)
{
  const glm::ivec2       &shape = z.shape;
  const std::vector<int> &di = HMAP_DINF_DI;
  const std::vector<int> &dj = HMAP_DINF_DJ;
  const size_t           &nb = di.size();
  const int               nx = shape.x;
  const int               ny = shape.y;
  const int               ncells = nx * ny;

  Array facc = constant(shape, 1.f);
  // use raw pointer to avoid operator() overhead
  float *facc_ptr = facc.vector.data();

  std::vector<float> dinf = flow_direction_dinf_flat(z, talus_ref);

  // build nidp — parallelized with atomic increments
  std::vector<int> nidp(ncells, 0); // int, not uint8_t (atomic-safe)

#pragma omp parallel for schedule(static)
  for (int j = 1; j < ny - 1; ++j)
    for (int i = 1; i < nx - 1; ++i)
    {
      const int base = (j * nx + i) * nb;
      for (size_t k = 0; k < nb; ++k)
      {
        if (dinf[base + k] > 0.f)
        {
          const int nidx = (j + dj[k]) * nx + (i + di[k]);
#pragma omp atomic
          nidp[nidx]++;
        }
      }
    }

  // seed queue with source cells (nidp == 0)
  std::vector<int> queue;
  queue.reserve(ncells / 4);

  for (int j = 1; j < ny - 1; ++j)
    for (int i = 1; i < nx - 1; ++i)
      if (nidp[j * nx + i] == 0) queue.push_back(j * nx + i);

  // topological accumulation, inherently sequential
  while (!queue.empty())
  {
    const int idx = queue.back();
    queue.pop_back();

    const int   i = idx % nx;
    const int   j = idx / nx;
    const float acc = facc_ptr[idx];
    const int   base = idx * nb;

    for (size_t k = 0; k < nb; ++k)
    {
      const float wgt = dinf[base + k];
      if (wgt == 0.f) continue;

      const int nidx = (j + dj[k]) * nx + (i + di[k]);
      facc_ptr[nidx] += acc * wgt;

      if (--nidp[nidx] == 0) queue.push_back(nidx);
    }
  }

  fill_borders(facc);
  return facc;
}

Array flow_accumulation_dinf_perturbed(const Array &z,
                                       float        talus_ref,
                                       int          nsamples,
                                       glm::vec2    kw,
                                       uint         seed,
                                       float        amp,
                                       const Array *p_perturb_scaling,
                                       glm::vec4    bbox)
{
  const glm::ivec2 shape = z.shape;
  Array            facc(shape);

  for (int n = 0; n < nsamples; ++n)
  {
    Array zp = z;
    Array dz = noise(hmap::NoiseType::PERLIN,
                     shape,
                     kw,
                     seed + n,
                     nullptr,
                     nullptr,
                     nullptr,
                     bbox);

    if (p_perturb_scaling) dz *= *p_perturb_scaling;
    zp += amp * dz;

    facc += flow_accumulation_dinf(zp, talus_ref);
  }

  return facc / float(nsamples);
}

std::vector<Array> flow_direction_dinf(const Array &z, float talus_ref)
{
  const std::vector<int>   di = HMAP_DINF_DI;
  const std::vector<int>   dj = HMAP_DINF_DJ;
  const std::vector<float> c = HMAP_DINF_C;
  const std::vector<float> ecl = HMAP_DINF_ECL;
  const int                nb = di.size();

  // the flow-partition exponent is defined locally based on the local
  // talus in [1, 10] (Qin et al 2007)
  Array p = Array(z.shape);
  {
    Array talus = gradient_talus(z) / talus_ref;
    clamp(talus, 0.f, 1.f);
    p = 10.f * talus + 1.f;
    clamp(p, 1.f, 3.f);
  }

  // memory consuming... every 8 direction needs a full array
  std::vector<Array> dinf(nb, {z.shape});

  for (int j = 1; j < z.shape.y - 1; j++)
    for (int i = 1; i < z.shape.x - 1; i++)
    {
      for (int k = 0; k < nb; k++)
      {
        float dz = z(i, j) - z(i + di[k], j + dj[k]);
        if (dz > 0.f) dinf[k](i, j) = std::pow(dz * c[k], p(i, j)) * ecl[k];
      }

      // normalize
      float sum = 0.f;
      for (int k = 0; k < nb; k++)
        sum += dinf[k](i, j);

      if (sum > 0.f)
        for (int k = 0; k < nb; k++)
          dinf[k](i, j) /= sum;
    }

  return dinf;
}

std::vector<float> flow_direction_dinf_flat(const Array &z, float talus_ref)
{
  const glm::ivec2         &shape = z.shape;
  const std::vector<int>   &di = HMAP_DINF_DI;
  const std::vector<int>   &dj = HMAP_DINF_DJ;
  const std::vector<float> &c = HMAP_DINF_C;
  const std::vector<float> &ecl = HMAP_DINF_ECL;
  const size_t              nb = di.size();

  // precompute log(c[k]) once — eliminates one log() per neighbor per cell
  float log_c[8];
  for (size_t k = 0; k < nb; ++k)
    log_c[k] = c[k] > 0.f ? std::log(c[k])
                          : -std::numeric_limits<float>::infinity();

  std::vector<float> dinf(shape.x * shape.y * nb, 0.f);

  Array talus = gradient_talus(z);
  talus /= talus_ref;
  clamp_max(talus, 1.f);

#pragma omp parallel for schedule(static)
  for (int j = 1; j < shape.y - 1; ++j)
    for (int i = 1; i < shape.x - 1; ++i)
    {
      const float pij = std::clamp(10.f * talus(i, j) + 1.f, 1.f, 3.f);
      const float zij = z(i, j);

      float sum = 0.f;
      float tmp[8];

      for (size_t k = 0; k < nb; ++k)
      {
        float dz = zij - z(i + di[k], j + dj[k]);
        if (dz > 0.f)
        {
          const float v = fast_exp(pij * (fast_log(dz) + log_c[k])) * ecl[k];
          tmp[k] = v;
          sum += v;
        }
        else
          tmp[k] = 0.f;
      }

      if (sum > 0.f)
      {
        const float inv = 1.f / sum;
        const int   idx = (j * shape.x + i) * nb;
        for (size_t k = 0; k < nb; ++k)
          dinf[idx + k] = tmp[k] * inv;
      }
    }

  return dinf;
}

Array flow_direction_dinf_angle(const Array &z, float talus_ref)
{
  const std::vector<int>   di = HMAP_DINF_DI;
  const std::vector<int>   dj = HMAP_DINF_DJ;
  const std::vector<float> c = HMAP_DINF_C;
  const std::vector<float> ecl = HMAP_DINF_ECL;

  const uint nb = di.size();

  // flow-partition exponent (Qin et al. 2007)
  Array p = Array(z.shape);
  {
    Array talus = gradient_talus(z) / talus_ref;
    clamp_max(talus, 1.f);
    p = 10.f * talus + 1.f;
    clamp(p, 1.f, 3.f);
  }

  Array angle(z.shape, 0.f); // dominant flow direction in radians

  std::vector<float> cell_angles;

  for (uint k = 0; k < nb; k++)
  {
    float alpha = std::atan2((float)dj[k], (float)di[k]);
    cell_angles.push_back(alpha);
  }

  for (int j = 1; j < z.shape.y - 1; j++)
    for (int i = 1; i < z.shape.x - 1; i++)
    {
      float sum = 0.f;

      for (uint k = 0; k < nb; k++)
      {
        float dz = z(i, j) - z(i + di[k], j + dj[k]);
        if (dz > 0.f)
        {
          float w = std::pow(dz * c[k], p(i, j)) * ecl[k];
          angle(i, j) += w * cell_angles[k];
          sum += w;
        }
      }

      if (sum) angle(i, j) /= sum;
    }

  return angle;
}

} // namespace hmap
