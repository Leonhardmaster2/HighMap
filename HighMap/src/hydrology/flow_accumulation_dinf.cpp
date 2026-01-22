/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>
#include <list>

#include "highmap/array.hpp"
#include "highmap/boundary.hpp"
#include "highmap/filters.hpp"
#include "highmap/gradient.hpp"
#include "highmap/hydrology.hpp"
#include "highmap/primitives.hpp"
#include "highmap/range.hpp"

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
  const glm::ivec2       shape = z.shape;
  const std::vector<int> di = HMAP_DINF_DI;
  const std::vector<int> dj = HMAP_DINF_DJ;
  const int              nb = di.size();

  // --- initial accumulation = 1 per cell

  Array facc = constant(z.shape, 1.f);

  // --- compute Dinf directions (flattened)

  Array zf = z;
  laplace(zf);
  std::vector<float> dinf = flow_direction_dinf_flat(zf, talus_ref);

  // --- number of incoming drainage paths (integer!)

  std::vector<uint8_t> nidp(shape.x * shape.y, 0);

  for (int j = 1; j < shape.y - 1; ++j)
    for (int i = 1; i < shape.x - 1; ++i)
    {
      const int base = (j * shape.x + i) * nb;
      for (int k = 0; k < nb; ++k)
      {
        if (dinf[base + k] > 0.f)
        {
          int ni = i + di[k];
          int nj = j + dj[k];
          nidp[nj * shape.x + ni]++;
        }
      }
    }

  // --- initialize queue with sources

  std::vector<int> queue;
  queue.reserve(shape.x * shape.y);

  for (int j = 1; j < shape.y - 1; ++j)
    for (int i = 1; i < shape.x - 1; ++i)
      if (nidp[j * shape.x + i] == 0) queue.push_back(j * shape.x + i);

  // --- topological accumulation

  while (!queue.empty())
  {
    int idx = queue.back();
    queue.pop_back();

    int i = idx % shape.x;
    int j = idx / shape.x;

    const float acc = facc(i, j);
    const int   base = idx * nb;

    for (int k = 0; k < nb; ++k)
    {
      float wgt = dinf[base + k];
      if (wgt == 0.f) continue;

      int ni = i + di[k];
      int nj = j + dj[k];
      int nidx = nj * shape.x + ni;

      facc(ni, nj) += acc * wgt;

      if (--nidp[nidx] == 0) queue.push_back(nidx);
    }
  }

  fill_borders(facc);
  return facc;
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
  const glm::ivec2         shape = z.shape;
  const std::vector<int>   di = HMAP_DINF_DI;
  const std::vector<int>   dj = HMAP_DINF_DJ;
  const std::vector<float> c = HMAP_DINF_C;
  const std::vector<float> ecl = HMAP_DINF_ECL;
  const uint               nb = di.size();

  std::vector<float> dinf(shape.x * shape.y * nb, 0.f);

  Array talus = gradient_talus(z);
  talus /= talus_ref;
  clamp_max(talus, 1.f);

  for (int j = 1; j < shape.y - 1; ++j)
    for (int i = 1; i < shape.x - 1; ++i)
    {
      // const float pij = 10.f * talus(i, j) + 1.f;
      const float pij = std::clamp(10.f * talus(i, j) + 1.f, 1.f, 8.f);

      float sum = 0.f;
      float tmp[8];

      for (uint k = 0; k < nb; ++k)
      {
        float dz = z(i, j) - z(i + di[k], j + dj[k]);
        if (dz > 0.f)
        {
          float v = std::pow(dz * c[k], pij) * ecl[k];
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
        for (uint k = 0; k < nb; ++k)
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
