/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <numeric>

#include "macrologger.h"

#include "highmap/filters.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/operator.hpp"

namespace hmap::gpu
{

Array spectral_equalizer(const Array              &array,
                         const std::vector<float> &weights,
                         int                       ir_min,
                         int                       ir_max)
{
  const glm::ivec2 &shape = array.shape;
  const size_t      nbands = weights.size();

  if (nbands < 1) return array;

  // --- Define Gaussian half-width

  std::vector<float> tmp = logspace(float(ir_min),
                                    float(ir_max),
                                    nbands - 1,
                                    true);

  std::vector<int> irs;
  irs.reserve(tmp.size());

  int prev = 0;
  for (float v : tmp)
  {
    int r = std::max(1, (int)(v + 0.5f));

    // ensure monotonic increase for stability
    if (r <= prev) r = prev + 1;

    prev = r;
    irs.push_back(r);
  }

  // --- Blur pyramid

  std::vector<Array> blurred;
  blurred.reserve(irs.size() + 1);

  blurred.push_back(array); // level 0 = original

  for (int ir : irs)
  {
    Array wrk = array;
    gpu::smooth_cpulse(wrk, ir);
    blurred.push_back(std::move(wrk));
  }

  // --- Weighted rebuild

  Array out(shape);

  for (size_t k = 0; k < nbands; ++k)
  {
    Array band(shape);

    if (k < nbands - 1)
    {
      // band-pass: Bk - Bk+1
      band = blurred[k] - blurred[k + 1];
    }
    else
    {
      // last band = residual low frequencies
      band = blurred[k];
    }

    // weights are from low to high wavenumbers
    out += weights[nbands - 1 - k] * band;
  }

  return out;
}

} // namespace hmap::gpu
