/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <stdexcept>

#include "highmap/local_metrics.hpp"

namespace hmap::gpu
{

Array local_metrics(const Array &array, int ir, LocalMetrics metric)
{
  switch (metric)
  {
  case LocalMetrics::LM_LOCAL_ASPECT_VARIANCE:
    return gpu::local_aspect_variance(array, ir);
  case LocalMetrics::LM_LOCAL_MAX: return gpu::local_max(array, ir);
  case LocalMetrics::LM_LOCAL_MEDIAN_DEVIATION:
    return gpu::local_median_deviation(array, ir);
  case LocalMetrics::LM_LOCAL_MIN: return gpu::local_min(array, ir);
  case LocalMetrics::LM_LOCAL_RELIEF: return gpu::local_relief(array, ir);
  case LocalMetrics::LM_LOCAL_VARIANCE: return gpu::local_variance(array, ir);
  case LocalMetrics::LM_LOCAL_MEAN: return gpu::local_mean(array, ir);
  case LocalMetrics::LM_LOCAL_SKEWNESS: return gpu::local_skewness(array, ir);
  case LocalMetrics::LM_LOCAL_Z_SCORE: return gpu::local_z_score(array, ir);
  case LocalMetrics::LM_TOPOGRAPHIC_POSITION_INDEX:
    return gpu::topographic_position_index(array, ir);
  case LocalMetrics::LM_RELATIVE_ELEVATION:
    return gpu::relative_elevation(array, ir);
  case LocalMetrics::LM_RUGGEDNESS: return gpu::ruggedness(array, ir);
  case LocalMetrics::LM_RUGOSITY_CONCAVE:
    return gpu::rugosity(array, ir, /* convex */ false);
  case LocalMetrics::LM_RUGOSITY_CONVEX:
    return gpu::rugosity(array, ir, /* convex */ true);
  default: throw std::runtime_error("unknown LocalMetrics in local_metrics");
  }
}

} // namespace hmap::gpu
