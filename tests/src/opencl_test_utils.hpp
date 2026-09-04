/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

#pragma once

#include <gtest/gtest.h>

#include "highmap/opencl/gpu_opencl.hpp"

// GPU compatibility tests are meaningful only when the optional OpenCL
// backend is present. Keep them in the common test binary so the same source
// set exercises both configurations, but turn unavailable-backend cases into
// explicit skips instead of failures.
#define HMAP_SKIP_IF_NO_OPENCL()                                             \
  do                                                                         \
  {                                                                          \
    static const bool hmap_opencl_available = hmap::gpu::init_opencl();      \
    if (!hmap_opencl_available)                                              \
      GTEST_SKIP() << "OpenCL backend is disabled or no OpenCL device is available"; \
  } while (false)
