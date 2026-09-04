/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

#include <stdexcept>

#include <gtest/gtest.h>

#include "highmap/gradient.hpp"
#include "highmap/internal/opencl_run.hpp"
#include "highmap/opencl/gpu_opencl.hpp"

TEST(OpenCLBackend, BuildTimeContract)
{
#if HIGHMAP_HAS_OPENCL
  // A framework/device may be unavailable at runtime, so the ON contract is
  // verified by successful compilation and a non-crashing readiness probe.
  if (!hmap::gpu::init_opencl()) GTEST_SKIP() << "No OpenCL device is available";

  hmap::Array input({4, 4}, 0.25f);
  EXPECT_NO_THROW((void) hmap::gpu::laplacian_fract(input, 0.1f, 1));
#else
  EXPECT_FALSE(hmap::gpu::init_opencl());
  EXPECT_THROW(clwrapper::Run run("opencl_disabled_probe"), std::runtime_error);

  hmap::Array input({4, 4}, 0.25f);
  EXPECT_THROW((void) hmap::gpu::laplacian_fract(input, 0.1f, 1),
               std::runtime_error);
#endif
}
