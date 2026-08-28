/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the file LICENSE, distributed
 * with this software. */
#include "highmap/gpu/metal.hpp"

#include <stdexcept>

#if !HIGHMAP_HAS_METAL

namespace hmap::gpu::metal
{

namespace
{

[[noreturn]] void unavailable()
{
  throw std::runtime_error(
      "HighMap Metal backend is unavailable in this build; "
      "configure with a macOS SDK containing Metal.framework");
}

} // namespace

bool is_available()
{
  return false;
}

std::string device_name()
{
  return {};
}

DeviceCapabilities capabilities()
{
  return {};
}

ExecutionStats last_execution_stats()
{
  return {};
}

bool supports_noise(NoiseType)
{
  return false;
}

Array gradient_norm(const Array &)
{
  unavailable();
}

Array maximum_smooth(const Array &, const Array &, float)
{
  unavailable();
}

Array minimum_smooth(const Array &, const Array &, float)
{
  unavailable();
}

Array noise(NoiseType,
            glm::ivec2,
            glm::vec2,
            std::uint32_t,
            const Array *,
            const Array *,
            glm::vec4,
            glm::ivec2)
{
  unavailable();
}

Array advection_warp(const Array &,
                     const Array &,
                     const Array &,
                     const Array &,
                     float,
                     float,
                     const Array *)
{
  unavailable();
}

void thermal(Array &, const Array &, int)
{
  unavailable();
}

void hydraulic_vpipes(Array &,
                      float,
                      bool,
                      float,
                      int,
                      float,
                      float,
                      float,
                      float,
                      float,
                      float,
                      bool,
                      float,
                      Array *,
                      Array *,
                      Array *,
                      Array *,
                      Array *)
{
  unavailable();
}

} // namespace hmap::gpu::metal

#endif
