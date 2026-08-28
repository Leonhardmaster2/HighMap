/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the file LICENSE, distributed
 * with this software. */
#include "highmap/gpu/metal.hpp"

#include <stdexcept>
#include <utility>

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

DeviceArray::DeviceArray(std::shared_ptr<detail::DeviceArrayState> state)
    : state_(std::move(state))
{
}

DeviceArray::~DeviceArray() = default;

bool DeviceArray::empty() const noexcept
{
  return !state_;
}

glm::ivec2 DeviceArray::shape() const
{
  return {};
}

std::size_t DeviceArray::size() const
{
  return 0;
}

StorageMode DeviceArray::storage_mode() const
{
  return StorageMode::shared;
}

ResidencyState DeviceArray::residency_state() const
{
  return ResidencyState::host_valid;
}

std::string DeviceArray::debug_name() const
{
  return {};
}

void DeviceArray::set_debug_name(const std::string &)
{
  unavailable();
}

Array DeviceArray::to_array() const
{
  unavailable();
}

DeviceSession::DeviceSession(StorageMode)
{
  unavailable();
}

DeviceSession::DeviceSession(DeviceSession &&) noexcept = default;

DeviceSession &DeviceSession::operator=(DeviceSession &&) noexcept = default;

DeviceSession::~DeviceSession() noexcept = default;

DeviceArray DeviceSession::upload(const Array &, StorageMode)
{
  unavailable();
}

DeviceArray DeviceSession::allocate(glm::ivec2, StorageMode)
{
  unavailable();
}

DeviceArray DeviceSession::gradient_norm(DeviceArray)
{
  unavailable();
}

DeviceArray DeviceSession::maximum_smooth(DeviceArray,
                                           const DeviceArray &,
                                           float)
{
  unavailable();
}

DeviceArray DeviceSession::minimum_smooth(DeviceArray,
                                           const DeviceArray &,
                                           float)
{
  unavailable();
}

DeviceArray DeviceSession::noise(NoiseType,
                                 glm::ivec2,
                                 glm::vec2,
                                 std::uint32_t,
                                 const DeviceArray *,
                                 const DeviceArray *,
                                 glm::vec4,
                                 glm::ivec2)
{
  unavailable();
}

DeviceArray DeviceSession::advection_warp(DeviceArray,
                                           const DeviceArray &,
                                           const DeviceArray &,
                                           const DeviceArray &,
                                           float,
                                           float,
                                           const DeviceArray *)
{
  unavailable();
}

DeviceArray DeviceSession::advection_warp_texture(
    DeviceArray,
    const DeviceArray &,
    const DeviceArray &,
    const DeviceArray &,
    float,
    float,
    const DeviceArray *)
{
  unavailable();
}

DeviceArray DeviceSession::thermal(DeviceArray, const DeviceArray &, int)
{
  unavailable();
}

DeviceArray DeviceSession::thermal_ridge(DeviceArray,
                                          const DeviceArray &,
                                          int)
{
  unavailable();
}

DeviceArray DeviceSession::extrapolate_borders(DeviceArray)
{
  unavailable();
}

DeviceArray DeviceSession::linear_combine(DeviceArray,
                                           const DeviceArray &,
                                           float,
                                           float)
{
  unavailable();
}

DeviceArray DeviceSession::hydraulic_vpipes(
    DeviceArray,
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
    const DeviceArray *,
    DeviceArray *,
    DeviceArray *,
    DeviceArray *,
    DeviceArray *)
{
  unavailable();
}

Array DeviceSession::download(const DeviceArray &) const
{
  unavailable();
}

void DeviceSession::submit()
{
  unavailable();
}

void DeviceSession::finish() const
{
  unavailable();
}

ExecutionStats DeviceSession::stats() const
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
