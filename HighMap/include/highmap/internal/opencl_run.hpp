/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

/**
 * @file opencl_run.hpp
 * @brief Build-time OpenCL compatibility boundary for GPU wrapper sources.
 *
 * OpenCL-backed wrappers remain source-compatible when the optional backend is
 * disabled. The disabled implementation deliberately fails at construction
 * instead of silently returning incomplete data.
 */
#pragma once

#include <stdexcept>
#include <string>
#include <initializer_list>
#include <vector>

#ifndef HIGHMAP_HAS_OPENCL
#define HIGHMAP_HAS_OPENCL 1
#endif

#if HIGHMAP_HAS_OPENCL

#include "cl_wrapper/run.hpp"

#else

namespace clwrapper
{

[[noreturn]] inline void throw_opencl_disabled()
{
  throw std::runtime_error(
      "HighMap OpenCL backend is disabled at build time "
      "(HIGHMAP_ENABLE_OPENCL=OFF)");
}

/**
 * @brief Compile-only stand-in for CLWrapper's Run type.
 *
 * Keeping the wrapper translation units in the library preserves the public
 * hmap::gpu API and gives callers a deterministic error if they explicitly
 * request an OpenCL-only operation. No OpenCL headers, symbols, or framework
 * are required in this mode.
 */
class Run
{
public:
  explicit Run(const std::string &) { throw_opencl_disabled(); }
  ~Run() = default;

  template <typename... Args> void bind_arguments(Args &&...)
  {
    throw_opencl_disabled();
  }

  template <typename... Args> void set_argument(Args &&...)
  {
    throw_opencl_disabled();
  }

  template <typename T, typename Vector, typename... Args>
  void bind_buffer(const std::string &, Vector &&, Args &&...)
  {
    throw_opencl_disabled();
  }

  template <typename Vector, typename... Args>
  void bind_buffer(const std::string &, Vector &&, Args &&...)
  {
    throw_opencl_disabled();
  }

  template <typename... Args> void bind_imagef(Args &&...)
  {
    throw_opencl_disabled();
  }

  template <typename... Args> void execute(Args &&...)
  {
    throw_opencl_disabled();
  }

  void execute(std::initializer_list<int>, float * = nullptr)
  {
    throw_opencl_disabled();
  }

  template <typename... Args> void read_buffer(Args &&...)
  {
    throw_opencl_disabled();
  }

  template <typename... Args> void read_imagef(Args &&...)
  {
    throw_opencl_disabled();
  }

  template <typename... Args> void write_buffer(Args &&...)
  {
    throw_opencl_disabled();
  }

  template <typename... Args> void write_imagef(Args &&...)
  {
    throw_opencl_disabled();
  }

  void reset_argcount() { throw_opencl_disabled(); }
};

} // namespace clwrapper

#endif
