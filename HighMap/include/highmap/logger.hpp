/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file logger.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @brief Custom logger utility with spdlog-like formatting and source location.
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include <iostream>
#include <string_view>

#include <format>
#include <source_location>

#ifndef HIGHMAP_ENABLE_LOGS
#define HIGHMAP_ENABLE_LOGS 1
#endif

namespace hmap::log
{

#if !HIGHMAP_ENABLE_LOGS

template <typename... Args> inline void trace([[maybe_unused]] Args &&...args)
{
}

template <typename... Args> inline void info([[maybe_unused]] Args &&...args)
{
}

template <typename... Args> inline void warn([[maybe_unused]] Args &&...args)
{
}

template <typename... Args> inline void error([[maybe_unused]] Args &&...args)
{
}

#else

template <typename... Args> struct format_string_with_loc
{
  std::format_string<Args...> fmt;
  std::source_location        loc;

  template <typename S>
  consteval format_string_with_loc(
      const S                    &s,
      const std::source_location &l = std::source_location::current())
      : fmt(s), loc(l)
  {
  }
};

namespace detail
{

enum class Level
{
  Trace,
  Info,
  Warn,
  Error
};

inline const char *level_to_string(Level level)
{
  switch (level)
  {
  case Level::Trace: return "trace";
  case Level::Info: return "info";
  case Level::Warn: return "warn";
  case Level::Error: return "error";
  }
  return "info";
}

constexpr std::string_view get_filename(std::string_view path)
{
  const size_t pos = path.find_last_of("/\\");
  return (pos == std::string_view::npos) ? path : path.substr(pos + 1);
}

inline void log_impl(Level                       level,
                     std::string_view            msg,
                     const std::source_location &loc)
{
  std::ostream &out = (level == Level::Warn || level == Level::Error)
                          ? std::cerr
                          : std::cout;
  out << "[" << level_to_string(level) << "] [" << get_filename(loc.file_name())
      << ":" << loc.line() << " (" << loc.function_name() << ")] " << msg
      << std::endl;
}

} // namespace detail

// -----------------------------------------------------------------------------
// trace
// -----------------------------------------------------------------------------

template <typename... Args>
inline void trace(format_string_with_loc<std::type_identity_t<Args>...> fl,
                  Args &&...args)
{
  std::string msg = std::vformat(fl.fmt.get(), std::make_format_args(args...));
  detail::log_impl(detail::Level::Trace, msg, fl.loc);
}

inline void trace(
    std::string_view            message,
    const std::source_location &loc = std::source_location::current())
{
  detail::log_impl(detail::Level::Trace, message, loc);
}

// -----------------------------------------------------------------------------
// info
// -----------------------------------------------------------------------------

template <typename... Args>
inline void info(format_string_with_loc<std::type_identity_t<Args>...> fl,
                 Args &&...args)
{
  std::string msg = std::vformat(fl.fmt.get(), std::make_format_args(args...));
  detail::log_impl(detail::Level::Info, msg, fl.loc);
}

inline void info(
    std::string_view            message,
    const std::source_location &loc = std::source_location::current())
{
  detail::log_impl(detail::Level::Info, message, loc);
}

// -----------------------------------------------------------------------------
// warn
// -----------------------------------------------------------------------------

template <typename... Args>
inline void warn(format_string_with_loc<std::type_identity_t<Args>...> fl,
                 Args &&...args)
{
  std::string msg = std::vformat(fl.fmt.get(), std::make_format_args(args...));
  detail::log_impl(detail::Level::Warn, msg, fl.loc);
}

inline void warn(
    std::string_view            message,
    const std::source_location &loc = std::source_location::current())
{
  detail::log_impl(detail::Level::Warn, message, loc);
}

// -----------------------------------------------------------------------------
// error
// -----------------------------------------------------------------------------

template <typename... Args>
inline void error(format_string_with_loc<std::type_identity_t<Args>...> fl,
                  Args &&...args)
{
  std::string msg = std::vformat(fl.fmt.get(), std::make_format_args(args...));
  detail::log_impl(detail::Level::Error, msg, fl.loc);
}

inline void error(
    std::string_view            message,
    const std::source_location &loc = std::source_location::current())
{
  detail::log_impl(detail::Level::Error, message, loc);
}

#endif

} // namespace hmap::log
