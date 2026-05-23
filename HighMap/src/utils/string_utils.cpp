/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>

namespace hmap
{

std::filesystem::path add_filename_suffix(
    const std::filesystem::path &file_path,
    const std::string           &suffix)
{

  std::filesystem::path stem = file_path.stem();
  std::filesystem::path ext = file_path.extension();
  std::filesystem::path new_path = file_path.parent_path() /
                                   std::filesystem::path(stem.string() +
                                                         suffix + ext.string());
  return new_path;
}

std::filesystem::path make_unique_temp_dir(const std::string &prefix)
{
  std::filesystem::path base = std::filesystem::temp_directory_path();

  std::random_device                      rd;
  std::mt19937_64                         gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  for (;;)
  {
    std::filesystem::path dir = base /
                                (prefix + "_" + std::to_string(dis(gen)));

    if (std::filesystem::create_directory(dir)) return dir;
  }
}

std::string zfill(const std::string &str, int n_zero)
{
  return std::string(n_zero - std::min(n_zero, (int)str.length()), '0') + str;
}

} // namespace hmap
