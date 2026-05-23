/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <fstream>
#include <string>
#include <vector>

#include "highmap/algebra.hpp"

namespace hmap
{

void to_csv(const std::vector<glm::vec3> &xyz, const std::string &fname)
{
  std::ofstream f(fname, std::ios::out);
  if (!f.is_open()) return;

  for (const auto &p : xyz)
    f << p.x << "," << p.y << "," << p.z << "\n";

  f.close();
}

} // namespace hmap
