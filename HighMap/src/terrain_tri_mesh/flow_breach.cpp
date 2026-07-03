/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstddef>
#include <vector>

#include "highmap/terrain_tri_mesh.hpp"

namespace hmap
{

void TerrainTriMesh::flow_breach(float epsilon)
{
  auto paths = compute_shortest_paths_to_hull(true);

  std::vector<bool>   visited(points.size(), false);
  std::vector<size_t> path;

  for (size_t start = 0; start < points.size(); ++start)
  {
    std::vector<size_t> path = this->path_to_hull(start, paths);

    for (size_t i = 0; i < path.size() - 1; ++i)
    {
      size_t child = path[i + 1];
      size_t parent = path[i];
      float  required = points[parent].z - epsilon;
      if (points[child].z > required) points[child].z = required;
    }
  }
}

} // namespace hmap
