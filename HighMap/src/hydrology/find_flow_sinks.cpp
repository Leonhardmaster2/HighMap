/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <sys/types.h> // for size_t, uint

#include <vector> // for vector

#include "highmap/array.hpp"               // for Array
#include "highmap/hydrology/hydrology.hpp" // for d8_compute_ndip, find_flo...

namespace hmap
{

void find_flow_apex(const Array &z, std::vector<int> &is, std::vector<int> &js)
{
  Array d8 = flow_direction_d8(z);
  Array nidp = d8_compute_ndip(d8);

  is.clear();
  js.clear();

  const int nx = z.shape.x;
  const int ny = z.shape.y;

  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    {
      if (nidp(i, j) == 0)
      {
        is.push_back(i);
        js.push_back(j);
      }
    }
}

void find_flow_sinks(const Array &z, std::vector<int> &is, std::vector<int> &js)
{
  is.clear();
  js.clear();

  const std::vector<int> di = {-1, -1, 0, 1, 1, 1, 0, -1};
  const std::vector<int> dj = {0, 1, 1, 1, 0, -1, -1, -1};
  const uint             nb = di.size();

  for (int j = 1; j < z.shape.y - 1; j++)
    for (int i = 1; i < z.shape.x - 1; i++)
    {
      // count neighbor cells with a higher elevation than the
      // current cell
      int n_higher_cells = 0;
      for (size_t k = 0; k < nb; k++)
      {
        int ik = i + di[k];
        int jk = j + dj[k];
        if (z(i, j) < z(ik, jk)) n_higher_cells++;
      }

      if (n_higher_cells == 8)
      {
        is.push_back(i);
        js.push_back(j);
      }
    }
}

std::vector<glm::ivec2> find_flow_sinks(const Array &z)
{
  std::vector<glm::ivec2> indices;

  const std::vector<int> di = {-1, -1, 0, 1, 1, 1, 0, -1};
  const std::vector<int> dj = {0, 1, 1, 1, 0, -1, -1, -1};
  const uint             nb = di.size();

  for (int j = 1; j < z.shape.y - 1; j++)
    for (int i = 1; i < z.shape.x - 1; i++)
    {
      // count neighbor cells with a higher elevation than the
      // current cell
      int n_higher_cells = 0;
      for (size_t k = 0; k < nb; k++)
      {
        int ik = i + di[k];
        int jk = j + dj[k];
        if (z(i, j) < z(ik, jk)) n_higher_cells++;
      }

      if (n_higher_cells == 8) indices.push_back({i, j});
    }

  return indices;
}

std::vector<glm::ivec2> find_flow_sinks_border(const Array &z)
{
  std::vector<glm::ivec2> indices;

  const int rows = z.shape.x;
  const int cols = z.shape.y;

  const int di[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
  const int dj[8] = {0, 1, 1, 1, 0, -1, -1, -1};

  auto is_inside = [&](int i, int j)
  { return i >= 0 && j >= 0 && i < rows && j < cols; };

  for (int j = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i)
    {
      // only border cells
      if (!(i == 0 || j == 0 || i == rows - 1 || j == cols - 1)) continue;

      int valid_neighbors = 0;
      int higher_neighbors = 0;

      for (int k = 0; k < 8; ++k)
      {
        int ni = i + di[k];
        int nj = j + dj[k];

        if (!is_inside(ni, nj)) continue;

        valid_neighbors++;

        if (z(i, j) < z(ni, nj)) higher_neighbors++;
      }

      // sink: all valid neighbors are higher
      if (valid_neighbors > 0 && higher_neighbors == valid_neighbors)
        indices.push_back(glm::ivec2(i, j));
    }

  return indices;
}

} // namespace hmap
