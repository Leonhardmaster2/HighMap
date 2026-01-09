/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/filters.hpp"
#include "highmap/hydrology.hpp"
#include "highmap/morphology.hpp"
#include "highmap/range.hpp"

#include "highmap/primitives.hpp"
#include "highmap/transform.hpp"

namespace hmap
{

Array flow_fixing(const Array &z,
                  float        riverbed_talus,
                  int          iterations,
                  int          prefilter_ir,
                  bool         carve_riverbed,
                  bool         smooth_river_bottom,
                  float        talus_riverbank,
                  uint         seed,
                  float        riverbank_noise_ratio,
                  float        merging_distance,
                  const Array *p_noise_x,
                  const Array *p_noise_y)
{
  // local node type for heap queues
  struct Node
  {
    float h;
    int   i;
    int   j;

    bool operator<(const Node &other) const
    {
      return h > other.h; // min-heap
    }
  };

  struct NodePath
  {
    float     h;  // elevation at end point
    Vec2<int> p0; // start
    Vec2<int> p1; // end

    bool operator<(const NodePath &other) const
    {
      return h < other.h; // max-heap
    }
  };

  //
  const Vec2<int> shape = z.shape;
  Array           zb = z;
  size_t          n_sinks = 0;

  // neighbor search
  const int di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dj[8] = {0, 1, 1, 1, 0, -1, -1, -1};

  auto is_inside = [&shape](int i, int j)
  { return i >= 0 && i < shape.x && j >= 0 && j < shape.y; };

  // --- main loop

  for (int it = 0; it < iterations; ++it)
  {
    Array zf = zb;
    smooth_cpulse(zf, prefilter_ir);

    std::vector<Vec2<int>> sinks = find_flow_sinks(zf);
    Mat<int>               is_sink(shape, 0);

    for (const auto &p : sinks)
      is_sink(p) = 1;

    if (sinks.size() == n_sinks)
      break;
    else
      n_sinks = sinks.size();

    // --- flow breaching: 1st pass

    std::vector<Node> queue;
    queue.reserve(shape.x * shape.y);

    Mat<int>                                    visited(shape, 0);
    Mat<Vec2<int>>                              flow_map(shape, {0, 0});
    std::map<Vec4<int>, std::vector<Vec2<int>>> breach_history;

    // --- initialize heap queue with the lowest cell of each border

    for (int i : {0, shape.x - 1})
    {
      float vmin = 1e30f;
      int   jmin = 0;
      for (int j = 0; j < shape.y; ++j)
      {
        if (zb(i, j) < vmin)
        {
          vmin = zb(i, j);
          jmin = j;
        }
      }
      queue.push_back({vmin, i, jmin});
    }

    for (int j : {0, shape.y - 1})
    {
      float vmin = 1e30f;
      int   imin = 0;
      for (int i = 0; i < shape.x; ++i)
      {
        if (zb(i, j) < vmin)
        {
          vmin = zb(i, j);
          imin = i;
        }
      }
      queue.push_back({vmin, imin, j});
    }

    // --- traverse the queue

    std::make_heap(queue.begin(), queue.end());

    while (!queue.empty())
    {
      std::pop_heap(queue.begin(), queue.end());
      const Node c = queue.back();
      queue.pop_back();

      for (int k = 0; k < 8; ++k)
      {
        int ni = c.i + di[k];
        int nj = c.j + dj[k];

        if (is_inside(ni, nj) && visited(ni, nj) == 0)
        {
          // store flow direction
          flow_map(ni, nj) = {c.i, c.j};
          visited(ni, nj) = 1;
          // queue.push_back({zb(ni, nj), ni, nj});
          queue.push_back({zb(ni, nj) + 1.f, ni, nj});
          // queue.push_back({std::abs(zb(ni, nj) - zb(c.i, c.j)) + c.h, ni,
          // nj});
          std::push_heap(queue.begin(), queue.end());

          // if the current cell is a sink, "breach" the
          // heightmap by following the reverse flow direction
          // in order to connect this sink to another sink, or
          // to connect this connect to the domain border
          if (is_sink(ni, nj))
          {
            int bi = ni;
            int bj = nj;

            bool                   keep_breaching = true;
            std::vector<Vec2<int>> path = {{bi, bj}};

            // stop on a boundary or at a sink
            while (keep_breaching &&
                   (bi > 0 && bi < shape.x - 1 && bj > 0 && bj < shape.y - 1))
            {
              Vec2<int> tmp = flow_map(bi, bj);
              bi = tmp.x;
              bj = tmp.y;
              path.push_back({bi, bj});
              if (is_sink(bi, bj)) keep_breaching = false;
            }

            // after the breaching path has been identifier,
            // follow the this path and make sure the
            // elevation is monotonic along this path
            if (path.size() > 2)
            {
              Vec2<int> p0 = path.front();
              Vec2<int> p1 = path.back();
              if (p0 != p1)
              {
                // store the breaching path for the second
                // pass of the algorithm
                Vec4<int> key = {p0.x, p0.y, p1.x, p1.y};
                breach_history[key] = path;

                for (size_t r = 0; r < path.size() - 1; ++r)
                {
                  if (zb(path[r + 1]) > zb(path[r]))
                    zb(path[r + 1]) = zb(path[r]) - riverbed_talus;
                  else
                    zb(path[r + 1]) -= riverbed_talus;
                }
              }
            }
          }
        } // if visited
      } // neighbors k-loop
    } // queue

    // --- 2nd pass

    // breach again from top to bottom (hence the -z in the cost) to
    // ensure overall elevations are coherent between the sinks
    std::vector<NodePath> queue_path;
    queue.reserve(breach_history.size());

    for (const auto &[key, path] : breach_history)
    {
      Vec2<int> p0 = {key.a, key.b};
      Vec2<int> p1 = {key.c, key.d};
      queue_path.push_back({z(p1), p0, p1});
    }
    std::make_heap(queue_path.begin(), queue_path.end());

    while (!queue.empty())
    {
      std::pop_heap(queue_path.begin(), queue_path.end());
      const NodePath current = queue_path.back();
      queue_path.pop_back();

      Vec4<int> key = {current.p0.x, current.p0.y, current.p1.x, current.p1.y};
      const std::vector<Vec2<int>> &path = breach_history[key];

      // breach again
      for (size_t r = 0; r < path.size() - 1; ++r)
      {
        if (zb(path[r + 1]) > zb(path[r]))
          zb(path[r + 1]) = zb(path[r]) - riverbed_talus;
        else
          zb(path[r + 1]) -= riverbed_talus;
      }
    }

  } // main it

  // --- carve the river

  if (carve_riverbed)
  {
    Array mask(shape);
    Array zr = zb;

    for (int j = 0; j < shape.y; j++)
      for (int i = 0; i < shape.x; i++)
        if (zb(i, j) != z(i, j)) mask(i, j) = 1.f;

    int ir = 2;
    expand_talus(zr, mask, talus_riverbank, seed, ir, riverbank_noise_ratio);

    if (smooth_river_bottom) laplace(zr);

    // transition mask
    mask = distance_transform(mask);
    mask = exp(-mask / merging_distance);
    laplace(mask);

    warp(mask, p_noise_x, p_noise_y);

    return lerp(z, zr, mask);
  }
  else
  {
    return zb;
  }
}

} // namespace hmap
