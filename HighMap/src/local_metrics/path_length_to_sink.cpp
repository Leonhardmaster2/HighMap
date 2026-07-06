/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <vector>

#include "highmap/array.hpp"

namespace hmap
{

Array path_length_to_outlet(const Array &z)
{
  const glm::ivec2 &shape = z.shape;
  Array             out(shape, -1.f);

  constexpr glm::ivec2 offsets[8] =
      {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};

  std::vector<glm::ivec2> stack;
  stack.reserve(shape.x);

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      if (out(i, j) >= 0.f) continue;

      stack.clear();

      glm::ivec2 p(i, j);

      while (true)
      {
        if (out(p) == -2.f || out(p) >= 0.f) break;

        stack.push_back(p);

        float zp = z(p);

        float      best_z = zp;
        glm::ivec2 next = p;

        for (const glm::ivec2 &d : offsets)
        {
          glm::ivec2 q = p + d;

          // flow leaves the domain
          if (q.x < 0 || q.x >= shape.x || q.y < 0 || q.y >= shape.y)
          {
            out(p) = 0.f;
            next = q;
            break;
          }

          if (z(q) < best_z)
          {
            best_z = z(q);
            next = q;
          }
        }

        if (next.x < 0 || next.x >= shape.x || next.y < 0 || next.y >= shape.y)
          break;

        // internal sink
        if (next == p)
        {
          out(p) = -2.f; // temporary marker
          break;
        }

        p = next;
      }

      float value = out(p);

      if (value == -2.f)
      {
        // path ends in a sink: every upstream cell gets sink marker -2.f
        for (const auto &c : stack)
          out(c) = -2.f;
      }
      else
      {
        // path reaches the outlet
        int length = static_cast<int>(value);

        for (auto it = stack.rbegin(); it != stack.rend(); ++it)
        {
          ++length;
          out(it->x, it->y) = static_cast<float>(length);
        }
      }
    }

  // clean-up markers
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
      if (out(i, j) == -2.f) out(i, j) = 0.f;

  return out;
}

Array path_length_to_sink(const Array &z)
{
  const glm::ivec2 &shape = z.shape;
  Array             out(shape, -1.f);

  constexpr glm::ivec2 offsets[8] =
      {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};

  std::vector<glm::ivec2> stack;
  stack.reserve(shape.x);

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      if (out(i, j) >= 0.f) continue;

      stack.clear();

      glm::ivec2 p(i, j);

      while (true)
      {
        // already known
        if (out(p) >= 0.f) break;

        stack.push_back(p);

        float zp = z(p);

        float      best_z = zp;
        glm::ivec2 next = p;

        // find steepest lower neighbour
        for (const glm::ivec2 &d : offsets)
        {
          glm::ivec2 q = p + d;

          if (q.x < 0 || q.x >= shape.x || q.y < 0 || q.y >= shape.y) continue;

          if (z(q) < best_z)
          {
            best_z = z(q);
            next = q;
          }
        }

        // sink reached
        if (next == p)
        {
          out(p) = 0.f;
          break;
        }

        p = next;
      }

      // Propagate distances back upstream
      int length = static_cast<int>(out(p));

      for (auto it = stack.rbegin(); it != stack.rend(); ++it)
      {
        ++length;
        out(it->x, it->y) = static_cast<float>(length);
      }
    }

  return out;
}

} // namespace hmap
