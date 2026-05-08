/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/boundary.hpp"
#include "highmap/features.hpp"
#include "highmap/filters.hpp"
#include "highmap/local_metrics.hpp"
#include "highmap/morphology.hpp"
#include "highmap/range.hpp"

namespace hmap
{

Array area_remove(const Array &array,
                  float        threshold_size,
                  float        background_value,
                  float        fill_value)
{
  const glm::ivec2 &shape = array.shape;

  // label connected components
  std::map<float, float> area;

  // remove background here
  Array mask = is_non_zero(array - background_value);
  Array labels = connected_components(mask,
                                      /* surface_threshold */ 0.f,
                                      /* background_value */ 0.f,
                                      &area);

  if (area.empty()) return array;

  // build output
  Array out = array;

  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
    {
      if (labels(i, j) != 0.f && area[labels(i, j)] < threshold_size)
        out(i, j) = fill_value;
    }

  return out;
}

Array border(const Array &array, int ir)
{
  return array - erosion(array, ir);
}

Array closing(const Array &array, int ir)
{
  return erosion(dilation(array, ir), ir);
}

Array closing_by_reconstruction(const Array &array, int ir, float k_smooth_max)
{
  Array marker = dilation(array, ir);
  return reconstruction_by_erosion(marker, array, ir, k_smooth_max);
}

Array contour_smoothing(const Array &array, int ir, float transition_ratio)
{
  Array edt = distance_transform(is_zero(array)) - distance_transform(array);
  smooth_cpulse(edt, 2 * ir);

  float width = transition_ratio * ir;
  edt /= width;
  clamp(edt, -1.f, 1.f);
  edt = smoothstep3(0.5f * edt + 0.5f);

  return edt;
}

Array dilation(const Array &array, int ir)
{
  return local_max(array, ir);
}

Array dilation_expand_border_only(const Array &array, int ir)
{
  const glm::ivec2 &shape = array.shape;
  Array             out = local_max(array, ir);

  // only keep result in the "background" to leave initial vlaues
  // untouched
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      if (array(i, j) != 0.f) out(i, j) = array(i, j);
    }

  return out;
}

Array dilation_expand_min_value_border_only(const Array &array)
{
  const glm::ivec2 &shape = array.shape;
  Array             out(shape);

  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      // non-background pixels are left untouched
      if (array(i, j) != 0.f)
      {
        out(i, j) = array(i, j);
        continue;
      }

      // for background pixels: find min non-zero neighbor
      float vmin_non_zero = std::numeric_limits<float>::max();
      for (int r = -1; r <= 1; ++r)
        for (int s = -1; s <= 1; ++s)
        {
          if (r == 0 && s == 0) continue;

          const int ni = i + r;
          const int nj = j + s;

          if (ni >= 0 && ni < shape.x && nj >= 0 && nj < shape.y &&
              array(ni, nj) != 0.f)
            vmin_non_zero = std::min(vmin_non_zero, array(ni, nj));
        }

      // expand only if a non-zero neighbor was found
      out(i, j) = (vmin_non_zero != std::numeric_limits<float>::max())
                      ? vmin_non_zero
                      : 0.f;
    }

  return out;
}

Array erosion(const Array &array, int ir)
{
  return local_min(array, ir);
}

void flood_fill(Array &array,
                int    i,
                int    j,
                float  fill_value,
                float  background_value)
{
  if (array(i, j) != background_value) return; // nothing to do

  const glm::ivec2 &shape = array.shape;

  std::vector<glm::ivec2> queue;
  queue.reserve(shape.x * shape.y);

  // mark and enqueue seed
  array(i, j) = fill_value;
  queue.push_back({i, j});

  while (!queue.empty())
  {
    auto c = queue.back();
    queue.pop_back();

    // 4-connected neighbors
    constexpr int di[] = {-1, 1, 0, 0};
    constexpr int dj[] = {0, 0, -1, 1};

    for (int d = 0; d < 4; ++d)
    {
      const int ni = c.x + di[d];
      const int nj = c.y + dj[d];

      if (ni >= 0 && ni < shape.x && nj >= 0 && nj < shape.y &&
          array(ni, nj) == background_value)
      {
        array(ni, nj) = fill_value;
        queue.push_back({ni, nj});
      }
    }
  }
}

Array morphological_black_hat(const Array &array, int ir)
{
  return closing(array, ir) - array;
}

Array morphological_gradient(const Array &array, int ir)
{
  float vmin = array.min();
  return dilation(array - vmin, ir) - erosion(array - vmin, ir);
}

Array morphological_laplacian(const Array &array, int ir)
{
  float vmin = array.min();
  return dilation(array - vmin, ir) + erosion(array - vmin, ir) - 2.f * array;
}

Array morphological_top_hat(const Array &array, int ir)
{
  return array - opening(array, ir);
}

Array opening(const Array &array, int ir)
{
  return dilation(erosion(array, ir), ir);
}

Array opening_by_reconstruction(const Array &array, int ir, float k_smooth_min)
{
  Array marker = erosion(array, ir);
  return reconstruction_by_dilation(marker, array, ir, k_smooth_min);
}

Array helper_reconstruction_impl(
    const Array                                              &marker,
    const Array                                              &mask,
    int                                                       ir,
    float                                                     k_smooth,
    std::function<Array(const Array &, int)>                  morph_op,
    std::function<Array(const Array &, const Array &, float)> clamp_op)
{
  constexpr float tol = 1e-6f;
  Array           current = marker;
  Array           next;

  while (true)
  {
    next = morph_op(current, ir);
    next = clamp_op(next, mask, k_smooth);

    float diff = 0.f;
    for (int j = 0; j < current.shape.y; ++j)
      for (int i = 0; i < current.shape.x; ++i)
        diff = std::max(diff, std::abs(next(i, j) - current(i, j)));

    std::swap(current, next);

    if (diff < tol) break;
  }
  return current;
}

Array reconstruction_by_dilation(const Array &marker,
                                 const Array &mask,
                                 int          ir,
                                 float        k_smooth_min)
{
  constexpr float tol = 1e-6f;
  Array           current = marker;
  Array           next;

  while (true)
  {
    next = dilation(current, ir);
    next = minimum_smooth(next, mask, k_smooth_min);

    float diff = 0.f;
    for (int j = 0; j < current.shape.y; ++j)
      for (int i = 0; i < current.shape.x; ++i)
        diff = std::max(diff, std::abs(next(i, j) - current(i, j)));

    std::swap(current, next);
    if (diff < tol) break;
  }
  return current;
}

Array reconstruction_by_erosion(const Array &marker,
                                const Array &mask,
                                int          ir,
                                float        k_smooth_max)
{
  constexpr float tol = 1e-6f;
  Array           current = marker;
  Array           next;

  while (true)
  {
    next = erosion(current, ir);
    next = maximum_smooth(next, mask, k_smooth_max);

    float diff = 0.f;
    for (int j = 0; j < current.shape.y; ++j)
      for (int i = 0; i < current.shape.x; ++i)
        diff = std::max(diff, std::abs(next(i, j) - current(i, j)));

    std::swap(current, next);
    if (diff < tol) break;
  }
  return current;
}

// Array reconstruction_by_erosion(const Array &marker,
//                                 const Array &mask,
//                                 int          ir,
//                                 float        k_smooth_max)
// {
//   constexpr float tol = 1e-6f;

//   Array current = marker;
//   Array next;

//   while (true)
//   {
//     next = erosion(current, ir);

//     // clamp to mask
//     next = maximum_smooth(next, mask, k_smooth_max);

//     float diff = abs(next - current).max();

//     if (diff < tol) // convergence
//       break;

//     current = next;
//   }

//   return current;
// }

// helper

void helper_thinning(Array &in, int iter)
{
  Array marker(in.shape);

  for (int j = 1; j < in.shape.y - 1; j++)
    for (int i = 1; i < in.shape.x - 1; i++)
    {
      int a = (in(i - 1, j) == 0.f && in(i - 1, j + 1) == 1.f) +
              (in(i - 1, j + 1) == 0.f && in(i, j + 1) == 1.f) +
              (in(i, j + 1) == 0.f && in(i + 1, j + 1) == 1.f) +
              (in(i + 1, j + 1) == 0.f && in(i + 1, j) == 1.f) +
              (in(i + 1, j) == 0.f && in(i + 1, j - 1) == 1.f) +
              (in(i + 1, j - 1) == 0.f && in(i, j - 1) == 1.f) +
              (in(i, j - 1) == 0.f && in(i - 1, j - 1) == 1.f) +
              (in(i - 1, j - 1) == 0.f && in(i - 1, j) == 1.f);
      int b = in(i - 1, j) + in(i - 1, j + 1) + in(i, j + 1) +
              in(i + 1, j + 1) + in(i + 1, j) + in(i + 1, j - 1) +
              in(i, j - 1) + in(i - 1, j - 1);
      int m1 = iter == 0 ? (in(i - 1, j) * in(i, j + 1) * in(i + 1, j))
                         : (in(i - 1, j) * in(i, j + 1) * in(i, j - 1));
      int m2 = iter == 0 ? (in(i, j + 1) * in(i + 1, j) * in(i, j - 1))
                         : (in(i - 1, j) * in(i + 1, j) * in(i, j - 1));

      if (a == 1 && (b >= 2 && b <= 6) && m1 == 0 && m2 == 0)
        marker(i, j) = 1.f;
    }

  for (int j = 0; j < in.shape.y; j++)
    for (int i = 0; i < in.shape.x; i++)
      in(i, j) *= 1.f - marker(i, j);
}

Array relative_distance_from_skeleton(const Array &array,
                                      int          ir_search,
                                      bool         zero_at_borders,
                                      int          ir_erosion)
{
  const glm::ivec2 &shape = array.shape;

  Array border = array - erosion(array, ir_erosion);
  Array sk = skeleton(array, zero_at_borders);

  Array rdist(shape);

  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
      // only work for cells within the non-zero regions
      if (array(i, j) != 0.f)
      {
        // find the closest skeleton and border cells
        float dmax_sk = std::numeric_limits<float>::max();
        float dmax_bd = std::numeric_limits<float>::max();

        int p1 = std::max(i - ir_search, 0);
        int p2 = std::min(i + ir_search + 1, shape.x);
        int q1 = std::max(j - ir_search, 0);
        int q2 = std::min(j + ir_search + 1, shape.y);

        for (int q = q1; q < q2; q++)
          for (int p = p1; p < p2; p++)
          {
            // distance to skeleton
            if (sk(p, q) == 1.f)
            {
              float d2 = (float)((i - p) * (i - p) + (j - q) * (j - q));
              if (d2 < dmax_sk) dmax_sk = d2;
            }

            // distance to border
            if (border(p, q) == 1.f)
            {
              float d2 = (float)((i - p) * (i - p) + (j - q) * (j - q));
              if (d2 < dmax_bd) dmax_bd = d2;
            }
          }

        // relative distance (from 1.f on the skeleton to 0.f and
        // the border)
        float sum = dmax_bd + dmax_sk;
        if (sum) rdist(i, j) = dmax_bd / sum;
      }

  return rdist;
}

Array remove_endpoints(const Array &array,
                       int          iterations,
                       float        background_value)
{
  const glm::ivec2 &shape = array.shape;
  Array             wrk = array;
  Array             out(shape);

  for (int it = 0; it < iterations; ++it)
  {
    for (int j = 0; j < shape.y; ++j)
      for (int i = 0; i < shape.x; ++i)
      {
        // Skip background pixels immediately
        if (wrk(i, j) == background_value)
        {
          out(i, j) = background_value;
          continue;
        }

        int n = 0;
        for (int r = -1; r <= 1; ++r)
          for (int s = -1; s <= 1; ++s)
          {
            if (r == 0 && s == 0) continue;

            const int ni = i + r;
            const int nj = j + s;

            if (ni >= 0 && ni < shape.x && nj >= 0 && nj < shape.y &&
                wrk(ni, nj) != background_value)
              ++n;
          }

        // also remove "lonely" cells, with no surrounding pixels
        out(i, j) = (n < 2) ? background_value : wrk(i, j);
      }

    std::swap(wrk, out);
  }

  return wrk; // result is in wrk after the last swap
}

Array skeleton(const Array &array, bool zero_at_borders)
{
  // https://github.com/krishraghuram/Zhang-Suen-Skeletonization
  Array sk = generate_buffered_array(array, {1, 1, 1, 1});
  set_borders(sk, 0.f, 1);

  Array prev;
  Array diff;

  do
  {
    prev = sk;

    helper_thinning(sk, 0);
    helper_thinning(sk, 1);

    diff = sk - prev;

  } while (count_non_zero(diff) > 0);

  // remove padding
  sk = sk.extract_slice({1, sk.shape.x - 1, 1, sk.shape.y - 1});

  // set border to zero
  if (zero_at_borders) zeroed_borders(sk);

  return sk;
}

} // namespace hmap
