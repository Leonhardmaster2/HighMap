/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cmath>

#include "highmap/array.hpp"
#include "highmap/internal/validation.hpp"

namespace hmap
{

Array harmonic_interpolation(const Array &array,
                             const Array &mask_fixed_values,
                             int          iterations_max,
                             float        tolerance,
                             float        omega)
{
  if (!validate_non_empty(array)) return Array();
  if (!validate_same_shape(array, mask_fixed_values)) return Array();

  Array out = array;
  int   nx = out.shape.x;
  int   ny = out.shape.y;

  if (omega <= 0.f)
  {
    float pi = static_cast<float>(M_PI);
    float rho = 0.5f * (std::cos(pi / static_cast<float>(nx)) +
                        std::cos(pi / static_cast<float>(ny)));
    omega = 2.f / (1.f + std::sqrt(std::max(0.f, 1.f - rho * rho)));
  }

  for (int it = 0; it < iterations_max; ++it)
  {
    float max_diff = 0.f;

    // sweep over interior
    for (int j = 1; j < ny - 1; ++j)
      for (int i = 1; i < nx - 1; ++i)
      {
        if (mask_fixed_values(i, j) > 0.f) continue;

        float new_val = 0.25f * (out(i - 1, j) + out(i + 1, j) + out(i, j - 1) +
                                 out(i, j + 1));
        float diff = new_val - out(i, j);
        out(i, j) += omega * diff; // SOR update
        max_diff = std::max(max_diff, std::abs(diff));
      }

    if (max_diff < tolerance) break;
  }

  return out;
}

Array harmonic_interpolation(const Array &array,
                             const Array &mask_fixed_values,
                             const Array &dx,
                             const Array &dy,
                             int          iterations_max,
                             float        tolerance,
                             float        omega)
{
  if (!validate_non_empty(array)) return Array();
  if (!validate_same_shape(array, mask_fixed_values)) return Array();
  if (!validate_same_shape(array, dx)) return Array();
  if (!validate_same_shape(array, dy)) return Array();

  Array out = array;
  int   nx = out.shape.x;
  int   ny = out.shape.y;

  if (omega <= 0.f)
  {
    float pi = static_cast<float>(M_PI);
    float rho = 0.5f * (std::cos(pi / static_cast<float>(nx)) +
                        std::cos(pi / static_cast<float>(ny)));
    omega = 2.f / (1.f + std::sqrt(std::max(0.f, 1.f - rho * rho)));
  }

  for (int it = 0; it < iterations_max; ++it)
  {
    float max_diff = 0.f;

    // sweep over interior
    for (int j = 1; j < ny - 1; ++j)
      for (int i = 1; i < nx - 1; ++i)
      {
        if (mask_fixed_values(i, j) > 0.f) continue;

        float dx_c = dx(i, j);
        float dy_c = dy(i, j);

        float k_w = 0.5f * (dx(i - 1, j) + dx_c);
        float k_e = 0.5f * (dx(i + 1, j) + dx_c);
        float k_s = 0.5f * (dy(i, j - 1) + dy_c);
        float k_n = 0.5f * (dy(i, j + 1) + dy_c);

        float c_tot = k_w + k_e + k_s + k_n;
        if (c_tot < 1e-8f) continue;

        float u_star = (k_w * out(i - 1, j) + k_e * out(i + 1, j) +
                        k_s * out(i, j - 1) + k_n * out(i, j + 1)) /
                       c_tot;

        float diff = u_star - out(i, j);
        out(i, j) += omega * diff; // SOR update
        max_diff = std::max(max_diff, std::abs(diff));
      }

    if (max_diff < tolerance) break;
  }

  return out;
}

} // namespace hmap
