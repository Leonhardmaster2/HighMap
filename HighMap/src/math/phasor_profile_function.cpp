/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "highmap/math/profiles.hpp"
#include "highmap/operator.hpp"

namespace hmap
{

std::function<float(float)> get_phasor_profile_function(
    PhasorProfile phasor_profile,
    float         delta,
    float        *p_profile_avg)
{

  std::function<float(float)> fct;

  // phi in [-pi, pi]

  switch (phasor_profile)
  {

  case PhasorProfile::PP_COSINE_BULKY:
    fct = [](float phi)
    {
      float t = phi / M_PI;
      t = std::fmod(t + 2.f, 2.f) - 1.f;
      float v = std::cos(M_PI * t * t);
      return v;
    };
    break;

  case PhasorProfile::PP_COSINE_PEAKY:
    fct = [](float phi)
    {
      float t = phi / M_PI;
      t = std::fmod(t + 2.f, 2.f) - 1.f;
      t = std::abs(t);
      float v = -std::cos(M_PI * (t - 1.f) * (t - 1.f));
      return v;
    };
    break;

  case PhasorProfile::PP_COSINE_SQUARE:
    fct = [delta](float phi)
    {
      float t = phi / M_PI;
      t = std::fmod(t + 2.f, 2.f) - 1.f;
      t = std::abs(t);

      float tn = std::pow(t, 1.f + delta);
      t = tn / (tn + std::pow(1.f - t, 1.f + delta));

      return std::cos(M_PI * t);
    };
    break;

  case PhasorProfile::PP_COSINE_STD:
    fct = [](float phi) { return -std::cos(phi); };
    break;

  case PhasorProfile::PP_TRIANGLE:
    fct = [](float phi)
    {
      float t = phi / M_PI;
      t = 4.f * std::abs(0.5f * t - std::floor(0.5f * t + 0.5f)) - 1.f;
      return t;
    };
    break;

  case PhasorProfile::PP_DUNE:
    fct = [delta](float phi)
    {
      float tc = std::clamp(delta, 0.f, 1.f);
      float t = 0.5f * phi / M_PI + 0.5f; // in [0, 1]
      float v;

      if (t < tc) // dune top position
      {
        float r = t / tc;
        v = r * r * (3.f - 2.f * r);
      }
      else
      {
        float r = (t - 1.f) / (tc - 1.f);
        v = r * r * (2.f - r);
      }

      return 2.f * v - 1.f; // in [-1, 1]
    };

    break;

  default:
    throw std::invalid_argument(
        "Invalid phasor profile request in hmap::get_phasor_profile_function");
  }

  if (p_profile_avg)
  {
    *p_profile_avg = 0.f;
    int                nd = 50;
    std::vector<float> phi = linspace(-M_PI, M_PI, nd);
    for (auto &v : phi)
      *p_profile_avg += fct(v);
    *p_profile_avg /= (float)nd;
  }

  return fct;
}

} // namespace hmap
