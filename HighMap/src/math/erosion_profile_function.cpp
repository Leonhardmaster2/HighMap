/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>
#include <functional>
#include <type_traits>
#include <vector>

#include "highmap/math/profiles.hpp"
#include "highmap/operator.hpp"

namespace hmap
{

float helper_wrap_phi(float phi)
{
  phi = std::fmod(phi + M_PI, 2.f * M_PI);

  if (phi < 0.f) phi += 2.f * M_PI;

  return phi - M_PI;
}

template <typename F> auto helper_make_wrapped(F f)
{
  return [f](float phi)
  {
    phi = helper_wrap_phi(phi);
    return f(phi);
  };
}

std::function<float(float)> get_erosion_profile_function(
    const ErosionProfile &erosion_profile,
    float                 delta,
    float                &profile_avg)
{
  std::function<float(float)> lambda_p;

  // phi always normalized in [-pi, pi)

  switch (erosion_profile)
  {
  case ErosionProfile::EP_COSINE:
    lambda_p = helper_make_wrapped([](float phi)
                                   { return 0.5f - 0.5f * std::cos(phi); });
    break;

  case ErosionProfile::EP_COSINE_BULK:
    lambda_p = helper_make_wrapped(
        [](float phi)
        { return 0.5f - 0.5f * std::cos(M_PI * std::pow(phi / M_PI, 2)); });
    break;

  case ErosionProfile::EP_COSINE_PEAK:
    lambda_p = helper_make_wrapped(
        [](float phi)
        {
          float t = phi / M_PI;
          return 0.5f * std::cos(M_PI * t * t) + 0.5f;
        });
    break;

  case ErosionProfile::EP_PARABOL:
    lambda_p = helper_make_wrapped([](float phi)
                                   { return std::pow(phi / M_PI, 2); });
    break;

  case ErosionProfile::EP_SAW_SHARP:
    lambda_p = helper_make_wrapped(
        [](float phi)
        {
          float t = phi / M_PI;
          return 0.5f * (t - std::floor(t)) + 0.5f;
        });
    break;

  case ErosionProfile::EP_SAW_SMOOTH:
  {
    float n = 1.f + 0.02f / delta;
    float dn = 2.f * n;

    float coeff = std::pow(1.f / dn, 1.f / dn);
    coeff = 1.f / coeff;

    lambda_p = helper_make_wrapped(
        [n, coeff](float phi)
        {
          float t = phi / M_PI;

          float a = std::abs(t);
          t = coeff * t * (1.f - std::pow(a, 2.f * n));

          return 0.5f * (1.f + t);
        });
  }
  break;

  case ErosionProfile::EP_SHARP_VALLEYS:
    lambda_p = helper_make_wrapped(
        [delta](float phi)
        {
          float t = phi / M_PI;
          return (1.f - t * t) / (1.f + t * t / delta);
        });
    break;

  case ErosionProfile::EP_SQRT:
    lambda_p = helper_make_wrapped(
        [](float phi) { return std::pow(std::abs(phi) / M_PI, 0.5f); });
    break;

  case ErosionProfile::EP_TRIANGLE_GRENIER:
    lambda_p = helper_make_wrapped(
        [delta](float phi)
        {
          float t = phi / M_PI;

          return std::sqrt((1.f + 2.f * std::sqrt(delta)) * t * t + delta) -
                 std::sqrt(delta);
        });
    break;

  case ErosionProfile::EP_TRIANGLE_SHARP:
    lambda_p = helper_make_wrapped(
        [](float phi)
        {
          float t = phi / M_PI;
          return 1.f - std::abs(t);
        });
    break;

  case ErosionProfile::EP_TRIANGLE_SMOOTH:
  {
    // https://mathematica.stackexchange.com/questions/38293

    float coeff = 0.5f / (std::acos(delta - 1.f) / M_PI - 0.5f);

    lambda_p = helper_make_wrapped(
        [delta, coeff](float phi)
        {
          return 0.5f + coeff * (std::acos((1.f - delta) *
                                           std::sin(phi + 0.5f * M_PI)) /
                                     M_PI -
                                 0.5f);
        });
  }
  break;
  }

  // average value

  profile_avg = 0.f;

  {
    int                nd = 50;
    std::vector<float> phi = linspace(-M_PI, M_PI, nd);

    for (auto &v : phi)
      profile_avg += lambda_p(v);

    profile_avg /= (float)nd;
  }

  return lambda_p;
}

} // namespace hmap
