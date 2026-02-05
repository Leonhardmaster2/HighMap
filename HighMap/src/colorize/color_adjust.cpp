/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <glm/glm.hpp>

#include "highmap/colorize.hpp"

namespace hmap
{

inline uint32_t helper_hash_u32(uint32_t x)
{
  x ^= x >> 16;
  x *= 0x7feb352d;
  x ^= x >> 15;
  x *= 0x846ca68b;
  x ^= x >> 16;
  return x;
}

inline float helper_hash_01(uint32_t x)
{
  return float(helper_hash_u32(x)) * (1.0f / float(0xffffffffu));
}

void color_adjust(VirtualTexture &tex, ColorAdjust param, const ComputeMode &cm)
{

  auto lambda = [&param](std::vector<Array *> &p, const TileRegion &region)
  {
    Array &r = *p[0];
    Array &g = *p[1];
    Array &b = *p[2];

    const float inv_gamma = 1.f / param.gamma;

    for (int j = 0; j < region.shape.y; ++j)
      for (int i = 0; i < region.shape.x; ++i)
      {
        glm::vec3 c{r(i, j), g(i, j), b(i, j)};

        // levels
        c = (c - param.in_min) / (param.in_max - param.in_min);
        c = glm::clamp(c, 0.0f, 1.0f);

        // exposure
        c *= std::exp2(param.exposure);

        // tonemap
        if (param.filmic_tonemap) c = c / (glm::vec3(1.0f) + c); // Reinhard

        if (param.aces_tonemap)
        {
          glm::vec3 a = c * (2.51f * c + 0.03f);
          glm::vec3 b = c * (2.43f * c + 0.59f) + 0.14f;
          c = a / b;
        }

        if (param.agx_tonemap)
        {
          // Scene-linear → log2
          glm::vec3 x = glm::max(c, glm::vec3(1e-6f));
          x = glm::log2(x);

          // Normalize log range (AGX working range)
          x = (x + 12.0f) / 14.0f;
          x = glm::clamp(x, 0.0f, 1.0f);

          // AGX contrast curve (smooth filmic sigmoid)
          x = x * x * (3.0f - 2.0f * x);

          // Back to linear
          c = glm::exp2(x * 14.0f - 12.0f);

          // Gamut / saturation compression (very important for AGX look)
          float luma = glm::dot(c, glm::vec3(0.2126f, 0.7152f, 0.0722f));
          c = glm::mix(glm::vec3(luma), c, 1.1f);
        }

        // contrast
        c = (c - 0.5f) * param.contrast + 0.5f;

        // saturation
        float l = glm::dot(c, glm::vec3(0.2126f, 0.7152f, 0.0722f));
        c = glm::mix(glm::vec3(l), c, param.saturation);

        // temperature
        c.r *= 1.0f + param.temperature;
        c.b *= 1.0f - param.temperature;

        // gamma
        c = glm::pow(c, glm::vec3(inv_gamma));

        // hash dithering
        int x = region.key.tx * region.shape.x + i;
        int y = region.key.ty * region.shape.y + j;

        float n = helper_hash_01(uint32_t(x) ^ (uint32_t(y) << 16));
        n = (n - 0.5f) * param.dither_amp;

        c += glm::vec3(n);

        // final clamp
        c = glm::clamp(c, 0.0f, 1.0f);

        r(i, j) = c.r;
        g(i, j) = c.g;
        b(i, j) = c.b;
      }
  };

  for_each_tile(tex, lambda, cm);
}

} // namespace hmap
