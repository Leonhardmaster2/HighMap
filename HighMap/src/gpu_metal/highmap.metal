// Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
// General Public License. The full license is in the file LICENSE.

#include <metal_stdlib>

using namespace metal;

struct GridParams
{
  int nx;
  int ny;
};

struct BinarySmoothParams
{
  int nx;
  int ny;
  float k;
};

struct NoiseParams
{
  int nx;
  int ny;
  int noise_id;
  float kx;
  float ky;
  uint seed;
  int has_noise_x;
  int has_noise_y;
  int period_x;
  int period_y;
  float bbox0;
  float bbox1;
  float bbox2;
  float bbox3;
};

struct NoiseFbmParams
{
  int nx;
  int ny;
  int noise_id;
  float kx;
  float ky;
  uint seed;
  int octaves;
  float weight;
  float persistence;
  float lacunarity;
  int has_ctrl_param;
  int has_noise_x;
  int has_noise_y;
  int period_x;
  int period_y;
  float bbox0;
  float bbox1;
  float bbox2;
  float bbox3;
};

struct SmoothCpulseParams
{
  int nx;
  int ny;
  int ir;
  int pass;
  float weight_sum;
};

struct NormalizeParams
{
  int nx;
  int ny;
  float from_min;
  float from_max;
  float to_min;
  float to_max;
};

struct AdvectionParams
{
  int nx;
  int ny;
  float advection_length;
  float value_persistence;
};

struct ThermalParams
{
  int nx;
  int ny;
};

struct BorderParams
{
  int nx;
  int ny;
};

struct LinearCombineParams
{
  int nx;
  int ny;
  float weight1;
  float weight2;
};

struct HydraulicParams
{
  int nx;
  int ny;
  float dt;
  float water_height;
  float k_capacity;
  float k_erode;
  float k_depose;
  float k_discharge_exp;
  float downcutting_max_depth_ratio;
  int flux_diffusion;
  float flux_diffusion_strength;
  float evap_factor;
  float initial_water_volume;
  float rain_volume;
  int maintain_water_volume;
};

struct ReduceParams
{
  int count;
  int threads_per_group;
};

struct MinMaxParams
{
  int count;
  int threads_per_group;
};

inline uint index_at(int x, int y, int nx)
{
  return uint(y * nx + x);
}

inline float load_clamped(device const float *data,
                          int                 x,
                          int                 y,
                          int                 nx,
                          int                 ny)
{
  return data[index_at(clamp(x, 0, nx - 1), clamp(y, 0, ny - 1), nx)];
}

inline float mirror_unit(float x)
{
  float t = x - 2.f * floor(x * 0.5f);
  return t <= 1.f ? t : 2.f - t;
}

inline int mirror_index(int x, int n)
{
  if (n <= 1) return 0;
  int period = 2 * n;
  int value = x % period;
  if (value < 0) value += period;
  return value < n ? value : period - value - 1;
}

inline float sample_mirrored_linear(device const float *data,
                                    float2              normalized_pos,
                                    int                  nx,
                                    int                  ny)
{
  // OpenCL's normalized linear image sampler used by the legacy kernel maps
  // normalized coordinates directly into the image's unnormalized sample
  // space. Keep that convention here so an Array-backed Metal sampler has the
  // same half-pixel behavior as the existing OpenCL path.
  float x = mirror_unit(normalized_pos.x) * float(nx) - 0.5f;
  float y = mirror_unit(normalized_pos.y) * float(ny) - 0.5f;
  int x0 = int(floor(x));
  int y0 = int(floor(y));
  float tx = fract(x);
  float ty = fract(y);

  float v00 = data[index_at(mirror_index(x0, nx), mirror_index(y0, ny), nx)];
  float v10 = data[index_at(mirror_index(x0 + 1, nx), mirror_index(y0, ny), nx)];
  float v01 = data[index_at(mirror_index(x0, nx), mirror_index(y0 + 1, ny), nx)];
  float v11 = data[index_at(mirror_index(x0 + 1, nx),
                           mirror_index(y0 + 1, ny),
                           nx)];
  return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

inline uint wang_hash(uint seed)
{
  seed = (seed ^ 61u) ^ (seed >> 16u);
  seed *= 9u;
  seed ^= seed >> 4u;
  seed *= 0x27d4eb2du;
  seed ^= seed >> 15u;
  return seed;
}

inline float seeded_random(uint seed)
{
  uint state = wang_hash(seed);
  state ^= state << 13u;
  state ^= state >> 17u;
  state ^= state << 5u;
  return float(state) * (1.f / 4294967296.f);
}

inline float hash12(float2 p, float fseed)
{
  float value = sin(dot(p, float2(127.1f, 311.7f)) + fseed) * 43758.5453123f;
  return value - floor(value);
}

inline float2 gradient22(float2 p, float fseed)
{
  float angle = 6.2831853f * hash12(p, fseed);
  return float2(cos(angle), sin(angle));
}

inline float2 wrap_lattice(float2 p, int2 period)
{
  if (period.x > 0) p.x -= float(period.x) * floor(p.x / float(period.x));
  if (period.y > 0) p.y -= float(period.y) * floor(p.y / float(period.y));
  return p;
}

inline float perlin(float2 p, float fseed, int2 period)
{
  float2 cell = floor(p);
  float2 f = fract(p);
  float2 g00 = gradient22(wrap_lattice(cell, period), fseed);
  float2 g10 = gradient22(wrap_lattice(cell + float2(1.f, 0.f), period), fseed);
  float2 g01 = gradient22(wrap_lattice(cell + float2(0.f, 1.f), period), fseed);
  float2 g11 = gradient22(wrap_lattice(cell + float2(1.f, 1.f), period), fseed);

  float n00 = dot(g00, f);
  float n10 = dot(g10, f - float2(1.f, 0.f));
  float n01 = dot(g01, f - float2(0.f, 1.f));
  float n11 = dot(g11, f - float2(1.f, 1.f));
  float2 u = f * f * f * (f * (f * 6.f - 15.f) + 10.f);
  return 1.42857f * mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
}

inline float simplex2(float2 p, float fseed)
{
  const float k1 = 0.366025403f;
  const float k2 = 0.211324865f;
  float2 p2 = 0.5f * p;
  float2 cell = floor(p2 + dot(p2, float2(k1)));
  float2 a = p2 - cell + dot(cell, float2(k2));
  float m = step(a.y, a.x);
  float2 offset = float2(m, 1.f - m);
  float2 b = a - offset + float2(k2);
  float2 c = a - 1.f + 2.f * float2(k2);
  float n0 = max(0.5f - dot(a, a), 0.f);
  float n1 = max(0.5f - dot(b, b), 0.f);
  float n2 = max(0.5f - dot(c, c), 0.f);
  n0 = n0 * n0 * n0 * n0 * dot(gradient22(cell, fseed), a);
  n1 = n1 * n1 * n1 * n1 * dot(gradient22(cell + offset, fseed), b);
  n2 = n2 * n2 * n2 * n2 * dot(gradient22(cell + 1.f, fseed), c);
  return 100.f * (n0 + n1 + n2);
}

inline float value_noise(float2 p, float fseed, int2 period, bool linear)
{
  float2 cell = floor(p);
  float2 f = fract(p);
  float v00 = hash12(wrap_lattice(cell, period), fseed);
  float v10 = hash12(wrap_lattice(cell + float2(1.f, 0.f), period), fseed);
  float v01 = hash12(wrap_lattice(cell + float2(0.f, 1.f), period), fseed);
  float v11 = hash12(wrap_lattice(cell + float2(1.f, 1.f), period), fseed);
  float2 u = linear ? f : f * f * (3.f - 2.f * f);
  return 2.f * mix(mix(v00, v10, u.x), mix(v01, v11, u.x), u.y) - 1.f;
}

inline float max_smooth(float a, float b, float k)
{
  float h = max(k - fabs(a - b), 0.f) / k;
  return max(a, b) + h * h * h * k / 6.f;
}

inline float evaluate_noise(float2 p, int noise_id, float fseed, int2 period)
{
  if (noise_id == 1) return perlin(p, fseed, period);
  if (noise_id == 2) return 2.f * fabs(perlin(p, fseed, period)) - 1.f;
  if (noise_id == 3) return max_smooth(perlin(p, fseed, period), 0.f, 0.5f);
  if (noise_id == 4) return simplex2(p, fseed);
  if (noise_id == 6) return value_noise(p, fseed, period, false);
  if (noise_id == 9) return value_noise(p, fseed, period, true);
  return 0.f;
}

inline float evaluate_noise_fbm(float2       p,
                               int          noise_id,
                               float        fseed,
                               int2         period,
                               int          octaves,
                               float        weight,
                               float        persistence,
                               float        lacunarity)
{
  float n = 0.f;
  float nf = 1.f;
  float na = 0.6f;
  int2 per = period;
  for (int i = 0; i < octaves; ++i)
  {
    float v = evaluate_noise(p * nf, noise_id, fseed, per);
    n += v * na;
    na *= (1.f - weight) + weight * min(v + 1.f, 2.f) * 0.5f;
    na *= persistence;
    nf *= lacunarity;
    per = int2(per.x > 0 ? int(float(per.x) * lacunarity + 0.5f) : 0,
               per.y > 0 ? int(float(per.y) * lacunarity + 0.5f) : 0);
  }
  return n;
}

kernel void gradient_norm(device const float *array [[buffer(0)]],
                          device float       *g_norm [[buffer(1)]],
                          constant GridParams &p [[buffer(2)]],
                          uint2                gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  int x = int(gid.x);
  int y = int(gid.y);
  float dx = x == 0 ? load_clamped(array, 1, y, p.nx, p.ny) -
                          load_clamped(array, 0, y, p.nx, p.ny)
                    : x == p.nx - 1
                          ? load_clamped(array, x, y, p.nx, p.ny) -
                                load_clamped(array, x - 1, y, p.nx, p.ny)
                          : 0.5f * (load_clamped(array, x + 1, y, p.nx, p.ny) -
                                    load_clamped(array, x - 1, y, p.nx, p.ny));
  float dy = y == 0 ? load_clamped(array, x, 1, p.nx, p.ny) -
                          load_clamped(array, x, 0, p.nx, p.ny)
                    : y == p.ny - 1
                          ? load_clamped(array, x, y, p.nx, p.ny) -
                                load_clamped(array, x, y - 1, p.nx, p.ny)
                          : 0.5f * (load_clamped(array, x, y + 1, p.nx, p.ny) -
                                    load_clamped(array, x, y - 1, p.nx, p.ny));
  g_norm[index_at(x, y, p.nx)] = length(float2(dx, dy));
}

inline float smooth_maximum(float a, float b, float k)
{
  if (k <= 0.f) return max(a, b);
  float h = max(k - fabs(a - b), 0.f) / k;
  return max(a, b) + h * h * h * k / 6.f;
}

inline float smooth_minimum(float a, float b, float k)
{
  if (k <= 0.f) return min(a, b);
  float h = max(k - fabs(a - b), 0.f) / k;
  return min(a, b) - h * h * h * k / 6.f;
}

kernel void maximum_smooth(device const float *array1 [[buffer(0)]],
                           device const float *array2 [[buffer(1)]],
                           device float *output [[buffer(2)]],
                           constant BinarySmoothParams &p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  uint i = index_at(int(gid.x), int(gid.y), p.nx);
  output[i] = smooth_maximum(array1[i], array2[i], p.k);
}

kernel void minimum_smooth(device const float *array1 [[buffer(0)]],
                           device const float *array2 [[buffer(1)]],
                           device float *output [[buffer(2)]],
                           constant BinarySmoothParams &p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  uint i = index_at(int(gid.x), int(gid.y), p.nx);
  output[i] = smooth_minimum(array1[i], array2[i], p.k);
}

kernel void noise(device float       *output [[buffer(0)]],
                  device const float *noise_x [[buffer(1)]],
                  device const float *noise_y [[buffer(2)]],
                  constant NoiseParams &p [[buffer(3)]],
                  uint2                gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  uint index = index_at(int(gid.x), int(gid.y), p.nx);
  float fseed = seeded_random(p.seed);
  float dx = p.has_noise_x ? noise_x[index] : 0.f;
  float dy = p.has_noise_y ? noise_y[index] : 0.f;
  float x = p.kx * (float(gid.x) / float(p.nx) * (p.bbox1 - p.bbox0) + p.bbox0) + p.kx * dx;
  float y = p.ky * (float(gid.y) / float(p.ny) * (p.bbox3 - p.bbox2) + p.bbox2) + p.ky * dy;
  output[index] = evaluate_noise(float2(x, y),
                                 p.noise_id,
                                 fseed,
                                 int2(p.period_x, p.period_y));
}

kernel void noise_fbm(device float       *output [[buffer(0)]],
                      device const float *ctrl_param [[buffer(1)]],
                      device const float *noise_x [[buffer(2)]],
                      device const float *noise_y [[buffer(3)]],
                      constant NoiseFbmParams &p [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  uint index = index_at(int(gid.x), int(gid.y), p.nx);
  float fseed = seeded_random(p.seed);
  float ct = p.has_ctrl_param ? ctrl_param[index] : 1.f;
  float dx = p.has_noise_x ? noise_x[index] : 0.f;
  float dy = p.has_noise_y ? noise_y[index] : 0.f;
  float x = p.kx * (float(gid.x) / float(p.nx) * (p.bbox1 - p.bbox0) + p.bbox0) +
            p.kx * dx;
  float y = p.ky * (float(gid.y) / float(p.ny) * (p.bbox3 - p.bbox2) + p.bbox2) +
            p.ky * dy;
  float new_weight = (1.f - ct) + ct * p.weight;
  output[index] = evaluate_noise_fbm(float2(x, y),
                                     p.noise_id,
                                     fseed,
                                     int2(p.period_x, p.period_y),
                                     p.octaves,
                                     new_weight,
                                     p.persistence,
                                     p.lacunarity);
}

kernel void smooth_cpulse(device const float *input [[buffer(0)]],
                          device float       *output [[buffer(1)]],
                          constant SmoothCpulseParams &p [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  float value = 0.f;
  for (int k = -p.ir; k <= p.ir; ++k)
  {
    int x = p.pass == 0 ? int(gid.x) + k : int(gid.x);
    int y = p.pass == 0 ? int(gid.y) : int(gid.y) + k;
    x = clamp(x, 0, p.nx - 1);
    y = clamp(y, 0, p.ny - 1);
    float d = float(abs(k)) / float(p.ir);
    float w = exp(-0.5f * d * d * 9.f) / p.weight_sum;
    value += input[index_at(x, y, p.nx)] * w;
  }
  output[index_at(int(gid.x), int(gid.y), p.nx)] = value;
}

kernel void normalize(device const float *input [[buffer(0)]],
                      device float       *output [[buffer(1)]],
                      constant NormalizeParams &p [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  uint i = index_at(int(gid.x), int(gid.y), p.nx);
  if (p.from_min == p.from_max)
    output[i] = p.to_min;
  else
    output[i] = p.to_min + (input[i] - p.from_min) *
                            (p.to_max - p.to_min) /
                            (p.from_max - p.from_min);
}

kernel void advection_warp(device const float *z [[buffer(0)]],
                            device const float *field [[buffer(1)]],
                            device const float *dx [[buffer(2)]],
                            device const float *dy [[buffer(3)]],
                            device const float *mask [[buffer(4)]],
                            device float *output [[buffer(5)]],
                            constant AdvectionParams &p [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  constexpr uint max_steps = 1024u;
  float2 position = (float2(gid) + 0.5f) / float2(p.nx, p.ny);
  float2 path[max_steps];
  uint path_length = 0u;
  float z_previous = sample_mirrored_linear(z, position, p.nx, p.ny);
  float2 direction = float2(0.f);
  float drx = 1.f / float(p.nx);
  float dry = 1.f / float(p.ny);
  uint limit = min(max_steps, uint(ceil(p.advection_length / min(drx, dry))));

  for (uint step = 0u; step < limit; ++step) {
    path[path_length++] = position;
    float2 velocity = float2(sample_mirrored_linear(dx, position, p.nx, p.ny),
                              sample_mirrored_linear(dy, position, p.nx, p.ny));
    float length = metal::length(velocity);
    if (length > 0.f) direction = velocity / length;
    position += float2(direction.x * drx, direction.y * dry);
    if (position.x < 0.f || position.x > 1.f || position.y < 0.f || position.y > 1.f) break;
    float z_current = sample_mirrored_linear(z, position, p.nx, p.ny);
    if (z_current - z_previous < -1e-5f) break;
    z_previous = z_current;
  }

  float value = path_length > 0u
                    ? sample_mirrored_linear(field, path[path_length - 1u], p.nx, p.ny)
                    : 0.f;
  for (int step = int(path_length) - 1; step >= 0; --step) {
    float new_value = sample_mirrored_linear(field, path[step], p.nx, p.ny);
    float ratio = sample_mirrored_linear(mask, path[step], p.nx, p.ny) * p.value_persistence;
    value = ratio * value + (1.f - ratio) * new_value;
  }
  output[index_at(int(gid.x), int(gid.y), p.nx)] = value;
}

// Texture-backed advection is intentionally an experiment rather than a
// replacement for the buffer path. DeviceArray owns buffers; the host-side
// session performs explicit buffer/texture blits around this kernel so their
// cost remains measurable.
kernel void advection_warp_texture(
    texture2d<float, access::sample> z [[texture(0)]],
    texture2d<float, access::sample> field [[texture(1)]],
    texture2d<float, access::sample> dx [[texture(2)]],
    texture2d<float, access::sample> dy [[texture(3)]],
    texture2d<float, access::sample> mask [[texture(4)]],
    texture2d<float, access::write> output [[texture(5)]],
    constant AdvectionParams       &p [[buffer(0)]],
    uint2                           gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  constexpr sampler sampler(coord::normalized,
                            address::mirrored_repeat,
                            filter::linear);
  constexpr uint max_steps = 1024u;
  float2 position = (float2(gid) + 0.5f) / float2(p.nx, p.ny);
  float2 path[max_steps];
  uint path_length = 0u;
  float z_previous = z.sample(sampler, position).r;
  float2 direction = float2(0.f);
  float drx = 1.f / float(p.nx);
  float dry = 1.f / float(p.ny);
  uint limit = min(max_steps, uint(ceil(p.advection_length / min(drx, dry))));

  for (uint step = 0u; step < limit; ++step) {
    path[path_length++] = position;
    float2 velocity = float2(dx.sample(sampler, position).r,
                              dy.sample(sampler, position).r);
    float length = metal::length(velocity);
    if (length > 0.f) direction = velocity / length;
    position += float2(direction.x * drx, direction.y * dry);
    if (position.x < 0.f || position.x > 1.f ||
        position.y < 0.f || position.y > 1.f) break;
    float z_current = z.sample(sampler, position).r;
    if (z_current - z_previous < -1e-5f) break;
    z_previous = z_current;
  }

  float value = path_length > 0u
                    ? field.sample(sampler, path[path_length - 1u]).r
                    : 0.f;
  for (int step = int(path_length) - 1; step >= 0; --step) {
    float new_value = field.sample(sampler, path[step]).r;
    float ratio = mask.sample(sampler, path[step]).r * p.value_persistence;
    value = ratio * value + (1.f - ratio) * new_value;
  }
  output.write(float4(value), uint2(gid.x, gid.y));
}

inline float thermal_exchange(float self, float other, float distance, float talus)
{
  float max_difference = distance * talus;
  float rate = 0.2f;
  if (self > other)
    return self - other > max_difference
               ? -rate * ((self - other) - max_difference) / distance
               : 0.f;
  return other - self > max_difference
             ? rate * ((other - self) - max_difference) / distance
             : 0.f;
}

kernel void thermal_pass(device const float *z_in [[buffer(0)]],
                         device const float *talus [[buffer(1)]],
                         device float       *z_out [[buffer(2)]],
                         constant ThermalParams &p [[buffer(3)]],
                         uint2                gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  int x = int(gid.x);
  int y = int(gid.y);
  uint index = index_at(x, y, p.nx);
  if (x == 0 || x == p.nx - 1 || y == 0 || y == p.ny - 1)
  {
    z_out[index] = z_in[index];
    return;
  }

  constexpr float diagonal = 1.414f;
  const int dx[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
  const int dy[8] = {0, 1, -1, 0, -1, 1, -1, 1};
  const float distance[8] = {1.f, 1.f, 1.f, 1.f, diagonal, diagonal, diagonal, diagonal};
  float value = z_in[index];
  float amount = 0.f;
  for (uint k = 0; k < 8; ++k)
    amount += thermal_exchange(value,
                               load_clamped(z_in, x + dx[k], y + dy[k], p.nx, p.ny),
                               distance[k],
                               talus[index]);
  z_out[index] = value + amount;
}

kernel void thermal_ridge_pass(device const float *z_in [[buffer(0)]],
                               device const float *talus [[buffer(1)]],
                               device float       *z_out [[buffer(2)]],
                               constant ThermalParams &p [[buffer(3)]],
                               uint2                gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  int x = int(gid.x);
  int y = int(gid.y);
  uint index = index_at(x, y, p.nx);
  if (x == 0 || x == p.nx - 1 || y == 0 || y == p.ny - 1)
  {
    z_out[index] = z_in[index];
    return;
  }

  constexpr float diagonal = 1.414f;
  const int dx[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
  const int dy[8] = {0, 1, -1, 0, -1, 1, -1, 1};
  const float distance[8] = {1.f, 1.f, 1.f, 1.f, diagonal, diagonal, diagonal, diagonal};
  const float value = z_in[index];
  float sum = 0.f;
  float slope_max = 0.f;
  for (uint k = 0; k < 8; ++k)
  {
    float dz = (value - load_clamped(z_in, x + dx[k], y + dy[k], p.nx, p.ny)) /
               distance[k];
    if (dz > 0.f) sum += dz;
    slope_max = max(slope_max, fabs(dz));
  }

  const float t = talus[index];
  float amp = slope_max > 0.f ? clamp(1.f - t / slope_max, 0.f, 1.f) : 0.f;
  amp = amp * amp * (3.f - 2.f * amp);
  z_out[index] = value + 0.25f * (t - 0.5f * sum) * amp;
}

kernel void extrapolate_horizontal(device const float *z_in [[buffer(0)]],
                                   device float       *z_out [[buffer(1)]],
                                   constant BorderParams &p [[buffer(2)]],
                                   uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  const int x = int(gid.x);
  const int y = int(gid.y);
  const uint index = index_at(x, y, p.nx);
  float value = z_in[index];
  if (x == 0)
    value = 2.f * load_clamped(z_in, 1, y, p.nx, p.ny) -
            load_clamped(z_in, 2, y, p.nx, p.ny);
  else if (x == p.nx - 1)
    value = 2.f * load_clamped(z_in, p.nx - 2, y, p.nx, p.ny) -
            load_clamped(z_in, p.nx - 3, y, p.nx, p.ny);
  z_out[index] = value;
}

kernel void extrapolate_vertical(device const float *z_in [[buffer(0)]],
                                 device float       *z_out [[buffer(1)]],
                                 constant BorderParams &p [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  const int x = int(gid.x);
  const int y = int(gid.y);
  const uint index = index_at(x, y, p.nx);
  float value = z_in[index];
  if (y == 0)
    value = 2.f * load_clamped(z_in, x, 1, p.nx, p.ny) -
            load_clamped(z_in, x, 2, p.nx, p.ny);
  else if (y == p.ny - 1)
    value = 2.f * load_clamped(z_in, x, p.ny - 2, p.nx, p.ny) -
            load_clamped(z_in, x, p.ny - 3, p.nx, p.ny);
  z_out[index] = value;
}

kernel void linear_combine(device const float *array1 [[buffer(0)]],
                           device const float *array2 [[buffer(1)]],
                           device float       *output [[buffer(2)]],
                           constant LinearCombineParams &p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  uint i = index_at(int(gid.x), int(gid.y), p.nx);
  output[i] = p.weight1 * array1[i] + p.weight2 * array2[i];
}

kernel void hydraulic_flow_pass(device const float *z [[buffer(0)]],
                                device const float *fl [[buffer(1)]],
                                device const float *fr [[buffer(2)]],
                                device const float *ft [[buffer(3)]],
                                device const float *fb [[buffer(4)]],
                                device const float *d1 [[buffer(5)]],
                                device float *fl_out [[buffer(6)]],
                                device float *fr_out [[buffer(7)]],
                                device float *ft_out [[buffer(8)]],
                                device float *fb_out [[buffer(9)]],
                                constant HydraulicParams &p [[buffer(10)]],
                                uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  int x = int(gid.x);
  int y = int(gid.y);
  uint i = index_at(x, y, p.nx);
  float h0 = z[i] + d1[i];
  float fl_new = max(0.f, fl[i] + p.dt * (h0 - load_clamped(z, x - 1, y, p.nx, p.ny) - load_clamped(d1, x - 1, y, p.nx, p.ny)));
  float fr_new = max(0.f, fr[i] + p.dt * (h0 - load_clamped(z, x + 1, y, p.nx, p.ny) - load_clamped(d1, x + 1, y, p.nx, p.ny)));
  float ft_new = max(0.f, ft[i] + p.dt * (h0 - load_clamped(z, x, y + 1, p.nx, p.ny) - load_clamped(d1, x, y + 1, p.nx, p.ny)));
  float fb_new = max(0.f, fb[i] + p.dt * (h0 - load_clamped(z, x, y - 1, p.nx, p.ny) - load_clamped(d1, x, y - 1, p.nx, p.ny)));
  if (p.flux_diffusion) {
    float diffusion = p.flux_diffusion_strength * (fl_new + fr_new + ft_new + fb_new);
    float retained = 1.f - 4.f * p.flux_diffusion_strength;
    fl_new = retained * fl_new + diffusion;
    fr_new = retained * fr_new + diffusion;
    ft_new = retained * ft_new + diffusion;
    fb_new = retained * fb_new + diffusion;
  }
  float sum = fl_new + fr_new + ft_new + fb_new;
  float scale = sum > 1e-5f ? clamp(d1[i] / (sum * p.dt), 0.f, 1.f) : 0.f;
  fl_out[i] = fl_new * scale;
  fr_out[i] = fr_new * scale;
  ft_out[i] = ft_new * scale;
  fb_out[i] = fb_new * scale;
}

kernel void hydraulic_water_pass(device const float *z [[buffer(0)]],
                                 device const float *fl [[buffer(1)]],
                                 device const float *fr [[buffer(2)]],
                                 device const float *ft [[buffer(3)]],
                                 device const float *fb [[buffer(4)]],
                                 device const float *d1 [[buffer(5)]],
                                 device float *d2 [[buffer(6)]],
                                 device float *u [[buffer(7)]],
                                 device float *v [[buffer(8)]],
                                 constant HydraulicParams &p [[buffer(9)]],
                                 uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  int x = int(gid.x);
  int y = int(gid.y);
  uint i = index_at(x, y, p.nx);
  float delta = p.dt * (load_clamped(fr, x - 1, y, p.nx, p.ny) +
                        load_clamped(ft, x, y - 1, p.nx, p.ny) +
                        load_clamped(fl, x + 1, y, p.nx, p.ny) +
                        load_clamped(fb, x, y + 1, p.nx, p.ny) -
                        fl[i] - fr[i] - ft[i] - fb[i]);
  float depth = max(0.f, d1[i] + delta);
  d2[i] = depth;
  u[i] = 0.5f * (load_clamped(fr, x - 1, y, p.nx, p.ny) - fl[i] + fr[i] - load_clamped(fl, x + 1, y, p.nx, p.ny)) /
         max(0.001f * p.water_height, depth);
  v[i] = 0.5f * (load_clamped(ft, x, y - 1, p.nx, p.ny) - fb[i] + ft[i] - load_clamped(fb, x, y + 1, p.nx, p.ny)) /
         max(0.001f * p.water_height, depth);
}

kernel void hydraulic_erosion_pass(device const float *z [[buffer(0)]],
                                   device const float *d2 [[buffer(1)]],
                                   device const float *u [[buffer(2)]],
                                   device const float *v [[buffer(3)]],
                                   device const float *s [[buffer(4)]],
                                   device float *z_out [[buffer(5)]],
                                   device float *s_out [[buffer(6)]],
                                   constant HydraulicParams &p [[buffer(7)]],
                                   uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  int x = int(gid.x);
  int y = int(gid.y);
  uint i = index_at(x, y, p.nx);
  float dzx = 0.5f * (load_clamped(z, x + 1, y, p.nx, p.ny) - load_clamped(z, x - 1, y, p.nx, p.ny) +
                      load_clamped(s, x + 1, y, p.nx, p.ny) - load_clamped(s, x - 1, y, p.nx, p.ny));
  float dzy = 0.5f * (load_clamped(z, x, y + 1, p.nx, p.ny) - load_clamped(z, x, y - 1, p.nx, p.ny) +
                      load_clamped(s, x, y + 1, p.nx, p.ny) - load_clamped(s, x, y - 1, p.nx, p.ny));
  float talus = length(float2(dzx, dzy));
  float dzn = max(1e-3f, float(p.nx) * talus);
  float salpha = max(0.01f, dzn / length(float2(1.f, dzn)));
  float speed = length(float2(u[i], v[i]));
  float depth = min(d2[i] / p.water_height, p.downcutting_max_depth_ratio);
  float capacity = p.k_capacity * pow(depth * speed, p.k_discharge_exp) * salpha;
  float sediment = s[i];
  float terrain = z[i];
  if (capacity > sediment) {
    float amount = p.k_erode * (capacity - sediment);
    z_out[i] = terrain - amount;
    s_out[i] = sediment + amount;
  } else {
    float amount = p.k_depose * (sediment - capacity);
    z_out[i] = terrain + amount;
    s_out[i] = sediment - amount;
  }
}

kernel void hydraulic_sediment_pass(device const float *u [[buffer(0)]],
                                    device const float *v [[buffer(1)]],
                                    device const float *s_in [[buffer(2)]],
                                    device float *s_out [[buffer(3)]],
                                    constant HydraulicParams &p [[buffer(4)]],
                                    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  int x = int(gid.x);
  int y = int(gid.y);
  float2 position = float2(float(x) + 0.5f - p.dt * u[index_at(x, y, p.nx)],
                           float(y) + 0.5f - p.dt * v[index_at(x, y, p.nx)]);
  float2 normalized = position / float2(p.nx, p.ny);
  s_out[index_at(x, y, p.nx)] = sample_mirrored_linear(s_in, normalized, p.nx, p.ny);
}

kernel void hydraulic_evaporate(device const float *d2 [[buffer(0)]],
                                device float *d_out [[buffer(1)]],
                                constant HydraulicParams &p [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  uint i = index_at(int(gid.x), int(gid.y), p.nx);
  float value = max(0.f, d2[i] * p.evap_factor);
  d_out[i] = value;
}

kernel void hydraulic_sum_tiles(device const float *values [[buffer(0)]],
                                device float *partials [[buffer(1)]],
                                constant ReduceParams &p [[buffer(2)]],
                                uint gid [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint group [[threadgroup_position_in_grid]])
{
  threadgroup float reduction_shared[256];
  reduction_shared[lid] = gid < uint(p.count) ? values[gid] : 0.f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = uint(p.threads_per_group) / 2u; stride > 0u; stride /= 2u)
  {
    if (lid < stride) reduction_shared[lid] += reduction_shared[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0u) partials[group] = reduction_shared[0];
}

kernel void hydraulic_sum_reduce(device const float *values [[buffer(0)]],
                                 device float *partials [[buffer(1)]],
                                 constant ReduceParams &p [[buffer(2)]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint group [[threadgroup_position_in_grid]])
{
  threadgroup float reduction_shared[256];
  const uint input_index = group * uint(p.threads_per_group) + lid;
  reduction_shared[lid] = input_index < uint(p.count) ? values[input_index] : 0.f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = uint(p.threads_per_group) / 2u; stride > 0u; stride /= 2u)
  {
    if (lid < stride) reduction_shared[lid] += reduction_shared[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0u) partials[group] = reduction_shared[0];
}

kernel void minmax_tiles(device const float *values [[buffer(0)]],
                         device float2      *partials [[buffer(1)]],
                         constant MinMaxParams &p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint group [[threadgroup_position_in_grid]])
{
  threadgroup float2 reduction_shared[256];
  reduction_shared[lid] = gid < uint(p.count)
                              ? float2(values[gid], values[gid])
                              : float2(INFINITY, -INFINITY);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = uint(p.threads_per_group) / 2u; stride > 0u; stride /= 2u)
  {
    if (lid < stride)
      reduction_shared[lid] = float2(min(reduction_shared[lid].x,
                                        reduction_shared[lid + stride].x),
                                    max(reduction_shared[lid].y,
                                        reduction_shared[lid + stride].y));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) partials[group] = reduction_shared[0];
}

kernel void minmax_reduce(device const float2 *values [[buffer(0)]],
                          device float2       *partials [[buffer(1)]],
                          constant MinMaxParams &p [[buffer(2)]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]])
{
  threadgroup float2 reduction_shared[256];
  uint input_index = group * uint(p.threads_per_group) + lid;
  reduction_shared[lid] = input_index < uint(p.count)
                              ? values[input_index]
                              : float2(INFINITY, -INFINITY);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = uint(p.threads_per_group) / 2u; stride > 0u; stride /= 2u)
  {
    if (lid < stride)
      reduction_shared[lid] = float2(min(reduction_shared[lid].x,
                                        reduction_shared[lid + stride].x),
                                    max(reduction_shared[lid].y,
                                        reduction_shared[lid + stride].y));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) partials[group] = reduction_shared[0];
}

kernel void hydraulic_rescale(device float *d [[buffer(0)]],
                              device const float *rain [[buffer(1)]],
                              device const float *sum [[buffer(2)]],
                              constant HydraulicParams &p [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(p.nx) || gid.y >= uint(p.ny)) return;
  uint i = index_at(int(gid.x), int(gid.y), p.nx);
  float current = sum[0];
  float correction = p.rain_volume > 0.f ? (p.initial_water_volume - current) / p.rain_volume : 0.f;
  d[i] += correction * rain[i];
}
