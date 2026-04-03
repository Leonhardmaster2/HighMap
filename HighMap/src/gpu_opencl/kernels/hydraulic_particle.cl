R""(
/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#define VELOCITY_INIT 1.f
#define VELOCITY_MIN 0.01f
#define VOLUME_MIN 0.001f
#define SLOPE_MIN 0.0001f
#define MAX_IT 5000

// --- HELPERS

float helper_bilinear_interp(const float f00,
                             const float f10,
                             const float f01,
                             const float f11,
                             const float u,
                             const float v)
{
  float a10 = f10 - f00;
  float a01 = f01 - f00;
  float a11 = f11 - f10 - f01 + f00;

  return f00 + a10 * u + a01 * v + a11 * u * v;
}

inline float helper_sample_height(global float *z,
                                  int           i,
                                  int           j,
                                  int           nx,
                                  int           ny,
                                  float         u,
                                  float         v)
{
  if (i == nx - 1) i--;
  if (j == ny - 1) j--;

  return helper_bilinear_interp(z[linear_index(i, j, nx)],
                                z[linear_index(i + 1, j, nx)],
                                z[linear_index(i, j + 1, nx)],
                                z[linear_index(i + 1, j + 1, nx)],
                                u,
                                v);
}

inline void helper_bilinear_deposition(global float *z,
                                       int           i,
                                       int           j,
                                       int           nx,
                                       int           ny,
                                       float         u,
                                       float         v,
                                       float         amount)
{
  if (i == nx - 1) i--;
  if (j == ny - 1) j--;

  float d1 = (1.f - u) * (1.f - v);
  float d2 = u * (1.f - v);
  float d3 = (1.f - u) * v;
  float d4 = u * v;

  z[linear_index(i, j, nx)] -= amount * d1;
  z[linear_index(i + 1, j, nx)] -= amount * d2;
  z[linear_index(i, j + 1, nx)] -= amount * d3;
  z[linear_index(i + 1, j + 1, nx)] -= amount * d4;
}

// --- MAIN KERNEL

void kernel hydraulic_particle(global float *z_in,
                               global float *bedrock,
                               global float *moisture_map,
                               global float *elevation_shift,
                               const int     nx,
                               const int     ny,
                               const int     nparticles,
                               const uint    seed,
                               const float   c_capacity,
                               const float   c_erosion,
                               const float   c_deposition,
                               const float   c_inertia,
                               const float   c_gravity,
                               const float   drag_rate,
                               const float   evap_rate,
                               const int     enable_directional_bias,
                               const float   angle_bias,
                               const int     has_bedrock,
                               const int     has_moisture_map,
                               const int     has_elevation_shift)
{
  float dt = 1.f;

  int id = get_global_id(0); // particle id
  if (id >= nparticles) return;

  uint   rng = pcg_hash(seed + id * 2u);
  float2 pos = {(nx - 2) * rand(&rng), (ny - 2) * rand(&rng)};
  int    i, j;
  float  u, v;

  update_interp_param(pos, &i, &j, &u, &v);

  float vel = VELOCITY_INIT;
  float volume = has_moisture_map > 0 ? moisture_map[linear_index(i, j, nx)]
                                      : 1.f;

  float2 dir = {0.f, 0.f};
  if (enable_directional_bias)
  {
    float alpha_bias = angle_bias / 180.f * 3.1459f;
    dir = (float2)(cos(alpha_bias), sin(alpha_bias));
  }

  float s = 0.f;
  float evap_factor = 1.f - evap_rate;
  int   count = 0;

  while (volume > 1e-3f && count < MAX_IT)
  {
    count++; // for pathological cases...

    update_interp_param(pos, &i, &j, &u, &v);

    // stop if the particle reaches the domain limits
    if (!is_inside_gap(i, j, nx, ny, 1)) break;

    float2 gz;

    // compute gradient
    {
      // --- Sobel with optional elevation_shift
      int idx00 = linear_index(i - 1, j - 1, nx);
      int idx01 = linear_index(i - 1, j, nx);
      int idx02 = linear_index(i - 1, j + 1, nx);

      int idx10 = linear_index(i, j - 1, nx);
      int idx12 = linear_index(i, j + 1, nx);

      int idx20 = linear_index(i + 1, j - 1, nx);
      int idx21 = linear_index(i + 1, j, nx);
      int idx22 = linear_index(i + 1, j + 1, nx);

      // load terrain
      float z00 = z_in[idx00];
      float z01 = z_in[idx01];
      float z02 = z_in[idx02];

      float z10 = z_in[idx10];
      float z12 = z_in[idx12];

      float z20 = z_in[idx20];
      float z21 = z_in[idx21];
      float z22 = z_in[idx22];

      // add elevation_shift if present
      if (has_elevation_shift)
      {
        z00 += elevation_shift[idx00];
        z01 += elevation_shift[idx01];
        z02 += elevation_shift[idx02];

        z10 += elevation_shift[idx10];
        z12 += elevation_shift[idx12];

        z20 += elevation_shift[idx20];
        z21 += elevation_shift[idx21];
        z22 += elevation_shift[idx22];
      }

      // compute Sobel gradient
      gz.x = -z00 - 2.f * z01 - z02 + z20 + 2.f * z21 + z22;
      gz.y = -z00 - 2.f * z10 - z20 + z02 + 2.f * z12 + z22;
      gz *= 0.125f;
    }

    // ensure minimum slope
    float gz_norm = length(gz);

    if (gz_norm < SLOPE_MIN)
    {
      if (gz_norm > 0.f)
        gz *= SLOPE_MIN / gz_norm;
      else
        gz = SLOPE_MIN * dir;
    }

    // particle goes downhill, opposite local gradient
    float2 new_dir = (float2)(-gz.x, -gz.y);
    dir = mix(new_dir, dir, c_inertia);

    if (length(dir))
      dir /= length(dir);
    else
      dir = (float2)(0.f, 0.f);

    if (vel < VELOCITY_MIN) break;

    float zp = helper_sample_height(z_in, i, j, nx, ny, u, v);

    // backup previous position
    int   ip = i;
    int   jp = j;
    float up = u;
    float vp = v;

    // move particle
    pos += dt * dir;

    // elevation at new position
    update_interp_param(pos, &i, &j, &u, &v);
    if (!is_inside(i, j, nx, ny)) break;

    float z = helper_sample_height(z_in, i, j, nx, ny, u, v);
    float dz = zp - z;
    float sc = max(c_capacity * volume * vel * dz, 0.f);
    float delta_sc = dt * (sc - s);
    float amount = 0.f;

    // if more sediments than capacity or if uphill motion
    if (delta_sc < 0.f || dz < 0.f)
    {
      // deposition
      amount = (dz < 0.f) ? -min(-dz, s) : c_deposition * delta_sc;
      helper_bilinear_deposition(z_in, ip, jp, nx, ny, up, vp, amount);
    }
    else
    {
      // erosion
      amount = c_erosion * delta_sc;
      helper_bilinear_deposition(z_in, ip, jp, nx, ny, up, vp, amount);
    }

    s += amount;

    // bedrock limit enforcement
    if (amount > 0.f && has_bedrock != 0) // erosion
    {
      z_in[linear_index(ip, jp, nx)] = max(bedrock[linear_index(ip, jp, nx)],
                                           z_in[linear_index(ip, jp, nx)]);
    }

    vel = sqrt(max(0.f, vel * vel - dz * c_gravity));
    volume *= evap_factor;
  }
}
)""
