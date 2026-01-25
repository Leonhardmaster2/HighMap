R""(
/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#define VELOCITY_MIN 0.005f
#define VOLUME_MIN 0.001f

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
                                  float         u,
                                  float         v)
{
  return helper_bilinear_interp(z[linear_index(i, j, nx)],
                                z[linear_index(i + 1, j, nx)],
                                z[linear_index(i, j + 1, nx)],
                                z[linear_index(i + 1, j + 1, nx)],
                                u,
                                v);
}

inline float2 helper_compute_gradient(global float *z, int i, int j, int nx)
{
  float f_p1_0 = z[linear_index(i + 1, j, nx)];
  float f_m1_0 = z[linear_index(i - 1, j, nx)];
  float f_0_p1 = z[linear_index(i, j + 1, nx)];
  float f_0_m1 = z[linear_index(i, j - 1, nx)];

  return (float2)(0.5f * (f_p1_0 - f_m1_0), 0.5f * (f_0_p1 - f_0_m1));
}

// Smoothed radial erosion kernel
inline void helper_radial_kernel_deposition(global float *z,
                                            int           i,
                                            int           j,
                                            int           nx,
                                            int           ny,
                                            float         amount,
                                            int           radius)
{
  float dmax = (float)radius;
  float sum = 0.f;

  // normalization
  for (int p = -radius; p <= radius; ++p)
    for (int q = -radius; q <= radius; ++q)
      if (is_inside(i + p, j + q, nx, ny))
        sum += max(0.f, 1.f - length((float2)(p, q)) / dmax);

  // apply
  for (int p = -radius; p <= radius; ++p)
    for (int q = -radius; q <= radius; ++q)
    {
      float w = max(0.f, 1.f - length((float2)(p, q)) / dmax) / sum;
      int   in = i + p;
      int   jn = j + q;
      if (w > 0.f && is_inside(in, jn, nx, ny))
        z[linear_index(in, jn, nx)] -= amount * w;
    }
}

inline void helper_bilinear_deposition(global float *z,
                                       int           i,
                                       int           j,
                                       int           nx,
                                       float         u,
                                       float         v,
                                       float         amount)
{
  float d1 = (1.f - u) * (1.f - v);
  float d2 = u * (1.f - v);
  float d3 = (1.f - u) * v;
  float d4 = u * v;

  z[linear_index(i, j, nx)] -= amount * d1;
  z[linear_index(i + 1, j, nx)] -= amount * d2;
  z[linear_index(i, j + 1, nx)] -= amount * d3;
  z[linear_index(i + 1, j + 1, nx)] -= amount * d4;
}

void kernel hydraulic_particle(global float *z_in,
                               global float *bedrock,
                               global float *moisture_map,
                               const int     nx,
                               const int     ny,
                               const int     nparticles,
                               const uint    seed,
                               const float   c_capacity,
                               const float   c_erosion,
                               const float   c_deposition,
                               const float   c_inertia,
                               const float   drag_rate,
                               const float   evap_rate,
                               const int     has_bedrock,
                               const int     has_moisture_map)
{
  float dt = 1.f;

  int id = get_global_id(0); // particle id
  if (id > nparticles) return;

  uint rng_state = wang_hash(seed + id);

  float2 pos = {(nx - 2) * rand(&rng_state), (ny - 2) * rand(&rng_state)};
  int    i, j;
  float  u, v;

  update_interp_param(pos, &i, &j, &u, &v);

  float2 vel = {0.f, 0.f};
  float  volume = has_moisture_map > 0 ? moisture_map[linear_index(i, j, nx)]
                                       : 1.f;

  float s = 0.f;

  float evap_factor = 1.f - evap_rate;

  while (volume > 1e-3f)
  {
    update_interp_param(pos, &i, &j, &u, &v);

    // stop if the particle reaches the domain limits
    if (i < 1 || i > nx - 2 || j < 1 || j > ny - 2) break;

    float2 gz = helper_compute_gradient(z_in, i, j, nx);

    // particle goes downhill, opposite local gradient
    vel += dt * (float2)(-gz.x, -gz.y) / c_inertia;
    vel *= (1.f - dt * drag_rate);

    float vnorm = length(vel);

    if (vnorm < VELOCITY_MIN) break;

    float zp = helper_sample_height(z_in, i, j, nx, u, v);

    // backup previous position
    int   ip = i;
    int   jp = j;
    float up = u;
    float vp = v;

    // move particle
    pos += dt * vel;

    // elevation at new position
    update_interp_param(pos, &i, &j, &u, &v);
    if (i < 1 || i > nx - 2 || j < 1 || j > ny - 2) break;

    float z = helper_sample_height(z_in, i, j, nx, u, v);
    float dz = zp - z;
    float sc = max(c_capacity * volume * vnorm * dz, 0.0001f);
    float delta_sc = dt * (sc - s);
    float amount = 0.f;

    // if more sediments than capacity or if uphill motion
    if (delta_sc < 0.f || dz < 0.f)
    {
      // deposition
      amount = (dz < 0.f) ? -min(-dz, s) : c_deposition * delta_sc;
      helper_bilinear_deposition(z_in, ip, jp, nx, up, vp, amount);

      /* int ir = 3; */
      /* helper_radial_kernel_deposition(z_in, ip, jp, nx, ny, amount, ir); */
    }
    else
    {
      // erosion
      amount = c_erosion * delta_sc;

      int ir = 2;
      helper_radial_kernel_deposition(z_in, ip, jp, nx, ny, amount, ir);
    }

    s += amount;

    // bedrock limit enforcement
    if (amount > 0.f && has_bedrock != 0) // erosion
    {
      z_in[linear_index(ip, jp, nx)] = max(bedrock[linear_index(ip, jp, nx)],
                                           z_in[linear_index(ip, jp, nx)]);

      z_in[linear_index(ip + 1, jp, nx)] = max(
          bedrock[linear_index(ip + 1, jp, nx)],
          z_in[linear_index(ip + 1, jp, nx)]);

      z_in[linear_index(ip, jp + 1, nx)] = max(
          bedrock[linear_index(ip, jp + 1, nx)],
          z_in[linear_index(ip, jp + 1, nx)]);

      z_in[linear_index(ip + 1, jp + 1, nx)] = max(
          bedrock[linear_index(ip + 1, jp + 1, nx)],
          z_in[linear_index(ip + 1, jp + 1, nx)]);
    }

    volume *= evap_factor;
  }
}
)""
