#include "highmap.hpp"

#include "highmap/dbg/timer.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  shape = {512, 512};
  shape = {1024, 1024};

  glm::vec2 kw = {4.f, 4.f};
  int       seed = 1;
  glm::vec4 bbox = {0.f, 1.f, 0.f, 1.f};
  size_t    count = 10000;

  hmap::Cloud cloud = hmap::random_cloud(count,
                                         seed,
                                         hmap::PointSamplingMethod::RND_LHS,
                                         bbox);

  // hmap::Cloud cloud = hmap::random_cloud_jittered(count,
  //                                                 {1.f, 1.f},
  //                                                 {0.f, 0.f},
  //                                                 seed,
  //                                                 bbox);

  cloud.snap_points_to_bounding_box(bbox);

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                   shape,
                                   kw,
                                   seed,
                                   8,
                                   0.7f);
  z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CONE_SMOOTH, 2.f);
  hmap::remap(z0);

  z0.dump("z0.png");

  cloud.set_values_from_array(z0, bbox);

  cloud.to_csv("cloud0.csv");

  auto db = hmap::DrainageBasin(cloud.to_vec3());
  // db.set_outlets({5, 10, 100});
  // db.set_outlets(hmap::find_border_minima(db.get_xyz()));
  // db.set_outlets(hmap::sample_border_points(db.get_xyz(), 5));
  // db.remap();

  int iterations = 5000;

  // std::vector<float> erodibility(cloud.get_npoints(), 1.f);
  auto erodibility = cloud.get_values();
  for (auto &v : erodibility)
    v = std::pow(1.f - v, 1.f);

  auto max_slope = cloud.get_values();

  if (false)
  {
    float smin = 1.f;
    float smax = 8.f;
    for (auto &v : max_slope)
      v = (smax - smin) * v + smin;
  }
  else
  {
    for (auto &v : max_slope)
      v = 4.f;
  }

  float m_exp = 0.8f;
  float uplift_rate = 0.2f;
  float tolerance = 1e-3f;

  hmap::Timer::Start("main_loop");

  int sub_it = 1;

  for (int it = 0; it < iterations; ++it)
  {
    std::vector acc(db.size(), 0.f);

    if (it % sub_it == 0)
    {
      hmap::Timer::Start("update_stream_tree");
      db.update_stream_tree();
      hmap::Timer::Stop("update_stream_tree");
    }

    hmap::Timer::Start("compute_vertex_areas");
    auto area = db.compute_vertex_areas();
    hmap::Timer::Stop("compute_vertex_areas");

    hmap::Timer::Start("accumulate_area_by_outlet");
    db.accumulate_area_by_outlet(area, acc);
    hmap::Timer::Stop("accumulate_area_by_outlet");

    hmap::Timer::Start("compute_response_times");
    auto response_times = db.compute_response_times(acc, erodibility, m_exp);
    hmap::Timer::Stop("compute_response_times");

    hmap::Timer::Start("update_elevations");
    float diff = db.update_elevations(response_times, uplift_rate, max_slope);
    hmap::Timer::Stop("update_elevations");

    LOG_DEBUG("%d %f", it, diff);

    if (it % sub_it == 0)
    {
      if (diff < tolerance) break;
    }
  }

  hmap::Timer::Stop("main_loop");

  db.update_stream_tree();
  db.smooth_mesh(0.5f, 10);
  // db.smooth_mesh_taubin(0.50f, -0.55f, 40);

  db.remap();
  db.to_csv("db.csv");

  //
  if (true)
  {
    // elevation
    std::vector<float> xc, yc, zc;
    for (const auto &p : db.get_xyz())
    {
      xc.push_back(p.x);
      yc.push_back(p.y);
      zc.push_back(p.z);
    }

    hmap::Array za = hmap::interpolate2d(
        shape,
        xc,
        yc,
        zc,
        hmap::InterpolationMethod2D::ITP2D_NNI);
    za.dump();

    // difference
    auto zinit = cloud.get_values();

    for (size_t k = 0; k < db.size(); ++k)
      zc[k] = zinit[k] - zc[k];

    hmap::Array zd = hmap::interpolate2d(
        shape,
        xc,
        yc,
        zc,
        hmap::InterpolationMethod2D::ITP2D_NNI);
    zd.dump("diff.png");

    // flow accumulation
    auto               area = db.compute_vertex_areas();
    std::vector<float> flow(db.size(), 1.f);
    db.accumulate_area_by_outlet(area, flow);

    for (auto &v : flow)
      v = std::log10(v + 1.f);

    hmap::Array facc = hmap::interpolate2d(
        shape,
        xc,
        yc,
        flow,
        hmap::InterpolationMethod2D::ITP2D_DELAUNAY);
    facc.dump("facc.png");
  }

  //
  if (true)
  {
    std::vector<float> xc, yc, zc;
    for (const auto &p : db.get_xyz())
    {
      xc.push_back(p.x);
      yc.push_back(p.y);
      zc.push_back(p.z);
    }

    hmap::Graph graph(hmap::Cloud(xc, yc, zc));

    auto receivers = db.get_receivers();

    for (size_t k = 0; k < db.size(); ++k)
    {
      size_t r = receivers[k];
      graph.add_edge({int(k), int(r)});
    }

    // order
    if (false)
    {
      auto order = db.compute_strahler_order();
      for (size_t k = 0; k < db.size(); ++k)
      {
        graph.points[k].v = float(order[k]);
      }
    }

    // flow acc
    if (false)
    {
      auto               area = db.compute_vertex_areas();
      std::vector<float> flow(db.size(), 1.f);
      db.accumulate_area_by_outlet(area, flow);

      for (size_t k = 0; k < db.size(); ++k)
      {
        graph.points[k].v = flow[k];
      }
    }

    hmap::Array zg(shape);
    graph.to_array(zg, bbox, false);
    zg.dump("zg.png");

    // fill
    // if (true)
    // {
    //   hmap::Array zgi = hmap::gpu::harmonic_interpolation(zg,
    //   hmap::is_non_zero(zg)); zgi.dump("zgi.png");
    // }

    hmap::Array zgi = zg;
    hmap::expand_talus(zgi, hmap::is_non_zero(zgi), 16.f / shape.x, 0);
    zgi.dump("zgi.png");
  }
}
