/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cmath>

#include "macrologger.h"

#include "highmap/interpolate2d.hpp"

namespace hmap
{

NaturalNeighborInterpolator::~NaturalNeighborInterpolator()
{
  if (this->handle) nnai_destroy(this->handle);
  if (this->d) delaunay_destroy(this->d);
}

void NaturalNeighborInterpolator::build(const std::vector<float> &xin,
                                        const std::vector<float> &yin)
{
  int npoints = static_cast<int>(xin.size());

  // create delaunay triangulation
  std::vector<point> pin(npoints);
  for (int i = 0; i < npoints; ++i)
  {
    pin[i].x = double(xin[i]);
    pin[i].y = double(yin[i]);
    pin[i].z = double(0.0); // can be set later via values array
  }

  d = delaunay_build(npoints, pin.data(), 0, nullptr, 0, nullptr);

  // create nnai interpolator
  handle = nnai_build(d, int(this->nout), this->xout.data(), this->yout.data());
  nnai_setwmin(handle, double(-1e30));
}

void NaturalNeighborInterpolator::interpolate(
    const std::vector<float> &values_in,
    std::vector<float>       &values_out) const
{
  if (!handle) return;

  values_out.resize(this->nout);

  // convert float input to double
  std::vector<double> values_in_d(values_in.begin(), values_in.end());
  std::vector<double> values_out_d(this->nout);

  // call the nnai C function
  nnai_interpolate(handle, values_in_d.data(), values_out_d.data());

  // convert double output back to float
  for (size_t k = 0; k < this->nout; ++k)
    values_out[k] = float(values_out_d[k]);
}

void NaturalNeighborInterpolator::setup_output_points(
    const std::vector<float> &x,
    const std::vector<float> &y)
{
  this->xout.clear();
  this->yout.clear();

  this->xout.reserve(x.size());
  this->yout.reserve(y.size());

  for (const auto &v : x)
    this->xout.push_back(double(v));

  for (const auto &v : y)
    this->yout.push_back(double(v));

  this->nout = this->xout.size();
}

} // namespace hmap
