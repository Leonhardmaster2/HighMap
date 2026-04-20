/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <limits>
#include <map>

#include "macrologger.h"

#include "highmap/array.hpp"
#include "highmap/boundary.hpp"

namespace hmap
{

Array connected_components(const Array            &array,
                           float                   surface_threshold,
                           float                   background_value,
                           std::map<float, float> *p_surfaces,
                           std::map<float, std::array<float, 2>> *p_centroids)
{
  const glm::ivec2 &shape = array.shape;

  // neighbor search pattern
  const float  di[4] = {0, -1, -1, -1};
  const float  dj[4] = {-1, -1, 0, 1};
  const size_t nb = 4;

  // padding: one cell with a non-background value on the borders
  const int npi = shape.x + 2;
  const int npj = shape.y + 2;

  Array labels = Array(glm::ivec2(npi, npj));
  Array array_pad = generate_buffered_array(array, {1, 1, 1, 1});
  set_borders(array_pad, std::numeric_limits<float>::max(), 1);

  // --- first pass of labelling

  int                                 current_label = 0;
  std::map<float, std::vector<float>> labels_mapping = {};

  // ===========================================
  //  /!\ i, j LOOP ORDER MATERS, DO NOT CHANGE
  // ===========================================

  for (int i = 0; i < npi; i++)   // i first
    for (int j = 0; j < npj; j++) // j second
      if (array_pad(i, j) != background_value)
      {
        // scan neighbors and count those that are "background"
        std::vector<int> nbrs_label = {};
        nbrs_label.reserve(nb);

        for (size_t k = 0; k < nb; k++)
        {
          int p = i + di[k];
          int q = j + dj[k];
          if ((p > 0) and (p < npi) and (q > 0) and (q < npj))
            if (array_pad(p, q) != background_value)
              nbrs_label.push_back(labels(p, q));
        }

        if (nbrs_label.size() == 0)
          labels(i, j) = current_label++;
        else if (nbrs_label.size() == 1)
          // set current label to the value of the only
          // non-background neighbor
          labels(i, j) = nbrs_label.back();
        else
        {
          int lmin = *std::min_element(nbrs_label.begin(), nbrs_label.end());
          labels(i, j) = lmin;

          // keep track of links between label values
          for (auto &v : nbrs_label)
            if (lmin != v) labels_mapping[lmin].push_back(v);
        }
      }

  // --- relabel components

  // reverse the labels mapping
  std::map<float, float> labels_mapping_reverse = {};

  for (auto &[key, v] : labels_mapping)
    for (auto &s : v)
      labels_mapping_reverse[s] = key;

  // find the root label by "traversing" the mapping, starting from
  // the higher labels
  for (auto it = labels_mapping_reverse.rbegin();
       it != labels_mapping_reverse.rend();
       it++)
  {
    float label_root = it->first;
    while (true)
    {
      if (labels_mapping_reverse.contains(label_root))
        label_root = labels_mapping_reverse[label_root];
      else
        break;
    }
    it->second = label_root;
  }

  for (int j = 0; j < npj; j++)
    for (int i = 0; i < npi; i++)
    {
      if ((labels(i, j) > 0) && (labels_mapping_reverse.contains(labels(i, j))))
        labels(i, j) = labels_mapping_reverse[labels(i, j)];
    }

  // --- clean-up, remove single-cell labels and filter by surface

  std::map<float, float> labels_surface = {};

  for (int j = 0; j < npj; j++)
    for (int i = 0; i < npi; i++)
    {
      if (labels(i, j) > 0.f) labels_surface[labels(i, j)] += 1.f;
    }

  // filter
  for (int j = 0; j < labels.shape.y; j++)
    for (int i = 0; i < labels.shape.x; i++)
    {
      if (labels_surface[labels(i, j)] <= surface_threshold)
        labels(i, j) = background_value;
    }

  // --- removing padding before returning the result

  labels = labels.extract_slice({1, npi - 1, 1, npj - 1});

  // --- relabeling

  std::vector<float> used_labels = labels.unique_values();

  // edge case of one label not to zero, add manually the background
  // which not present in the label array
  if (used_labels.size() == 1 && used_labels.back() != 0.f)
    used_labels.insert(used_labels.begin(), 0.f);

  // relabel...
  size_t           max_label = size_t(used_labels.back());
  std::vector<int> label_remap(max_label + 1, 0);

  int next = 0;
  for (const auto &v : used_labels)
    label_remap[int(v)] = next++;

  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
    {
      int lbl = int(labels(i, j));
      if (lbl != 0) labels(i, j) = label_remap[lbl];
    }

  // --- update surface and centroids

  labels_surface.clear();
  std::map<float, std::array<float, 2>> labels_centroids = {};

  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
    {
      if (labels(i, j) > 0)
      {
        labels_surface[labels(i, j)] += 1.f;
        labels_centroids[labels(i, j)][0] += float(i);
        labels_centroids[labels(i, j)][1] += float(j);
      }
    }

  // --- outputs

  if (p_surfaces) *p_surfaces = std::move(labels_surface);
  if (p_centroids) *p_centroids = std::move(labels_centroids);

  return labels;
}

} // namespace hmap
