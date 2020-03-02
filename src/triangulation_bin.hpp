/* Copyright 2019 Greg Tucker
//
// This file is part of brille.
//
// brille is free software: you can redistribute it and/or modify it under the
// terms of the GNU Affero General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// brille is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with brille. If not, see <https://www.gnu.org/licenses/>.            */
#ifndef _TRIANGULATION_BIN_HPP_
#define _TRIANGULATION_BIN_HPP_

#include <vector>
#include <array>
#include <numeric>
#include <omp.h>

#include "bin_utilities.hpp"
#include "arrayvector.hpp"
#include "polyhedron.hpp"
#include "utilities.hpp"
#include "debug.hpp"

class BinTet {
public:
  using index_t = unsigned long;
  using Bounds = std::vector<double>;
  using BinContents = std::vector<index_t>;
private:
  SimpleTet triangulation_;         //! The triangulated points and tetrahedra
  Polyhedron p_;
  std::array<Bounds,3> boundaries_; //! The bin boundaries in three dimensions
  std::vector<index_t> map_;        //! Bin index to in-polyhedron bin index map
  std::vector<BinContents> bins_;   //! In-polyhedron bins with intersecting tetrahera contents
public:
  BinTet() {}
  BinTet(const SimpleTet& t, const double v=-1.): triangulation_(t), p_(t.get_vertices()) {
    this->determine_boundaries(v);
  }
  BinTet(const SimpleTet& t, const Polyhedron& p, const double v=-1.): triangulation_(t), p_(p) {
    this->determine_boundaries(v);
  }
  // SimpleTet methods:
  const ArrayVector<double>& get_vertices() const {return triangulation_.get_vertices();}
  const ArrayVector<index_t>& get_vertices_per_tetrahedron() const {return triangulation_.get_vertices_per_tetrahedron();}
  // new methods
  size_t count_empty_bins() const {
    return std::count_if(bins_.begin(), bins_.end(), [](const BinContents& x){return x.size()==0;});
  }
  size_t count() const {
    std::vector<size_t> c;
    std::transform(bins_.begin(), bins_.end(), std::back_inserter(c), [](const BinContents& x){return x.size();});
    return std::accumulate(c.begin(), c.end(), 0u);
  }
  size_t bin_size() const {return bins_.size();}
  std::array<index_t,3> bin_shape() const {
    return {boundaries_[0].size()-1, boundaries_[1].size()-1, boundaries_[2].size()-1};
  }
  std::array<index_t,3> bin_span() const {
    auto sz=this->bin_shape();
    return {1, sz[0], sz[0]*sz[1]};
  }
  index_t bin_index(const ArrayVector<double>& x) const {
    return bin_index(x,this->bin_span());
  }
  index_t bin_index(const ArrayVector<double>& x, const std::array<index_t,3>& sp) const {
    return map_[map_index(x, sp)];
  }
  index_t map_index(const ArrayVector<double>& x) const {
    return map_index(x,this->bin_span());
  }
  index_t map_index(const ArrayVector<double>& x, const std::array<index_t,3>& sp) const {
    index_t idx{0};
    for (int i=0; i<3; ++i)
      idx += sp[i]*find_bin(boundaries_[i], x.getvalue(0,i));
    return idx;
  }
  index_t map_sub2idx(const index_t i, const index_t j, const index_t k) const {
    return map_sub2idx(i,j,k,this->bin_span());
  }
  index_t map_sub2idx(const index_t i, const index_t j, const index_t k, const std::array<index_t,3>& sp) const{
    return sp[0]*i + sp[1]*j + sp[2]*k;
  }
  index_t map_sub2idx(const std::array<index_t,3>& sub) const{
    return map_sub2idx(sub,this->bin_span());
  }
  index_t map_sub2idx(const std::array<index_t,3>& sub, const std::array<index_t,3>& sp) const{
    index_t idx{0};
    for (index_t d=0; d<3u; ++d) idx += sp[d]*sub[d];
    return idx;
  }
  std::array<index_t,3> map_idx2sub(const index_t idx) const{
    return map_idx2sub(idx,this->bin_span());
  }
  std::array<index_t,3> map_idx2sub(const index_t idx, const std::array<index_t,3>& sp) const{
    std::array<index_t,3> sub{{0,0,0}};
    index_t rem{idx};
    for (index_t d=3u; d--; ){
      sub[d] = rem/sp[d];
      rem -= sub[d]*sp[d];
    }
    return sub;
  }
  //
  std::array<Bounds,3> determine_boundaries(const double volin=-1.){
    // pick a volume for the cubic cells
    double vol = volin > 0 ? volin : triangulation_.maximum_volume();
    double len = std::cbrt(vol);
    // find the bounding box corners of the polyhedron
    std::vector<double> vmin = p_.get_vertices().min().to_std();
    std::vector<double> vmax = p_.get_vertices().max().to_std();
    // construct the bin boundary vectors
    std::array<Bounds,3> b;
    for (int i=0; i<3; ++i){
      b[i].push_back(len);
      while (b[i].back() < vmax[i]) b[i].push_back(b[i].back()+len);
    }
    return this->set_boundaries(b);
  }
  std::array<Bounds,3> set_boundaries(const std::array<Bounds,3>& b){
    boundaries_ = b;
    /* Reset the map_ vector */
    // find the number of logical bins specified by the boundaries
    auto nbins = this->bin_size();
    // ensure that the map can store an entry for each logical bin
    map_.resize(nbins);
    // set all logical bins to an invalid in-polyhedron bin index
    // (the highest possible valid index is nbins-1)
    std::fill(map_.begin(), map_.end(), nbins);
    // there are no longer any valid bins_, so empty it out
    bins_.resize(0);
    // then refill them:
    this->fill_bins();
    return boundaries_;
  }
  std::array<Bounds,3> get_boundaries() const {return boundaries_;}
  bool fill_bins(){
    auto shape = this->bin_shape();
    auto nbins = this->bin_size();
    auto span = this->bin_span();
    index_t count{0};
    Polyhedron cube, cube_poly;
    // First check for bin-polyhedron intersections so that we can keep track of
    // whether any in-polyhedron bins are devoid of intersecing tetrahedra
    for (index_t k=0; k<shape[2]; ++k)
    for (index_t j=0; j<shape[1]; ++j)
    for (index_t i=0; i<shape[0]; ++i) {
      cube = bin_polyhedron(i,j,k);
      // checking for an intersection of the cube and polyhedron should be fast
      cube_poly = cube.intersection(p_);
      // if the cube and polyhedron intersect, this is an in-polyhedron bin
      if (cube_poly.get_vertices().size()>3 && !approx_scalar(cube_poly.get_volume(),0.))
        map_[map_sub2idx(i,j,k,span)] = count++; // so it gets a valid mapping
    }
    // Now check whether the in-polyhedron bins intersect with tetrahedra
    // This nested for loop scheme *must* match bin_span() since bins_.push_back is used
    for (index_t k=0; k<shape[2]; ++k)
    for (index_t j=0; j<shape[1]; ++j)
    for (index_t i=0; i<shape[0]; ++i)
    if (map_[map_sub2idx(i,j,k,span)] < nbins) {
      cube = bin_polyhedron(i,j,k);
      // check against the triangulated tetrahedra since the cube and polyhedron intersect
      // and store their indices (which might be an empty vector)
      bins_.push_back(triangulation_.tetrahedra_intersecting_with(cube));
      // change this to debug_update_if to preprocess-filter out the check
      // in release code once this method is verified working
      info_update_if(map_[map_sub2idx(i,j,k,span)] != bins_.size()+1,"tetrahedra assigned to wrong bin?!");
    }
    return count == bins_.size() && count <= nbins;
  }

  BinContents bin_contents(const ArrayVector<double>& x) const {
    return bin_contents(map_index(x));
  }
  BinContents bin_contents(const index_t idx) const {
    if (idx < map_.size() && map_[idx] < bins_.size())
      return bins_[map_[idx]];
    if (idx < map_.size())
      throw std::runtime_error("Point outside of mapped bins");
    throw std::runtime_error("Invalid bin index");
  }
  Polyhedron bin_polyhedron(const index_t i, const index_t j, const index_t k) const {
    // construct a rectangular prism for bin (i,j,k)
    std::array<double,3> mnc{{boundaries_[0][i],   boundaries_[1][j],   boundaries_[2][k]}};
    std::array<double,3> mxc{{boundaries_[0][i+1], boundaries_[1][j+1], boundaries_[2][k+1]}};
    return polyhedron_box(mnc, mxc);
  }
  bool indices_weights(const ArrayVector<double>& x, std::vector<std::pair<index_t,double>>& iw) const {
    const auto & vt{triangulation_.get_vertices_per_tetrahedron()};
    iw.clear();
    std::array<double,4> w{{0,0,0,0}};
    for (auto t: this->bin_contents(x))
    if (triangulation_.tetrahedra_contains(t, x, w)){
      for (int j=0; j<4; ++j) if (!approx_scalar(w[j],0.))
        iw.push_back(std::make_pair(vt.getvalue(t,j),w[j]));
      return true; // tetrahedra t has x inside, indices and weights set
    }
    return false;
  }
};


#endif
