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
#ifndef _NET_HPP_
#define _NET_HPP_

#include <vector>
#include <array>
#include <algorithm>
#include <omp.h>

#include "arrayvector.hpp"
#include "latvec.hpp"
#include "polyhedron.hpp"
#include "utilities.hpp"
#include "debug.hpp"
#include "triangulation_bin.hpp"
#include "interpolation_data.hpp"

/*! \brief A triangulated linear interpolation data class with binned vertices for fast point location

The `Net` class containes a binned tetrahedral triangulation object and
a general InterpolationData objects.
It can be constructed from a bounding `Polyhedron` and the information required
to create a `SimpleTet` object, or by a set of pre-defined vertices in which
case a convex hull polyhedron is constructed and the vertices are triangulated.

In the case of pre-defined vertices the interpolation data can be provided at
creation time as well.
*/
template <class T>
class Net {
  BinTet bt_;
  InterpolationData<T> data_;
public:
  using index_t = unsigned long;
  // Triangulate the polyhedron
  Net(const Polyhedron& p, const double v=-1, const bool addG=false): bt_(SimpleTet(p,v,addG),p) {}
  // Triangulate the specified points
  Net(const ArrayVector<double>& v): bt_(SimpleTet(v), Polyhedron(v)){}
  // Triangulate the specified points and add the specified data
  template<class... A>
  Net(const ArrayVector<double>& v, A... a): bt_(SimpleTet(v), Polyhedron(v)){
    // perform some sort of validation that the points in `verts` have the same
    // order as the vertices in bt_?
    this->replace_data(a...);
  }
  // Triangulate the specified points inside of the provided polyhedron
  Net(const Polyhedron& p, const ArrayVector<double>& v): bt_(SimpleTet(v), p) {}
  // Triangulate the specified points inside of the provided polyhedron
  // and add the specified data
  template<class... A>
  Net(const Polyhedron& p, const ArrayVector<double>& v, A... a): bt_(SimpleTet(v), p) {
    this->replace_data(a...);
  }
  // BinTet methods:
  std::array<BinTet::Bounds,3> get_boundaries() const { return bt_.get_boundaries(); }
  std::array<BinTet::Bounds,3> set_boundaries(const std::array<BinTet::Bounds,3>& b) {return bt_.set_boundaries(b);}
  bool indices_weights(const ArrayVector<double>& x, std::vector<std::pair<index_t,double>>& iw) const {
    return bt_.indices_weights(x,iw);
  }
  const ArrayVector<double>& get_vertices() const { return bt_.get_vertices();}
  const ArrayVector<index_t>& get_vertices_per_tetrahedron() const {return bt_.get_vertices_per_tetrahedron();}
  // InterpolationData methods:
  const InterpolationData<T>& data(void) const {return data_;}
  template<typename... A> void replace_data(A... args) { data_.replace_data(args...); }
  template<template<class> class A>
  ArrayVector<double> debye_waller(const A<double>& Q, const std::vector<double>& M, const double t_K) const{
    return data_.debye_waller(Q,M,t_K);
  }
  // new methods
  template<class R> unsigned check_before_interpolating(const ArrayVector<R>& x) const{
    unsigned int mask = 0u;
    if (this->data_.size()==0)
      throw std::runtime_error("Net must be filled before interpolating!");
    if (x.numel()!=3u)
      throw std::runtime_error("Net requires x values which are three-vectors.");
    return mask;
  }
  // Mixed BinTet/InterpolationData methods:
  ArrayVector<T> interpolate_at(const ArrayVector<double>& x) const {
    verbose_update("Single thread interpolation at ",x.size()," points");
    this->check_before_interpolating(x);
    ArrayVector<T> out(data_.numel(), x.size());
    std::vector<std::pair<BinTet::index_t, double>> iw;
    for (size_t i=0; i<x.size(); ++i){
      verbose_update("Locating ",x.to_string(i));
      if (!this->indices_weights(x.extract(i),iw))
        throw std::runtime_error("Point not found in BinTet");
      data_.interpolate_at(iw, out, i);
    }
    return out;
  }
  ArrayVector<T> interpolate_at(const ArrayVector<double>& x, const int threads) const {
    this->check_before_interpolating(x);
    omp_set_num_threads( (threads > 0) ? threads : omp_get_max_threads() );
    verbose_update("Parallel interpolation at ",x.size()," points with ",threads," threads");
    // shared between threads
    ArrayVector<T> out(data_.numel(), x.size(), T(0)); // initialise to zero so that we can use a reduction
    // private to each thread
    std::vector<std::pair<BinTet::index_t, double>> iw;
    // OpenMP < v3.0 (VS uses v2.0) requires signed indexes for omp parallel
    long long xsize = unsigned_to_signed<long long, size_t>(x.size());
    size_t n_unfound{0};
  #pragma omp parallel for default(none) shared(x,out,xsize) private(iw) reduction(+:n_unfound) schedule(dynamic)
    for (long long si=0; si<xsize; ++si){
      size_t i = signed_to_unsigned<size_t, long long>(si);
      if (this->indices_weights(x.extract(i), iw)){
        data_.interpolate_at(iw, out, i);
      } else {
        ++n_unfound;
      }
    }
    std::runtime_error("interpolate at failed to find "+std::to_string(n_unfound)+" point"+(n_unfound>1?"s.":"."));
    return out;
  }
};


#endif
