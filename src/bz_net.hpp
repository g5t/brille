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

typedef long slong;

#ifndef _BZ_NET_
#define _BZ_NET_

#include "bz.hpp"
#include "net.hpp"

template<class T> class BrillouinZoneNet3: public Net<T>{
  BrillouinZone brillouinzone;
  using index_t = unsigned long;
public:
  BrillouinZoneNet3(const BrillouinZone& bz, const double vol=-1):
    Net<T>(bz.get_ir_polyhedron(), vol), brillouinzone(bz) {}
  template<typename R>
  BrillouinZoneNet3(const BrillouinZone& bz, const LQVec<R>& pts):
    brillouinzone(bz) {
    // The input points *MUST* be moved into the first/ir-Brillouin zone NOW!
    // Plus we *must* deal with the input data, if present.
    LQVec<double> q(bz.get_lattice(),pts.size());
    LQVec<int> tau(bz.get_lattice(), pts.size());
    std::vector<std::array<int,9>> rots(pts.size());
    bz.ir_moveinto(pts, q, tau, rots);
    this->Net<T>::reset(bz.get_ir_polyhedron(), q.get_xyz());
  }
  template<typename R,typename I>
  BrillouinZoneNet3(const BrillouinZone& bz, const LQVec<R>& pts, const ArrayVector<T>& data, const std::array<I,3>& elements):
    brillouinzone(bz) {
    // The input points *MUST* be moved into the first/ir-Brillouin zone NOW!
    // Plus we *must* deal with the input data, if present.
    LQVec<double> q(bz.get_lattice(),pts.size());
    LQVec<int> tau(bz.get_lattice(), pts.size());
    std::vector<std::array<int,9>> rots(pts.size());
    // the following finds q, τ and R such that pts = R q + τ
    bz.ir_moveinto(pts, q, tau, rots);
    ArrayVector<T> rotdata(data);
    // we have data(pts) but want data(q) to put into the Net
    if (elements[1]>0 || elements[2]>0){
      // we only need to do the rotation if there are vectors or tensors
      T tmp_v[3], tmp_m[9];
      std::vector<std::array<int,9>> invR(rots.size());
      for (size_t i=0; i<rots.size(); ++i)
        matrix_inverse(invR[i].data(), rots[i].data());
      size_t offset, sp{elements[0]+elements[1]+elements[2]};
      size_t branches = data.numel()/sp;
      for (size_t i=0; i<rotdata.size(); ++i)
      for (size_t b=0; b<branches; ++b){
        // skip over the scalar elements
        offset = b*sp + elements[0];
        // rotate any vectors
        for (size_t v=0; v<elements[1]/3; ++v){
          mul_mat_vec(tmp_v, 3u, invR[i].data(), rotdata.data(i, offset+v*3u));
          for (int j=0; j<3; ++j) rotdata.insert(tmp_v[j], i, offset+v*3u);
        }
        offset += elements[1];
        for (size_t m=0; m<elements[2]/9; ++m){
          // calculate R⁻¹ M R (rotate by R⁻¹) in two steps
          mul_mat_mat(tmp_m, 3u, rotdata.data(i, offset+m*9u), rots[i].data());
          mul_mat_mat(rotdata.data(i, offset+m*9u), 3u, invR[i].data(), tmp_m);
        }
      }
    }
    this->Net<T>::reset(bz.get_ir_polyhedron(), q.get_xyz(), rotdata, elements);
  }

  //! get the BrillouinZone object
  BrillouinZone get_brillouinzone(void) const {return this->brillouinzone;}
  //! get the vertices of the triangulation in absolute units
  const ArrayVector<double>& get_xyz(void) const {return this->get_vertices();}
  //! get the vertices of the triangulation in relative lattice units
  ArrayVector<double> get_hkl(void) const { return xyz_to_hkl(brillouinzone.get_lattice(),this->get_vertices());}
  //! get the indices forming the faces of the tetrahedra
  std::vector<std::array<index_t,4>> get_vertices_per_tetrahedron(void) const {return this->get_vertices_per_tetrahedron();}

  template<typename R> ArrayVector<T> interpolate_at(const LQVec<R>& x, const int nthreads, const bool no_move=false) const{
    LQVec<R> q(x.get_lattice(), x.size());
    LQVec<int> tau(x.get_lattice(), x.size());
    if (no_move){
      // Special mode for testing where no specified points are moved
      // IT IS IMPERITIVE THAT THE PROVIDED POINTS ARE *INSIDE* THE IRREDUCIBLE
      // POLYHEDRON otherwise the interpolation will fail or give garbage back.
      q = x;
    } else if (!brillouinzone.moveinto(x, q, tau, nthreads)){
      std::string msg;
      msg = "Moving all points into the first Brillouin zone failed.";
      throw std::runtime_error(msg);
    }
    ArrayVector<T> result = (nthreads < 2)
      ? this->Net<T>::interpolate_at(q.get_xyz())
      : this->Net<T>::interpolate_at(q.get_xyz(), nthreads);
    return result;
  }

  template<typename R> ArrayVector<T> ir_interpolate_at(const LQVec<R>& x, const int nthreads, const bool no_move=false) const{
    LQVec<R> ir_q(x.get_lattice(), x.size());
    LQVec<int> tau(x.get_lattice(), x.size());
    std::vector<std::array<int,9>> rots(x.size());
    if (no_move){
      // Special mode for testing where no specified points are moved
      // IT IS IMPERITIVE THAT THE PROVIDED POINTS ARE *INSIDE* THE IRREDUCIBLE
      // POLYHEDRON otherwise the interpolation will fail or give garbage back.
      ir_q = x;
      for (size_t i=0; i<x.size(); ++i) rots[i] = {1,0,0, 0,1,0, 0,0,1};
    } else if (!brillouinzone.ir_moveinto(x, ir_q, tau, rots, nthreads)){
      std::string msg;
      msg = "Moving all points into the irreducible Brillouin zone failed.";
      throw std::runtime_error(msg);
    }
    ArrayVector<T> ir_result = (nthreads < 2)
      ? this->Net<T>::interpolate_at(ir_q.get_xyz())
      : this->Net<T>::interpolate_at(ir_q.get_xyz(), nthreads);

    if (nthreads < 2)
      this->data().rotate_in_place(ir_result, rots);
    else
      this->data().rotate_in_place(ir_result, rots, nthreads);

    return ir_result;
  }
};

#endif // _BZ_MESH_
