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

#include "bz.hpp"

LQVec<double,brille::ref_ptr_t> BrillouinZone::get_ir_polyhedron_wedge_normals(void) const {
  auto ir_n = this->get_ir_normals();
  auto ir_p = this->get_ir_points();
  auto bz_n = this->get_normals();
  auto bz_p = this->get_points();
  for (size_t i=0; i<bz_n.size(0); ++i){
    // check if the irBZ face point is on a first BZ zone face too
    auto not_bz = dot(bz_n.view(i), ir_p - bz_p.view(i)).is(brille::cmp::neq, 0.);
    ir_n = ir_n.extract(not_bz);
    ir_p = ir_p.extract(not_bz);
  }
  // It is possible that, lacking inversion symmetry, we have found an
  // irreducible polyhedron which is comprised of two convex polyhedra that
  // are mutualy inverse. In such a case for every normal in ir_n there is also
  // its opposite also in ir_n, and so ir_n defines a wedge and an anti-wedge in
  // which no finite point can be inside
  if (ir_n.size(0)%2 == 0 /* [lacks inversion] or this->no_ir_mirroring? */){
    std::vector<bool> no_inverse(ir_n.size(0), true), keep(ir_n.size(0), true);
    for (size_t i=0; i<ir_n.size(0)-1; ++i) if (no_inverse[i])
    for (size_t j=i+1; j<ir_n.size(0); ++j) if (no_inverse[j])
    if ((ir_n.view(i)+ir_n.view(j)).all(brille::cmp::eq, 0.)) {
      no_inverse[i] = no_inverse[j] = keep[j] = false;
      break;
    }
    if (0==std::count(no_inverse.begin(), no_inverse.end(), true))
      ir_n = ir_n.extract(keep);
  }
  // the remaining irBZ faces are the irreducible reciprocal space wedge
  // which we store with the opposite sign in this->ir_wedge_normals
  return -ir_n;
}

Polyhedron BrillouinZone::get_polyhedron(void) const {return this->polyhedron;}
Polyhedron BrillouinZone::get_ir_polyhedron(const bool true_ir) const {
  // If the ir polyhedron fills the first BZ using the symmetry operations
  // of the pointgroup (plus time inversion, if included), then no ir_mirroring
  // is to be performed. In this case or if the 'true_ir' was requested, return
  // the computed ir_polyhedron
  if (this->no_ir_mirroring || !true_ir) return this->ir_polyhedron;
  // Otherwise, the ir_polyedron is only half of the true (non-convex)
  // irreducible polyhedron, so add the lack of inversion symmetry explicitly.
  return this->ir_polyhedron + this->ir_polyhedron.mirror();
}
bool BrillouinZone::check_ir_polyhedron(void){
  profile_update("Start BrillouinZone::check_ir_polyhedron");
  this->check_if_mirroring_needed(); // move this to end of wedge_brute_force?
  PointSymmetry fullps = this->outerlattice.get_pointgroup_symmetry(this->time_reversal?1:0);
  double volume_goal = this->polyhedron.get_volume() / static_cast<double>(fullps.size());
  Polyhedron irbz = this->get_ir_polyhedron(), rotated;
  if (!brille::approx::scalar(irbz.get_volume(), volume_goal)){
    debug_update("The current 'irreducible' polyhedron has the wrong volume");
    debug_update("Since ",irbz.get_volume()," != ",volume_goal);
    return false;
  }
  /* TODO FIXME -- make a LatticePolyhedron class to handle this? */
  /* Our Polyhedron class holds the vertices, and plane information of a
     convex polyhedron expressed in an orthonormal frame tied to the
     conventional reciprocal space lattice. The conversion from lattice
     vectors to absolute vectors is done with a transformation matrix, B.
     The rotation and rotoinversions, R, stored in the PointSymmetry object
     are expressed in units of the conventional real space lattice.

     Reciprocal space vectors rotate via the transpose of R, Rᵀ; and since the
     polyhedron contains vertices, points, and normals of the form Bx and we
     need to apply Rᵀ to x directly, we want to "rotate" each part of the
     polyhedron by B Rᵀ B⁻¹, e.g., B Rᵀ B⁻¹ B x = B Rᵀ x.
  */
  // double B[9], invB[9], RtinvB[8];
  std::array<double,9> B, invB, RtinvB, BRtinvB;
  this->outerlattice.get_B_matrix(B.data());
  verbose_update("B\n",B);
  brille::utils::matrix_inverse(invB.data(), B.data());
  verbose_update("inverse(B)\n",invB);
  std::array<int,9> Rt;
  // get the operations of the pointgroup which are not 1 or -1
  // keeping -1 would probably be ok, but hopefully it won't hurt to remove it now
  PointSymmetry ps = fullps.higher(1);
  for (size_t i=0; i<ps.size(); ++i){
    Rt = transpose(ps.get(i)); // Rᵀ
    debug_update("\n\n\n\n\ntranspose(R)\n",Rt);
    brille::utils::multiply_matrix_matrix(RtinvB.data(), Rt.data(), invB.data()); // Rᵀ*B⁻¹
    debug_update("transpose(R) inverse(B)\n",RtinvB);
    brille::utils::multiply_matrix_matrix(BRtinvB.data(), B.data(), RtinvB.data()); // B*Rᵀ*B⁻¹
    debug_update("Rotate the polyhedron by\n", BRtinvB);
    rotated = irbz.rotate(BRtinvB);
    debug_update("Checking for intersection ",i);
    if (irbz.intersects(rotated)){
      debug_update("The current 'irreducible' polyhedron intersects with itself and therefore is not correct.");
      return false;
    }
  }
  // volume is right and no intersections
  profile_update("  End BrillouinZone::check_ir_polyhedron");
  return true;
}

// first Brillouin zone
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_vertices(void) const {
  return LQVec<double,brille::ref_ptr_t>::from_invA(this->outerlattice, this->polyhedron.get_vertices());
}
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_primitive_vertices(void) const {
  auto v = this->get_vertices();
  if (this->isprimitive()) v = transform_to_primitive(this->outerlattice, v);
  return v;
}
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_points(void) const {
  return LQVec<double,brille::ref_ptr_t>::from_invA(this->outerlattice, this->polyhedron.get_points());
}
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_primitive_points(void) const {
  LQVec<double,brille::ref_ptr_t> p = this->get_points();
  if (this->isprimitive()) p = transform_to_primitive(this->outerlattice, p);
  return p;
}
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_normals(void) const {
  return LQVec<double,brille::ref_ptr_t>::from_invA(this->outerlattice, this->polyhedron.get_normals());
}
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_primitive_normals(void) const {
  LQVec<double,brille::ref_ptr_t> n = this->get_normals();
  if (this->isprimitive()) n = transform_to_primitive(this->outerlattice, n);
  return n;
}
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_half_edges(void) const{
  return LQVec<double,brille::ref_ptr_t>::from_invA(this->outerlattice, this->polyhedron.get_half_edges());
}
std::vector<std::vector<int>> BrillouinZone::get_faces_per_vertex(void) const {
  return this->polyhedron.get_faces_per_vertex();
}
std::vector<std::vector<int>> BrillouinZone::get_vertices_per_face(void) const {
  return this->polyhedron.get_vertices_per_face();
}
// irreducible first Brillouin zone
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_ir_vertices(void) const {
  Polyhedron irp = this->get_ir_polyhedron();
  return LQVec<double,brille::ref_ptr_t>::from_invA(this->outerlattice, irp.get_vertices());
}
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_ir_primitive_vertices(void) const {
  LQVec<double,brille::ref_ptr_t> v = this->get_ir_vertices();
  if (this->isprimitive()) v = transform_to_primitive(this->outerlattice, v);
  return v;
}
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_ir_points(void) const {
  Polyhedron irp = this->get_ir_polyhedron();
  return LQVec<double,brille::ref_ptr_t>::from_invA(this->outerlattice, irp.get_points());
}
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_ir_primitive_points(void) const {
  LQVec<double,brille::ref_ptr_t> p = this->get_ir_points();
  if (this->isprimitive()) p = transform_to_primitive(this->outerlattice, p);
  return p;
}
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_ir_normals(void) const {
  Polyhedron irp = this->get_ir_polyhedron();
  return LQVec<double,brille::ref_ptr_t>::from_invA(this->outerlattice, irp.get_normals());
}
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_ir_primitive_normals(void) const {
  LQVec<double,brille::ref_ptr_t> n = this->get_ir_normals();
  if (this->isprimitive()) n = transform_to_primitive(this->outerlattice, n);
  return n;
}
std::vector<std::vector<int>> BrillouinZone::get_ir_faces_per_vertex(void) const {
  return this->get_ir_polyhedron().get_faces_per_vertex();
}
std::vector<std::vector<int>> BrillouinZone::get_ir_vertices_per_face(void) const {
  return this->get_ir_polyhedron().get_vertices_per_face();
}

// irreducible reciprocal space
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_ir_wedge_normals(void) const {
  LQVec<double,brille::ref_ptr_t> out(this->outerlattice, 0u);
  if (this->ir_wedge_normals.size(0))
    out = LQVec<double,brille::ref_ptr_t>(this->outerlattice, this->ir_wedge_normals);
  return out;
}
//
LQVec<double,brille::ref_ptr_t> BrillouinZone::get_primitive_ir_wedge_normals(void) const {
  LQVec<double,brille::ref_ptr_t> lqwn(this->outerlattice, 0u);
  if (this->ir_wedge_normals.size(0)){
    lqwn = LQVec<double,brille::ref_ptr_t>(this->outerlattice, this->ir_wedge_normals);
    if (this->isprimitive())
      lqwn = transform_to_primitive(this->outerlattice, lqwn);
  }
  return lqwn;
}


void BrillouinZone::print() const {
  std::string msg = "BrillouinZone with ";
  msg += std::to_string(this->vertices_count()) + " vertices and ";
  msg += std::to_string(this->faces_count()) + " faces";
  std::cout << msg << std::endl;
}


void BrillouinZone::irreducible_vertex_search(){
  using namespace brille;
  /* We need to check for three-plane intersections for all combinations of two
     1st Brillouin zone planes and one irreducible reciprocal space normal and
     two irreducible reciprocal space normals and one 1st Brillouin zone plane.
  */
  size_t Nbz = this->get_normals().size(0);
  size_t Nir = this->ir_wedge_normals.size(0);

  if (0==Nir){
    this->ir_polyhedron = this->polyhedron;
    return;
  }

  // for which there are M*(N*(N-1))/2 + N*(M*(M-1))/2 total possible combinations
  size_t n21 = ((Nbz*(Nbz-1))>>1)*Nir;
  size_t n12 = ((Nir*(Nir-1))>>1)*Nbz;
  size_t n03 = 0;
  for (size_t i=2; i<Nir; ++i) n03 += (i*(i-1))>>1;
  verbose_update("Checking {",n21,", ",n12,", ",n03,"} {2:1, 1:2, 0:3} zone:wedge 3-plane intersection points");

  auto bznormals = this->get_normals();
  auto bzpoints = this->get_points();

  // We will create a polyhedron using (some of) these normals. It is imperitive
  // that the polyhedron normals point *outwards* from the centre of the polyhedron
  // while we've thus far defined the wedge normals to point *into* the irreducible
  // reciprocal space wedge.
  auto irnormals = -1.0*this->get_ir_wedge_normals();
  auto vertices30 = this->get_vertices();
  std::vector<std::vector<int>> i30 = this->get_faces_per_vertex();

  LQVec<double,brille::ref_ptr_t> vertices21(bznormals.get_lattice(), n21);
  Array<int,brille::ref_ptr_t> i21(n21,3);
  LQVec<double,brille::ref_ptr_t> vertices12(bznormals.get_lattice(), n12);
  Array<int,brille::ref_ptr_t> i12(n12,3);
  LQVec<double,brille::ref_ptr_t> vertices03(bznormals.get_lattice(), n03);
  Array<int,brille::ref_ptr_t> i03(n03,3);

  int c21=0, c12=0, c03=0;
  shape_t isub({0,0});
  if (n21){ // protect against Nbz=0, since size_t(0)-1 = 4294967294 or 18446744073709551615 if its 32- or 64-bit
    for (ind_t i=0  ; i<(Nbz-1); ++i)
    for (ind_t j=i+1; j< Nbz   ; ++j)
    for (ind_t k=0  ; k< Nir   ; ++k)
    if (intersect_at(bznormals.view(i), bzpoints.view(i),
                     bznormals.view(j), bzpoints.view(j),
                     irnormals.view(k),                   vertices21, c21)){
      isub[0]=c21++;
      isub[1]=0; i21[isub]=i; isub[1]=1; i21[isub]=j; isub[1]=2; i21[isub]=k;
    }
  }
  if (n12){ // protect against Nir=0, since size_t(0)-1 = 4294967294 or 18446744073709551615 if its 32- or 64-bit
    for (ind_t i=0  ; i< Nbz   ; ++i)
    for (ind_t j=0  ; j<(Nir-1); ++j)
    for (ind_t k=j+1; k< Nir   ; ++k)
    if (intersect_at(bznormals.view(i), bzpoints.view(i),
                     irnormals.view(j),
                     irnormals.view(k),                   vertices12, c12)){
      isub[0]=c12++;
      isub[1]=0; i12[isub]=i; isub[1]=1; i12[isub]=j; isub[1]=2; i12[isub]=k;
    }
  }
  if (n03){
    for (ind_t i=0  ; i<(Nir-2); ++i)
    for (ind_t j=i+1; j<(Nir-1); ++j)
    for (ind_t k=j+1; k< Nir   ; ++k)
    if (intersect_at(irnormals.view(i),
                     irnormals.view(j),
                     irnormals.view(k),                   vertices03, c03)){
      isub[0]=c03++;
      isub[1]=0; i03[isub]=i; isub[1]=1; i03[isub]=j; isub[1]=2; i03[isub]=k;
    }
  }
  verbose_update("Intersections found");
  // make sure we shrink all sets of vertices to just those found!
  // plus remove any intersection points outside of the first Brillouin zone
  this->shrink_and_prune_outside(static_cast<size_t>(c21), vertices21, i21);
  this->shrink_and_prune_outside(static_cast<size_t>(c12), vertices12, i12);
  this->shrink_and_prune_outside(static_cast<size_t>(c03), vertices03, i03);
  verbose_update("Intersections pruned");
  // Now we have four lists of vertices, plus lists of the normal vectors
  // and on-plane points, which define the three planes that intersect at each
  // vertex.
  // We want to combine these lists:

  int max_bz_idx=0, max_ir_idx=0;
  for (auto i: i30) for (int j: i) if (j > max_bz_idx) max_bz_idx = j;
  for (ind_t i=0; i<i21.size(0); ++i){
    isub[0]=i;
    isub[1]=0; if (i21[isub]>max_bz_idx) max_bz_idx = i21[isub];
    isub[1]=1; if (i21[isub]>max_bz_idx) max_bz_idx = i21[isub];
    isub[1]=2; if (i21[isub]>max_ir_idx) max_ir_idx = i21[isub];
  }
  for (ind_t i=0; i<i12.size(0); ++i){
    isub[0]=i;
    isub[1]=0; if (i12[isub]>max_bz_idx) max_bz_idx = i12[isub];
    isub[1]=1; if (i12[isub]>max_ir_idx) max_ir_idx = i12[isub];
    isub[1]=2; if (i12[isub]>max_ir_idx) max_ir_idx = i12[isub];
  }
  for (ind_t i=0; i<i03.size(0); ++i){
    isub[0]=i;
    for (ind_t j=0; j<3u; ++j){
      isub[1]=j; if (i03[isub]>max_ir_idx) max_ir_idx = i03[isub];
    }
  }
  max_bz_idx++; max_ir_idx++; // since we count from 0, and need one more element than the highest index.
  std::vector<bool> bz_face_present(max_bz_idx, false), ir_face_present(max_ir_idx, false);
  for (auto i: i30) for (int j: i) bz_face_present[j] = true;
  for (ind_t i=0; i<i21.size(0); ++i){ isub[0]=i;
    isub[1]=0; bz_face_present[i21[isub]] = true;
    isub[1]=1; bz_face_present[i21[isub]] = true;
    isub[1]=2; ir_face_present[i21[isub]] = true;
  }
  for (ind_t i=0; i<i12.size(0); ++i){ isub[0]=i;
    isub[1]=0; bz_face_present[i12[isub]] = true;
    isub[1]=1; ir_face_present[i12[isub]] = true;
    isub[1]=2; ir_face_present[i12[isub]] = true;
  }
  for (ind_t i=0; i<i03.size(0); ++i){ isub[0]=1;
    for (ind_t j=0; j<3u; ++j){ isub[1]=j;
      ir_face_present[i03[isub]] = true;
    }
  }
  ind_t bz_faces = std::count(bz_face_present.begin(), bz_face_present.end(), true);
  ind_t ir_faces = std::count(ir_face_present.begin(), ir_face_present.end(), true);

  ind_t total_verts;
  total_verts  = vertices30.size(0) + vertices12.size(0);
  total_verts += vertices21.size(0) + vertices03.size(0);

  LQVec<double,brille::ref_ptr_t> all_verts(bznormals.get_lattice(), total_verts);
  LQVec<double,brille::ref_ptr_t> all_norms(bznormals.get_lattice(), bz_faces+ir_faces);
  LQVec<double,brille::ref_ptr_t> all_point(bznormals.get_lattice(), bz_faces+ir_faces);
  Array<int,brille::ref_ptr_t> all_ijk(total_verts,3);

  std::vector<size_t> bz_face_mapped(max_bz_idx, 0u), ir_face_mapped(max_ir_idx, 0u);

  size_t face_idx=0;
  verbose_update("Combine ", i30.size(), " 3:0 normals and plane-points");
  for (auto i: i30) for (int j: i)
    if (0==bz_face_mapped[j]){
      all_norms.set(face_idx, bznormals.view(j));
      all_point.set(face_idx,  bzpoints.view(j));
      bz_face_mapped[j] = ++face_idx; // so that bz_face_mapped is the index+1
    }
  verbose_update("Combine ", i21.size(0), " 2:1 normals and plane-points");
  for (ind_t i=0; i<i21.size(0); ++i){
    isub[0]=i;
    isub[1]=0;
    if (0==bz_face_mapped[i21[isub]]){
      all_norms.set(face_idx, bznormals.view(i21[isub]));
      all_point.set(face_idx,  bzpoints.view(i21[isub]));
      bz_face_mapped[i21[isub]] = ++face_idx;
    }
    isub[1]=1;
    if (0==bz_face_mapped[i21[isub]]){
      all_norms.set(face_idx, bznormals.view(i21[isub]));
      all_point.set(face_idx,  bzpoints.view(i21[isub]));
      bz_face_mapped[i21[isub]] = ++face_idx;
    }
    isub[1]=2;
    if (0==ir_face_mapped[i21[isub]]){
      all_norms.set(face_idx, irnormals.view(i21[isub]));
      all_point.set(face_idx, 0.0*all_point.extract(i21[isub]));
      ir_face_mapped[i21[isub]] = ++face_idx;
    }
  }
  verbose_update("Combine ", i12.size(0), " 1:2 normals and plane-points");
  for (ind_t i=0; i<i12.size(0); ++i){
    isub[0]=i;
    isub[1]=0;
    if (0==bz_face_mapped[i12[isub]]){
      all_norms.set(face_idx, bznormals.view(i12[isub]));
      all_point.set(face_idx,  bzpoints.view(i12[isub]));
      bz_face_mapped[i12[isub]] = ++face_idx;
    }
    isub[1]=1;
    if (0==ir_face_mapped[i12[isub]]){
      all_norms.set(face_idx, irnormals.view(i12[isub]));
      all_point.set(face_idx, 0.0*all_point.extract(i12[isub]));
      ir_face_mapped[i12[isub]] = ++face_idx;
    }
    isub[1]=2;
    if (0==ir_face_mapped[i12[isub]]){
      all_norms.set(face_idx, irnormals.view(i12[isub]));
      all_point.set(face_idx, 0.0*all_point.extract(i12[isub]));
      ir_face_mapped[i12[isub]] = ++face_idx;
    }
  }
  verbose_update("Combine ", i03.size(0), " 0:3 normals and plane-points");
  for (ind_t i=0; i<i03.size(0); ++i){
    isub[0]=i;
    for (ind_t j=0; j<3u; ++j){
      isub[1]=j;
      if (0==ir_face_mapped[i03[isub]]){
        all_norms.set(face_idx, irnormals.view(i03[isub]));
        all_point.set(face_idx, 0.0*all_point.extract(i03[isub]));
        ir_face_mapped[i03[isub]] = ++face_idx;
      }
    }
  }
  verbose_update("Normals and plane-points combined");

  ind_t vert_idx=0;
  shape_t vsub({0,0});
  verbose_update("Combine ", i30.size(), " 3:0 vertices and planes-per-vertex");
  for (ind_t i=0; i<i30.size(); ++i){
    all_verts.set(vert_idx, vertices30.view(i));
    vsub[0] = vert_idx++;
    vsub[1] = 0; all_ijk[vsub] = bz_face_mapped[i30[i][0]]-1;
    vsub[1] = 1; all_ijk[vsub] = bz_face_mapped[i30[i][1]]-1;
    vsub[1] = 2; all_ijk[vsub] = bz_face_mapped[i30[i][2]]-1;
  }
  verbose_update("Combine ", i21.size(0), " 2:1 vertices and planes-per-vertex");
  for (size_t i=0; i<i21.size(0); ++i){
    all_verts.set(vert_idx, vertices21.view(i));
    isub[0] = i; vsub[0] = vert_idx++;
    isub[1] = vsub[1] = 0; all_ijk[vsub] = bz_face_mapped[i21[isub]]-1;
    isub[1] = vsub[1] = 1; all_ijk[vsub] = bz_face_mapped[i21[isub]]-1;
    isub[1] = vsub[1] = 2; all_ijk[vsub] = ir_face_mapped[i21[isub]]-1;
  }
  verbose_update("Combine ", i12.size(0), " 1:2 vertices and planes-per-vertex");
  for (size_t i=0; i<i12.size(0); ++i){
    all_verts.set(vert_idx, vertices12.view(i));
    isub[0] = i; vsub[0] = vert_idx++;
    isub[1] = vsub[1] = 0; all_ijk[vsub] = bz_face_mapped[i12[isub]]-1;
    isub[1] = vsub[1] = 1; all_ijk[vsub] = ir_face_mapped[i12[isub]]-1;
    isub[1] = vsub[1] = 2; all_ijk[vsub] = ir_face_mapped[i12[isub]]-1;
  }
  verbose_update("Combine ", i03.size(0), " 0:3 vertices and planes-per-vertex");
  for (size_t i=0; i<i03.size(0); ++i){
    all_verts.set(vert_idx, vertices03.view(i));
    isub[0] = i; vsub[0] = vert_idx++;
    isub[1] = vsub[1] = 0; all_ijk[vsub] = ir_face_mapped[i03[isub]]-1;
    isub[1] = vsub[1] = 1; all_ijk[vsub] = ir_face_mapped[i03[isub]]-1;
    isub[1] = vsub[1] = 2; all_ijk[vsub] = ir_face_mapped[i03[isub]]-1;
  }
  verbose_update("Vertices and planes-per-vertex combined");
  // four lists now combined into one.
  //      all_ijk   -- (N,3) array of which three planes intersected at a vertex
  //      all_verts -- (N,) vector of intersection vertex locations
  //      all_norms -- (N,) vector of plane normals (indexed by all_ijk)
  //      all_point -- (N,) vector of on-plane points (indexed by all_ijk)

  // Find which vertices are inside the irreducible wedge
  const bool constructing{true}; // here we *are* building-up the irreducible Brillouin zone
  auto keep = this->isinside_wedge(all_verts, constructing);
  // and pull out those vertices and their intersecting plane indices
  all_verts = all_verts.extract(keep);
  all_ijk   = all_ijk.extract(keep);

  // it is imperitive that the xyz coordinate system of the irreducible
  // polyhedron is the same as that used by the Brillouin zone polyhedron.
  // Deal with this in a special function elsewhere.
  //
  // Creating a Polyhedron object automatically keeps only unique vertices
  // and facet planes which are polygons, plus finds the vertex to facet
  // and facet to vertex indexing required for, e.g., plotting
  this->set_ir_polyhedron(all_verts, all_point, all_norms);
  // this->set_ir_polyhedron(all_verts, all_point, all_norms, all_ijk);
  verbose_update("Found a ",this->ir_polyhedron.string_repr());
}

void BrillouinZone::voro_search(const int extent){
  profile_update("Start BrillouinZone::voro_search with ",extent," extent");
  using namespace brille;
  std::array<double, 3> bbmin{1e3,1e3,1e3}, bbmax{-1e3,-1e3,-1e3};
  LQVec<int,brille::ref_ptr_t> primtau(this->lattice, make_relative_neighbour_indices(extent));
  size_t ntau = primtau.size(0);
  std::vector<size_t> perm(ntau);
  std::iota(perm.begin(), perm.end(), 0u); // {0u, 1u, 2u, ..., ntau-1}
  std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b){
    return primtau.norm(a) < primtau.norm(b);
  });
  // verbose_update("unsorted primtau\n",primtau.to_string(),norm(primtau).to_string());
  verbose_update("unsorted primtau\n",cat(1,primtau,norm(primtau)).to_string());
  verbose_update("permutation:",perm);
  primtau.permute(perm);
  verbose_update("sorted primtau\n",cat(1,primtau,norm(primtau)).to_string());
  // the first Brillouin zone polyhedron will be expressed in absolute units
  // in the xyz frame of the conventional reciprocal lattice
  auto tau = transform_from_primitive(this->outerlattice, primtau).get_xyz();
  shape_t ij{0,0};
  for (size_t i=0; i<ntau; ++i){
    ij[0]=i;
    for (size_t j=0; j<3u; ++j){
      ij[1]=j;
      if (tau[ij] < bbmin[j]) bbmin[j] = tau[ij];
      if (tau[ij] > bbmax[j]) bbmax[j] = tau[ij];
    }
  }
  // create an initialize the voro++ voronoicell object
  verbose_update("Construct the bounding polyhedron from ",bbmin," to ",bbmax);
  Polyhedron voronoi = polyhedron_box(bbmin, bbmax);
  verbose_update("Bounding polyhedron with volume = ",voronoi.get_volume());
  // and then use the reciprocal lattice points to subdivide the cell until
  // only the first Brillouin zone is left:
  Polyhedron firstbz = Polyhedron::bisect(voronoi, tau/norm(tau), tau/2.0);
  this->polyhedron = firstbz;
  profile_update("  End BrillouinZone::voro_search with ",extent," extent");
}
