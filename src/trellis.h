#include <vector>
#include <array>
#include <algorithm>
#include <omp.h>
#include "arrayvector.h"
#include "latvec.h"
#include "polyhedron.h"
#include "unsignedtosigned.h"
#include "debug.h"
#include "triangulation_simple.h"

#ifndef _TRELLIS_H_
#define _TRELLIS_H_

template<class T>
static size_t find_bin(const std::vector<T>& bin_edges, const T x){
  auto found_at = std::find_if(bin_edges.begin(), bin_edges.end(), [x](double b){return b>x;});
  return std::distance(bin_edges.begin(), found_at) - 1;
}

enum class NodeType {empty, cube, polyhedron};

class GeneralNode{
public:
  GeneralNode() {}
  virtual ~GeneralNode() = default;
  //
  virtual NodeType type(void) const {return NodeType::empty;}
  virtual size_t vertex_count() const {return 0u;}
  virtual std::vector<size_t> vertices(void) const {return std::vector<size_t>();}
  virtual std::vector<std::array<size_t,4>> vertices_per_tetrahedron(void) const {return std::vector<std::array<size_t,4>>();}
  virtual bool indices_weights(const ArrayVector<double>&, const ArrayVector<double>&, std::vector<size_t>&, std::vector<double>&) const {return false;};
};
class CubeNode: public GeneralNode {
  std::array<size_t, 8> vertex_indices;
public:
  CubeNode(): vertex_indices({0,0,0,0,0,0,0,0}) {}
  CubeNode(const std::array<size_t,8>& vi): vertex_indices(vi) {}
  CubeNode(const std::vector<size_t>& vi): vertex_indices({0,0,0,0,0,0,0,0}) {
    if (vi.size() != 8) throw std::logic_error("CubeNode objects take 8 indices.");
    for (size_t i=0; i<8u; ++i) vertex_indices[i] = vi[i];
  }
  NodeType type(void) const {return NodeType::cube;}
  size_t vertex_count() const { return 8u;}
  std::vector<size_t> vertices(void) const {
    std::vector<size_t> out;
    for (auto v: vertex_indices) out.push_back(v);
    return out;
  }
  bool indices_weights(const ArrayVector<double>&, const ArrayVector<double>&, std::vector<size_t>&, std::vector<double>&) const;
};
class PolyNode: public GeneralNode {
  std::vector<std::array<size_t,4>> vi_t;
public:
  PolyNode() {};
  // actually constructing the tetrahedra from, e.g., a Polyhedron object will
  // need to be done elsewhere
  PolyNode(const std::vector<std::array<size_t,4>>& vit): vi_t(vit) {}
  NodeType type(void) const {return NodeType::polyhedron;}
  // count-up the number fo unique vertices in the tetrahedra-triangulated polyhedron
  size_t vertex_count() const { return this->vertices().size();}
  std::vector<size_t> vertices(void) const {
    std::vector<size_t> out;
    for (auto tet: vi_t) for (auto idx: tet)
    if (std::find(out.begin(), out.end(), idx)==out.end()) out.push_back(idx);
    return out;
  }
  std::vector<std::array<size_t,4>> vertices_per_tetrahedron(void) const {return vi_t;}
  bool indices_weights(const ArrayVector<double>&, const ArrayVector<double>&, std::vector<size_t>&, std::vector<double>&) const;
private:
  bool tetrahedra_contains(const size_t, const ArrayVector<double>&, const ArrayVector<double>&, std::array<double,4>&) const;
};

template<class T> class TrellisData{
  typedef std::vector<size_t> ShapeType;
  typedef std::array<unsigned,4> ElementsType;
  ArrayVector<T> data_;   //!< The stored ArrayVector indexed like the holding-PolyhedronTrellis' vertices
  ShapeType shape_;       //!< A std::vector to indicate a possible higher-dimensional shape of each `data` array
  ElementsType elements_; //!< The number of scalars, normalised eigenvector elements, vector elements, and matrix elements per data array
  size_t branches_;       //!< The number of branches contained per data array
public:
  TrellisData(): data_({0,0}), shape_({0,0}), elements_({0,0,0,0}), branches_(0){};
  size_t size(void) const {return data_.size();}
  size_t numel(void) const {return data_.numel();}
  const ArrayVector<T>& data(void) const {return data_;}
  const ShapeType& shape(void) const {return shape_;}
  const ElementsType& elements(void) const {return elements_;}
  size_t branches(void) const {return branches_;}
  //
  void interpolate_at(const std::vector<size_t>& indicies, const std::vector<double>& weights, ArrayVector<T>& out, const size_t to) const {
    new_unsafe_interpolate_to(data_, elements_, branches_, indicies, weights, out, to);
  }
  void replace_data(const ArrayVector<T>& nd, const ShapeType& ns, const ElementsType& ne){
    data_ = nd;
    shape_ = ns;
    elements_ = ne;
    // check the input for correctness
    size_t total_elements = 1u;
    // scalar + eigenvector + vector + matrix*matrix elements
    size_t known_elements = static_cast<size_t>(ne[0])+static_cast<size_t>(ne[1])+static_cast<size_t>(ne[2])
                          + static_cast<size_t>(ne[3])*static_cast<size_t>(ne[3]);
    // no matter what, shape[0] should be the number of gridded points
    if (ns.size()>2){
      // if the number of dimensions of the shape array is greater than two,
      // the second element is the number of modes per point                    */
      branches_ = ns[1];
      for (size_t i=2u; i<ns.size(); ++i) total_elements *= ns[i];
    } else {
      // shape is [n_points, n_elements] or [n_points,], so there is only one mode
      branches_ = 1u;
      total_elements = ns.size() > 1 ? ns[1] : 1u;
    }
    if (0 == known_elements) elements_[0] = total_elements;
    if (known_elements && known_elements != total_elements){
      std::string msg;
      msg = "Inconsistent elements: " + std::to_string(known_elements) + " = ";
      msg += std::to_string(elements_[0]) + "+" + std::to_string(elements_[1]) + "+";
      msg += std::to_string(elements_[2]) + "+" + std::to_string(elements_[3]) + "² ≠ ";
      msg += std::to_string(total_elements);
      throw std::runtime_error(msg);
    }
  }
  void replace_data(const ArrayVector<T>& nd, const ElementsType& ne=ElementsType({0,0,0,0})){
    ShapeType ns{nd.size(), nd.numel()};
    return this->replace_data(nd, ns, ne);
  }
  // Calculate the Debye-Waller factor for the provided Q points and ion masses
  template<template<class> class A> ArrayVector<double> debye_waller(const A<double>& Q, const std::vector<double>& M, const double t_K) const;
private:
  ArrayVector<double> debye_waller_sum(const LQVec<double>& Q, const double beta) const;
  ArrayVector<double> debye_waller_sum(const ArrayVector<double>& Q, const double t_K) const;
};

template<typename T> class PolyhedronTrellis{
  Polyhedron polyhedron_;                        //!< the Polyhedron bounding the Trellis
  TrellisData<T> data_;                          //!< [optional] data stored at each Trellis vertex
  ArrayVector<double> vertices_;                 //!< The Trellis intersections inside the bounding Polyhedron
  std::vector<size_t> node_index_;               //!< For each Trellis node, the index into its respective NodeType vector
  std::vector<NodeType> node_type_;              //!< The NodeType for each Trellis node
  std::vector<CubeNode> cube_nodes_;             //!< The node objects for each unrestricted Trellis node
  std::vector<PolyNode> poly_nodes_;             //!< The node objects for each Polyheadron-bounded Trellis node
  std::array<std::vector<double>,3> boundaries_; //!< The coordinates of the Trellis intersections, which bound the Trellis nodes
public:
  explicit PolyhedronTrellis(const Polyhedron& polyhedron, const double node_fraction=1.0);
  // PolyhedronTrellis(const Polyhedron& polyhedron, const size_t node_ratio);
  PolyhedronTrellis(): vertices_({3,0}) {
    std::vector<double> everything;
    everything.push_back(std::numeric_limits<double>::lowest());
    everything.push_back((std::numeric_limits<double>::max)());
    this->boundaries(everything, everything, everything);
  }
  size_t expected_vertex_count() const {
    size_t count = 1u;
    for (size_t i=0; i<3u; ++i) count *= boundaries_[i].size();
    return count;
  }
  size_t vertex_count() const { return vertices_.size(); }
  const ArrayVector<double>& vertices(void) const { return vertices_; }
  const ArrayVector<double>& vertices(const ArrayVector<double>& v){
    if (v.numel()==3) vertices_ = v;
    return vertices_;
  }
  ArrayVector<double> cube_vertices(void) const {
    std::vector<bool> keep(vertices_.size(), false);
    for (auto node: cube_nodes_) for (auto idx: node.vertices()) keep[idx]=true;
    return vertices_.extract(keep);
  }
  ArrayVector<double> poly_vertices(void) const {
    std::vector<bool> keep(vertices_.size(), false);
    for (auto node: poly_nodes_) for (auto idx: node.vertices()) keep[idx]=true;
    return vertices_.extract(keep);
  }
  std::vector<std::array<size_t,4>> vertices_per_tetrahedron(void) const {
    std::vector<std::array<size_t,4>> out;
    for (auto node: poly_nodes_)
    for (auto tet: node.vertices_per_tetrahedron()) out.push_back(tet);
    return out;
  }
  bool indices_weights(const ArrayVector<double>& x, std::vector<size_t>& indices, std::vector<double>& weights) const {
    if (x.size()!=1u || x.numel()!=3u)
      throw std::runtime_error("The indices and weights can only be found for one point at a time.");
    size_t idx = this->node_index(x);
    switch (node_type_[idx]){
      case NodeType::cube:
      info_update("Interpolate at ",x.to_string("")," between cube vertices\n",cube_nodes_[idx].vertices());
      return cube_nodes_[idx].indices_weights(vertices_, x, indices, weights);
      case NodeType::polyhedron:
      info_update("Interpolate at ",x.to_string("")," between polyhedron vertices\n",poly_nodes_[idx].vertices());
      return cube_nodes_[idx].indices_weights(vertices_, x, indices, weights);
      default:
      return false;
    }
  }
  template<class R> unsigned check_before_interpolating(const ArrayVector<R>& x) const{
    unsigned int mask = 0u;
    if (this->data_.size()==0)
      throw std::runtime_error("The trellis must be filled before interpolating!");
    if (x.numel()!=3u)
      throw std::runtime_error("PolyhedronTrellis requires x values which are three-vectors.");
    return mask;
  }
  ArrayVector<T> interpolate_at(const ArrayVector<double>& x) const {
    this->check_before_interpolating(x);
    ArrayVector<T> out(data_.numel(), x.size());
    std::vector<size_t> indices;
    std::vector<double> weights;
    for (size_t i=0; i<x.size(); ++i){
      verbose_update("Locating ",x.to_string(i));
      if (!this->indices_weights(x.extract(i), indices, weights))
        throw std::runtime_error("Point not found in PolyhedronTrellis");
      verbose_update("Interpolate between vertices ", indices," with weights ",weights);
      data_.interpolate_at(indices, weights, out, i);

    }
    return out;
  }
  ArrayVector<T> interpolate_at(const ArrayVector<double>& x, const int threads) const {
    this->check_before_interpolating(x);
    omp_set_num_threads( (threads > 0) ? threads : omp_get_max_threads() );
    // shared between threads
    ArrayVector<T> out(data_.numel(), x.size());
    // private to each thread
    std::vector<size_t> indices;
    std::vector<double> weights;
    // OpenMP < v3.0 (VS uses v2.0) requires signed indexes for omp parallel
    long xsize = unsigned_to_signed<long, size_t>(x.size());
  #pragma omp parallel for shared(x, out) private(indices, weights)
    for (long si=0; si<xsize; ++si){
      size_t i = signed_to_unsigned<size_t, long>(si);
      if (!this->indices_weights(x.extract(i), indices, weights))
        throw std::runtime_error("Point not found in PolyhedronTrellis");
      data_.interpolate_at(indices, weights, out, i);
    }
    return out;
  }
  size_t node_count() {
    size_t count = 1u;
    for (size_t i=0; i<3u; ++i) count *= boundaries_[i].size()-1;
    return count;
  }
  std::array<size_t,3> size() const {
    std::array<size_t,3> s;
    for (size_t i=0; i<3u; ++i) s[i] = boundaries_[i].size()-1;
    return s;
  }
  std::array<size_t,3> span() const {
    std::array<size_t,3> s{1,0,0}, sz=this->size();
    for (size_t i=1; i<3; ++i) s[i] = sz[i-1]*s[i-1];
    return s;
  }
  const std::array<std::vector<double>,3>& boundaries(void) const {return boundaries_;}
  // Find the appropriate node for an arbitrary point:
  std::array<size_t,3> node_subscript(const std::array<double,3>& p) const {
    std::array<size_t,3> sub{0,0,0}, sz=this->size();
    for (size_t dim=0; dim<3u; ++dim) sub[dim] = find_bin(boundaries_[dim], p[dim]);
    return sub;
  }
  std::array<size_t,3> node_subscript(const ArrayVector<double>& p) const {
    std::array<size_t,3> sub{0,0,0};
    for (size_t dim=0; dim<3u; ++dim){
      // find the first boundary no larger than the value
      sub[dim] = find_bin(boundaries_[dim], p.getvalue(0, dim));
      info_update("value ",p.getvalue(0, dim)," falls into bin ",sub[dim]," for bins with boundaries\n",boundaries_[dim]);
    }
    info_update("point ",p.to_string("")," has indices ",sub);
    return sub;
  }
  // find the node linear index for a point
  template <class R> size_t node_index(const R& p) const { return this->sub2idx(this->node_subscript(p)); }

  const CubeNode& cube_node(const size_t idx) const {
    if (idx >= node_index_.size() || NodeType::cube != node_type_[idx])
      throw std::runtime_error("Out-of-bounds or non-cube node");
    return cube_nodes_[node_index_[idx]];
  }
  const PolyNode& poly_node(const size_t idx) const {
    if (idx >= node_index_.size() || NodeType::polyhedron != node_type_[idx])
      throw std::runtime_error("Out-of-bounds or non-polyhedron node");
    return poly_nodes_[node_index_[idx]];
  }

  std::string to_string(void) const {
    std::string str = "(";
    for (auto i: this->size()) str += " " + std::to_string(i);
    str += " )";
    return str;
  }
  // Get a constant reference to the stored data
  const TrellisData<T>& data(void) const {return data_;}
  // Replace the data stored in the object
  template<typename... A> void replace_data(A... args) { data_.replace_data(args...); }
  // Calculate the Debye-Waller factor for the provided Q points and ion masses
  template<template<class> class A>
  ArrayVector<double> debye_waller(const A<double>& Q, const std::vector<double>& M, const double t_K) const{
    return data_.debye_waller(Q,M,t_K);
  }
private:
  size_t sub2idx(const std::array<size_t,3>& sub) const {
    return this->sub2idx(sub, this->span());
  }
  std::array<size_t,3> idx2sub(const size_t idx) const {
    return this->idx2sub(idx, this->span());
  }
  size_t sub2idx(const std::array<size_t,3>& sub, const std::array<size_t,3>& sp) const {
    size_t idx=0;
    for (size_t dim=0; dim<3u; ++dim) idx += sp[dim]*sub[dim];
    info_update("span ",sp," gives subscript ",sub," as linear ",idx);
    return idx;
  }
  std::array<size_t,3> idx2sub(const size_t idx, const std::array<size_t,3>& sp) const {
    std::array<size_t,3> sub{0,0,0};
    size_t rem{idx};
    for (size_t dim=3u; dim--;){
      sub[dim] = rem/sp[dim];
      rem -= sub[dim]*sp[dim];
    }
    return sub;
  }
};

#include "trellis.hpp"
#endif