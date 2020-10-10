#ifndef ARRAY_HPP
#define ARRAY_HPP

#include <functional>
#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include <math.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>

#include "subscript.hpp"
#include "utilities.hpp"
#include "comparisons.hpp"
#include "approx.hpp"
#include "types.hpp"

namespace brille {

// struct ref_ptr{
//   using type = char;
// };
// using ref_ptr_t = typename ref_ptr::type; // use char by default since it's small

using ref_ptr_t = char;

/*! \brief A multidimensional shared data array with operator overloads

The `Array<T,P>`` object holds a multidimensional shared data object plus
information necessary to index into that array with an offset, different shape,
and/or different stride than the underlying data contains.

The `Array<T,P>` object provides overloads for basic operations, a number of
utility methods, and methods to create offset and/or reduced-range mutable
or immutable views into the shared array.
*/
template<class T, class P=ref_ptr_t>
class Array{
public:
  using ref_t = std::shared_ptr<P>;
protected:
  T*    _data;     //! A (possilby shared) raw pointer
  ind_t _num;      //! The number of elements stored under the raw pointer
  bool  _own;      //! Whether we need to worry about memory management of the pointer
  ref_t _ref;      //! A shared_ptr for reference counting an owned raw pointer
  bool  _mutable;  //! Whether we are allowed to modify the memory under the pointer
  shape_t _offset; //! A multidimensional offset different from {0,…,0} if a view
  shape_t _shape;  //! The multidimensional shape of this view, prod(_shape) ≤ _num
  shape_t _stride; //! The stride of our multidimensional view
public:
  // accessors
  T* data() {return _data;}
  T* data(const ind_t& idx){ return _data + this->l2l_d(idx);}
  T* data(const shape_t& idx){ return _data + this->s2l_d(idx);}
  const T* data() const {return _data;}
  const T* data(const ind_t& idx) const {return _data + this->l2l_d(idx);}
  const T* data(const shape_t& idx) const {return _data + this->s2l_d(idx);}
  ind_t raw_size() const {return _num;}
  bool own() const {return _own;}
  ref_t ref() const {return _ref;}
  bool  ismutable(void) const {return _mutable;}
  bool  isconst(void) const {return !_mutable;}
  ind_t ndim(void) const {return _shape.size();}
  ind_t numel(void) const {
    auto nel = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<>());
    return this->ndim() ? static_cast<ind_t>(nel) : 0u;
  }
  shape_t offset(void) const {return _offset;}
  ind_t size(const ind_t dim) const {
    assert(dim < _shape.size());
    return _shape[dim];
  }
  shape_t shape(void) const {return _shape;}
  shape_t stride(void) const {return _stride;}
  shape_t cstride() const {
    shape_t cs(_stride);
    for (auto& s: cs) s *= sizeof(T);
    return cs;
  }
  bool is_column_ordered() const { // stride {1,2,4,8} is column ordered
    //return std::is_sorted(_stride.begin(), _stride.end(), [](ind_t a, ind_t b){return a <= b;});
    for (size_t i=1; i<_stride.size(); ++i) if (_stride[i] < _stride[i-1]) return false;
    return true;
  }
  bool is_row_ordered() const { // stride {8,4,2,1} is row ordered
    // is_sorted doesn't like the vector {1,1} with this predicate?!
    //return std::is_sorted(_stride.begin(), _stride.end(), [](ind_t a, ind_t b){return a >= b;});
    for (size_t i=1; i<_stride.size(); ++i) if (_stride[i] > _stride[i-1]) return false;
    return true;
  }
  bool is_contiguous() const {
    shape_t expected(1, 1);
    if (this->is_row_ordered()){
      for (auto s = _shape.rbegin(); s!=_shape.rend(); ++s)
        expected.push_back(expected.back()*(*s));
      expected.pop_back();
      return std::equal(_stride.begin(), _stride.end(), expected.rbegin());
    }
    if (this->is_column_ordered()){
      for (auto s : _shape) expected.push_back(expected.back()*s);
      // expected has one too many elements
      return std::equal(_stride.begin(), _stride.end(), expected.begin());
    }
    return false;
  }
  // empty initializer
  explicit Array()
  : _data(nullptr), _num(0), _own(false), _ref(std::make_shared<P>()),
  _mutable(false), _offset({0}), _shape({0}), _stride({1})
  {}
  // 1D initializer
  Array(T* data, const ind_t num, const bool own, const bool mut=true)
  : _data(data), _num(num), _own(own), _ref(std::make_shared<P>()),
    _mutable(mut), _offset({0}), _shape({num}), _stride({1})
  {
    this->init_check();
  }
  // ND initializer
  Array(T* data, const ind_t num, const bool own,
    const shape_t& shape, const shape_t& stride, const bool mut=true)
  : _data(data), _num(num), _own(own), _ref(std::make_shared<P>()),
    _mutable(mut), _offset(shape.size(),0), _shape(shape), _stride(stride)
  {
    this->init_check();
  }
  // ND initializer with reference-counter specified (used in pybind11 wrapper)
  Array(T* data, const ind_t num, const bool own, ref_t ref,
    const shape_t& shape, const shape_t& stride, const bool mut=true)
  : _data(data), _num(num), _own(own), _ref(ref),
    _mutable(mut), _offset(shape.size(),0), _shape(shape), _stride(stride)
  {
    this->init_check();
  }
  // ND view constructor
  Array(T* data, const ind_t num, const bool own, const ref_t& ref,
    const shape_t& offset, const shape_t& shape, const shape_t& stride, const bool mut=true)
  : _data(data), _num(num), _own(own), _ref(ref), _mutable(mut), _offset(offset),
    _shape(shape), _stride(stride)
  {
    this->init_check();
  }
  // 2D allocate new memory constructor
  Array(const ind_t s0, const ind_t s1)
  : _mutable(true), _offset({0,0}), _shape({s0,s1}), _stride({s1,1})
  {
    this->construct();
    this->init_check();
  }
  // 2D initialize new memory constructor
  Array(const ind_t s0, const ind_t s1, const T init)
  : _mutable(true), _offset({0,0}), _shape({s0,s1}), _stride({s1,1})
  {
    this->construct(init);
    this->init_check();
  }
  // ND allocate new memory constructor
  Array(const shape_t& shape)
  : _mutable(true), _offset(shape.size(), 0), _shape(shape)
  {
    this->set_stride();
    this->construct();
    this->init_check();
  }
  // ND initialize new memory constructor
  Array(const shape_t& shape, const T init)
  : _mutable(true), _offset(shape.size(), 0), _shape(shape)
  {
    this->set_stride();
    this->construct(init);
    this->init_check();
  }
  // ND allocate new memory with specified stride constructor
  Array(const shape_t& shape, const shape_t& stride)
  : _mutable(true), _offset(shape.size(), 0), _shape(shape), _stride(stride)
  {
    this->construct();
    this->init_check();
  }
  // ND initialize new memory with specified stride constructor
  Array(const shape_t& shape, const shape_t& stride, const T init)
  : _mutable(true), _offset(shape.size(), 0), _shape(shape), _stride(stride)
  {
    this->construct(init);
    this->init_check();
  }
  // otherwise possibly ambiguous constructor from std::vector
  static Array<T,P> from_std(const std::vector<T>& data){
    ind_t num = static_cast<ind_t>(data.size());
    T* d = new T[num]();
    for (ind_t i=0; i<num; ++i) d[i] = data[i];
    // take ownership of d while creating the Array
    return Array<T,P>(d, num, true);
  }
  template<class R, size_t Nel>
  static Array<T,P> from_std(const std::vector<std::array<R,Nel>>& data){
    shape_t shape{static_cast<ind_t>(data.size()), static_cast<ind_t>(Nel)};
    shape_t stride{shape[1], 1};
    ind_t num = shape[0]*shape[1];
    T* d = new T[num]();
    ind_t x{0};
    for (ind_t i=0; i<shape[0]; ++i) for (ind_t j=0; j<shape[1]; ++j)
      d[x++] = static_cast<T>(data[i][j]);
    return Array<T,P>(d, num, true, shape, stride);
  }

  // Rule-of-five definitions since we may not own the associated raw array:
  Array(const Array<T,P>& o)
  : _data(o._data), _num(o._num), _own(o._own), _ref(o._ref),
    _mutable(o._mutable), _offset(o._offset), _shape(o._shape), _stride(o._stride)
  {}
  ~Array(){
    if (_own && _ref.use_count()==1 && _data != nullptr) delete[] _data;
  }
  Array<T,P>& operator=(const Array<T,P>& other){
    if (this != &other){
      if (_own){
        T* old_data = _data;
        ref_t old_ref = _ref;
        _data = other._data;
        _ref = other._ref;
        if (old_ref.use_count()==1 && old_data != nullptr) delete[] old_data;
      } else {
        _data = other._data;
        _ref = other._ref;
      }
      _own = other.own();
      _num = other.raw_size();
      _mutable = other.ismutable();
      _offset = other.offset();
      _shape = other.shape();
      _stride = other.stride();
    }
    return *this;
  }
  // Array(Array<T,P>&& other) noexcept
  // : _data(std::exchange(other._data, nullptr))
  // {
  //   std::swap(_num, other._num);
  //   std::swap(_own, other._own);
  //   std::swap(_ref, other._ref);
  //   std::swap(_mutable, other._mutable);
  //   std::swap(_offset, other._offset);
  //   std::swap(_shape, other._shape);
  //   std::swap(_stride, other._stride);
  // }
  // Array<T,P>& operator=(Array<T,P>&& other) noexcept {
  //   std::swap(_data, other._data);
  //   std::swap(_num, other._num);
  //   std::swap(_own, other._own);
  //   std::swap(_ref, other._ref);
  //   std::swap(_mutable, other._mutable);
  //   std::swap(_offset, other._offset);
  //   std::swap(_shape, other._shape);
  //   std::swap(_stride, other._stride);
  //   return *this;
  // }


  // type casting requires a copy
  // the reference pointer type does not need to be P since a new raw array is made
  // handles also Array<T,P>(Array<T,Q>&) reference type conversion
  template<class R, class RP>
  Array(const Array<R,RP>& other)
  : _mutable(true), _offset(other.ndim(),0), _shape(other.shape())
  {
    this->set_stride();
    this->construct();
    for (auto x: SubIt(_shape)) _data[s2l_d(x)] = static_cast<T>(other[x]);
  }
  template<class R, class RP>
  Array<T,P>& operator=(const Array<R,RP>& other){
    _mutable = true;
    _shape = other.shape();
    _offset = shape_t(_shape.size(), 0);
    this->set_stride();
    this->construct();
    for (auto x: SubIt(_shape)) _data[s2l_d(x)] = static_cast<T>(other[x]);
    return *this;
  }

  // modifiers
  bool make_mutable() {_mutable = true; return _mutable;}
  bool make_immutable() {_mutable = false; return _mutable;}
  Array<T,ref_ptr_t> decouple() const {
    if (!_own || _ref.use_count() > 1) return this->_decouple();
    return *this;
  }

  // data accessors
        T& operator[](ind_t    lin)       {return _data[this->l2l_d(lin)];}
  const T& operator[](ind_t    lin) const {return _data[this->l2l_d(lin)];}

        T& operator[](shape_t& sub)       {return _data[this->s2l_d(sub)];}
  const T& operator[](shape_t& sub) const {return _data[this->s2l_d(sub)];}
protected: // so inherited classes can calculate subscript indexes into their data
  ind_t l2l_d(const ind_t l) const {
    shape_t sub = lin2sub(l, _stride);
    return offset_sub2lin(_offset, sub, _stride);
  }
  ind_t s2l_d(const shape_t& s) const {
    assert(s.size() == _offset.size());
    return offset_sub2lin(_offset, s, _stride);
  }
private:
  ind_t size_from_shape(const shape_t& s) const {
    // std::reduce can perform the same operation in parallel, but isn't implemented
    // in the gcc libraries until v9.3. Since the number of dimensions is (probably)
    // small a serial algorithm is not a giant speed hit.
    size_t sz = std::accumulate(s.begin(), s.end(), ind_t(1), std::multiplies<ind_t>());
    return static_cast<ind_t>(sz);
  }
  ind_t size_from_shape() const {return this->size_from_shape(_shape);}
  void construct() {
    _num = this->size_from_shape();
    if (_num > 0){
      _ref = std::make_shared<P>();
      _data = new T[_num]();
      _own = true;
    } else {
      _data = nullptr;
      _own = false;
    }
  }
  void construct(const T init){
    this->construct();
    if (_num > 0 && _data != nullptr) std::fill(_data, _data+_num, init);
  }
  void set_stride(void){
    _stride.clear();
    _stride.push_back(1);
    for (auto s = _shape.rbegin(); s!=_shape.rend(); ++s)
      _stride.push_back(_stride.back()*(*s));
    _stride.pop_back();
    std::reverse(_stride.begin(), _stride.end());
  }
  void init_check(void){
    if (_shape.size() != _offset.size() || _shape.size() != _stride.size())
    throw std::runtime_error("Attempting to construct Array with inconsistent offset, shape, and strides");
    shape_t sh;
    for (size_t i=0; i<_shape.size(); ++i) sh.push_back(_offset[i]+_shape[i]);
    ind_t offset_size = this->size_from_shape(sh);
    if (_num < offset_size) {
      std::string msg = "The offset { ";
      for (auto x: _offset) msg += std::to_string(x) + " ";
      msg += "} and size { ";
      for (auto x: _shape) msg += std::to_string(x) + " ";
      msg += "} of an Array must not exceed the allocated pointer size ";
      msg += std::to_string(_num);
      throw std::runtime_error(msg);
    }
  }
  shape_t calculate_stride(const shape_t& sh) const {
    shape_t st(sh.size(), 1u);
    if (_stride.front() < _stride.back()){
      for (size_t i=1; i<st.size(); ++i) st[i] = st[i-1]*sh[i-1];
    } else {
      for (size_t i=st.size()-1; i--;) st[i] = st[i+1]*sh[i+1];
    }
    return st;
  }
  void reset_stride(){
    if (!this->is_contiguous())
      throw std::runtime_error("Re-calculating non-contigous strides is not yet working");
    _stride = this->calculate_stride(_shape);
  }
  Array<T,ref_ptr_t> _decouple() const {
    ind_t nnum = this->size_from_shape(_shape);
    T* new_data = new T[nnum]();
    if (nnum == _num) {
      std::copy(_data, _data+_num, new_data);
    } else {
      //subscript conversion necessary due to offset
      //                                   vvv(no offset)vvvv          vvv(offset)vvv
      for (auto x: SubIt(_shape)) new_data[sub2lin(x,_stride)] = _data[this->s2l_d(x)];
    }
    bool new_own = true; // always take ownership of C++ allocated memory
    auto new_ref = std::make_shared<ref_ptr_t>(); // always use the default with C++ created arrays
    bool new_mut = true; // new allocated memory should be mutable
    return Array<T,ref_ptr_t>(new_data, nnum, new_own, new_ref, _shape, _stride, new_mut);
  }
public:
  // sub-array access
  Array<T,P> view() const; // whole array non-owning view
  //! View a single sub-array at index `i` or an offset-smaller array if single=false;
  Array<T,P> view(ind_t i) const;
  //! View the sub-arrays from `i` to `j-1`
  Array<T,P> view(ind_t i, ind_t j) const;
  Array<T,P> view(const shape_t&) const;
  // duplication of one or more sub-arrays:
  Array<T,ref_ptr_t>
  extract(ind_t i) const;
  template<class I, class IP>
  std::enable_if_t<std::is_integral_v<I>, Array<T,ref_ptr_t>>
  extract(const Array<I,IP>& i) const;
  template<typename I>
  std::enable_if_t<std::is_integral_v<I>, Array<T,ref_ptr_t>>
  extract(const std::vector<I>& i) const;
  template<typename I, size_t Nel>
  std::enable_if_t<std::is_integral_v<I>, Array<T,ref_ptr_t>>
  extract(const std::array<I,Nel>& i) const;
  template<typename I, size_t Nel>
  std::enable_if_t<std::is_integral_v<I>, Array<T,ref_ptr_t>>
  extract(const std::vector<std::array<I,Nel>>& i) const;
  template<class RP>
  Array<T,ref_ptr_t>
  extract(const Array<bool,RP>& i) const;
  Array<T,ref_ptr_t>
  extract(const std::vector<bool>& i) const;
  template<class RP>
  bool set(const ind_t i, const Array<T,RP>& in);
  template<class R, class RP>
  bool set(const ind_t i, const Array<R,RP>& in);
  bool set(const ind_t i, const std::vector<T>& in);
  template<size_t Nel> bool set(const ind_t i, const std::array<T,Nel>& in);
  T set(const shape_t& sub, T in);
  template<class RP>
  Array<T,P>& append(const ind_t, const Array<T,RP>&);
  std::string to_string() const;
  std::string to_string(const ind_t) const;

  Array<T,P>& reshape(const shape_t& ns);
  Array<T,P>& resize(const shape_t&, T init=T(0));
  template<class I> Array<T,P>& resize(const I, T init=T(0));
  bool all(ind_t n=0) const;
  bool any(ind_t n=0) const;
  ind_t count(ind_t n=0) const;
  ind_t first(ind_t n=0) const;
  ind_t last(ind_t n=0) const;
  // bool all() const;
  // bool any() const;
  // ind_t count() const;
  // ind_t first() const;
  // ind_t last() const;
  bool all(T val, ind_t n=0) const;
  bool any(T val, ind_t n=0) const;
  ind_t count(T val, ind_t n=0) const;
  ind_t first(T val, ind_t n=0) const;
  ind_t last(T val, ind_t n=0) const;
  Array<int,ref_ptr_t> round() const;
  Array<int,ref_ptr_t> floor() const;
  Array<int,ref_ptr_t> ceil() const;
  Array<T,ref_ptr_t> sum(ind_t dim=0) const;
  Array<T,ref_ptr_t> prod(ind_t dim=0) const;
  Array<T,ref_ptr_t> min(ind_t dim=0) const;
  Array<T,ref_ptr_t> max(ind_t dim=0) const;
  T sum() const;
  T prod() const;
  template<class R, size_t Nel>
  bool match(ind_t i, ind_t j, const std::array<R,Nel>& rot, int order=1) const;
  bool match(ind_t i, ind_t j, ops op=ops::plus, T val=T{0}) const;
  bool all(cmp expr, T val) const;
  bool any(cmp expr, T val) const;
  ind_t first(cmp expr, T val) const;
  ind_t last(cmp expr, T val) const;
  ind_t count(cmp expr, T val) const;
  Array<bool,ref_ptr_t> is(cmp expr, T val) const;
  std::vector<ind_t> find(cmp expr, T val) const;
  template<class R, class RP>
  Array<bool,ref_ptr_t>
  is(cmp expr, const Array<R,RP>& that) const;
  template<class R>
  std::vector<bool>
  is(cmp expr, const std::vector<R>& val) const;
  template<class R, class RP>
  bool
  is(const Array<R,RP>& that) const;
  std::vector<bool> is_unique() const;
  std::vector<ind_t> unique_idx() const;
  Array<T,ref_ptr_t> unique() const;
  Array<T,ref_ptr_t>  operator-() const;
  Array<T,P>& operator +=(const T&);
  Array<T,P>& operator -=(const T&);
  Array<T,P>& operator *=(const T&);
  Array<T,P>& operator /=(const T&);
  template<class R, class RP> Array<T,P>& operator +=(const Array<R,RP>&);
  template<class R, class RP> Array<T,P>& operator -=(const Array<R,RP>&);
  template<class R, class RP> Array<T,P>& operator *=(const Array<R,RP>&);
  template<class R, class RP> Array<T,P>& operator /=(const Array<R,RP>&);
  T dot(ind_t i, ind_t j) const;
  T norm(ind_t i) const;
  template<typename I, typename=std::enable_if_t<std::is_integral<I>::value>>
  void permute(std::vector<I>& p);
  bool swap(ind_t a, ind_t b);
  bool swap(ind_t i, ind_t a, ind_t b);
  std::vector<T> to_std() const;
  T* ptr(const ind_t i0);
  T* ptr(const shape_t& partial_subscript);
  const T* ptr(const ind_t i0) const;
  const T* ptr(const shape_t& partial_subscript) const;
  T& val(const ind_t i0);
  T& val(const shape_t& partial_subscript);
  template<typename I> T& val(std::initializer_list<I> l);
  const T& val(const ind_t i0) const;
  const T& val(const shape_t& partial_subscript) const;
  template<typename I> const T& val(std::initializer_list<I> l) const;

  Array<T,P> contiguous_copy() const;
  // ^^^^^^^^^^ IMPLEMENTED  ^^^^^^^^^^^vvvvvvvvv TO IMPLEMENT vvvvvvvvvvvvvvvvv

};

template<class T,class P>
class ArrayIt {
public:
  Array<T,P> array;
  SubIt<ind_t> subit;
public:
  // constructing with array(a) does not copy the underlying data:
  explicit ArrayIt()
  : array(), subit()
  {}
  ArrayIt(const Array<T,P>& a, const SubIt<ind_t>& s)
  : array(a), subit(s)
  {}
  ArrayIt(const Array<T,P>& a)
  : array(a), subit(a.shape()) // initialises to first element, e.g., {0,…,0}
  {}
  //
  ArrayIt<T,P> begin() const {
    return ArrayIt(array);
  }
  ArrayIt<T,P> end() const {
    return ArrayIt(array, subit.end());
  }
  ArrayIt<T,P>& operator++() {
    ++subit;
    return *this;
  }
  const SubIt<ind_t>& iterator() const {return subit;}
  template<class Q>
  bool operator==(const ArrayIt<T,Q>& other) const {
    // add checking to ensure array and other.array point to the same data?
    return subit == other.iterator();
  }
  template<class Q>
  bool operator!=(const ArrayIt<T,Q>& other) const {
    return subit != other.iterator();
  }
  const T& operator*()  const {return array[*subit];}
  const T* operator->() const {return &(array[*subit]);}
  T& operator*()  {return array[*subit];}
  T* operator->() {return &(array[*subit]);}
};


#include "array.tpp"
} // end namespace brille
#endif // ARRAY_HPP
