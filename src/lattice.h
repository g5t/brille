/*! \file */
#ifndef _LATTICE_CLASS_H_
#define _LATTICE_CLASS_H_

#include <assert.h>
// #include <vector>
// #include "linear_algebra.h"
#include "primitive.h"


// forward declare the two types of lattices so that they can be mutually-referential
class Lattice;
class Direct;
class Reciprocal;

enum class AngleUnit { not_provided, radian, degree, pi };

template<class T, class I> void latmat_to_lenang(const T* latmat, const I c, const I r, T* len, T* ang){
  T n[9];
  // compute the dot product of each row with itself
  for (int i=0; i<3; ++i)  for (int j=0; j<3; ++j)  len[i] += latmat[i*c+j*r]*latmat[i*c+j*r];
  // the lattice vector lengths are the square root of this
  for (int i=0; i<3; ++i) len[i] = std::sqrt(len[i]);
  // normalize the row vectors, leaving only angle information
  for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) n[i*c+j*r] = latmat[i*c+j*r]/len[i];
  // take the dot product between cyclically permuted rows: 0=1⋅2, 1=2⋅0, 2=0⋅1
  for (int i=0; i<3; ++i) for (int j=0; j<3; ++j)  ang[i] += n[c*((i+1)%3)+j*r]*n[c*((i+2)%3)+j*r];
  // the lattice angles are the arccosines of these dot products of normalized lattice vectors
  for (int i=0; i<3; ++i) ang[i] = std::acos(ang[i]);
}

/*! \brief A class to hold information about a space-spanning lattice in three dimensions

A space-spanning lattice in N dimensions has N basis vectors which can be
described fully by their N lengths and the `sum(1:N-1)` angles between each set
of basis vectors, or `sum(1:N)` scalars in total. Storing only their lengths and
angles is therefore always more efficient than storing all N² components of the
basis vectors in an orthonormal frame.
This class stores the 3 lengths and 3 angles required to describe a
3 dimensional space-spanning lattice, plus the volume of the lattice unit cell
and an optional Hall number if a non-Primitive lattice is desired.
*/
class Lattice{
protected:
  double len[3]; //!< basis vector lengths
  double ang[3]; //!< basis vector angles ordered θ₁₂, θ₀₂, θ₀₁, in radian
  double volume; //!< volume of the unit cell formed by the basis vectors
  int hall;      //!< Hall number of the non-Primitive lattice (`hall>1`)
protected:
  double unitvolume() const;
  Lattice inner_star() const;
  template<class I>
  void set_len_pointer(const double *lvec, const I span){
    for (int i=0;i<3;i++) this->len[i] = lvec[i*span];
  }
  template<class I>
  void set_ang_pointer(const double *avec, const I span, const AngleUnit angle_unit){
    AngleUnit au = angle_unit;
    if (au == AngleUnit::not_provided){
      double minang = (std::numeric_limits<double>::max)();
      double maxang = (std::numeric_limits<double>::lowest)();
      for (int i=0; i<3; ++i){
        if (avec[i*span] < minang) minang = avec[i*span];
        if (avec[i*span] > maxang) maxang = avec[i*span];
      }
      if (minang < 0.) throw std::runtime_error("Unexpected negative inter-facial cell angle");
      // 1 is not a good separator between π-radian and radian, since 1 radian ≈ 57.3°
      // au = (maxang < 1.0) ? AngleUnit::pi : (maxang < 2.0*PIOVERTWO) ? AngleUnit::radian : AngleUnit::degree;
      au = (maxang < 2.0*PIOVERTWO) ? AngleUnit::radian : AngleUnit::degree;
    }
    double conversion = (AngleUnit::radian == au) ? 1.0 : PIOVERTWO*((AngleUnit::degree == au) ? 1.0/90.0 : 2.0);
    for (int i=0;i<3;i++) this->ang[i] = avec[i*span]*conversion;
  }
  void set_len_scalars(const double, const double, const double);
  void set_ang_scalars(const double, const double, const double);
  void check_hall_number(const int h);
  void check_IT_name(const std::string itname);
public:
  //! Construct the Lattice from a matrix of the basis vectors
  Lattice(const double *, const int h=1);
  //! Construct the Lattice from a possibly-not-contiguous matrix of the basis vectors
  template<class I>//, typename=typename std::enable_if<std::is_integral<I>::value>::type>
  Lattice(const double * latmat, std::vector<I>& strides, const int h){
    double l[3]={0,0,0}, a[3]={0,0,0};
    latmat_to_lenang(latmat,strides[0]/sizeof(double),strides[1]/sizeof(double),l,a);
    this->set_len_pointer(l,1);
    this->set_ang_pointer(a,1, AngleUnit::radian);
    this->volume=this->calculatevolume();
    this->check_hall_number(h);
  }
  /*! \brief Construct the lattice from two vectors of the lengths and angles

  @param lengths    A pointer to the first of three basis vector lengths in
                    arbitrary length units
  @param lenstrides The first element must contain the stride in bytes between
                    basis vector lenght entries
  @param angles     A pointer to the first of three inter-basis-vector angles in
                    units of pi, radian, or degree
  @param angstrides The first element must contain the stride in bytes between
                    angle entries
  @param h          The hall number specifying the lattice symmetries
  @param au         An enum which identifies which units the provided angles
                    use. If omitted or AngleUnit::not_provided, an attempt is
                    made to guess the units. If all provided angles have values
                    less than π then they are assumed to be in units of radian,
                    otherwise they are assumed to be in units of degrees.
  */
  template<class I>//, typename=typename std::enable_if<std::is_integral<I>::value>::type>
  Lattice(const double * lengths, std::vector<I>& lenstrides, const double * angles, std::vector<I>& angstrides, const int h, const AngleUnit au=AngleUnit::not_provided){
    this->set_len_pointer(lengths,lenstrides[0]/sizeof(double));
    this->set_ang_pointer(angles,angstrides[0]/sizeof(double), au);
    this->volume=this->calculatevolume();
    this->check_hall_number(h);
  }
  //! Construct the Lattice from a vector of the basis vector lengths and a vector of the basis vector angles
  Lattice(const double *, const double *, const int h=1, const AngleUnit au=AngleUnit::not_provided);
  //! Construct the Lattice from the three scalar lengths and three scalar angles
  Lattice(const double la=1.0, const double lb=1.0, const double lc=1.0, const double al=PIOVERTWO, const double bl=PIOVERTWO, const double cl=PIOVERTWO, const int h=1);
  //! Construct the Lattice from a matrix of the basis vectors, specifying an International Tables symmetry name instead of a Hall number
  Lattice(const double *, const std::string);
  //! Construct the lattice from vectors, specifying an International Tables symmetry name instead of a Hall number
  Lattice(const double *, const double *, const std::string, const AngleUnit au=AngleUnit::not_provided);
  template<class I>//, typename=typename std::enable_if<std::is_integral<I>::value>::type>
  Lattice(const double * latmat, std::vector<I>& strides, const std::string itname){
    double l[3]={0,0,0}, a[3]={0,0,0};
    latmat_to_lenang(latmat,strides[0]/sizeof(double),strides[1]/sizeof(double),l,a);
    this->set_len_pointer(l,1);
    this->set_ang_pointer(a,1, AngleUnit::radian);
    this->volume=this->calculatevolume();
    this->check_IT_name(itname);
  }
  //! Construct the lattice from two possibly-not-contiguous vectors of the lengths and angles
  template<class I>//, typename=typename std::enable_if<std::is_integral<I>::value>::type>
  Lattice(const double * lengths, std::vector<I>& lenstrides, const double * angles, std::vector<I>& angstrides, const std::string itname, const AngleUnit au=AngleUnit::not_provided){
    this->set_len_pointer(lengths,lenstrides[0]/sizeof(double));
    this->set_ang_pointer(angles,angstrides[0]/sizeof(double), au);
    this->volume=this->calculatevolume();
    this->check_IT_name(itname);
  }
  //! Construct the lattice from scalars, specifying an International Tables symmetry name instead of a Hall number
  Lattice(const double, const double, const double, const double, const double, const double, const std::string);
  virtual ~Lattice() = default;
  //! copy constructor
  Lattice(const Lattice& other){
    for (int i=0; i<3; ++i){
      this->len[i] = other.len[i];
      this->ang[i] = other.ang[i];
    }
    this->volume = other.volume;
    this->hall = other.hall;
  }
  //! explicit assignment operator
  // required for gcc 9+
  Lattice& operator=(const Lattice& other){
    for (int i=0; i<3; ++i){
      this->len[i] = other.len[i];
      this->ang[i] = other.ang[i];
    }
    this->volume = other.volume;
    this->hall = other.hall;
    return *this;
  }
  //! Return the first basis vector length
  double get_a     () const {return len[0];}
  //! Return the second basis vector length
  double get_b     () const {return len[1];}
  //! Return the third basis vector length
  double get_c     () const {return len[2];}
  //! Return the angle between the second and third basis vectors in radian
  double get_alpha () const {return ang[0];}
  //! Return the angle between the first and third basis vectors in radian
  double get_beta  () const {return ang[1];}
  //! Return the angle between the first and second basis vectors in radian
  double get_gamma () const {return ang[2];}
  //! Return the volume of the parallelpiped unit cell formed by the basis vectors
  double get_volume() const {return volume;}
  //! Calculate and return the unit cell volume
  double calculatevolume();
  /*! Calculate the metric tensor of the Lattice
  @param[out] mt Pointer to memory which can store 9 doubles
  */
  void get_metric_tensor(double * mt) const ;
  /*! Calculate the covariant metric tensor of the Lattice -- this is typically referred to as **the** metric tensor
  @param[out] mt Pointer to memory which can store 9 doubles
  */
  void get_covariant_metric_tensor(double *mt) const ;
  /*! Calculate the contravariant metric tensor of the Lattice -- the inverse of **the** metric tensor
  @param[out] mt Pointer to memory which can store 9 doubles
  */
  void get_contravariant_metric_tensor(double *mt) const ;
  // some functions don't logically make sense for this base class, but
  // do for the derived classes. define them here for funsies
  //! Determine if the passed Lattice represents the same space-spanning lattice
  bool issame(const Lattice) const; // this should really have a tolerance
  /*! Determine if the passed Lattice represents an equivalent space-spanning
  lattice within the specified tolerance. Simultaneous permutations of lengths
  and angles are considered as equivalent --
  e.g., (a,b,c)(α,β,γ) ≡ (b,c,a)(β,γ,α) ≡ (c,a,b)(γ,α,β),
  as are antipermutations,
  e.g., (a,b,c)(α,β,γ) ≡ (a,c,b)(α,γ,β) ≡ (c,b,a)(γ,β,α) ≡ (b,a,c)(β,α,γ).
  */
  bool isapprox(const Lattice) const;
  /*! Determine if the passed Lattice is a permutation of the space-spanning
  lattice within the specified tolerance. The equivalence is encoded in a
  signed integer:

  | returned value | permutation |
  | --- | --- |
  | 1 | (a,b,c)(α,β,γ) |
  | 2 | (b,c,a)(β,γ,α) |
  | 3 | (c,a,b)(γ,α,β) |
  | -1 | (a,c,b)(α,γ,β) |
  | -2 | (c,b,a)(γ,β,α) |
  | -3 | (b,a,c)(β,α,γ) |
  | 0 | no equivalent permutation |
  */
  int ispermutation(const Lattice) const;
  //! Print the basis vector lengths and angles to the console
  virtual void print();
  //! Return a string representation of the basis vector lengths and angles
  virtual std::string string_repr();
  //! Return the Hall number of the Lattice
  int get_hall() const {return hall;}
  //! Set the symmetry of the Lattice by changing the Hall number
  int set_hall(const int h) { check_hall_number(h); return hall; }
  //! Return the Spacegroup object of the Lattice
  Spacegroup get_spacegroup_object() const { return Spacegroup(hall); }
  //! Return the Pointgroup object of the Lattice
  Pointgroup get_pointgroup_object() const { return Spacegroup(hall).get_pointgroup(); }
  //! Return the Spacegroup symmetry operation object of the Lattice
  Symmetry get_spacegroup_symmetry() const { return make_spacegroup_symmetry_object(hall); }
  //! Return the Pointgroup Symmetry operation object of the Lattice
  PointSymmetry get_pointgroup_symmetry(const int time_reversal=0) const { return make_pointgroup_symmetry_object(hall, time_reversal); }
  //! Check whether the pointgroup has the space-inversion operator, ̄1.
  bool has_space_inversion() const {
    PointSymmetry ps = make_pointgroup_symmetry_object(hall, 0); // don't add time-reversal symmetry.
    return ps.has_space_inversion();
  }
};

/*! \brief A space-spanning Lattice that exists in real space

The Direct Lattice describes a space-spanning lattice in the real three
dimensions. The class is a logical wrapper to the Lattice class and
defines new versions of some methods.
*/
class Direct: public Lattice{
public:
  template<class ...Types> Direct(Types ... args): Lattice(args...){}
  Direct(Lattice lat): Lattice(lat){}
  //! Return the inverse Reciprocal lattice
  Reciprocal star() const;
  //! Return the basis vectors expressed in *an* orthonormal frame with a* along x
  void get_xyz_transform(double*) const;
  void get_xyz_transform(double*, const size_t, const size_t) const;
  template<class I> void get_xyz_transform(double*, std::vector<I>&) const;
  //! Return the inverse of the basis vectors expressed in *an* orthonormal frame where a* is along x
  void get_inverse_xyz_transform(double*) const;
  void get_inverse_xyz_transform(double*, const size_t, const size_t) const;
  template<class I> void get_inverse_xyz_transform(double*, std::vector<I>&) const;
  //! Return the basis vectors expressed in *an* orthonormal frame with a along x
  void get_lattice_matrix(double*) const;
  void get_lattice_matrix(double*, const size_t, const size_t) const;
  template<class I> void get_lattice_matrix(double*, std::vector<I>&) const;
  //! Always false
  bool isstar(const Direct) const;
  //! Determine if a Reciprocal lattice is the inverse of this lattice
  bool isstar(const Reciprocal) const;
  void print() override;
  std::string string_repr() override;
  //! For non-Primitive Direct lattices, return the equivalent Primitive lattice
  Direct primitive(void) const;
};
/*! \brief A space-spanning Lattice that exists in reciprocal space

The Reciprocal Lattice describes a space-spanning lattice in the reciprocal
three dimensions of momentum. The class is a logical wrapper to the Lattice
class and defines new versions of some methods.
*/
class Reciprocal: public Lattice{
public:
  template<class ...Types> Reciprocal(Types ... args): Lattice(args...){}
  Reciprocal(Lattice lat): Lattice(lat){}
  //! Return the inverse Direct lattice
  Direct star() const;
  //! Return the Busing-Levey B matrix http://dx.doi.org/10.1107/S0365110X67000970
  void get_B_matrix(double*) const;
  void get_B_matrix(double*, const size_t, const size_t) const;
  template<class I> void get_B_matrix(double*, std::vector<I>&) const;
  //! Return the basis vectors expressed in *an* orthonormal frame with a* along x
  void get_xyz_transform(double*) const;
  void get_xyz_transform(double*, const size_t, const size_t) const;
  template<class I> void get_xyz_transform(double*, std::vector<I>&) const;
  //! Return the inverse of the basis vectors expressed in *an* orthonormal frame where a* is along x
  void get_inverse_xyz_transform(double*) const;
  void get_inverse_xyz_transform(double*, const size_t, const size_t) const;
  template<class I> void get_inverse_xyz_transform(double*, std::vector<I>&) const;
  //! Return the basis vectors expressed in *an* orthonormal frame with a along x
  void get_lattice_matrix(double*) const;
  void get_lattice_matrix(double*, const size_t, const size_t) const;
  template<class I> void get_lattice_matrix(double*, std::vector<I>&) const;
  //! Always false
  bool isstar(const Reciprocal) const;
  //! Determine if a Direct lattice is the inverse of this lattice
  bool isstar(const Direct) const;
  void print() override;
  std::string string_repr() override;
  //! For non-Primitive Reciprocal lattices, return the equivalent Primitive Reciprocal lattice
  Reciprocal primitive(void) const;
};

/*! \brief Type information for Lattice and LatVec objects

Some templated functions require internal variables or return types which
depend on *which* subtype of Lattice or LatVec are provided. This traits struct
provides the typename of an appropriate Lattice subclass and its inverse for
those cases.

The two `using` types are `type` and `star`, defined based on the templated
typename as

| template typename | type | star |
| --- | --- | --- |
| Direct | Direct | Reciprocal |
| Reciprocal | Reciprocal | Direct |
| LDVec | Direct | Reciprocal |
| LQVec | Reciprocal | Direct |

*/
template <typename T> struct LatticeTraits{
  using type = void;
  using star = void;
};
#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<> struct LatticeTraits<Direct>{
  using type = Direct;
  using star = Reciprocal;
};
template<> struct LatticeTraits<Reciprocal>{
  using type = Reciprocal;
  using star = Direct;
};
#endif

#endif
