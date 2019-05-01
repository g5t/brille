#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <vector>

#include "symbz.h"
#include "arithmetic.h"
#include "linear_algebra.h"
#include "lattice.h"
#include "bz.h"
#include "grid.h"
#include "bz_grid.h"

#ifndef __BINDING_H
#define __BINDING_H

namespace py = pybind11;
using namespace pybind11::literals; // bring in "[name]"_a to be interpreted as py::arg("[name]")

template<typename T> py::array_t<T> av2np(const ArrayVector<T> av){
	std::vector<ssize_t> shape(2); // ArrayVectors are 2D by default
	shape[0] = av.size();
	shape[1] = av.numel();
	auto np = py::array_t<T,py::array::c_style>(shape);
	T *rptr = (T*) np.request().ptr;
	for (size_t i =0; i< av.size(); i++)
		for (size_t j=0; j< av.numel(); j++)
			rptr[i*av.numel()+j] = av.getvalue(i,j);
	return np;
}


std::string long_version(){
	using namespace symbz::version;
	std::string v = version_number;
	if (!std::string(git_revision).empty()){
		v += "-" + std::string(git_revision).substr(0,7)
		   + "@" + git_branch;
	}
	return v;
}

typedef long slong; // ssize_t is only defined for gcc?


template<class T>
void declare_bzgridq(py::module &m, const std::string &typestr) {
    using Class = BrillouinZoneGrid3<T>;
    std::string pyclass_name = std::string("BZGridQ") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    // Initializer (BrillouinZone, [half-]Number_of_steps vector)
    .def(py::init([](BrillouinZone &b, py::array_t<size_t,py::array::c_style> pyN){
      py::buffer_info bi = pyN.request();
      if (bi.ndim != 1) throw std::runtime_error("N must be a 1-D array");
      if (bi.shape[0] < 3) throw std::runtime_error("N must have three elements");
      Class cobj( b, (size_t*)bi.ptr );
      return cobj;
    }),py::arg("brillouinzone"),py::arg("halfN"))
    // Initializer (BrillouinZone, step_size vector, flag_for_whether_step_size_is_in_rlu_or_inverse_angstrom)
    .def(py::init([](BrillouinZone &b, py::array_t<double,py::array::c_style> pyD, const bool& isrlu){
      py::buffer_info bi = pyD.request();
      if (bi.ndim != 1) throw std::runtime_error("stepsize must be a 1-D array");
      if (bi.shape[0] < 3) throw std::runtime_error("stepsize must have three elements");
      return Class( b, (double*)bi.ptr, isrlu ? 1 : 0 );
    }),py::arg("brillouinzone"),py::arg("step"),py::arg("rlu")=true)
    .def_property_readonly("brillouinzone",[](const Class& cobj){ return cobj.get_brillouinzone();} )
    .def_property_readonly("rlu",[](const Class& cobj){ return av2np(cobj.get_grid_hkl());} )
    .def_property_readonly("invA",[](const Class& cobj){ return av2np(cobj.get_grid_xyz());} )
    .def_property_readonly("mapped_rlu",[](const Class& cobj){ return av2np(cobj.get_mapped_hkl());} )
    .def_property_readonly("mapped_invA",[](const Class& cobj){ return av2np(cobj.get_mapped_xyz());} )
    .def("fill",[](Class& cobj, py::array_t<T,py::array::c_style> pydata){
      py::buffer_info bi = pydata.request();
      ssize_t ndim = bi.ndim;
      /* ndim  assumed interpolation data type  numel /
      /   1              scalar                   1   /
      /   2              vector                shape[1]  /
      /   3       matrix / rank 2 tensor       shape[1]*shape[2]       /
      /   N         rank N-1 tensor            prod(shape,1,N-1)      */
      size_t numel=1, numarrays=bi.shape[0];
      if (ndim > 1) for (ssize_t i=1; i<ndim; ++i) numel *= bi.shape[i];
      ArrayVector<T> data(numel, numarrays, (T*)bi.ptr);
      ArrayVector<size_t> shape(1,ndim);
      for (ssize_t i=0; i<ndim; ++i) shape.insert(bi.shape[i], (size_t)i );
      int mapExceedsNewData = cobj.check_map(data);
      if (mapExceedsNewData) throw std::runtime_error("There are less provided data arrays than unique integers in the mapping.");
      cobj.replace_data(data,shape); // no error, so this will work for sure
    })
    .def_property("map",
      /*get map*/ [](Class& cobj){
        std::vector<ssize_t> shape(3); // the map is 3D
        shape[0] = cobj.size(0);
        shape[1] = cobj.size(1);
        shape[2] = cobj.size(2);
        auto np = py::array_t<slong,py::array::c_style>(shape);
        size_t nret = cobj.unsafe_get_map( (slong*)np.request().ptr );
        if (nret != shape[0]*shape[1]*shape[2])
          // I guess nret is smaller, otherwise we probably already encountered a segfault
          throw std::runtime_error("Something has gone horribly wrong with getting the map.");
        return np;
      },
      /*set map*/ [](Class& cobj, py::array_t<slong,py::array::c_style> pymap){
        py::buffer_info bi = pymap.request();
        if (bi.ndim != 3) throw std::runtime_error("The mapping must be a 3 dimensional array");
        for (size_t i=0; i<3; ++i) if (bi.shape[i]!=cobj.size(i))
          throw std::runtime_error("The new map shape must match the old map"); // or we could resize it, but that is more difficult
        if (cobj.maximum_mapping( (slong*)bi.ptr ) > cobj.num_data() )
          throw std::runtime_error("The largest integer in the new mapping exceeds the number of data elements.");
        cobj.unsafe_set_map( (slong*)bi.ptr ); //no error, so this works.
    })
    .def("interpolate_at",[](Class& cobj, py::array_t<double,py::array::c_style> pyX, const bool& moveinto){
      py::buffer_info bi = pyX.request();
      if ( bi.shape[bi.ndim-1] !=3 )
        throw std::runtime_error("Interpolation requires one or more 3-vectors");
      ssize_t npts = 1;
      if (bi.ndim > 1) for (ssize_t i=0; i<bi.ndim-1; i++) npts *= bi.shape[i];
      BrillouinZone b = cobj.get_brillouinzone();
      Reciprocal lat = b.get_lattice();
      LQVec<double> qv(lat,npts, (double*)bi.ptr ); //memcopy
      if (moveinto){
        LQVec<double> Qv(qv); // second memcopy
        LQVec<int>  tauv(lat,npts); // filled by moveinto
        bool success = b.moveinto(&Qv,&qv,&tauv);
        if (!success)
          throw std::runtime_error("failed to move all Q into the first Brillouin Zone");
      }
      // do the interpolation for each point in qv
      ArrayVector<T> lires = cobj.linear_interpolate_at(qv);
      // and then make sure we return an numpy array of appropriate size:
      std::vector<ssize_t> outshape;
      for (ssize_t i=0; i < bi.ndim-1; ++i) outshape.push_back(bi.shape[i]);
      if (cobj.data_ndim() > 1){
        ArrayVector<size_t> data_shape = cobj.data_shape();
        // the shape of each element is data_shape[1,...,data_ndim-1]
        for (ssize_t i=1; i<data_shape.size(); ++i) outshape.push_back( data_shape.getvalue(i) );
      }
      size_t total_elements = 1;
      for (ssize_t osi : outshape) total_elements *= osi;
      if (total_elements != lires.numel()*lires.size() ){
        std::cout << "outshape: (";
        for (ssize_t osi : outshape) std::cout << osi << "," ;
        std::cout << "\b)" << std::endl;
        printf("Why do we expect %u total elements but only get back %u? (%u × %u)\n",total_elements,lires.numel()*lires.size(),lires.numel(),lires.size());
        throw std::runtime_error("error determining output size");
      }
      auto liout = py::array_t<T,py::array::c_style>(outshape);
      T *rptr = (T*) liout.request().ptr;
      for (size_t i =0; i< lires.size(); i++)
        for (size_t j=0; j< lires.numel(); j++)
          rptr[i*lires.numel()+j] = lires.getvalue(i,j);
      return liout;
    },py::arg("Q"),py::arg("moveinto")=true);
}

template<class T>
void declare_bzgridqe(py::module &m, const std::string &typestr) {
    using Class = BrillouinZoneGrid4<T>;
    std::string pyclass_name = std::string("BZGridQE") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    // Initializer (BrillouinZone, [half-]Number_of_steps vector)
    .def(py::init([](BrillouinZone &b, py::array_t<double,py::array::c_style> pySpec, py::array_t<size_t,py::array::c_style> pyN){
      py::buffer_info b0 = pySpec.request();
      if (b0.ndim!=1) throw std::runtime_error("spec must be a 1-D array");
      if (b0.shape[0]<3) throw std::runtime_error("spec must have three elements");
      py::buffer_info bi = pyN.request();
      if (bi.ndim != 1) throw std::runtime_error("N must be a 1-D array");
      if (bi.shape[0] < 3) throw std::runtime_error("N must have three elements");
      Class a( b, (double*)b0.ptr, (size_t*)bi.ptr );
      return a;
    }),py::arg("brillouinzone"),py::arg("spec"),py::arg("halfN"))
    // Initializer (BrillouinZone, step_size vector, flag_for_whether_step_size_is_in_rlu_or_inverse_angstrom)
    .def(py::init([](BrillouinZone &b, py::array_t<double,py::array::c_style> pySpec, py::array_t<double,py::array::c_style> pyD, const bool& isrlu){
      py::buffer_info b0 = pySpec.request();
      if (b0.ndim!=1) throw std::runtime_error("spec must be a 1-D array");
      if (b0.shape[0]<3) throw std::runtime_error("spec must have three elements");
      py::buffer_info bi = pyD.request();
      if (bi.ndim != 1) throw std::runtime_error("stepsize must be a 1-D array");
      if (bi.shape[0] < 3) throw std::runtime_error("stepsize must have three elements");
      return Class( b, (double*)b0.ptr, (double*)bi.ptr, isrlu ? 1 : 0 );
    }),py::arg("brillouinzone"),py::arg("spec"),py::arg("step"),py::arg("rlu")=true)
    .def_property_readonly("brillouinzone",[](const Class& a){ return a.get_brillouinzone();} )
    .def_property_readonly("rlu_Q",[](const Class& a){ return av2np(a.get_grid_hkl());} )
    .def_property_readonly("invA_Q",[](const Class& a){ return av2np(a.get_grid_xyz());} )
    .def_property_readonly("mapped_rlu_Q",[](const Class& a){ return av2np(a.get_mapped_hkl());} )
    .def_property_readonly("mapped_invA_Q",[](const Class& a){ return av2np(a.get_mapped_xyz());} )
    .def_property_readonly("rlu",[](const Class& a){ return av2np(a.get_grid_hkle());} )
    .def_property_readonly("invA",[](const Class& a){ return av2np(a.get_grid_xyzw());} )
    .def_property_readonly("mapped_rlu",[](const Class& a){ return av2np(a.get_mapped_hkle());} )
    .def_property_readonly("mapped_invA",[](const Class& a){ return av2np(a.get_mapped_xyzw());} )
    .def("fill",[](Class& a, py::array_t<T,py::array::c_style> pydata){
      py::buffer_info bi = pydata.request();
      ssize_t ndim = bi.ndim;
      size_t numel=1, numarrays=bi.shape[0];
      if (ndim > 1) for (ssize_t i=1; i<ndim; ++i) numel *= bi.shape[i];
      ArrayVector<T> data(numel, numarrays, (T*)bi.ptr);
      ArrayVector<size_t> shape(1,ndim);
      for (ssize_t i=0; i<ndim; ++i) shape.insert(bi.shape[i], (size_t)i );
      int mapExceedsNewData = a.check_map(data);
      if (mapExceedsNewData) throw std::runtime_error("There are less provided data arrays than unique integers in the mapping.");
      a.replace_data(data,shape); // no error, so this will work for sure
      // return mapExceedsNewData; // let the calling function deal with this?
    })
    .def_property("map",
      /*get map*/ [](Class& a){
        std::vector<ssize_t> shape(4); // the map is 4D
        for (int i=0; i<4; i++) shape[i] = a.size(i);
        auto np = py::array_t<slong,py::array::c_style>(shape);
        size_t nret = a.unsafe_get_map( (slong*)np.request().ptr );
        if (nret != shape[0]*shape[1]*shape[2]*shape[3])
          // I guess nret is smaller, otherwise we probably already encountered a segfault
          throw std::runtime_error("Something has gone horribly wrong with getting the map.");
        return np;
      },
      /*set map*/ [](Class& a, py::array_t<slong,py::array::c_style> pymap){
        py::buffer_info bi = pymap.request();
        if (bi.ndim != 4) throw std::runtime_error("The mapping must be a 4 dimensional array");
        for (size_t i=0; i<4; ++i) if (bi.shape[i]!=a.size(i))
          throw std::runtime_error("The new map shape must match the old map"); // or we could resize it, but that is more difficult
        if (a.maximum_mapping( (slong*)bi.ptr ) > a.num_data() )
          throw std::runtime_error("The largest integer in the new mapping exceeds the number of data elements.");
        a.unsafe_set_map( (slong*)bi.ptr ); //no error, so this works.
    })
    .def("interpolate_at",[](Class& a, py::array_t<double,py::array::c_style> pyX, const bool& moveinto){
      py::buffer_info bi = pyX.request();
      if ( bi.shape[bi.ndim-1] !=4 )
        throw std::runtime_error("Interpolation requires one or more 4-vectors");
      ssize_t npts = 1;
      if (bi.ndim > 1) for (ssize_t i=0; i<bi.ndim-1; i++) npts *= bi.shape[i];
      BrillouinZone b = a.get_brillouinzone();
      Reciprocal lat = b.get_lattice();
      ArrayVector<double> qEv(4u,npts, (double*)bi.ptr ); //memcopy
      if (moveinto){
        LQVec<double> Qv(lat, qEv, 0); // 0 ==> truncate qEv such that it has numel()==3.
        LQVec<double> qv(lat,npts); // filled by moveinto
        LQVec<int>  tauv(lat,npts); // filled by moveinto
        bool success = b.moveinto(&Qv,&qv,&tauv);
        if (!success)
          throw std::runtime_error("failed to move all Q into the first Brillouin Zone");
        // replace the first three elements of qEv with qv.
        for (size_t i=0; i<npts; ++i) for(size_t j=0; j<3u; ++j) qEv.insert(qv.getvalue(i,j), i,j);
      }
      // do the interpolation for each point in qv
      ArrayVector<T> lires = a.linear_interpolate_at(qEv);
      // and then make sure we return an numpy array of appropriate size:
      std::vector<ssize_t> outshape;
      for (ssize_t i=0; i < bi.ndim-1; ++i) outshape.push_back(bi.shape[i]);
      if (a.data_ndim() > 1){
        ArrayVector<size_t> data_shape = a.data_shape();
        // the shape of each element is data_shape[1,...,data_ndim-1]
        for (ssize_t i=1; i<data_shape.size(); ++i) outshape.push_back( data_shape.getvalue(i) );
      }
      size_t total_elements = 1;
      for (ssize_t osi : outshape) total_elements *= osi;
      if (total_elements != lires.numel()*lires.size() ){
        std::cout << "outshape: (";
        for (ssize_t osi : outshape) std::cout << osi << "," ;
        std::cout << "\b)" << std::endl;
        printf("Why do we expect %u total elements but only get back %u? (%u × %u)\n",total_elements,lires.numel()*lires.size(),lires.numel(),lires.size());
        throw std::runtime_error("error determining output size");
      }
      auto liout = py::array_t<T,py::array::c_style>(outshape);
      T *rptr = (T*) liout.request().ptr;
      for (size_t i =0; i< lires.size(); i++)
        for (size_t j=0; j< lires.numel(); j++)
          rptr[i*lires.numel()+j] = lires.getvalue(i,j);
      return liout;
    },py::arg("QE"),py::arg("moveinto")=true);
  }


#endif
