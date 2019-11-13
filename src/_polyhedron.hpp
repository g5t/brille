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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <thread>

#include "_c_to_python.hpp"
#include "polyhedron.hpp"
#include "utilities.hpp"

#ifndef __POLYHEDRON_H
#define __POLYHEDRON_H

namespace py = pybind11;
typedef long slong; // ssize_t is only defined for gcc?

template<class T>
void declare_polyhedron(py::module &m, const std::string &typestr) {
    std::string pyclass_name = std::string("Polyhedron") + typestr;
    py::class_<T>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def_property_readonly("vertices",[](const T& o){return av2np(o.get_vertices());})
    .def_property_readonly("points",[](const T& o){return av2np(o.get_points());})
    .def_property_readonly("normals",[](const T& o){return av2np(o.get_normals());})
    .def_property_readonly("vertices_per_face",&T::get_vertices_per_face)
    .def_property_readonly("faces_per_vertex",&T::get_faces_per_vertex)
    .def_property_readonly("volume",&T::get_volume)
    .def_property_readonly("mirror",&T::mirror)
    .def_property_readonly("centre",&T::centre)
    .def("intersection",&T::intersection)
    .def("rotate",[](const T& o, py::array_t<double> rot){
      py::buffer_info info = rot.request();
      if (info.ndim != 2)
        throw std::runtime_error("Number of dimensions of rotation matrix must be two");
      if (info.shape[0]!=3 || info.shape[1]!=3)
        throw std::runtime_error("Rotation matrix must be 3x3");
      std::array<double, 9> crot;
      double* ptr = (double*) info.ptr;
      auto s0 = info.strides[0]/sizeof(double);
      auto s1 = info.strides[1]/sizeof(double);
      for (size_t i=0; i<3u; ++i) for (size_t j=0; j<3u; ++j)
      crot[i*3u+j] = ptr[i*s0 + j*s1];
      return o.rotate(crot);
    });
  }
#endif