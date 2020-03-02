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
#ifndef _BIN_UTILITIES_HPP_
#define _BIN_UTILITIES_HPP_

#include <vector>
#include <algorithm>
#include "utilities.hpp"

/*
  Storing the lowest bin boundary (zero[3]), constant difference (step[3]),
  and number of bins (size[3]) and directly calculating the bin for a given x
  is a possible solution, but one which is no faster than storing bin bounds.
  Since storing the boundaries can enable non-uniform bins this seems like
  the better long-term solution.
*/
// template<class T, class I>
// I find_bin(const T zero, const T step, const I size, const T x){
//   // we want to find i such that 0 <= i*step < x-zero < (i+1)*step
//   I i = x > zero ?  static_cast<I>(std::floor((x-zero)/static_cast<T>(step))) : 0;
//   return i < size ? i : size-1;
// }
// template<class T, class I>
// int on_boundary(const T zero, const T step, const I size, const T x, const I i){
//   // if x is infinitesimally smaller than zero+step*(i+1)
//   if (i+1<size && approx_scalar(zero+step*(i+1),x)) return  1;
//   // if x is infinitesimally larger than zero+step*i
//   if (i  >0    && approx_scalar(zero+step*(i  ),x)) return -1;
//   return 0;
// }
template<class T>
size_t find_bin(const std::vector<T>& bin_edges, const T x){
  auto found_at = std::find_if(bin_edges.begin(), bin_edges.end(), [x](double b){return b>x;});
  size_t d = std::distance(bin_edges.begin(), found_at);
  return d>0 ? d-1 : d;
}
template<class T>
int on_boundary(const std::vector<T>& bin_edges, const T x, const size_t i){
  // if (i==0) then above d was *either* 0 or 1, otherwise d = i + 1;
  // if (i==0) we can't go lower in either case, so no problem.
  if (i+2<bin_edges.size() && approx_scalar(bin_edges[i+1],x)) return  1;
  if (i  >0                && approx_scalar(bin_edges[i  ],x)) return -1;
  return 0;
}

#endif
