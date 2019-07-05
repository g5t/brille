/*! \file */
/* Copyright (C) 2008 Atsushi Togo
 All rights reserved.

 This file is part of spglib. https://github.com/atztogo/spglib

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.

 * Neither the name of the phonopy project nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE. */

#ifndef __pointgroup_H__
#define __pointgroup_H__

#include <vector>
#include <array>
#include "symmetry.h"

enum class Holohedry {_, triclinic, monoclinic, orthogonal, tetragonal, trigonal, hexagonal, cubic};
enum class Laue {_, _1, _2m, _mmm, _4m, _4mmm, _3, _3m, _6m, _6mmm, _m3, _m3m};

typedef struct {
  int number;
  char symbol[6];
  char schoenflies[4];
  Holohedry holohedry;
  Laue laue;
} Pointgroup;

Pointgroup ptg_get_transformation_matrix(int *transform_mat, const int *rotations, const int num_rotations);
Pointgroup ptg_get_pointgroup(const int pointgroup_number);
PointSymmetry ptg_get_pointsymmetry(const int *rotations, const int num_rotations);

int get_pointgroup_rotations_hall_number(int *rotations, const int max_size, const int hall_number, const int is_time_reversal);

std::vector<std::array<int,3>> rotation_axis_and_perpendicular_vectors(const int* rot);

#endif
