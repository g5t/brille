# brille
A C++ library for symmetry operations and linear interpolation within an
irreducible part of the first Brillouin zone.
Wrapped for use in python using [pybind11](https://github.com/pybind/pybind11).

![brille for Horace](brille.png)

# Dependencies
## TetGen
A modified version of [TetGen](http://tetgen.org) is used to create
refined tetrahedral meshes in the irreducible portion of the first Brillouin
zone.

The modified version is included as part of this repository.

# Installation
From the root folder of this repository use Python 3 to build and install this
library.

`python setup.py install`
