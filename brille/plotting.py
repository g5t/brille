# Copyright 2019 Greg Tucker
#
# This file is part of brille.
#
# brille is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# brille is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with brille. If not, see <https://www.gnu.org/licenses/>.

"""Provides functionality for drawing first Brillouin zones in 3D."""
import collections
import numpy as np
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.colors import get_named_colors_mapping
import brille

def _check_axes(axs=None):
    if axs is None:
        if pp.get_fignums() and isinstance(pp.gca(), Axes3D):
            axs = pp.gca()
        else:
            axs = Axes3D(pp.figure())
    return axs

def plot(*args, **kwds):
    bz_types = (brille.BrillouinZone,
                brille.BZGridQcomplex, brille.BZGridQ,
                brille.BZMeshQcomplex, brille.BZMeshQ,
                brille.BZNestQcomplex, brille.BZNestQ,
                brille.BZTrellisQcomplex, brille.BZTrellisQ)
    if len(args) is 1:
        if isinstance(args[0], bz_types):
            return plot_bz(*args, **kwds)
        if isinstance(args[0], brille.Polyhedron):
            return plot_polyhedron(*args, **kwds)
        else:
            return plot_points(*args, **kwds)
    if len(args) is 2:
        if (isinstance(args[1], np.ndarray)) and not issubclass(args[1].dtype.type, np.integer):
            return plot_points_with_lines(*args, **kwds)
        else:
            return plot_tetrahedra(*args, **kwds)
    else:
        raise Exception("Unknown number of non-keyword arguments for plot")


# pylint: disable=c0103
def plot_points(x, axs=None, title=None, show=True):
    """Plot the N points contained in the (N,3) ndarray x."""
    axs = _check_axes(axs)
    axs.scatter(x[:, 0], x[:, 1], x[:, 2], s=10)
    if title is not None:
        axs.set_title(title)
    if show:
        pp.show()

def plot_points_with_lines(x, y, axs=None, title=None, show=True):
    """Plot the N points contained in the (N,3) ndarray x with lines y.

    The M line segments defined by the (M+1,3) ndarray y are drawn before the
    points in x.
    """
    axs = _check_axes(axs)
    axs.plot(y[:, 0], y[:, 1], y[:, 2])
    axs.scatter(x[:, 0], x[:, 1], x[:, 2], s=10)
    if title is not None:
        axs.set_title(title)
    if show:
        pp.show()

# pylint: disable=r0912,r0913,r0914,r0915
def plot_bz(bz, axs=None, origin=None, Q=None, units='invA', irreducible=True,
            face_vectors=False, show=True,
            color='b', edgecolor='k', linewidth=1, alpha=0.2):
    """Plot a BrillouinZone or BZGridQ[complex] object.

    Draw the faces of a first Brillouin zone with color, edgecolor, linewidth,
    and alpha specified. The plotting units are selectable via the keyword
    `units` with valid values 'rlu', 'invA', or 'primitive' and are relative
    lattice units of the reciprocal space spanning lattice, inverse ångstrom,
    or relative lattice units of the primitive reciprocal space spanning
    lattice, respectively. The face vectors defining the Brillouin zone can be
    drawn as well if the keyword `face_vectors` is set to True.

    If a (N,3) numpy.ndarray is provided via the keyword `Q` it will be treated
    as points in the specified units of reciprocal space.
    If a BZGridQ or BZGridQcomplex object is provided and `Q` is omitted or
    None, then the mapped grid points in 'rlu' or 'invA' will be set to `Q`.
    """
    # pylint: disable=no-member
    axs = _check_axes(axs)
    types_with_points = (brille.BZGridQcomplex, brille.BZGridQ,
                         brille.BZMeshQcomplex, brille.BZMeshQ,
                         brille.BZNestQcomplex, brille.BZNestQ,
                         brille.BZTrellisQcomplex, brille.BZTrellisQ)
    if isinstance(bz, types_with_points):
        if Q is None:
            if units == 'rlu':
                Q = bz.rlu
            elif units == 'invA':
                Q = bz.invA
        bz = bz.BrillouinZone
    if origin is not None and not isinstance(origin, np.ndarray):
        origin = np.array(origin)
    if origin is None or origin.size != 3 or origin.ndim > 1:
        origin = np.array((0, 0, 0))
    # we always draw the 1st Brillouin zone
    if units == 'rlu':
        verts = bz.vertices
    elif units == 'primitive':
        verts = bz.vertices_primitive
    else:
        verts = bz.vertices_invA
    bzcolor = color if not irreducible else "w"
    bzedgecolor = edgecolor if not irreducible else "0.5"
    bzlinestyle = '-' if not irreducible else '--'
    bzalpha = alpha if not irreducible else 0

    # the 1st Brillouin zone has on-face points equal to half the normals
    polybz, xyz_min, xyz_max = _make_poly_collection(verts,
                                                     bz.vertices_per_face,
                                                     origin=origin,
                                                     color=bzcolor,
                                                     edgecolor=bzedgecolor,
                                                     linestyle=bzlinestyle,
                                                     linewidth=linewidth,
                                                     alpha=bzalpha)
    if irreducible:
        if units == 'rlu':
            ir_verts = bz.ir_vertices
        elif units == 'primitive':
            ir_verts = bz.ir_vertices_primitive
        else:
            ir_verts = bz.ir_vertices_invA
        if ir_verts.size > 0:
            polyir, _, _ = _make_poly_collection(ir_verts,
                                                 bz.ir_vertices_per_face,
                                                 origin=origin,
                                                 color=color,
                                                 edgecolor=edgecolor,
                                                 linestyle='-',
                                                 linewidth=linewidth,
                                                 alpha=alpha)
            axs.add_collection3d(polyir)
    axs.add_collection3d(polybz)
    if face_vectors:
        if units == 'rlu':
            norms = bz.normals
            point = bz.points
        elif units == 'primitive':
            norms = bz.normals_primitive
            point = bz.points_primitive
        else:
            norms = bz.normals_invA
            point = bz.points_invA
        fvecs = [np.array([p, p+n]) for p, n in zip(point, norms)]
        lcol = Line3DCollection(fvecs)
        axs.add_collection3d(lcol)
    axs.set_xlim(left=xyz_min[0], right=xyz_max[0])
    axs.set_ylim(bottom=xyz_min[1], top=xyz_max[1])
    axs.set_zlim(bottom=xyz_min[2], top=xyz_max[2])
    if isinstance(Q, np.ndarray) and Q.ndim == 2 and Q.shape[1] == 3:
        axs.scatter(Q[:, 0], Q[:, 1], Q[:, 2])
    # axs.set_aspect('equal', 'box') # removed from newer Matplotlib
    # axs.auto_scale_xyz(1.,1.,1.) # supposed-workaround, probably need to set scaling based on figure size and view
    if show:
        pp.show()
    return axs

def _make_poly_collection(verts, vpf, origin=None, color='b', edgecolor='k',
                          linestyle='-', linewidth=1, alpha=0.5):
    # vpf lists the ordered vertices which make up each facet
    # for each facet, pick-out the vertices which define its polygon face
    patches = [np.array([verts[j, :] for j in i]) for i in vpf]
    # if an origin has been provided, add it to the patches
    if origin is not None and origin.ndim == 1 and origin.shape[0] == 3:
        for p in patches:
            p += origin
    # find the extent of the patches
    xyz_min = np.array([x.min() for x in np.vsplit(verts.transpose(), 3)])
    xyz_max = np.array([x.max() for x in np.vsplit(verts.transpose(), 3)])
    # plus some nice-for-plotting padding
    dif = xyz_max-xyz_min
    xyz_min -= dif/20
    xyz_max += dif/20
    # and create the collection of polygons in 3D
    collection = Poly3DCollection(patches, edgecolor=edgecolor,
                                  linestyle=linestyle, linewidth=linewidth,
                                  alpha=alpha)
    # which requires that the face color be set after the fact
    collection.set_facecolor(color)
    return (collection, xyz_min, xyz_max)

def __cube(p_0, p_1):
    """Return the patches of a cube bounded by points p_0 and p_1."""
    d_x = np.array((p_1[0]-p_0[0], 0, 0))
    d_y = np.array((0, p_1[1]-p_0[1], 0))
    d_z = np.array((0, 0, p_1[2]-p_0[2]))
    verts = p_0+np.array([d_x-d_x,      # 0 (000)
                          d_x,          # 1 (100)
                          d_x+d_y,      # 2 (110)
                          d_y,          # 3 (010)
                          d_z,          # 4 (001)
                          d_z+d_x,      # 5 (101)
                          d_z+d_x+d_y,  # 6 (111)
                          d_z+d_y])     # 7 (011)
    idx = np.array([[0, 1, 2, 3],   # (000)-(100)-(110)-(010)
                    [0, 1, 5, 4],   # (000)-(100)-(101)-(001)
                    [0, 4, 7, 3],   # (000)-(001)-(011)-(010)
                    [4, 5, 6, 7],   # (001)-(101)-(111)-(011)
                    [6, 2, 1, 5],   # (111)-(110)-(100)-(101)
                    [2, 6, 7, 3]])  # (110)-(111)-(011)-(010)
    patches = [verts[x] for x in idx]
    return patches

def plot_polyhedron(poly, axs=None, setlims=True, show=True, **kwds):
    """Plot a polyhedron"""
    # pylint: disable=no-member
    axs = _check_axes(axs)
    # the 1st Brillouin zone has on-face points equal to half the normals
    coll, xyz_min, xyz_max = _make_poly_collection(poly.vertices,
                                                   poly.vertices_per_face,
                                                   **kwds)
    axs.add_collection3d(coll)
    if setlims:
        axs.set_xlim(left=xyz_min[0], right=xyz_max[0])
        axs.set_ylim(bottom=xyz_min[1], top=xyz_max[1])
        axs.set_zlim(bottom=xyz_min[2], top=xyz_max[2])
    if show:
        pp.show()
    return axs

def plot_tetrahedron(verts, axs=None, show=True, **kwds):
    if not (verts.ndim == 2 and verts.shape[0]==4 and verts.shape[1]==3):
        raise RuntimeError('Input are not the vertices of a tetrahedron')
    vpf = np.array([[0,1,2],[0,3,1],[3,2,1],[0,2,3]])
    pc, _, _ = _make_poly_collection(verts, vpf, **kwds)
    # Add the Poly3DCollection to existing or new axes:
    axs = _check_axes(axs)
    axs.add_collection3d(pc)
    if show:
        pp.show()
    return axs

def plot_tetrahedra(allverts, tetidx, axs=None, **kwds):
    if not (allverts.ndim == 2 and allverts.shape[1] == 3):
        raise RuntimeError('Vertices are not the correct shape')
    if isinstance(tetidx, list):
        tetidx = np.array(tetidx)
    if not (tetidx.ndim == 2 and tetidx.shape[1] == 4):
        raise RuntimeError('Tetrahedra indexes are not the correct shape')

    colours = kwds.pop('color', get_named_colors_mapping().keys())
    if isinstance(colours, collections.Iterable):
        colours = list(colours)
    if isinstance(colours, str) or (isinstance(colours, (list, tuple)) and len(colours)==3):
        colours = [colours]
    if not isinstance(colours, np.ndarray):
        colours = np.array(colours);
    #if not 'str' in colours.dtype.name:
    if colours.shape[0] < tetidx.shape[0]:
        colours = np.tile(colours, 1+tetidx.shape[0]//colours.shape[0])
    colours = colours[0:tetidx.shape[0]]

    # we want to ensure all tetrahedra end up in the same set of axes
    axs = _check_axes(axs)
    for tet, colour in zip(tetidx, colours):
        plot_tetrahedron(allverts[tet], color=colour, **kwds)