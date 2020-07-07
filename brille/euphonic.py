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
"""
Define a class BrEu to act as the interface between brille and Euphonic.

Thus enabling efficient interpolation of CASTEP-derived phonons at arbitrary
Q points.
"""
import numpy as np
import spglib

from euphonic import ForceConstants
from euphonic import QpointPhononModes as EuQpointPhononModes

import brille
from brille.spglib import BrSpgl, BrQωε
from brille.evn import degenerate_check
from brille.utilities import broaden_modes, half_cpu_count

class BrEu:
    """
    Efficient interpolation of phonon intensity at arbitrary Q points.

    The Euphonic data classes can be used to interpolate CASTEP dynamical
    force matrices at arbitary Q points. It can then use that information to
    determine eigen values (excitation energy) and eigen vectors
    (ion displacements) for the 3×[number of ions] phonon branches. Finally
    Q and the eigen vectors can be used to determine the structure factors of
    the phonon branches, which are proportional to the doubly-differential
    cross section measured by neutron scattering.

    brille provides classes to hold arbitrarily-shaped data at the points of a
    grid filling the first irreducible Brillouin zone of the primitive lattice
    reciprocal unit cell. The brille classes can then use linear interpolation
    to estimate their held-data at points between grid-points, the translational
    and rotational symmetry of the primitive lattice to find an equivalent
    first-irreducible-Brillouin-zone point, q, for any arbitrary reciprocal
    space point, Q.

    The BrEu object uses lattice information from the Euphonic object and the
    package spglib to determine the conventional unit cell equivalent to that
    used in, e.g., CASTEP calculations. This unit cell information is then used
    to construct the primitive first irreducible Brillouin zone and a brille
    grid. The gridded-points are then used by the Euphonic object to calculate
    ωᵢ(q) and ϵᵢⱼ(q), which are placed in the brille object at their respective
    grid points. When an external request is made to the BrEu object to
    calculate Sᵢ(Q) it first uses the filled brille object to interpolate ωᵢ(Q)
    and ϵᵢⱼ(Q) and then the Euphonic object to convert Q and ϵᵢⱼ(Q) into Sᵢ(Q).
    The more-likely use for BrEu, however, will be in calculating S(Q,ω) in
    which case ωᵢ(Q) and Sᵢ(Q) are calculated as above and then a simple
    distribution (selected by the caller) is used to broaden each of the i
    phonon branches before combining their intensities.
    """

    # pylint: disable=r0913,r0914
    def __init__(self, FCData,
                 scattering_lengths=None, cell_is_primitive=None,
                 sort=True, vf=0, hall=None, parallel=False, **kwds):
        """Initialize a new BrEu object from an existing Euphonic object."""
        if not isinstance(FCData, ForceConstants):
            msg = "Unexpected data type {}, expect failures."
            print(msg.format(type(FCData)))
        self.data = FCData
        if not isinstance(scattering_lengths, dict):
            scattering_lengths = {k: 1 for k in np.unique(self.data.crystal.atom_type)}
        self.scattering_lengths = scattering_lengths
        self.brspgl = BrSpgl(self.data.crystal, hall=hall)
        # Construct the BZGrid, by default using the conventional unit cell
        grid_q = self.__define_grid(**kwds)
        # Calculate ωᵢ(Q) and ⃗ϵᵢⱼ(Q), and fill the BZGrid:
        # Select only those keyword arguments which Euphonic expects:
        cfp_keywords = ('asr', 'dipole', 'eta_scale', 'reduce_qpts', 'fall_back_on_python')
        cfp_dict = {k: kwds[k] for k in cfp_keywords if k in kwds}
        # splitting and insert_gamma can modify the number of returned Q points
        # ensure we provide sensible (for brille) defaults
        cfp_dict['splitting'] = kwds.get('splitting', False)
        cfp_dict['insert_gamma'] = kwds.get('insert_gamma', False)
        # the parallel keyword should apply to the Euphonic code too, unless
        # if use_c is already present
        cfp_dict['use_c'] = kwds.get('use_c', parallel)
        if cfp_dict['use_c']:
            cfp_dict['n_threads'] = kwds.get('n_threads', half_cpu_count())
        if cfp_dict['splitting'] or cfp_dict['insert_gamma']:
            print('Options which modify the number of Q points can not be used with brille. Expect problems.')
        # calculate_qpoint_phonon_modes returns the frequencies and eigenvectors
        # equivalent to the properties .freqs and .eigenvecs
        # but we need to make sure we grab _freqs (or _reduced_freqs)
        # since Euphonic no longer attempts to handle varying units
        # internally
        qωε = self.data.calculate_qpoint_phonon_modes(grid_q, **cfp_dict)
        # freq = self.data.freqs.to('millielectron_volt').magnitude
        freq = qωε.frequencies.to('meV').magnitude  # (n_pt, n_br)
        vecs = qωε.eigenvectors # (n_pt, n_br, n_io, 3)
        vecs = degenerate_check(grid_q, freq, vecs)
        self._fill_grid(freq, vecs, vf=vf, sort=sort)
        self.parallel = parallel

    def _fill_grid(self, freq, vecs, vf=0, sort=False):
        n_pt = self.grid.invA.shape[0]
        n_io = self.data.crystal.n_atoms
        n_br = 3*n_io
        if freq.shape == (n_pt, n_br):
            freq = freq.reshape(n_pt, n_br, 1)
        if freq.shape != (n_pt, n_br, 1):
            raise Exception('freqiencies have wrong shape')
        if vecs.shape != (n_pt, n_br, n_io, 3):
            raise Exception('eigenvectors have wrong shape')
        # We must provide extra content information to enable efficient
        # interpolation and possible rotations. All arrays provided to fill
        # must be 3+ dimensional -- the first dimension is over the points of
        # the grid, the second dimension is over the phonon branches, and any
        # additional dimensions are collapsed from highest to lowest dimension
        # (indexed as row-ordered linear indexing)
        # So, the vectors in (n_pt, n_br, n_io, 3) end up as
        # (n_pt, n_br, 3*n_io) with each three-vector's elements contiguous in
        # memory. The extra information provided details *what* the last
        # collapsed dimension contains, as a tuple, list, or 1-D array
        #   ( the number of scalar elements,
        #     the total number of vectors elements [must be divisible by 3],
        #     the total number of matrix elements [must be divisible by 9],
        #     how the vectors/matrices behave under rotation [0, 1, or 2] )
        # The last dimension elements must be ordered as scalars, then vectors
        # then matrices. The three rotation behaviour values are 0 ≡ Realspace
        # vectors, 1 ≡ Reciprocal space vectors, 2 ≡ (Realspace) Axial vectors.
        # Any missing entries from the tuple/list/array are treated as zeros.
        #
        # For the case of phonon eigenvalues and eigenvectors: there is one
        # eigenvalue per branch (scalars do not change under rotation), and
        # each branch eigenvector is comprised of n_ions displacement 3-vectors
        # which transform via the phonon Γ function,
        # so [1,0,0,0] ≡ (1,) and [0,n_ions*3,0,3] ≡ (0,3*n_io,0,3)
        freq_el = (1,)
        freq_wght = (13605.693, 0., 0.) # Rydberg/meV
        vecs_el = (0, 3*n_io, 0, 3, 0, vf) # n_scalar, n_vector, n_matrix, rotates_like (phonon eigenvectors), scalar cost function, vector cost function
        vecs_wght = (0., 1., 0.)
        self.grid.fill(freq, freq_el, freq_wght, self.brspgl.orthogonal_to_conventional_eigenvectors(vecs), vecs_el, vecs_wght, sort)


    def sort_branches(self):
        """Sort the phonon branches stored at all mapped grip points.

        By comparing the difference in phonon branch energy and the angle
        between the branch eigenvectors it is possible to determine a cost
        matrix for assigning the branches on one grid point to those on a
        neighbouring grid point. The Munkres' Assignment algorithm is then used
        to determine a local branch permutation, which is ultimately used in
        determining a global branch permutation for each grid point.

        The cost for each branch-branch assignment is the weighted sum of the
        difference in eigen energies and the angle between eigenvectors:

            Cᵢⱼ = [energy_weight]*√(ωᵢ-ωⱼ)²
                + [angle_weight]*acos(<ϵᵢ,ϵⱼ>/|ϵᵢ||ϵⱼ|)

        """
        if sort and callable(getattr(self.grid, 'sort', None)):
            self.grid.sort()

    # pylint: disable=c0103,w0613,no-member
    def __define_grid(self, mesh=False, nest=False, **kwds):
        brillouin_zone = self.brspgl.get_conventional_BrillouinZone()
        if mesh:
            self.__make_mesh(brillouin_zone, **kwds)
        elif nest:
            self.__make_nest(brillouin_zone, **kwds)
        else:
            self.__make_trellis(brillouin_zone, **kwds)

        # We need to make sure that we pass gridded Q points in the primitive
        # lattice, since that is what Euphonic expects:
        return self.brspgl.conventional_to_input_Q(self.grid.rlu)

    def __make_mesh(self, bz, max_size=-1, max_points=-1, num_levels=3, **kwds):
        self.grid = brille.BZMeshQdc(bz, max_size, num_levels, max_points)

    def __make_trellis(self, bz, max_volume=None, number_density=None, always_triangulate=False, **kwds):
        if max_volume is not None:
            self.grid = brille.BZTrellisQdc(bz, max_volume, always_triangulate)
        #elif number_density is not None:
        #    self.grid = brille.BZTrellisQdc(bz, number_density)
        else:
            raise Exception("keyword 'max_volume' or 'number_density' required")

    def __make_nest(self, bz, max_branchings=5, max_volume=None, number_density=None, **kwds):
        if max_volume is not None:
            self.grid = brille.BZNestQdc(bz, max_volume, max_branchings)
        elif number_density is not None:
            self.grid = brille.BZNestQdc(bz, number_density, max_branchings)
        else:
            raise Exception("keyword 'max_volume' or 'number_density' required")

    def __call__(self, *args, **kwargs):
        """Calculate and return Sᵢ(Q) and ωᵢ(Q) or S(Q,ω) depending on input.

        If one positional argument is provided it is assumed to be Q in which
        case both the intensity, Sᵢ(Q), and eigen-energy, ωᵢ(Q), for all phonon
        branches are returned as a tuple.
        If two positional arguments are provided they are assumed to be Q and ω
        in which case Sᵢ(Q) and ωᵢ(Q) are used in conjunction with keyword
        arguments 'resfun' and 'param' to calculate a convolved S(Q,ω).
        """
        if len(args) < 1:
            raise RuntimeError('At least one argument, Q, is required')
        elif len(args) is 1:
            return self.s_q(*args, **kwargs) # let the caller figure out how to handle a euphonic.StructureFactor
        elif len(args) is 2:
            return self.s_qw(*args, kwargs) # keep kwargs as a dictionary
        else:
            raise RuntimeError('Only one or two arguments expected, (Q,) or (Q,ω), expected')


    def s_q(self, q_hkl, interpolate=True, **kwargs):
        """Calculate Sᵢ(Q) where Q = (q_h,q_k,q_l)."""
        qωε = self.QpointPhononModes(q_hkl, **kwargs)
        # Finally calculate Sᵢ(Q)
        if interpolate:
            sf = qωε.calculate_structure_factor(self.data.crystal, self.scattering_lengths, **kwargs)
        else:
            # make the Euphonc.QpointPhononModes object, frequencies default to meV (good)
            euqpm = EuQpointPhononModes(self.data.crystal, qωε.Q, qωε.ω, qωε.ε)
            # using InterpolationData.calculate_structure_factor
            # which only allows a limited number of keyword arguments
            sf_keywords = ('dw',)
            sf_dict = {k: kwargs[k] for k in sf_keywords if k in kwargs}
            sf = euqpm.calculate_structure_factor(self.scattering_lengths, **sf_dict)
        return sf

    def dw(self, q_hkl, temperature=0):
        """Calculates the Debye-Waller factor using the Brillouin zone grid."""
        meVs2A2 = self.data.crystal.atom_mass.to('meV*s**2/angstrom**2').magnitude
        return self.grid.debye_waller(q_hkl, meVs2A2, temperature)

    def QpointPhononModes(self, q_pt, moveinto=True, interpolate=True, dw=None, temperature=5., threads=-1, **kwds):
        """Calculate ωᵢ(Q) where Q = (q_h,q_k,q_l)."""
        if interpolate:
            # Interpolate the previously-stored eigen values/vectors for each Q
            # each grid point has a (n_br, 1) values array and a (n_br, n_io, 3)
            # eigenvectors array and interpolate_at returns a tuple with the
            # first entry a (n_pt, n_br, 1) values array and the second a
            # (n_pt, n_br, n_io, 3) eigenvectors array
            if dw:
                mass = self.data.crystal.atom_mass.to('unified_atomic_mass_unit').magnitude
                frqs, vecs, Wd = self.grid.ir_interpolate_at_dw(q_pt, mass , temperature, self.parallel, threads, not moveinto)
                return BrQωε(q_pt, np.squeeze(frqs), vecs, Wd, temperature)
            else:
                frqs, vecs = self.grid.ir_interpolate_at(q_pt, self.parallel, threads, not moveinto)
                return BrQωε(q_pt, np.squeeze(frqs), vecs)
        else:
            cfp_kwds = ('asr', 'dipole', 'eta_scale', 'splitting',
                        'insert_gamma', 'reduce_qpts', 'fall_back_on_python')
            cfp_dict = {k: kwds[k] for k in cfp_kwds if k in kwds}
            cfp_dict['use_c'] = kwds.get('use_c', parallel)
            if cfp_dict['use_c']:
                cfp_dict['n_threads'] = kwds.get('n_threads', half_cpu_count())
            euqpm = self.data.calculate_qpoint_phonon_modes(self.brspgl.conventional_to_input_Q(q_pt), **cfp_dict)
            frqs = euqpm.frequencies
            vecs = euqpm.eigenvectors
            return BrQωε(q_pt, euqpm.frequencies, euqpm.eigenvectors)

    def w_q(self, q_pt, **kwds):
        qωε = self.QpointPhononModes(q_pt, **kwds)
        return qωε.ω

    def s_qw(self, q_hkl, energy, p_dict):
        """Calculate S(Q,E) for Q = (q_h, q_k, q_l) and E=energy.

        The last input, p_dict, should be a dict with keys 'resfun' and 'param'
        controlling the phonon-linewidth.

        ========================== ===================== ================
        Linewidth function           'resfun' (one of)       'param'
        ========================== ===================== ================
        Simple Harmonic Oscillator 'sho', 's'                 fwhm
        Gaussian                   'gauss', 'g'               fwhm
        Lorentzian                 'lorentz', 'lor', 'l'      fwhm
        Voigt                      'voi', 'v'            [g_fwhm, l_fwhm]
        ========================== ===================== ================

        For each linewidth function, the full name is also a valid value for
        'resfun', e.g., 'resfun':'Simple Harmonic Oscillator'.
        Functions taking a single 'param' value will use the first element in
        any non-scalar value.
        The Simple Harmonic Oscillator function looks for an additional key,
        'temperature', in p_dict to optionally include the temperature.

        Additional keys in p_dict are allowed and are passed on to Euphonic
        as keyword arguments to the calculate_structure_factor method.

        """
        res_par_tem = ('delta',)
        if 'resfun' in p_dict and 'param' in p_dict:
            res_par_tem = (p_dict['resfun'].replace(' ', '').lower(),
                           p_dict['param'])
        n_pt = energy.size
        n_br = 3*self.data.crystal.n_atoms
        # Check if we might perform the Bose factor correction twice:
        # Replicate Euphonic's behaviour of calc_bose=True by default,
        # and T=5 by default.
        if res_par_tem[0] in ('s', 'sho', 'simpleharmonicoscillator'):
            # pull out T, or 5 if it's not present
            temp_k = p_dict.get('temperature', 5)
            # If calc_bose is present
            if 'calc_bose' in p_dict:
                # keep T if it's True, discard T if it's False
                temp_k = temp_k if p_dict['calc_bose'] else 0
            # Prevent Euphonic from performing the Bose correction twice
            p_dict['calc_bose'] = False
            res_par_tem = (*res_par_tem, temp_k)
        # Calculate Sᵢ(Q) after interpolating ωᵢ(Q) and ⃗ϵᵢⱼ(Q)
        if 'unique_q' in p_dict and p_dict['unique_q']:
            # Avoid repeated Q entries for, e.g., (Q,E) maps
            # Finding unique points is 𝒪(q_hkl.shape[0])
            q_hkl, u_inv = np.unique(q_hkl, return_inverse=True, axis=0)
            s_i = self.s_q(q_hkl, **p_dict)[u_inv]
            omega = (s_i.frequencies.to('millielectron_volt')).magnitude[u_inv]
        else:
            s_i = self.s_q(q_hkl, **p_dict)
            omega = (s_i.frequencies.to('millielectron_volt')).magnitude
        # The resulting array *is* be (n_pt,n_br)

        shapein = energy.shape
        energy = energy.flatten()[:, None]
        # ωᵢ(Q)  is (n_pt,n_br)
        # Sᵢ(Q)  is (n_pt,n_br)
        # energy is (n_pt,1) [instead of (n_pt,)]

        # Broaden and then sum over the n_br branches
        s_q_e = broaden_modes(energy, omega, s_i.structure_factors.magnitude, res_par_tem).sum(1)
        if s_q_e.shape != shapein:
            s_q_e = s_q_e.reshape(shapein)
        return s_q_e
