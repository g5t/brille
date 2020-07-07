import numpy as np
from scipy import special
from scipy.stats import norm, cauchy

def broaden_modes(energy, omega, s_i, res_par_tem):
    """Compute S(Q,E) for a number of dispersion relations and intensities.

    Given any number of dispersion relations, ω(Q), and the intensities of the
    modes which they represent, S(Q), plus energy-broadening information in
    the form of a function name plus parameters (if required), calculate S(Q,E)
    at the provided energy positions.

    The energy positions must have shape (Npoints,).
    The dispersion and intensities must have been precalculated and should have
    shape similar to (Npoints, Nmodes). This function calls one of five
    available broadening functions, a simple harmonic oscillator, gaussian,
    lorentzian, voigt, or delta function.
    The retuned S(Q,E) array will have shape (Npoints, Nmodes).
    """
    if res_par_tem[0] in ('s', 'sho', 'simpleharmonicoscillator'):
        s_q_e = sho(energy, omega, s_i, res_par_tem[1], res_par_tem[2])
    elif res_par_tem[0] in ('g', 'gauss', 'gaussian'):
        s_q_e = gaussian(energy, omega, s_i, res_par_tem[1])
    elif res_par_tem[0] in ('l', 'lor', 'lorentz', 'lorentzian'):
        s_q_e = lorentzian(energy, omega, s_i, res_par_tem[1])
    elif res_par_tem[0] in ('v', 'voi', 'voigt'):
        s_q_e = voigt(energy, omega, s_i, res_par_tem[1])
    elif res_par_tem[0] in ('d', 'del', 'delta'):
        s_q_e = delta(energy, omega, s_i)
    else:
        print("Unknown function {}".format(res_par_tem[0]))
        s_q_e = s_i
    return s_q_e


def delta(x_0, x_i, y_i):
    """
    Compute the δ-function.

    y₀ = yᵢ×δ(x₀-xᵢ)
    """
    y_0 = np.zeros(y_i.shape, dtype=y_i.dtype)
    # y_0 = np.zeros_like(y_i)
    y_0[x_0 == x_i] = y_i[x_0 == x_i]
    return y_0


def gaussian(x_0, x_i, y_i, fwhm):
    """Compute the normal distribution with full-width-at-half-maximum fwhm."""
    if not np.isscalar(fwhm):
        fwhm = fwhm[0]
    sigma = fwhm/np.sqrt(np.log(256))
    z_0 = (x_0-x_i)/sigma
    y_0 = norm.pdf(z_0) * y_i
    return y_0


def lorentzian(x_0, x_i, y_i, fwhm):
    """Compute the Cauchy distribution with full-width-at-half-maximum fwhm."""
    if not np.isscalar(fwhm):
        fwhm = fwhm[0]
    gamma = fwhm/2
    z_0 = (x_0-x_i)/gamma
    y_0 = cauchy.pdf(z_0) * y_i
    return y_0


def voigt(x_0, x_i, y_i, params):
    """Compute the convolution of a normal and Cauchy distribution.

    The Voigt function is the exact convolution of a normal distribution (a
    Gaussian) with full-width-at-half-max gᶠʷʰᵐ and a Cauchy distribution
    (a Lorentzian) with full-with-at-half-max lᶠʷʰᵐ. Computing the Voigt
    function exactly is computationally expensive, but it can be approximated
    to (almost always nearly) machine precision quickly using the [Faddeeva
    distribution](http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package).

    The Voigt distribution is the real part of the Faddeeva distribution,
    given an appropriate rescaling of the parameters. See, e.g.,
    https://en.wikipedia.org/wiki/Voigt_profile.
    """
    if np.isscalar(params):
        g_fwhm = params
        l_fwhm = 0
    else:
        g_fwhm = params[0]
        l_fwhm = params[1]
    if l_fwhm == 0:
        return gaussian(x_0, x_i, y_i, g_fwhm)
    if g_fwhm == 0:
        return lorentzian(x_0, x_i, y_i, l_fwhm)

    area = np.sqrt(np.log(2)/np.pi)
    gamma = g_fwhm/2
    real_z = np.sqrt(np.log(2))*(x_0-x_i)/gamma
    imag_z = np.sqrt(np.log(2))*np.abs(l_fwhm/g_fwhm)
    # pylint: disable=no-member
    y_0 = area*np.real(special.wofz(real_z + 1j*imag_z))/gamma
    return y_0


def sho(x_0, x_i, y_i, fwhm, t_k):
    """Compute the Simple-Harmonic-Oscillator distribution."""
    # (partly) ensure that all inputs have the same shape:
    if np.isscalar(fwhm):
        fwhm = fwhm * np.ones(y_i.shape)
    if np.isscalar(t_k):
        t_k = t_k * np.ones(y_i.shape)
    if x_0.ndim < x_i.ndim or (x_0.shape[1] == 1 and x_i.shape[1] > 1):
        x_0 = np.repeat(x_0, x_i.shape[1], 1)
    # include the Bose factor if the temperature is non-zero
    bose = x_0 / (1-np.exp(-11.602*x_0/t_k))
    bose[t_k == 0] = 1.0
    # We need x₀² the same shape as xᵢ
    x_02 = x_0**2
    # and to ensure that only valid (finite) modes are included
    flag = (x_i != 0) * np.isfinite(x_i)
    # create an output array
    y_0 = np.zeros(y_i.shape)
    # flatten everything so that we can use logical indexing
    # keeping the original output shape
    outshape = y_0.shape
    bose = bose.flatten()
    fwhm = fwhm.flatten()
    y_0 = y_0.flatten()
    x_i = x_i.flatten()
    y_i = y_i.flatten()
    x_02 = x_02.flatten()
    flag = flag.flatten()
    # and actually calculate the distribution
    part1 = bose[flag]*(4/np.pi)*fwhm[flag]*x_i[flag]*y_i[flag]
    part2 = ((x_02[flag]-x_i[flag]**2)**2 + 4*fwhm[flag]**2*x_02[flag])
    # if the brille object is holding complex values (it is) then its returned
    # interpolated values are all complex too, even, e.g., energies which are
    # purely real with identically zero imaginary components.
    # The division of two purely-real complex numbers in Python will annoyingly
    # raise a warning about discarding the imaginary part. So preempt it here.
    if not np.isclose(np.sum(np.abs(np.imag(part1))+np.abs(np.imag(part2))), 0.):
        raise RuntimeError('Unexpected imaginary components.')
    y_0[flag] = np.real(part1)/np.real(part2)
    return y_0.reshape(outshape)

def half_cpu_count():
    import os
    count = os.cpu_count()
    if 'sched_get_affinity' in dir(os):
        count = len(os.sched_getaffinity(0))
    return count//2
