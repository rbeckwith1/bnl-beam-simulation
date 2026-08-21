"""
Matched-ellipse amplitude relation + initial bunch generation. This is
method-agnostic: every method initializes matched to whatever its own
starting (usually pre-jump/pre-ramp) voltage is, then hands off to its own
voltage program for tracking.

Convention: eps_l is the FULL matched-ellipse AREA, eps_l = pi*a_t*a_E, in
ns*GeV (== eV*s numerically -- GeV=1e9 eV, ns=1e-9 s cancel exactly).
"""

import numpy as np


def matched_ellipse_amplitudes(eps_l_ns_GeV, a_coef, b_coef):
    """a_t [ns], a_E [GeV] such that pi*a_t*a_E = eps_l_ns_GeV, with aspect
    ratio a_E/a_t = sqrt(|b/a|) fixed by the machine at the reference voltage."""
    aspect_E_over_t = np.sqrt(abs(b_coef / a_coef))
    a_t = np.sqrt(eps_l_ns_GeV / (np.pi * aspect_E_over_t))
    a_E = a_t * aspect_E_over_t
    return a_t, a_E

def initial_bunch_gaussian(N, a_t, a_E, n_sigma_edge=1.0, truncate=6.0, rng=None):
    """N particles Gaussian-distributed, with a_t, a_E treated as the
    n_sigma_edge-sigma half-widths.
    time [ns], dE [GeV].

    truncate: redraw any particle beyond this many sigma so no one starts
    outside the matched ellipse edge (set to None to allow untruncated tails).
    """
    rng = rng or np.random
    sigma_t = a_t / n_sigma_edge
    sigma_E = a_E / n_sigma_edge

    time = rng.normal(0.0, sigma_t, N)
    dE = rng.normal(0.0, sigma_E, N)

    if truncate is not None:
        r_sigma = np.sqrt((time / sigma_t)**2 + (dE / sigma_E)**2)
        bad = r_sigma > truncate
        while np.any(bad):
            n_bad = bad.sum()
            time[bad] = rng.normal(0.0, sigma_t, n_bad)
            dE[bad] = rng.normal(0.0, sigma_E, n_bad)
            r_sigma = np.sqrt((time / sigma_t)**2 + (dE / sigma_E)**2)
            bad = r_sigma > truncate

    return time, dE

def initial_bunch_gaussian_from_rms(N, sigma_t, sigma_E, truncate=5.0, rng=None):
    """
    N particles Gaussian-distributed with sigma_t, sigma_E as the
    actual 1-sigma RMS widths (no ellipse/emittance conversion).
    time [ns], dE [GeV].
    truncate: redraw particles beyond this many RMS-sigma so no one
    starts unphysically far out (set to None to allow untruncated tails).
    """
    rng = rng or np.random
    time = rng.normal(0.0, sigma_t, N)
    dE = rng.normal(0.0, sigma_E, N)
    if truncate is not None:
        r_sigma = np.sqrt((time / sigma_t)**2 + (dE / sigma_E)**2)
        bad = r_sigma > truncate
        while np.any(bad):
            n_bad = bad.sum()
            time[bad] = rng.normal(0.0, sigma_t, n_bad)
            dE[bad] = rng.normal(0.0, sigma_E, n_bad)
            r_sigma = np.sqrt((time / sigma_t)**2 + (dE / sigma_E)**2)
            bad = r_sigma > truncate
    return time, dE

def initial_bunch_ellipse_family(N, a_t, a_E, J=1.5, rng=None):
    """
    Generate an elliptical phase-space distribution with

        f(t, dE) ∝ [1 - (t/a_t)^2 - (dE/a_E)^2]^(J - 1/2)

    whose time projection is

        I(t) ∝ [1 - (t/a_t)^2]^J.

    J = 0.5 gives a uniformly filled ellipse.
    """
    rng = rng or np.random

    if J < 0.5:
        raise ValueError("J must be >= 0.5")

    # Uniform angle around the phase-space ellipse
    theta = 2.0 * np.pi * rng.rand(N)

    u = 1.0 - rng.rand(N)**(1.0 / (J + 0.5))
    r = np.sqrt(u)

    # Convert normalized polar coordinates into
    # longitudinal phase-space coordinates.
    #
    # This gives:
    #     r^2 = (time/a_t)^2 + (dE/a_E)^2

    time = a_t * r * np.cos(theta)
    dE = a_E * r * np.sin(theta)

    return time, dE

# def initial_bunch_uniform(N, a_t, a_E, rng=None):
#     """N particles uniformly filling the matched ellipse (time [ns], dE [GeV])."""
#     rng = rng or np.random
#     theta = 2.0 * np.pi * rng.rand(N)
#     r = np.sqrt(rng.rand(N))
#     time = a_t * r * np.cos(theta)
#     dE = a_E * r * np.sin(theta)
#     return time, dE

def initial_bunch(N, a_t, a_E, method="ellipse", rng=None, **kwargs):

    if method == "ellipse":
        return initial_bunch_ellipse_family(
            N, a_t, a_E, rng=rng, **kwargs
        )

    elif method == "gaussian":
        return initial_bunch_gaussian(
            N, a_t, a_E, rng=rng, **kwargs
        )

    elif method == "gaussian_rms":
        return initial_bunch_gaussian_from_rms(
            N, a_t, a_E, rng=rng, **kwargs
        )

    else:
        raise ValueError(f"unknown method: {method!r}")
