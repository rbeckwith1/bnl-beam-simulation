"""
Matched-ellipse amplitude relation + initial bunch generation. This is
method-agnostic: every method initializes matched to whatever its own
starting (usually pre-jump/pre-ramp) voltage is, then hands off to its own
voltage program for tracking.

Convention: eps_l is the FULL matched-ellipse AREA, eps_l = pi*a_t*a_E, in
ns*GeV (== eV*s numerically -- GeV=1e9 eV, ns=1e-9 s cancel exactly).

TODO(Rosalyn): confirm with Dr. Brooks whether BNL's Bbat/Bbrat convention
reports the FULL ellipse area, area/pi ("normalized"), or a 4*sigma area --
this affects the prefactor used here.
"""

import numpy as np


def matched_ellipse_amplitudes(eps_l_ns_GeV, a_coef, b_coef):
    """a_t [ns], a_E [GeV] such that pi*a_t*a_E = eps_l_ns_GeV, with aspect
    ratio a_E/a_t = sqrt(|b/a|) fixed by the machine at the reference voltage."""
    aspect_E_over_t = np.sqrt(abs(b_coef / a_coef))
    a_t = np.sqrt(eps_l_ns_GeV / (np.pi * aspect_E_over_t))
    a_E = a_t * aspect_E_over_t
    return a_t, a_E


def initial_bunch(N, a_t, a_E, rng=None):
    """N particles uniformly filling the matched ellipse (time [ns], dE [GeV])."""
    rng = rng or np.random
    theta = 2.0 * np.pi * rng.rand(N)
    r = np.sqrt(rng.rand(N))
    time = a_t * r * np.cos(theta)
    dE = a_E * r * np.sin(theta)
    return time, dE
