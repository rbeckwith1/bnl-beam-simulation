"""
RF kick (exact nonlinear form, above-transition convention: stable phase
is pi), the combined one-turn map, and the linearized a/b coefficients used
for the linear synchrotron-frequency estimate and matched-ellipse aspect
ratio. Any method-specific Vrf is passed in by the caller -- nothing here
hardcodes a particular voltage program.
"""

import numpy as np

from core.constants import h
from core.kinematics import T0_ns, revolution_time


def rf_kick(t_ns, dE, Vrf, phi_ref=np.pi, T0_ns=T0_ns):
    """
    phi = 2*pi*h*t_ns/T0_ns + phi_ref; dE_new = dE + Vrf*(sin(phi) - sin(phi_ref)).

    phi_ref defaults to pi -- the above-transition stationary-bucket stable
    phase (verified by check_fixed_point_stability) -- in which case
    sin(phi_ref)=0 and this reduces exactly to the original
    dE + Vrf*sin(phi) formula.

    For an accelerating bucket, phi_ref = pi - phi_s (see
    core.acceleration.AccelerationProgram): the -sin(phi_ref) term is what
    keeps dE a deviation *relative to the synchronously accelerating
    reference particle*, whose own energy gain (Vrf*sin(phi_s) per turn) is
    applied separately via ReferenceParticle.accelerate().

    T0_ns defaults to the fixed module-level constant; pass the current
    ReferenceParticle.T0_ns instead when the reference energy is changing.
    """
    phi = 2.0 * np.pi * h * t_ns / T0_ns + phi_ref
    return dE + Vrf * (np.sin(phi) - np.sin(phi_ref))


def one_turn_map(t_ns, dE, Vrf, phi_ref=np.pi, T0_ns=T0_ns):
    """DRIFT (using current dE) then KICK (using post-drift t). Uses the
    fixed-K0 module drift_map -- fine for the linear/Jacobian synchrotron-
    frequency and separatrix diagnostics, which are always evaluated at a
    single snapshot K0, not inside the accelerating tracking loop itself."""
    from core.kinematics import drift_map
    t_new = drift_map(t_ns, dE)
    dE_new = rf_kick(t_new, dE, Vrf, phi_ref=phi_ref, T0_ns=T0_ns)
    return t_new, dE_new


def compute_a_coefficient():
    """a in dt/dn = a*dE [ns/GeV], numerical derivative of revolution_time at dE=0."""
    dE_step = 1.0e-6
    T_plus = revolution_time(dE_step)
    T_minus = revolution_time(-dE_step)
    dTdE = (T_plus - T_minus) / (2.0 * dE_step)
    return dTdE * 1e9


def compute_b_coefficient(Vrf_ref):
    """b in ddE/dn = b*t [GeV/ns], linearizing the RF kick about (t=0, dE=0)."""
    return -Vrf_ref * 2.0 * np.pi * h / T0_ns


def check_fixed_point_stability(a, b):
    """Require a*b < 0 (omega_s^2 = -a*b > 0) for (t=0, dE=0) to be a stable center."""
    prod = a * b
    if prod >= 0:
        raise RuntimeError(
            f"Fixed point (t=0, dE=0) is NOT stable: a*b = {prod:.6e} >= 0. "
            f"Re-check the RF phase convention (phi offset)."
        )
    print(f"Fixed-point stability check passed: a={a:.6e} ns/GeV, "
          f"b={b:.6e} GeV/ns, a*b={prod:.6e} < 0 (stable).")

