"""
Synchrotron frequency via two independent methods, used as a cross-check:
  Method 1: linear (a,b) estimate
  Method 2: Jacobian of the exact one-turn map

Call get_omega_s(Vrf_ref) to get the working value (Method 2, exact) plus
both estimates printed for comparison -- this replaces the block that used
to be copy-pasted near the top of every method script.
"""

import warnings
import numpy as np

from core.rf import compute_a_coefficient, compute_b_coefficient, one_turn_map


def compute_exact_jacobian(Vrf_ref):
    eps_t = 1.0e-3   # ns
    eps_E = 1.0e-8   # GeV

    def m(t_ns, dE):
        return one_turn_map(t_ns, dE, Vrf_ref)

    t_p, dE_p = m(eps_t, 0.0)
    t_m, dE_m = m(-eps_t, 0.0)
    dtdt = (t_p - t_m) / (2 * eps_t)
    ddEdt = (dE_p - dE_m) / (2 * eps_t)

    t_pE, dE_pE = m(0.0, eps_E)
    t_mE, dE_mE = m(0.0, -eps_E)
    dtdE = (t_pE - t_mE) / (2 * eps_E)
    ddEdE = (dE_pE - dE_mE) / (2 * eps_E)

    return np.array([[dtdt, dtdE], [ddEdt, ddEdE]])


def get_omega_s(Vrf_ref, verbose=True):
    """Returns (omega_s, T_s_turns, a_coef, b_coef) using the exact Jacobian
    method as the working value; Method 1 (linear a,b) is computed too and
    printed as a cross-check."""
    a_coef = compute_a_coefficient()
    b_coef = compute_b_coefficient(Vrf_ref)

    omega_s_lin = np.sqrt(-a_coef * b_coef)
    T_s_lin_turns = 2.0 * np.pi / omega_s_lin

    J_exact = compute_exact_jacobian(Vrf_ref)
    eigvals, _ = np.linalg.eig(J_exact)

    if np.any(np.abs(np.abs(eigvals) - 1.0) > 1e-3):
        warnings.warn(
            f"Jacobian eigenvalues not on unit circle (|lambda|={np.abs(eigvals)}); "
            f"linear approx may be inaccurate or fixed point unstable here."
        )

    mu_s = np.abs(np.angle(eigvals[0]))
    if mu_s == 0:
        raise RuntimeError("Jacobian eigenvalues purely real; fixed point not elliptic.")

    T_s_jac_turns = 2.0 * np.pi / mu_s
    omega_s_jac = mu_s

    if verbose:
        print(f"  Method 1 (linear a,b):        T_s = {T_s_lin_turns:.3f} turns")
        print(f"  Method 2 (exact map Jacobian): T_s = {T_s_jac_turns:.3f} turns")
        print(f"  Relative difference: "
              f"{abs(T_s_jac_turns - T_s_lin_turns) / T_s_jac_turns * 100:.3f} %")

    return omega_s_jac, T_s_jac_turns, a_coef, b_coef
