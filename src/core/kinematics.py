"""
Reference-particle kinematics and the exact drift map.

This is pulled directly from the "REFERENCE PARTICLE KINEMATICS" and
"EXACT REVOLUTION TIME AND DRIFT MAP" sections that used to be duplicated
at the top of every method script. It depends only on core.constants, so
adiabatic/non_adiabatic/resonant all import the same numbers.
"""

import numpy as np

from core.constants import K0, mp, c, L0, gamma_t, alpha_p, h


def _derive(K0_val):
    """E0, p0, beta0, gamma0, T0, T_rf for a reference particle at kinetic
    energy K0_val. Single source of truth -- used both for the fixed
    module-level constants below (non-accelerating methods) and inside
    ReferenceParticle (accelerating methods)."""
    E0_ = K0_val + mp
    p0_ = np.sqrt(E0_**2 - mp**2)
    beta0_ = p0_ / E0_
    gamma0_ = E0_ / mp
    T0_ = L0 / (beta0_ * c)
    T_rf_ = T0_ / h
    return E0_, p0_, beta0_, gamma0_, T0_, T_rf_


E0, p0, beta0, gamma0, T0, T_rf = _derive(K0)

if gamma0 <= gamma_t:
    raise ValueError(
        f"Reference particle (gamma0={gamma0:.4f}) is not above transition "
        f"(gamma_t={gamma_t}); this codebase assumes above-transition operation."
    )

T0_ns = T0 * 1e9
T_rf_ns = T_rf * 1e9

eta_slip = alpha_p - 1.0 / gamma0**2   # linear slip factor (>0 above transition)


def revolution_time(dE):
    """Exact revolution period [s] of a particle with energy deviation dE [GeV],
    relative to the FIXED reference K0 above. Non-accelerating methods only --
    see ReferenceParticle for the turn-by-turn-updated version."""
    E = E0 + dE
    p = np.sqrt(np.maximum(E**2 - mp**2, 1e-30))
    beta = p / E
    dp_over_p0 = (p - p0) / p0
    C = L0 * (1.0 + alpha_p * dp_over_p0)
    return C / (beta * c)


def drift_map(t_ns, dE):
    """One-turn drift: update arrival-time deviation t [ns] at fixed dE [GeV]."""
    T = revolution_time(dE)
    dt_ns = (T - T0) * 1e9
    return t_ns + dt_ns


def wrap_to_bucket(time_ns):
    """Wrap a time coordinate into (-T_rf_ns/2, +T_rf_ns/2]."""
    return ((time_ns + T_rf_ns / 2) % T_rf_ns) - T_rf_ns / 2


class ReferenceParticle:
    """
    Mutable reference-particle state for runs where K0 changes turn by turn
    (acceleration). Non-accelerating methods can ignore this entirely and
    keep using the fixed T0 / T0_ns / T_rf_ns constants above -- those are
    computed once and never touched by this class.

    track_bunch() always builds one of these internally now; when no
    AccelerationProgram is supplied (or it's disabled), .accelerate() is
    simply never called, so K0 stays fixed at its initial value and this
    behaves identically to the old fixed-constant module.
    """

    def __init__(self, K0_init=K0):
        self.K0 = K0_init
        self._refresh()

    def _refresh(self):
        self.E0, self.p0, self.beta0, self.gamma0, self.T0, self.T_rf = _derive(self.K0)
        self.T0_ns = self.T0 * 1e9
        self.T_rf_ns = self.T_rf * 1e9

    def revolution_time(self, dE):
        E = self.E0 + dE
        p = np.sqrt(np.maximum(E**2 - mp**2, 1e-30))
        beta = p / E
        dp_over_p0 = (p - self.p0) / self.p0
        C = L0 * (1.0 + alpha_p * dp_over_p0)
        return C / (beta * c)

    def drift_map(self, t_ns, dE):
        T = self.revolution_time(dE)
        return t_ns + (T - self.T0) * 1e9

    def wrap_to_bucket(self, t_ns):
        return ((t_ns + self.T_rf_ns / 2) % self.T_rf_ns) - self.T_rf_ns / 2

    def accelerate(self, dK0_turn):
        """Apply one turn's worth of reference energy gain [GeV] and
        recompute all derived quantities (T0, T_rf, etc.) at the new K0."""
        if dK0_turn != 0.0:
            self.K0 += dK0_turn
            self._refresh()


def print_summary():
    print("=" * 70)
    print("AGS REFERENCE PARTICLE / MACHINE PARAMETERS")
    print("=" * 70)
    print(f"  Total energy E0            = {E0:.6f} GeV")
    print(f"  gamma0                     = {gamma0:.6f}   (gamma_t = {gamma_t})")
    print(f"  Slip factor eta            = {eta_slip:.6e}")
    print(f"  Revolution period T0       = {T0_ns:.6f} ns")
    print(f"  RF period T_rf = T0/h      = {T_rf_ns:.6f} ns")
