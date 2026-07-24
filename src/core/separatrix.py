
"""
Continuous-Hamiltonian approximation to the drift-then-kick map, used only
to draw the separatrix for diagnostics/animation (see LIMITATION note below
-- unchanged from the original derivation).

Wrapped as a class (instead of module-level globals) so each method script
builds ONE Separatrix object sized for its own max voltage, rather than
sharing/overwriting global grid state.
"""

import warnings
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

from core.constants import h
from core.kinematics import T0, T0_ns, revolution_time


class Separatrix:
    """
    LIMITATION: this is a continuous (Hamiltonian-flow) approximation to a
    discrete drift-THEN-kick map, exact only for small per-turn phase
    advance. Standard accelerator-physics approximation, not an exact
    invariant of the tracked map.
    """

    def __init__(self, Vrf_max_expected):
        self._dE_grid_max = self._estimate_bucket_dE_scale(Vrf_max_expected)
        self._dE_grid = np.linspace(-self._dE_grid_max, self._dE_grid_max, 8001)
        integrand = (revolution_time(self._dE_grid) - T0) * 1e9
        F_grid = np.concatenate([[0.0], np.cumsum(
            0.5 * (integrand[1:] + integrand[:-1]) * np.diff(self._dE_grid))])
        zero_idx = np.argmin(np.abs(self._dE_grid))
        self._F_grid = F_grid - F_grid[zero_idx]

    @staticmethod
    def _estimate_bucket_dE_scale(Vrf_max_expected):
        Vrf_hi = Vrf_max_expected * 1.2
        H_sep_hi = Vrf_hi * (T0_ns / (2.0 * np.pi * h))
        target_hi = 2.0 * H_sep_hi
        dE_trial = 0.2
        for _ in range(20):
            val, _ = quad(lambda e: (revolution_time(e) - T0) * 1e9, 0, dE_trial)
            if val >= target_hi:
                return dE_trial
            dE_trial *= 1.5
        return dE_trial

    def F_of_dE(self, dE_val):
        return np.interp(dE_val, self._dE_grid, self._F_grid)

    @staticmethod
    def G_of_t(t_ns, Vrf, phi_ref=np.pi):
        """
        RF-potential piece of the continuous Hamiltonian, generalized to an
        arbitrary reference phase phi_ref (= pi - phi_s for an accelerating
        bucket, phi_s = synchronous phase). At phi_ref=pi (stationary
        bucket, phi_s=0) the sin(phi_ref)*t_ns term vanishes and this
        reduces EXACTLY to the original -Vrf*(T0_ns/2*pi*h)*cos(...)
        formula -- verified numerically to float noise.
        """
        phase = 2.0 * np.pi * h * t_ns / T0_ns + phi_ref
        return (Vrf * (T0_ns / (2.0 * np.pi * h)) * np.cos(phase)
                + Vrf * np.sin(phi_ref) * t_ns)

    def separatrix_H(self, Vrf, phi_ref=np.pi):
        """
        H at the unstable fixed point. For a stationary bucket (phi_ref=pi)
        that's the traditional t_u = T0_ns/(2h). For an accelerating bucket
        the unstable fixed point (in absolute RF phase) sits at
        pi - phi_ref, i.e. at deviation-phase phi_particle_u = pi - 2*phi_ref
        relative to the reference particle -- solved from
        sin(phi_ref + phi_particle) = sin(phi_ref) (the kick vanishing
        condition), same result as the phi_u = pi - phi_ref convention used
        for the accelerating-bucket separatrix in the original script.
        G_of_t is periodic in t_ns with period T_rf_ns, so it doesn't matter
        that this t_u differs by a half-integer number of periods from the
        old formula at phi_ref=pi -- they land on the same G value.
        """
        phi_particle_u = np.pi - 2.0 * phi_ref
        t_u = phi_particle_u * T0_ns / (2.0 * np.pi * h)
        return self.G_of_t(t_u, Vrf, phi_ref)

    def unstable_fixed_point_t_ns(self, phi_ref=np.pi):
        """
        Time-coordinate (t_ns, deviation from the reference particle) of
        the unstable fixed point for the given phi_ref. Same convention as
        separatrix_H/separatrix_dE: phi_ref = pi - phi_s, phi_ref = pi for
        a stationary bucket. The stable fixed point is always t_ns = 0 in
        this coordinate.
        """
        phi_particle_u = np.pi - 2.0 * phi_ref
        return phi_particle_u * T0_ns / (2.0 * np.pi * h)

    def separatrix_dE(self, t_array, Vrf, phi_ref=np.pi, dE_search_max=None):
        """
        Solve H(t, dE) = H_sep for dE; returns (dE_pos, dE_neg) branches.

        phi_ref defaults to pi (stationary bucket). Pass phi_ref = pi -
        phi_s for an accelerating snapshot -- e.g. from
        snapshots["phi_s"][i] in core.tracking's output, phi_ref =
        np.pi - phi_s -- to get the shrunken/asymmetric accelerating bucket
        instead of the stationary one.

        CAVEAT: F_of_dE (the drift/energy part of H) is still built once,
        in __init__, from the FIXED K0 in core.constants -- it does not
        track K0 climbing during acceleration. For the phi_s <~ 30 deg,
        small-dK0 ramps this scaffold is currently used for, that's a minor
        approximation; for a large energy excursion, F_of_dE would need to
        be rebuilt from a core.kinematics.ReferenceParticle at the
        snapshot's current K0 instead.
        """
        if dE_search_max is None:
            dE_search_max = 0.5 * self._dE_grid_max

        if Vrf <= 0:
            return (np.full_like(t_array, np.nan, dtype=float),
                     np.full_like(t_array, np.nan, dtype=float))

        H_sep = self.separatrix_H(Vrf, phi_ref)
        dE_pos = np.full_like(t_array, np.nan, dtype=float)
        dE_neg = np.full_like(t_array, np.nan, dtype=float)

        for i, t in enumerate(t_array):
            target = H_sep - self.G_of_t(t, Vrf, phi_ref)

            bracket_hi = min(dE_search_max, self._dE_grid_max)
            found = False
            for _ in range(6):
                if self.F_of_dE(bracket_hi) >= target:
                    found = True
                    break
                if bracket_hi >= self._dE_grid_max:
                    break
                bracket_hi = min(bracket_hi * 2.0, self._dE_grid_max)
            if found and target >= 0.0:
                try:
                    dE_pos[i] = brentq(lambda e: self.F_of_dE(e) - target, 0.0, bracket_hi)
                except ValueError as exc:
                    warnings.warn(f"Brent failed (positive branch) at t={t:.3f} ns: {exc}")

            bracket_lo = max(-dE_search_max, -self._dE_grid_max)
            found = False
            for _ in range(6):
                if self.F_of_dE(bracket_lo) >= target:
                    found = True
                    break
                if bracket_lo <= -self._dE_grid_max:
                    break
                bracket_lo = max(bracket_lo * 2.0, -self._dE_grid_max)
            if found and target >= 0.0:
                try:
                    dE_neg[i] = brentq(lambda e: self.F_of_dE(e) - target, bracket_lo, 0.0)
                except ValueError as exc:
                    warnings.warn(f"Brent failed (negative branch) at t={t:.3f} ns: {exc}")
        
        return dE_pos, dE_neg

    @property
    def dE_grid_max(self):
        return self._dE_grid_max
    