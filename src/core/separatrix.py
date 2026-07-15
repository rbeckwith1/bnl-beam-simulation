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
    def G_of_t(t_ns, Vrf):
        return -Vrf * (T0_ns / (2.0 * np.pi * h)) * np.cos(2.0 * np.pi * h * t_ns / T0_ns)

    def separatrix_H(self, Vrf):
        t_u = T0_ns / (2.0 * h)
        return self.G_of_t(t_u, Vrf)

    def separatrix_dE(self, t_array, Vrf, dE_search_max=None):
        """Solve H(t, dE) = H_sep for dE; returns (dE_pos, dE_neg) branches."""
        if dE_search_max is None:
            dE_search_max = 0.5 * self._dE_grid_max

        if Vrf <= 0:
            return (np.full_like(t_array, np.nan, dtype=float),
                     np.full_like(t_array, np.nan, dtype=float))

        H_sep = self.separatrix_H(Vrf)
        dE_pos = np.full_like(t_array, np.nan, dtype=float)
        dE_neg = np.full_like(t_array, np.nan, dtype=float)

        for i, t in enumerate(t_array):
            target = H_sep - self.G_of_t(t, Vrf)

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
