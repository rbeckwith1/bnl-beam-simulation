"""
Resonant (parametric) bunching: RF voltage modulation near 2*omega_s, with a
raised-cosine ramp-up of the modulation depth so the switch-on itself does
not inject a broadband transient that could be mistaken for resonant growth.

    V(n) = Vrf_mean * (1 + depth(n) * cos(omega_mod * tau + mod_phase))

where tau = n - start_turn and omega_mod = resonance_ratio * omega_s *
(1 + detuning) (resonance_ratio=2.0 is the classic quadrupole parametric
resonance condition).

NOTE on mod_phase: the envelope orientation angle theta_Q evolves at
2*omega_s. Whether the modulation AMPLIFIES or SUPPRESSES the envelope
oscillation depends on the relative phase between cos(omega_mod*tau +
mod_phase) and cos(2*theta_Q(n)) -- shifting mod_phase by pi/2 turns
parametric driving into parametric damping (or vice versa); shifting by pi
just reverses which quadrature is driven. There is no a-priori "correct"
mod_phase = 0; check it against the Q1/Q2/theta_Q diagnostics after a run.
"""

import numpy as np

class ResonantProgram:
    def __init__(self, Vrf_mean, mod_depth, omega_s, resonance_ratio=2.0,
                 detuning=0.0, start_turn=0, ramp_turns=0, mod_phase=0.0,
                 stop_turn=None, rampdown_turns=0, V_start_level=None):
        self.Vrf_mean = Vrf_mean
        self.mod_depth = mod_depth
        self.omega_mod = resonance_ratio * omega_s * (1.0 + detuning)
        self.start_turn = start_turn
        self.ramp_turns = ramp_turns
        self.mod_phase = mod_phase
        self.stop_turn = stop_turn
        self.rampdown_turns = rampdown_turns
        # if set, the mean itself ramps V_start_level -> Vrf_mean over the same
        # ramp_turns window as the depth ramp-up (None = old behavior, mean fixed)
        self.V_start_level = V_start_level

    def _ramp_frac(self, n):
        tau_up = n - self.start_turn
        if self.ramp_turns > 0 and tau_up < self.ramp_turns:
            frac = tau_up / self.ramp_turns
            return 0.5 * (1.0 - np.cos(np.pi * frac))
        return 1.0

    def mean_at_turn(self, n):
        if self.V_start_level is None or n < self.start_turn:
            return self.Vrf_mean
        smooth = self._ramp_frac(n)
        return self.V_start_level + (self.Vrf_mean - self.V_start_level) * smooth

    def depth_at_turn(self, n):
        # unchanged from your version
        if n < self.start_turn:
            return 0.0
        tau_up = n - self.start_turn
        if self.ramp_turns > 0 and tau_up < self.ramp_turns:
            return self.mod_depth * self._ramp_frac(n)
        if self.stop_turn is None:
            return self.mod_depth
        if n < self.stop_turn:
            return self.mod_depth
        tau_down = n - self.stop_turn
        if self.rampdown_turns <= 0:
            return 0.0
        if tau_down < self.rampdown_turns:
            ramp_frac = tau_down / self.rampdown_turns
            return self.mod_depth * 0.5 * (1.0 + np.cos(np.pi * ramp_frac))
        return 0.0

    def __call__(self, turn):
        if turn < self.start_turn:
            V = self.Vrf_mean
        else:
            tau = turn - self.start_turn
            mean_n = self.mean_at_turn(turn)
            depth_n = self.depth_at_turn(turn)
            # amplitude scaled by Vrf_mean (not mean_n) so mean and amplitude
            # ramp in lockstep -> V(n) stays within [Vrf_mean(1-mod_depth), V_max]
            # for the whole ramp, no transient overshoot past V_max
            amp_n = self.Vrf_mean * depth_n
            V = mean_n + amp_n * np.cos(self.omega_mod * tau + self.mod_phase)
        return float(np.clip(V, 1.0e-9, None))