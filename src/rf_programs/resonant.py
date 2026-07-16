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
                 detuning=0.0, start_turn=0, ramp_turns=0, mod_phase=0.0):
        self.Vrf_mean = Vrf_mean
        self.mod_depth = mod_depth
        self.omega_mod = resonance_ratio * omega_s * (1.0 + detuning)
        self.start_turn = start_turn
        self.ramp_turns = ramp_turns
        self.mod_phase = mod_phase

    def depth_at_turn(self, n):
        """Effective modulation depth at turn n: 0 before start_turn, then a
        raised-cosine ramp 0 -> mod_depth over ramp_turns (zero slope at
        both ends). ramp_turns <= 0 recovers an instantaneous switch-on."""
        if n < self.start_turn:
            return 0.0
        tau = n - self.start_turn
        if self.ramp_turns <= 0:
            return self.mod_depth
        ramp_frac = min(tau / self.ramp_turns, 1.0)
        smooth = 0.5 * (1.0 - np.cos(np.pi * ramp_frac))
        return self.mod_depth * smooth

    def __call__(self, turn):
        if turn < self.start_turn:
            V = self.Vrf_mean
        else:
            tau = turn - self.start_turn
            depth_n = self.depth_at_turn(turn)
            V = self.Vrf_mean * (1.0 + depth_n *
                                  np.cos(self.omega_mod * tau + self.mod_phase))
        return float(np.clip(V, 1.0e-9, None))   # keep strictly positive/physical
