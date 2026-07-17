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
                 stop_turn=None, rampdown_turns=0):
        """
        Parameters
        ----------
        stop_turn : int or None
            Turn at which the modulation begins ramping back down to zero.
            None (default) => no ramp-down; modulation holds at mod_depth
            forever once ramp_turns has completed (old behavior, unchanged).
        rampdown_turns : int
            Turns over which depth ramps mod_depth -> 0, using the same
            raised-cosine shape as the ramp-up (zero slope at both ends).
            rampdown_turns <= 0 with stop_turn set recovers an instantaneous
            switch-off at stop_turn.
        """
        self.Vrf_mean = Vrf_mean
        self.mod_depth = mod_depth
        self.omega_mod = resonance_ratio * omega_s * (1.0 + detuning)
        self.start_turn = start_turn
        self.ramp_turns = ramp_turns
        self.mod_phase = mod_phase
        self.stop_turn = stop_turn
        self.rampdown_turns = rampdown_turns

    def depth_at_turn(self, n):
        """Effective modulation depth at turn n.

        Phases, in order:
          0.0                                   for n < start_turn
          raised-cosine ramp 0 -> mod_depth      over [start_turn, start_turn+ramp_turns]
          held at mod_depth                      over [start_turn+ramp_turns, stop_turn]
          raised-cosine ramp mod_depth -> 0       over [stop_turn, stop_turn+rampdown_turns]
          0.0                                    for n >= stop_turn + rampdown_turns

        If stop_turn is None, the "held" phase simply continues forever
        (original behavior). ramp_turns/rampdown_turns <= 0 recover
        instantaneous switch-on/off at their respective turns.
        """
        if n < self.start_turn:
            return 0.0

        # ramp-up phase
        tau_up = n - self.start_turn
        if self.ramp_turns > 0 and tau_up < self.ramp_turns:
            ramp_frac = tau_up / self.ramp_turns
            smooth = 0.5 * (1.0 - np.cos(np.pi * ramp_frac))
            return self.mod_depth * smooth

        # no ramp-down configured -> hold at full depth indefinitely
        if self.stop_turn is None:
            return self.mod_depth

        # before stop_turn -> holding at full depth
        if n < self.stop_turn:
            return self.mod_depth

        # ramp-down phase
        tau_down = n - self.stop_turn
        if self.rampdown_turns <= 0:
            return 0.0
        if tau_down < self.rampdown_turns:
            ramp_frac = tau_down / self.rampdown_turns
            smooth = 0.5 * (1.0 + np.cos(np.pi * ramp_frac))  # 1 -> 0
            return self.mod_depth * smooth

        # fully ramped down
        return 0.0

    def __call__(self, turn):
        if turn < self.start_turn:
            V = self.Vrf_mean
        else:
            tau = turn - self.start_turn
            depth_n = self.depth_at_turn(turn)
            V = self.Vrf_mean * (1.0 + depth_n *
                                  np.cos(self.omega_mod * tau + self.mod_phase))
        return float(np.clip(V, 1.0e-9, None))   # keep strictly positive/physical