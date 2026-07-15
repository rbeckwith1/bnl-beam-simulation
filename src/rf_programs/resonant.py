"""
Resonant (parametric) bunching: RF voltage modulation at 2*omega_s, with a
raised-cosine ramp-up of the modulation depth.

TODO(Rosalyn): paste your actual modulation formula + ramp-up profile here.
This stub keeps the same shape (base voltage +/- modulation, ramped in over
`ramp_turns`) so run_resonant.py is runnable while you migrate the real one.
"""

import numpy as np


class ResonantProgram:
    def __init__(self, Vrf_base, mod_depth, omega_s, ramp_turns):
        self.Vrf_base = Vrf_base
        self.mod_depth = mod_depth
        self.omega_mod = 2.0 * omega_s
        self.ramp_turns = ramp_turns

    def __call__(self, turn):
        ramp = min(1.0, turn / self.ramp_turns) if self.ramp_turns > 0 else 1.0
        raised_cosine_ramp = 0.5 * (1 - np.cos(np.pi * ramp))   # TODO: replace with your exact profile
        return self.Vrf_base * (1.0 + raised_cosine_ramp * self.mod_depth * np.cos(self.omega_mod * turn))
