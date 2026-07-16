"""
Reference-particle acceleration program.

Ramps the synchronous phase phi_s from 0 -> phi_s_final via a raised-cosine
(smoothstep) profile, starting at start_turn over ramp_turns. This is what
turns a stationary bucket (phi_ref = pi, no net energy gain per turn) into
an accelerating one:

    phi_ref      = pi - phi_s
    dK0/turn     = Vrf * sin(phi_s)

TOGGLE: set enabled=False (or just don't pass an AccelerationProgram to
track_bunch at all -- it defaults to None) to get phi_s=0, dK0=0 every
turn, i.e. the exact original stationary-bucket behavior. There is no
separate code path for "off" -- it's the same formulas evaluated at
phi_s=0, so there's nothing that can silently drift out of sync between
the accelerating and non-accelerating cases.
"""

import numpy as np


def smoothstep(r):
    r = np.clip(r, 0.0, 1.0)
    return 3.0 * r**2 - 2.0 * r**3


class AccelerationProgram:
    def __init__(self, phi_s_final_deg, start_turn, ramp_turns, enabled=True):
        self.phi_s_final = np.deg2rad(phi_s_final_deg)
        self.start_turn = start_turn
        self.ramp_turns = max(ramp_turns, 1)
        self.enabled = enabled

    def __call__(self, turn, Vrf):
        """Returns (dK0_turn [GeV], phi_s [rad], phi_ref [rad]) for this turn."""
        if not self.enabled or turn < self.start_turn:
            phi_s = 0.0
        else:
            r = (turn - self.start_turn) / self.ramp_turns
            phi_s = self.phi_s_final * smoothstep(r)

        dK0_turn = Vrf * np.sin(phi_s)
        phi_ref = np.pi - phi_s
        return dK0_turn, phi_s, phi_ref