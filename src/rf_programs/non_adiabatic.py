"""
Non-adiabatic (fast) RF voltage jump: hold at Vrf_low, then a constant-slew
(linear in V) ramp up to Vrf_high over jump_turns.

SHAPE CHANGE: previously a smoothstep (3r**2 - 2r**3). Now linear (r),
matching AdiabaticProgram so that ramp DURATION is the only variable
distinguishing the two methods. Smoothstep sits above the old quadratic
adiabatic shape everywhere on (0,1) -- by up to 93 kV at r = 2/3 -- so the
two programs were never comparable even at identical endpoints and duration.

This is the method where the (pending) AGS dV/dt limit is most likely to
bind: peak slew = (Vrf_high - Vrf_low) / jump_turns, and jump_turns is small
by construction. Check `self.slew` against the machine limit before quoting
any result from this program.
"""


class NonAdiabaticProgram:
    def __init__(self, Vrf_low, Vrf_high, jump_start_turn, jump_turns):
        self.Vrf_low = Vrf_low
        self.Vrf_high = Vrf_high
        self.jump_start_turn = jump_start_turn
        self.jump_turns = jump_turns
        # constant for a linear ramp; same voltage units as Vrf_low, per turn
        self.slew = (Vrf_high - Vrf_low) / jump_turns

    def __call__(self, turn):
        if turn < self.jump_start_turn:
            return self.Vrf_low
        r = (turn - self.jump_start_turn) / self.jump_turns
        r = max(0.0, min(1.0, r))
        ramp_shape = r   # linear / constant slew rate
        return self.Vrf_low + (self.Vrf_high - self.Vrf_low) * ramp_shape