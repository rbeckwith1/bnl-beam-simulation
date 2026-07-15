"""
Adiabatic (slow) RF voltage ramp: hold at Vrf_low, then a quadratic
(r**2) ease-in ramp up to Vrf_high, ramp shape ported from the original
rf_bucket_motion.py `voltage_ramp` function.

NOTE(Rosalyn): the original script actually started the ramp at Vrf=0 kV
with a raw uniform-box initial distribution (not the matched-ellipse init
used elsewhere in this codebase). A Vrf=0 reference point can't be used to
linearize/match a bucket (b_coefficient -> 0), so run_adiabatic.py instead
starts the ramp from Vrf_low (matching non_adiabatic's low value) so the
matched-ellipse initialization and synchrotron-frequency cross-check have a
well-defined bucket at turn 0. Flag if AGS actually runs the real ramp from
literal 0 kV -- that would need a separate (non-matched-ellipse) init path.
"""


class AdiabaticProgram:
    def __init__(self, Vrf_low, Vrf_high, ramp_start_turn, ramp_turns):
        self.Vrf_low = Vrf_low
        self.Vrf_high = Vrf_high
        self.ramp_start_turn = ramp_start_turn
        self.ramp_turns = ramp_turns

    def __call__(self, turn):
        if turn < self.ramp_start_turn:
            return self.Vrf_low
        r = (turn - self.ramp_start_turn) / self.ramp_turns
        r = max(0.0, min(1.0, r))
        ramp_shape = r**2   # quadratic ease-in, from original voltage_ramp()
        return self.Vrf_low + (self.Vrf_high - self.Vrf_low) * ramp_shape