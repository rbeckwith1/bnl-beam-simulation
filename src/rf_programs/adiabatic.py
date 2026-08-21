"""
Adiabatic (slow) RF voltage ramp: hold at Vrf_low, then a constant-slew
(linear in V) ramp up to Vrf_high.

NOTE(Rosalyn): the original script actually started the ramp at Vrf=0 kV
with a raw uniform-box initial distribution (not the matched-ellipse init
used elsewhere in this codebase). A Vrf=0 reference point can't be used to
linearize/match a bucket (b_coefficient -> 0), so run_adiabatic.py instead
starts the ramp from Vrf_low (matching non_adiabatic's low value) so the
matched-ellipse initialization and synchrotron-frequency cross-check have a
well-defined bucket at turn 0. Flag if AGS actually runs the real ramp from
literal 0 kV -- that would need a separate (non-matched-ellipse) init path.

Linear is the standard shape here because its slew rate dV/dn is constant,
so feasibility against the (pending) AGS dV/dt limit is a single number
rather than a curve with a hidden interior peak. Under a binding slew-rate
limit a linear ramp is also time-optimal, since it runs at the limit
throughout. Its cost is the largest peak adiabaticity violation of the
shapes tested (alpha is maximal at r=0, where s'(0) != 0 puts a slew-rate
step into the weakest bucket of the ramp).
"""


class AdiabaticProgram:
    def __init__(self, Vrf_low, Vrf_high, ramp_start_turn, ramp_turns):
        self.Vrf_low = Vrf_low
        self.Vrf_high = Vrf_high
        self.ramp_start_turn = ramp_start_turn
        self.ramp_turns = ramp_turns
        # constant for a linear ramp; same voltage units as Vrf_low, per turn
        self.slew = (Vrf_high - Vrf_low) / ramp_turns

    def __call__(self, turn):
        if turn < self.ramp_start_turn:
            return self.Vrf_low
        r = (turn - self.ramp_start_turn) / self.ramp_turns
        r = max(0.0, min(1.0, r))
        ramp_shape = r   # linear / constant slew rate
        return self.Vrf_low + (self.Vrf_high - self.Vrf_low) * ramp_shape