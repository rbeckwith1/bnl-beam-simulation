"""Non-adiabatic RF voltage program: hold low, then a fast smoothstep jump."""


class NonAdiabaticProgram:
    def __init__(self, Vrf_low, Vrf_high, jump_start_turn, jump_turns):
        self.Vrf_low = Vrf_low
        self.Vrf_high = Vrf_high
        self.jump_start_turn = jump_start_turn
        self.jump_turns = jump_turns

    def __call__(self, turn):
        if turn < self.jump_start_turn:
            return self.Vrf_low
        r = (turn - self.jump_start_turn) / self.jump_turns
        r = max(0.0, min(1.0, r))
        ramp_shape = 3.0 * r**2 - 2.0 * r**3   # smoothstep, zero slope at both ends
        return self.Vrf_low + (self.Vrf_high - self.Vrf_low) * ramp_shape
