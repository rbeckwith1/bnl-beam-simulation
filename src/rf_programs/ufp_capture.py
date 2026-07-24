"""
UFP release-and-capture bunching.

"""
import numpy as np


def _raised_cosine(frac):
    frac = np.clip(frac, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * frac))


class UFPReleaseCaptureProgram:
    def __init__(self, phi_s_hold_deg, phase_jump_deg,
                 jump_out_start_turn, jump_turns, dwell_turns):
        self.phi_s_hold_rad = np.deg2rad(phi_s_hold_deg)
        self.phi_ref_hold = np.pi - self.phi_s_hold_rad
        self.phase_jump_rad = np.deg2rad(phase_jump_deg)

        self.jump_turns = jump_turns
        self.jump_out_start = jump_out_start_turn
        self.jump_out_end = jump_out_start_turn + jump_turns

        self.dwell_turns = dwell_turns
        self.jump_back_start = self.jump_out_end + dwell_turns
        self.jump_back_end = self.jump_back_start + jump_turns

    def phi_ref(self, turn):
        """'phi_ref as a function of turn alone."""
        if turn < self.jump_out_start:
            return self.phi_ref_hold

        if turn < self.jump_out_end:
            if self.jump_turns <= 0:
                return self.phi_ref_hold + self.phase_jump_rad
            frac = (turn - self.jump_out_start) / self.jump_turns
            return self.phi_ref_hold + self.phase_jump_rad * _raised_cosine(frac)

        if turn < self.jump_back_start:
            return self.phi_ref_hold + self.phase_jump_rad

        if turn < self.jump_back_end:
            if self.jump_turns <= 0:
                return self.phi_ref_hold
            frac = (turn - self.jump_back_start) / self.jump_turns
            return self.phi_ref_hold + self.phase_jump_rad * (1.0 - _raised_cosine(frac))

        return self.phi_ref_hold

    def __call__(self, turn, Vrf_n=None):
        phi_ref = self.phi_ref(turn)
        phi_s = np.pi - phi_ref
        dK0 = 0.0
        return dK0, phi_s, phi_ref