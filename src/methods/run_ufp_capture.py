"""
UFP release-and-capture bunching run.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import kinematics
from core.rf import compute_a_coefficient, compute_b_coefficient, check_fixed_point_stability
from core.synchrotron import get_omega_s
from core.separatrix import Separatrix
from core.tracking import track_bunch
from core.diagnostics import save_csv, save_standard_plots
from core.animation import render_animation
from rf_programs.ufp_capture import UFPReleaseCaptureProgram
from core.bunch_init import matched_ellipse_amplitudes, initial_bunch
from core.cartoon_plots import render_storyboard
from core.stability import add_stability_columns, report_instabilities 

RNG_SEED = 12345
np.random.seed(RNG_SEED)

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SRC_DIR, "results", "ufp_capture")
os.makedirs(OUT_DIR, exist_ok=True)

# --- method-specific configuration ---
V_hold = 320e3 / 1e9
phi_s_hold_deg = 0.0
phi_ref_hold = np.pi - np.deg2rad(phi_s_hold_deg)

jump_out_start_turn = 200   # let the matched bunch sit quietly first, then release
jump_turns = 5
dwell_turns = 900
phase_jump_deg = 180.0

N = 10000
n_turns = jump_out_start_turn + jump_turns + dwell_turns + jump_turns + 2000  # margin after capture

kinematics.print_summary()
a_coef = compute_a_coefficient()
b_coef = compute_b_coefficient(V_hold)   # valid: computed at phi_ref=pi, before any jump
check_fixed_point_stability(a_coef, b_coef)

omega_s, T_s_turns, a_coef, b_coef = get_omega_s(V_hold)
eps_l_ns_GeV = 1.35


a_t, a_E = matched_ellipse_amplitudes(eps_l_ns_GeV, a_coef, b_coef)
print(f"Diagnostic normalization amplitudes: a_t = {a_t:.3f} ns, a_E = {a_E*1e3:.4f} MeV")

separatrix = Separatrix(Vrf_max_expected=V_hold)
t_ufp_ns = separatrix.unstable_fixed_point_t_ns(phi_ref_hold)
print(f"Old UFP (pre-jump) at t = {t_ufp_ns:.4f} ns -- this is where the new SFP "
      f"should land after the jump")
# sanity check: the new UFP after the jump should land back near t=0
t_ufp_after_jump = separatrix.unstable_fixed_point_t_ns(
    phi_ref_hold + np.deg2rad(phase_jump_deg))
print(f"New UFP (post-jump) at t = {t_ufp_after_jump:.4f} ns (should be ~0)")

# --- real matched bunch at the SFP, no jitter hacks needed ---
time0, dE0 = initial_bunch(N, a_t, a_E)   # centered at (0, 0) = the SFP

phase_program = UFPReleaseCaptureProgram(
    phi_s_hold_deg=phi_s_hold_deg,
    phase_jump_deg=phase_jump_deg,
    jump_out_start_turn=jump_out_start_turn,
    jump_turns=jump_turns,
    dwell_turns=dwell_turns,
)

def voltage_program(turn):
    return V_hold

df, snapshots, time_init_for_color = track_bunch(
    time0, dE0, voltage_program, n_turns, a_t, a_E,
    acceleration_program=phase_program,
    snapshot_every=10, max_frames=400,
    stop_after_best_compression=False,
)

df = add_stability_columns(df, Nb=1.5e12)
episodes = report_instabilities(df)

print(df['unstable'].sum())  
print(episodes)

# check specifically at your compression minimum
best_row = df.loc[df["time_sigma_ns"].idxmin()]
print(best_row[["turn", "time_sigma_ns", "dE_sigma_MeV", "Zon_ohm", "unstable"]])

df.plot(x="turn", y="Zon_ohm", logy=True)


save_csv(df, f"{OUT_DIR}/diagnostics.csv")

ENABLE_PLOTS = True
ENABLE_ANIMATION = False
ENABLE_CARTOON = True

if ENABLE_PLOTS:
    save_standard_plots(df, OUT_DIR)

if ENABLE_ANIMATION:
    render_animation(
    snapshots, time_init_for_color, a_t, a_E, separatrix, T_s_turns,
    f"{OUT_DIR}/animation.mp4",
    extra_info=f"UFP dwell={dwell_turns} turns, phase jump={phase_jump_deg} deg "
               f"over {jump_turns} turns",
    center_on_bunch=True,
)

print(f"Minimum RMS bunch length: {df.time_sigma_ns.min():.3f} ns "
      f"at turn {df.loc[df.time_sigma_ns.idxmin(), 'turn']:.0f}")

if ENABLE_CARTOON: 
    snapshot_every = 10  # must match what you passed to track_bunch above
 
    # Phase boundaries, in turns:
    release_end_turn = jump_out_start_turn + jump_turns
    capture_start_turn = release_end_turn + dwell_turns
    capture_end_turn = capture_start_turn + jump_turns
     
    # Hand-picked turns that tell the release -> dwell -> capture story, since
    # the two jumps are only 5 turns wide and evenly-spaced panels would very
    # likely miss them in a ~2900-turn run:
    panel_turns = [
        0,                                   # matched bunch sitting at the SFP
        jump_out_start_turn,                 # just before release
        release_end_turn + 10,               # just after release, drifting toward old UFP
        release_end_turn + dwell_turns // 2, # mid-dwell
        capture_start_turn,                  # just before the capture jump
        min(capture_end_turn + 50, n_turns - 10),  # just after capture, settling at new SFP
    ]
    panel_indices = sorted(set(
        min(t // snapshot_every, len(snapshots["turns"]) - 1) for t in panel_turns
    ))
     
    render_storyboard(
        snapshots, time_init_for_color, a_t, a_E, separatrix, T_s_turns,
        f"{OUT_DIR}/storyboard.png",
        panel_indices=panel_indices,
        ncols=len(panel_indices),         # single inline row
        center_on_bunch=False,            # set True if you want a fixed zoomed window instead
        suptitle="Unstable fixed point release/capture",
        extra_info=(f"release: start={jump_out_start_turn}, dur={jump_turns} turns | "
                    f"dwell={dwell_turns} turns | "
                    f"capture jump: {phase_jump_deg} deg over {jump_turns} turns"),
    )
    # vector version for print quality on the poster:
    render_storyboard(
        snapshots, time_init_for_color, a_t, a_E, separatrix, T_s_turns,
        f"{OUT_DIR}/storyboard.pdf",
        panel_indices=panel_indices, ncols=len(panel_indices),
        suptitle="Unstable fixed point release/capture",    )
    
    