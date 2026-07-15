"""
The one turn-by-turn tracking loop, shared by adiabatic / non_adiabatic /
resonant. This is the piece that used to get copy-pasted and silently
diverge between method scripts -- now there is exactly one copy.

Each method supplies its own `voltage_program(turn) -> Vrf [GeV]` callable;
everything else (drift, kick, diagnostics, quadrupole moments, early-stop
logic, snapshotting) is identical across methods.
"""

import numpy as np
import pandas as pd

from core.kinematics import T0, T0_ns, T_rf_ns, wrap_to_bucket
from core.constants import h


def track_bunch(time0, dE0, voltage_program, n_turns, a_t, a_E,
                 log_every=1, snapshot_every=None, max_frames=None,
                 stop_after_best_compression=True, rows_past_best_to_stop=20,
                 min_turn_for_stop=0):
    """
    Run the tracking loop.

    Parameters
    ----------
    time0, dE0 : initial particle coordinates [ns], [GeV] (from bunch_init)
    voltage_program : callable(turn:int) -> Vrf [GeV]
    a_t, a_E : matched-ellipse amplitudes, used to normalize quadrupole moments
    snapshot_every : turns between animation snapshots; None disables snapshotting

    Returns
    -------
    df : pandas.DataFrame of per-turn diagnostics
    snapshots : dict with 'turns', 'times', 'dEs', 'Vrf' lists (empty if disabled)
    """
    time = time0.copy()
    dE = dE0.copy()
    time_init_for_color = time0.copy()

    rows = []
    snapshots = {"turns": [], "times": [], "dEs": [], "Vrf": []}

    best_time_sigma = np.inf
    best_turn = -1
    rows_since_best = 0
    stop_turn = None

    for n in range(n_turns):
        Vrf_n = voltage_program(n)

        # --- drift ---
        from core.kinematics import revolution_time
        T = revolution_time(dE)
        dt_ns = (T - T0) * 1e9
        time = wrap_to_bucket(time + dt_ns)

        # --- kick ---
        phi = 2.0 * np.pi * h * time / T0_ns + np.pi
        dE = dE + Vrf_n * np.sin(phi)

        if n % log_every == 0:
            t2 = np.mean(time**2)
            dE2 = np.mean(dE**2)
            t_dE = np.mean(time * dE)
            time_sigma = np.sqrt(np.mean((time - np.mean(time))**2))
            dE_sigma_GeV = np.sqrt(np.mean((dE - np.mean(dE))**2))
            eps_rms = np.sqrt(max(t2 * dE2 - t_dE**2, 0.0))

            Q1 = np.mean((time / a_t)**2 - (dE / a_E)**2)
            Q2 = 2.0 * np.mean((time / a_t) * (dE / a_E))
            Q_amp = np.sqrt(Q1**2 + Q2**2)
            theta_Q = 0.5 * np.arctan2(Q2, Q1)

            rows.append({
                "turn": n, "Vrf_kV": Vrf_n * 1e9 / 1e3,
                "time_mean_ns": np.mean(time), "time_sigma_ns": time_sigma,
                "time_min_ns": np.min(time), "time_max_ns": np.max(time),
                "dE_mean_MeV": np.mean(dE) * 1e3, "dE_sigma_MeV": dE_sigma_GeV * 1e3,
                "dE_min_MeV": np.min(dE) * 1e3, "dE_max_MeV": np.max(dE) * 1e3,
                "t2": t2, "dE2": dE2, "t_dE": t_dE, "eps_rms": eps_rms,
                "Q1": Q1, "Q2": Q2, "Q_amp": Q_amp, "theta_Q": theta_Q,
            })

            if stop_after_best_compression and n > min_turn_for_stop:
                if time_sigma < best_time_sigma:
                    best_time_sigma, best_turn, rows_since_best = time_sigma, n, 0
                else:
                    rows_since_best += 1
                if rows_since_best > rows_past_best_to_stop:
                    stop_turn = n
                    break

        if snapshot_every and (n % snapshot_every == 0):
            snapshots["turns"].append(n)
            snapshots["times"].append(time.copy())
            snapshots["dEs"].append(dE.copy())
            snapshots["Vrf"].append(Vrf_n)
            if max_frames and len(snapshots["turns"]) >= max_frames:
                pass  # caller's snapshot_every should already respect max_frames

    df = pd.DataFrame(rows)
    if stop_turn is not None:
        print(f"Stopped early at turn {stop_turn}: best compression at turn "
              f"{best_turn} (time_sigma={best_time_sigma:.4f} ns).")
    return df, snapshots, time_init_for_color
