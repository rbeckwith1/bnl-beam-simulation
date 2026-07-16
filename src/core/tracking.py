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

from core.kinematics import ReferenceParticle
from core.rf import rf_kick


def track_bunch(time0, dE0, voltage_program, n_turns, a_t, a_E,
                 acceleration_program=None,
                 log_every=1, snapshot_every=None, max_frames=None,
                 stop_after_best_compression=True, rows_past_best_to_stop=20):
    """
    Run the tracking loop.

    Parameters
    ----------
    time0, dE0 : initial particle coordinates [ns], [GeV] (from bunch_init)
    voltage_program : callable(turn:int) -> Vrf [GeV]
    acceleration_program : callable(turn:int, Vrf:float) -> (dK0_turn, phi_s,
        phi_ref), e.g. core.acceleration.AccelerationProgram. Pass None (the
        default) for a stationary bucket -- phi_ref stays pi and the
        reference K0 never moves, i.e. the original non-accelerating
        behavior, byte-for-byte. Passing an AccelerationProgram with
        enabled=False has the identical effect, since it returns
        (0.0, 0.0, pi) for every turn.
    a_t, a_E : matched-ellipse amplitudes, used to normalize quadrupole moments
    snapshot_every : turns between animation snapshots; None disables snapshotting

    Returns
    -------
    df : pandas.DataFrame of per-turn diagnostics
    snapshots : dict with 'turns', 'times', 'dEs', 'Vrf', 'K0', 'phi_s' lists
        (empty if disabled)
    """
    ref = ReferenceParticle()   # starts at core.constants.K0; only moves if
                                 # acceleration_program actually returns dK0 != 0

    time = time0.copy()
    dE = dE0.copy()
    time_init_for_color = time0.copy()

    rows = []
    snapshots = {"turns": [], "times": [], "dEs": [], "Vrf": [], "K0": [], "phi_s": []}

    best_time_sigma = np.inf
    best_turn = -1
    rows_since_best = 0
    stop_turn = None

    for n in range(n_turns):
        Vrf_n = voltage_program(n)

        # --- drift (uses the reference's CURRENT T0; fixed unless
        #     acceleration has moved K0) ---
        time = ref.wrap_to_bucket(ref.drift_map(time, dE))

        # --- this turn's synchronous phase / phi_ref / reference energy gain ---
        if acceleration_program is not None:
            dK0_turn, phi_s, phi_ref = acceleration_program(n, Vrf_n)
        else:
            dK0_turn, phi_s, phi_ref = 0.0, 0.0, np.pi

        # --- kick (phi_ref=pi, dK0=0 reduces exactly to the old dE+Vrf*sin(phi)) ---
        dE = rf_kick(time, dE, Vrf_n, phi_ref=phi_ref, T0_ns=ref.T0_ns)

        # --- advance the reference particle itself (no-op if dK0_turn==0) ---
        ref.accelerate(dK0_turn)

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
                "K0_GeV": ref.K0, "phi_s_deg": np.rad2deg(phi_s),
                "time_mean_ns": np.mean(time), "time_sigma_ns": time_sigma,
                "time_min_ns": np.min(time), "time_max_ns": np.max(time),
                "dE_mean_MeV": np.mean(dE) * 1e3, "dE_sigma_MeV": dE_sigma_GeV * 1e3,
                "dE_min_MeV": np.min(dE) * 1e3, "dE_max_MeV": np.max(dE) * 1e3,
                "t2": t2, "dE2": dE2, "t_dE": t_dE, "eps_rms": eps_rms,
                "Q1": Q1, "Q2": Q2, "Q_amp": Q_amp, "theta_Q": theta_Q,
            })

            if stop_after_best_compression:
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
            snapshots["K0"].append(ref.K0)
            snapshots["phi_s"].append(phi_s)
            if max_frames and len(snapshots["turns"]) >= max_frames:
                pass  # caller's snapshot_every should already respect max_frames

    df = pd.DataFrame(rows)
    if stop_turn is not None:
        print(f"Stopped early at turn {stop_turn}: best compression at turn "
              f"{best_turn} (time_sigma={best_time_sigma:.4f} ns).")
    return df, snapshots, time_init_for_color
