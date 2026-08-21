import numpy as np
import pandas as pd
from scipy.special import beta as beta_func

pi = np.pi
clight = 299792458.0
e = 1.602176634e-19

v00 = 938.272013e6      # proton rest energy, eV
circ = 807.12           # AGS circumference, m
gammat = 8.5            # AGS transition gamma

Z_MACHINE = 10.0        # estimated AGS |Z/n|, Ohm



def zon_vs_turn(
    gamma_arr,
    sigma_tau_arr,
    sigma_E_arr,
    Nb,
    F=1.0,
    J=1.5,
):
    """
    Calculate the Boussard / Keil-Schnell longitudinal impedance
    threshold turn-by-turn.

    Binomial longitudinal profile:
        I(t) = I_peak * [1 - (t/T)^2]^J,  |t| <= T

    For this family:
        sigma_tau = T / sqrt(2J + 3)

    and normalization to total bunch charge Q gives:

        I_peak = Q / [T * B(1/2, J+1)]

    where B is the beta function.

    For J = 3/2:

        B(1/2, 5/2) = 3*pi/8

    so:

        I_peak = 8Q / (3*pi*T)

    Parameters
    ----------
    gamma_arr : array
        Reference relativistic gamma per turn.
    sigma_tau_arr : array [s]
        RMS bunch length from the simulation.
    sigma_E_arr : array [eV]
        RMS energy spread from the simulation.
    Nb : float
        Number of protons in the bunch.
    F : float
        Boussard/Keil-Schnell form factor associated with the
        assumed longitudinal distribution.
    J : float
        Binomial distribution exponent.

    Returns
    -------
    Zon : array [Ohm]
        Threshold |Z/n| per turn.
    I_peak : array [A]
        Peak current calculated from the binomial distribution.
    T : array [s]
        Half-support corresponding to the selected J.
    rmsdpp : array
        RMS relative momentum spread.
    """

    gamma_arr = np.asarray(gamma_arr, dtype=float)
    sigma_tau_arr = np.asarray(sigma_tau_arr, dtype=float)
    sigma_E_arr = np.asarray(sigma_E_arr, dtype=float)

    # --- machine / reference particle -------------------------------------

    beta = np.sqrt(1.0 - 1.0 / gamma_arr**2)

    eta = 1.0 / gammat**2 - 1.0 / gamma_arr**2

    E = gamma_arr * v00

    # --- bunch-shape geometry ---------------------------------------------

    # For I(t) ~ [1 - (t/T)^2]^J:
    #
    # sigma_tau = T / sqrt(2J + 3)
    #
    # therefore:
    T = np.sqrt(2.0 * J + 3.0) * sigma_tau_arr

    # Total bunch charge
    Q = e * Nb

    # Peak current from normalization of the binomial distribution:
    #
    # Q = integral_{-T}^{T} I(t) dt
    #   = I_peak * T * B(1/2, J+1)
    #
    # therefore:
    # I_peak = Q / [T * B(1/2, J+1)]
    shape_integral = beta_func(0.5, J + 1.0)

    I_peak = Q / (T * shape_integral)

    # For J = 3/2, this is exactly:
    #
    # I_peak = 8Q / (3*pi*T)

    # --- RMS relative momentum spread -------------------------------------

    rmsdpp = sigma_E_arr / (beta**2 * E)

    # --- Boussard / Keil-Schnell threshold --------------------------------

    Zon = (
        2.0
        * pi
        * beta**2
        * E
        * rmsdpp**2
        * np.abs(eta)
        * F
        / I_peak
    )

    return Zon, I_peak, T, rmsdpp

def add_stability_columns(
    df,
    Nb,
    F=1.0,
    J=1.5,
    Z_machine=Z_MACHINE,
):
    """
    Add microwave-instability diagnostics to a track_bunch() dataframe.
    """

    df = df.copy()

    gamma_arr = 1.0 + df["K0_GeV"].to_numpy() * 1e9 / v00

    sigma_tau_s = df["time_sigma_ns"].to_numpy() * 1e-9
    sigma_E_eV = df["dE_sigma_MeV"].to_numpy() * 1e6

    Zon, I_peak, T, rmsdpp = zon_vs_turn(
        gamma_arr,
        sigma_tau_s,
        sigma_E_eV,
        Nb,
        F=F,
        J=J,
    )

    df["gamma"] = gamma_arr

    # Record assumptions
    df["distribution_J"] = J
    df["form_factor_F"] = F

    # Diagnostics
    df["half_bunch_length_ns"] = T * 1e9
    df["peak_current_A"] = I_peak
    df["rms_dpp"] = rmsdpp

    # Stability threshold
    df["Zon_ohm"] = Zon
    df["unstable"] = df["Zon_ohm"] <= Z_machine

    return df


def find_instability_windows(df):
    """
    Group consecutive logged unstable points into instability episodes.

    Returns
    -------
    list of dict
    """

    if not df["unstable"].any():
        return []

    unstable = df["unstable"].to_numpy()
    turns = df["turn"].to_numpy()

    change = np.diff(unstable.astype(int))

    starts = np.where(change == 1)[0] + 1
    if unstable[0]:
        starts = np.insert(starts, 0, 0)

    ends = np.where(change == -1)[0]
    if unstable[-1]:
        ends = np.append(ends, len(unstable) - 1)

    episodes = []

    for s, e in zip(starts, ends):

        window = df.iloc[s:e + 1]

        worst_idx = window["Zon_ohm"].idxmin()
        worst = window.loc[worst_idx]

        episodes.append({
            "start_turn": int(turns[s]),
            "end_turn": int(turns[e]),
            "n_logged_points": int(e - s + 1),

            "min_Zon_ohm": float(worst["Zon_ohm"]),
            "turn_at_min_Zon": int(worst["turn"]),

            "time_sigma_at_min": float(worst["time_sigma_ns"]),
            "dE_sigma_at_min": float(worst["dE_sigma_MeV"]),

            "peak_current_at_min_A": float(worst["peak_current_A"]),
            "rms_dpp_at_min": float(worst["rms_dpp"]),
        })

    return episodes


def report_instabilities(df, verbose=True):
    """
    Print a summary of instability episodes.
    """

    episodes = find_instability_windows(df)

    if not episodes:

        if verbose:
            print(
                f"No instability windows found "
                f"(Z/n stayed above {Z_MACHINE:.1f} Ohm "
                f"for all logged turns)."
            )

        return episodes

    if verbose:

        print(f"Found {len(episodes)} instability window(s):")

        for i, ep in enumerate(episodes, 1):

            print(
                f"  [{i}] turns "
                f"{ep['start_turn']}-{ep['end_turn']} | "
                f"worst Z/n = {ep['min_Zon_ohm']:.3f} Ohm "
                f"at turn {ep['turn_at_min_Zon']} | "
                f"sigma_tau = {ep['time_sigma_at_min']:.3f} ns | "
                f"sigma_dE = {ep['dE_sigma_at_min']:.3f} MeV | "
                f"I_peak = {ep['peak_current_at_min_A']:.3f} A"
            )

    return episodes