import numpy as np
import pandas as pd

pi = np.pi
clight = 299792458.0
e = 1.602176634e-19
v00 = 938.272013e6      # proton rest mass, eV
circ = 807.12           # AGS circumference, m
gammat = 8.5            # AGS transition gamma

def zon_vs_turn(gamma_arr, sigma_tau_arr, sigma_E_arr, Nb, F=1.0):
    """
    gamma_arr, sigma_tau_arr [s], sigma_E_arr [eV] : per-turn arrays from  sim
    Nb : number of protons in bunch
    Returns Z/n [Ohms] per turn (Boussard/Keil-Schnell threshold)
    """
    gamma_arr = np.asarray(gamma_arr)
    sigma_tau_arr = np.asarray(sigma_tau_arr)
    sigma_E_arr = np.asarray(sigma_E_arr)

    beta = np.sqrt(1. - 1./gamma_arr**2)
    eta = 1./gammat**2 - 1./gamma_arr**2
    E = gamma_arr * v00                      # total energy, eV
    vrev = clight * beta
    trev = circ / vrev
    frev = 1. / trev

    bc = e * Nb                              # bunch charge, C
    wrev = 2.*pi*vrev/circ
    sigo = wrev * sigma_tau_arr              # rms bunch length in orbit angle
    curr = bc * frev * 2*pi / (np.sqrt(2*pi) * sigo)   # peak current

    rmsdpp = sigma_E_arr / (gamma_arr * v00 * beta**2)
    fwhm = 2. * np.sqrt(2. * np.log(2.))
    dppF = fwhm * rmsdpp                      # <-- add this

    Zon = 2.*pi*beta**2*E*dppF**2*np.abs(eta)*F/curr   # use dppF, not rmsdpp
    return Zon



def add_stability_columns(df, Nb, F=1.0):
    """
    Post-process a track_bunch() df to add microwave-instability
    diagnostics (Keil-Schnell/Boussard) per logged turn.
    """
    df = df.copy()

    # K0_GeV is kinetic energy -> gamma  (confirm this against
    # core.kinematics.ReferenceParticle if unsure)
    gamma_arr = 1.0 + df["K0_GeV"].to_numpy() * 1e9 / v00

    sigma_tau_s = df["time_sigma_ns"].to_numpy() * 1e-9
    sigma_E_eV = df["dE_sigma_MeV"].to_numpy() * 1e6

    Zon = zon_vs_turn(gamma_arr, sigma_tau_s, sigma_E_eV, Nb, F=F)

    df["gamma"] = gamma_arr
    df["Zon_ohm"] = Zon
    df["unstable"] = df["Zon_ohm"] <= 10.0
    return df

def find_instability_windows(df):
    """
    Group consecutive 'unstable' turns into episodes and summarize each.

    Returns
    -------
    list of dict, one per contiguous unstable episode, with:
        start_turn, end_turn, n_turns, min_Zon_ohm, turn_at_min_Zon,
        time_sigma_at_min, dE_sigma_at_min
    """
    if not df["unstable"].any():
        return []

    unstable = df["unstable"].to_numpy()
    turns = df["turn"].to_numpy()

    # find boundaries where unstable flips True/False
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
        worst = window.loc[window["Zon_ohm"].idxmin()]
        episodes.append({
            "start_turn": int(turns[s]),
            "end_turn": int(turns[e]),
            "n_turns": int(e - s + 1),
            "min_Zon_ohm": float(worst["Zon_ohm"]),
            "turn_at_min_Zon": int(worst["turn"]),
            "time_sigma_at_min": float(worst["time_sigma_ns"]),
            "dE_sigma_at_min": float(worst["dE_sigma_MeV"]),
        })
    return episodes


def report_instabilities(df, verbose=True):
    """
    Print a summary of instability episodes found in df (from
    add_stability_columns). Returns the episode list either way.
    """
    episodes = find_instability_windows(df)

    if not episodes:
        if verbose:
            print("No instability windows found (Z/n stayed above 10 Ohm "
                  "for all logged turns).")
        return episodes

    if verbose:
        print(f"Found {len(episodes)} instability window(s):")
        for i, ep in enumerate(episodes, 1):
            print(f"  [{i}] turns {ep['start_turn']}-{ep['end_turn']} "
                  f"({ep['n_turns']} turns) | "
                  f"worst Z/n = {ep['min_Zon_ohm']:.3f} Ohm at turn "
                  f"{ep['turn_at_min_Zon']} "
                  f"(sigma_tau={ep['time_sigma_at_min']:.3f} ns, "
                  f"sigma_dE={ep['dE_sigma_at_min']:.3f} MeV)")
    return episodes