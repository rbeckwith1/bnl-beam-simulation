"""Shared diagnostic outputs: CSV + the standard set of turn-history plots.
Every method calls these pointed at its own results/<method>/ folder."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"Diagnostics saved to: {path}")


def _save_plot(x, y, xlabel, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  saved {fname}")


def _save_bunch_length_plot(x, y, xlabel, ylabel, title, fname):
    """Like _save_plot, but also marks the minimum bunch length with a
    labeled marker and a legend entry giving its value and turn."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, lw=1.2, label="RMS bunch length")

    i_min = y.idxmin() if hasattr(y, "idxmin") else int(min(range(len(y)), key=lambda i: y[i]))
    x_min = x[i_min]
    y_min = y[i_min]
    ax.plot(x_min, y_min, marker="o", ms=7, mfc="red", mec="black", zorder=5,
            linestyle="None",
            label=f"Min: {y_min:.3f} ns @ turn {x_min:g}")
    ax.legend(loc="upper left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  saved {fname}")


def save_standard_plots(df, out_dir):
    """The plot_1..7 set, standardized across methods so results/ folders
    are directly comparable between adiabatic/non_adiabatic/resonant."""
    _save_plot(df.turn, df.Vrf_kV, "Turn", "V_rf [kV]",
               "RF voltage vs. turn", f"{out_dir}/plot_1_Vrf_vs_turn.png")
    _save_bunch_length_plot(df.turn, df.time_sigma_ns, "Turn", "RMS bunch length [ns]",
               "RMS bunch length vs. turn", f"{out_dir}/plot_2_time_sigma_vs_turn.png")
    _save_plot(df.turn, df.dE_sigma_MeV, "Turn", "RMS energy spread [MeV]",
               "RMS energy spread vs. turn", f"{out_dir}/plot_3_dE_sigma_vs_turn.png")
    _save_plot(df.turn, df.eps_rms, "Turn", "RMS emittance-like quantity [ns*GeV]",
               "RMS longitudinal emittance vs. turn", f"{out_dir}/plot_4_eps_rms_vs_turn.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df.turn, df.Q1, label="Q1")
    ax.plot(df.turn, df.Q2, label="Q2")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Quadrupole moments (normalized)")
    ax.set_title("Q1, Q2 vs. turn")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot_5_Q1_Q2_vs_turn.png", dpi=140)
    plt.close(fig)

    _save_plot(df.turn, df.Q_amp, "Turn", "Q_amp",
               "Quadrupole amplitude vs. turn", f"{out_dir}/plot_6_Qamp_vs_turn.png")
    _save_plot(df.turn, df.theta_Q, "Turn", "theta_Q [rad]",
               "Quadrupole orientation vs. turn", f"{out_dir}/plot_7_thetaQ_vs_turn.png")
