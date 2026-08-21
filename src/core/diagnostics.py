"""Shared diagnostic outputs: CSV + the standard set of turn-history plots.
Every method calls these pointed at its own results/<method>/ folder."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 9,
})

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
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def _save_bunch_length_plot(x, y, xlabel, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, lw=1.2, label="RMS bunch length")

    i_min = y.idxmin() if hasattr(y, "idxmin") else int(min(range(len(y)), key=lambda i: y[i]))
    x_min = x[i_min]
    y_min = y[i_min]

    ax.plot(
        x_min, y_min,
        marker="o",
        ms=6,
        mfc="red",
        mec="black",
        linestyle="None",
        zorder=5,
        label=f"Min: {y_min:.3f} ns @ turn {x_min:g}"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
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
    
def save_initial_distribution(time, dE, out_dir, Nb, J=1.5):
    # ============================================================
    # 2D longitudinal phase-space distribution
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        time,
        dE * 1e3,   # GeV -> MeV
        s=2,
        alpha=0.35,
        rasterized=True,
    )

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(r"$\Delta E$ [MeV]")
    ax.set_title("Initial longitudinal phase-space distribution")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        f"{out_dir}/plot_0_initial_distribution.png",
        dpi=140,
    )
    plt.close(fig)



    # ============================================================
    # 1D current profile
    # ============================================================
    n_bins = 50
    
    counts, edges = np.histogram(time, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    # Bin width: ns -> seconds
    dt = (edges[1] - edges[0]) * 1e-9
    
    # Each macroparticle represents Nb / N real protons
    particles_per_macro = Nb / len(time)
    
    # Simulated current [A]
    current = counts * particles_per_macro * e / dt
    
    
    # ============================================================
    # Analytical J = 3/2 profile
    # ============================================================
    J = 3 / 2
    
    # Half-width of distribution
    T = np.max(np.abs(time))
    
    # Use a smooth time grid for analytical curve
    t_analytic = np.linspace(-T, T, 500)
    
    shape = np.clip(
        1.0 - (t_analytic / T)**2,
        0.0,
        None
    )**J
    
    # Normalize so integral I(t) dt = Nb * e
    integral_shape = np.trapezoid(shape, t_analytic * 1e-9)
    I0 = Nb * e / integral_shape
    
    current_analytic = I0 * shape
    
    
    # ============================================================
    # Plot
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.step(
        centers,
        current,
        where="mid",
        lw=1.5,
        label="Simulated bunch",
    )
    
    ax.plot(
        t_analytic,
        current_analytic,
        "--",
        lw=2,
        label=r"Analytical $J=3/2$",
    )
    
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Current [A]")
    ax.set_title("Initial bunch current profile")
    ax.grid(alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    
    fig.savefig(
        f"{out_dir}/plot_0_initial_current.png",
        dpi=140,
    )
    
    plt.close(fig)
