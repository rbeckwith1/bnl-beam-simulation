import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
from scipy.optimize import brentq

# Initial Longitudinal Phase Space Distribution
log_rows = []

N = 10000                  # Number of particles

n_turns = 10000
k = 0.0005

# Synchronous particle
K0 = 24          # GeV -AGS parameter
mp = 0.938272      # GeV -rest mass of proton
c = 299792458      # m/s
L0 = 807.1         # m -circumference of AGS

# Reference fractional momentum deviation
gamma_t = 8.45
alpha_p = 1 / gamma_t**2

E0_total = K0 + mp
gamma0 = E0_total / mp
beta0 = np.sqrt(1 - 1/gamma0**2)
p0 = np.sqrt(E0_total**2 - mp**2)
T0 = L0 / (beta0 * c)

# -----------------------------
# RF / bucket setup
h = 6
Vrf_max = 320e3 / 1e9  # GeV

T_rf_ns = (T0 / h) * 1e9

# -----------------------------
# Initial filled oval distribution
bucket_fraction = 1/3
time_width = bucket_fraction * T_rf_ns

a_t = time_width / 2     # horizontal semi-axis [ns]
a_E = 0.015            # vertical semi-axis [GeV] 

theta = 2 * np.pi * np.random.rand(N)
r = np.sqrt(np.random.rand(N))   # sqrt gives uniform filling of ellipse

time = a_t * r * np.cos(theta)
dE = a_E * r * np.sin(theta)

# Save initial coordinates (used for coloring particles)
dE_initial = dE.copy()
initial_time = time.copy()

# Plots
fig, ax = plt.subplots(figsize=(6,5))

sc = ax.scatter(time, dE * 1000, c=initial_time, cmap="coolwarm", s=3, alpha=0.6)

ax.set_xlim(-T_rf_ns/2, T_rf_ns/2)
ax.set_ylim(-275, 275)
ax.set_xlabel("Arrival Time Deviation (ns)")
ax.set_ylabel("Energy Deviation (MeV)")
ax.grid(True)

title = ax.set_title("Turn 0")

current_turn = 0

sep_line_plus, = ax.plot([], [], "k-", linewidth=2, label="Separatrix")
sep_line_minus, = ax.plot([], [], "k-", linewidth=2)
ax.legend()


def wrap_to_bucket(time_ns, T_rf_ns):
    """
    Wrap time into one RF bucket centered at 0:
    [-T_rf/2, +T_rf/2)
    """
    return ((time_ns + T_rf_ns/2) % T_rf_ns) - T_rf_ns/2

def F_(dE):
    """
    Exact drift Hamiltonian term F(dE), where dF/ddE = (T(dE)-T0)*1e9.
    dE is in GeV.
    F has units ns*GeV/turn.
    """

    E0 = K0 + mp
    P0 = np.sqrt(E0**2 - mp**2)
    beta0 = P0 / E0
    T0 = L0 / (beta0 * c)

    E = K0 + dE + mp
    P = np.sqrt(E**2 - mp**2)

    return 1e9 * (
        (L0 / c) * (
            (1 - alpha_p) * (P - P0)
            + alpha_p * (E**2 - E0**2) / (2 * P0)
        )
        - T0 * dE
    )


def G_(q_ns, Vrf):
    """
    Matches RF kick:
        dE = dE + Vrf * sin(2*pi*q/T_rf + pi)
    which equals:
        dE = dE - Vrf * sin(2*pi*q/T_rf)
    """

    omega = 2 * np.pi / T_rf_ns  # rad/ns

    return -(Vrf / omega) * np.cos(omega * q_ns)

def separatrix_curve(Vrf, n_points=1000, p_max=0.5):
    """
    Exact-Hamiltonian RF bucket separatrix.
    Returns time in ns and dE in MeV.
    """

    q_vals = np.linspace(-T_rf_ns / 2, T_rf_ns / 2, n_points)
    if Vrf <= 0:
        q_vals = np.linspace(-T_rf_ns / 2, T_rf_ns / 2, n_points)
        return q_vals, np.full_like(q_vals, np.nan), np.full_like(q_vals, np.nan)
    
    # unstable fixed point coordinates
    q_saddle = T_rf_ns / 2
    p_saddle = 0.0

    # Hamiltonian of the unstable fixed point
    H_sep = F_(p_saddle) + G_(q_saddle, Vrf)

    p_plus = np.full_like(q_vals, np.nan)
    p_minus = np.full_like(q_vals, np.nan)

    for i, q in enumerate(q_vals):

        # F(dE) calculation
        target = H_sep - G_(q, Vrf)

        # Calculating how far target is from F(dE) calculated in F_
        def root_func(p):
            return F_(p) - target

        # Positive branch
        try:
            p_plus[i] = brentq(root_func, 0.0, p_max)
        except ValueError:
            pass

        # Negative branch
        try:
            p_minus[i] = brentq(root_func, -p_max, 0.0)
        except ValueError:
            pass

    return q_vals, p_plus * 1000, p_minus * 1000

def voltage_program(turn, Vrf_low, Vrf_high, jump_start_turn, jump_turns):
    """
    Non-adiabatic voltage program.

    Holds low voltage, then quickly jumps to max voltage.
    Vrf values are in GeV.
    """

    if turn < jump_start_turn:
        return Vrf_low

    r = (turn - jump_start_turn) / jump_turns
    r = np.clip(r, 0, 1)

    # smooth but fast transition
    ramp_shape = 3*r**2 - 2*r**3

    return Vrf_low + (Vrf_high - Vrf_low) * ramp_shape

stop_sim = False

def update(frame):
    global time, dE, current_turn, stop_sim

    if stop_sim:
        return sc, title, sep_line_plus, sep_line_minus

    for i in range(turns_per_frame):

        turn_number = current_turn

        # -----------------------------
        # Drift update
        # -----------------------------
        K = K0 + dE
        E_total = K + mp
        gamma = E_total / mp
        beta = np.sqrt(1 - 1/gamma**2)
        p = np.sqrt(E_total**2 - mp**2)

        delta = (p - p0) / p0
        L = L0 * (1 + alpha_p * delta)
        T = L / (beta * c)

        time = time + (T - T0) * 1e9
        time = wrap_to_bucket(time, T_rf_ns)

        # -----------------------------
        # Non-adiabatic voltage jump
        # -----------------------------
        Vrf_low = 20e3 / 1e9
        Vrf_high = 320e3 / 1e9

        jump_start_turn = 200
        jump_turns = 50

        Vrf = voltage_program(
            current_turn,
            Vrf_low,
            Vrf_high,
            jump_start_turn,
            jump_turns
        )

        # -----------------------------
        # RF kick

        
        phi = 2 * np.pi * h * time / (T0 * 1e9) + np.pi
        dE = dE + Vrf * np.sin(phi)

        # -----------------------------
        # Log stats
        # -----------------------------
        log_rows.append({
            "turn": turn_number,

            "Vrf_kV": Vrf * 1e9 / 1e3,

            "dE_avg_GeV": np.mean(dE),
            "dE_sigma_GeV": np.std(dE),
            "dE_min_GeV": np.min(dE),
            "dE_max_GeV": np.max(dE),

            "time_avg_ns": np.mean(time),
            "time_sigma_ns": np.std(time),
            "time_min_ns": np.min(time),
            "time_max_ns": np.max(time),
        })

        

        # Stop shortly after strongest compression
        if current_turn > jump_start_turn + jump_turns + 20 and len(log_rows) > 50:
            recent = pd.DataFrame(log_rows)
        
            after_jump = recent[recent["turn"] > jump_start_turn + jump_turns]
        
            if len(after_jump) > 20:
                best_idx = after_jump["time_sigma_ns"].idxmin()
                latest_idx = after_jump.index[-1]
        
                if latest_idx - best_idx > 20:
                    stop_sim = True
            
        current_turn += 1
    # -----------------------------
    # Update particles
    # -----------------------------
    sc.set_offsets(np.column_stack((time, dE * 1000)))

    title.set_text(
        f"Turn {current_turn}, Vrf = {Vrf * 1e9 / 1e3:.1f} kV"
    )

    # -----------------------------
    # Update separatrix
    # -----------------------------
    t_sep, dE_plus, dE_minus = separatrix_curve(Vrf)

    sep_line_plus.set_data(t_sep, dE_plus)
    sep_line_minus.set_data(t_sep, dE_minus)

    return sc, title, sep_line_plus, sep_line_minus

turns_per_frame = 10
n_frames = 100

ani = FuncAnimation(fig, update, frames=n_frames, interval=30, blit=True)

writer = FFMpegWriter(fps=30)
ani.save("rf_bucket_motion.mp4", writer=writer, dpi=150)

log_df = pd.DataFrame(log_rows)
log_df.to_csv("turn_log.csv", index=False)

print("T0 =", T0*1e9, "ns")

log_df = pd.read_csv("turn_log.csv")

plt.figure(figsize=(6,4))
plt.plot(log_df["turn"], log_df["dE_sigma_GeV"] * 1000)
plt.xlabel("Turn")
plt.ylabel("Energy spread sigma [MeV]")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(log_df["turn"], log_df["time_sigma_ns"])
plt.xlabel("Turn")
plt.ylabel("Time spread sigma [ns]")
plt.grid(True)
plt.show()
