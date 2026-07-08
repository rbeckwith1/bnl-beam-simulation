import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
# Initial Longitudinal Phase Space Distribution

#start with test particle
N = 10000                  # Number of particles

sigma_t = 1             # Arrival time RMS (ns)
sigma_dE = 0.001           # Energy deviation RMS


dE = np.random.uniform(-0.02, 0.02, N)  # GeV = ±200 MeV
time = np.random.uniform(-250, 250, N)    # ns

# Save initial coordinates (used for coloring particles)
time_initial = time.copy()
dE_initial = dE.copy()


# Fixed Plot Limits

padding_t = 0.5      # ns
padding_dE = 0.0005  # GeV

t_min = np.min(time) - padding_t
t_max = np.max(time) + padding_t

e_min = np.min(dE) - padding_dE
e_max = np.max(dE) + padding_dE

# Parameters
n_turns = 50000
k = 0.0005

initial_time = time.copy()
initial_time = time.copy()

# Synchronous particle
K0 = 24          # GeV -AGS parameter
mp = 0.938272      # GeV -rest mass of proton
c = 299792458      # m/s
L0 = 807.1         # m -circumference of AGS

# Reference fractional momentum deviation
gamma_t = 8.45
alpha_p = 1 / gamma_t**2
h = 6

E0_total = K0 + mp
gamma0 = E0_total / mp
beta0 = np.sqrt(1 - 1/gamma0**2)
p0 = np.sqrt(E0_total**2 - mp**2)
T0 = L0 / (beta0 * c)

# RF period
T_rf_ns = (T0 / h) * 1e9

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
log_rows = []

def wrap_to_bucket(time_ns, T_rf_ns):
    """
    Wrap arrival-time deviation into one RF bucket:
    [-T_rf/2, +T_rf/2)
    """
    return ((time_ns + T_rf_ns/2) % T_rf_ns) - T_rf_ns/2

# Separatrix line
sep_line, = ax.plot([], [], "k-", lw=2, label="Separatrix")
ax.legend(loc="upper right")


def drift_phase_per_turn(dE_grid, K0):
    """
    Returns dphi/dturn caused by the drift, using the same model as the particles.
    """

    E0_total = K0 + mp
    gamma0 = E0_total / mp
    beta0 = np.sqrt(1 - 1 / gamma0**2)
    p0 = np.sqrt(E0_total**2 - mp**2)
    T0 = L0 / (beta0 * c)
    T_rf_ns = (T0 / h) * 1e9

    K = K0 + dE_grid
    E_total = K + mp
    gamma = E_total / mp
    beta = np.sqrt(1 - 1 / gamma**2)
    p = np.sqrt(E_total**2 - mp**2)

    delta = (p - p0) / p0
    L = L0 * (1 + alpha_p * delta)
    T = L / (beta * c)

    F_ns = (T - T0) * 1e9

    return 2 * np.pi * F_ns / T_rf_ns

def accelerating_separatrix(K0, Vrf, phi_ref, dE_max=0.4):
    """
    Computes the accelerating bucket separatrix using the same nonlinear drift
    model as the simulation.
    """

    if Vrf <= 0:
        return np.array([]), np.array([])

    # Energy grid
    dE_grid = np.linspace(-dE_max, dE_max, 4001)

    # Exact drift term from your code
    phi_dot = drift_phase_per_turn(dE_grid, K0)

    # Integrate phi_dot with respect to dE
    H_E = np.zeros_like(dE_grid)
    zero_index = np.argmin(np.abs(dE_grid))

    for j in range(zero_index + 1, len(dE_grid)):
        H_E[j] = H_E[j-1] + 0.5 * (
            phi_dot[j] + phi_dot[j-1]
        ) * (dE_grid[j] - dE_grid[j-1])

    for j in range(zero_index - 1, -1, -1):
        H_E[j] = H_E[j+1] - 0.5 * (
            phi_dot[j] + phi_dot[j+1]
        ) * (dE_grid[j+1] - dE_grid[j])

    # RF potential
    def H_phi(phi):
        return Vrf * (np.cos(phi) + phi * np.sin(phi_ref))

    # Unstable fixed point
    phi_u = np.pi - phi_ref

    while phi_u > phi_ref:
        phi_u -= 2 * np.pi

    H_sep = H_phi(phi_u)

    phi_grid = np.linspace(phi_u, phi_u + 2*np.pi, 1500)

    needed_energy_H = H_sep - H_phi(phi_grid)

    # Split positive and negative branches
    pos = dE_grid >= 0
    neg = dE_grid <= 0

    H_pos = H_E[pos]
    dE_pos = dE_grid[pos]

    H_neg = H_E[neg][::-1]
    dE_neg = dE_grid[neg][::-1]

    dE_upper = np.full_like(phi_grid, np.nan)
    dE_lower = np.full_like(phi_grid, np.nan)

    valid_upper = (needed_energy_H >= H_pos.min()) & (needed_energy_H <= H_pos.max())
    valid_lower = (needed_energy_H >= H_neg.min()) & (needed_energy_H <= H_neg.max())

    dE_upper[valid_upper] = np.interp(
        needed_energy_H[valid_upper],
        H_pos,
        dE_pos
    )

    dE_lower[valid_lower] = np.interp(
        needed_energy_H[valid_lower],
        H_neg,
        dE_neg
    )

    # Current RF bucket spacing
    E0_total = K0 + mp
    gamma0 = E0_total / mp
    beta0 = np.sqrt(1 - 1 / gamma0**2)
    T0 = L0 / (beta0 * c)
    T_rf_ns = (T0 / h) * 1e9

    time_grid = (phi_grid - phi_ref) * T_rf_ns / (2*np.pi)

    x = np.concatenate([time_grid, time_grid[::-1]])
    y = np.concatenate([dE_upper, dE_lower[::-1]])

    good = np.isfinite(x) & np.isfinite(y)

    return x[good], y[good]

def update(frame):
    global time, dE, current_turn, K0

    if frame == 0:
        sc.set_offsets(np.column_stack((time, dE * 1000)))
        title.set_text(
            f"Turn {current_turn}, "
            f"Vrf = 0.0 kV, "
            f"K0 = {K0:.3f} GeV"
        )

    for i in range(turns_per_frame):

        turn_number = current_turn

        # Reference particle update
        E0_total = K0 + mp
        gamma0 = E0_total / mp
        beta0 = np.sqrt(1 - 1 / gamma0**2)
        p0 = np.sqrt(E0_total**2 - mp**2)
        T0 = L0 / (beta0 * c)
        T_rf_ns = (T0 / h) * 1e9


        # Particle drift update
        K = K0 + dE
        E_total = K + mp
        gamma = E_total / mp
        beta = np.sqrt(1 - 1 / gamma**2)
        p = np.sqrt(E_total**2 - mp**2)

        delta = (p - p0) / p0
        L = L0 * (1 + alpha_p * delta)
        T = L / (beta * c)

        # wrap time to single bucket
        time = time + (T - T0) * 1e9
        time = wrap_to_bucket(time, T_rf_ns)

        # RF voltage ramp
        Vrf_initial = 0
        Vrf_final = 320e3 / 1e9
        ramp_turns = 100000 #number of turns to obtain final voltage

        # nonlinear model when ramp_shape^n | linear when multiplied by const
        r = min(current_turn / ramp_turns, 1.0)
        ramp_shape = r**1.5
        Vrf = Vrf_initial + (Vrf_final - Vrf_initial) * ramp_shape
        
        # RF kick with accelerating reference phase
        phi_s = np.deg2rad(30)

        # Above transition: stable accelerating phase is shifted by pi
        phi_ref = np.pi - phi_s

        # Particle phase relative to reference particle
        phi = phi_ref + 2 * np.pi * time / T_rf_ns

        kick = Vrf * np.sin(phi)
        kick_ref = Vrf * np.sin(phi_ref)

        # Update energy deviation relative to reference
        dE += kick - kick_ref

        # Update reference kinetic energy
        K0 += kick_ref

        # Logging
        log_rows.append({
            "turn": turn_number,
            "K0_GeV": K0,
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

        current_turn += 1

    sc.set_offsets(np.column_stack((time, dE * 1000)))
    
    sep_x, sep_y = accelerating_separatrix(
        K0=K0,
        Vrf=Vrf,
        phi_ref=phi_ref,
        dE_max=0.4
    )

    sep_line.set_data(sep_x, sep_y * 1000)

    title.set_text(
        f"Turn {current_turn}, "
        f"Vrf = {Vrf * 1e9 / 1e3:.1f} kV, "
        f"K0 = {K0:.3f} GeV"
    )

    return sc, title, sep_line

turns_per_frame = 100
n_frames = 500

def init():
    sc.set_offsets(np.column_stack((time, dE * 1000)))
    sep_line.set_data([], [])
    title.set_text(f"Turn {current_turn}")
    return sc, title, sep_line

ani = FuncAnimation(
    fig,
    update,
    frames=n_frames + 1,
    init_func=init,
    interval=30,
    blit=False
)

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
