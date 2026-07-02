import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Initial Longitudinal Phase Space Distribution

#start with test particle
N = 10000                  # Number of particles

sigma_t = 1             # Arrival time RMS (ns)
sigma_dE = 0.001           # Energy deviation RMS


dE = np.random.uniform(-0.02, 0.02, N)  # GeV = ±200 MeV
time = np.random.uniform(-250, 250, N)    # ns

# distribution = "uniform"  # "gaussian" or "uniform"

# if distribution == "gaussian":

#     # Initial Gaussian bunch
#     time = np.random.normal(0, sigma_t, N)
#     dE = np.random.normal(0, sigma_dE, N)

# elif distribution == "uniform":

#     # Uniform bunch with approximately the same width
#     time = np.random.uniform(-20*sigma_t, 20*sigma_t, N)
#     dE = np.random.uniform(-8*sigma_dE, 8*sigma_dE, N)

# else:
#     raise ValueError("distribution must be 'gaussian' or 'uniform'")

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
n_turns = 100000
k = 0.0005

print(np.min(time), np.max(time))

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

E0_total = K0 + mp
gamma0 = E0_total / mp
beta0 = np.sqrt(1 - 1/gamma0**2)
p0 = np.sqrt(E0_total**2 - mp**2)
T0 = L0 / (beta0 * c)

# Plots
n_plots = 10
plot_turns = set(np.linspace(0, n_turns, n_plots, dtype=int))

fig, ax = plt.subplots(figsize=(6,5))

sc = ax.scatter(time, dE * 1000, c=initial_time, cmap="coolwarm", s=3, alpha=0.6)

ax.set_xlim(-800, 800)
ax.set_ylim(-275, 275)
ax.set_xlabel("Arrival Time Deviation (ns)")
ax.set_ylabel("Energy Deviation (MeV)")
ax.grid(True)

title = ax.set_title("Turn 0")

current_turn = 0

def update(frame):
    global time, dE, current_turn

    for _ in range(turns_per_frame):

        # Drift update
        K = K0 + dE
        E_total = K + mp
        gamma = E_total / mp
        beta = np.sqrt(1 - 1/gamma**2)
        p = np.sqrt(E_total**2 - mp**2)

        delta = (p - p0) / p0
        L = L0 * (1 + alpha_p * delta)
        T = L / (beta * c)

        time = time + (T - T0) * 1e9

        # RF voltage ramp
        Vrf_initial = 50e3 / 1e9
        Vrf_final = 320e3 / 1e9
        ramp_turns = 100000

        ramp_fraction = min(current_turn / ramp_turns, 1.0)
        Vrf = Vrf_initial + (Vrf_final - Vrf_initial) * ramp_fraction

        # RF kick
        h = 6
        phi = 2 * np.pi * h * time / (T0 * 1e9)

        dE = dE - Vrf * np.sin(phi)

        current_turn += 1

    sc.set_offsets(np.column_stack((time, dE * 1000)))
    title.set_text(
        f"Turn {current_turn}, Vrf = {Vrf * 1e9 / 1e3:.1f} kV"
    )
    return sc, title

turns_per_frame = 100
n_frames = 1000

ani = FuncAnimation(fig, update, frames=n_frames, interval=30, blit=True)

writer = FFMpegWriter(fps=30)
ani.save("rf_bucket_motion.mp4", writer=writer, dpi=150)
print("T0 =", T0*1e9, "ns")