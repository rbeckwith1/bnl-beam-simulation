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
        ramp_shape = r**2
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

    title.set_text(
        f"Turn {current_turn}, "
        f"Vrf = {Vrf * 1e9 / 1e3:.1f} kV, "
        f"K0 = {K0:.3f} GeV"
    )

    return sc, title

turns_per_frame = 100
n_frames = 500

def init():
    sc.set_offsets(np.column_stack((time, dE * 1000)))
    title.set_text(f"Turn {current_turn}")
    return sc, title

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
