import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Single Particle Longitudinal Tracking
# -------------------------------------------------

# Initial particle coordinates
# time > 0 means particle arrives late
time = np.array([400.0])       # ns
dE = np.array([0.0])         # GeV

# Save initial values
time0 = time[0]
dE0 = dE[0]

# -------------------------------------------------
# AGS / Proton Parameters
# -------------------------------------------------

K0 = 24.0          # GeV, synchronous kinetic energy
mp = 0.938272      # GeV, proton rest mass
c = 299792458      # m/s
L0 = 807.1         # m, AGS circumference

gamma_t = 8.45
alpha_p = 1 / gamma_t**2

E0_total = K0 + mp
gamma0 = E0_total / mp
beta0 = np.sqrt(1 - 1 / gamma0**2)
p0 = np.sqrt(E0_total**2 - mp**2)
T0 = L0 / (beta0 * c)

eta = alpha_p - 1 / gamma0**2

print("gamma0 =", gamma0)
print("gamma_t =", gamma_t)
print("eta =", eta)
print("T0 =", T0 * 1e9, "ns")

# -------------------------------------------------
# RF Parameters
# -------------------------------------------------

Vrf = 320e3 / 1e9   # GeV, 320 keV for AGS
h = 6               # harmonic number for AGS

# -------------------------------------------------
# Tracking Settings
# -------------------------------------------------

n_turns = 50000

time_history = []
dE_history = []
turn_history = []

# -------------------------------------------------
# Main Tracking Loop
# -------------------------------------------------

for turn in range(n_turns + 1):

    time_history.append(time[0])
    dE_history.append(dE[0] * 1000)  # GeV -> MeV
    turn_history.append(turn)

    # -----------------------------
    # Drift update
    # -----------------------------
    K = K0 + dE
    E_total = K + mp
    gamma = E_total / mp
    beta = np.sqrt(1 - 1 / gamma**2)
    p = np.sqrt(E_total**2 - mp**2)

    delta = (p - p0) / p0

    L = L0 * (1 + alpha_p * delta)
    T = L / (beta * c)

    time = time + (T - T0) * 1e9

    # -----------------------------
    # RF kick
    # -----------------------------
    phi = 2 * np.pi * h * time / (T0 * 1e9)

    # Negative sign gives stable motion above transition
    dE = dE - Vrf * np.sin(phi)

# -------------------------------------------------
# Plot single-particle phase-space trajectory
# -------------------------------------------------

plt.figure(figsize=(6, 5))
plt.plot(time_history, dE_history, linewidth=1)
plt.scatter(time_history[0], dE_history[0], label="Start", s=40)
plt.scatter(time_history[-1], dE_history[-1], label="End", s=40)

plt.xlabel("Arrival Time Deviation (ns)")
plt.ylabel("Energy Deviation (MeV)")
plt.title("Single Particle Synchrotron Motion")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
