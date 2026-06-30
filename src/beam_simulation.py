import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Simulation Parameters
# -----------------------------

N = 10000                     # Number of particles

sigma_t = 1.0                 # ns
sigma_E = 1.0                 # MeV

# -----------------------------
# Generate Gaussian bunch
# -----------------------------

time = np.random.normal(0, sigma_t, N)
energy = np.random.normal(0, sigma_E, N)

# -----------------------------
# Plot longitudinal phase space
# -----------------------------

plt.figure(figsize=(8,6))

plt.scatter(time, energy,
            s=2,
            alpha=0.35)

plt.xlabel("Arrival Time (ns)")
plt.ylabel("Energy Deviation (MeV)")
plt.title("Initial Proton Bunch")

plt.grid(True)

plt.show()