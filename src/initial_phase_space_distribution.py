import numpy as np
import matplotlib.pyplot as plt


# Initial phase space distrbution

N = 10000                     # num of particles

sigma_t = 1.0                 # ns
sigma_E = 1.0                 # MeV


# Generate initial Gaussian bunch (ICs)


time = np.random.normal(0, sigma_t, N)
energy = np.random.normal(0, sigma_E, N)

# Save initial values
time_initial = time.copy()
energy_initial = energy.copy()


# Parameters
n_turns = 10000       # number of times around the ring total
k = 0.0005           # time slip per turn, ns/MeV

n_plots = 5
plot_turns = np.linspace(0, n_turns, n_plots, dtype=int)

fig, axes = plt.subplots(1, n_plots, figsize=(18,4))

plot_index = 0

for turn in range(n_turns + 1):

    # Plot if this is one of the requested turns
    if turn == plot_turns[plot_index]:
        axes[plot_index].scatter(time, energy, s=1, alpha=0.3)
        axes[plot_index].set_title(f"Turn {turn}")
        axes[plot_index].set_xlabel("Time")
        axes[plot_index].set_ylabel("Energy")
        axes[plot_index].grid(True)

        plot_index += 1
        if plot_index >= n_plots:
            break

    # Drift (one turn)
    time = time - k * energy

plt.tight_layout()
plt.show()