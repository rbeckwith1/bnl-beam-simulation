import numpy as np
import matplotlib.pyplot as plt


# Initial Longitudinal Phase Space Distribution

N = 5000                  # Number of particles

sigma_t = 1.0              # Arrival time RMS (ns)
sigma_dE = 0.001           # Energy deviation RMS

distribution = "uniform"  # "gaussian" or "uniform"

if distribution == "gaussian":

    # Initial Gaussian bunch
    time = np.random.normal(0, sigma_t, N)
    dE = np.random.normal(0, sigma_dE, N)

elif distribution == "uniform":

    # Uniform bunch with approximately the same width
    time = np.random.uniform(-2*sigma_t, 2*sigma_t, N)
    dE = np.random.uniform(-2*sigma_dE, 2*sigma_dE, N)

else:
    raise ValueError("distribution must be 'gaussian' or 'uniform'")

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
n_turns = 500
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
n_plots = 5
plot_turns = set(np.linspace(0, n_turns, n_plots, dtype=int))

initial_time = time.copy()

fig, axes = plt.subplots(1, n_plots, figsize=(18,4))

plot_index = 0

for turn in range(n_turns + 1):

    if turn in plot_turns:

        ax = axes[plot_index]

        sc = ax.scatter(
            time,
            dE * 1000,          # GeV -> MeV for plotting
            c=initial_time,
            cmap="coolwarm",
            s=1,
            alpha=0.5
        )

        ax.set_title(f"Turn {turn}")
        ax.set_xlabel("Arrival Time Deviation (ns)")
        ax.set_ylabel("Energy Deviation (MeV)")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-30, 30)
        ax.grid(True)

        plot_index += 1

    
# Physical drift update
    K = K0 + dE
    E_total = K + mp
    gamma = E_total / mp
    beta = np.sqrt(1 - 1/gamma**2)
    p = np.sqrt(E_total**2 - mp**2)

    delta = (p - p0) / p0

    L = L0 * (1 + alpha_p * delta)
    T = L / (beta * c)

    time = time + (T - T0) * 1e9
    time = time + (T - T0) * 1e9

    # # Optional RF kick
    # Vrf = 0.00001   # 10 keV   # reference vlaue for V0 GeV
    # h = 12 # reference value for the harmonic number
    
    # phi = 2 * np.pi * h * time / (T0*1e9)
    # dE = dE + Vrf * np.sin(phi)

    if plot_index >= n_plots:
        break

plt.tight_layout()
plt.show()

print("T0 =", T0*1e9, "ns")
print("max(T-T0) =", np.max(T-T0)*1e12, "ps")
print("min(T-T0) =", np.min(T-T0)*1e12, "ps")