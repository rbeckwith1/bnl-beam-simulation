import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import cumulative_trapezoid

from constants import c, mp, K0, L0, alpha_p, h, phi_s


# -----------------------------
# Reference particle
# -----------------------------

E0 = K0 + mp
gamma0 = E0 / mp
beta0 = np.sqrt(1 - 1 / gamma0**2)
p0 = np.sqrt(E0**2 - mp**2)
T0 = L0 / (beta0 * c)


# -----------------------------
# Model parameters
# -----------------------------

Vrf = 0.00001  # GeV
q_range = (-400, 400)      # ns
p_range = (-0.03, 0.03)  # GeV


# -----------------------------
# qdot = f(p)
# -----------------------------

def f(p):
    K = K0 + p
    E = K + mp

    gamma = E / mp
    beta = np.sqrt(1 - 1 / gamma**2)
    momentum = np.sqrt(E**2 - mp**2)

    delta = (momentum - p0) / p0

    L = L0 * (1 + alpha_p * delta)
    T = L / (beta * c)

    return (T - T0) * 1e9  # ns/turn


# -----------------------------
# pdot = g(q)
# -----------------------------

def g(q):
    phi = 2 * np.pi * h * q / (T0 * 1e9)
    return Vrf * np.sin(phi + phi_s)  # GeV/turn


# -----------------------------
# Fixed points
# -----------------------------

def fixed_point_equations(x):
    q, p = x
    return [f(p), g(q)]


guesses = [
    [0, 0],
    [100, 0],
    [-100, 0],
    [300, 0],
    [-300, 0],
]

fixed_points = []

for guess in guesses:
    sol = root(fixed_point_equations, guess)

    if sol.success:
        q_fp, p_fp = sol.x

        duplicate = any(
            np.isclose(q_fp, old_q, atol=1e-3)
            and np.isclose(p_fp, old_p, atol=1e-6)
            for old_q, old_p in fixed_points
        )

        if not duplicate:
            fixed_points.append((q_fp, p_fp))


print("Fixed points:")
for q_fp, p_fp in fixed_points:
    print(f"q = {q_fp:.6f} ns, p = {p_fp * 1000:.6f} MeV")


# -----------------------------
# Phase-space flow plot
# -----------------------------

q_vals = np.linspace(q_range[0], q_range[1], 400)
p_vals = np.linspace(p_range[0], p_range[1], 400)

Q, P = np.meshgrid(q_vals, p_vals)

F = f(P)
G = g(Q)

plt.figure(figsize=(8, 6))
plt.streamplot(Q, P * 1000, F, G * 1000, density=1.3)

for q_fp, p_fp in fixed_points:
    plt.plot(q_fp, p_fp * 1000, "ko")

plt.xlabel("Arrival time deviation q [ns]")
plt.ylabel("Energy deviation p [MeV]")
plt.title("Longitudinal Phase-Space Flow")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Hamiltonian contours
# -----------------------------

# Integrate f(p) with respect to p
Hp = cumulative_trapezoid(f(p_vals), p_vals, initial=0)

# Integrate -g(q) with respect to q
Hq = cumulative_trapezoid(-g(q_vals), q_vals, initial=0)

# Combine to make H(q, p)
# H has shape matching meshgrid: rows are p, columns are q
H = Hp[:, None] + Hq[None, :]


# -----------------------------
# Plot flow + Hamiltonian contours
# -----------------------------

plt.figure(figsize=(10, 6))

# Hamiltonian contours
contours = plt.contour(
    Q,
    P * 1000,
    H,
    levels=30,
    cmap="viridis"
)

plt.clabel(contours, inline=True, fontsize=7)

# Phase-space flow
plt.streamplot(
    Q,
    P * 1000,
    F,
    G * 1000,
    density=1.2,
    color="gray",
    linewidth=0.8,
    arrowsize=0.8
)

# Fixed points
for q_fp, p_fp in fixed_points:
    plt.plot(q_fp, p_fp * 1000, "ko")

plt.xlabel("Arrival time deviation q [ns]")
plt.ylabel("Energy deviation p [MeV]")
plt.title("Longitudinal Phase-Space Flow and Hamiltonian Contours")
plt.grid(True)
plt.tight_layout()
plt.show()