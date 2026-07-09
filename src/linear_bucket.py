import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
# Initial Longitudinal Phase Space Distribution

#start with test particle

N = 10000 # num of particles

K0 = 24
mp = 0.938272
c = 299792458
L0 = 807.1
gamma_t = 8.667
alpha_p = 1 / gamma_t**2
h = 6

Vrf = 320e3 / 1e9
phi_s = np.deg2rad(30)
phi_ref = np.pi - phi_s

# Parameters
n_turns = 70000
k = 0.0005


# Synchronous particle
K0 = 24          # GeV -AGS parameter
mp = 0.938272      # GeV -rest mass of proton
c = 299792458      # m/s
L0 = 807.1         # m -circumference of AGS

# Reference fractional momentum deviation
gamma_t = 8.667
alpha_p = 1 / gamma_t**2
h = 6

E0_total = K0 + mp
gamma0 = E0_total / mp
beta0 = np.sqrt(1 - 1/gamma0**2)
p0 = np.sqrt(E0_total**2 - mp**2)
T0 = L0 / (beta0 * c)

# RF period
T_rf_ns = (T0 / h) * 1e9

current_turn = 0
log_rows = []

def wrap_to_bucket(time_ns, T_rf_ns):
    """
    Wrap arrival-time deviation into one RF bucket:
    [-T_rf/2, +T_rf/2)
    """
    return ((time_ns + T_rf_ns/2) % T_rf_ns) - T_rf_ns/2


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


def linear_drift_slope(K0, eps=1e-5):
    """
    Numerically estimate a = d(phi_dot)/d(dE) at dE = 0.
    Units: rad/turn/GeV
    """
    dE_test = np.array([-eps, eps])
    phi_dot = drift_phase_per_turn(dE_test, K0)

    return (phi_dot[1] - phi_dot[0]) / (2 * eps)

def approximate_accelerating_bucket(K0, Vrf, phi_s, dE_max_grid=0.4, n_points=1500):
    """
    Approximate accelerating bucket using:
        H = 0.5*C*dE^2 + Vrf*(cos(phi) + phi*sin(phi_s))

    where C = d(phi_dot)/d(dE) near dE = 0.

    This linearizes the drift/energy part, but keeps the nonlinear RF potential.
    """

    if Vrf <= 0:
        return np.array([]), np.array([])

    # RF period
    E0_total = K0 + mp
    gamma0 = E0_total / mp
    beta0 = np.sqrt(1 - 1 / gamma0**2)
    T0 = L0 / (beta0 * c)
    T_rf_ns = (T0 / h) * 1e9

    # Linear drift coefficient
    C = linear_drift_slope(K0)

    # Stable and unstable phases for accelerating bucket
    phi_stable = np.pi - phi_s
    phi_unstable = phi_s

    # RF potential part
    def U(phi):
        return Vrf * (np.cos(phi) + phi * np.sin(phi_s))

    # Separatrix Hamiltonian at unstable fixed point
    H_sep = U(phi_unstable)

    # Sweep phase across the bucket
    phi_grid = np.linspace(phi_unstable, phi_unstable + 2*np.pi, n_points)

    # Energy part needed:
    # 0.5*C*dE^2 = H_sep - U(phi)
    needed = H_sep - U(phi_grid)

    # Need sign convention to produce positive dE^2
    C_abs = abs(C)

    dE_squared = 2 * needed / C_abs

    valid = dE_squared >= 0

    dE_upper = np.full_like(phi_grid, np.nan)
    dE_lower = np.full_like(phi_grid, np.nan)

    dE_upper[valid] = np.sqrt(dE_squared[valid])
    dE_lower[valid] = -np.sqrt(dE_squared[valid])

    # Convert phase to time relative to stable phase
    time_grid = (phi_grid - phi_stable) * T_rf_ns / (2*np.pi)

    x = np.concatenate([time_grid, time_grid[::-1]])
    y = np.concatenate([dE_upper, dE_lower[::-1]])

    good = np.isfinite(x) & np.isfinite(y)

    return x[good], y[good]

def linear_bucket_ellipse(K0, Vrf, phi_ref, J, n_points=1000):
    if Vrf <= 0:
        return np.array([]), np.array([])

    E0_total = K0 + mp
    gamma0 = E0_total / mp
    beta0 = np.sqrt(1 - 1 / gamma0**2)
    T0 = L0 / (beta0 * c)
    T_rf_ns = (T0 / h) * 1e9

    a = linear_drift_slope(K0)
    b = Vrf * np.cos(phi_ref)

    a_abs = abs(a)
    b_abs = abs(b)

    omega_s = np.sqrt(a_abs * b_abs)

    H = J * omega_s

    dE_max = np.sqrt(2 * H / a_abs)
    phi_max = np.sqrt(2 * H / b_abs)

    theta = np.linspace(0, 2*np.pi, n_points)

    phi = phi_max * np.cos(theta)
    dE = dE_max * np.sin(theta)

    time = phi * T_rf_ns / (2*np.pi)

    return time, dE


# Ramp / acceleration controls
# -----------------------------
Vrf_initial = 5e3 / 1e9       # GeV, small nonzero voltage
Vrf_final = 320e3 / 1e9       # GeV
ramp_start_turn = 0
ramp_turns = 300000

accel_start_turn = 5000000      # delay acceleration
phi_s = np.deg2rad(30)
phi_ref = np.pi - phi_s

Vrf = Vrf_initial

# -----------------------------
# Initial small filled ellipse
dE_initial_bucket = 0.003  # GeV

a0 = abs(linear_drift_slope(K0))
b0 = abs(Vrf_initial * np.cos(phi_ref))
omega0 = np.sqrt(a0 * b0)

H0 = 0.5 * a0 * dE_initial_bucket**2
J_bunch = H0 / omega0

sep_x, sep_y = linear_bucket_ellipse(
    K0=K0,
    Vrf=Vrf_initial,
    phi_ref=phi_ref,
    J=J_bunch
)

time_max = np.max(np.abs(sep_x))
dE_max = np.max(np.abs(sep_y))

theta = np.random.uniform(0, 2*np.pi, N)
r = np.sqrt(np.random.uniform(0, 1, N))

time = r * time_max * np.cos(theta)
dE = r * dE_max * np.sin(theta)

time_initial = time.copy()
dE_initial = dE.copy()


# -----------------------------
# Plot setup

fig, ax = plt.subplots(figsize=(6, 5))

sc = ax.scatter(
    time,
    dE * 1000,
    c=time_initial,
    cmap="coolwarm",
    s=3,
    alpha=0.6
)

sep_line, = ax.plot(
    sep_x,
    sep_y * 1000,
    "k-",
    lw=2,
    label="Linear Bucket Ellipse"
)


ax.set_xlim(-50, 50)
ax.set_ylim(-20, 20)

ax.set_xlabel("Arrival Time Deviation (ns)")
ax.set_ylabel("Energy Deviation (MeV)")
ax.grid(True)
ax.legend(loc="upper right")

title = ax.set_title(
    f"Turn 0, Vrf = {Vrf * 1e9 / 1e3:.1f} kV, K0 = {K0:.3f} GeV"
)

def smoothstep(r):
    r = np.clip(r, 0, 1)
    return 3*r**2 - 2*r**3


def acceleration_ramp(Vrf, current_turn, accel_start_turn, accel_ramp_turns, phi_s_final):
    """
    Smoothly ramp reference acceleration by ramping synchronous phase.

    Before ramp:
        phi_s = 0      -> no acceleration
        phi_ref = pi   -> stationary bucket

    After ramp:
        phi_s = phi_s_final
        phi_ref = pi - phi_s
        dK0_turn = Vrf*sin(phi_s)
    """

    if current_turn < accel_start_turn:
        phi_s = 0.0
    else:
        r = (current_turn - accel_start_turn) / accel_ramp_turns
        phi_s = phi_s_final * smoothstep(r)

    dK0_turn = Vrf * np.sin(phi_s)

    phi_ref = np.pi - phi_s

    return dK0_turn, phi_s, phi_ref

def voltage_ramp(turn, Vrf_initial, Vrf_final, ramp_start_turn, ramp_turns):
    if turn < ramp_start_turn:
        return Vrf_initial

    r = (turn - ramp_start_turn) / ramp_turns
    r = np.clip(r, 0, 1)

    # smooth slow-start/slow-end ramp
    ramp_shape = 3*r**2 - 2*r**3

    return Vrf_initial + (Vrf_final - Vrf_initial) * ramp_shape

def update(frame):
    global time, dE, current_turn, K0, Vrf

    for i in range(turns_per_frame):

        turn_number = current_turn

        # Voltage ramp
        Vrf = voltage_ramp(
            turn=current_turn,
            Vrf_initial=Vrf_initial,
            Vrf_final=Vrf_final,
            ramp_start_turn=ramp_start_turn,
            ramp_turns=ramp_turns
        )

        # RF period for current reference energy
        E0_total = K0 + mp
        gamma0 = E0_total / mp
        beta0 = np.sqrt(1 - 1 / gamma0**2)
        T0 = L0 / (beta0 * c)
        T_rf_ns = (T0 / h) * 1e9

        # Phase coordinate near synchronous particle
        phi = 2 * np.pi * time / T_rf_ns

        accel_start_turn = 20000
        accel_ramp_turns = 40000
        phi_s_final = np.deg2rad(30)

        # Smooth acceleration ramp
        dK0_turn, phi_s, phi_ref = acceleration_ramp(
            Vrf=Vrf,
            current_turn=current_turn,
            accel_start_turn=accel_start_turn,
            accel_ramp_turns=accel_ramp_turns,
            phi_s_final=phi_s_final
        )
        
        # Linear coefficients
        a = linear_drift_slope(K0)
        b = Vrf * np.cos(phi_ref)

        # Linear drift
        phi = phi + a * dE

        # Linear RF restoring kick
        dE = dE + b * phi

        # Convert back to time
        time = phi * T_rf_ns / (2*np.pi)

        K0 += dK0_turn

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

    sep_x, sep_y = linear_bucket_ellipse(
    K0=K0,
    Vrf=Vrf,
    phi_ref=phi_ref,
    J=J_bunch
)

    sep_line.set_data(sep_x, sep_y * 1000)

    title.set_text(
        f"Turn {current_turn}, "
        f"Vrf = {Vrf * 1e9 / 1e3:.1f} kV, "
        f"K0 = {K0:.3f} GeV"
    )

    return sc, title, sep_line

turns_per_frame = 100
n_frames = 700

def init():
    global time, dE, current_turn, K0, Vrf

    sc.set_offsets(np.column_stack((time, dE * 1000)))

    sep_x, sep_y = linear_bucket_ellipse(
    K0=K0,
    Vrf=Vrf,
    phi_ref=phi_ref,
    J=J_bunch
)

    sep_line.set_data(sep_x, sep_y * 1000)

    title.set_text(
        f"Turn {current_turn}, "
        f"Vrf = {Vrf * 1e9 / 1e3:.1f} kV, "
        f"K0 = {K0:.3f} GeV"
    )

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
