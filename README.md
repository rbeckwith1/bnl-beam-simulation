# AGS Longitudinal Beam Dynamics Simulation

## Overview

This repository contains a Python-based longitudinal beam dynamics simulation developed to study proton bunch compression in Brookhaven National Laboratory's Alternating Gradient Synchrotron (AGS) for potential muon collider applications.

The simulation tracks macroparticles turn-by-turn in longitudinal phase space, ((\Delta t,\Delta E)), using a drift-kick model:

* **Drift:** particles with different energies have different revolution periods and therefore accumulate arrival-time offsets.
* **RF kick:** particles receive an energy kick determined by their arrival phase and the RF voltage and phase on that turn.

The model operates above transition energy using AGS machine parameters and includes RF bucket dynamics, synchrotron motion, separatrix calculations, configurable initial bunch distributions, and turn-by-turn diagnostics.

The primary goal of the study was to compare several methods for compressing an AGS proton bunch toward an RMS bunch length of approximately **2 ns**.

## Compression Methods

Four RF programs are implemented using the same underlying longitudinal tracking model.

### Adiabatic Bunching

The RF voltage is increased slowly compared with the synchrotron period, allowing the bunch to remain approximately matched to the evolving RF bucket. This method provides modest compression and serves primarily as a validation case for the tracking model.

The simulated final distribution can be compared with the theoretical matched distribution at the final RF voltage to verify the implementation of adiabatic longitudinal dynamics.

### Non-Adiabatic Bunching

The bunch is initialized at a low RF voltage and the voltage is rapidly increased. The sudden change mismatches the bunch to the new bucket, causing the distribution to rotate in longitudinal phase space and temporarily reach a short bunch length.

The main optimization parameters are:

* starting RF voltage;
* voltage-jump duration.

### Unstable Fixed-Point Bunching

The RF phase is shifted by (180^\circ), placing the bunch near the unstable fixed point. The distribution evolves along the nonlinear phase-space flow near the separatrix before the RF phase is restored and the bunch is recaptured in the stable bucket.

The primary optimization parameter is the number of turns spent near the unstable fixed point before recapture.

### Resonant Bunching

The RF voltage is sinusoidally modulated near twice the synchrotron frequency,

[
f_{\mathrm{mod}} \approx 2f_s,
]

to excite the quadrupole, or breathing, oscillation of the bunch. Compression develops over multiple oscillations rather than through a single rapid rotation.

The primary optimization parameter is the RF modulation depth.

## Repository Structure

The code uses a shared-core architecture so that the compression methods use the same tracking physics, diagnostics, and machine model.

```text
.
├── core/
│   ├── ...
│   └── shared longitudinal tracking and diagnostics
│
├── rf_programs/
│   ├── ...
│   └── method-specific RF voltage/phase programs
│
├── methods/
│   ├── ...
│   └── run/configuration scripts for each compression method
│
└── ...
```

### `core/`

Contains the physics and utilities shared across the compression methods, including:

* AGS/reference-particle kinematics;
* relativistic quantities and slip factor;
* longitudinal drift and RF kick calculations;
* synchrotron frequency and period calculations;
* RF bucket and separatrix calculations;
* turn-by-turn macroparticle tracking;
* RMS bunch-length and energy-spread diagnostics;
* phase-space plotting;
* storyboard and animation generation; and
* common initialization utilities.

Keeping these calculations in the shared core ensures that changes to the underlying physics are applied consistently to every compression method.

### `rf_programs/`

Defines the method-specific RF programs.

These routines determine the RF voltage and, where applicable, RF phase on each turn. This separates the compression strategy from the underlying particle tracking.

For example, the non-adiabatic method implements a rapid voltage transition, the resonant method adds a sinusoidal voltage modulation, and the unstable fixed-point method controls the RF phase jump and recapture.

### `methods/`

Contains the run scripts for the individual compression methods. These scripts primarily specify simulation parameters such as:

* number of macroparticles and turns;
* initial and final RF voltage;
* longitudinal emittance;
* initial bunch distribution;
* ramp or jump duration;
* modulation depth;
* unstable fixed-point dwell time; and
* output/diagnostic settings.

The scripts then call the common tracking and plotting routines from `core/`.

## Initial Bunch Distribution

The simulation supports elliptical longitudinal phase-space distributions of the form

[
f(t,\Delta E)
\propto
\left[
1-\left(\frac{t}{T}\right)^2
-\left(\frac{\Delta E}{\Delta E_{\max}}\right)^2
\right]^{J-1/2}.
]

Integrating over energy gives the corresponding longitudinal profile

[
I(t)\propto
\left[
1-\left(\frac{t}{T}\right)^2
\right]^J.
]

A uniformly filled phase-space ellipse corresponds to (J=1/2).

The primary compression-method comparison was performed using this (J=1/2) distribution and a longitudinal emittance of approximately **1.35 eV s** so that all methods could be compared from identical initial conditions.

Measured AGS wall-current-monitor profiles were subsequently digitized and compared with several analytical distributions. A (J=3/2) profile provided better agreement with the measured bunch shapes and should therefore be used in future studies intended to make quantitative predictions of AGS performance.

The existing compression results should consequently be interpreted primarily as **comparative benchmarks between methods**, rather than definitive predictions of achievable AGS bunch lengths.

## Parameter Scans

Optimization routines are included to study the dominant control parameter for each compression method.

The scans used in the initial study include:

* **Non-adiabatic:** starting RF voltage and voltage-transition duration;
* **Unstable fixed point:** dwell time before recapture;
* **Resonant:** RF modulation depth.

Typical outputs include:

* minimum RMS bunch length;
* turn at which maximum compression occurs;
* RMS bunch length versus turn;
* parameter-scan heat maps;
* longitudinal phase-space snapshots; and
* animations of the bunch evolution.

These scans are intended to compare trends and identify promising operating regions. They should not be interpreted as global optimization of the complete AGS RF program.

## Longitudinal Stability

The repository also includes analysis of the longitudinal microwave-instability threshold using a Boussard/Keil--Schnell-type criterion.

The stability calculation uses the simulated bunch length and energy spread to estimate the longitudinal impedance threshold during the compression cycle. Peak current calculations can account for the assumed longitudinal distribution rather than treating every bunch as Gaussian.

This analysis is important because stronger compression increases the peak current and can reduce the margin against collective instability. Stability considerations also affect the appropriate initial longitudinal emittance and therefore the quantitative compression results.

## Acceleration

The tracking model includes an optional acceleration mode in which the synchronous phase is shifted away from the stationary-bucket value and the reference-particle energy changes turn-by-turn.

When enabled, the model:

1. updates the reference-particle energy from the synchronous RF kick;
2. evaluates particle kicks relative to the evolving synchronous phase; and
3. updates diagnostics associated with the accelerating bucket.

Acceleration can be enabled or disabled from the method configuration scripts. The primary bunch-compression comparison was performed using stationary-bucket conditions.

## Outputs and Diagnostics

Depending on the run configuration, the simulation can generate:

* RMS bunch length versus turn;
* RMS energy spread versus turn;
* RF voltage and phase histories;
* longitudinal phase-space distributions;
* RF separatrix overlays;
* initial bunch current profiles;
* parameter-scan heat maps;
* minimum-bunch-length summaries;
* phase-space storyboards;
* animations; and
* longitudinal stability/impedance estimates.

These diagnostics are useful both for comparing compression methods and for checking the underlying longitudinal dynamics.

## Important Assumptions and Limitations

The current simulation is primarily a **single-particle longitudinal tracking model**. Important limitations include:

* Collective effects such as longitudinal space charge and wakefields are not directly included in the particle tracking.
* Machine longitudinal impedance is evaluated separately through stability estimates rather than self-consistently through wakefield tracking.
* The primary compression comparison used a (J=1/2) initial distribution, while measured AGS profiles are better represented by approximately (J=3/2).
* The longitudinal emittance used in the original comparison should be revisited together with the stability constraints.
* RF programs are idealized and do not yet fully model cavity bandwidth, beam loading, or experimentally achievable voltage and phase slew rates.
* Parameter scans generally vary only the dominant control parameter and therefore do not establish global optima.
* Resonant operation assumes a prescribed relationship between the modulation frequency and synchrotron frequency; maintaining this condition under changing machine conditions requires further study.
* Acceleration has been tested only over relatively small changes in reference energy and would require additional validation for large energy ramps.

## Interpretation of Results

Under the baseline assumptions used for the original method comparison, the three non-adiabatic compression techniques produced minimum RMS bunch lengths near the 2 ns design target, with the rapid voltage-jump method producing the strongest compression.

These values should be treated as **simulation benchmarks**, not concrete predictions of AGS performance. More realistic studies should incorporate the measured (J=3/2) initial distribution, updated longitudinal emittance and stability constraints, realistic AGS RF programs, and collective effects.

The main purpose of the current codebase is therefore to provide a common framework for studying and comparing candidate longitudinal compression schemes and for progressively adding the machine physics needed for a realistic AGS bunch-compression model.

## Future Development

The main next steps are:

* initialize compression studies using the measured (J=3/2) bunch distribution;
* refine the initial longitudinal emittance using AGS measurements and stability constraints;
* incorporate realistic RF voltage and phase slew rates;
* include collective longitudinal effects and machine impedance in the tracking;
* extend optimization to multiple simultaneous RF parameters;
* investigate combined or multi-stage compression schemes; and
* benchmark the tracking model against established longitudinal beam-dynamics tools such as BLonD.
