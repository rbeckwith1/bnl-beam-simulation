# AGS Longitudinal Beam Dynamics Simulation

## Overview

A Python simulation of longitudinal beam dynamics for proton bunch compression at the AGS (Alternating Gradient Synchrotron) at BNL, targeting an RMS bunch length around 2 ns. It tracks a population of macro-particles (time deviation and energy deviation relative to a reference particle) turn by turn around the ring, through a drift step (particles of different energy take different amounts of time to go around) and an RF kick step (the RF cavity nudges each particle's energy depending on its arrival phase). This is standard longitudinal phase-space tracking, done above transition energy, where the stable RF phase convention is shifted by 180 degrees relative to the more commonly taught below-transition case.

## Bunching Strategies

The codebase studies three different bunching strategies, each with its own RF voltage program but sharing all the same underlying physics:

- **Non-adiabatic** — hold the RF voltage low, then jump it up quickly (over a timescale much shorter than one synchrotron oscillation period). This deliberately mismatches the bunch to the new, stronger bucket, causing it to rotate and briefly compress in phase space.
- **Adiabatic** — ramp the RF voltage up slowly (over many synchrotron periods), so the bunch stays matched to the bucket throughout and compresses gradually without the rotation/filamentation seen in the non-adiabatic case.
- **Resonant (parametric)** — hold the RF voltage roughly constant but modulate it at close to twice the synchrotron frequency, which can parametrically drive or damp the bunch's envelope (quadrupole/breathing) oscillation, depending on the relative phase of the modulation.

## Architecture

The repository is organized around a shared-core structure so that a physics fix or feature addition made once applies automatically to all three methods, rather than needing to be copy-pasted across separate scripts:

- **`core/`** — everything that's identical across all three methods: reference-particle kinematics, the drift and RF-kick physics, synchrotron frequency calculation (via two independent methods as a cross-check), the separatrix (bucket boundary) calculation used for diagnostics, the shared turn-by-turn tracking loop, standard diagnostic plots, and the phase-space animation renderer.
- **`rf_programs/`** — the one thing that's genuinely different per method: a small class for each method that, given a turn number, returns the RF voltage for that turn.
- **`methods/`** — one run script per method, each just a short configuration block (voltage levels, timing, particle count, target emittance) plus calls into the shared core machinery.

## Acceleration

All three methods support an optional, toggleable acceleration mode layered on top of the base (stationary-bucket) physics. Real machine operation eventually needs to accelerate the beam (ramp its energy) as well as bunch it, which introduces a "synchronous phase" (`phi_s`) — the RF phase at which the reference particle sits, off from the peak of the RF waveform, so that it gains a little energy every turn instead of none.

With acceleration switched off, every number produced is bit-for-bit identical to the original stationary-bucket behavior. Turning acceleration on is controlled by a small number of settings per method script: whether it's enabled at all, what turn it starts ramping on, how many turns that ramp takes, and what synchronous phase it ramps to.

Physically, enabling acceleration does two linked things every turn:
1. Nudges the reference particle's own energy upward (proportional to the sine of the synchronous phase).
2. Shifts every other particle's RF kick to be measured relative to that shifted reference phase rather than the old fixed one.

An accelerating bucket is smaller and asymmetric compared to a stationary one — the RF bucket's separatrix shrinks as the synchronous phase increases, because some of the RF voltage swing is spent maintaining acceleration rather than providing pure restoring force. The diagnostic/animation code reflects this: the drawn bucket boundary shrinks and shifts correctly as acceleration ramps up. The animation also displays the reference particle's current energy and how much it's increased since the start, alongside the turn number, RF voltage, and synchrotron period readouts.

**Caveat:** the bucket-boundary calculation assumes the reference energy used to compute revolution times is fixed at its initial value, even during an accelerating run. For the modest energy increases studied so far (roughly a tenth of a percent to a few tenths of a percent change in reference energy), this is a small approximation; it would need to be revisited for a much larger acceleration ramp.

## Known Limitations

- The "stop early once bunch length stops improving" logic (used by default in the adiabatic and resonant methods) is a patience-counter with no concept of a warm-up period. It's appropriate for a pure bunching run where the goal is to stop right at the compression minimum, but is a poor fit for any run that also accelerates, since bunch length isn't trying to reach a new minimum once the ramp starts. Disable it explicitly for any run where acceleration is enabled.
- A diagnostic option for the maximum number of animation frames is not currently enforced anywhere in the code, so animations may end up longer/heavier than that setting implies.
- Whether the resonance condition (modulation frequency equal to twice the synchrotron frequency) still holds once the bucket has shrunk under acceleration is an open question the code does not currently check for.
- BNL's internal convention for reporting longitudinal emittance (whether it's the full phase-space ellipse area, that area divided by pi, or a four-sigma area) is still to be confirmed with Dr. Brooks, and affects the initial bunch sizing used across all three methods.
