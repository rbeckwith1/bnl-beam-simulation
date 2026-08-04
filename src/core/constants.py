"""
Machine constants for the AGS. These describe the machine itself, not any
particular bunching method -- nothing here should ever differ between
adiabatic / non-adiabatic / resonant runs. Method-specific knobs (RF voltage
programs, jump timing, modulation depth, target emittance, N/n_turns) live in
rf_programs/ and methods/, not here.
"""

# Reference particle / ring
K0 = 24.0            # reference kinetic energy [GeV]
mp = 0.938272         # proton rest mass [GeV]
c = 299792458.0       # speed of light [m/s]
L0 = 807.1            # AGS circumference [m]

# Transition / momentum compaction
gamma_t = 8.5
alpha_p = 1.0 / gamma_t**2

# RF harmonic number
h = 6
