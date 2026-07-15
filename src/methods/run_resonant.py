"""
Resonant (parametric) bunching run. TODO(Rosalyn): mirror
run_non_adiabatic.py, swapping in ResonantProgram + your real modulation
config once migrated.
"""
import os
from rf_programs.resonant import ResonantProgram

OUT_DIR = "results/resonant"
os.makedirs(OUT_DIR, exist_ok=True)

# TODO: port the rest of your resonant script here, same pattern as
# run_non_adiabatic.py
