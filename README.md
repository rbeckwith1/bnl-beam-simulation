# bnl-beam-simulation — reorg notes

## Git workflow
```
git checkout phase-space-flow
git branch archive/phase-space-flow      # frozen fallback, do NOT touch again
git push origin archive/phase-space-flow
git checkout -b reorg                    # do all restructuring here
```
Merge `reorg` into `main` only after checking that `run_non_adiabatic.py`
reproduces the old CSV within numerical noise.

## Why this split
Comparing your three method scripts, ~80% was identical machinery
(kinematics, drift map, RF kick, synchrotron frequency x2 methods,
separatrix, per-turn diagnostics, animation). Only two things actually
differ per method:
1. The voltage program (`rf_programs/`)
2. A handful of config numbers (N, n_turns, target emittance, jump/ramp/mod
   timing) — these live at the top of each `methods/run_*.py`, nowhere else.

Everything else lives in `core/` exactly once. That's the fix for
"I copy a file, tweak a feature, and the sibling method goes stale" — there
is no longer a sibling copy to go stale.

## Status
- `run_non_adiabatic.py` — fully migrated from your example script, should
  run as-is (`cd src && python -m methods.run_non_adiabatic`).
- `run_adiabatic.py` / `run_resonant.py` — stubs. `rf_programs/adiabatic.py`
  and `rf_programs/resonant.py` are placeholder voltage programs (linear
  ramp / basic 2*omega_s modulation) — swap in your real ones, then copy the
  body of `run_non_adiabatic.py` and change the config block + import.
- `results/` is gitignored (regenerable output, not source). Keep it that
  way — don't commit CSVs/mp4s/pngs to the repo.

## Open item carried over
`core/bunch_init.py` still has the TODO about confirming the Bbat/Bbrat
emittance convention with Dr. Brooks — resolve that once, it fixes all
three methods at once now instead of three separate fixes.
