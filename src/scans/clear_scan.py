import os
import shutil
import sys
import pandas as pd

SCANS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR   = os.path.dirname(SCANS_DIR)
ROOT_DIR  = os.path.join(SRC_DIR, "results", "resonant_single_run")
HIST_DIR  = os.path.join(ROOT_DIR, "histories")
SUMMARY_PATH = os.path.join(ROOT_DIR, "scan_summary.csv")

mode = input("Mode: (p)rune orphaned rows only, or (w)ipe everything? [p/w] ").strip().lower()
PRUNE = mode == "p"

if PRUNE:
    if not os.path.exists(SUMMARY_PATH):
        print("No scan_summary.csv found, nothing to prune.")
        raise SystemExit

    df = pd.read_csv(SUMMARY_PATH)
    exists_mask = df["run_id"].apply(
        lambda rid: os.path.exists(os.path.join(HIST_DIR, f"{rid}.csv"))
    )
    orphaned = df[~exists_mask]

    if orphaned.empty:
        print("No orphaned rows found.")
        raise SystemExit

    print(f"Found {len(orphaned)} orphaned row(s):")
    print(orphaned[["run_id", "mod_depth", "ramp_turns", "V_start_kV"]])
    resp = input("Drop these rows from scan_summary.csv? [y/N] ")
    if resp.strip().lower() != "y":
        print("Aborted, nothing changed.")
        raise SystemExit

    df[exists_mask].to_csv(SUMMARY_PATH, index=False)
    print(f"Dropped {len(orphaned)} row(s), summary rewritten.")
    raise SystemExit

# ================= full wipe (original behavior) =================
resp = input(f"This will delete all files in:\n  {HIST_DIR}\nand remove:\n  {SUMMARY_PATH}\nProceed? [y/N] ")
if resp.strip().lower() != "y":
    print("Aborted, nothing deleted.")
    raise SystemExit

shutil.rmtree(HIST_DIR, ignore_errors=True)
os.makedirs(HIST_DIR, exist_ok=True)
if os.path.exists(SUMMARY_PATH):
    os.remove(SUMMARY_PATH)
print(f"Cleared {ROOT_DIR}")
print("  histories/ reset to empty")
print("  scan_summary.csv removed")