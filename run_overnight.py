#!/usr/bin/env python3
"""
run_overnight.py
================
Build binary → run parameter scan → run interpolation validation.

Streams all output to the terminal AND appends to overnight.log so you
can check progress in the morning with:  tail -f overnight.log

Usage (from the mySVJ directory):
    python run_overnight.py              # full run
    python run_overnight.py --skip-scan  # skip scan if already done
"""

import subprocess
import sys
import time
import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

SCAN_CFG    = 'scan_regression.cfg'
SCAN_SCRIPT = 'scan_regression.py'
VAL_SCRIPT  = 'validate_interpolation.py'
SCAN_OUT    = Path('simulated/regression_scan.npz')
LOG_FILE    = 'overnight.log'

# Validation: 5000 random points at the same event count as the scan.
# validate_interpolation.py saves atomically after every point, so
# whatever finishes before morning is kept.
VAL_N_POINTS = 5000
VAL_N_EVENTS = 2000   # keep consistent with scan for comparable statistics


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    text = msg + '\n'
    sys.stdout.write(text)
    sys.stdout.flush()
    with open(LOG_FILE, 'a') as f:
        f.write(text)


def banner(msg):
    bar  = '=' * 64
    text = f"\n{bar}\n  {msg}\n{bar}\n"
    sys.stdout.write(text)
    sys.stdout.flush()
    with open(LOG_FILE, 'a') as f:
        f.write(text)


def run(cmd, check=True):
    """Run cmd, streaming output to stdout and overnight.log simultaneously."""
    log(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    with open(LOG_FILE, 'a') as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        proc.wait()
    if check and proc.returncode != 0:
        log(f"\nERROR: command exited with code {proc.returncode}.")
        sys.exit(proc.returncode)
    return proc.returncode


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    skip_scan = '--skip-scan' in sys.argv

    banner(f"SVJ overnight run  [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
    log(f"  Log file  : {LOG_FILE}")
    log(f"  Scan cfg  : {SCAN_CFG}")
    log(f"  Scan out  : {SCAN_OUT}")
    log(f"  Val points: {VAL_N_POINTS}  x  {VAL_N_EVENTS} events each")

    # ── Step 1: rebuild binary ────────────────────────────────────────────────
    banner(f"STEP 1: Rebuild binary  [{time.strftime('%H:%M:%S')}]")
    run(['make', 'svj_regression'])
    log("Binary OK.")

    # ── Step 2: parameter scan ────────────────────────────────────────────────
    if skip_scan or SCAN_OUT.exists():
        banner(f"STEP 2: Scan — SKIPPED (output already exists)")
        if not SCAN_OUT.exists():
            log(f"ERROR: --skip-scan given but {SCAN_OUT} not found.")
            sys.exit(1)
    else:
        banner(f"STEP 2: Parameter scan  [{time.strftime('%H:%M:%S')}]")
        t0 = time.time()
        run([sys.executable, SCAN_SCRIPT, SCAN_CFG])
        elapsed = time.time() - t0
        banner(f"Scan complete in {elapsed/3600:.2f} h  [{time.strftime('%H:%M:%S')}]")

    if not SCAN_OUT.exists():
        log(f"ERROR: {SCAN_OUT} missing after scan — cannot validate.")
        sys.exit(1)

    # ── Step 3: validation ────────────────────────────────────────────────────
    banner(f"STEP 3: Validation  [{time.strftime('%H:%M:%S')}]")
    log(f"  {VAL_N_POINTS} random points, {VAL_N_EVENTS} events each")
    log(f"  Results saved atomically to simulated/validation_results.npy\n")
    # check=False: keyboard interrupt or morning kill leaves saved results intact
    rc = run(
        [sys.executable, VAL_SCRIPT,
         str(VAL_N_POINTS), str(VAL_N_EVENTS), SCAN_CFG],
        check=False,
    )

    banner(f"Overnight run finished  [{time.strftime('%H:%M:%S')}]")
    if rc != 0:
        log(f"  Validation exited with code {rc} (may have been interrupted — that's fine).")


if __name__ == '__main__':
    main()
