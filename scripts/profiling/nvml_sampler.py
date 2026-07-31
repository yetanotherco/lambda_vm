#!/usr/bin/env python3
"""Sample GPU utilization to a CSV, timestamped with epoch nanoseconds.

Designed to run alongside an instruments-enabled prove: the spans' `start_ns`
epoch field (crypto/stark/src/instruments.rs) aligns with this CSV so
phase_table.py can attribute GPU busy%% per phase.

Usage:  nvml_sampler.py -o util.csv [-i 0.1] [-d 0]
Stop with SIGINT/SIGTERM (run_profile.sh kills it when the prove exits).

CSV columns: epoch_ns,gpu_util_pct,mem_util_pct,vram_used_mib,sm_clock_mhz,power_w,temp_c
"""

import argparse
import signal
import subprocess
import sys
import time

STOP = False


def on_sig(_sig, _frm):
    global STOP
    STOP = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-i", "--interval", type=float, default=0.1, help="seconds")
    ap.add_argument("-d", "--device", type=int, default=0)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    query = (
        "utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu"
    )
    cmd = [
        "nvidia-smi",
        f"--id={args.device}",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]

    with open(args.out, "w") as f:
        f.write("epoch_ns,gpu_util_pct,mem_util_pct,vram_used_mib,sm_clock_mhz,power_w,temp_c\n")
        while not STOP:
            t0 = time.time()
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=5
                ).stdout.strip()
            except subprocess.TimeoutExpired:
                out = ""
            ns = time.time_ns()
            if out:
                vals = [v.strip() for v in out.splitlines()[0].split(",")]
                f.write(f"{ns},{','.join(vals)}\n")
                f.flush()
            # keep a steady cadence regardless of nvidia-smi latency
            sleep = args.interval - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
