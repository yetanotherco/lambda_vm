#!/usr/bin/env bash
# Dump the measurement environment as JSON to stdout. Attach one of these to
# every profiling artifact — numbers without their clocks/driver/sha are noise.
set -euo pipefail

q() { # first line of a command's stdout, or "" on failure
  "$@" 2>/dev/null | head -1 || true
}

GIT_SHA="$(q git rev-parse HEAD)"
GIT_DIRTY="$(test -n "$(git status --porcelain 2>/dev/null)" && echo true || echo false)"

SMI_QUERY="name,compute_cap,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit,temperature.gpu,memory.total,memory.free,persistence_mode"
SMI="$(q nvidia-smi "--query-gpu=${SMI_QUERY}" --format=csv,noheader)"

GOVERNOR="$(q cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
PARANOID="$(q cat /proc/sys/kernel/perf_event_paranoid)"

python3 - "$GIT_SHA" "$GIT_DIRTY" "$SMI" "$GOVERNOR" "$PARANOID" <<'PY'
import json, os, platform, shutil, subprocess, sys, datetime

sha, dirty, smi, governor, paranoid = sys.argv[1:6]
def ver(cmd, *args):
    if not shutil.which(cmd):
        return None
    try:
        out = subprocess.run([cmd, *args], capture_output=True, text=True, timeout=10)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception:
        return None

smi_fields = [s.strip() for s in smi.split(",")] if smi else []
smi_keys = ["gpu", "compute_cap", "driver", "sm_clock", "sm_clock_max",
            "mem_clock", "power_limit", "temp_c", "vram_total", "vram_free",
            "persistence_mode"]

print(json.dumps({
    "date": datetime.datetime.now().isoformat(timespec="seconds"),
    "hostname": platform.node(),
    "kernel": platform.release(),
    "git_sha": sha,
    "git_dirty": dirty == "true",
    "gpu": dict(zip(smi_keys, smi_fields)) if smi_fields else None,
    "cpu_governor": governor or None,
    "perf_event_paranoid": paranoid or None,
    "nvcc": ver("nvcc", "--version"),
    "nsys": ver("nsys", "--version"),
    "ncu": ver("ncu", "--version"),
    "rustc": ver("rustc", "--version"),
    "env": {k: v for k, v in os.environ.items()
            if k.startswith(("LAMBDA_VM_", "CUDARC_", "TABLE_PARALLELISM", "CUDA_"))},
}, indent=2))
PY
