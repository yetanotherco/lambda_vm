#!/usr/bin/env bash
# Put the box in (or out of) a known measurement state: locked GPU clocks,
# persistence mode, performance CPU governor. Run before every profiling
# session; numbers taken with floating clocks are not comparable across runs.
#
# Usage:
#   sudo scripts/profiling/bench_mode.sh on [sm_clock_mhz]
#   sudo scripts/profiling/bench_mode.sh off
#
# Default lock clock = 90% of the GPU's max SM clock (sustainable without
# thermal throttle on most chassis; pass an explicit MHz to override). Note on
# GeForce: -lgc (graphics/SM clock) works, -lmc (memory clock) is often
# rejected — we lock what we can and *record* the rest via capture_env.sh.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "on" && "$MODE" != "off" ]]; then
  echo "usage: bench_mode.sh on [sm_clock_mhz] | off" >&2
  exit 2
fi

if [[ "$MODE" == "off" ]]; then
  nvidia-smi -rgc || true
  if command -v cpupower >/dev/null; then
    cpupower frequency-set -g schedutil >/dev/null || \
      cpupower frequency-set -g ondemand >/dev/null || true
  fi
  echo "bench mode OFF (clocks unlocked, governor restored)"
  exit 0
fi

nvidia-smi -pm 1 >/dev/null

MAX_SM="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')"
CLOCK="${2:-$(( MAX_SM * 90 / 100 ))}"
nvidia-smi -lgc "${CLOCK},${CLOCK}"
# Memory clock lock: best effort (GeForce usually refuses).
nvidia-smi -lmc "$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1 | tr -d ' ')" \
  2>/dev/null || echo "note: -lmc rejected (normal on GeForce); memory clock floats — recorded by capture_env.sh"

if command -v cpupower >/dev/null; then
  cpupower frequency-set -g performance >/dev/null || true
elif ls /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor >/dev/null 2>&1; then
  # no cpupower (e.g. Debian without linux-cpupower): set governor via sysfs
  echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
else
  echo "note: no cpufreq control available; CPU governor unchanged" >&2
fi

echo "bench mode ON: SM clock locked to ${CLOCK} MHz (max ${MAX_SM}), persistence on, governor=performance"
echo "verify no throttling during long sessions: nvidia-smi -q -d PERFORMANCE | grep -A6 'Clocks Event Reasons'"
