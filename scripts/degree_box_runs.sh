#!/usr/bin/env bash
#
# degree_box_runs.sh — DEGREE-LANE EXPERIMENT (temporary, not for merge).
#
# Two modes, deliberately separated because they have OPPOSITE scheduling
# requirements. Running the wrong one under load silently ruins it.
#
#   guest   CONTENTION-IMMUNE. Deliverables are guest cycles and keccak calls:
#           retired-instruction counts for a fixed (ELF, blob), i.e. exact
#           integers that do not move with CPU load. Safe to run concurrently
#           with other work on a busy box; only the wall estimate stretches.
#
#   timing  IDLE BOX REQUIRED. Deliverables are wall clock and peak RSS. These
#           are NOT contention-immune: a paired ABBA against a 60-core
#           neighbour measures the neighbour, not the arm. Do not start this
#           while anything else heavy is running.
#
# Usage:
#   scripts/degree_box_runs.sh guest   [OUTDIR]
#   scripts/degree_box_runs.sh timing  [OUTDIR]
#
# Arm SHAs (signed; the pair differs in exactly one line of prover/src/lib.rs):
#   D3_SHA  bb3da304  VM_MAX_DEGREE=3
#   D7_SHA  a3cd3c0c  VM_MAX_DEGREE=7
#   FIXED   21dacd34  fixed-cells A/B/C arm
# Both arms are PRE-PR-949 (the IR pre-capture gate), so the RATIO is internally
# valid; only absolute cycle counts would shift if 949 lands. Stamp that on
# every number.

set -uo pipefail

MODE="${1:-}"
OUT="${2:-$HOME/degree_runs_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"

D3_SHA=bb3da304
D7_SHA=a3cd3c0c
FIXED_SHA=21dacd34

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/run.log"; }

case "$MODE" in
guest)
  log "=== GUEST-CYCLE ARMS (contention-immune; concurrent OK) ==="
  log "REF_A=$D7_SHA (d=7 declared, 6 parts)  REF_B=$D3_SHA (d=3, 2 parts)  preset=blowup4"

  # Fail fast rather than let the guest build die deep in build-std.
  : "${SYSROOT_DIR:=$HOME/.lambda-vm-sysroot}"
  if [ ! -d "$SYSROOT_DIR" ]; then
    log "FATAL: SYSROOT_DIR=$SYSROOT_DIR missing — build it first (worktree setup recipe)."
    exit 1
  fi
  export SYSROOT_DIR
  # Shared across the two ref worktrees: build-std is the expensive part and
  # this roughly halves the second guest build.
  export GUEST_TARGET_DIR="${GUEST_TARGET_DIR:-/tmp/degree_guest_td}"
  export HOST_TARGET_DIR="${HOST_TARGET_DIR:-/tmp/degree_host_td}"

  timeout 10800 scripts/bench_recursion_cycles.sh "$D7_SHA" "$D3_SHA" blowup4 \
    2>&1 | tee -a "$OUT/guest_blowup4.log"
  log "guest arms exit=$?"

  # The measurements themselves, kept next to the log so one scp takes it all.
  cp -v /tmp/recursion_cycles_run/result_* "$OUT/" 2>/dev/null || true
  log "=== GUEST DONE — collect $OUT ==="
  ;;

timing)
  log "=== TIMING ARMS — REQUIRES AN IDLE BOX ==="
  # Loud, because a paired ABBA run against a busy box produces numbers that
  # look fine and mean nothing.
  loadavg=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
  log "1-minute load average: $loadavg"
  if awk "BEGIN{exit !($loadavg > 4)}"; then
    log "WARNING: load average $loadavg suggests the box is NOT idle."
    log "Wall-clock and peak-RSS arms are meaningless under contention."
    if [ "${FORCE_BUSY:-0}" != "1" ]; then
      log "Refusing to start. Re-run with FORCE_BUSY=1 only if you know the box is free."
      exit 1
    fi
    log "FORCE_BUSY=1 set — proceeding, but treat every timing number as suspect."
  fi

  log "--- A: fixed-cells constraint-degree ladder (SHA $FIXED_SHA) ---"
  git checkout -q "$FIXED_SHA" || { log "FATAL: cannot check out $FIXED_SHA"; exit 1; }
  for R in 18 20 22; do
    log "rows_log2=$R"
    LVM_DEGREE_ROWS_LOG2=$R LVM_DEGREE_BLOWUP=4 LVM_DEGREE_REPS=3 \
      timeout 3600 cargo test -p stark --release degree_fixed_cells_sweep -- \
        --ignored --nocapture --test-threads=1 2>&1 \
      | grep '^DEGREEFIXED' | tee -a "$OUT/fixed_cells.log"
  done

  log "--- A': fixed-cells peak RSS, ONE ARM PER PROCESS ---"
  BIN=$(cargo test -p stark --release --no-run --message-format=json 2>/dev/null |
    python3 -c '
import sys, json
for l in sys.stdin:
    try: m = json.loads(l)
    except Exception: continue
    if (m.get("reason") == "compiler-artifact"
            and m.get("target", {}).get("name") == "stark"
            and m.get("profile", {}).get("test")):
        print(m["executable"])
')
  if [ -n "${BIN:-}" ]; then
    for ARM in A C B; do
      LVM_DEGREE_ARM=$ARM LVM_DEGREE_REPS=1 LVM_DEGREE_ROWS_LOG2=22 LVM_DEGREE_BLOWUP=4 \
        timeout 1800 /usr/bin/time -v "$BIN" degree_fixed_cells_sweep --ignored --nocapture 2>&1 |
        grep -E '^DEGREEFIXED|Maximum resident set size' | tee -a "$OUT/fixed_cells_rss.log"
    done
  else
    log "WARNING: could not locate the stark test binary; RSS arms skipped."
  fi

  log "--- B: VM prover sweep B0-B5 + C0 (rebuilds per degree) ---"
  # The sweep now builds/verifies the ASM artifact itself and aborts on the
  # first unmeasurable arm; a fresh clone has no executor/program_artifacts/.
  timeout 14400 scripts/degree_prover_sweep.sh all_instructions_64 3 \
    2>&1 | tee -a "$OUT/prover_sweep.log"

  log "=== TIMING DONE — collect $OUT ==="
  ;;

*)
  echo "usage: $0 guest|timing [OUTDIR]" >&2
  echo "  guest  = contention-immune, concurrent OK" >&2
  echo "  timing = requires an IDLE box" >&2
  exit 2
  ;;
esac
