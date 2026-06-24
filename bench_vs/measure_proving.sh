#!/usr/bin/env bash
#
# Accurate per-step proving measurement for heavy ethrex blocks.
#
# Produces, per block size and per run:
#   - the wall-clock TIMELINE tree (execute / trace_build{p0..p3to5} / air / proving)
#   - the timeline as JSON (for diffing/plotting)               -> $OUT/<size>tx_run<n>.json
#   - peak RSS + swap counters from `/usr/bin/time -v`          -> $OUT/<size>tx_run<n>.log
#
# Hard rule: a run that swapped is INVALID. We swapoff first (if permitted) and
# assert Swaps==0 + ~0 major page faults afterwards; otherwise the run is flagged.
#
# Usage:
#   bash bench_vs/measure_proving.sh [SIZES...] [--runs N] [--threads T]
#   bash bench_vs/measure_proving.sh 2 5                 # default 3 timed runs each
#   bash bench_vs/measure_proving.sh 5 --runs 5 --threads 64
#
# NOTE: peak RSS ~ 6.5 GB per million cpu-ops. Measured: 2tx~62GB, 5tx~122GB,
# 10tx~240GB. On a 128GB box cap at 5tx; 10tx needs ~384GB to stay out of swap.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

SIZES=()
RUNS=3
THREADS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    *) SIZES+=("$1"); shift;;
  esac
done
[ ${#SIZES[@]} -eq 0 ] && SIZES=(2 5)

export SYSROOT_DIR="${SYSROOT_DIR:-$HOME/.lambda-vm-sysroot}"
[ -n "$THREADS" ] && export RAYON_NUM_THREADS="$THREADS"

OUT="$ROOT/bench_vs/results/proving_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo ">>> results: $OUT   sizes=${SIZES[*]}  runs=$RUNS  threads=${RAYON_NUM_THREADS:-all}"

# Total RAM (GB) and a soft check we won't obviously swap.
TOTAL_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "?")
echo ">>> total RAM: ${TOTAL_GB} GB"

# Best-effort disable swap so an over-budget run OOMs cleanly instead of silently
# swapping (swap poisons the timing). Needs privilege; warn if it fails.
if swapoff -a 2>/dev/null; then echo ">>> swap disabled"; else echo ">>> WARN: could not swapoff (need sudo) — will verify Swaps==0 per run instead"; fi

echo ">>> building guest ELF + bench (instruments)"
make executor/program_artifacts/rust/ethrex.elf >/dev/null
CARGO_PROFILE_RELEASE_DEBUG=1 cargo bench -p lambda-vm-prover \
    --bench profile_ethrex --features "parallel,instruments" --no-run 2>&1 | tail -2
BIN=$(ls -t target/release/deps/profile_ethrex-* 2>/dev/null | grep -v '\.d$' | head -1)

for n in "${SIZES[@]}"; do
  fixture="executor/tests/ethrex_${n}_transfers.bin"
  if [ ! -f "$fixture" ]; then
    echo ">>> generating ${n}-tx fixture"
    ( cd tooling/ethrex-fixtures && cargo run --release -- "$n" "$ROOT/$fixture" )
  fi

  echo ">>> ${n}-tx: 1 warmup (discarded) + $RUNS timed"
  "$BIN" "ethrex_${n}_transfers" >/dev/null 2>&1 || { echo "  !! warmup failed (OOM?) — skipping ${n}tx"; continue; }

  for r in $(seq 1 "$RUNS"); do
    log="$OUT/${n}tx_run${r}.log"
    LAMBDA_VM_TIMELINE_JSON="$OUT/${n}tx_run${r}.json" \
      /usr/bin/time -v "$BIN" "ethrex_${n}_transfers" >"$log" 2>&1 || true
    swaps=$(grep -i "Swaps" "$log" | grep -oE "[0-9]+$" || echo "?")
    majf=$(grep -i "Major.*page faults" "$log" | grep -oE "[0-9]+$" || echo "?")
    rss=$(grep -i "Maximum resident" "$log" | grep -oE "[0-9]+$" || echo "?")
    total=$(grep -i "completed in" "$log" | tail -1)
    flag=""; [ "$swaps" != "0" ] && flag=" *** SWAPPED — INVALID ***"
    printf "  run %s: RSS=%sKB swaps=%s majflt=%s  %s%s\n" "$r" "$rss" "$swaps" "$majf" "$total" "$flag"
  done
done

echo
echo ">>> TIMELINE (median-ish: last run per size):"
for n in "${SIZES[@]}"; do
  echo "----- ${n}tx -----"
  awk '/=== TIMELINE/{p=1} p; /^TOTAL|completed in/{if(p)exit}' "$OUT/${n}tx_run${RUNS}.log" 2>/dev/null || true
done
echo ">>> JSON per run in $OUT (diff/plot across sizes & runs)"
