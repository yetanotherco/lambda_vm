#!/usr/bin/env bash
#
# Accurate per-step proving measurement for heavy ethrex blocks.
#
# Per size & run, captures:
#   - wall-clock TIMELINE tree (printed live)
#   - timeline JSON                              -> $OUT/<size>tx_run<n>.json
#   - peak RSS + swap counters (/usr/bin/time -v) -> $OUT/<size>tx_run<n>.log
#
# A run that SWAPPED is INVALID (swap poisons timing). We swapoff if permitted
# and flag any run with Swaps!=0.
#
# Usage:
#   bash bench_vs/measure_proving.sh [SIZES...] [--runs N] [--threads T]
#   bash bench_vs/measure_proving.sh 2 5
#   bash bench_vs/measure_proving.sh 5 --runs 5 --threads 64
#
# Footprint ~6.5 GB / 1M cpu-ops: 2tx~62GB, 5tx~122GB, 10tx~240GB.

# NOTE: intentionally NOT using `set -e` — we handle errors explicitly so one
# failed run never silently kills the whole sweep.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

SIZES=(); RUNS=3; THREADS=""
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

# Pick a peak-memory wrapper.
TIME_BIN=""
if [ -x /usr/bin/time ]; then TIME_BIN=/usr/bin/time
elif command -v gtime >/dev/null 2>&1; then TIME_BIN=$(command -v gtime); fi
[ -n "$TIME_BIN" ] && echo ">>> mem wrapper: $TIME_BIN -v" || echo ">>> WARN: no GNU time; peak RSS/swaps will be unavailable"

if swapoff -a 2>/dev/null; then echo ">>> swap disabled"; else echo ">>> WARN: could not swapoff (need sudo) — will check Swaps per run"; fi

echo ">>> [build] ethrex guest ELF"
if ! make executor/program_artifacts/rust/ethrex.elf; then echo "!! guest ELF build FAILED"; exit 1; fi
echo ">>> [build] instrumented bench"
if ! CARGO_PROFILE_RELEASE_DEBUG=1 cargo bench -p lambda-vm-prover \
      --bench profile_ethrex --features "parallel,instruments" --no-run; then
  echo "!! bench build FAILED"; exit 1
fi
BIN=$(ls -t target/release/deps/profile_ethrex-* 2>/dev/null | grep -v '\.d$' | head -1)
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then echo "!! bench binary not found at target/release/deps/profile_ethrex-*"; exit 1; fi
echo ">>> bench binary: $BIN"

run_once() {  # $1=size  $2=label(log/json basename)
  local n="$1" base="$2"
  local log="$OUT/${base}.log" json="$OUT/${base}.json"
  echo "    [$(date +%H:%M:%S)] running ${base} ..."
  if [ -n "$TIME_BIN" ]; then
    LAMBDA_VM_TIMELINE_JSON="$json" "$TIME_BIN" -v "$BIN" "ethrex_${n}_transfers" >"$log" 2>&1
  else
    LAMBDA_VM_TIMELINE_JSON="$json" "$BIN" "ethrex_${n}_transfers" >"$log" 2>&1
  fi
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "    !! ${base} exited rc=$rc (OOM/kill?) — tail:"
    tail -5 "$log" | sed 's/^/       /'
    return 1
  fi
  return 0
}

for n in "${SIZES[@]}"; do
  echo
  echo "================= ${n}-tx ================="
  fixture="executor/tests/ethrex_${n}_transfers.bin"
  if [ ! -f "$fixture" ]; then
    echo ">>> generating ${n}-tx fixture"
    ( cd tooling/ethrex-fixtures && cargo run --release -- "$n" "$ROOT/$fixture" ) || { echo "!! fixture gen failed"; continue; }
  fi

  echo ">>> warmup (discarded) — this takes the full proving time, be patient"
  if ! run_once "$n" "${n}tx_warmup"; then
    echo "!! ${n}-tx warmup failed — likely exceeds RAM (would swap). Skipping ${n}-tx."
    continue
  fi

  for r in $(seq 1 "$RUNS"); do
    if run_once "$n" "${n}tx_run${r}"; then
      log="$OUT/${n}tx_run${r}.log"
      rss=$(grep -i "Maximum resident" "$log" | grep -oE "[0-9]+" | tail -1)
      swaps=$(grep -i "Swaps" "$log" | grep -oE "[0-9]+$")
      majf=$(grep -i "Major.*page faults" "$log" | grep -oE "[0-9]+$")
      [ -n "$rss" ] && echo "      peak RSS: $(( rss / 1024 / 1024 )) GB   swaps=${swaps:-?}   majflt=${majf:-?}"
      [ "${swaps:-0}" != "0" ] && echo "      *** SWAPPED — RUN INVALID ***"
      # live timeline tree
      sed -n '/=== TIMELINE/,/completed in/p' "$log" | sed 's/^/      /'
    fi
  done
done

echo
echo ">>> done. logs + JSON in: $OUT"
