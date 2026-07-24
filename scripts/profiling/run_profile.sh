#!/usr/bin/env bash
# One command -> a complete profiling bundle for a workload:
#   * N instrumented prove runs (timeline JSON + GPU-util CSV + prove time)
#   * aggregated phase table (phase_table.py), with per-phase GPU util
#   * optionally one nsys-traced run + per-phase GPU-busy report
#   * capture_env.sh JSON alongside everything
#
# Usage:
#   scripts/profiling/run_profile.sh [options] <elf> [--private-input <bin>]
# Options:
#   --runs N          instrumented runs (default 3; run 1 is additionally
#                     saved as "cold" — module load, twiddle build, mempool
#                     growth make it slower by construction)
#   --nsys            add one run under `nsys profile` + sqlite phase report
#   --gpu-metrics     add --gpu-metrics-devices=0 to nsys (needs profiling perms)
#   --continuations   prove with --continuations
#   --out DIR         output dir (default reports/<elf>_<sha>_<timestamp>)
#   --no-build        reuse ./target/release/cli as-is
# Env:
#   PROFILE_FEATURES  cli features (default "nvtx,jemalloc-stats")
#   EXTRA_PROVE_ARGS  appended to every `cli prove`
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
HERE="$ROOT/scripts/profiling"

RUNS=3 NSYS=0 GPU_METRICS=0 CONT=0 OUT="" BUILD=1 ELF="" INPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --nsys) NSYS=1; shift ;;
    --gpu-metrics) GPU_METRICS=1; shift ;;
    --continuations) CONT=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    --private-input) INPUT="$2"; shift 2 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) ELF="$1"; shift ;;
  esac
done
[[ -n "$ELF" && -f "$ELF" ]] || { echo "usage: run_profile.sh [options] <elf> [--private-input <bin>]" >&2; exit 2; }

SHA="$(git rev-parse --short HEAD)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-reports/$(basename "${ELF%.elf}")_${SHA}_${STAMP}}"
mkdir -p "$OUT"

FEATURES="${PROFILE_FEATURES:-nvtx,jemalloc-stats}"
PROVE_ARGS=(prove "$ELF" -o "$OUT/proof.bin" --time)
[[ -n "$INPUT" ]] && PROVE_ARGS+=(--private-input "$INPUT")
[[ "$CONT" == 1 ]] && PROVE_ARGS+=(--continuations)
# shellcheck disable=SC2206  # intentional word-split of extra args
PROVE_ARGS+=(${EXTRA_PROVE_ARGS:-})

if [[ "$BUILD" == 1 ]]; then
  echo "==> Building cli (features: $FEATURES, release+debuginfo+frame-pointers)"
  CARGO_PROFILE_RELEASE_DEBUG=1 RUSTFLAGS="${RUSTFLAGS:-} -Cforce-frame-pointers=yes" \
    cargo build --release -p cli --features "$FEATURES"
fi
CLI="$ROOT/target/release/cli"

"$HERE/capture_env.sh" > "$OUT/env.json"
echo "==> Environment captured to $OUT/env.json"

TIMELINES=() UTILS=()
for i in $(seq 1 "$RUNS"); do
  TL="$OUT/timeline_run${i}.json"
  UT="$OUT/util_run${i}.csv"
  python3 "$HERE/nvml_sampler.py" -o "$UT" -i 0.1 & SAMPLER=$!
  echo "==> Run $i/$RUNS"
  LAMBDA_VM_TIMELINE_JSON="$TL" "$CLI" "${PROVE_ARGS[@]}" | tee "$OUT/run${i}.log" \
    | grep -E "Proving time|TIMELINE" || true
  kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null || true
  rm -f "$OUT/proof.bin"
  if [[ ! -f "$TL" ]]; then
    echo "    WARNING: run $i produced no timeline JSON (binary built without instruments?)"
  elif [[ "$i" == 1 ]]; then
    cp "$TL" "$OUT/timeline_cold.json"   # run 1 = cold (module load, twiddles, mempool)
  else
    TIMELINES+=("$TL"); UTILS+=(--util "$UT")
  fi
done

if [[ ${#TIMELINES[@]} -gt 0 ]]; then
  python3 "$HERE/phase_table.py" "${UTILS[@]}" "${TIMELINES[@]}" > "$OUT/phase_table.md"
  echo "==> Warm phase table: $OUT/phase_table.md"
  python3 "$HERE/phase_table.py" "$OUT/timeline_cold.json" > "$OUT/phase_table_cold.md"
  python3 "$HERE/timeline_to_perfetto.py" "${TIMELINES[0]}" > "$OUT/trace_perfetto.json"
fi

if [[ "$NSYS" == 1 ]]; then
  command -v nsys >/dev/null || { echo "ERROR: nsys not installed (setup_machine.sh)"; exit 1; }
  NSYS_ARGS=(profile -t cuda,nvtx,osrt --cuda-memory-usage=true -o "$OUT/nsys_report")
  [[ "$GPU_METRICS" == 1 ]] && NSYS_ARGS+=(--gpu-metrics-devices=0)
  if [[ -n "${LAMBDA_VM_NSYS_CAPTURE_SPAN:-}" ]]; then
    NSYS_ARGS+=(--capture-range=cudaProfilerApi --capture-range-end=stop)
    echo "==> nsys capture gated on span '$LAMBDA_VM_NSYS_CAPTURE_SPAN'"
  fi
  echo "==> nsys run"
  nsys "${NSYS_ARGS[@]}" "$CLI" "${PROVE_ARGS[@]}" > "$OUT/nsys_run.log" 2>&1 || {
    tail -20 "$OUT/nsys_run.log"; exit 1; }
  rm -f "$OUT/proof.bin"
  nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum \
    "$OUT/nsys_report.nsys-rep" > "$OUT/nsys_stats.txt" 2>&1 || true
  # --force-overwrite: `nsys stats` above already materializes the sqlite
  nsys export --type sqlite --force-overwrite true -o "$OUT/nsys_report.sqlite" "$OUT/nsys_report.nsys-rep"
  python3 "$HERE/nsys_phase_busy.py" "$OUT/nsys_report.sqlite" > "$OUT/phase_busy.md"
  echo "==> GPU phase-busy report: $OUT/phase_busy.md"
fi

echo
echo "==> Bundle: $OUT"
ls -la "$OUT"
