#!/bin/bash
# GPU profiling harness: nsys timeline of the proving section (ethrex workload).
#
# Runs `cli prove` on the ethrex guest under Nsight Systems with CUDA + NVTX
# tracing. The capture window is opened by cuProfilerStart at the start of the
# `proving` phase (see prover/src/lib.rs), so CPU-side execution + trace build
# are excluded from the profile.
#
# Designed for rented GPU boxes (Vast.ai) WITHOUT sudo:
#   * nsys is located in PATH / the CUDA toolkit, or downloaded as a .deb and
#     extracted into $HOME with `dpkg -x` (no root needed).
#   * If perf_event_paranoid blocks CPU sampling, sampling is disabled
#     automatically (CUDA + NVTX tracing does not need perf events).
#   * Nsight Compute (ncu) needs GPU perf counters, which the host driver
#     usually restricts (RmProfilingAdminOnly=1) — reported, not required.
#
# Usage:
#   scripts/profile_gpu.sh                 # ethrex 5-tx block
#   TX_COUNT=10 scripts/profile_gpu.sh     # bigger block
#   CONTINUATIONS=1 scripts/profile_gpu.sh # continuation mode (large workloads)
#   NO_NSYS=1 scripts/profile_gpu.sh       # instruments-only run (no profiler)
#
# Env knobs:
#   TX_COUNT        ethrex transfer count (default 5)
#   OUT_DIR         output directory (default /tmp/gpu_profile/<timestamp>)
#   FEATURES        cli build features (default jemalloc-stats,nvtx,prover/cuda)
#   CONTINUATIONS   1 -> prove with --continuations --epoch-size-log2 $EPOCH_SIZE_LOG2
#   EPOCH_SIZE_LOG2 epoch size for continuation mode (default 20)
#   NSYS_BIN        path to nsys (skips discovery)
#   NSYS_DEB_URL    override URL for the user-space nsys install
#   NO_NSYS         1 -> skip nsys, just run the instrumented prove
#
# Outputs in $OUT_DIR:
#   prove_ethrex_<N>tx.nsys-rep   timeline (open in nsys-ui locally)
#   prove_ethrex_<N>tx.sqlite     exported DB (for scripts/analyze_nsys.py)
#   stats_*.csv                   nvtx/kernel/memcpy/api summaries
#   timeline.json                 host-side span tree (instruments)
#   prove.log                     cli output (timing report)

set -euo pipefail

TX_COUNT="${TX_COUNT:-5}"
FEATURES="${FEATURES:-jemalloc-stats,nvtx,prover/cuda}"
CONTINUATIONS="${CONTINUATIONS:-0}"
EPOCH_SIZE_LOG2="${EPOCH_SIZE_LOG2:-20}"
NO_NSYS="${NO_NSYS:-0}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
OUT_DIR="${OUT_DIR:-/tmp/gpu_profile/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

if [ "$CONTINUATIONS" = "1" ]; then
  CONT_ARGS="--continuations --epoch-size-log2 $EPOCH_SIZE_LOG2"
else
  CONT_ARGS=""
fi

# --- 0. Environment sanity -------------------------------------------------

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found — this script must run on a GPU box." >&2
  exit 1
fi
echo "==> GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"

PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 4)"
SAMPLING_ARGS=""
if [ "$PARANOID" -gt 2 ]; then
  echo "==> perf_event_paranoid=$PARANOID (>2): disabling nsys CPU sampling (CUDA+NVTX tracing unaffected)"
  SAMPLING_ARGS="--sample=none --cpuctxsw=none"
fi

if grep -q 'RmProfilingAdminOnly: 1' /proc/driver/nvidia/params 2>/dev/null; then
  echo "==> NOTE: GPU perf counters are admin-only on this host (RmProfilingAdminOnly=1)."
  echo "    nsys timelines work fine; ncu per-kernel deep dives will NOT (ERR_NVGPUCTRPERM)."
fi

# --- 1. Locate or user-install nsys -----------------------------------------

find_nsys() {
  if [ -n "${NSYS_BIN:-}" ] && [ -x "$NSYS_BIN" ]; then echo "$NSYS_BIN"; return; fi
  if command -v nsys >/dev/null 2>&1; then command -v nsys; return; fi
  local c
  for c in /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems*/bin/nsys \
           "$HOME"/.local/nsight-systems/opt/nvidia/nsight-systems*/*/target-linux-x64/nsys \
           "$HOME"/.local/nsight-systems/opt/nvidia/nsight-systems*/*/bin/nsys; do
    if [ -x "$c" ]; then echo "$c"; return; fi
  done
  echo ""
}

install_nsys_userspace() {
  # Extract the nsight-systems-cli .deb into $HOME — no root required.
  local dest="$HOME/.local/nsight-systems"
  local cache="$HOME/.cache/lambda-nsys"
  mkdir -p "$dest" "$cache"
  local deb=""
  echo "==> nsys not found; attempting user-space install into $dest" >&2
  if [ -n "${NSYS_DEB_URL:-}" ]; then
    ( cd "$cache" && curl -fLO "$NSYS_DEB_URL" )
    deb="$(ls -t "$cache"/*.deb 2>/dev/null | head -1)"
  elif command -v apt-get >/dev/null 2>&1; then
    # `apt-get download` needs no root; Vast CUDA images ship the NVIDIA repo.
    ( cd "$cache" && apt-get download nsight-systems-cli >/dev/null 2>&1 ) || true
    deb="$(ls -t "$cache"/nsight-systems-cli*.deb 2>/dev/null | head -1)"
  fi
  if [ -z "$deb" ]; then
    echo "ERROR: could not obtain an nsys .deb. Set NSYS_DEB_URL to e.g." >&2
    echo "  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-cli-<ver>_amd64.deb" >&2
    echo "or rerun with NO_NSYS=1 for an instruments-only run." >&2
    exit 1
  fi
  dpkg -x "$deb" "$dest"
}

NSYS=""
if [ "$NO_NSYS" != "1" ]; then
  NSYS="$(find_nsys)"
  if [ -z "$NSYS" ]; then
    install_nsys_userspace
    NSYS="$(find_nsys)"
    [ -n "$NSYS" ] || { echo "ERROR: nsys still not found after extraction." >&2; exit 1; }
  fi
  echo "==> nsys: $NSYS ($("$NSYS" --version 2>/dev/null | head -1 || true))"
fi

# --- 2. Build the instrumented cli ------------------------------------------

echo "==> Building cli (features: $FEATURES)"
cargo build --release -p cli --features "$FEATURES"
CLI="$ROOT/target/release/cli"

# --- 3. Guest ELF + fixture (same flow as bench_abba.sh) ---------------------

ELF_REL="executor/program_artifacts/rust/ethrex.elf"
INPUT_REL="executor/tests/ethrex_${TX_COUNT}_transfers.bin"
if [ ! -f "$ELF_REL" ]; then
  echo "==> Building ethrex guest ELF (missing)"
  export SYSROOT_DIR="${SYSROOT_DIR:-$HOME/.lambda-vm-sysroot}"
  make "$ELF_REL"
fi
if [ ! -f "$INPUT_REL" ]; then
  echo "==> Generating ethrex ${TX_COUNT}-transfer fixture (missing)"
  ( cd tooling/ethrex-fixtures && cargo build --release )
  tooling/ethrex-fixtures/target/release/ethrex-fixtures "$TX_COUNT" "$INPUT_REL" distinct
fi
ELF="$ROOT/$ELF_REL"
INPUT="$ROOT/$INPUT_REL"

# --- 4. Warm-up run (JIT/caches/mempool; also validates the workload) --------

echo "==> Warm-up prove (untraced; includes in-process GPU event timing)"
LAMBDA_VM_TIMELINE_JSON="$OUT_DIR/timeline.json" \
LAMBDA_VM_GPU_TIMELINE=1 \
LAMBDA_VM_GPU_TIMELINE_JSON="$OUT_DIR/gpu_timeline_chrome.json" \
  "$CLI" prove "$ELF" --private-input "$INPUT" -o "$OUT_DIR/proof.bin" --time $CONT_ARGS \
  >"$OUT_DIR/prove.log" 2>&1 || { tail -30 "$OUT_DIR/prove.log" >&2; exit 1; }
grep -o 'Proving time: [0-9.]*' "$OUT_DIR/prove.log" | tail -1 || true

if [ "$NO_NSYS" = "1" ]; then
  echo "==> NO_NSYS=1: done. Instruments report in $OUT_DIR/prove.log, spans in timeline.json"
  exit 0
fi

# --- 5. Profiled run ----------------------------------------------------------

REP="$OUT_DIR/prove_ethrex_${TX_COUNT}tx"
echo "==> nsys profile -> $REP.nsys-rep"
# -c cudaProfilerApi: capture starts/stops at cuProfilerStart/Stop around the
# proving section, so execution + trace build are outside the recording.
# Continuation mode proves per epoch and does not hit the monolithic-path
# profiler bracket, so capture the whole process there — the epoch-interleaved
# CPU (execute/trace-build) vs GPU (prove) alternation is itself the picture.
if [ "$CONTINUATIONS" = "1" ]; then
  CAPTURE_ARGS=""
else
  CAPTURE_ARGS="-c cudaProfilerApi --capture-range-end=stop"
fi
# shellcheck disable=SC2086
"$NSYS" profile \
  -t cuda,nvtx,osrt $SAMPLING_ARGS $CAPTURE_ARGS \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$REP" \
  "$CLI" prove "$ELF" --private-input "$INPUT" -o "$OUT_DIR/proof.bin" --time $CONT_ARGS \
  >"$OUT_DIR/prove_profiled.log" 2>&1 || { tail -30 "$OUT_DIR/prove_profiled.log" >&2; exit 1; }
grep -o 'Proving time: [0-9.]*' "$OUT_DIR/prove_profiled.log" | tail -1 || true

# --- 6. Summaries + sqlite export ---------------------------------------------

echo "==> nsys stats"
"$NSYS" stats \
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv --output "$OUT_DIR/stats" "$REP.nsys-rep" >/dev/null 2>&1 || \
  echo "WARNING: nsys stats failed (older nsys?); the .nsys-rep is still usable." >&2

echo "==> nsys export --type sqlite"
"$NSYS" export --type sqlite --force-overwrite=true -o "$REP.sqlite" "$REP.nsys-rep" >/dev/null 2>&1 || \
  echo "WARNING: sqlite export failed; analyze_nsys.py needs it." >&2

if [ -f "$ROOT/scripts/analyze_nsys.py" ] && [ -f "$REP.sqlite" ] && command -v python3 >/dev/null 2>&1; then
  echo "==> analyze_nsys.py"
  python3 "$ROOT/scripts/analyze_nsys.py" "$REP.sqlite" | tee "$OUT_DIR/bottlenecks.md" || true
fi

echo ""
echo "==> Done. Outputs in $OUT_DIR:"
ls -lh "$OUT_DIR" | sed 's/^/    /'
echo ""
echo "    To inspect the timeline locally:  scp <box>:$REP.nsys-rep . && nsys-ui $(basename "$REP").nsys-rep"
