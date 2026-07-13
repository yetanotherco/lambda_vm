#!/usr/bin/env bash
#
# bench_abba_gpu.sh — interleaved A/B/B/A of the GPU composition path (ON vs OFF)
# on the ethrex workload, using ONE cuda `cli` binary and the
# LAMBDA_VM_DISABLE_GPU_COMPOSITION runtime toggle.
#
# WHY a toggle instead of two refs (as in bench_abba.sh): the change under test
# is a single feature-gated branch (round-2 constraint eval on GPU vs CPU). One
# binary flipped by an env var isolates exactly that branch with zero build/ref
# differences — no compiler/codegen drift between the two sides.
#
#   A = GPU composition path ON  (LAMBDA_VM_DISABLE_GPU_COMPOSITION unset)
#   B = GPU composition path OFF (=1 -> CPU per-row accumulation)
#
# CONVENTION: reported % = (A - B)/B = (GPU_on - CPU)/CPU.  NEGATIVE = GPU faster.
#
# USAGE:  scripts/bench_abba_gpu.sh [N_PAIRS=10] [TX_COUNT=20]
#   Env:  REBUILD=1        force a cli rebuild
#         CUDARC_PIN=<ver> pin math-cuda's cudarc to a CUDA version (rented-box
#                          driver may lack cudarc-latest symbols)
#         BENCH_FEATURES   cli features (default: jemalloc-stats,prover/cuda)

set -euo pipefail
N_PAIRS="${1:-10}"
TX_COUNT="${2:-20}"
BENCH_FEATURES="${BENCH_FEATURES:-jemalloc-stats,prover/cuda}"

ELF_REL="executor/program_artifacts/rust/ethrex.elf"
INPUT_REL="executor/tests/ethrex_${TX_COUNT}_transfers.bin"
WORK="/tmp/abba_gpu_run"
PROOF="/tmp/abba_gpu_proof.bin"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
# shellcheck source=scripts/lib/bench_abba_common.sh
. "$ROOT/scripts/lib/bench_abba_common.sh"
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 required." >&2; exit 1; }
mkdir -p "$WORK"

[ -f "$ELF_REL" ] || { echo "ERROR: $ELF_REL missing (copy the ethrex guest ELF in)." >&2; exit 1; }
if [ ! -f "$INPUT_REL" ]; then
  echo "==> Generating ethrex ${TX_COUNT}-transfer fixture"
  ( cd tooling/ethrex-fixtures && cargo build --release )
  tooling/ethrex-fixtures/target/release/ethrex-fixtures "$TX_COUNT" "$INPUT_REL" distinct
fi
ELF="$ROOT/$ELF_REL"
INPUT="$ROOT/$INPUT_REL"

if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$WORK/cli" ]; then
  # CUDARC_PIN compat shim (no-op with a warning on post-pin checkouts) — see
  # cudarc_pin_apply in scripts/lib/bench_abba_common.sh.
  cudarc_pin_apply crypto/math-cuda/Cargo.toml
  echo "==> Building cli (features: $BENCH_FEATURES)"
  # Restore the tracked Cargo.toml whether the build succeeds or fails, so a
  # build error (set -e) can't leave the sed edit above in the working tree.
  build_rc=0
  cargo build --release -p cli --features "$BENCH_FEATURES" || build_rc=$?
  [ -n "${CUDARC_PIN:-}" ] && git checkout -- crypto/math-cuda/Cargo.toml
  if [ "$build_rc" -ne 0 ]; then
    echo "ERROR: cargo build failed (rc=$build_rc)." >&2
    exit "$build_rc"
  fi
  cp target/release/cli "$WORK/cli"
else
  echo "==> Reusing cached cli (REBUILD=1 to force)"
fi

run_prove() {  # $1 = 0|1 (disable-flag) -> proving time (s)
  local out
  out="$(LAMBDA_VM_DISABLE_GPU_COMPOSITION="$1" "$WORK/cli" prove "$ELF" \
        --private-input "$INPUT" -o "$PROOF" --time 2>&1)"
  rm -f "$PROOF"
  extract_prove_time "$out"
}

# Warm-up (PTX load, pools, pinned alloc) so pair 1 isn't an outlier.
echo "==> Warm-up prove"
run_prove 0 >/dev/null

echo "==> $N_PAIRS pairs, ethrex ${TX_COUNT} txs  (A=GPU-comp ON, B=OFF; - = GPU faster)"
printf 'pair,a_time,b_time\n' > "$WORK/pairs.csv"
for i in $(seq 1 "$N_PAIRS"); do
  if [ $((i % 2)) -eq 1 ]; then
    a="$(run_prove 0)"; b="$(run_prove 1)"        # odd: A then B
  else
    b="$(run_prove 1)"; a="$(run_prove 0)"        # even: B then A (ABBA)
  fi
  printf '%d,%s,%s\n' "$i" "$a" "$b" >> "$WORK/pairs.csv"
  printf '   pair %2d/%d  A(GPU)=%ss  B(CPU)=%ss  %+.2f%%\n' \
    "$i" "$N_PAIRS" "$a" "$b" "$(awk "BEGIN{print ($a-$b)/$b*100}")"
done

# Shared paired analysis (paired-t + exact Wilcoxon + stability + verdict) —
# same statistics as bench_abba.sh, by construction.
abba_stats "$WORK/pairs.csv" "GPU" "CPU" "GPU-composition ABBA"
