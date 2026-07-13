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
  # CUDARC_PIN: compat shim for benching *pre-pin* checkouts. Newer trees pin
  # cudarc's CUDA version permanently in crypto/math-cuda/Cargo.toml, so the
  # `cuda-version-from-build-system` anchor is gone and this sed no-ops on them.
  if [ -n "${CUDARC_PIN:-}" ]; then
    if grep -q '"cuda-version-from-build-system"' crypto/math-cuda/Cargo.toml; then
      sed -i "s/\"cuda-version-from-build-system\"/\"${CUDARC_PIN}\"/; /\"fallback-latest\"/d" \
        crypto/math-cuda/Cargo.toml
      echo "    cudarc pinned to ${CUDARC_PIN}"
    else
      # Post-pin checkouts already hard-pin cudarc in Cargo.toml, so the anchor
      # is gone and the sed would silently no-op. Warn rather than mislead.
      echo "    WARNING: CUDARC_PIN=${CUDARC_PIN} ignored — cudarc is already" >&2
      echo "             pinned in crypto/math-cuda/Cargo.toml (no build-system anchor to rewrite)." >&2
    fi
  fi
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
  local out t
  out="$(LAMBDA_VM_DISABLE_GPU_COMPOSITION="$1" "$WORK/cli" prove "$ELF" \
        --private-input "$INPUT" -o "$PROOF" --time 2>&1)"
  rm -f "$PROOF"
  t="$(printf '%s\n' "$out" | grep -o 'Proving time: [0-9.]*' | awk '{print $3}')"
  if [ -z "$t" ]; then
    echo "ERROR: could not parse 'Proving time':" >&2; printf '%s\n' "$out" >&2; exit 1
  fi
  echo "$t"
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

python3 - "$WORK/pairs.csv" <<'PY'
import sys, csv, math
rows = list(csv.DictReader(open(sys.argv[1])))
A = [float(r['a_time']) for r in rows]   # GPU composition ON
B = [float(r['b_time']) for r in rows]   # GPU composition OFF (CPU)
n = len(A)
d = [(a - b) / b * 100.0 for a, b in zip(A, B)]
mean = sum(d) / n
var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
sd = math.sqrt(var); se = sd / math.sqrt(n) if n else float('inf')
TT = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,
      10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,
      18:2.101,19:2.093,20:2.086,25:2.060,30:2.042,40:2.021,50:2.009,60:2.000}
df = n - 1
tc = TT.get(df) or (1.96 if df > 120 else TT[min(TT, key=lambda k: abs(k - df))])
lo, hi = mean - tc * se, mean + tc * se
def median(xs):
    s = sorted(xs); m = len(s)
    return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2
med = median(d)
print("\n=== GPU-composition ABBA  (A=GPU ON, B=CPU; - = GPU faster) ===")
print(f"  pairs: {n}   mean A (GPU): {sum(A)/n:.3f}s   mean B (CPU): {sum(B)/n:.3f}s")
print(f"  paired-t  mean {mean:+.2f}%   sd {sd:.2f}%   se {se:.2f}%   95% CI [{lo:+.2f}%, {hi:+.2f}%]")
print(f"  median    {med:+.2f}%")
if hi < 0:
    print(f"  => GPU path faster by ~{-mean:.2f}% (CI below 0)")
elif lo > 0:
    print(f"  => GPU path slower by ~{mean:.2f}% (CI above 0)")
else:
    print(f"  => inconclusive at n={n} (CI straddles 0); point ~{med:+.2f}%")
PY
