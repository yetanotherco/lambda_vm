#!/usr/bin/env bash
#
# bench_abba.sh — interleaved A/B/B/A paired prover benchmark.
#
# WHY: comparing a PR against a separately-recorded (cached) baseline conflates the
# code delta with machine drift between the two measurement sessions. For small
# (~1%) prover changes that drift is the dominant error. Measuring both binaries
# *interleaved on the same machine in the same session* cancels the drift (it hits
# both sides equally), and a paired analysis over the A/B pairs is far more powerful
# than an unpaired two-sample test.
#
# WHAT IT DOES:
#   1. Builds the ethrex guest ELF + 5-transfer fixture once (identical for both
#      sides — a prover-only change doesn't touch the guest).
#   2. Builds the `cli` prover at REF_A and REF_B (skips the build and reuses the
#      cached binaries if they already exist; set REBUILD=1 to force).
#   3. Runs N_PAIRS interleaved pairs in A B B A ... order (alternating which side
#      runs first each pair, to cancel linear drift). Use an EVEN N_PAIRS.
#   4. Reports BOTH a paired-t 95% CI (sensitive to outliers) AND a robust
#      median + Wilcoxon signed-rank result (shrugs off transient slow runs).
#
# CONVENTION: reported % = (PR - baseline)/baseline, matching the classic /bench.
# NEGATIVE = PR FASTER (improvement); positive = regression.
#
# USAGE:
#   scripts/bench_abba.sh REF_A [REF_B] [N_PAIRS]
#     REF_A    REQUIRED — ref or SHA to evaluate (the PR side)
#     REF_B    baseline   (default: origin/main)
#     N_PAIRS  pairs      (default: 20 -> 40 runs, ~33 min on ethrex)
#   Env: REBUILD=1 forces a rebuild even if cached binaries exist.
#        BENCH_FEATURES=<list> cargo features for the cli build (default: jemalloc-stats).
#          The GPU ABBA workflow passes "jemalloc-stats,prover/cuda" to bench the GPU path.
#        CONTINUATIONS=1 proves with --continuations (epochs; flat peak memory) on
#          both sides — needed for large workloads that OOM monolithically.
#        EPOCH_SIZE_LOG2=<n> continuation epoch size (default 20; min 18).
#        TX_COUNT=<n> ethrex transfer fixture to prove (default 5; use 20 for a
#          large continuation trace where GPU-residency wins are visible).
#
#   Sizing (ethrex pair-noise sd ~1.2%, 80% power): ~12 pairs for a 1% effect,
#   ~18 for 0.8%, ~32 for 0.6%. Default 20 -> solid on 0.8-1%, ~60% power at 0.6%
#   (if a 20-pair run straddles 0 on a ~0.6%-looking effect, extend to 32).
#
#   scripts/bench_abba.sh origin/my-pr-branch                # vs main, 20 pairs
#   scripts/bench_abba.sh origin/my-pr-branch origin/main 32 # 32 pairs (~0.6%)

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bench_abba.sh REF_A [REF_B=origin/main] [N_PAIRS=20]" >&2
  echo "  REF_A: ref or SHA to evaluate (the PR side)" >&2
  exit 2
fi
REF_A="$1"
REF_B="${2:-origin/main}"
N_PAIRS="${3:-20}"
# cli build features. Default matches the CPU bench; the GPU ABBA workflow overrides
# with "jemalloc-stats,prover/cuda" to exercise the CUDA prover path.
BENCH_FEATURES="${BENCH_FEATURES:-jemalloc-stats}"
# Continuation mode: split execution into epochs (flat peak memory) instead of a
# single monolithic prove. Required for large workloads (e.g. ethrex-20tx) that
# OOM monolithically, and the only way to bench a per-epoch GPU-residency change
# at a realistic trace size. When on, both binaries prove with
# `--continuations --epoch-size-log2 $EPOCH_SIZE_LOG2`.
CONTINUATIONS="${CONTINUATIONS:-0}"
EPOCH_SIZE_LOG2="${EPOCH_SIZE_LOG2:-20}"
# ethrex transfer-count fixture to prove (executor/tests/ethrex_${TX_COUNT}_transfers.bin).
TX_COUNT="${TX_COUNT:-5}"
if [ "$CONTINUATIONS" = "1" ]; then
  CONT_ARGS="--continuations --epoch-size-log2 $EPOCH_SIZE_LOG2"
else
  CONT_ARGS=""
fi

ELF_REL="executor/program_artifacts/rust/ethrex.elf"
INPUT_REL="executor/tests/ethrex_${TX_COUNT}_transfers.bin"
WORK="/tmp/abba_run"
WT="/tmp/abba_wt"
PROOF="/tmp/abba_proof.bin"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
# shellcheck source=scripts/lib/bench_abba_common.sh
. "$ROOT/scripts/lib/bench_abba_common.sh"

# Fail fast on the toolchain the final stats step needs, before the ~30-min build.
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is required (final stats step)." >&2; exit 1; }

echo "==> Refs"
git fetch origin --quiet || echo "WARNING: 'git fetch origin' failed -- resolving against possibly-stale local refs." >&2
SHA_A="$(git rev-parse "$REF_A")"
SHA_B="$(git rev-parse "$REF_B")"
echo "   A (PR)       $REF_A  -> ${SHA_A:0:10}"
echo "   B (baseline) $REF_B  -> ${SHA_B:0:10}"
if [ $((N_PAIRS % 2)) -ne 0 ]; then
  echo "   WARNING: N_PAIRS=$N_PAIRS is odd; use an even count so AB/BA orders balance."
fi
echo "   pairs=$N_PAIRS  (=$((N_PAIRS * 2)) prove runs)"

mkdir -p "$WORK"

# --- 1. Guest ELF + fixture (identical for both sides; build once if missing) ---
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
ELF="$(cd "$(dirname "$ELF_REL")" && pwd)/$(basename "$ELF_REL")"
INPUT="$(cd "$(dirname "$INPUT_REL")" && pwd)/$(basename "$INPUT_REL")"

# --- 2. Build (or reuse) both prover binaries ---
need_build=0
if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$WORK/cli_A" ] || [ ! -x "$WORK/cli_B" ]; then
  need_build=1
elif [ "$(cat "$WORK/cli_A.sha" 2>/dev/null)" != "$SHA_A $BENCH_FEATURES" ] || \
     [ "$(cat "$WORK/cli_B.sha" 2>/dev/null)" != "$SHA_B $BENCH_FEATURES" ]; then
  # Cache persists on the self-hosted runner; rebuild if it's for different refs (a
  # different PR, or main advanced) OR a different feature set (e.g. CPU vs prover/cuda),
  # so we never benchmark stale binaries. The marker stores "<sha> <features>".
  echo "==> Cached binaries are for different refs/features; rebuilding."
  need_build=1
fi
if [ "$need_build" = "1" ]; then
  cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; }
  trap cleanup EXIT
  git worktree remove --force "$WT" 2>/dev/null || true
  echo "==> Building both prover binaries in isolated worktree $WT"
  git worktree add --detach "$WT" "$SHA_B" >/dev/null
  build_cli() {  # $1=sha $2=out (shared target dir -> 2nd build is incremental)
    echo "==> Building cli @ ${1:0:10} -> $2  (features: $BENCH_FEATURES)"
    # -f: discard any prior worktree edit (e.g. the CUDARC_PIN sed below) before switching
    # refs, so the checkout can't conflict.
    git -C "$WT" checkout --quiet -f "$1"
    # CUDARC_PIN compat shim (no-op with a warning on post-pin shas) — see
    # cudarc_pin_apply in scripts/lib/bench_abba_common.sh.
    cudarc_pin_apply "$WT/crypto/math-cuda/Cargo.toml"
    if ! ( cd "$WT" && cargo build --release -p cli --features "$BENCH_FEATURES" >"$WORK/build_$2.log" 2>&1 ); then
      echo "ERROR: cargo build failed for $2 (@ ${1:0:10}). Tail of $WORK/build_$2.log:" >&2
      tail -40 "$WORK/build_$2.log" >&2
      exit 1
    fi
    cp "$WT/target/release/cli" "$WORK/$2"
    # Marker = "<sha> <features>" so the cache invalidates on either changing.
    echo "$1 $BENCH_FEATURES" > "$WORK/$2.sha"
  }
  build_cli "$SHA_B" cli_B
  build_cli "$SHA_A" cli_A
  cleanup
  trap - EXIT
else
  echo "==> Reusing cached binaries (refs + features match; REBUILD=1 to force):"
  echo "     cli_A=${SHA_A:0:10}  cli_B=${SHA_B:0:10}  features=$BENCH_FEATURES"
fi

# --- 3. Interleaved A/B/B/A measurement (fresh CSV -- pre-committed batch) ---
run_prove() {  # $1=binary -> echoes proving time (s)
  local out
  # shellcheck disable=SC2086  # CONT_ARGS is intentionally word-split (0 or 2 args)
  out="$("$1" prove "$ELF" --private-input "$INPUT" -o "$PROOF" --time $CONT_ARGS 2>&1)"
  rm -f "$PROOF"
  extract_prove_time "$out"
}

echo "==> Running $N_PAIRS interleaved pairs  (improvement: - = PR faster)"
printf 'pair,a_time,b_time\n' > "$WORK/pairs.csv"
for i in $(seq 1 "$N_PAIRS"); do
  if [ $((i % 2)) -eq 1 ]; then          # odd pair: A then B
    a="$(run_prove "$WORK/cli_A")"; b="$(run_prove "$WORK/cli_B")"
  else                                   # even pair: B then A (ABBA pattern)
    b="$(run_prove "$WORK/cli_B")"; a="$(run_prove "$WORK/cli_A")"
  fi
  printf '%d,%s,%s\n' "$i" "$a" "$b" >> "$WORK/pairs.csv"
  printf '   pair %2d/%d   A=%ss  B=%ss   PR %+.2f%% (-=faster)\n' \
    "$i" "$N_PAIRS" "$a" "$b" "$(awk "BEGIN{print ($a-$b)/$b*100}")"
done

# --- 4. Paired t-test + robust median/Wilcoxon (shared analysis) ---
abba_stats "$WORK/pairs.csv" "PR" "base" "ABBA paired result"
