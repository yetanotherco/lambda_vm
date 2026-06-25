#!/usr/bin/env bash
#
# bench_abba.sh — interleaved A/B/B/A paired benchmark for the ethrex prover bench.
#
# WHY: comparing a PR against a separately-recorded (cached) baseline conflates the
# code delta with machine drift between the two measurement sessions. For small
# (~1%) prover changes that drift is the dominant error. Measuring both binaries
# *interleaved on the same machine in the same session* cancels the drift (it hits
# both sides equally), and a paired analysis over the A/B pairs is far more powerful
# than an unpaired two-sample test. This resolves ~1% deltas that the 5% gate and
# the cached-baseline comparison cannot.
#
# WHAT IT DOES:
#   1. Builds the ethrex guest ELF + 20-transfer fixture once (identical for both
#      sides — PR #696 only touches the prover, not the guest).
#   2. Builds the `cli` prover at REF_A and REF_B into two separate binaries, using
#      an isolated git worktree so your current checkout is never touched.
#   3. Runs N_PAIRS interleaved pairs in A B B A ... order (alternating which side
#      goes first each pair, to cancel linear drift).
#   4. Reports a paired-t 95% CI on the per-pair % difference in proving time, and a
#      verdict (real improvement / real regression / inconclusive).
#
# USAGE:
#   scripts/bench_abba.sh [REF_A] [REF_B] [N_PAIRS]
#
#   REF_A    PR ref to evaluate   (default: origin/perf/logup-fingerprint-constants, PR #696)
#   REF_B    baseline ref         (default: origin/main)
#   N_PAIRS  number of A/B pairs   (default: 8  -> 16 prove runs, ~0.5% detection floor)
#
#   # PR #696 vs main, default 8 pairs:
#   scripts/bench_abba.sh
#   # explicit, quick 5-pair run:
#   scripts/bench_abba.sh origin/perf/logup-fingerprint-constants origin/main 5
#
# Positive % = REF_A (the PR) is FASTER than REF_B (baseline), i.e. an improvement.

set -euo pipefail

REF_A="${1:-origin/perf/logup-fingerprint-constants}"
REF_B="${2:-origin/main}"
N_PAIRS="${3:-8}"

ELF_REL="executor/program_artifacts/rust/ethrex.elf"
INPUT_REL="executor/tests/ethrex_bench_20.bin"
WORK="/tmp/abba_run"
WT="/tmp/abba_wt"
PROOF="/tmp/abba_proof.bin"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "==> Fetching refs"
git fetch origin --quiet || echo "   (fetch failed; using local refs)"
SHA_A="$(git rev-parse "$REF_A")"
SHA_B="$(git rev-parse "$REF_B")"
echo "   A (PR)       $REF_A  -> ${SHA_A:0:10}"
echo "   B (baseline) $REF_B  -> ${SHA_B:0:10}"
echo "   pairs=$N_PAIRS  (=$((N_PAIRS * 2)) prove runs)"

mkdir -p "$WORK"

# --- 1. Guest ELF + fixture (identical for both sides; build once if missing) ---
if [ ! -f "$ELF_REL" ]; then
  echo "==> Building ethrex guest ELF (missing)"
  export SYSROOT_DIR="${SYSROOT_DIR:-$HOME/.lambda-vm-sysroot}"
  make "$ELF_REL"
fi
if [ ! -f "$INPUT_REL" ]; then
  echo "==> Generating ethrex 20-transfer fixture (missing)"
  ( cd tooling/ethrex-fixtures && cargo build --release )
  tooling/ethrex-fixtures/target/release/ethrex-fixtures 20 "$INPUT_REL" distinct
fi
ELF="$(cd "$(dirname "$ELF_REL")" && pwd)/$(basename "$ELF_REL")"
INPUT="$(cd "$(dirname "$INPUT_REL")" && pwd)/$(basename "$INPUT_REL")"

# --- 2. Build both prover binaries in an isolated worktree (main tree untouched) ---
cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; }
trap cleanup EXIT
git worktree remove --force "$WT" 2>/dev/null || true
echo "==> Creating build worktree at $WT"
git worktree add --detach "$WT" "$SHA_B" >/dev/null

build_cli() {  # $1=sha  $2=out-name  (shared target dir -> 2nd build is incremental)
  echo "==> Building cli @ ${1:0:10} -> $2"
  git -C "$WT" checkout --quiet "$1"
  ( cd "$WT" && cargo build --release -p cli --features jemalloc-stats >/dev/null 2>&1 )
  cp "$WT/target/release/cli" "$WORK/$2"
}
build_cli "$SHA_B" cli_B
build_cli "$SHA_A" cli_A
cleanup
trap - EXIT

# --- 3. Interleaved A/B/B/A measurement ---
run_prove() {  # $1=binary  -> echoes proving time in seconds
  local out t
  out="$("$1" prove "$ELF" --private-input "$INPUT" -o "$PROOF" --time 2>/dev/null)"
  rm -f "$PROOF"
  t="$(printf '%s\n' "$out" | grep -o 'Proving time: [0-9.]*' | awk '{print $3}')"
  if [ -z "$t" ]; then
    echo "ERROR: could not parse 'Proving time' from cli output:" >&2
    printf '%s\n' "$out" >&2
    exit 1
  fi
  echo "$t"
}

echo "==> Running $N_PAIRS interleaved pairs"
printf 'pair,a_time,b_time\n' > "$WORK/pairs.csv"
for i in $(seq 1 "$N_PAIRS"); do
  if [ $((i % 2)) -eq 1 ]; then          # odd pair: A then B
    a="$(run_prove "$WORK/cli_A")"; b="$(run_prove "$WORK/cli_B")"
  else                                   # even pair: B then A (ABBA pattern)
    b="$(run_prove "$WORK/cli_B")"; a="$(run_prove "$WORK/cli_A")"
  fi
  printf '%d,%s,%s\n' "$i" "$a" "$b" >> "$WORK/pairs.csv"
  printf '   pair %2d/%d   A=%ss  B=%ss  (A-B=%+.3fs)\n' \
    "$i" "$N_PAIRS" "$a" "$b" "$(awk "BEGIN{print $a-$b}")"
done

# --- 4. Paired t-test on the per-pair % difference ---
python3 - "$WORK/pairs.csv" <<'PY'
import sys, csv, math
rows = list(csv.DictReader(open(sys.argv[1])))
A = [float(r['a_time']) for r in rows]   # PR
B = [float(r['b_time']) for r in rows]   # baseline
n = len(A)
# per-pair improvement: positive => PR (A) faster than baseline (B)
d = [(b - a) / b * 100.0 for a, b in zip(A, B)]
mean = sum(d) / n
var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
sd = math.sqrt(var)
se = sd / math.sqrt(n) if n > 0 else float('inf')
# two-sided 95% t critical values by degrees of freedom
TT = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,
      10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,
      18:2.101,19:2.093,20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,
      26:2.056,27:2.052,28:2.048,29:2.045,30:2.042}
df = n - 1
tc = TT.get(df, 2.042 if df > 30 else TT[min(TT, key=lambda k: abs(k - df))])
lo, hi = mean - tc * se, mean + tc * se
mean_a, mean_b = sum(A)/n, sum(B)/n
print("\n=== ABBA paired result ===")
print(f"  pairs: {n}   df: {df}   t(0.025,df): {tc}")
print(f"  mean A (PR):       {mean_a:8.3f} s")
print(f"  mean B (baseline): {mean_b:8.3f} s")
print(f"  per-pair improvement (B->A): mean {mean:+.2f}%   sd {sd:.2f}%   se {se:.2f}%")
print(f"  95% CI on improvement: [{lo:+.2f}%, {hi:+.2f}%]")
if lo > 0:
    print(f"  VERDICT: REAL IMPROVEMENT — PR is faster by {mean:.2f}% (95% CI entirely > 0)")
elif hi < 0:
    print(f"  VERDICT: REAL REGRESSION — PR is slower by {-mean:.2f}% (95% CI entirely < 0)")
else:
    print(f"  VERDICT: INCONCLUSIVE — 95% CI includes 0. Re-run with more pairs to tighten.")
print(f"\n  raw pairs CSV: {sys.argv[1]}")
PY
