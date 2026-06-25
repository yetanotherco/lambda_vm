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
#   1. Builds the ethrex guest ELF + 20-transfer fixture once (identical for both
#      sides — a prover-only change doesn't touch the guest).
#   2. Builds the `cli` prover at REF_A and REF_B (skips the build and reuses the
#      cached binaries if they already exist; set REBUILD=1 to force).
#   3. Runs N_PAIRS interleaved pairs in A B B A ... order (alternating which side
#      runs first each pair, to cancel linear drift). Use an EVEN N_PAIRS.
#   4. Reports BOTH a paired-t 95% CI (sensitive to outliers) AND a robust
#      median + Wilcoxon signed-rank result (shrugs off transient slow runs).
#
# CONVENTION: every reported number is an IMPROVEMENT, positive = PR FASTER.
#
# USAGE:
#   scripts/bench_abba.sh [REF_A] [REF_B] [N_PAIRS]
#     REF_A    PR ref     (default: origin/perf/logup-fingerprint-constants, PR #696)
#     REF_B    baseline   (default: origin/main)
#     N_PAIRS  pairs      (default: 20 -> 40 runs, ~33 min on ethrex)
#   Env: REBUILD=1 forces a rebuild even if cached binaries exist.
#
#   Sizing (ethrex pair-noise sd ~1.2%, 80% power): ~12 pairs for a 1% effect,
#   ~18 for 0.8%, ~32 for 0.6%. Default 20 -> solid on 0.8-1%, ~60% power at 0.6%
#   (if a 20-pair run straddles 0 on a ~0.6%-looking effect, extend to 32).
#
#   scripts/bench_abba.sh                                   # PR #696 vs main, 20 pairs
#   scripts/bench_abba.sh origin/perf/logup-fingerprint-constants origin/main 32

set -euo pipefail

REF_A="${1:-origin/perf/logup-fingerprint-constants}"
REF_B="${2:-origin/main}"
N_PAIRS="${3:-20}"

ELF_REL="executor/program_artifacts/rust/ethrex.elf"
INPUT_REL="executor/tests/ethrex_bench_20.bin"
WORK="/tmp/abba_run"
WT="/tmp/abba_wt"
PROOF="/tmp/abba_proof.bin"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "==> Refs"
git fetch origin --quiet || echo "   (fetch failed; using local refs)"
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
  echo "==> Generating ethrex 20-transfer fixture (missing)"
  ( cd tooling/ethrex-fixtures && cargo build --release )
  tooling/ethrex-fixtures/target/release/ethrex-fixtures 20 "$INPUT_REL" distinct
fi
ELF="$(cd "$(dirname "$ELF_REL")" && pwd)/$(basename "$ELF_REL")"
INPUT="$(cd "$(dirname "$INPUT_REL")" && pwd)/$(basename "$INPUT_REL")"

# --- 2. Build (or reuse) both prover binaries ---
need_build=0
if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$WORK/cli_A" ] || [ ! -x "$WORK/cli_B" ]; then
  need_build=1
fi
if [ "$need_build" = "1" ]; then
  cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; }
  trap cleanup EXIT
  git worktree remove --force "$WT" 2>/dev/null || true
  echo "==> Building both prover binaries in isolated worktree $WT"
  git worktree add --detach "$WT" "$SHA_B" >/dev/null
  build_cli() {  # $1=sha $2=out (shared target dir -> 2nd build is incremental)
    echo "==> Building cli @ ${1:0:10} -> $2"
    git -C "$WT" checkout --quiet "$1"
    ( cd "$WT" && cargo build --release -p cli --features jemalloc-stats >/dev/null 2>&1 )
    cp "$WT/target/release/cli" "$WORK/$2"
    echo "$1" > "$WORK/$2.sha"
  }
  build_cli "$SHA_B" cli_B
  build_cli "$SHA_A" cli_A
  cleanup
  trap - EXIT
else
  echo "==> Reusing cached binaries (set REBUILD=1 to force a rebuild):"
  echo "     cli_A: $(cat "$WORK/cli_A.sha" 2>/dev/null || echo '?')  ($(date -r "$WORK/cli_A" 2>/dev/null || echo 'built earlier'))"
  echo "     cli_B: $(cat "$WORK/cli_B.sha" 2>/dev/null || echo '?')  ($(date -r "$WORK/cli_B" 2>/dev/null || echo 'built earlier'))"
  echo "     (requested A=${SHA_A:0:10} B=${SHA_B:0:10} -- verify these match before trusting results)"
fi

# --- 3. Interleaved A/B/B/A measurement (fresh CSV -- pre-committed batch) ---
run_prove() {  # $1=binary -> echoes proving time (s)
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

echo "==> Running $N_PAIRS interleaved pairs  (improvement: + = PR faster)"
printf 'pair,a_time,b_time\n' > "$WORK/pairs.csv"
for i in $(seq 1 "$N_PAIRS"); do
  if [ $((i % 2)) -eq 1 ]; then          # odd pair: A then B
    a="$(run_prove "$WORK/cli_A")"; b="$(run_prove "$WORK/cli_B")"
  else                                   # even pair: B then A (ABBA pattern)
    b="$(run_prove "$WORK/cli_B")"; a="$(run_prove "$WORK/cli_A")"
  fi
  printf '%d,%s,%s\n' "$i" "$a" "$b" >> "$WORK/pairs.csv"
  printf '   pair %2d/%d   A=%ss  B=%ss   PR %+.2f%% (+=faster)\n' \
    "$i" "$N_PAIRS" "$a" "$b" "$(awk "BEGIN{print ($b-$a)/$b*100}")"
done

# --- 4. Paired t-test + robust median/Wilcoxon ---
python3 - "$WORK/pairs.csv" <<'PY'
import sys, csv, math

rows = list(csv.DictReader(open(sys.argv[1])))
A = [float(r['a_time']) for r in rows]   # PR
B = [float(r['b_time']) for r in rows]   # baseline
n = len(A)
# per-pair improvement: positive => PR (A) faster than baseline (B)
d = [(b - a) / b * 100.0 for a, b in zip(A, B)]

# ---- parametric: paired t ----
mean = sum(d) / n
var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
sd = math.sqrt(var)
se = sd / math.sqrt(n) if n else float('inf')
TT = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,
      10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,
      18:2.101,19:2.093,20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,
      26:2.056,27:2.052,28:2.048,29:2.045,30:2.042,35:2.030,40:2.021,50:2.009,
      60:2.000,80:1.990,120:1.980}
df = n - 1
tc = TT.get(df) or (1.96 if df > 120 else TT[min(TT, key=lambda k: abs(k - df))])
lo, hi = mean - tc * se, mean + tc * se

# ---- robust: median + Wilcoxon signed-rank (tie-averaged ranks, normal approx) ----
def median(xs):
    s = sorted(xs); m = len(s)
    return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2

nz = [x for x in d if x != 0.0]
m = len(nz)
order = sorted(range(m), key=lambda i: abs(nz[i]))
ranks = [0.0] * m
i = 0
while i < m:                                   # average ranks within ties on |d|
    j = i
    while j + 1 < m and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
        j += 1
    avg = (i + 1 + j + 1) / 2.0
    for k in range(i, j + 1):
        ranks[order[k]] = avg
    i = j + 1
Wp = sum(r for r, x in zip(ranks, nz) if x > 0)
Wn = sum(r for r, x in zip(ranks, nz) if x < 0)
mu = m * (m + 1) / 4.0
sig = math.sqrt(m * (m + 1) * (2 * m + 1) / 24.0) if m else 0.0
z = (Wp - mu - (0.5 if Wp > mu else -0.5)) / sig if sig else 0.0   # continuity-corrected
p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
med = median(d)

print("\n=== ABBA paired result  (improvement: + = PR faster) ===")
print(f"  pairs: {n}   mean A (PR): {sum(A)/n:.3f}s   mean B (base): {sum(B)/n:.3f}s")
print()
print(f"  [parametric] paired-t   mean {mean:+.2f}%   sd {sd:.2f}%   se {se:.2f}%")
print(f"               95% CI: [{lo:+.2f}%, {hi:+.2f}%]   (t df={df} = {tc})")
print(f"  [robust]     median {med:+.2f}%   Wilcoxon W+={Wp:.0f} W-={Wn:.0f}  z={z:+.2f}  p~={p:.3f}")
print()
if lo > 0 and p < 0.05:
    print(f"  VERDICT: REAL IMPROVEMENT - PR faster by ~{mean:.2f}% (t-CI and Wilcoxon agree)")
elif hi < 0 and p < 0.05:
    print(f"  VERDICT: REAL REGRESSION - PR slower by ~{-mean:.2f}% (t-CI and Wilcoxon agree)")
elif (lo > 0) != (p < 0.05):
    print(f"  VERDICT: BORDERLINE - parametric and robust disagree; suspect outlier pair(s).")
    print(f"           Trust the median ({med:+.2f}%); add pairs or inspect the per-pair list.")
else:
    print(f"  VERDICT: INCONCLUSIVE - effect not separable from 0 at n={n}.")
    print(f"           Point estimate ~{med:+.2f}% (median). Need more pairs to resolve.")
print(f"\n  raw pairs: {sys.argv[1]}")
PY
