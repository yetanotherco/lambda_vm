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
#   1. Builds the ethrex guest ELF and obtains the workload fixture once (identical
#      for both sides — a prover-only change doesn't touch the guest). Which fixture
#      is WORKLOAD's business; see below.
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
#          Ignored when WORKLOAD=real.
#        WORKLOAD=real|synthetic|keccak (default real) what to prove. `real`
#          fetches the real-block fixture (block identity lives in the Makefile) and
#          forces --continuations; TX_COUNT and CONTINUATIONS do not apply to it.
#        KECCAK_PERMS=<n> permutations for WORKLOAD=keccak (default 5461).
#
#   WORKLOAD=keccak proves a guest that does nothing but N keccak-f[1600] ecalls
#   (scripts/gen_keccak_bench.sh), so KECCAK_RND is ~86% of committed cells and a
#   keccak-table change shows up close to undiluted — where the same change on a real
#   block is scaled down by keccak's share of that trace. Use it to size a keccak
#   change, then a `real` run to see what it is worth in production. It needs far
#   fewer pairs than `real`: 5461 permutations prove in ~13 s on a 32-core box, so 6
#   pairs is ~3 minutes, and both sides run an identical generated ELF with no fixture.
#
#   ON GPU: pass BENCH_FEATURES="jemalloc-stats,prover/cuda". Mind the LDE threshold —
#   the GPU path only engages once padded_rows * blowup clears 2^19, and KECCAK_RND
#   commits 24 rows per permutation, so at blowup 2 the default KECCAK_PERMS=5461
#   stays on the CPU. Use KECCAK_PERMS=21845 (or >=10922) for a GPU run, or the two
#   sides will be compared on a path neither of them took.
#
#   scripts/bench_abba.sh 6a280121 6a280121^ 6         # WORKLOAD=keccak: isolate #889
#
#   On WORKLOAD: a real block is keccak- and trie-bound, while the synthetic option is
#   N plain transfers — ecrecover-heavy over a near-empty state — so a prover change can
#   move the two in opposite directions. `real` is the default here, in benchmark-gpu.yml
#   and in /bench, so the screen and its tiebreaker resolve the same workload; pass
#   WORKLOAD=synthetic to reproduce a number recorded against that fixture.
#
#   Sizing at WORKLOAD=real, from the paired t-test (resolvable 95% delta =
#   t* x sd / sqrt(N)). The pair-delta sd on the bench runner is NOT yet measured; the
#   two columns bracket it between 1.0% (the GPU box's measured 0.64% plus margin) and
#   2.0% (sqrt(2) x the runner's measured 1.43% single-run CV):
#
#     pairs   wall      resolves (sd 2.0% / sd 1.0%)
#      8      ~50 min    1.7% / 0.8%
#     12      ~72 min    1.3% / 0.6%   <- workflow default
#     20      ~1h55m     0.9% / 0.5%
#     32      ~3h        0.7% / 0.4%
#
#   Wall assumes epoch 2^22 (158.8 s per prove, two per pair) plus ~8 min of setup.
#   The first real ABBA run MEASURES that sd — read it off the `sd` field of the
#   paired-t line printed below — and this table should be re-pinned to it.
#
#   scripts/bench_abba.sh origin/my-pr-branch                # vs main, 20 pairs
#   scripts/bench_abba.sh origin/my-pr-branch origin/main 32 # 32 pairs

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
CONTINUATIONS_EXPLICIT="${CONTINUATIONS+1}"
CONTINUATIONS="${CONTINUATIONS:-0}"
EPOCH_SIZE_LOG2="${EPOCH_SIZE_LOG2:-20}"
# ethrex transfer-count fixture to prove (executor/tests/ethrex_${TX_COUNT}_transfers.bin).
TX_COUNT="${TX_COUNT:-5}"
WORKLOAD="${WORKLOAD:-real}"
case "$WORKLOAD" in
  synthetic|real|keccak) ;;
  *) echo "ERROR: WORKLOAD must be 'synthetic', 'real' or 'keccak' (got '$WORKLOAD')." >&2; exit 2 ;;
esac
# WORKLOAD=keccak: permutation count for the saturated keccak guest. 5461 puts
# 131,064 KECCAK_RND rows just under 2^17, so almost nothing is lost to padding
# and the measured delta is not diluted by padded rows that both sides carry.
KECCAK_PERMS="${KECCAK_PERMS:-5461}"
# A real block sits far past the monolithic memory ceiling (~4.9 GB of peak heap per
# million cycles puts it in the hundreds of GB), so continuations are forced rather
# than offered: CONTINUATIONS=0 here could only produce an OOM.
if [ "$WORKLOAD" = "real" ] && [ "$CONTINUATIONS" != "1" ]; then
  echo "   NOTE: WORKLOAD=real always proves with continuations (monolithic would OOM)."
  CONTINUATIONS=1
fi
# The keccak figures on record were taken as single-epoch continuation proofs;
# default to that so a new reading is comparable to them. The trace is small
# enough that monolithic also works — set CONTINUATIONS=0 for it.
if [ "$WORKLOAD" = "keccak" ] && [ -z "$CONTINUATIONS_EXPLICIT" ]; then
  CONTINUATIONS=1
fi
if [ "$CONTINUATIONS" = "1" ]; then
  CONT_ARGS="--continuations --epoch-size-log2 $EPOCH_SIZE_LOG2"
else
  CONT_ARGS=""
fi

ELF_REL="executor/program_artifacts/rust/ethrex.elf"
# Resolved after the cd to the repo root (WORKLOAD=real reads it from the Makefile).
INPUT_REL=""
WORK="/tmp/abba_run"
WT="/tmp/abba_wt"
PROOF="/tmp/abba_proof.bin"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

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

# The real block's identity lives in the Makefile and nowhere else, so repointing it
# never needs an edit here.
if [ "$WORKLOAD" = "real" ]; then
  INPUT_REL="$(make -s print-real-block-fixture)"
  echo "   workload=real  $INPUT_REL  (continuations, epoch 2^$EPOCH_SIZE_LOG2)"
elif [ "$WORKLOAD" = "keccak" ]; then
  echo "   workload=keccak  $KECCAK_PERMS permutations  ($((KECCAK_PERMS * 24)) KECCAK_RND rows)"
else
  INPUT_REL="executor/tests/ethrex_${TX_COUNT}_transfers.bin"
  echo "   workload=synthetic  ${TX_COUNT}tx  $INPUT_REL"
fi

mkdir -p "$WORK"

# --- 1. Guest ELF + fixture (identical for both sides; build once if missing) ---
if [ "$WORKLOAD" = "keccak" ]; then
  # Generated, not checked in: the permutation count is a parameter. Costs
  # milliseconds, so it is regenerated unconditionally rather than cached — a
  # stale ELF from an earlier run at a different KECCAK_PERMS would silently
  # measure the wrong workload.
  ELF_REL="$WORK/keccak_${KECCAK_PERMS}.elf"
  echo "==> Generating keccak guest ($KECCAK_PERMS permutations)"
  scripts/gen_keccak_bench.sh "$KECCAK_PERMS" "$ELF_REL"
elif [ ! -f "$ELF_REL" ]; then
  echo "==> Building ethrex guest ELF (missing)"
  export SYSROOT_DIR="${SYSROOT_DIR:-$HOME/.lambda-vm-sysroot}"
  make "$ELF_REL"
fi
if [ "$WORKLOAD" = "keccak" ]; then
  : # no fixture; the guest is self-contained
elif [ "$WORKLOAD" = "real" ]; then
  # ~1 MB, gitignored, never in a fresh checkout — and a rented GPU box is always a
  # fresh checkout. Fetched by URL + sha256, not built: no converter, no ethrex host
  # dependency tree, so this costs seconds on the box. Unconditional on purpose: the
  # target hashes whatever is on disk on every invocation, which is what catches a
  # copy left behind by an earlier run in the same rental. A match costs ~35 ms.
  echo "==> Verifying ethrex real-block fixture (fetches on a digest miss)"
  make ethrex-real-block-fixture
elif [ ! -f "$INPUT_REL" ]; then
  echo "==> Generating ethrex ${TX_COUNT}-transfer fixture (missing)"
  ( cd tooling/ethrex-fixtures && cargo build --release )
  tooling/ethrex-fixtures/target/release/ethrex-fixtures "$TX_COUNT" "$INPUT_REL" distinct
fi
ELF="$(cd "$(dirname "$ELF_REL")" && pwd)/$(basename "$ELF_REL")"
if [ -n "$INPUT_REL" ]; then
  INPUT="$(cd "$(dirname "$INPUT_REL")" && pwd)/$(basename "$INPUT_REL")"
  INPUT_ARGS=(--private-input "$INPUT")
else
  INPUT_ARGS=()
fi

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
    # CUDARC_PIN: pin math-cuda's cudarc to a fixed CUDA version and drop fallback-latest, so
    # cudarc binds a known driver-symbol set instead of its newest (which can request symbols
    # the rented box's driver doesn't export, e.g. cuDevSmResourceSplit -> runtime panic).
    if [ -n "${CUDARC_PIN:-}" ]; then
      sed -i "s/\"cuda-version-from-build-system\"/\"${CUDARC_PIN}\"/; /\"fallback-latest\"/d" \
        "$WT/crypto/math-cuda/Cargo.toml"
      echo "    cudarc pinned to ${CUDARC_PIN}"
    fi
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

# --- 2b. keccak gate: both sides must see the same permutation count ---
# The whole point of this workload is that the permutations dominate, so the
# count is checked against BOTH binaries rather than assumed: if a ref changed
# the accelerator's ecall ABI, the guest would fall back to something else on
# one side and the "speedup" would be an artefact.
if [ "$WORKLOAD" = "keccak" ]; then
  for side in A B; do
    calls="$("$WORK/cli_$side" execute "$ELF" --cycles 2>/dev/null | awk '/^Keccak calls:/ {print $3}')"
    if [ "$calls" != "$KECCAK_PERMS" ]; then
      echo "ERROR: cli_$side counted '${calls:-none}' keccak permutations, expected $KECCAK_PERMS." >&2
      exit 1
    fi
  done
  echo "==> Gate: both binaries execute exactly $KECCAK_PERMS keccak permutations"
fi

# --- 3. Interleaved A/B/B/A measurement (fresh CSV -- pre-committed batch) ---
run_prove() {  # $1=binary -> echoes proving time (s)
  local out t
  # shellcheck disable=SC2086  # CONT_ARGS is intentionally word-split (0 or 2 args)
  # ${a[@]+"${a[@]}"} rather than "${a[@]}": under `set -u`, bash < 4.4 treats an
  # empty array expansion as an unbound variable, and WORKLOAD=keccak has no
  # --private-input to pass.
  out="$("$1" prove "$ELF" ${INPUT_ARGS[@]+"${INPUT_ARGS[@]}"} -o "$PROOF" --time $CONT_ARGS 2>&1)"
  rm -f "$PROOF"
  t="$(printf '%s\n' "$out" | grep -o 'Proving time: [0-9.]*' | awk '{print $3}')"
  if [ -z "$t" ]; then
    echo "ERROR: could not parse 'Proving time' from cli output:" >&2
    printf '%s\n' "$out" >&2
    exit 1
  fi
  echo "$t"
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

# --- 4. Paired t-test + robust median/Wilcoxon ---
python3 - "$WORK/pairs.csv" <<'PY'
import sys, csv, math

rows = list(csv.DictReader(open(sys.argv[1])))
A = [float(r['a_time']) for r in rows]   # PR
B = [float(r['b_time']) for r in rows]   # baseline
n = len(A)
# per-pair delta = (PR - baseline)/baseline: negative => PR (A) faster than baseline (B)
d = [(a - b) / b * 100.0 for a, b in zip(A, B)]

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

# ---- robust: median + Wilcoxon signed-rank (tie-averaged ranks, EXACT p, pure stdlib) ----
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
z = (Wp - mu - (0.5 if Wp > mu else -0.5)) / sig if sig else 0.0   # normal approx (display only)
# EXACT two-sided p: enumerate the signed-rank null distribution. Each rank is +/- with
# prob 1/2, so the count of assignments giving W+=v is the coeff of x^v in prod(1 + x^rank)
# -- build it with a generating-function DP. Double the ranks so tie-averaged (half-integer)
# ranks become integers. No scipy; exact even at small n where the normal approx is loose.
if m:
    ir = [int(round(2 * r)) for r in ranks]
    poly = [1]
    for r in ir:
        nxt = [0] * (len(poly) + r)
        for v, c in enumerate(poly):
            if c:
                nxt[v] += c          # this rank negative -> adds 0 to W+
                nxt[v + r] += c      # this rank positive -> adds r to W+
        poly = nxt
    Wp2 = int(round(2 * Wp))
    p = min(1.0, 2.0 * min(sum(poly[:Wp2 + 1]), sum(poly[Wp2:])) / (1 << m))
else:
    p = 1.0
med = median(d)

# ---- server stability (byproduct): run-to-run jitter + within-session drift ----
def cv(xs):
    mm = sum(xs) / len(xs)
    s = math.sqrt(sum((x - mm) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return (s / mm * 100.0) if mm else 0.0
mA, mB = sum(A) / n, sum(B) / n
cvA, cvB = cv(A), cv(B)
# reconstruct execution order (odd pair: A,B ; even pair: B,A) and normalize each
# run by its binary's mean so the A/B offset drops out, leaving pure machine drift.
seq = []
for i in range(n):
    seq += ([('A', A[i]), ('B', B[i])] if (i + 1) % 2 else [('B', B[i]), ('A', A[i])])
nrm = [(t / (mA if lbl == 'A' else mB) - 1) * 100 for lbl, t in seq]
N = len(nrm); mi = (N - 1) / 2.0; mn = sum(nrm) / N
denom = sum((i - mi) ** 2 for i in range(N))
slope = (sum((i - mi) * (nrm[i] - mn) for i in range(N)) / denom) if denom else 0.0
half = N // 2
drift_shift = sum(nrm[half:]) / (N - half) - sum(nrm[:half]) / half

print("\n=== ABBA paired result  (improvement: - = PR faster) ===")
print(f"  pairs: {n}   mean A (PR): {sum(A)/n:.3f}s   mean B (base): {sum(B)/n:.3f}s")
print()
print(f"  [parametric] paired-t   mean {mean:+.2f}%   sd {sd:.2f}%   se {se:.2f}%")
print(f"               95% CI: [{lo:+.2f}%, {hi:+.2f}%]   (t df={df} = {tc})")
pstr = f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"
print(f"  [robust]     median {med:+.2f}%   Wilcoxon W+={Wp:.0f} W-={Wn:.0f}  p(exact)={pstr}  (z={z:+.2f})")
print()
print("  --- server stability (this run; compare across servers) ---")
print(f"  run-to-run jitter:    A CV {cvA:.2f}%   B CV {cvB:.2f}%        (lower = steadier)")
print(f"  within-session drift: {slope * N:+.2f}% over the run, 1st->2nd half {drift_shift:+.2f}%")
print(f"    (jitter -> Tier-1 cached gate floor; drift -> whether the cached baseline can be trusted)")
print()
if hi < 0 and p < 0.05:
    print(f"  VERDICT: REAL IMPROVEMENT - PR faster by ~{-mean:.2f}% (t-CI and Wilcoxon agree)")
elif lo > 0 and p < 0.05:
    print(f"  VERDICT: REAL REGRESSION - PR slower by ~{mean:.2f}% (t-CI and Wilcoxon agree)")
elif (hi < 0) != (p < 0.05):
    print(f"  VERDICT: BORDERLINE - parametric and robust disagree; suspect outlier pair(s).")
    print(f"           Trust the median ({med:+.2f}%); add pairs or inspect the per-pair list.")
else:
    print(f"  VERDICT: INCONCLUSIVE - effect not separable from 0 at n={n}.")
    print(f"           Point estimate ~{med:+.2f}% (median). Need more pairs to resolve.")
print(f"\n  raw pairs: {sys.argv[1]}")
PY
