#!/usr/bin/env bash
#
# bench_verify.sh — interleaved A/B/B/A paired verifier benchmark (PR vs main).
# Positive numbers are improvements (PR faster).
#
# Usage: scripts/bench_verify.sh REF_A [REF_B=origin/main] [N_PAIRS=20]
#   REF_A/REF_B  refs to compare (A = PR side); N_PAIRS even, default 20 (~4 min).
#   Env: REBUILD=1 forces rebuild + re-prove; BENCH_FEATURES=<list> (default: jemalloc-stats).
#        PROVE_PER_SIDE=auto|1|0 (default auto): 1 = each side proves+verifies its
#        own proof (required when REF_A changes the proof format); 0 = force one
#        shared proof (best precision); auto = share if the PR binary can verify the
#        baseline's proof, else fall back to per-side.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bench_verify.sh REF_A [REF_B=origin/main] [N_PAIRS=20]" >&2
  echo "  REF_A: ref or SHA to evaluate (the PR side)" >&2
  exit 2
fi
REF_A="$1"
REF_B="${2:-origin/main}"
N_PAIRS="${3:-20}"
BENCH_FEATURES="${BENCH_FEATURES:-jemalloc-stats}"

ELF_REL="executor/program_artifacts/rust/ethrex.elf"
INPUT_REL="executor/tests/ethrex_bench_20.bin"
WORK="/tmp/verify_run"
WT="/tmp/verify_wt"
PROOF_B="$WORK/proof_b.bin"   # baseline's proof (cached in $WORK, keyed like the binaries)
PROOF_A="$WORK/proof_a.bin"   # PR's proof (cached likewise)

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Fail fast on the toolchain the final stats step needs, before the build.
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
echo "   pairs=$N_PAIRS  (=$((N_PAIRS * 2)) verify runs)"

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

# --- 2. Build (or reuse) both cli binaries ---
need_build=0
if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$WORK/cli_A" ] || [ ! -x "$WORK/cli_B" ]; then
  need_build=1
elif [ "$(cat "$WORK/cli_A.sha" 2>/dev/null)" != "$SHA_A $BENCH_FEATURES" ] || \
     [ "$(cat "$WORK/cli_B.sha" 2>/dev/null)" != "$SHA_B $BENCH_FEATURES" ]; then
  echo "==> Cached binaries are for different refs/features; rebuilding."
  need_build=1
fi
if [ "$need_build" = "1" ]; then
  cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; }
  trap cleanup EXIT
  git worktree remove --force "$WT" 2>/dev/null || true
  echo "==> Building both cli binaries in isolated worktree $WT"
  git worktree add --detach "$WT" "$SHA_B" >/dev/null
  build_cli() {  # $1=sha $2=out (shared target dir -> 2nd build is incremental)
    echo "==> Building cli @ ${1:0:10} -> $2  (features: $BENCH_FEATURES)"
    git -C "$WT" checkout --quiet -f "$1"
    if ! ( cd "$WT" && cargo build --release -p cli --features "$BENCH_FEATURES" >"$WORK/build_$2.log" 2>&1 ); then
      echo "ERROR: cargo build failed for $2 (@ ${1:0:10}). Tail of $WORK/build_$2.log:" >&2
      tail -40 "$WORK/build_$2.log" >&2
      exit 1
    fi
    cp "$WT/target/release/cli" "$WORK/$2"
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

# --- 3. Prove, then interleaved A/B/B/A verify measurement ---
# Default: both sides verify ONE shared proof (proved by the baseline), which gives
# ABBA the tightest precision (no proof-specific variance). That only works when both
# binaries share the same proof (de)serialization format. A PR that changes the proof
# format cannot deserialize the baseline's proof, so we detect that and fall back to
# per-side proofs (each binary proves and verifies its own). PROVE_PER_SIDE overrides.
PROVE_PER_SIDE="${PROVE_PER_SIDE:-auto}"

prove_once() {  # $1=binary $2=proof-path
  if ! "$1" prove "$ELF" --private-input "$INPUT" -o "$2" --time >"$WORK/prove_$(basename "$2").log" 2>&1; then
    echo "ERROR: prove failed for $1. Tail of log:" >&2
    tail -20 "$WORK/prove_$(basename "$2").log" >&2
    exit 1
  fi
}
verify_time() {  # $1=binary $2=proof-path -> echoes time on success, empty on failure (never exits)
  local out
  out="$("$1" verify "$2" "$ELF" --time 2>&1)" || true
  printf '%s\n' "$out" | grep -o 'Verification time: [0-9.]*' | awk '{print $3}' || true
}
run_verify() {  # $1=binary $2=proof-path -> echoes verification time (s), exits on failure
  local t
  t="$(verify_time "$1" "$2")"
  if [ -z "$t" ]; then
    echo "ERROR: could not parse 'Verification time' from '$1 verify $2':" >&2
    "$1" verify "$2" "$ELF" --time >&2 2>&1 || true
    echo "HINT: if REF_A changes the proof format, run with PROVE_PER_SIDE=1." >&2
    exit 1
  fi
  echo "$t"
}

# Both sides prove their own proof (needed for the proof-size row; per-side verify
# needs both). Proofs are cached in $WORK like the binaries, marker
# "<sha> <features> <ELF+input hash>". Bytes are non-deterministic (parallel grinding)
# but size + verify cost are structural, so reusing a cached proof is valid. The prove
# call passes no proof-option flags; if it ever gains one (--blowup, ...), add it to the marker.
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi; }
PROOF_KEY_INPUT="$(cat "$ELF" "$INPUT" | sha256_of | cut -c1-16)"
prove_cached() {  # $1=binary $2=proof-path $3=sha
  local marker="$3 $BENCH_FEATURES $PROOF_KEY_INPUT"
  if [ "${REBUILD:-0}" != "1" ] && [ -f "$2" ] && [ "$(cat "$2.sha" 2>/dev/null)" = "$marker" ]; then
    echo "==> Reusing cached proof for ${3:0:10} ($(basename "$2"))"
  else
    echo "==> Proving with $(basename "$1") (${3:0:10})"
    prove_once "$1" "$2"
    echo "$marker" > "$2.sha"
  fi
}
prove_cached "$WORK/cli_B" "$PROOF_B" "$SHA_B"
prove_cached "$WORK/cli_A" "$PROOF_A" "$SHA_A"

# Proof sizes (bytes) for the Proof size row.
SIZE_B="$(wc -c < "$PROOF_B" | tr -d '[:space:]')"
SIZE_A="$(wc -c < "$PROOF_A" | tr -d '[:space:]')"

# Decide whether both sides can VERIFY one shared proof (baseline's) — tightest
# timing precision — or must verify their own (when the PR changes the proof format).
per_side=0
case "$PROVE_PER_SIDE" in
  1) per_side=1 ;;
  0) per_side=0 ;;
  *)  # auto: can the PR binary deserialize + verify the baseline's proof?
      if [ -z "$(verify_time "$WORK/cli_A" "$PROOF_B")" ]; then
        echo "==> PR binary cannot verify the baseline's proof (likely a proof-format change);"
        echo "    verifying per-side (set PROVE_PER_SIDE=0 to force shared)."
        per_side=1
      fi ;;
esac

if [ "$per_side" = "1" ]; then
  MODE="per-side"
  echo "==> Per-side verify: each binary verifies its OWN proof."
  PROOF_FOR_A="$PROOF_A"
  PROOF_FOR_B="$PROOF_B"
else
  MODE="shared"
  echo "==> Shared verify: both sides verify the baseline's proof (best precision)."
  PROOF_FOR_A="$PROOF_B"
  PROOF_FOR_B="$PROOF_B"
fi

echo "==> Running $N_PAIRS interleaved pairs  (improvement: + = PR faster)"
printf 'pair,a_time,b_time\n' > "$WORK/pairs.csv"
for i in $(seq 1 "$N_PAIRS"); do
  if [ $((i % 2)) -eq 1 ]; then          # odd pair: A then B
    a="$(run_verify "$WORK/cli_A" "$PROOF_FOR_A")"; b="$(run_verify "$WORK/cli_B" "$PROOF_FOR_B")"
  else                                   # even pair: B then A (ABBA pattern)
    b="$(run_verify "$WORK/cli_B" "$PROOF_FOR_B")"; a="$(run_verify "$WORK/cli_A" "$PROOF_FOR_A")"
  fi
  printf '%d,%s,%s\n' "$i" "$a" "$b" >> "$WORK/pairs.csv"
  printf '   pair %2d/%d   A=%ss  B=%ss   PR %+.2f%% (+=faster)\n' \
    "$i" "$N_PAIRS" "$a" "$b" "$(awk "BEGIN{print ($b-$a)/$b*100}")"
done
# Proofs are kept in $WORK as a cache (invalidated by their .sha markers), not deleted.

# --- 4. Paired t-test + robust median/Wilcoxon (same stats as bench_abba.sh) ---
SIZE_A="$SIZE_A" SIZE_B="$SIZE_B" MODE="$MODE" python3 - "$WORK/pairs.csv" <<'PY'
import sys, csv, math, os

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
# EXACT two-sided p via generating-function DP over the signed-rank null distribution.
if m:
    ir = [int(round(2 * r)) for r in ranks]
    poly = [1]
    for r in ir:
        nxt = [0] * (len(poly) + r)
        for v, c in enumerate(poly):
            if c:
                nxt[v] += c
                nxt[v + r] += c
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
seq = []
for i in range(n):
    seq += ([('A', A[i]), ('B', B[i])] if (i + 1) % 2 else [('B', B[i]), ('A', A[i])])
nrm = [(t / (mA if lbl == 'A' else mB) - 1) * 100 for lbl, t in seq]
N = len(nrm); mi = (N - 1) / 2.0; mn = sum(nrm) / N
denom = sum((i - mi) ** 2 for i in range(N))
slope = (sum((i - mi) * (nrm[i] - mn) for i in range(N)) / denom) if denom else 0.0
half = N // 2
drift_shift = sum(nrm[half:]) / (N - half) - sum(nrm[:half]) / half

# Markdown table (rendered directly in the PR comment) + paired detail.
sign = lambda v: f"+{v:.2f}" if v >= 0 else f"{v:.2f}"
icon = "🟢" if (lo > 0 and p < 0.05) else "🔴" if (hi < 0 and p < 0.05) else "⚪"
print("\n=== Verify ABBA result ===")
print()

# Proof size row: exact (the .bin byte size), no ABBA. + = PR smaller = better.
size_b = float(os.environ.get('SIZE_B', 0))   # main
size_a = float(os.environ.get('SIZE_A', 0))   # PR
size_impr = (size_b - size_a) / size_b * 100.0 if size_b else 0.0
size_icon = "🟢" if size_impr > 0.005 else "🔴" if size_impr < -0.005 else "⚪"
to_mib = lambda b: b / (1024.0 * 1024.0)

print("| Metric | main | PR | Δ |")
print("|--------|------|----|---|")
print(f"| **Verify time** | {mB:.3f}s | {mA:.3f}s | {sign(mean)}% {icon} |")
print(f"| **Proof size** | {to_mib(size_b):.2f} MiB | {to_mib(size_a):.2f} MiB | {sign(size_impr)}% {size_icon} |")
print()
print("```")
print(f"  pairs: {n}   mean A (PR): {mA:.3f}s   mean B (main): {mB:.3f}s")
print(f"  [parametric] paired-t   mean {mean:+.2f}%   sd {sd:.2f}%   se {se:.2f}%")
print(f"               95% CI: [{lo:+.2f}%, {hi:+.2f}%]   (t df={df} = {tc})")
pstr = f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"
print(f"  [robust]     median {med:+.2f}%   Wilcoxon W+={Wp:.0f} W-={Wn:.0f}  p(exact)={pstr}  (z={z:+.2f})")
print()
print(f"  run-to-run jitter:    A CV {cvA:.2f}%   B CV {cvB:.2f}%        (lower = steadier)")
print(f"  within-session drift: {slope * N:+.2f}% over the run, 1st->2nd half {drift_shift:+.2f}%")
print("```")
if lo > 0 and p < 0.05:
    print(f"\n> 🟢 **REAL IMPROVEMENT** — PR verifies ~{mean:.2f}% faster (paired-t and Wilcoxon agree).")
elif hi < 0 and p < 0.05:
    print(f"\n> 🔴 **REAL REGRESSION** — PR verifies ~{-mean:.2f}% slower (paired-t and Wilcoxon agree).")
elif (lo > 0) != (p < 0.05):
    print(f"\n> ⚪ **BORDERLINE** — parametric and robust disagree; suspect outlier pair(s). Trust the median ({med:+.2f}%); add pairs.")
else:
    print(f"\n> ⚪ **INCONCLUSIVE** — effect not separable from 0 at n={n} (point estimate ~{med:+.2f}%). Add pairs to resolve.")
PY
