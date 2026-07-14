# bench_abba_common.sh — the statistics + helpers behind the ABBA prover bench.
#
# Sourced by scripts/bench_abba.sh. The paired-t + exact-Wilcoxon analysis is a
# large block; keeping it here lets the bench script stay focused on orchestration
# (build the two refs, run the A/B/B/A loop) rather than statistics.
#
# Provides:
#   extract_prove_time "<cli output>"   -> echoes the proving time (s) or dies
#   cudarc_pin_apply   <Cargo.toml>     -> CUDARC_PIN compat sed with no-op warn
#   abba_stats <pairs.csv> <label_A> <label_B> <header>
#                                       -> paired-t + exact Wilcoxon + stability

# Parse `Proving time: <s>` out of a cli prove run's output; die loudly (with
# the full output on stderr) when it is missing, so a crashed prove can't
# silently feed an empty sample into the stats.
extract_prove_time() {  # $1 = full cli output -> echoes time (s)
  local t
  t="$(printf '%s\n' "$1" | grep -o 'Proving time: [0-9.]*' | awk '{print $3}')"
  if [ -z "$t" ]; then
    echo "ERROR: could not parse 'Proving time' from cli output:" >&2
    printf '%s\n' "$1" >&2
    exit 1
  fi
  echo "$t"
}

# CUDARC_PIN: compat shim for benching *pre-pin* checkouts. Newer trees pin
# cudarc's CUDA version permanently in crypto/math-cuda/Cargo.toml (feature
# `cuda-12080`), so the `cuda-version-from-build-system` anchor is gone and the
# sed no-ops on them — warn rather than mislead. On an older checkout it still
# swaps in the pin + drops fallback-latest, so cudarc binds a known
# driver-symbol set instead of its newest (which can request symbols the rented
# box's driver doesn't export, e.g. cuDevSmResourceSplit -> runtime panic).
# No-op when CUDARC_PIN is unset.
cudarc_pin_apply() {  # $1 = path to the checkout's crypto/math-cuda/Cargo.toml
  [ -n "${CUDARC_PIN:-}" ] || return 0
  if grep -q '"cuda-version-from-build-system"' "$1"; then
    sed -i "s/\"cuda-version-from-build-system\"/\"${CUDARC_PIN}\"/; /\"fallback-latest\"/d" "$1"
    echo "    cudarc pinned to ${CUDARC_PIN}"
  else
    echo "    WARNING: CUDARC_PIN=${CUDARC_PIN} ignored — cudarc is already" >&2
    echo "             pinned in $1 (no build-system anchor to rewrite)." >&2
  fi
}

# Paired analysis over an ABBA pairs.csv (columns pair,a_time,b_time):
# paired-t 95% CI, robust median + EXACT Wilcoxon signed-rank, server-stability
# diagnostics, and a verdict. Labels name the two sides in the report
# (e.g. "PR"/"baseline" or "GPU ON"/"CPU"). Convention everywhere:
# delta = (A - B)/B, NEGATIVE = A faster.
abba_stats() {  # $1=pairs.csv  $2=label_A  $3=label_B  $4=report header
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys, csv, math

path, LA, LB, HDR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rows = list(csv.DictReader(open(path)))
A = [float(r['a_time']) for r in rows]
B = [float(r['b_time']) for r in rows]
n = len(A)
# per-pair delta = (A - B)/B: negative => A faster than B
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

print(f"\n=== {HDR}  (improvement: - = {LA} faster) ===")
print(f"  pairs: {n}   mean A ({LA}): {sum(A)/n:.3f}s   mean B ({LB}): {sum(B)/n:.3f}s")
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
    print(f"  VERDICT: REAL IMPROVEMENT - {LA} faster by ~{-mean:.2f}% (t-CI and Wilcoxon agree)")
elif lo > 0 and p < 0.05:
    print(f"  VERDICT: REAL REGRESSION - {LA} slower by ~{mean:.2f}% (t-CI and Wilcoxon agree)")
elif (hi < 0) != (p < 0.05):
    print(f"  VERDICT: BORDERLINE - parametric and robust disagree; suspect outlier pair(s).")
    print(f"           Trust the median ({med:+.2f}%); add pairs or inspect the per-pair list.")
else:
    print(f"  VERDICT: INCONCLUSIVE - effect not separable from 0 at n={n}.")
    print(f"           Point estimate ~{med:+.2f}% (median). Need more pairs to resolve.")
print(f"\n  raw pairs: {path}")
PY
}
