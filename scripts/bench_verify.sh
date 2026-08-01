#!/usr/bin/env bash
#
# bench_verify.sh — interleaved A/B/B/A paired verifier benchmark (PR vs main).
# Reported % = (PR - baseline)/baseline, matching the classic /bench:
# NEGATIVE numbers are improvements (PR faster/smaller); positive = regression.
#
# Arms over one block, both at blowup=2 / 219 queries:
#   monolithic     one VmProof for the whole execution.
#   continuations  the same block proved as 2^CONT_EPOCH_LOG2-cycle epochs and verified
#                  as a ContinuationProof bundle — what /bench proves and what an L2
#                  actually runs, so a verifier change that only moves per-epoch or
#                  aggregation cost is invisible in the monolithic arm alone.
# WORKLOAD picks the block: `synthetic` (default) is the ethrex 20-tx one and runs both
# arms; `real` runs the continuation arm only, since a real block does not fit
# monolithically. See "Workload" below.
# The continuation arm is best-effort: if its prove or verify fails (it is the
# memory-hungry one) the arm is skipped with a note and the monolithic verdict still
# posts, rather than failing the whole bench.
#
# Usage: scripts/bench_verify.sh REF_A [REF_B=origin/main] [N_PAIRS=20]
#   REF_A/REF_B  refs to compare (A = PR side); N_PAIRS even, default 20 (~5-6 min).
#   Env: REBUILD=1 forces rebuild + re-prove; BENCH_FEATURES=<list> (default: jemalloc-stats).
#        PROVE_PER_SIDE=auto|1|0 (default auto): 1 = each side proves+verifies its
#        own proof (required when REF_A changes the proof format); 0 = force one
#        shared proof (best precision); auto = share if the PR binary can verify the
#        baseline's proof, else fall back to per-side. Decided per arm.
#        CONT_PAIRS=<n> pairs for the continuation arm (even, default 8; 0 skips it).
#        Fewer than N_PAIRS because one continuation verify covers every epoch proof
#        plus the aggregation, so it costs multiples of a monolithic verify. Don't go
#        below 6: the exact Wilcoxon's smallest attainable two-sided p is 2/2^n, so at
#        n=4 it is 0.125 and the arm can only ever report BORDERLINE, however large and
#        clean the effect.
#        WORKLOAD=synthetic|real (default synthetic) which BLOCK both arms prove.
#        `real` fetches the real-block fixture (identity lives in the Makefile) and runs
#        the continuation arm ONLY — a real block is hundreds of GB monolithically, so
#        that arm is skipped rather than left to OOM. See "Workload" below.
#        CONT_EPOCH_LOG2=<n> continuation epoch size (default 20, min 18). For
#        WORKLOAD=real prefer the calibrated tier for the box you are on — 2^22 on the
#        bench runner or a 64 GiB machine, 2^23 on a 128 GiB one (see
#        tooling/ethrex-block-converter/README.md, "Choosing the epoch size"); the default
#        below is chosen for the SYNTHETIC arm. 20 matches
#        scripts/bench_abba.sh, so this arm proves the same bundle shape /bench-abba
#        already proves on the same server — and 20 txs at 2^20 is strictly cheaper than
#        the 100 txs at 2^20 that /bench-abba runs by default, so it can't be the thing
#        that OOMs the box. (`cli prove --epoch-size-log2 --help` measured ethrex 10tx at ~9.5 GB
#        for 2^20 vs ~15.8 GB for 2^21.) Note this does NOT match
#        bench_recursion_cycles.sh's BLOCK_EPOCH_LOG2=21: that arm needs FEW epochs so
#        the bundle fits the guest's 512 MiB private-input cap, a constraint that
#        doesn't apply to host-side verification.
#
# Workload. What this script measures is VERIFY cost, and verify cost is structural in
# the proof — table mix, trace lengths, query count — so the block decides what is being
# measured, not the loop around it. The synthetic default is 20 plain transfers:
# ecrecover-heavy over a near-empty state. A real block inverts that mix (keccak- and
# trie-bound), so a verifier change can move the two differently.
#
# `synthetic` stays the default because it is what `/bench-verify` runs and what every
# number recorded so far used; `real` is the representative one, and costs a ~2.6 min
# continuation prove per side (cached in $WORK afterwards) before any verify run.
# Both sides always prove the same block, so a comparison is never mixed.

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
CONT_PAIRS="${CONT_PAIRS:-8}"
CONT_EPOCH_LOG2="${CONT_EPOCH_LOG2:-20}"
WORKLOAD="${WORKLOAD:-synthetic}"
case "$WORKLOAD" in
  synthetic|real) ;;
  *) echo "ERROR: WORKLOAD must be 'synthetic' or 'real' (got '$WORKLOAD')." >&2; exit 2 ;;
esac
# WORKLOAD=real skips the monolithic arm, so CONT_PAIRS=0 on top of it would build both
# binaries, prove nothing and report nothing. Fail before the ~30-min build rather than
# after it.
if [ "$WORKLOAD" = "real" ] && [ "${CONT_PAIRS}" -eq 0 ] 2>/dev/null; then
  echo "ERROR: WORKLOAD=real runs the continuation arm only, so CONT_PAIRS=0 would measure nothing." >&2
  exit 2
fi

ELF_REL="executor/program_artifacts/rust/ethrex.elf"
# Resolved after the cd to the repo root (WORKLOAD=real reads it from the Makefile).
INPUT_REL=""
WORK="/tmp/verify_run"
WT="/tmp/verify_wt"
PROOF_B="$WORK/proof_b.bin"   # baseline's proof (cached in $WORK, keyed like the binaries)
PROOF_A="$WORK/proof_a.bin"   # PR's proof (cached likewise)
CPROOF_B="$WORK/cproof_b.bin" # baseline's continuation bundle (cached likewise)
CPROOF_A="$WORK/cproof_a.bin" # PR's continuation bundle (cached likewise)

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
# Validate both counts BEFORE any building/proving. Two ways a bad value bites otherwise:
# under `set -u` a non-numeric one makes the arithmetic below die with a bare
# "abc: unbound variable", and a value that makes `seq` produce nothing yields a
# header-only pairs CSV whose ZeroDivisionError only surfaces in the stats step at the
# very END — after both arms have been measured, so the run fails with nothing to show
# and CI posts "Run failed" instead of the results it already had.
for v in N_PAIRS CONT_PAIRS; do
  if ! [[ "${!v}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $v must be a non-negative integer (got '${!v}')." >&2
    exit 2
  fi
done
if [ "$N_PAIRS" -lt 2 ]; then
  echo "ERROR: N_PAIRS must be >= 2 (got $N_PAIRS)." >&2
  exit 2
fi
if [ "$CONT_PAIRS" -eq 1 ]; then
  echo "ERROR: CONT_PAIRS must be 0 (skip the arm) or >= 2 (got $CONT_PAIRS)." >&2
  exit 2
fi
# CONT_EPOCH_LOG2 is the one knob with a hard floor (MIN_CONTINUATION_EPOCH_SIZE_LOG2 in
# bin/cli/src/main.rs). Catch it here rather than letting clap reject it after two cli
# builds and the whole monolithic arm, which would then degrade to a bland
# "continuation prove failed" note that doesn't say why.
if ! [[ "$CONT_EPOCH_LOG2" =~ ^[0-9]+$ ]] || [ "$CONT_EPOCH_LOG2" -lt 18 ]; then
  echo "ERROR: CONT_EPOCH_LOG2 must be an integer >= 18 (got '$CONT_EPOCH_LOG2')." >&2
  exit 2
fi
# A warning, not an error: 2..5 pairs is a legitimate quick smoke run, and the monolithic
# arm already accepts N_PAIRS=2 (the workflow clamps its own input to [2,40]). Just say
# the verdict can't reach significance so nobody reads BORDERLINE as a real result.
if [ "$CONT_PAIRS" -ge 2 ] && [ "$CONT_PAIRS" -lt 6 ]; then
  echo "   WARNING: CONT_PAIRS=$CONT_PAIRS < 6; the exact Wilcoxon's smallest attainable"
  echo "            two-sided p is 2/2^n, so this arm can only ever report BORDERLINE."
fi
if [ $((N_PAIRS % 2)) -ne 0 ]; then
  echo "   WARNING: N_PAIRS=$N_PAIRS is odd; use an even count so AB/BA orders balance."
fi
if [ $((CONT_PAIRS % 2)) -ne 0 ]; then
  echo "   WARNING: CONT_PAIRS=$CONT_PAIRS is odd; use an even count so AB/BA orders balance."
fi
# The real block's identity lives in the Makefile and nowhere else, so repointing it
# never needs an edit here. A real block cannot be proven monolithically (~4.9 GB of peak
# heap per million cycles puts it in the hundreds of GB), so that arm is skipped outright
# rather than left to OOM halfway through a rented or shared machine's run.
RUN_MONO=1
if [ "$WORKLOAD" = "real" ]; then
  INPUT_REL="$(make -s print-real-block-fixture)"
  RUN_MONO=0
  BLOCK_LABEL="ethrex real block $(basename "$INPUT_REL")"
else
  INPUT_REL="executor/tests/ethrex_bench_20.bin"
  BLOCK_LABEL="ethrex 20-tx block"
fi

echo "   workload=$WORKLOAD  $INPUT_REL"
if [ "$RUN_MONO" = "1" ]; then
  echo "   pairs=$N_PAIRS  (=$((N_PAIRS * 2)) verify runs)"
else
  echo "   monolithic arm SKIPPED (a real block exceeds the monolithic memory ceiling)"
fi
echo "   continuation pairs=$CONT_PAIRS  epoch=2^$CONT_EPOCH_LOG2"

mkdir -p "$WORK"
# Drop any previous run's monolithic report before measuring. It is the CI fallback for a
# run that dies mid-continuation-arm, so a leftover from an earlier run would be posted as
# if it belonged to this one.
rm -f "$WORK/result_mono.txt"

# --- 1. Guest ELF + fixture (identical for both sides; build once if missing) ---
if [ ! -f "$ELF_REL" ]; then
  echo "==> Building ethrex guest ELF (missing)"
  export SYSROOT_DIR="${SYSROOT_DIR:-$HOME/.lambda-vm-sysroot}"
  make "$ELF_REL"
fi
if [ "$WORKLOAD" = "real" ]; then
  # ~1 MB, gitignored, never in a fresh checkout. Fetched by URL + sha256, not built:
  # no converter and no ethrex host dependency tree on this path. Unconditional on
  # purpose: the target hashes whatever is already on disk on every invocation, which
  # is how a stale copy left by an earlier block or an interrupted fetch gets caught.
  # A match costs ~35 ms, so there is nothing to gate it on.
  echo "==> Verifying ethrex real-block fixture (fetches on a digest miss)"
  make ethrex-real-block-fixture
elif [ ! -f "$INPUT_REL" ]; then
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

# The trailing "$@" on each of these carries the arm's extra flags: empty for the
# monolithic arm, `--continuations` (plus `--epoch-size-log2` when proving) for the
# continuation one. Failures RETURN non-zero instead of exiting so the caller decides
# whether the arm is fatal (monolithic) or skippable (continuations).
prove_once() {  # $1=binary $2=proof-path $3...=extra prove args
  local bin="$1" out="$2"; shift 2
  if ! "$bin" prove "$ELF" --private-input "$INPUT" -o "$out" --time "$@" \
       >"$WORK/prove_$(basename "$out").log" 2>&1; then
    echo "ERROR: prove failed for $bin. Tail of log:" >&2
    tail -20 "$WORK/prove_$(basename "$out").log" >&2
    return 1
  fi
}
verify_time() {  # $1=binary $2=proof-path $3...=extra verify args -> time, empty on failure
  local bin="$1" proof="$2"; shift 2
  local out
  out="$("$bin" verify "$proof" "$ELF" --time "$@" 2>&1)" || true
  printf '%s\n' "$out" | grep -o 'Verification time: [0-9.]*' | awk '{print $3}' || true
}
run_verify() {  # $1=binary $2=proof-path $3...=extra verify args -> time (s), 1 on failure
  local bin="$1" proof="$2"; shift 2
  local t
  t="$(verify_time "$bin" "$proof" "$@")"
  if [ -z "$t" ]; then
    echo "ERROR: could not parse 'Verification time' from '$bin verify $proof $*':" >&2
    "$bin" verify "$proof" "$ELF" --time "$@" >&2 2>&1 || true
    echo "HINT: if REF_A changes the proof format, run with PROVE_PER_SIDE=1." >&2
    return 1
  fi
  echo "$t"
}

# Both sides prove their own proof (needed for the proof-size row; per-side verify
# needs both). Proofs are cached in $WORK like the binaries, marker
# "<sha> <features> <ELF+input hash>". Bytes are non-deterministic (parallel grinding)
# but size + verify cost are structural, so reusing a cached proof is valid. Any extra
# prove flags (--continuations, --epoch-size-log2) go into the marker too; if the call
# ever gains one that is NOT passed through here (--blowup, ...), add it as well.
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi; }
PROOF_KEY_INPUT="$(cat "$ELF" "$INPUT" | sha256_of | cut -c1-16)"
prove_cached() {  # $1=binary $2=proof-path $3=sha $4...=extra prove args
  local bin="$1" out="$2" sha="$3"; shift 3
  # The extra args are part of the marker: a monolithic and a continuation proof of the
  # same (ref, features, ELF+input) must never share a cache entry.
  local marker="$sha $BENCH_FEATURES $PROOF_KEY_INPUT $*"
  if [ "${REBUILD:-0}" != "1" ] && [ -f "$out" ] && [ "$(cat "$out.sha" 2>/dev/null)" = "$marker" ]; then
    echo "==> Reusing cached proof for ${sha:0:10} ($(basename "$out"))"
    return 0
  fi
  echo "==> Proving with $(basename "$bin") (${sha:0:10}) $*"
  # Wipe the old sidecar before proving: on failure neither it nor the .sha is rewritten,
  # so a previous run's count would survive and get printed next to the NEW epoch size —
  # e.g. change CONT_EPOCH_LOG2, have the prove fail, and the skip note claims the old
  # epoch count for a bundle that no longer exists.
  rm -f "$out.epochs"
  prove_once "$bin" "$out" "$@" || return 1
  # Persist the epoch count next to the proof rather than parsing the prove log at report
  # time: on a cache hit the prove is skipped entirely, so that log is stale or gone. The
  # sidecar is written with the proof and invalidated with it. Continuation proves only —
  # `cli prove` prints no "Epochs:" line in monolithic mode, so no sidecar appears there.
  local ep
  ep="$(awk -F': ' '/^Epochs:/{print $2; exit}' "$WORK/prove_$(basename "$out").log")"
  # `if` rather than `[ -n "$ep" ] && ...`: a failing &&-list is fatal under `set -e`.
  if [ -n "$ep" ]; then printf '%s\n' "$ep" > "$out.epochs"; fi
  echo "$marker" > "$out.sha"
}
SIZE_B=0; SIZE_A=0
if [ "$RUN_MONO" = "1" ]; then
  prove_cached "$WORK/cli_B" "$PROOF_B" "$SHA_B" || exit 1
  prove_cached "$WORK/cli_A" "$PROOF_A" "$SHA_A" || exit 1

  # Proof sizes (bytes) for the Proof size row.
  SIZE_B="$(wc -c < "$PROOF_B" | tr -d '[:space:]')"
  SIZE_A="$(wc -c < "$PROOF_A" | tr -d '[:space:]')"
fi

# Decide whether both sides can VERIFY one shared proof (baseline's) — tightest
# timing precision — or must verify their own. In auto mode, distinguish a real
# proof-format change (deserialize error → benign) from the PR *rejecting* a valid
# baseline proof (a verify regression). Both fall back to per-side, but they mean
# very different things, so carry the reason into the report — otherwise a real
# backward-compat break gets silently reclassified as a format change and shown green.
# Decided per arm (sets MODE/PER_SIDE_NOTE/PROOF_FOR_A/PROOF_FOR_B): a PR can change the
# continuation bundle format without touching the monolithic one, or vice versa.
decide_mode() {  # $1=baseline proof $2=PR proof $3...=extra verify args
  local pb="$1" pa="$2"; shift 2
  local per_side=0
  PER_SIDE_NOTE=""
  case "$PROVE_PER_SIDE" in
    1) per_side=1; PER_SIDE_NOTE="forced via PROVE_PER_SIDE=1" ;;
    0) per_side=0 ;;
    *)  local probe
        probe="$("$WORK/cli_A" verify "$pb" "$ELF" --time "$@" 2>&1 || true)"
        if printf '%s\n' "$probe" | grep -q 'Verification time'; then
          per_side=0                                   # PR verifies main's proof -> shared
        elif printf '%s\n' "$probe" | grep -q 'Failed to deserialize'; then
          per_side=1
          PER_SIDE_NOTE="PR can't deserialize the baseline's proof — proof-format change"
          echo "==> $PER_SIDE_NOTE; verifying per-side."
        else
          per_side=1
          PER_SIDE_NOTE="⚠️ PR REJECTS the baseline's valid proof — likely a VERIFY REGRESSION, not a format change"
          echo "==> $PER_SIDE_NOTE"
          echo "    verifying per-side, but the Verify-time numbers below are NOT a safe signal."
        fi ;;
  esac
  if [ "$per_side" = "1" ]; then
    MODE="per-side"
    echo "==> Per-side verify: each binary verifies its OWN proof."
    PROOF_FOR_A="$pa"
    PROOF_FOR_B="$pb"
  else
    MODE="shared"
    echo "==> Shared verify: both sides verify the baseline's proof (best precision)."
    PROOF_FOR_A="$pb"
    PROOF_FOR_B="$pb"
  fi
}

run_abba() {  # $1=pairs $2=csv $3...=extra verify args
  local pairs="$1" csv="$2"; shift 2
  local i a b
  echo "==> Running $pairs interleaved pairs  (improvement: - = PR faster)"
  printf 'pair,a_time,b_time\n' > "$csv"
  for i in $(seq 1 "$pairs"); do
    if [ $((i % 2)) -eq 1 ]; then          # odd pair: A then B
      a="$(run_verify "$WORK/cli_A" "$PROOF_FOR_A" "$@")" || return 1
      b="$(run_verify "$WORK/cli_B" "$PROOF_FOR_B" "$@")" || return 1
    else                                   # even pair: B then A (ABBA pattern)
      b="$(run_verify "$WORK/cli_B" "$PROOF_FOR_B" "$@")" || return 1
      a="$(run_verify "$WORK/cli_A" "$PROOF_FOR_A" "$@")" || return 1
    fi
    printf '%d,%s,%s\n' "$i" "$a" "$b" >> "$csv"
    printf '   pair %2d/%d   A=%ss  B=%ss   PR %+.2f%% (-=faster)\n' \
      "$i" "$pairs" "$a" "$b" "$(awk "BEGIN{print ($a-$b)/$b*100}")"
  done
}

# --- 4. Paired t-test + robust median/Wilcoxon (same stats as bench_abba.sh) ---
# Both arms are reported AFTER all measuring is done: bench-verify.yml extracts the PR
# comment with `sed -n '/<!-- verify-abba-report -->/,$p'`, so anything printed between
# the two tables (per-pair progress) would land in the comment.
print_stats() {  # $1=csv $2=title $3=size_a $4=size_b $5=mode $6=note
  TITLE="$2" SIZE_A="$3" SIZE_B="$4" MODE="$5" PER_SIDE_NOTE="$6" python3 - "$1" <<'PY'
import sys, csv, math, os

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
icon = "🟢" if (hi < 0 and p < 0.05) else "🔴" if (lo > 0 and p < 0.05) else "⚪"
mode = os.environ.get('MODE', 'shared')
per_side_note = os.environ.get('PER_SIDE_NOTE', '')

print(f"\n#### {os.environ.get('TITLE', '')}")
print()

# Proof size row: exact (the .bin byte size), no ABBA. - = PR smaller = better.
size_b = float(os.environ.get('SIZE_B', 0))   # main
size_a = float(os.environ.get('SIZE_A', 0))   # PR
size_impr = (size_a - size_b) / size_b * 100.0 if size_b else 0.0
size_icon = "🟢" if size_impr < -0.005 else "🔴" if size_impr > 0.005 else "⚪"
to_mib = lambda b: b / (1024.0 * 1024.0)

# Say per row how it was measured. Only the timing row is ABBA; the byte size is one
# exact reading per side. Without this the reader applies the ABBA/statistics framing to
# every number in the comment, including the ones it does not describe.
# In per-side mode A and B verify different proofs, so label that too (M2).
vt_qual = f"ABBA, {n} pairs, per-side" if mode == "per-side" else f"ABBA, {n} pairs"
print("| Metric | main | PR | Δ |")
print("|--------|------|----|---|")
print(f"| **Verify time** ({vt_qual}) | {mB:.3f}s | {mA:.3f}s | {sign(mean)}% {icon} |")
print(f"| **Proof size** (exact, 1 reading) | {to_mib(size_b):.2f} MiB | {to_mib(size_a):.2f} MiB | {sign(size_impr)}% {size_icon} |")

# Surface why per-side kicked in (format change vs possible regression) so a green
# table can't silently hide a backward-compat verify break (M1/M2).
if mode == "per-side":
    print()
    print(f"> **Per-side** ({per_side_note or 'each side verified its own proof'}): "
          "A/B/B/A cancels machine drift but not proof-specific variance — read the Verify-time Δ as approximate.")
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
if hi < 0 and p < 0.05:
    print(f"\n> 🟢 **REAL IMPROVEMENT** — PR verifies ~{-mean:.2f}% faster (paired-t and Wilcoxon agree).")
elif lo > 0 and p < 0.05:
    print(f"\n> 🔴 **REAL REGRESSION** — PR verifies ~{mean:.2f}% slower (paired-t and Wilcoxon agree).")
elif (hi < 0) != (p < 0.05):
    print(f"\n> ⚪ **BORDERLINE** — parametric and robust disagree; suspect outlier pair(s). Trust the median ({med:+.2f}%); add pairs.")
else:
    print(f"\n> ⚪ **INCONCLUSIVE** — effect not separable from 0 at n={n} (point estimate ~{med:+.2f}%). Add pairs to resolve.")
PY
}

MONO_SKIP=""
if [ "$RUN_MONO" = "1" ]; then
  decide_mode "$PROOF_B" "$PROOF_A"
  run_abba "$N_PAIRS" "$WORK/pairs.csv" || exit 1
  MONO_MODE="$MODE"
  MONO_NOTE="$PER_SIDE_NOTE"
else
  MONO_SKIP="not run: a real block exceeds the monolithic memory ceiling, so only the continuation arm below applies"
fi
# Render the monolithic report NOW, not at the end with the continuation one. Both are
# still emitted together at the end (the extractor needs them contiguous after the
# anchor), but computing this one here means it also survives the run dying during the
# continuation arm. The best-effort CONT_SKIP path only covers a clean non-zero exit; a
# step timeout or an OOM kill takes the whole process down, and CI would then post
# "Run failed" and throw away a monolithic verdict it had already measured. bench-verify.yml
# falls back to this file in that case.
MONO_TITLE="$BLOCK_LABEL · monolithic · blowup=2, 219 queries"
if [ -n "$MONO_SKIP" ]; then
  MONO_REPORT="$(printf '#### %s\n\n_(Monolithic arm %s.)_\n' "$MONO_TITLE" "$MONO_SKIP")"
else
  MONO_REPORT="$(print_stats "$WORK/pairs.csv" "$MONO_TITLE" \
    "$SIZE_A" "$SIZE_B" "$MONO_MODE" "$MONO_NOTE")"
fi
printf '%s\n' "$MONO_REPORT" > "$WORK/result_mono.txt"

# --- 3b. Same measurement over a CONTINUATION bundle of the same block ---------
# Best-effort: this is the memory-hungry arm (the whole bundle is materialised to
# serialize it), so any failure here degrades to a note in the report instead of
# sinking the monolithic verdict above.
CONT_SKIP=""
CONT_ARGS=(--continuations --epoch-size-log2 "$CONT_EPOCH_LOG2")
if [ "$CONT_PAIRS" -eq 0 ]; then
  CONT_SKIP="skipped (CONT_PAIRS=0)"
elif ! prove_cached "$WORK/cli_B" "$CPROOF_B" "$SHA_B" "${CONT_ARGS[@]}"; then
  CONT_SKIP="baseline continuation prove failed"
elif ! prove_cached "$WORK/cli_A" "$CPROOF_A" "$SHA_A" "${CONT_ARGS[@]}"; then
  CONT_SKIP="PR continuation prove failed"
else
  CSIZE_B="$(wc -c < "$CPROOF_B" | tr -d '[:space:]')"
  CSIZE_A="$(wc -c < "$CPROOF_A" | tr -d '[:space:]')"
  decide_mode "$CPROOF_B" "$CPROOF_A" --continuations
  CONT_MODE="$MODE"
  CONT_NOTE="$PER_SIDE_NOTE"
  if ! run_abba "$CONT_PAIRS" "$WORK/pairs_cont.csv" --continuations; then
    CONT_SKIP="continuation verify failed mid-run"
  fi
fi
if [ -n "$CONT_SKIP" ]; then
  echo "==> Continuation arm $CONT_SKIP"
fi
# Proofs are kept in $WORK as a cache (invalidated by their .sha markers), not deleted.


echo
# Machine anchor for bench-verify.yml's extractor; an HTML comment so it doesn't render
# in the PR comment (the arm headings below are the human entry point).
echo "<!-- verify-abba-report -->"
# Arm titles follow the same `workload · mode · params` shape as the recursion cycle
# regimes (bench_recursion_cycles.sh), so every table in the PR comment says what it
# proved and how, and no two arms can be confused for each other.
# Epoch COUNT alongside the epoch SIZE: the size alone doesn't tell you the bundle shape,
# and the count is what drives verify cost and bundle size. Read from the sidecars written
# at prove time. Show both sides when they disagree — a PR that changes epoch splitting is
# exactly the kind of thing this arm should surface, not average away. Falls back to no
# parenthetical if either side is unknown (e.g. an older cached proof with no sidecar),
# because a wrong count is worse than a missing one. Only consulted when the arm actually
# ran: with CONT_PAIRS=0 nothing validates a bundle this run, so a sidecar left in the
# long-lived $WORK by an earlier run would otherwise be reported as this run's count.
epochs_of() { local f="$1.epochs"; [ -s "$f" ] && head -1 "$f" | tr -d '[:space:]' || true; }
CONT_EPOCHS=""
if [ -z "$CONT_SKIP" ]; then
  EPO_A="$(epochs_of "$CPROOF_A")"; EPO_B="$(epochs_of "$CPROOF_B")"
  if [ -n "$EPO_A" ] && [ -n "$EPO_B" ]; then
    if [ "$EPO_A" = "$EPO_B" ]; then
      CONT_EPOCHS=" ($EPO_A epochs)"
    else
      CONT_EPOCHS=" (main $EPO_B / PR $EPO_A epochs)"
    fi
  fi
fi
CONT_TITLE="$BLOCK_LABEL · continuations, epoch 2^$CONT_EPOCH_LOG2$CONT_EPOCHS · blowup=2, 219 queries"
printf '%s\n' "$MONO_REPORT"
if [ -n "$CONT_SKIP" ]; then
  echo
  echo "#### $CONT_TITLE"
  echo
  echo "_(Continuation arm $CONT_SKIP — see the workflow log. Does not affect the monolithic verdict above.)_"
else
  print_stats "$WORK/pairs_cont.csv" "$CONT_TITLE" \
    "$CSIZE_A" "$CSIZE_B" "$CONT_MODE" "$CONT_NOTE"
fi
