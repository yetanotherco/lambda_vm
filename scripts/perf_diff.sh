#!/usr/bin/env bash
#
# perf_diff.sh — symbol-level profile diff of two prover builds on the ethrex
# fixture. Companion to bench_abba.sh: once ABBA says a regression is REAL,
# this localizes it — `perf diff` reports per-symbol self-time deltas between
# the two binaries, which is the ground truth the source-level audits can't
# see (inlining, register pressure, allocator time).
#
# Builds mirror bench_abba.sh exactly (release, jemalloc-stats) plus debug
# symbols (CARGO_PROFILE_RELEASE_DEBUG=1 — see the note in the workspace
# Cargo.toml); debug=1 does not change optimization, so the profiled binary
# is the benched binary.
#
# USAGE (on the bench server):
#   scripts/perf_diff.sh REF_A [REF_B=origin/main]
#   Env: WORKLOAD=real|synthetic (default real) picks the block to profile;
#        EPOCH_SIZE_LOG2=<n> (default 22) sizes the epoch, WORKLOAD=real only.
#          22 is the calibrated bench-runner tier, matching /bench; use 23 on a
#          128 GiB box (tooling/ethrex-block-converter/README.md, "Choosing the epoch size").
#
# Pick the workload that matches the run you are localizing, because the symbol
# mix follows the block: the real default is 50.78M cycles, 10,478 keccak calls and
# 116 ecsm calls, and the synthetic option (20 plain transfers) inverts that at
# 8.73M cycles, 411 keccak, 80 ecsm — so a hot symbol in one need not be hot in the other.
# Both counts are from the same guest ELF (merge fdb92f67, main @ 9ccdaf2, clang 21);
# they move with guest optimisation (#861's thin LTO) and ~2% with the clang major, so
# pin the ELF when quoting one.
#
# WORKLOAD=real also switches to a continuation prove (monolithic would need ~240 GB
# at that trace length), which is 158.8 s per recording on the bench runner — five
# recordings, so budget ~13 min of proving, plus ~1.2 GB of disk per bundle and ~52 GB
# of RAM at the default epoch.
#
# Produces:
#   - two perf-diff tables (recorded twice per side, interleaved B A B A —
#     symbols whose delta repeats across both tables are real, one-off
#     deltas are sampling noise)
#   - top self-time report per side
# Requires: perf. If kernel.perf_event_paranoid > 2, run:
#   sudo sysctl kernel.perf_event_paranoid=1

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: perf_diff.sh REF_A [REF_B=origin/main]" >&2
  exit 2
fi
REF_A="$1"
REF_B="${2:-origin/main}"
WORKLOAD="${WORKLOAD:-real}"
# 2^22: the calibrated tier for the bench server this script targets, same as
# /bench's real-block arm. Memory picks it, not speed — that server peaks at ~52 GB on
# a >=64 GiB floor, and 2^23 measured 60 GiB on a roomier box, so it would not fit here.
EPOCH_SIZE_LOG2="${EPOCH_SIZE_LOG2:-22}"
case "$WORKLOAD" in
  synthetic|real) ;;
  *) echo "ERROR: WORKLOAD must be 'synthetic' or 'real' (got '$WORKLOAD')." >&2; exit 2 ;;
esac

ELF_REL="executor/program_artifacts/rust/ethrex.elf"
WORK="/tmp/perf_diff"
WT="/tmp/perf_diff_wt"
PROOF="/tmp/perf_diff_proof.bin"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# The real block's name lives in the Makefile and nowhere else, so repointing the
# benchmark to a different block never needs an edit here.
if [ "$WORKLOAD" = "real" ]; then
  INPUT_REL="$(make -s print-real-block-fixture)"
  CONT_ARGS="--continuations --epoch-size-log2 $EPOCH_SIZE_LOG2"
else
  INPUT_REL="executor/tests/ethrex_bench_20.bin"
  CONT_ARGS=""
fi

command -v perf >/dev/null 2>&1 || { echo "ERROR: perf not installed (linux-tools)." >&2; exit 1; }
[ -f "$ELF_REL" ] || { echo "ERROR: missing $ELF_REL — run bench_abba.sh once (it builds the guest)." >&2; exit 1; }
if [ "$WORKLOAD" = "real" ]; then
  # ~1 MB, gitignored, never in a fresh checkout; fetch rather than abort. This is a
  # URL + sha256 download, not a build. Unconditional on purpose: the target hashes
  # whatever is on disk on every invocation, which is how a stale copy gets caught.
  # A match costs ~35 ms.
  echo "==> Verifying ethrex real-block fixture (fetches on a digest miss)"
  make ethrex-real-block-fixture
elif [ ! -f "$INPUT_REL" ]; then
  echo "ERROR: missing $INPUT_REL — run bench_abba.sh once (it builds the fixture)." >&2
  exit 1
fi
echo "==> Workload: $WORKLOAD ($INPUT_REL${CONT_ARGS:+, continuations epoch 2^$EPOCH_SIZE_LOG2})"

echo "==> Refs"
git fetch origin --quiet || echo "WARNING: 'git fetch origin' failed -- resolving against possibly-stale local refs." >&2
SHA_A="$(git rev-parse "$REF_A")"
SHA_B="$(git rev-parse "$REF_B")"
echo "   A (PR)       $REF_A  -> ${SHA_A:0:10}"
echo "   B (baseline) $REF_B  -> ${SHA_B:0:10}"

mkdir -p "$WORK"

# --- Build both binaries with debug symbols (cached per SHA) ---
need_build=0
if [ ! -x "$WORK/cli_A" ] || [ ! -x "$WORK/cli_B" ]; then
  need_build=1
elif [ "$(cat "$WORK/cli_A.sha" 2>/dev/null)" != "$SHA_A" ] || [ "$(cat "$WORK/cli_B.sha" 2>/dev/null)" != "$SHA_B" ]; then
  need_build=1
fi
if [ "$need_build" = "1" ]; then
  cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; }
  trap cleanup EXIT
  git worktree remove --force "$WT" 2>/dev/null || true
  echo "==> Building both binaries (release + debug symbols) in $WT"
  git worktree add --detach "$WT" "$SHA_B" >/dev/null
  build_cli() { # $1=sha $2=out
    echo "==> Building cli @ ${1:0:10} -> $2"
    git -C "$WT" checkout --quiet "$1"
    if ! ( cd "$WT" && CARGO_PROFILE_RELEASE_DEBUG=1 cargo build --release -p cli --features jemalloc-stats >"$WORK/build_$2.log" 2>&1 ); then
      echo "ERROR: build failed for $2. Tail of $WORK/build_$2.log:" >&2
      tail -40 "$WORK/build_$2.log" >&2
      exit 1
    fi
    cp "$WT/target/release/cli" "$WORK/$2"
    echo "$1" > "$WORK/$2.sha"
  }
  build_cli "$SHA_B" cli_B
  build_cli "$SHA_A" cli_A
  cleanup
  trap - EXIT
else
  echo "==> Reusing cached binaries (cli_A=${SHA_A:0:10} cli_B=${SHA_B:0:10})"
fi

# --- Record: warmup, then B A B A (interleaved so drift hits both sides) ---
record() { # $1=binary $2=out.data
  # shellcheck disable=SC2086  # CONT_ARGS is a deliberate multi-word flag list (empty when synthetic)
  perf record -F 599 -o "$WORK/$2" -- \
    "$WORK/$1" prove "$ELF_REL" --private-input "$INPUT_REL" $CONT_ARGS -o "$PROOF" --time \
    >"$WORK/$2.log" 2>&1
  rm -f "$PROOF"
  grep -o 'Proving time: [0-9.]*' "$WORK/$2.log" || true
}
echo "==> Warmup (B, not recorded)"
# shellcheck disable=SC2086
"$WORK/cli_B" prove "$ELF_REL" --private-input "$INPUT_REL" $CONT_ARGS -o "$PROOF" --time >/dev/null 2>&1
rm -f "$PROOF"
echo "==> Recording B (main), run 1";  record cli_B B1.data
echo "==> Recording A (PR),   run 1";  record cli_A A1.data
echo "==> Recording B (main), run 2";  record cli_B B2.data
echo "==> Recording A (PR),   run 2";  record cli_A A2.data

# --- Reports ---
echo
echo "=== perf diff, run 1  (Delta column: + = PR spends MORE self-time there) ==="
perf diff "$WORK/B1.data" "$WORK/A1.data" 2>/dev/null | head -60
echo
echo "=== perf diff, run 2  (a symbol is REAL only if it repeats here) ==="
perf diff "$WORK/B2.data" "$WORK/A2.data" 2>/dev/null | head -60
echo
echo "=== top self-time, B (main) run 1 ==="
perf report -i "$WORK/B1.data" --stdio --no-children --percent-limit 0.5 2>/dev/null | head -45
echo
echo "=== top self-time, A (PR) run 1 ==="
perf report -i "$WORK/A1.data" --stdio --no-children --percent-limit 0.5 2>/dev/null | head -45
echo
echo "Raw data in $WORK (perf report -i $WORK/A1.data for interactive drill-down)."
