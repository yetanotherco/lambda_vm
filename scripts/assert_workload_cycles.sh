#!/usr/bin/env bash
# Refuse to benchmark a workload the guest rejected.
#
# `run_stateless_guest` cannot fail. On an input whose schema it does not recognise it
# commits `successful_validation = 0` and exits cleanly, so a fixture from before the
# ethrex 26 bump runs 496 cycles instead of aborting -- measured, against 3,730,771 for
# the same guest on its own 10-transfer fixture. Proving that reads as a ~99% improvement,
# in green, on both sides of an A/B.
#
# Every benchmark entry point reuses whatever fixture is already on disk, so on a
# persistent runner this is the common case and not a corner case. The real block is
# pinned twice -- by sha256 in the Makefile and by a filename carrying the block number --
# but the synthetic names carry neither the rev nor the schema:
# `executor/tests/ethrex_bench_20.bin` is gitignored, so it survives a rev bump, a
# `git clean -fd` and every `git status`, and `ethrex_<n>_transfers.bin` is merely
# untracked. The digest pin cannot see the other half of the pair either: moving only the
# guest (`scripts/set_ethrex_rev.sh --guest-only`, which is how a benchmark varies the
# guest across revs) leaves a fixture that still matches its pin and an ELF that may no
# longer accept it.
#
# One cheap execution turns that whole class of mistake -- stale fixture, wrong fork,
# fixture built against another ethrex rev -- into a hard stop.
#
# Usage: scripts/assert_workload_cycles.sh <cli> <elf> <input> real|synthetic
#
# A standalone script rather than a sourceable function, because
# .github/workflows/benchmark-pr.yml calls it straight from a `run:` block.
set -euo pipefail

# Far above a rejected run (496 cycles) and far below the smallest honest one.
#
# The real block runs 30.5M cycles today, so 20M is ~34% of margin: enough for a guest
# change that genuinely gets cheaper, and still 40,000x above a rejection. A floor of 1M
# would have tolerated a 97% collapse, which is the only kind of breakage this can see --
# WHICH block is being proven is pinned by the fixture's sha256 and by the ethrex rev, not
# by a cycle count, so this is not the place to re-pin the workload's identity.
#
# The synthetic floor stays two orders of magnitude lower because a small TX_COUNT is a
# legitimate workload: a 1-transaction block ran 1.80M cycles before the bump and an empty
# one 0.99M, and TX_COUNT is a knob, so nothing here knows which size to expect.
MIN_CYCLES_REAL=20000000
MIN_CYCLES_SYNTHETIC=100000

# A caller that already has the count (scripts/bench_recursion_scaling.sh records one per
# block size) asks for the floor instead of paying a second execution, so the constants
# above stay the only copy.
if [ "${1:-}" = "--floor" ]; then
  case "${2:-}" in
    real)      echo "$MIN_CYCLES_REAL"; exit 0 ;;
    synthetic) echo "$MIN_CYCLES_SYNTHETIC"; exit 0 ;;
    *) echo "usage: ${0##*/} --floor real|synthetic" >&2; exit 2 ;;
  esac
fi

if [ "$#" -ne 4 ]; then
  echo "usage: ${0##*/} <cli> <elf> <input> real|synthetic" >&2
  echo "       ${0##*/} --floor real|synthetic" >&2
  exit 2
fi
CLI="$1"; ELF="$2"; INPUT="$3"; WORKLOAD="$4"
case "$WORKLOAD" in
  real)      FLOOR=$MIN_CYCLES_REAL ;;
  synthetic) FLOOR=$MIN_CYCLES_SYNTHETIC ;;
  *) echo "ERROR: workload must be 'real' or 'synthetic' (got '$WORKLOAD')." >&2; exit 2 ;;
esac

# Kept whole instead of piped straight into awk, so a run that never reaches a cycle
# count can be told apart from one that ran and was rejected -- and so its own diagnostic
# survives to be printed. `|| true`: a failing command substitution would otherwise abort
# under `set -e` before either message below, which is fail-closed but silent.
out="$("$CLI" execute "$ELF" --private-input "$INPUT" --cycles 2>&1)" || true
cycles="$(printf '%s\n' "$out" | awk '/^Cycles:/ {print $2}')"
# Every non-numeric answer -- absent, a usage error, several lines -- is "no count".
case "${cycles:-}" in ''|*[!0-9]*) cycles="" ;; esac

if [ -z "$cycles" ]; then
  echo "ERROR: 'cli execute' printed no cycle count, so the run never got as far as being" >&2
  echo "       accepted or rejected. That is a different fault -- a missing file, an" >&2
  echo "       execution error, a cli that does not implement a syscall this ELF uses." >&2
  echo "         cli   $CLI" >&2
  echo "         ELF   $ELF" >&2
  echo "         input $INPUT" >&2
  echo "       Tail of its output:" >&2
  printf '%s\n' "$out" | tail -5 | sed 's/^/         /' >&2
  exit 1
fi
if [ "$cycles" -lt "$FLOOR" ]; then
  echo "ERROR: the workload executed only $cycles cycles, below the $FLOOR floor for the" >&2
  echo "       $WORKLOAD workload. The guest almost certainly rejected the input:" >&2
  echo "         ELF   $ELF" >&2
  echo "         input $INPUT" >&2
  echo "       Rebuild the fixture at this ethrex rev: 'make regen-real-block-fixture' for" >&2
  echo "       the real block, or delete the synthetic one and let the caller regenerate it" >&2
  echo "       (tooling/ethrex-fixtures)." >&2
  exit 1
fi
echo "==> Workload executes: $cycles cycles ($WORKLOAD floor $FLOOR)"
