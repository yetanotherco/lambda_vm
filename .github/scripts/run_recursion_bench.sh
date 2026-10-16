#!/usr/bin/env bash
#
# Runs scripts/bench_recursion_cycles.sh across the regimes /bench-verify reports and
# appends each result — or an explicit failure note — to /tmp/recursion_result.txt for
# the bench-verify.yml PR comment.
#
# Two regimes, deliberately:
#   min            cheap canary over the `empty` diagnostic program (blowup=2, 1 query).
#                  Seconds per ref, so it catches a broken guest before the expensive
#                  regime runs, and it's the one arm whose absolute cycle count is
#                  meaningless on its own.
#   blowup2-block  the representative regime: a REAL ethrex 20-tx block proved via
#                  CONTINUATIONS and verified in-VM at a real query count (blowup=2,
#                  219 queries — the same options the verifier arms above use). Real
#                  prover minutes per ref; the dumped blob is cached by ref SHA so a
#                  repeat run skips re-proving.
#
# The `empty`-program full-query regimes (blowup2/blowup4) used to run here too. They
# only ever varied the query count over a trivial inner trace, which blowup2-block now
# covers at a realistic trace size, so they were dropped to pay for the 20-tx block
# instead. They still work for manual runs:
#   scripts/bench_recursion_cycles.sh <sha> origin/main blowup2
#
# Usage: .github/scripts/run_recursion_bench.sh HEAD_SHA
set -euo pipefail

HEAD_SHA="$1"
RESULT=/tmp/recursion_result.txt
: > "$RESULT"

run_preset() {
  local preset="$1"
  local log="/tmp/recursion_out_${preset}.txt"
  if scripts/bench_recursion_cycles.sh "$HEAD_SHA" origin/main "$preset" 2>&1 | tee "$log"; then
    { echo; sed -n '/<!-- recursion-cycle-report -->/,$p' "$log"; } >> "$RESULT"
  else
    # Say it FAILED, not that it's "unavailable for these refs": the old wording read as
    # a ref-capability limit and hid a real infra bug (a shared guest target dir that
    # poisoned every regime after the first) for as long as it was there.
    { echo; echo "_(${preset} regime FAILED — see the workflow log.)_"; } >> "$RESULT"
  fi
}

run_preset min
# Post-result's raw-log fallback reads /tmp/recursion_out.txt (unsuffixed).
cp -f /tmp/recursion_out_min.txt /tmp/recursion_out.txt

# blowup2-block: blowup=2 verifier over a REAL ethrex block proved with continuations
# (via the `continuation` guest). Needs origin/main's RECURSION_DUMP_EPOCH_LOG2 support.
# BLOCK_TXS/BLOCK_EPOCH_LOG2 keep the script's own defaults; blowup=2 matches the query
# count the verifier arms use, and at 20 txs / 2^21 the bundle is ~350 MB, inside the
# guest's 512 MiB MAX_PRIVATE_INPUT_SIZE (see that script's header for the measurements).
if git grep -q RECURSION_DUMP_EPOCH_LOG2 origin/main -- prover/src/tests/ 2>/dev/null; then
  run_preset blowup2-block
else
  { echo; echo "_(blowup2-block real-ethrex-block regime needs \`origin/main\` to support RECURSION_DUMP_EPOCH_LOG2 — not merged yet.)_"; } >> "$RESULT"
fi
