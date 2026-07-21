#!/usr/bin/env bash
#
# Runs scripts/bench_recursion_cycles.sh across every preset regime (min,
# blowup2/blowup4, blowup4-block) and appends each result — or an
# "unavailable" note when the ref/preset combo isn't supported — to
# /tmp/recursion_result.txt for the bench-verify.yml PR comment.
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
    { echo; sed -n '/=== Recursion-guest cycle/,$p' "$log"; } >> "$RESULT"
  else
    { echo; echo "_(${preset} regime unavailable for these refs — see the workflow log.)_"; } >> "$RESULT"
  fi
}

run_preset min
# Post-result's raw-log fallback reads /tmp/recursion_out.txt (unsuffixed).
cp -f /tmp/recursion_out_min.txt /tmp/recursion_out.txt

# blowup2/blowup4: full-query base-layer regimes over the `empty` diagnostic
# program. Need origin/main's RECURSION_DUMP_PRESET support to dump a
# non-min blob; checked once up front instead of failing each preset in turn.
if git grep -q RECURSION_DUMP_PRESET origin/main -- prover/src/tests/ 2>/dev/null; then
  run_preset blowup2
  run_preset blowup4
else
  { echo; echo "_(blowup2/blowup4 full-query regimes need \`origin/main\` to have the preset-aware dump test (RECURSION_DUMP_PRESET) — not merged yet, so only \`min\` is compared for this PR.)_"; } >> "$RESULT"
fi

# blowup4-block: same blowup=4 verifier over a REAL ethrex block (via the
# `continuation` guest). Needs origin/main's RECURSION_DUMP_EPOCH_LOG2 support.
if git grep -q RECURSION_DUMP_EPOCH_LOG2 origin/main -- prover/src/tests/ 2>/dev/null; then
  run_preset blowup4-block
else
  { echo; echo "_(blowup4-block real-ethrex-block regime needs \`origin/main\` to support RECURSION_DUMP_EPOCH_LOG2 — not merged yet.)_"; } >> "$RESULT"
fi
