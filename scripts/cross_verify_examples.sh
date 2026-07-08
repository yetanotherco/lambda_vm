#!/usr/bin/env bash
#
# cross_verify_examples.sh — cross-version verification of the example AIRs.
#
# WHY: the single-source constraints migration must preserve the constraint
# system EXACTLY — order, indices, num_base split, degrees, zerofier shape.
# Prove/verify within one version cannot see a self-consistent drift (a
# version that reorders constraints still accepts its own proofs). Verifying
# each side's proofs with the OTHER side's verifier does: the verifier
# recomputes the OOD constraint evaluations from ITS OWN constraint
# definitions against the other side's commitments, so any semantic
# difference fails loudly. Needs no proof determinism.
#
# WHAT IT DOES:
#   1. Builds the stark `examples_cli` example binary at REF_OLD and REF_NEW
#      (isolated worktree, same pattern as scripts/bench_abba.sh).
#   2. Per example AIR: prove NEW -> verify OLD, and prove OLD -> verify NEW.
#   3. Prints a per-example, per-direction PASS/FAIL table; exits nonzero if
#      any direction fails. A failing direction is a REAL migration finding —
#      diagnose and fix the migration, never the old side.
#
# USAGE:
#   scripts/cross_verify_examples.sh REF_OLD REF_NEW
#     REF_OLD  ref or SHA with the pre-migration constraint system
#     REF_NEW  ref or SHA with the migrated constraint system
#   Env: WORK  work/output dir   (default /tmp/cross_verify_examples)
#        WT    build worktree    (default /tmp/cross_verify_wt)

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: cross_verify_examples.sh REF_OLD REF_NEW" >&2
  exit 2
fi
REF_OLD="$1"
REF_NEW="$2"

EXAMPLES=(
  simple_fibonacci
  fibonacci_2_columns
  fibonacci_2_cols_shifted
  fibonacci_multi_column
  quadratic_air
  fibonacci_rap
  dummy_air
  simple_addition
  read_only_memory
  read_only_memory_logup
  multi_table_lookup
)

WORK="${WORK:-/tmp/cross_verify_examples}"
WT="${WT:-/tmp/cross_verify_wt}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SHA_OLD="$(git rev-parse "$REF_OLD")"
SHA_NEW="$(git rev-parse "$REF_NEW")"
echo "==> Refs"
echo "   OLD $REF_OLD -> ${SHA_OLD:0:10}"
echo "   NEW $REF_NEW -> ${SHA_NEW:0:10}"

mkdir -p "$WORK"

# --- 1. Build both examples_cli binaries in an isolated worktree ---
cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; }
trap cleanup EXIT
git worktree remove --force "$WT" 2>/dev/null || true
git worktree add --detach "$WT" "$SHA_OLD" >/dev/null
build_cli() { # $1=sha $2=out (shared target dir -> 2nd build is incremental)
  echo "==> Building examples_cli @ ${1:0:10} -> $2"
  git -C "$WT" checkout --quiet -f "$1"
  if ! (cd "$WT" && cargo build --release -p stark --features test-utils \
    --example examples_cli >"$WORK/build_$2.log" 2>&1); then
    echo "ERROR: cargo build failed for $2 (@ ${1:0:10}). Tail of $WORK/build_$2.log:" >&2
    tail -40 "$WORK/build_$2.log" >&2
    exit 1
  fi
  cp "$WT/target/release/examples/examples_cli" "$WORK/$2"
}
build_cli "$SHA_OLD" cli_old
build_cli "$SHA_NEW" cli_new
cleanup
trap - EXIT

# --- 2. Cross-verify every example in both directions ---
fail=0
check() { # $1=prover bin  $2=verifier bin  $3=example  $4=direction label
  local proof="$WORK/$3.$4.bin"
  if ! "$WORK/$1" prove "$3" -o "$proof" >"$WORK/$3.$4.prove.log" 2>&1; then
    echo "FAIL $4 : $3  (PROVE errored; see $WORK/$3.$4.prove.log)"
    fail=1
    return
  fi
  if "$WORK/$2" verify "$3" "$proof" >"$WORK/$3.$4.verify.log" 2>&1; then
    echo "PASS $4 : $3"
  else
    echo "FAIL $4 : $3  (VERIFY rejected; see $WORK/$3.$4.verify.log)"
    fail=1
  fi
}

echo "==> Cross-verifying ${#EXAMPLES[@]} examples, both directions"
for ex in "${EXAMPLES[@]}"; do
  check cli_new cli_old "$ex" "prove-NEW-verify-OLD"
  check cli_old cli_new "$ex" "prove-OLD-verify-NEW"
done

echo
if [ "$fail" = "0" ]; then
  echo "==> RESULT: all ${#EXAMPLES[@]} examples cross-verify in both directions."
else
  echo "==> RESULT: FAILURES above — the migration drifted from the old constraint system."
fi
exit "$fail"
