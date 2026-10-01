#!/usr/bin/env bash
#
# cross_verify_vm.sh — cross-version verification of the FULL VM prover/verifier.
#
# WHY: the single-source constraints migration must preserve the constraint
# system EXACTLY — order, indices, num_base split, per-constraint degree,
# zerofier shape, and the transcript. Prove/verify within one version cannot
# see a self-consistent drift (a version that reorders constraints still
# accepts its own proofs). Cross-verifying — each side's proofs checked by the
# OTHER side's verifier — does: the verifier recomputes the OOD constraint
# evaluations from ITS OWN constraint definitions against the other side's
# commitments, so any semantic difference fails loudly. Needs no proof
# determinism (this system's proofs are nondeterministic by design: grinding +
# order-free HashMap trace tables).
#
# This is the VM-scale analog of scripts/cross_verify_examples.sh: it builds the
# `cli` binary (cargo build --release -p cli) at REF_OLD and REF_NEW in an
# isolated worktree (same build-both-refs pattern as scripts/bench_abba.sh) and
# exchanges real VM proofs over a handful of small test ELFs.
#
# WHAT IT DOES:
#   1. Builds bin/cli at REF_OLD and REF_NEW (isolated worktree).
#   2. Per ELF: prove NEW -> verify OLD, and prove OLD -> verify NEW.
#   3. Prints a per-ELF, per-direction PASS/FAIL table; exits nonzero on any
#      failure. A failing direction is a REAL migration finding (ordering /
#      num_base / alpha-power indexing / zerofier grouping / transcript) —
#      diagnose and fix the NEW side, never the old one.
#
# USAGE:
#   scripts/cross_verify_vm.sh REF_OLD REF_NEW
#     REF_OLD  ref or SHA with the pre-migration (boxed) constraint system
#     REF_NEW  ref or SHA with the migrated (single-source) constraint system
#   Env: WORK  work/output dir   (default /tmp/cross_verify_vm)
#        WT    build worktree    (default /tmp/cross_verify_vm_wt)
#        ELFS  space-separated absolute ELF paths (default: a few small asm ELFs
#              from executor/program_artifacts/asm, built via
#              `make compile-programs-asm` if absent)

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: cross_verify_vm.sh REF_OLD REF_NEW" >&2
  exit 2
fi
REF_OLD="$1"
REF_NEW="$2"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# --- ELF fixtures: small asm programs the prove_elfs tests exercise. -----------
# The CLI consumes prebuilt ELF files; the asm artifacts are produced by
# `make compile-programs-asm` (a plain clang invocation, no sysroot needed).
ASM_DIR="$ROOT/executor/program_artifacts/asm"
DEFAULT_ELF_NAMES=(sub add arith_8)
if [ -z "${ELFS:-}" ]; then
  # Build the asm artifacts if the ones we need are missing.
  missing=0
  for n in "${DEFAULT_ELF_NAMES[@]}"; do
    [ -f "$ASM_DIR/$n.elf" ] || missing=1
  done
  if [ "$missing" = "1" ]; then
    echo "==> Building asm ELF artifacts (make compile-programs-asm)"
    make compile-programs-asm >/dev/null
  fi
  ELFS=""
  for n in "${DEFAULT_ELF_NAMES[@]}"; do
    ELFS="$ELFS $ASM_DIR/$n.elf"
  done
fi
# shellcheck disable=SC2206
ELF_LIST=($ELFS)

WORK="${WORK:-/tmp/cross_verify_vm}"
WT="${WT:-/tmp/cross_verify_vm_wt}"

SHA_OLD="$(git rev-parse "$REF_OLD")"
SHA_NEW="$(git rev-parse "$REF_NEW")"
echo "==> Refs"
echo "   OLD $REF_OLD -> ${SHA_OLD:0:10}"
echo "   NEW $REF_NEW -> ${SHA_NEW:0:10}"
echo "==> ELFs: ${ELF_LIST[*]}"

mkdir -p "$WORK"

# --- 1. Build both cli binaries in an isolated worktree ------------------------
cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; }
trap cleanup EXIT
git worktree remove --force "$WT" 2>/dev/null || true
git worktree add --detach "$WT" "$SHA_OLD" >/dev/null
build_cli() { # $1=sha $2=out (shared target dir -> 2nd build is incremental)
  echo "==> Building cli @ ${1:0:10} -> $2"
  git -C "$WT" checkout --quiet -f "$1"
  if ! (cd "$WT" && cargo build --release -p cli >"$WORK/build_$2.log" 2>&1); then
    echo "ERROR: cargo build failed for $2 (@ ${1:0:10}). Tail of $WORK/build_$2.log:" >&2
    tail -40 "$WORK/build_$2.log" >&2
    exit 1
  fi
  cp "$WT/target/release/cli" "$WORK/$2"
}
build_cli "$SHA_OLD" cli_old
build_cli "$SHA_NEW" cli_new
cleanup
trap - EXIT

# --- 2. Cross-verify every ELF in both directions -----------------------------
fail=0
check() { # $1=prover bin  $2=verifier bin  $3=elf path  $4=direction label
  local elf="$3"
  local tag
  tag="$(basename "$elf" .elf)"
  local proof="$WORK/$tag.$4.bin"
  if ! "$WORK/$1" prove "$elf" -o "$proof" >"$WORK/$tag.$4.prove.log" 2>&1; then
    echo "FAIL $4 : $tag  (PROVE errored; see $WORK/$tag.$4.prove.log)"
    fail=1
    return
  fi
  if "$WORK/$2" verify "$proof" "$elf" >"$WORK/$tag.$4.verify.log" 2>&1; then
    echo "PASS $4 : $tag"
  else
    echo "FAIL $4 : $tag  (VERIFY rejected; see $WORK/$tag.$4.verify.log)"
    fail=1
  fi
}

echo "==> Cross-verifying ${#ELF_LIST[@]} ELFs, both directions"
for elf in "${ELF_LIST[@]}"; do
  check cli_new cli_old "$elf" "prove-NEW-verify-OLD"
  check cli_old cli_new "$elf" "prove-OLD-verify-NEW"
done

echo
if [ "$fail" = "0" ]; then
  echo "==> RESULT: all ${#ELF_LIST[@]} ELFs cross-verify in both directions."
else
  echo "==> RESULT: FAILURES above — the migration drifted from the old constraint system."
fi
exit "$fail"
