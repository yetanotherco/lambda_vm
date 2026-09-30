#!/usr/bin/env bash
# O5 v10 — v9 (O3, md5 bea1219641a56ced9a9c44eab9064cb6) plus the fallback gate
# corrected and a host-peak ceiling. Immutable once launched; a fix goes to v11.
#
# What v10 changes vs v9 (nothing else — a byte-for-byte diff was checked):
#   1. THE FALLBACK GATE, the reason v9 voided every tree run. v9 read a
#      `host fallbacks` line the TREE harness never prints and defaulted the
#      unread value to 1 => exit 8 always. v10 reads the TWO lines the harness
#      now prints at its WHOLE-RUN block (item 2, per_table_aggregator_tests.rs):
#      `commit fallbacks` (the commit path) and `device fallbacks` (the
#      argue surface, math-cuda, bc6ce9ea2). Both must be 0; an ABSENT line is a
#      LOUD refuse (11 / 13), never a silent pass. device fallbacks is THE
#      primary gate — it is the surface wt16's net-negative hid on.
#   2. HOST_PEAK_HI_GIB, a REQUIRED one-sided ceiling on the `WHOLE RUN: host
#      peak` line (backstop for any host rise the fallback gate misses; wt16's
#      37.3 vs the no-retention 31.0 would have failed). Carried IN the command.
#   3. LFM_WHIR_RETENTION exported per arm (ruled b): A and B run off ONE binary,
#      differing only by this flag; EXPECT_RETENTION tracks it, and the
#      retention-line gate is now SEMANTIC (admitted 0 vs >0), not structural
#      (line present vs absent), so the disabled control's present-but-not-taken
#      line reads correctly.
#   Exit codes added: 11 commit-fallback line absent · 12 device fallbacks > 0 ·
#   13 device-fallback line absent · 14 host peak over the ceiling (or no line).
#
# O3 v9 — THE WHIR BLOCK ARTIFACT with THE LEAF-LAYER RETENTION read back.
#
# v8 (md5 7a30ea943ce56d1ce964bbd5f1ff17ec) byte-unchanged; this is v8 plus four
# read-backs and one REQUIRED input.
#
# ⛔ `EXPECT_RETENTION=yes|no` IS REQUIRED AND HAS NO DEFAULT. The ABBA runs this
# script on BOTH a tree that has the retention and one that does not, so "the
# line is missing" is a legitimate state in one arm and a defect in the other —
# and a script that cannot tell them apart would pass either way. The arm
# declares what it expects and this refuses on disagreement, which is the only
# form in which the absent line can still be an error.
#
# What v9 adds:
#   1. the `retention[leaf layers]` line, REQUIRED per EXPECT_RETENTION;
#   2. leaf_passes vs tree_builds, and the admitted/refused counts, lifted;
#   3. the device peak split into the BASE's window and the rest — the base is
#      where the retention's bytes are held, so a peak set by level 0 says
#      nothing about it;
#   4. `host fallbacks` from the pin line, a HARD gate: a fallback is H4's cliff
#      and means the lever is dead at that shape, not merely slower.
#
# A NEW file, DERIVED from A-tree-whir.v4.sh (wt9/wt10/wt11's script, immutable)
# by anchored edits; `diff A-tree-whir.v4.sh A-tree-whir.v5.sh` is the whole
# change. Every guard, every read-0 marker and every export of v4 is unchanged —
# the additions are section 8b's parse and its one refusal.
#
# ★ WHAT IS NEW, and it is a reading rather than a lever: at this sha
# `LAMBDA_VM_BASE_SPLIT=1` finally reaches the WHIR base, so the 84.5 s that
# wt9 printed as ONE number resolves into the producer's four stages, the
# prover's four, the argument's four slots and the global stage. The wall, the
# proof bytes, the pins and the census must all be UNMOVED against wt9/wt11 —
# this run adds timers, not changes.
# It is NOT an edit of A-tree-boxA.v*.sh either: that drives the STARK pipeline
# and is what the D-S control runs.
#
# Usage:
#   A-tree-whir.v4.sh <worktree> <elf> <input> [epoch_log2]
# with EXPECT_HEAD=<sha> AND ROOT_OPTION=A|B in the environment — both REQUIRED,
# no defaults (the harness has no default for the option either, :7467).
#
# ★ WHAT THIS MEASURES: the base epochs proven under WHIR with the RPX
# transcript, one LFM wrap per epoch from `whir_epoch_program`, THE WHIR GLOBAL
# STAGE (the cross-epoch program as ONE child, `prove_whir_global_child`), the
# interior to `RootOption::child_level(top)`, and THE BLOCK-ARTIFACT ROOT over
# `fan_in + 1` children — one run, nothing cached, nothing sliced.
#
# ⛔ THE HARNESS'S OWN CONTRACT (per_table_aggregator_tests.rs at db5ad5e5b):
#   LFM_TREE_PROVE_ROOT=1 REQUIRES LFM_TREE_LEVELS UNSET (:7450 — "a second and
#   contradictory spelling"); LFM_TREE_ROOT_OPTION is mandatory with it (:7467);
#   A_CACHE_DIR is REFUSED (:7425 — this driver caches nothing); A_BUNDLE_MODE
#   refused (:7364); GLOBAL_K / PARENT_MODE / SIZE_GLOBAL name stages that do
#   not exist on the WHIR path; the global stage that cannot be built PANICS
#   (:7677) — a run that ends `0 passed` after the base is that panic, not a
#   measurement. So v3's `[levels] [cache_dir]` positionals are GONE, not
#   defaulted: a caller passing them gets a usage refusal below.

set -uo pipefail

REPO="${1:?usage: A-tree-whir.v4.sh <worktree> <elf> <input> [epoch_log2]}"
ELF="${2:?path to the block ELF}"
INPUT="${3:?path to the block input .bin}"
EPOCH_LOG2="${4:-21}"
if [ $# -gt 4 ]; then
  echo "REFUSING TO MEASURE: $# arguments — v4 takes at most 4 (<worktree> <elf> \
<input> [epoch_log2]); the levels and the cache directory are not inputs of the \
root arm (the harness refuses both)" >&2
  exit 1
fi
ROOT_OPTION="${ROOT_OPTION:?set ROOT_OPTION=A|B — the root option is a NAMED INPUT with no default}"
case "$ROOT_OPTION" in A|B) ;; *)
  echo "REFUSING TO MEASURE: ROOT_OPTION must be A or B, got '$ROOT_OPTION'" >&2; exit 1;;
esac
EXPECT_RETENTION="${EXPECT_RETENTION:?set EXPECT_RETENTION=yes|no — the ABBA runs this on a tree with the retention and one without, and an absent line must be an ERROR in exactly one of them}"
case "$EXPECT_RETENTION" in yes|no) ;; *)
  echo "REFUSING TO MEASURE: EXPECT_RETENTION must be yes or no, got '$EXPECT_RETENTION'" >&2; exit 1;;
esac
HOST_PEAK_HI_GIB="${HOST_PEAK_HI_GIB:?set HOST_PEAK_HI_GIB=NN.N — the one-sided WHOLE-RUN host-peak ceiling in GiB. A rise above it fails the run (wt16 37.3 vs the no-retention baseline 31.0 would have failed). The PRIMARY gate is device fallbacks 0; this is the backstop that catches any host rise it does not.}"
case "$HOST_PEAK_HI_GIB" in
  ''|*[!0-9.]*) echo "REFUSING TO MEASURE: HOST_PEAK_HI_GIB must be a number in GiB, got '$HOST_PEAK_HI_GIB'" >&2; exit 1;;
esac
SAMPLE_MS="${SAMPLE_MS:-10}"
K="${K:-4}"
VRAM_BUDGET_MB="${VRAM_BUDGET_MB:-24000}"   # a number in MiB, or `query` = export nothing and let the device layer ask the driver (lb8: the pinned 24000 costs the WHIR base +4.47 s and 34.5 GB of host RSS against 11 GB)
VRAM_SCHED_MB="${VRAM_SCHED_MB:-}"
TEST=lfm::per_table_aggregator_tests::the_whir_production_tree_composes_to_a_root

export PATH="/root/.cargo/bin:$PATH"
cd "$REPO" || exit 2

# ---- 0. ONE OWNER ----------------------------------------------------------
exec 9>/root/.box.lock
if ! flock -n 9; then
  echo "REFUSING TO MEASURE: /root/.box.lock is held — another run owns this box" >&2
  exit 1
fi

# ---- 1. provenance ---------------------------------------------------------
: "${EXPECT_HEAD:?set EXPECT_HEAD=<sha> — a run whose commit is unknown cannot be scored}"
HEAD_SHA="$(git rev-parse --short=8 HEAD 2>/dev/null)"
if [ "${HEAD_SHA:-}" != "${EXPECT_HEAD:0:8}" ]; then
  echo "REFUSING TO MEASURE: HEAD is ${HEAD_SHA:-<none>}, expected ${EXPECT_HEAD:0:8}" >&2
  exit 1
fi

# The stronger check: does this tip CONTAIN the test? A launch off a tip that
# does not prints `0 passed; N filtered out`, exits 0, every precondition green.
# `grep -c` (reads to EOF) not `grep -q` — the SIGPIPE trap v3 documents.
LIST_OUT="$(mktemp /tmp/W_list.XXXX)"
LIST_ERR=/tmp/W_list_err.log
if ! cargo test --release -p lambda-vm-prover --features cuda --lib -- --list \
     >"$LIST_OUT" 2>"$LIST_ERR"; then
  echo "REFUSING TO MEASURE: could NOT list the tests (a build or toolchain \
failure, NOT a missing test) — see $LIST_ERR" >&2
  tail -20 "$LIST_ERR" >&2
  exit 1
fi
if [ "$(grep -c "^${TEST}: test$" "$LIST_OUT")" -ne 1 ]; then
  echo "REFUSING TO MEASURE: the build listed $(grep -c ': test$' "$LIST_OUT") \
tests and $TEST is NOT among them" >&2
  exit 1
fi
echo "test present: $TEST ($(grep -c ': test$' "$LIST_OUT") tests listed)"
rm -f "$LIST_OUT"

# ---- 1b. THE BASE THIS RUN USES, and why Fix A's check is NOT copied --------
# ⛔ A-tree-boxA.v3.sh asserts `continuation.rs` has exactly 3 `scope.spawn`
# sites as a content proxy for Fix A (111ca721). That check is STALE on this
# lineage and would refuse a tree that HAS the fix: 813b2f3b ("an epoch is
# prepared while the last one is being proved") added a FOURTH spawn for the
# epoch pipeline, so the count reads 4 again although 111ca721 is an ancestor.
# ⇒ It is not copied here, and the STARK script's own copy wants revisiting.
#
# What this run needs instead is a check about the file it ACTUALLY proves from.
# The WHIR base is `multilinear_continuation::prove_continuation`, which spawns
# nothing at all — so the two-concurrent-VramGate overlap Fix A closed cannot
# arise on this path by construction. Asserted, not assumed, because a future
# parallel WHIR base would reopen it silently.
#
# ⛔⛔ v1's VERSION OF THIS GUARD COULD NEVER PASS, and the bug is worth stating
# because the shape is everywhere. It read:
#
#     MSPAWNS="$(grep -c 'scope\.spawn' <file> 2>/dev/null || echo -1)"
#
# `grep -c` ALWAYS prints a count and signals "no match" through its EXIT
# STATUS, so on the 0-site file it printed "0" AND exited 1 — the `|| echo -1`
# then appended, MSPAWNS became the two-line string "0\n-1", and the comparison
# refused. On a file with sites the guard would have refused correctly; on the
# one state it exists to ACCEPT it could not pass. It cost a box launch.
#
# ⇒ THE FILE'S EXISTENCE AND ITS COUNT ARE NOW TWO READS. The `|| true` keeps
# grep's OUTPUT and discards its status, which is what line 120 already does
# with `nvidia-smi | grep -c . || true`.
# ⚠ The lesson is NOT "never use `|| echo`": the `cat`/`git rev-parse`/`date`
# sites below are the same shape and are SAFE, because those commands print
# NOTHING on their failing path. It is unsafe only after a command that prints
# on failure, and `grep -c` is exactly that command.
MCFILE=prover/src/multilinear_continuation.rs
if [ ! -r "$MCFILE" ]; then
  echo "REFUSING TO MEASURE: $MCFILE is missing or unreadable from $REPO — this \
is not the tree this script was written for." >&2
  exit 1
fi
MSPAWNS="$(grep -c 'scope\.spawn' "$MCFILE" || true)"
if [ "$MSPAWNS" != "0" ]; then
  echo "REFUSING TO MEASURE: $MCFILE has $MSPAWNS \
\`scope.spawn\` sites, expected 0. The WHIR base has become concurrent, so the \
two-VramGate overlap that Fix A closed on the STARK base may now be open here. \
Read gpu-two-vramgates-overlap before running, or set W_ALLOW_SPAWNS=1 if this \
check has gone stale the way the STARK one did." >&2
  [ "${W_ALLOW_SPAWNS:-0}" = "1" ] || exit 1
  echo "  (W_ALLOW_SPAWNS=1 — proceeding)" >&2
fi
# ★ PRINTED ON THE ACCEPTING PATH TOO. A guard that is silent when it passes
# leaves a reader unable to tell "checked and fine" from "never ran" — which is
# the same confusion v1's failure produced from the other side.
echo "spawn guard: $MCFILE has $MSPAWNS \`scope.spawn\` sites (expected 0)"

# ---- 2. modes this test does not consult -----------------------------------
if [ -n "${A_BUNDLE_MODE:-}" ]; then
  echo "REFUSING TO MEASURE: A_BUNDLE_MODE is exported and this driver does NOT \
consult it — LFM_TREE_LEVELS names the experiment. Unset it." >&2
  exit 1
fi
# ⛔ THE ROOT ARM IS THE ONLY ARM. Everything the harness refuses beside the
# root knobs is refused HERE first, in BOTH spellings — the bare run_ds.sh
# habit (GLOBAL_K=2) and the process form (LFM_TREE_GLOBAL_K=2) — so a stale
# export in the launcher's shell is a refusal on the record, not a knob that
# reached the process or silently did not. LFM_TREE_LEVELS and A_CACHE_DIR are
# the two v3 EXPORTED that the root arm must NOT: the harness refuses both.
for v in SIZE_ROOT STOP_AFTER_GLOBAL GLOBAL_K GLOBAL_MODE PARENT_MODE SIZE_GLOBAL \
         ROOT_MODE TOP_OVERLAP LEVELS PROVE_ROOT; do
  for name in "$v" "LFM_TREE_$v"; do
    if [ -n "$(eval "printf '%s' \"\${$name:-}\"")" ]; then
      echo "REFUSING TO MEASURE: $name is set in the launcher's environment. The \
root arm exports LFM_TREE_PROVE_ROOT and LFM_TREE_ROOT_OPTION itself and the \
harness refuses or ignores the rest — unset it." >&2
      exit 1
    fi
  done
done
if [ -n "${A_CACHE_DIR:-}" ]; then
  echo "REFUSING TO MEASURE: A_CACHE_DIR is set and the WHIR driver caches \
NOTHING (a shared \`wrap-0.rkyv\` would collide with a STARK tree). Unset it." >&2
  exit 1
fi

# ---- 2b. THE FIXTURE, BY CONTENT ------------------------------------------
#
# ⛔⛔ THE FILE NAME IS NOT THE FILE. A run handed
# `ethrex_mainnet_25368371.bin` under an OLDER ENCODING died 1.7 s into the base
# with `Panic … called `Result::unwrap()` on an `Err` value: Error { inner:
# Failure }` — a guest panic that says nothing about fixtures, after a full guest
# ELF build, on a box, in a slot somebody was waiting for. The two files differ
# by 27 bytes and share a name; only the sha separates them.
#
# ⇒ BOTH FIXTURES ARE PINNED BY CONTENT, and this runs BEFORE section 4's guest
# build rather than beside its readability probe: a wrong fixture should cost
# seconds, not a build. The readability probe stays where it is — it also feeds
# the artifact `missing` count — so this is an addition, not a move.
#
# ⚠ The input is an UNTRACKED repo copy (`executor/tests/…`), absent from a
# fresh worktree; `run_ds.sh` copies it from the whir worktree. So "the file is
# there" is exactly the assumption that fails, and it fails LATE.
WANT_ELF_SHA=8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a
WANT_INPUT_SHA=573004e62e3680a00d3cdbae19dc4897e2ec60d6ec0c1d05d9ef118cb8aef17f

# The repo's own idiom (`Makefile` prepare-sysroot): prefer sha256sum, fall back
# to shasum, and REFUSE when neither exists. A checksum that silently does not
# run is worse than none, because the log then looks like a verified run.
if command -v sha256sum >/dev/null 2>&1; then SHACMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then SHACMD="shasum -a 256"
else
  echo "REFUSING TO MEASURE: neither sha256sum nor shasum is available, so the \
fixtures cannot be verified by content. A run that skips this check is the run \
that measured the wrong file." >&2
  exit 1
fi

check_fixture() {
  what="$1"; path="$2"; want="$3"
  if [ ! -r "$path" ]; then
    echo "REFUSING TO MEASURE: the $what is missing or unreadable: $path" >&2
    exit 1
  fi
  bytes="$(wc -c < "$path" | tr -d ' ')"
  got="$($SHACMD "$path" | awk '{print $1}')"
  # ★ PRINTED WHETHER IT PASSES OR NOT, at full width. The diagnosis of the bad
  # run was a byte count in a header; the sha is what settles it next time.
  echo "$what: $bytes bytes  sha256 $got"
  if [ "$got" != "$want" ]; then
    echo "REFUSING TO MEASURE: the $what is not the file the record was proved \
from — same name, different contents." >&2
    echo "  wanted $want" >&2
    echo "  read   $got" >&2
    echo "  path   $path ($bytes bytes)" >&2
    exit 1
  fi
}
check_fixture "block ELF"   "$ELF"   "$WANT_ELF_SHA"
check_fixture "block input" "$INPUT" "$WANT_INPUT_SHA"
echo "fixtures verified by content against the record's pair"

# ---- 3. the card must be IDLE ----------------------------------------------
GPU_USED="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)"
GPU_APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)"
if [ "${GPU_USED:-99999}" -ge 500 ] || [ "${GPU_APPS:-1}" -ne 0 ]; then
  echo "REFUSING TO MEASURE: card not idle — ${GPU_USED} MiB used, ${GPU_APPS} compute app(s)" >&2
  nvidia-smi >&2
  exit 1
fi

# ---- 4. guest ELFs, enumerated from the build system ------------------------
echo "building guest ELFs (asm + rust + recursion)…"
if ! SYSROOT_DIR="${SYSROOT_DIR:-$HOME/.lambda-vm-sysroot}" \
     make compile-programs-asm compile-programs-rust compile-recursion-elfs \
        >/tmp/W_guest_build.log 2>&1; then
  echo "REFUSING TO MEASURE: guest ELF build failed — see /tmp/W_guest_build.log" >&2
  tail -20 /tmp/W_guest_build.log >&2
  exit 1
fi
git checkout -- executor/programs/rust/hint_min/Cargo.lock \
                executor/programs/rust/hint_multi/Cargo.lock 2>/dev/null || true
PROBE="$(mktemp)"
printf '__probe: ; @echo $(ASM_ARTIFACTS) $(RUST_ARTIFACTS) $(RECURSION_ARTIFACTS) $(RECURSION_VERIFIER_ARTIFACTS)\n' > "$PROBE"
ARTIFACTS="$(make --no-print-directory -f Makefile -f "$PROBE" __probe 2>/tmp/W_probe_err.log)"
rm -f "$PROBE"
[ -s /tmp/W_probe_err.log ] && { echo "  (make probe wrote stderr:)" >&2; tail -5 /tmp/W_probe_err.log >&2; }
expected=0; missing=0
for f in $ARTIFACTS; do
  expected=$((expected+1))
  [ -r "$f" ] || { missing=$((missing+1)); [ "$missing" -le 5 ] && echo "  MISSING: $f" >&2; }
done
[ -r "$ELF" ]   || { echo "  MISSING (block ELF): $ELF" >&2; missing=$((missing+1)); }
[ -r "$INPUT" ] || { echo "  MISSING (block input): $INPUT" >&2; missing=$((missing+1)); }
echo "ARTIFACTS expected=$expected missing=$missing"
if [ "$expected" -eq 0 ] || [ "$missing" -ne 0 ]; then
  echo "REFUSING TO MEASURE: artifact probe expected=$expected missing=$missing" >&2
  exit 1
fi

# ---- 5. the environment, EXPORTED and COUNTED -------------------------------
# Separate exports, never a `VAR=v \` chain: a `#` after a backslash ends the
# continuation and the command runs with a bare environment.
export LFM_CENSUS_ELF="$ELF"
export LFM_CENSUS_INPUT="$INPUT"
export LFM_CENSUS_EPOCH_LOG2="$EPOCH_LOG2"
export LAMBDA_VM_MAX_ROWS_LOG2="$EPOCH_LOG2"
export LAMBDA_VM_MEMPOOL_RELEASE_MB=0
# ★ THE ROOT, NAMED TWICE AND COUNTED: PROVE_ROOT decides where the interior
# stops (`child_level(top)`); the OPTION has no default in the harness.
# LFM_TREE_LEVELS is deliberately NOT exported (:7450).
export LFM_TREE_PROVE_ROOT=1
export LFM_TREE_ROOT_OPTION="$ROOT_OPTION"
# ★ THE RETENTION FLAG, EXPORTED PER ARM (ruled b): the ABBA's A and B run off
# ONE binary at this tip and differ ONLY by this flag. EXPECT_RETENTION tracks
# it — `yes` leaves the leaf-layer retention on (its default), `no` sets it to 0
# so item 3's `capture_leaves` is a no-op and the retention print reads
# admitted 0. Harmless on a build that predates that disable (nothing reads it),
# so wt17 (yes) is unaffected; it only bites the ABBA's control arm.
if [ "$EXPECT_RETENTION" = "no" ]; then
  export LFM_WHIR_RETENTION=0
else
  export LFM_WHIR_RETENTION=1
fi
# ★ THE POSTURE, EXPORTED AND COUNTED. The wrap proofs commit under the
# compile-time RPX block hasher either way; this names what the BASE is proven
# under, and the driver's posture note reports a mismatch rather than deciding.
export LAMBDA_VM_WHIR_HASH="${WHIR_HASH:-rpx}"
export TABLE_PARALLELISM="$K"
if [ "$VRAM_BUDGET_MB" = query ]; then
  unset LAMBDA_VM_VRAM_BUDGET_MB; VB_REQ=""; VB_NOTE="LAMBDA_VM_VRAM_BUDGET_MB=<UNSET by VRAM_BUDGET_MB=query: the device layer queries the driver>"
else
  export LAMBDA_VM_VRAM_BUDGET_MB="$VRAM_BUDGET_MB"; VB_REQ="LAMBDA_VM_VRAM_BUDGET_MB"; VB_NOTE=""
fi
[ -n "$VRAM_SCHED_MB" ] && export LAMBDA_VM_VRAM_SCHED_MB="$VRAM_SCHED_MB"
# The pass-5 tree knobs, so this runs at the D-S posture rather than at a
# default nobody chose. Named by the caller or left at the record's values.
export LFM_TREE_SIBLINGS_L0="${SIBLINGS_L0:-6}"
# ⓘ 4, not v1's 3: the D-S arms this run is scored against name it themselves —
# `ARM ds10a: LFM_TREE_SIBLINGS_L0=6 LFM_TREE_SIBLINGS=4 …`, and ds11a identical.
# A candidate measured at a different sibling count than its control is not a
# comparison, it is two experiments.
export LFM_TREE_SIBLINGS="${SIBLINGS:-4}"
export LFM_TREE_LEVEL_POOL="${LEVEL_POOL:-1}"
export LFM_PRECOMPUTED_TREE_CACHE_CAP="${TREE_CACHE_CAP:-64}"
# ⓘ THE FOUR `run_ds.sh` ALSO EXPORTS, so this run's configuration is the D-S
# arms' configuration and not a subset of it. Counted below rather than merely
# set, because a knob that never reached the process is indistinguishable in a
# log from a lever that did nothing.
#   LFM_EXEC_PARALLEL   the level-scheduled executor; the code default at this
#                       tip, exported so the log says so rather than implying it
#   LFM_PROVE_SPLIT     the per-phase prove stamps
#   LAMBDA_VM_BASE_SPLIT  the base's own stage lines
#                       ⛔ v4's note here read "on the WHIR base expect none",
#                       and it was RIGHT until this sha: the knob was exported
#                       on the WHIR arm "byte identical to the D-S exports" and
#                       reached nothing, because the WHIR base does not go
#                       through `continuation::prove_continuation`. 57% of the
#                       block's wall sat under a knob the log showed as set.
#                       It now reaches both pipelines and means the same thing
#                       on each — section 8b reads it back and REFUSES if the
#                       closure line is absent.
#   LFM_CARD_TRACE      the CARD HOLD lines
export LFM_EXEC_PARALLEL="${EXEC_PARALLEL:-1}"
export LFM_PROVE_SPLIT="${PROVE_SPLIT:-1}"
export LAMBDA_VM_BASE_SPLIT="${BASE_SPLIT:-1}"
export LFM_CARD_TRACE="${CARD_TRACE:-1}"
# ★ ROUND 2's KNOB, EXPORTED AND COUNTED: how many expected hit distances one
# device grind launch covers (`math-cuda/src/grinding.rs`). Default 8 = the
# record posture, so a run that does not name it IS the control.
#
# ⛔ The banner `★ GRIND KNOBS: scan N · grid G · stride …` is REQUIRED in
# section 8c. The search
# prints it on its FIRST device grind, so its absence is not a cosmetic miss:
# it means no device grind was reached and the factor this run is named after
# never acted on anything. A knob quoted in a posture line has to be shown READ
# on the path, and the banner is that proof.
export LAMBDA_VM_GRIND_SCAN_FACTOR="${GRIND_SCAN_FACTOR:-8}"
# ★ AND THE GRID — blocks per grind launch. Default 1024 = the record
# posture. stride = grid × block_dim is what the kernels' early exit leaves
# behind as overshoot past the first hit, so this is the knob with a
# mechanism; the scan factor is a ceiling the launch never reaches.
export LAMBDA_VM_GRIND_GRID="${GRIND_GRID:-1024}"
# ⛔ LFM_TREE_TOP_OVERLAP IS DELIBERATELY NOT SET, although run_ds.sh sets it:
# it folds the global slices into level 0's pool, and this driver has no global
# stage at all — it REFUSES the knob a few guards up. Named here so the
# difference from the D-S arms is a decision on the record rather than an
# omission someone later reads as one.

REQUIRED="LFM_CENSUS_ELF LFM_CENSUS_INPUT LFM_CENSUS_EPOCH_LOG2 \
LFM_TREE_PROVE_ROOT LFM_TREE_ROOT_OPTION LFM_WHIR_RETENTION \
LAMBDA_VM_MAX_ROWS_LOG2 LAMBDA_VM_MEMPOOL_RELEASE_MB LAMBDA_VM_WHIR_HASH \
TABLE_PARALLELISM $VB_REQ LFM_TREE_SIBLINGS_L0 \
LFM_TREE_SIBLINGS LFM_TREE_LEVEL_POOL LFM_PRECOMPUTED_TREE_CACHE_CAP \
LFM_EXEC_PARALLEL LFM_PROVE_SPLIT LAMBDA_VM_BASE_SPLIT LFM_CARD_TRACE \
LAMBDA_VM_GRIND_SCAN_FACTOR LAMBDA_VM_GRIND_GRID"
NREQ=0; NSET=0
echo "--- environment handed to cargo (REQUIRED) ---"
for v in $REQUIRED; do
  NREQ=$((NREQ+1))
  if val="$(printenv "$v")"; then NSET=$((NSET+1)); printf '  %s=%s\n' "$v" "$val"
  else printf '  %s=<UNSET>\n' "$v"; fi
done
echo "--- also exported (not counted) ---"
[ -n "$VB_NOTE" ] && echo "  $VB_NOTE"
env | grep -E '^(LAMBDA_VM_VRAM_SCHED_MB)' | sed 's/^/  /' || echo "  (none)"
if [ "$NSET" -ne "$NREQ" ]; then
  echo "REFUSING TO MEASURE: only $NSET of $NREQ required variables reached the \
environment; the run would measure a different thing under this name" >&2
  exit 1
fi
echo "environment: $NSET of $NREQ required present"
echo
echo "HEAD asserted: $HEAD_SHA on $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<detached>')"
echo "cgroup v2 memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo '-')"
echo "cgroup v1 limit:      $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo '-')"
echo "card idle at claim:   ${GPU_USED} MiB, ${GPU_APPS} apps"
echo

# ---- 6. the device sampler, on the clock the test also prints ---------------
TRACE="$(mktemp /tmp/W_tree_gpu.XXXX)"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits \
  -lms "$SAMPLE_MS" > "$TRACE" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

LOG="$(/usr/bin/time -v timeout 21600 nsys profile -t cuda,osrt --sample=none --cpuctxsw=none -o /root/prof/nsys/g6-wt1201 -f false cargo test --release -p lambda-vm-prover \
    --features cuda --lib "$TEST" -- --ignored --exact --nocapture --test-threads=1 2>&1)"
STATUS=$?
kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null

NOISE='^[[:space:]]+(Command being timed|User time|System time|Percent of CPU|Average|Major|Minor|Voluntary|Involuntary|Swaps|File system|Socket|Signals|Page size|Exit status)'
printf '%s\n' "$LOG" | grep -Ev "$NOISE"
echo
echo "exit status: $STATUS"

# ---- 7. did a test actually run? -------------------------------------------
PASSED="$(printf '%s' "$LOG" | sed -n 's/^test result:.*[^0-9]\([0-9][0-9]*\) passed.*/\1/p' | tail -1)"
echo "tests passed: ${PASSED:-<no result line>}"
if [ "${PASSED:-0}" -lt 1 ]; then
  echo "REFUSING TO REPORT: 0 tests passed — the filter matched nothing or the \
test aborted. This is NOT a green run." >&2
  exit 3
fi

# ---- 8. the read-0 lines, counted rather than eyeballed ---------------------
# ⛔ Each grep is for what the test PRINTS. A filter that matches nothing is
# indistinguishable from a measurement that produced nothing, so every count is
# printed even when it is zero and the notes below say what each should be.
echo
echo "=== READ 0 ==="
echo "WHIR banner:        $(printf '%s' "$LOG" | grep -c 'WHIR HASH')"
echo "wrap IDENTITY:      $(printf '%s' "$LOG" | grep -c 'wrap [0-9]* IDENTITY')"
echo "node IDENTITY:      $(printf '%s' "$LOG" | grep -cE 'L[0-9]+N[0-9]+ .*IDENTITY')"
echo "MARK lines:         $(printf '%s' "$LOG" | grep -c '   MARK ')"
echo "RPX device grinds:  $(printf '%s' "$LOG" | grep -c 'RPX device grinds')"
echo "GRIND KNOBS banner:  $(printf '%s' "$LOG" | grep -c '★ GRIND KNOBS:')"
echo "L0-HARVEST lines:   $(printf '%s' "$LOG" | grep -c 'L0-HARVEST')"
# ---- the wt9 markers, every one printed by the harness at db5ad5e5b's lineage.
echo "GLOBAL WRAP banner: $(printf '%s' "$LOG" | grep -c 'THE WHIR GLOBAL WRAP (the root')"
echo "GLOBAL census:      $(printf '%s' "$LOG" | grep -c 'CENSUS the WHIR GLOBAL wrap')"
echo "GLOBAL LFM_HASH:   $(printf '%s\n' "$LOG" | awk '/CENSUS the WHIR GLOBAL wrap/{f=1} f && /^ +LFM_HASH /{print; exit}')"
printf '%s\n' "$LOG" | grep -E 'the WHIR GLOBAL wrap: prove' | sed 's/^/GLOBAL wrap line:   /'
printf '%s\n' "$LOG" | grep -E 'WHIR INTERIOR COMPOSED' | sed 's/^/INTERIOR:           /'
echo "ROOT banner:        $(printf '%s' "$LOG" | grep -c 'THE WHIR BLOCK-ARTIFACT ROOT — option')"
echo "ROOT census:        $(printf '%s' "$LOG" | grep -c 'CENSUS the WHIR BLOCK-ARTIFACT ROOT')"
echo "ROOT LFM_HASH:     $(printf '%s\n' "$LOG" | awk '/CENSUS the WHIR BLOCK-ARTIFACT ROOT/{f=1} f && /^ +LFM_HASH /{print; exit}')"
printf '%s\n' "$LOG" | grep -E 'GPU dispatches during the WHIR' | sed 's/^/DISPATCHES:         /'
COMPRESSED="$(printf '%s' "$LOG" | grep -c 'THE BLOCK IS COMPRESSED UNDER WHIR')"
echo "COMPRESSED line:    $COMPRESSED (must be 1)"
printf '%s\n' "$LOG" | grep -A3 'THE BLOCK IS COMPRESSED UNDER WHIR' | sed 's/^/   /'
echo "NO-ROOT notice:     $(printf '%s' "$LOG" | grep -c 'NO BLOCK-ARTIFACT ROOT') (must be 0)"
echo "COULD-NOT-BUILD:    $(printf '%s' "$LOG" | grep -c 'COULD NOT BE BUILT') (must be 0)"
if [ "$COMPRESSED" -ne 1 ]; then
  echo "REFUSING TO REPORT: the block-artifact line is absent ($COMPRESSED) — a \
passing test without it is a different arm, not the artifact." >&2
  exit 4
fi
printf '%s\n' "$LOG" | grep -E 'wrap [0-9]* IDENTITY|L[0-9]+N[0-9]+ .*IDENTITY' \
  > /tmp/W_identity_ordered.txt
echo "identity lines kept at /tmp/W_identity_ordered.txt \
($(grep -c . /tmp/W_identity_ordered.txt) lines)"

# ---- 8b. THE BASE'S STAGE BREAKDOWN -----------------------------------------
#
# ⛔ THE COUNTS ARE DERIVED FROM THE EPOCH COUNT THE RUN ITSELF PRINTS, never
# from a remembered 15. A hardcoded expectation turns a changed epoch count into
# a red gate, and a gate that reddens for the wrong reason is how a real finding
# gets explained away.
#
# Per epoch the producer prints 5 lines (execute/collect/build/handoff/epoch)
# and the prover 5 (prep/absorb/commit/prove/prove_epoch); the global stage
# prints 5 more (prep/absorb/commit/prove/prove_global). So:
#   BASE EPOCH lines   = 10 * epochs + 5
#   WHIR PROVE SPLIT   = epochs + 1
echo
echo "=== THE BASE'S STAGE BREAKDOWN ==="
EPOCHS="$(printf '%s' "$LOG" | sed -n 's/.*base (WHIR): \([0-9][0-9]*\) epochs in.*/\1/p' | tail -1)"
BASE_SECS="$(printf '%s' "$LOG" | sed -n 's/.*base (WHIR): [0-9][0-9]* epochs in \([0-9.][0-9.]*\)s.*/\1/p' | tail -1)"
echo "epochs read from the log: ${EPOCHS:-<none>}  base wall: ${BASE_SECS:-<none>}s"
N_STAGE="$(printf '%s' "$LOG" | grep -c '^BASE EPOCH ')"
N_SPLIT="$(printf '%s' "$LOG" | grep -c '^WHIR PROVE SPLIT ')"
echo "BASE EPOCH lines:   $N_STAGE"
echo "WHIR PROVE SPLIT:   $N_SPLIT"
if [ -n "${EPOCHS:-}" ]; then
  echo "  expected: BASE EPOCH $((10 * EPOCHS + 5)), WHIR PROVE SPLIT $((EPOCHS + 1))"
  [ "$N_STAGE" -eq "$((10 * EPOCHS + 5))" ] || echo "  ⚠ BASE EPOCH count is NOT 10*epochs+5" >&2
  [ "$N_SPLIT" -eq "$((EPOCHS + 1))" ] || echo "  ⚠ WHIR PROVE SPLIT count is NOT epochs+1" >&2
fi
# The overlap falsifier: a mixed inner reading must never be read as clean.
echo "OVERLAPPED marks:   $(printf '%s' "$LOG" | grep -c 'OVERLAPPED') (must be 0)"
# The summed table the harness prints, and the verdict it does NOT assert.
printf '%s\n' "$LOG" | grep -E '── WHIR BASE SPLIT over|producer\[Σ\]|prover\[Σ\]|inside prove\[Σ\]|global \(in base\) wall|slowest single table|BOTTLENECK:|headroom if the producer'
# ★ THE REQUIRED MARKER. The harness runs arms A-D over its own records and
# prints this line only when all four pass. Absent = the breakdown did not
# close, or was never taken — in both cases the numbers above must not be
# quoted, so this is a REFUSAL and not a warning.
SPLIT_OK="$(printf '%s' "$LOG" | grep -c 'WHIR BASE SPLIT: closure GREEN')"
SPLIT_OFF="$(printf '%s' "$LOG" | grep -c 'WHIR BASE SPLIT: NOT TAKEN')"
echo "closure GREEN line: $SPLIT_OK (must be 1)"
echo "NOT-TAKEN line:     $SPLIT_OFF (must be 0 — the knob is exported above)"
if [ "$SPLIT_OK" -ne 1 ]; then
  echo "REFUSING TO REPORT: the base's closure line is absent ($SPLIT_OK). The \
stage table either did not close (arms A-D) or was never taken — either way the \
breakdown above is not a measurement anyone may quote." >&2
  exit 5
fi

# ---- 8c. ROUND 2's READING: THE GRIND SLOT, AND THE STAGES THAT SHARE IT ----
#
# ⛔ THE SAME DEVICE SEARCH SERVES ALL THREE STAGES — the base's chains, level
# 0's wraps and the interior's nodes. An arm that reads only the base cannot
# tell a lever from a transfer, so all three are printed together and the
# epoch sum is kept apart from the global's.
echo
echo "=== THE GRIND SLOT ==="
KNOBBANNER="$(printf '%s' "$LOG" | grep -c '★ GRIND KNOBS:')"
echo "GRIND KNOBS banner:  $KNOBBANNER (must be >= 1)"
printf '%s\n' "$LOG" | grep -m1 '★ GRIND KNOBS:' | sed 's/^/  /'
if [ "$KNOBBANNER" -lt 1 ]; then
  echo "REFUSING TO REPORT: no '★ GRIND KNOBS:' line. The knob is exported \
above and the device search prints it on its first grind, so its absence means \
the device grind was never reached — the knobs this run is named after did not \
act on it, and no number here may be attributed to them. The line carries the \
scan factor, the grid AND the stride each arm gets, so the posture is read \
rather than multiplied back out by whoever reads the log." >&2
  exit 6
fi
# The six chain slots summed over the EPOCH records only (`#n`). The global
# stage carries its own and is reported apart, never folded in.
printf '%s\n' "$LOG" | awk '
  /^WHIR PROVE SPLIT #/ { rows++; sum(); }
  /^WHIR PROVE SPLIT GLOBAL/ { g_seen = 1; gsum(); }
  function sum(   i) {
    for (i = 1; i <= NF; i++) {
      if      ($i == "grind")         g  += $(i+1)
      else if ($i == "sumcheck")      s  += $(i+1)
      else if ($i == "fold")          f  += $(i+1)
      else if ($i == "commit_folded") c  += $(i+1)
      else if ($i == "ood")           o  += $(i+1)
      else if ($i == "queries")       q  += $(i+1)
      else if ($i == "open_groups")   og += $(i+1)
    }
  }
  function gsum(   i) {
    for (i = 1; i <= NF; i++) {
      if      ($i == "grind")       gg  += $(i+1)
      else if ($i == "queries")     gq  += $(i+1)
      else if ($i == "open_groups") gog += $(i+1)
    }
  }
  END {
    printf "epoch records summed: %d\n", rows
    printf "chain[Σ epochs] grind %.2f · sumcheck %.2f · fold %.2f · commit_folded %.2f · ood %.2f · queries %.2f\n", g, s, f, c, o, q
    printf "  against open_groups[Σ epochs] %.2f — the six close it, arm E asserts it in-process\n", og
    if (g_seen) printf "global (apart, NEVER added in): grind %.2f · queries %.2f of open_groups %.2f\n", gg, gq, gog
    else        printf "global: NO record — the global stage did not print one\n"
  }'
# ★ THE HARNESS'S OWN SUMS, read rather than re-derived. The instrument prints
# the chain's six and the four inside QUERIES summed over the EPOCH records, so
# the launcher and the harness cannot disagree about a number they both report.
# `tree_rebuild`'s SHARE is round 3's kill condition: retention removes the
# rebuilds and nothing else, so if they are not the bulk of the query openings
# the lever is dead before any lifetime code is written.
echo
printf '%s\n' "$LOG" | grep -E 'chain\[Σ epochs\]|queries\[Σ epochs\]' | sed 's/^ */HARNESS:  /'
echo
echo "=== ★ THE LEAF-LAYER RETENTION ==="
printf '%s\n' "$LOG" | grep -E 'retention\[leaf layers\]' | sed 's/^ */HARNESS:  /'
RETLINE="$(printf '%s' "$LOG" | grep -c 'retention\[leaf layers\]')"
ADMITTED="$(printf '%s' "$LOG" | sed -n 's/.*retention\[leaf layers\] admitted \([0-9][0-9]*\).*/\1/p' | head -1)"
ADMITTED="${ADMITTED:-0}"
echo "retention line: $RETLINE · admitted: $ADMITTED (EXPECT_RETENTION=$EXPECT_RETENTION)"
# ★ RULED (b): the ABBA's A and B are ONE binary, differing only by
# LFM_WHIR_RETENTION. So the CONTROL arm's retention line is PRESENT and reads
# admitted 0 (the flag made capture_leaves a no-op) — NOT absent, as it was when
# A was an older build. The gate is therefore SEMANTIC (was anything retained),
# not structural (is the line there).
if [ "$EXPECT_RETENTION" = "yes" ]; then
  if [ "$RETLINE" -ne 1 ]; then
    echo "REFUSING TO REPORT: this arm was launched as a RETENTION build and the \
harness printed no 'retention[leaf layers]' line. Either the build predates the \
slice or the print was lost — in both cases no number here describes the lever." >&2
    exit 7
  fi
  if [ "$ADMITTED" -lt 1 ]; then
    echo "REFUSING TO REPORT: EXPECT_RETENTION=yes but admitted 0 — the lever did \
not fire (the cap refused everything, or LFM_WHIR_RETENTION was off). The wall \
here is a control, not a B arm." >&2
    exit 7
  fi
  NOTTAKEN="$(printf '%s' "$LOG" | grep -c 'NOT TAKEN: no leaf layer was retained')"
  REFUSED1="$(printf '%s' "$LOG" | grep -c 'FIRST REFUSAL at')"
  echo "  NOT-TAKEN marks: $NOTTAKEN · first-refusal marks: $REFUSED1 (under a cap, refusals are EXPECTED — the cost of refusing is one leaf pass, not a fallback)"
else
  if [ "$ADMITTED" -ne 0 ]; then
    echo "REFUSING TO REPORT: this arm was launched as a CONTROL (no retention) \
but admitted $ADMITTED leaf layers — LFM_WHIR_RETENTION did not disable the \
capture, so A and B are not one binary differing only by the flag." >&2
    exit 7
  fi
  echo "  retention: DISABLED for this arm (admitted 0) — the control"
fi

QSUM="$(printf '%s' "$LOG" | grep -c 'queries\[Σ epochs\]')"
echo "queries[Σ epochs] line: $QSUM (must be 1 — the four-slot instrument is in this build)"
if [ "$QSUM" -ne 1 ]; then
  echo "⚠ NO queries[Σ epochs] LINE: this build predates the four-slot split, so \
the QUERIES number above is one number again and round 3 has no reading here." >&2
fi

# The two other stages the same search serves. Level 0 prints its own wall; the
# interior prints a per-level table and the sum below names the levels it added,
# because a pooled number whose membership is not stated is not comparable.
echo
printf '%s\n' "$LOG" | grep -E '^   level 0: [0-9]+ wraps in' | sed 's/^ */LEVEL 0:  /'
printf '%s\n' "$LOG" | awk '
  /^level arity/ { intable = 1; print "INTERIOR TABLE (level arity cells instructions host-GiB argmax wall):"; next }
  intable && $1 ~ /^[0-9]+$/ && NF >= 7 { print "  " $0; w += $NF; lv = lv " " $1; n++; next }
  intable && $1 !~ /^[0-9]+$/ { intable = 0 }
  END {
    if (n) printf "INTERIOR: Σ wall %.1fs over %d level(s):%s\n", w, n, lv
    else   print "INTERIOR: no per-level table found"
  }'

echo
echo "=== DEVICE, ${SAMPLE_MS}ms sampling ($(grep -c . "$TRACE") samples) ==="
PEAK_LINE="$(sort -t, -k2 -n "$TRACE" | tail -1)"
PEAK_MIB="$(printf '%s' "$PEAK_LINE" | cut -d, -f2 | tr -d ' ')"
PEAK_TS="$(printf '%s' "$PEAK_LINE" | cut -d, -f1)"
PEAK_EPOCH="$(date -d "$PEAK_TS" +%s.%N 2>/dev/null || echo '?')"
echo "device peak ${PEAK_MIB} MiB at ${PEAK_TS} (epoch ${PEAK_EPOCH})"

# ★ THE PEAK SPLIT AT THE BASE'S BOUNDARY. The retained leaf layers live in the
# base and are gone by level 0, so a peak set above the base says nothing about
# what the lever holds. Split by SAMPLE INDEX rather than by parsing timestamps:
# the sampler runs every ${SAMPLE_MS} ms, so the base's window is its first
# base_secs*1000/SAMPLE_MS samples. ⚠ That assumes the sampler kept its period;
# it is a window, not a stopwatch, and the boundary is approximate by design.
BASE_SECS="$(printf '%s' "$LOG" | sed -n 's/.*base (WHIR): [0-9]* epochs in \([0-9.]*\)s.*/\1/p' | head -1)"
if [ -n "${BASE_SECS:-}" ]; then
  awk -F, -v ms="$SAMPLE_MS" -v base="$BASE_SECS" '
    { n++; v = $2 + 0; if (n <= base * 1000 / ms) { if (v > pb) pb = v; nb++ } else { if (v > pa) pa = v; na++ } }
    END {
      printf "device peak IN THE BASE window:  %d MiB over %d samples (~%.1f s)\n", pb, nb, base
      printf "device peak AFTER the base:      %d MiB over %d samples\n", pa, na
      print  "  the base window is the one the leaf layers are held in; a higher peak after it is level 0 and the interior, which hold none"
    }' "$TRACE"
else
  echo "⚠ no 'base (WHIR): N epochs in Ts' line — the peak cannot be split at the base boundary" >&2
fi

# ⛔ FALLBACKS, TWO SURFACES, BOTH HARD GATES. A fallback means device work was
# declined and done on the host instead, holding its buffers there to the end of
# the proof — H4's cliff, and under it the wall and peak describe a different
# pipeline. v9 read a `host fallbacks` line the TREE harness never prints and so
# defaulted to 1 and VOIDED every tree run (exit 8 on unread). v10 reads the two
# lines the harness now prints (per_table_aggregator_tests.rs, the WHOLE-RUN
# block) and refuses LOUDLY on absence rather than treating unread as a fallback.
#
#   commit fallbacks — a WHIR commitment declined the device (`host_fallbacks`,
#                      the commit path, one call site).
#   device fallbacks — an argue-surface reserve was refused in math-cuda
#                      (sumcheck/gkr/columns; `device_fallbacks`, bc6ce9ea2).
#                      THE surface wt16's regression hit, uncounted until now.
# Both must read 0; they are different surfaces, named and gated separately.
CFB="$(printf '%s' "$LOG" | sed -n 's/.*commit fallbacks \([0-9][0-9]*\).*/\1/p' | head -1)"
DFB="$(printf '%s' "$LOG" | sed -n 's/.*device fallbacks \([0-9][0-9]*\).*/\1/p' | head -1)"
printf '%s\n' "$LOG" | grep -E 'commit fallbacks|device fallbacks' | sed 's/^ */  /'
if [ -z "$CFB" ]; then
  echo "REFUSING TO REPORT: no 'commit fallbacks' line — this build predates the \
item-2 harness print. An absent fallback line is an ERROR, not a pass (v9's bug)." >&2
  exit 11
fi
if [ "$CFB" != "0" ]; then
  echo "REFUSING TO REPORT: commit fallbacks $CFB — a WHIR commitment fell back to \
the host, so this run's wall and device peak describe the host pipeline." >&2
  exit 8
fi
if [ -z "$DFB" ]; then
  echo "REFUSING TO REPORT: no 'device fallbacks' line — this build predates the \
math-cuda argue-surface counter (bc6ce9ea2) or its print. An absent fallback \
line is an ERROR, not a pass." >&2
  exit 13
fi
if [ "$DFB" != "0" ]; then
  echo "REFUSING TO REPORT: device fallbacks $DFB — an argue-surface reserve was \
refused and its work moved to the host. This IS wt16's failure mechanism: the \
wall and peak describe the host path, not the device one. THE PRIMARY GATE." >&2
  exit 12
fi

printf '%s\n' "$LOG" | grep -E 'host peak|WHOLE RUN: host peak'
# ⛔ THE HOST-PEAK CEILING, ONE-SIDED. wt16 rose 31.0 → 37.3 GiB when the
# uncapped retention pushed argue to the host; a capped B arm must stay near the
# no-retention baseline. A rise above the ceiling fails; a lower peak does not.
# The BACKSTOP — the primary gate is `device fallbacks 0` above, and while argue
# stays on the device the peak stays near baseline on its own.
RUNPEAK="$(printf '%s' "$LOG" | sed -n 's/.*WHOLE RUN: host peak \([0-9][0-9]*\.[0-9]*\) GiB.*/\1/p' | head -1)"
if [ -z "$RUNPEAK" ]; then
  echo "REFUSING TO REPORT: no 'WHOLE RUN: host peak' line to gate the ceiling against." >&2
  exit 14
fi
echo "WHOLE-RUN host peak: $RUNPEAK GiB · ceiling HOST_PEAK_HI_GIB=$HOST_PEAK_HI_GIB GiB"
if awk -v p="$RUNPEAK" -v c="$HOST_PEAK_HI_GIB" 'BEGIN { exit !(p > c) }'; then
  echo "REFUSING TO REPORT: WHOLE-RUN host peak $RUNPEAK GiB is above the ceiling \
$HOST_PEAK_HI_GIB GiB — the host rise this run was supposed not to have (wt16 was \
37.3; the no-retention baseline is 31.0)." >&2
  exit 14
fi
echo "trace kept at $TRACE"

cat <<'NOTES'

WHAT TO READ BACK — against the lead's wt9 pre-registration, in this order

  ★ 1. The banner and the header: `WHIR HASH: rpx256`, the guest, 2^21
       cycles/epoch, fan-in, and the stage list naming "the GLOBAL stage … and
       the ROOT" (:7511). A run whose stage list ends at the interior (:7515)
       ran without PROVE_ROOT and answers wt8's question again.
  ★ 2. "base: N epochs in T s" — wt8's band at the merged tip is 84.2 s; level
       0 24.0 s; 27.843 GiB device. Outside the bands, the base is the finding.
  ★ 3. The FIFTEEN `wrap k IDENTITY` lines at 145 + out_halves words — wt8's.
  ★ 4. ★ THE WHIR GLOBAL WRAP: the banner's `N cross-epoch sub-proofs over E
       epochs and P touched pages`, then `★ CENSUS the WHIR GLOBAL wrap: cells
       (instructions)` — the instruction count is the cross-epoch program's F1
       at the block (V1j's band 2–4 M with the hybrid), and the `LFM_HASH` row's
       committed height is the go/no-go the STARK global could not pass
       (2^21 × 329 columns = 26.22 GiB, aborted unsliced). Then `the WHIR GLOBAL
       wrap: prove · verify · N published words (E bookend roots)`: N = 2 + 4·E.
  ★ 5. `★★★ WHIR INTERIOR COMPOSED — C proof(s) at level L`: under option A the
       root REPLACES the top node, so L = top − 1 and C = fan_in (2 at level 3
       for 15 wraps under fan-in 2); a `1 proof at level 4` here is wt8's tree
       with a root bolted on top — a different artifact from the one named.
  ★ 6. ★★★ THE WHIR BLOCK-ARTIFACT ROOT — option A: its census, its LFM_HASH
       row, its dispatch count (must be > 0: the grind reached the device), and
       `★★★ THE BLOCK IS COMPRESSED UNDER WHIR … N published words (=
       root_schema_words(num_reg, out_halves, AssertOnly))` — the identity is
       printed with its inputs; read N against the form, never against a
       remembered number.
  ★ 7. The WHOLE-RUN wall against wt8's 142.5 s: what the global stage and the
       root ADD is the first version's price, scored against ds12's global
       slices (6.5 + 8.5 s) and its root arm.

  ★ 8. ★★★ THE BASE'S BREAKDOWN — what this run exists for. Read in this
       order, and read the CLOSURE LINE FIRST: without `closure GREEN` the
       table is not a measurement. Then
         · `producer[Σ] … handoff Xs` — THE BOTTLENECK. A large handoff share
           means the producer waits at the unbuffered channel and THE PROVER
           sets the wall, so nothing spent on execute/collect/build buys any;
           a handoff near zero means the producer does, and the prover idled.
           The script prints the verdict; it is a READING, not an assertion.
         · `prover[Σ] … prove Xs` then `inside prove[Σ] argue / open_groups /
           open_prepared` — which of the argument and the openings dominates is
           what picks round 2's lever. `argue` is a SERIAL loop over the
           epoch's tables, so it is a wall as well as a sum; `open_groups` is
           two full WHIR chains per epoch (`epoch_groups` = [n-1, 1]).
         · `slowest single table in any epoch` — the sum alone cannot tell
           fifty even tables from one that dominates, and those want opposite
           levers.
         · `global (in base) wall` — the cross-epoch stage, inside the 84.5 s.
  ★ 9. ⛔ THE CONTROL THAT MAKES 8 QUOTABLE: the wall, the census, the pins and
       the published words must be UNMOVED against wt9 (148.9 s) and wt11
       (149.4 s), band ≈ 0.5 s. Timers move no bytes. A base outside 84.5-85.0 s
       is the finding, not the breakdown.

  ⚠ ONE test run --exact. NOT a suite result.
  ⛔ `1 passed` AND the COMPRESSED line: the harness asserts the root under
     PROVE_ROOT, this script asserts the line — two sources, or no artifact.
  ⛔ `closure GREEN` is a THIRD source and it is required: exit 5 without it.
NOTES

exit "$STATUS"
