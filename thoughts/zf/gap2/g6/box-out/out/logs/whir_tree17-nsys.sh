#!/usr/bin/env bash
# whir_tree16.sh = whir_tree15.sh with v8 -> v9 (bea12196…, O3: EXPECT_RETENTION=yes|no REQUIRED, the device peak split at the base, fallbacks a hard gate) and the arm-declared EXPECT_RETENTION passed through and printed; serves wt16 and the ABBA wt17–wt20 by the tag argument. Derived by the lead 2026-09-19 23:4xZ.
# whir_tree15.sh = whir_tree14.sh (31ce1968…) with the A-tree script v5 → v8 (7a30ea94…, O2: both grind knobs REQUIRED + the queries[Σ epochs] read-back). Derived by the lead 2026-09-19 22:3xZ for wt15 at c766c2327.
# whir_tree14.sh <sha8> <tag> <worktree> — wt14: ROUND 2, THE GATED BLOCK RE-READ WITH THE SEVEN SLOTS (B2 da5008492; the v5 script; everything else = whir_tree13.sh).
# Derived from whir_tree8.sh (563e1327…): the worktree is an ARGUMENT (the harness
# lineage moves with W1h's tip), ROOT_OPTION=A is exported for the v4 script, and
# the v4 takes no `all` — LFM_TREE_LEVELS must be UNSET under PROVE_ROOT.
# Everything else byte-identical to wt8's exports (the query posture, never-purge,
# the four D-S process exports, SIBLINGS=4, W_ALLOW_SPAWNS=1).
set -u
SHA="${1:?usage: whir_tree9.sh <sha8> <tag> <worktree>}"
TAG="${2:?usage: whir_tree9.sh <sha8> <tag> <worktree>}"
WT="${3:?usage: whir_tree9.sh <sha8> <tag> <worktree>}"
[ -d "$WT/.git" ] || [ -f "$WT/.git" ] || { echo "WT REFUSING: worktree $WT is missing"; exit 3; }
LOG=/root/prof/${TAG}-tree.log
[ -e "$LOG" ] && { echo "WT REFUSING: $LOG exists — a tag is never reused"; exit 4; }
ELF=/root/fixtures/ethrex_8f826601.elf
INPUT=/root/fixtures/ethrex_mainnet_25368371_573004e6.bin
ELF_SHA_WANT=8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a
INPUT_SHA_WANT=573004e62e3680a00d3cdbae19dc4897e2ec60d6ec0c1d05d9ef118cb8aef17f
exec > "$LOG" 2>&1
echo "WT launcher start $(date -u +%Y-%m-%dT%H:%M:%SZ) tag=$TAG expect=$SHA worktree=$WT head=$(git -C "$WT" rev-parse --short=8 HEAD)"
ELF_SHA="$(sha256sum "$ELF" | cut -c1-64)"
INPUT_SHA="$(sha256sum "$INPUT" | cut -c1-64)"
echo "WT guest ELF:   $ELF $(stat -c %s "$ELF") B sha256 $ELF_SHA"
echo "WT block input: $INPUT $(stat -c %s "$INPUT") B sha256 $INPUT_SHA"
if [ "$ELF_SHA" != "$ELF_SHA_WANT" ] || [ "$INPUT_SHA" != "$INPUT_SHA_WANT" ]; then
  echo "WT REFUSING: a fixture is not the record's (want ELF $ELF_SHA_WANT, input $INPUT_SHA_WANT)"
  echo "WT _RC=9"; echo "WT DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit 9
fi
echo "WT spawn sites read by hand: multilinear_continuation.rs=$(grep -c 'scope\.spawn' "$WT/prover/src/multilinear_continuation.rs"; true) continuation.rs=$(grep -c 'scope\.spawn' "$WT/prover/src/continuation.rs"; true)"

export _RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1
export LFM_EXEC_PARALLEL=1 LFM_PROVE_SPLIT=1 LAMBDA_VM_BASE_SPLIT=1 LFM_CARD_TRACE=1
export SIBLINGS=4
export VRAM_BUDGET_MB=query
export W_ALLOW_SPAWNS=1
# ★ THE ROOT: option A (the root REPLACES the top node; the STARK D-S runs A).
export ROOT_OPTION=A
echo "WT launcher exports: _RJEM_MALLOC_CONF=$_RJEM_MALLOC_CONF LFM_EXEC_PARALLEL=$LFM_EXEC_PARALLEL LFM_PROVE_SPLIT=$LFM_PROVE_SPLIT LAMBDA_VM_BASE_SPLIT=$LAMBDA_VM_BASE_SPLIT LFM_CARD_TRACE=$LFM_CARD_TRACE SIBLINGS=$SIBLINGS VRAM_BUDGET_MB=$VRAM_BUDGET_MB W_ALLOW_SPAWNS=$W_ALLOW_SPAWNS ROOT_OPTION=$ROOT_OPTION LFM_TREE_TOP_OVERLAP=${LFM_TREE_TOP_OVERLAP:-<UNSET>} LFM_TREE_LEVELS=${LFM_TREE_LEVELS:-<UNSET>} A_CACHE_DIR=${A_CACHE_DIR:-<UNSET>}"
echo "WT script md5: $(md5sum /root/zf/g6/launch/A-tree-whir.v10-nsys.sh | cut -c1-32)"

echo "WT EXPECT_RETENTION=${EXPECT_RETENTION:?the arm must declare yes|no}"
EXPECT_RETENTION="$EXPECT_RETENTION" EXPECT_HEAD="$SHA" bash /root/zf/g6/launch/A-tree-whir.v10-nsys.sh "$WT" "$ELF" "$INPUT" 21
RC=$?
echo "WT _RC=$RC"
echo "WT DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
