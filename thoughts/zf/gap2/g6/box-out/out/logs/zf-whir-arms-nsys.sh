#!/usr/bin/env bash
# zf-whir-arms.sh — the ZF campaign's WHIR block arms, run through the UNCHANGED record chain.
#
#   bash zf-whir-arms.sh [--dry-run] <sha9> <first-wt-number> <arm>...
#     <arm> = NAME:KNOB=V[,KNOB=V...]  or  NAME:-   (no ZF knob)
#             KNOB is a ZF knob, short (CAP, FRI, ONE_ROW, WHIR_CAP, WHIR_FOLDS, WHIR_STACK) or full
#             (LAMBDA_VM_ZF_CAP ...). A fold list may contain commas: WHIR_FOLDS=4,4,3.
#   e.g. bash zf-whir-arms.sh 1a2b3c4d5 40 A:- B:WHIR_CAP=auto B:WHIR_CAP=auto A:-
#   env: ZF_BRANCH=<branch>  fetch only that branch (default: `git fetch origin`, all branches)
#        ZF_ALLOW_OTHER_KNOBS=1  let an arm set a non-ZF env var (printed; for controls only)
#
# Per arm, in the order given, one tag each (wt<N>, wt<N+1>, …): assert the card idle and the
# box lock free, re-assert HEAD, then call `/root/whir_tree17.sh <sha> <tag> <wt>` EXACTLY as
# confirm-defaults.sh does (wt39, the no-knob record): no tree knobs, the same `env -u` list,
# EXPECT_RETENTION=yes HOST_PEAK_HI_GIB=33.5 ROOT_OPTION=A VRAM_BUDGET_MB=query — plus that
# arm's ZF knobs and nothing else (every LAMBDA_VM_ZF_* is `env -u`'d first, so a knob left in
# the caller's shell cannot leak into an A arm).
#
# Read, not assumed (A-tree-whir.v10.sh, md5 88dd1587…, lines cited from the local copy
# thoughts/zf/launchers/box-root/A-tree-whir.v10.sh):
#   · v10 PASSES UNKNOWN ENV VARS THROUGH to cargo: it never runs `env -i`/`env -u`, it only
#     `export`s its own set (:328-407) and runs `cargo test` in the same shell (:451), so
#     LAMBDA_VM_ZF_* reach the test process. whir_tree17.sh also clears nothing (it exports
#     its own set and calls v10 with EXPECT_RETENTION/EXPECT_HEAD prefixed).
#     Its refusal list (:214-229) names only tree/root knobs — no LAMBDA_VM_ZF_* name.
#   · v10 TAKES /root/.box.lock ITSELF (`exec 9>/root/.box.lock; flock -n 9`, :115-119). A lock
#     held here would make that nested flock refuse, so this layer only PROBES the box lock
#     (`flock -n /root/.box.lock true`, the confirm-defaults idiom) and serialises ZF jobs on
#     its OWN lock, /root/zf/.zf.lock (shared with zf-stark-arms.sh and zf-gates.sh).
#   · whir_tree17.sh's exit status is its last `echo`; the record chain's verdict is the
#     `WT _RC=<n>` line it writes into the tree log. That line is what this script gates on.
#
# After each arm: copy the tree log to /root/zf/<tag>/, run zf_summary.py on it, and REFUSE to
# continue if the arm set a ZF knob and the log lacks a matching `ZF FORMAT:` banner (a
# no-knob arm may print the default banner or none; which one is reported). The newest
# lambda_vm_prover test binary is md5'd after every arm: an ABBA is ONE binary, and a knob
# read at build time would silently rebuild between arms.
#
# Outputs: /root/zf/<tag>/{tree.log,launcher.out,summary.md,banner.txt}; the run directory
# /root/zf/wt<first>-wt<last>/{driver.log,manifest.tsv,ab.md}. The record chain's own log stays
# at /root/prof/<tag>-tree.log.
#
# Exit codes: 0 all arms green + A/B written · 1 usage · 3 lock held / tag already used ·
# 4 git/worktree/HEAD · 5 preflight (launcher, fixture, python) · 9 card not idle ·
# 10 ZF banner missing or mismatched · 11 an arm's record chain failed (WT _RC != 0 or absent) ·
# 12 zf_summary.py could not read an arm log · 13 the prover binary changed between arms ·
# 14 the final A/B identity check is RED (same knobs, different program ids).
set -euo pipefail

HERE=/root/zf/bin
SUMMARY="$HERE/zf_summary.py"
SRC=/workspace/lambda_vm
WT=/workspace/lambda_vm-zf-whir
DRIVER=/root/zf/g6/launch/whir_tree17-nsys.sh
V10=/root/zf/g6/launch/A-tree-whir.v10-nsys.sh
V10_MD5=1ccde946d55ca1a1ee0bbdb146661128
DRIVER_MD5=6371860a77a408650494847cab97f49e
PROF=/root/prof
ZFROOT=/root/zf
ELF=/root/fixtures/ethrex_8f826601.elf
INPUT=/root/fixtures/ethrex_mainnet_25368371_573004e6.bin
ELF_SHA_WANT=8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a
INPUT_SHA_WANT=573004e62e3680a00d3cdbae19dc4897e2ec60d6ec0c1d05d9ef118cb8aef17f
ZF_ALL="LAMBDA_VM_ZF_CAP LAMBDA_VM_ZF_FRI LAMBDA_VM_ZF_ONE_ROW LAMBDA_VM_ZF_WHIR_CAP LAMBDA_VM_ZF_WHIR_FOLDS LAMBDA_VM_ZF_WHIR_STACK"
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export SYSROOT_DIR=/opt/lambda-vm-sysroot

log() { echo "ZFW $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
die() { local rc=$1; shift; log "REFUSE($rc): $*"; echo "VERDICT: REFUSED rc=$rc — $*"; exit "$rc"; }
usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
[ $# -ge 3 ] || usage
SHA9="$1"; N0="$2"; shift 2
[[ "$SHA9" =~ ^[0-9a-f]{9}$ ]] || die 1 "sha must be 9 lowercase hex characters, got '$SHA9'"
[[ "$N0" =~ ^[0-9]+$ ]] || die 1 "first wt number must be an integer, got '$N0'"
command -v python3 >/dev/null || die 5 "python3 not found"
[ -r "$SUMMARY" ] || die 5 "zf_summary.py not found next to this script ($SUMMARY)"

# ---- parse every arm BEFORE touching anything (one tested parser: zf_summary.py --parse-arm)
NAMES=(); KNOBS=(); TAGS=()
i=0
for spec in "$@"; do
  out="$(python3 "$SUMMARY" --parse-arm "$spec")" || die 1 "arm '$spec' refused by the parser (see above)"
  NAMES+=("$(printf '%s\n' "$out" | head -1)")
  KNOBS+=("$(printf '%s\n' "$out" | tail -n +2 | tr '\n' ' ' | sed 's/ $//')")
  TAGS+=("wt$((N0 + i))")
  i=$((i + 1))
done
NARM=${#TAGS[@]}
RUN="$ZFROOT/${TAGS[0]}-${TAGS[$((NARM - 1))]}"

log "plan: sha $SHA9 · worktree $WT · driver $DRIVER -> $V10 · $NARM arm(s)"
for ((k = 0; k < NARM; k++)); do
  log "  ${TAGS[$k]}  arm ${NAMES[$k]}  knobs: ${KNOBS[$k]:-<none>}"
done

# ---- a tag is never reused: check ALL of them before the first arm
for t in "${TAGS[@]}"; do
  [ ! -e "$PROF/$t-tree.log" ] || die 3 "$PROF/$t-tree.log exists — tag $t is used"
  [ ! -e "$ZFROOT/$t" ] || die 3 "$ZFROOT/$t exists — tag $t is used"
done
[ ! -e "$RUN" ] || die 3 "$RUN exists — this run id is used"

# ---- preflight (also the whole of --dry-run)
PRE=0
chk() { if eval "$2"; then log "  ok    $1"; else log "  FAIL  $1"; PRE=1; fi; }
log "preflight:"
chk "record driver $DRIVER md5 $DRIVER_MD5 (called unchanged)" "[ -r '$DRIVER' ] && [ \"\$(md5sum '$DRIVER' | cut -c1-32)\" = '$DRIVER_MD5' ]"
chk "launcher $V10 md5 $V10_MD5" "[ -r '$V10' ] && [ \"\$(md5sum '$V10' | cut -c1-32)\" = '$V10_MD5' ]"
chk "block ELF sha256 = record" "[ -r '$ELF' ] && [ \"\$(sha256sum '$ELF' | cut -c1-64)\" = '$ELF_SHA_WANT' ]"
chk "block input sha256 = record" "[ -r '$INPUT' ] && [ \"\$(sha256sum '$INPUT' | cut -c1-64)\" = '$INPUT_SHA_WANT' ]"
chk "source repo $SRC" "[ -d '$SRC/.git' ]"
chk "nvidia-smi present" "command -v nvidia-smi >/dev/null"
chk "flock present" "command -v flock >/dev/null"
chk "zf_summary.py parses a knob arm" "python3 '$SUMMARY' --parse-arm 'X:CAP=auto' >/dev/null"
if [ "$DRY" = 1 ]; then
  for ((k = 0; k < NARM; k++)); do
    log "  would run: env -u LFM_WHIR_PREFETCH -u LAMBDA_VM_TREE_BUSY_PROBE -u LFM_TREE_TOP_OVERLAP -u LFM_CENSUS_FAN_IN -u TOP_OVERLAP -u FAN_IN $(for z in $ZF_ALL; do printf -- '-u %s ' "$z"; done)${KNOBS[$k]} EXPECT_RETENTION=yes HOST_PEAK_HI_GIB=33.5 ROOT_OPTION=A VRAM_BUDGET_MB=query bash $DRIVER $SHA9 ${TAGS[$k]} $WT"
  done
  if [ "$PRE" = 0 ]; then echo "VERDICT: DRY-RUN OK — $NARM arm(s) parsed, preflight green, nothing run"; exit 0; fi
  echo "VERDICT: DRY-RUN PREFLIGHT RED — nothing run"; exit 5
fi
[ "$PRE" = 0 ] || die 5 "preflight red (see FAIL lines)"

# ---- single instance: OUR lock, never the box lock (v10 takes that one itself)
mkdir -p "$ZFROOT"
exec 8>"$ZFROOT/.zf.lock"
flock -n 8 || die 3 "$ZFROOT/.zf.lock is held — another ZF job is running"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
log "run dir $RUN (driver.log is this output)"
printf '# tag\tname\tknobs\tlog\n' > "$RUN/manifest.tsv"

card_idle() {
  local used apps
  used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)"
  log "card: ${used} MiB used, ${apps} compute app(s)"
  [ "${used:-99999}" -lt 500 ] && [ "${apps:-1}" -eq 0 ]
}
box_lock_free() { flock -n /root/.box.lock true; }
prover_running() {  # argv[0]-anchored: a prover TEST BINARY from any /workspace/lambda_vm* target
  ps -eww -o pid=,args= | awk '$2 ~ /^\/workspace\/lambda_vm[-a-z0-9]*\/target\/release\/deps\/lambda_vm_prover-/ {print $1}'
}
prover_bin() {  # the newest lambda_vm_prover test executable in the worktree's target
  local f b=""
  for f in "$WT"/target/release/deps/lambda_vm_prover-*; do
    [[ "$f" == *.d ]] && continue
    [ -x "$f" ] || continue
    if [ -z "$b" ] || [ "$f" -nt "$b" ]; then b="$f"; fi
  done
  printf '%s' "$b"
}

# ---- the worktree, detached at the sha
log "fetch: origin ${ZF_BRANCH:-<all branches>}"
if [ -n "${ZF_BRANCH:-}" ]; then git -C "$SRC" fetch -q origin "$ZF_BRANCH" || die 4 "git fetch origin $ZF_BRANCH failed"
else git -C "$SRC" fetch -q origin || die 4 "git fetch origin failed"; fi
FULL="$(git -C "$SRC" rev-parse --verify -q "${SHA9}^{commit}")" || die 4 "$SHA9 is not a commit in $SRC after the fetch (pushed? right ZF_BRANCH?)"
if [ ! -e "$WT/.git" ]; then
  log "creating worktree $WT at $FULL"
  git -C "$SRC" worktree add --detach "$WT" "$FULL" >/dev/null 2>&1 || die 4 "git worktree add $WT failed"
fi
checkout_sha() {
  # the guest build dirties these lock files (confirm-defaults.sh's reset, verbatim), then any
  # other tracked edit, then the detached checkout; the assert below is what decides
  git -C "$WT" checkout -q -- executor/programs/rust/hint_min/Cargo.lock executor/programs/rust/hint_multi/Cargo.lock executor/programs/rust/Cargo.lock recursion/Cargo.lock 2>/dev/null || true
  git -C "$WT" checkout -q -- . 2>/dev/null || true
  git -C "$WT" checkout -q --detach "$FULL" 2>/dev/null || true
  local head; head="$(git -C "$WT" rev-parse HEAD)"
  [ "$head" = "$FULL" ] || die 4 "HEAD of $WT is $head, expected $FULL (EXPECT_HEAD)"
  log "HEAD asserted: $head"
}
checkout_sha

BIN_MD5=""
for ((k = 0; k < NARM; k++)); do
  TAG="${TAGS[$k]}"; NAME="${NAMES[$k]}"; KN="${KNOBS[$k]}"
  ADIR="$ZFROOT/$TAG"; TLOG="$PROF/$TAG-tree.log"
  log "===== ARM $TAG ($((k + 1))/$NARM) name $NAME · ZF knobs: ${KN:-<none>}"
  [ ! -e "$TLOG" ] || die 3 "$TLOG appeared since the start — tag $TAG is used"
  card_idle || die 9 "card not idle before $TAG"
  box_lock_free || die 3 "/root/.box.lock is held before $TAG — another run owns the box"
  [ -z "$(prover_running)" ] || die 9 "a lambda_vm_prover test binary is running before $TAG (pids $(prover_running | tr '\n' ' '))"
  checkout_sha
  mkdir -p "$ADIR"
  UNSET=(-u LFM_WHIR_PREFETCH -u LAMBDA_VM_TREE_BUSY_PROBE -u LFM_TREE_TOP_OVERLAP -u LFM_CENSUS_FAN_IN -u TOP_OVERLAP -u FAN_IN)
  for z in $ZF_ALL; do UNSET+=(-u "$z"); done
  # shellcheck disable=SC2206  # KN is validated by the parser: NAME=VALUE words, no spaces
  SETK=($KN)
  log "  knobs exported to the process: ${KN:-<none>} (every other LAMBDA_VM_ZF_* unset)"
  T0=$(date +%s)
  set +e
  env "${UNSET[@]}" ${SETK[@]+"${SETK[@]}"} \
    EXPECT_RETENTION=yes HOST_PEAK_HI_GIB=33.5 ROOT_OPTION=A VRAM_BUDGET_MB=query \
    bash "$DRIVER" "$SHA9" "$TAG" "$WT" > "$ADIR/launcher.out" 2>&1
  DRC=$?
  set -e
  log "  record chain returned $DRC after $(( $(date +%s) - T0 )) s (its verdict is the WT _RC line)"
  [ -r "$TLOG" ] || die 11 "$TAG: no tree log at $TLOG — the driver refused before logging (see $ADIR/launcher.out: $(head -c 300 "$ADIR/launcher.out"))"
  cp "$TLOG" "$ADIR/tree.log"
  WRC="$(sed -n 's/^WT _RC=\([0-9][0-9]*\).*/\1/p' "$TLOG" | tail -1)"
  log "  WT _RC=${WRC:-<ABSENT>}"
  B="$(prover_bin)"
  if [ -n "$B" ]; then
    M="$(md5sum "$B" | cut -c1-32)"
    log "  prover binary $(basename "$B") md5 $M"
    if [ -z "$BIN_MD5" ]; then BIN_MD5="$M"
    elif [ "$M" != "$BIN_MD5" ]; then die 13 "$TAG ran a DIFFERENT prover binary ($M vs $BIN_MD5): not one binary — a knob is read at build time, or the tree changed"; fi
  else
    log "  ⚠ no lambda_vm_prover-* binary found under $WT/target/release/deps"
  fi
  set +e
  python3 "$SUMMARY" "$TLOG" > "$ADIR/summary.md" 2>&1; SRC_RC=$?
  # shellcheck disable=SC2086
  python3 "$SUMMARY" --check-banner "$TLOG" $KN > "$ADIR/banner.txt" 2>&1; BRC=$?
  set -e
  log "  $(cat "$ADIR/banner.txt")"
  grep -E '^\| (block wall|host peak \(WHOLE|base|level 0|interior|root stage|device peak|fallbacks|PROVED|SHAPE)' "$ADIR/summary.md" | sed "s/^/  $TAG /" || true
  grep -E '^identities:|PROBLEMS' "$ADIR/summary.md" | sed "s/^/  $TAG /" || true
  printf '%s\t%s\t%s\t%s\n' "$TAG" "$NAME" "${KN:--}" "$TLOG" >> "$RUN/manifest.tsv"
  [ "${WRC:-x}" = "0" ] || die 11 "$TAG: the record chain did not pass (WT _RC=${WRC:-ABSENT}) — v10's REFUSING line is in $TLOG"
  [ "$BRC" = 0 ] || die 10 "$TAG: $(cat "$ADIR/banner.txt") — no number from this arm may be quoted"
  [ "$SRC_RC" = 0 ] || die 12 "$TAG: zf_summary.py rc=$SRC_RC on $TLOG (see $ADIR/summary.md)"
  log "  $TAG green"
done

log "===== A/B over $NARM arm(s)"
set +e
python3 "$SUMMARY" --manifest "$RUN/manifest.tsv" > "$RUN/ab.md" 2>&1; ABRC=$?
set -e
sed -n '/^## A\/B/,$p' "$RUN/ab.md"
[ "$ABRC" = 0 ] || die 14 "A/B summary rc=$ABRC (4 = same knobs, different program ids) — see $RUN/ab.md"
echo "VERDICT: WHIR ARMS GREEN — $NARM arm(s) ${TAGS[0]}..${TAGS[$((NARM - 1))]} at $SHA9; A/B in $RUN/ab.md"
