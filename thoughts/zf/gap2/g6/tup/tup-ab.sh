#!/usr/bin/env bash
# TRACE-UPLOAD A/B — the default (A: one stream of pageable column copies, DECODE's prepared opening before the pipeline)
# against LAMBDA_VM_TRACE_UPLOAD=1 (B: the epoch's columns staged from 4 threads through staging pairs, zero tails of
# >= 64 KiB zeroed on the card; the opening on a helper beside epoch 0) on #1010's production tree, A B B A, at
# fix2/1010-trace-upload d0bf1ad8c (#1010's keep-futile head 73342bc66 + the knob, default off). Pre-registered in
# G6-LEDGER.md §7 (fix2/1010-g6-census), before the run.
#
# usage: bash tup-ab.sh <first-wt-number>        (lane tags: 1290 → wt1290..wt1293)
#
# The knob reaches the tree through the lead's harness, unchanged (zf-whir-arms.sh, md5 86951bc2), with
# ZF_ALLOW_OTHER_KNOBS=1. This script unsets LAMBDA_VM_TRACE_UPLOAD{,_THREADS} before the call: an A arm must read the
# default. Every arm prints which way it ran (COLUMNS UPLOAD … (pageable|staged x4); BASE HEAD (WHIR): DECODE prepared
# [(ahead)]); the readout refuses an arm whose lines are not its own.
# Then tup_readout.py (next to this script, md5 pinned) reads the arms from the harness's manifest.
# Output: /root/zf/tup-wt<N>/{tup.log,readout.txt}; the harness's run is /root/zf/wt<N>-wt<N+3>/.
# Exit: 0 green · 1 usage · 4 git/HEAD · 5 preflight · 9 card not idle · 11 harness red.
set -euo pipefail

EXPECT_HEAD=d0bf1ad8ca24ca04bef7072a346477d4c26dbdef
BRANCH=fix2/1010-trace-upload
HARNESS=/root/zf/bin/zf-whir-arms.sh
HARNESS_MD5=86951bc260eafb8461055611b501665c
PREFIX=wt
[ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]] || { sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 1; }
N0="$1"
SHA9=${EXPECT_HEAD:0:9}
SRC=/workspace/lambda_vm
SUMMARY_MD5=a0228cf0dcf71b26cf77a658d830382b
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READOUT="$HERE/tup_readout.py"
READOUT_MD5=1309e6265c87de78ce512e3f528066ea
ARMS=(A:- B:LAMBDA_VM_TRACE_UPLOAD=1 B:LAMBDA_VM_TRACE_UPLOAD=1 A:-)
LAST=$((N0 + ${#ARMS[@]} - 1))
RUN="/root/zf/${PREFIX}${N0}-${PREFIX}${LAST}"
OUT="/root/zf/tup-wt${N0}"
[ ! -e "$OUT" ] || { echo "VERDICT: REFUSED — $OUT exists (tag used)"; exit 1; }
mkdir -p "$OUT"
LOG="$OUT/tup.log"
say() { echo "TUP $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
refuse() { local rc=$1; shift; say "REFUSE($rc): $*"; echo "VERDICT: REFUSED rc=$rc — $*" | tee -a "$LOG"; exit "$rc"; }

say "TRACE-UPLOAD A/B · expect $EXPECT_HEAD ($BRANCH) · arms ${ARMS[*]} · tags $PREFIX$N0..$PREFIX$LAST"

# ---- the card must be idle before anything expensive
used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | { grep -c . || true; })"
say "card: ${used} MiB used, ${apps} compute app(s)"
[ "${used:-99999}" -lt 500 ] && [ "${apps:-1}" -eq 0 ] || refuse 9 "card not idle (${used} MiB, ${apps} apps)"

# ---- preflight: the pinned harness, its summary, the readout, and the sha on the branch
[ -r "$HARNESS" ] && [ "$(md5sum "$HARNESS" | cut -c1-32)" = "$HARNESS_MD5" ] \
  || refuse 5 "$HARNESS missing or not md5 $HARNESS_MD5"
[ "$(md5sum "$(dirname "$HARNESS")/zf_summary.py" | cut -c1-32)" = "$SUMMARY_MD5" ] \
  || refuse 5 "zf_summary.py next to the harness is not md5 $SUMMARY_MD5"
[ -r "$READOUT" ] && [ "$(md5sum "$READOUT" | cut -c1-32)" = "$READOUT_MD5" ] \
  || refuse 5 "$READOUT missing or not md5 $READOUT_MD5"
python3 -c 'import sys; assert sys.version_info >= (3, 8)' || refuse 5 "python3 >= 3.8 needed"
git -C "$SRC" fetch -q origin "$BRANCH" || refuse 4 "git fetch origin $BRANCH failed"
full="$(git -C "$SRC" rev-parse --verify -q "${SHA9}^{commit}")" || refuse 4 "$SHA9 is not a commit after the fetch"
[ "$full" = "$EXPECT_HEAD" ] || refuse 4 "$SHA9 resolves to $full, expected $EXPECT_HEAD"
say "HEAD asserted: $full (the harness re-asserts it in its own worktree before every arm)"

# ---- the arms, through the harness (it takes /root/zf/.zf.lock and checks the card before each arm)
unset LAMBDA_VM_TRACE_UPLOAD LAMBDA_VM_TRACE_UPLOAD_THREADS
say "harness: ZF_BRANCH=$BRANCH ZF_ALLOW_OTHER_KNOBS=1 bash $HARNESS $SHA9 $N0 ${ARMS[*]}"
set +e
ZF_BRANCH="$BRANCH" ZF_ALLOW_OTHER_KNOBS=1 bash "$HARNESS" "$SHA9" "$N0" "${ARMS[@]}" 2>&1 | tee -a "$LOG"
HRC=${PIPESTATUS[0]}
set -e
say "harness returned $HRC"

# ---- the readout, over whatever arms the manifest holds
READ_LINE="READOUT: not run (no manifest)"
if [ -r "$RUN/manifest.tsv" ]; then
  set +e
  python3 "$READOUT" "$RUN/manifest.tsv" > "$OUT/readout.txt" 2>&1
  RRC=$?
  set -e
  tee -a "$LOG" < "$OUT/readout.txt"
  READ_LINE="$(grep '^READOUT: ' "$OUT/readout.txt" | tail -1 || true)"
  [ -n "$READ_LINE" ] || READ_LINE="READOUT: rc=$RRC, no verdict line (see $OUT/readout.txt)"
fi
if [ "$HRC" -ne 0 ]; then
  echo "VERDICT: HARNESS RED rc=$HRC · ${READ_LINE#READOUT: }" | tee -a "$LOG"
  exit 11
fi
echo "VERDICT: ${READ_LINE#READOUT: }" | tee -a "$LOG"
