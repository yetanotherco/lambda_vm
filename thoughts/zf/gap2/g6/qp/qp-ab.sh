#!/usr/bin/env bash
# QUIET-PRODUCER A/B — the default (A) against LAMBDA_VM_QUIET_PRODUCER=1 (B: the WHIR base producer's stages and the
# prover's argues never overlap; timing only) on #1010's production tree, A B B A, at fix2/1010-quiet-producer
# 6e0b5f731 (#1010's next head 7c8272701 + the knob, default off). Pre-registered in G6-LEDGER.md §11
# (fix2/1010-g6-census), before the run: (a) β for stage 1b; (b) the quiet producer as a lever.
#
# usage: bash qp-ab.sh <first-wt-number>        (lane tags: 1314 → wt1314..wt1317)
# The knob reaches the tree through the lead's harness, unchanged (zf-whir-arms.sh, md5 86951bc2), with
# ZF_ALLOW_OTHER_KNOBS=1; this script unsets LAMBDA_VM_QUIET_PRODUCER first so an A arm reads the default. Then
# qp_readout.py (next to this script, md5 pinned) reads the arms from the harness's manifest.
# Output: /root/zf/qp-wt<N>/{qp.log,readout.txt}; the harness's run is /root/zf/wt<N>-wt<N+3>/.
# Exit: 0 green · 1 usage · 4 git/HEAD · 5 preflight · 9 card not idle · 11 harness red.
set -euo pipefail

EXPECT_HEAD=6e0b5f7310c7e144f52ceefe7f233ca4cf354304
BRANCH=fix2/1010-quiet-producer
HARNESS=/root/zf/bin/zf-whir-arms.sh
HARNESS_MD5=86951bc260eafb8461055611b501665c
PREFIX=wt
[ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]] || { sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 1; }
N0="$1"
SHA9=${EXPECT_HEAD:0:9}
SRC=/workspace/lambda_vm
SUMMARY_MD5=a0228cf0dcf71b26cf77a658d830382b
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READOUT="$HERE/qp_readout.py"
READOUT_MD5=2fa76515888a838addb2d8592d724dea
ARMS=(A:- B:LAMBDA_VM_QUIET_PRODUCER=1 B:LAMBDA_VM_QUIET_PRODUCER=1 A:-)
LAST=$((N0 + ${#ARMS[@]} - 1))
RUN="/root/zf/${PREFIX}${N0}-${PREFIX}${LAST}"
OUT="/root/zf/qp-wt${N0}"
[ ! -e "$OUT" ] || { echo "VERDICT: REFUSED — $OUT exists (tag used)"; exit 1; }
mkdir -p "$OUT"
LOG="$OUT/qp.log"
say() { echo "QP $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
refuse() { local rc=$1; shift; say "REFUSE($rc): $*"; echo "VERDICT: REFUSED rc=$rc — $*" | tee -a "$LOG"; exit "$rc"; }

say "QUIET-PRODUCER A/B · expect $EXPECT_HEAD ($BRANCH) · arms ${ARMS[*]} · tags $PREFIX$N0..$PREFIX$LAST"
used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | { grep -c . || true; })"
say "card: ${used} MiB used, ${apps} compute app(s)"
[ "${used:-99999}" -lt 500 ] && [ "${apps:-1}" -eq 0 ] || refuse 9 "card not idle (${used} MiB, ${apps} apps)"

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
say "HEAD asserted: $full"

unset LAMBDA_VM_QUIET_PRODUCER
say "harness: ZF_BRANCH=$BRANCH ZF_ALLOW_OTHER_KNOBS=1 bash $HARNESS $SHA9 $N0 ${ARMS[*]}"
set +e
ZF_BRANCH="$BRANCH" ZF_ALLOW_OTHER_KNOBS=1 bash "$HARNESS" "$SHA9" "$N0" "${ARMS[@]}" 2>&1 | tee -a "$LOG"
HRC=${PIPESTATUS[0]}
set -e
say "harness returned $HRC"

READ_LINE="READOUT: not run (no manifest)"
if [ -r "$RUN/manifest.tsv" ]; then
  set +e
  python3 "$READOUT" "$RUN/manifest.tsv" > "$OUT/readout.txt" 2>&1
  set -e
  tee -a "$LOG" < "$OUT/readout.txt"
  READ_LINE="$(grep '^READOUT: ' "$OUT/readout.txt" | tail -1 || true)"
  [ -n "$READ_LINE" ] || READ_LINE="READOUT: no verdict line (see $OUT/readout.txt)"
fi
if [ "$HRC" -ne 0 ]; then
  echo "VERDICT: HARNESS RED rc=$HRC · ${READ_LINE#READOUT: }" | tee -a "$LOG"
  exit 11
fi
echo "VERDICT: ${READ_LINE#READOUT: }" | tee -a "$LOG"
