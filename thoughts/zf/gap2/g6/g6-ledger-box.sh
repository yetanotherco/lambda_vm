#!/usr/bin/env bash
# g6-ledger-box.sh — lane G6 (SOTA §d Q5 and Q2): the #1010 base's ledger at 9e2728955, and its padding census, on FAST.
#
#   bash g6-ledger-box.sh            # the job: 4 arms, the trace, its export, the analysis, the secrets scan
#   DRY=1 bash g6-ledger-box.sh      # preflight, git, the generated launcher copies, 4 harness --dry-runs, the sampler's
#                                    # selftest, the analysis on the 09-25 trace wt90nsys, a log-only pass over the newest
#                                    # head-ahead tree log, the census on a synthetic log (CPU only: no card, no cargo)
#   G6_N0=<n>                        # the first tag number (default 1200): tags wt<n> .. wt<n+3>
#
# Four arms, in this order, each one call of the deployed harness /root/zf/bin/zf-whir-arms.sh (86951bc2), one tag each:
#   P1 wt<n>    #1010 whir-recursion-rpx @ 9e2728955, no knob                  the wall (nothing runs beside it)
#   N  wt<n+1>  the same, the launcher's `cargo test` under nsys, + the host sampler                 the trace
#   P2 wt<n+2>  the same as P1, + the host sampler                  the sampler's cost; host contention without nsys
#   C  wt<n+3>  fix2/1010-g6-census @ fd6da62a2 (9e2728955 + log lines behind LAMBDA_VM_ROW_CENSUS), knob
#               LAMBDA_VM_ROW_CENSUS=1, + the host sampler                               the padding census (Q2)
# N goes through generated copies of the record chain in which exactly one line changes (G1's method, asserted):
#   A-tree-whir.v10.sh (1 line: `cargo test` -> `nsys profile -t cuda,osrt --sample=none --cpuctxsw=none -o <rep> -f
#   false cargo test`) <- whir_tree17.sh (2 lines: its path) <- zf-whir-arms.sh (5 lines: HERE, DRIVER, DRIVER_MD5,
#   V10, V10_MD5).
# The sampler (g6_sampler.py) reads /proc/<pid>/task/*/schedstat of the prover test binary (matched on argv[0], or on
# its executable when argv[0] is relative; never on the command line), /proc/stat and the cgroup's cpu.stat every
# 0.05 s; it writes numbers and thread names only; it runs beside N, P2 and C, never beside P1; it is killed by PID.
# Then: nsys export to sqlite, `nsys stats` CSVs, g6_trace.py on N (trace + log + sampler) and --log-only on P1, P2 and
# C, g6_census.py on C, zf_summary.py over the four logs (walls, identities), text copies, a secrets scan of everything
# under $OUT, a tarball. The report and its sqlite stay in /root/prof/nsys and the sampler files in $G6/sampler; only
# $OUT (text and TSV) is for copying off.
#
# Exit: 0 done (C may have failed: the VERDICT says so) · 3 a lock is held · 4 git/HEAD · 5 preflight (a used tag too) ·
# 9 card not idle, a prover running, or too little disk · 11 a P arm failed · 12 P1/N/P2 ran different binaries ·
# 13 the nsys arm left no report · 14 analysis failed · 15 the secrets scan hit · 16 the sampler died at start.
set -euo pipefail

EXPECT_HEAD=9e2728955f8d1f0b6e71f4e21d48d592aab06f5c
EXPECT_TREE=4799a00364d1e012fe5fb0cc4c06bef6e86388c9       # = 33232d688's tree (the #1010 number of record, 40.20 s)
HEAD_BRANCH=whir-recursion-rpx
CENSUS_HEAD=fd6da62a20f5c55d7b62471f5c6fddd2459d8154
CENSUS_BRANCH=fix2/1010-g6-census
CENSUS_SHORTSTAT=" 27 files changed, 243 insertions(+)"      # git diff --shortstat EXPECT_HEAD CENSUS_HEAD (log lines only)
SHA9=${EXPECT_HEAD:0:9}
CSHA9=${CENSUS_HEAD:0:9}
N0="${G6_N0:-1200}"
[[ "$N0" =~ ^[0-9]+$ ]] || { echo "VERDICT: G6 FAILED rc=5 — G6_N0 must be an integer, got '$N0'"; exit 5; }
G6=/root/zf/g6
BIN=/root/zf/bin
LAUNCH=$G6/launch
OUT=$G6/out
SAMP=$G6/sampler
NSYSDIR=/root/prof/nsys
SRC=/workspace/lambda_vm
WT=/workspace/lambda_vm-zf-whir
MATCH=$WT/target/release/deps/lambda_vm_prover-
DRY="${DRY:-0}"
NSYS_FLAGS="-t cuda,osrt --sample=none --cpuctxsw=none"
SECRETS='JUPYTER_TOKEN|CONTAINER_API_KEY|GITHUB_SSH_KEY_B64|OPEN_BUTTON_TOKEN|apikey|api_key|BEGIN [A-Z ]*PRIVATE KEY'
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH

# pinned inputs (md5)
ZF_SUMMARY_MD5=a0228cf0dcf71b26cf77a658d830382b
WHIR_ARMS_MD5=86951bc260eafb8461055611b501665c
V10_MD5=88dd158786704e98df5a726a29df0088
TREE17_MD5=d243d42e4035e5f954f0be519c4aed38
G6_TRACE_MD5=175f0450adf082bc0dac31b0f8829f97
G6_SAMPLER_MD5=56eff730175a96591c7546c62fd33caf
G6_CENSUS_MD5=883228a366f97c1f540b92c078a2ffc2
# Disk this job needs on /root [E]: one nsys report + its sqlite, kept on FAST, <= 1.0 GB (G1's 61.7 s WHIR arm and the
# 09-25 107.8 s one: 0.43 GB of report); capture streams while recording <= 1.0 GB, transient; the census build in the
# harness worktree <= 1.0 GB (rebuilt in place); sampler files <= 0.1 GB; logs and out/ <= 0.1 GB.
NEED_GB=3
MIN_FREE_GB=$((NEED_GB + 5))

TP1="wt$N0"; TN="wt$((N0 + 1))"; TP2="wt$((N0 + 2))"; TC="wt$((N0 + 3))"
TAGS=("$TP1" "$TN" "$TP2" "$TC")
REP=$NSYSDIR/g6-$TN

log() { echo "G6 $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
free_gb() { df -BG --output=avail /root | tail -1 | tr -dc 0-9; }
VERDICT_DONE=""
die() { local rc=$1; shift; log "REFUSE($rc): $*"; VERDICT_DONE=1; echo "VERDICT: G6 FAILED rc=$rc — $*"; exit "$rc"; }
md5of() { md5sum "$1" | cut -c1-32; }

mkdir -p "$G6" "$LAUNCH" "$SAMP" "$NSYSDIR"
exec 7>"$G6/.g6.lock"
flock -n 7 || die 3 "$G6/.g6.lock is held — this job is already running"
SUFFIX=""
if [ "$DRY" = 1 ]; then SUFFIX="-dry"; fi
LOGF="$G6/driver-$(date -u +%Y%m%dT%H%M%SZ)$SUFFIX.log"
# The last word must reach the job's stdout even if tee is gone, and tee must see EOF at exit: close EVERY fd that may
# hold its pipe (bash parks a copy of fd 1 at 10+ inside a redirected compound command), then wait for it (job 220).
exec 9>&1
exec > >(tee -a "$LOGF") 2>&1
TEE_PID=$!
SPID=""
finish() {
  local rc=$? cmd=$BASH_COMMAND p n
  if [ -n "$SPID" ]; then kill -TERM "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; fi
  exec 1>&- 2>&-
  for p in /dev/fd/*; do n=${p##*/}; case "$n" in 0|1|2|9|255|*[!0-9]*) continue ;; esac; eval "exec $n>&-"; done
  wait "$TEE_PID" 2>/dev/null || true
  [ "$rc" = 0 ] || [ -n "$VERDICT_DONE" ] || echo "VERDICT: G6 FAILED rc=$rc — unexpected exit at: $cmd" | tee -a "$LOGF" >&9
}
trap finish EXIT
log "G6 ledger job · DRY=$DRY · #1010 $SHA9 ($TP1 P1, $TN N, $TP2 P2) · census $CSHA9 ($TC C) · this script md5 $(md5of "$0")"

# ------------------------------------------------------------------ preflight
PRE=0
chk() { if eval "$2"; then log "  ok    $1"; else log "  FAIL  $1"; PRE=1; fi; }
log "preflight:"
chk "g6_trace.py md5 $G6_TRACE_MD5" "[ \"\$(md5of $G6/g6_trace.py)\" = $G6_TRACE_MD5 ]"
chk "g6_sampler.py md5 $G6_SAMPLER_MD5" "[ \"\$(md5of $G6/g6_sampler.py)\" = $G6_SAMPLER_MD5 ]"
chk "g6_census.py md5 $G6_CENSUS_MD5" "[ \"\$(md5of $G6/g6_census.py)\" = $G6_CENSUS_MD5 ]"
chk "zf_summary.py md5 $ZF_SUMMARY_MD5 (harness a0228cf0)" "[ \"\$(md5of $BIN/zf_summary.py)\" = $ZF_SUMMARY_MD5 ]"
chk "zf-whir-arms.sh md5 $WHIR_ARMS_MD5" "[ \"\$(md5of $BIN/zf-whir-arms.sh)\" = $WHIR_ARMS_MD5 ]"
chk "A-tree-whir.v10.sh md5 $V10_MD5" "[ \"\$(md5of /root/A-tree-whir.v10.sh)\" = $V10_MD5 ]"
chk "whir_tree17.sh md5 $TREE17_MD5" "[ \"\$(md5of /root/whir_tree17.sh)\" = $TREE17_MD5 ]"
chk "nsys present ($(nsys --version 2>/dev/null | tail -1))" "command -v nsys >/dev/null"
chk "python3 with numpy and sqlite3" "python3 -c 'import numpy, sqlite3' 2>/dev/null"
for t in "${TAGS[@]}"; do
  chk "tag $t unused" "[ ! -e /root/prof/$t-tree.log ] && [ ! -e /root/zf/$t ] && [ ! -e /root/zf/$t-$t ] && [ ! -e $SAMP/$t.tsv ]"
done
chk "no nsys report or sqlite at $REP" "[ ! -e $REP.nsys-rep ] && [ ! -e $REP.sqlite ]"
[ "$DRY" = 1 ] || chk "$OUT is absent or empty (no earlier run's outputs to mix in)" "[ -z \"\$(ls -A $OUT 2>/dev/null)\" ]"
[ "$PRE" = 0 ] || die 5 "preflight red (see FAIL lines)"

card_idle() {
  local used apps
  used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)"
  log "card: ${used} MiB used, ${apps} compute app(s)"
  [ "${used:-99999}" -lt 500 ] && [ "${apps:-1}" -eq 0 ]
}
prover_running() {  # argv[0]-anchored: a prover test binary from any /workspace/lambda_vm* target; PIDs only
  ps -eww -o pid=,args= | awk '$2 ~ /^\/workspace\/lambda_vm[-a-z0-9]*\/target\/release\/deps\/lambda_vm_prover-/ {print $1}'
}
disk_ok() {  # <when>
  local f; f="$(free_gb || true)"
  log "disk: ${f:-?} GB free on /root ($1; need ${NEED_GB} GB + 5 GB margin = ${MIN_FREE_GB} GB)"
  [ -n "$f" ] && [ "$f" -ge "$MIN_FREE_GB" ] || die 9 "only ${f:-?} GB free on /root, below ${MIN_FREE_GB} GB ($1)"
}
disk_ok "preflight"
if [ "$DRY" != 1 ]; then
  card_idle || die 9 "the card is not idle (used >= 500 MiB or a compute app is running)"
  [ -z "$(prover_running)" ] || die 9 "a lambda_vm_prover test binary is running"
  flock -n /root/.box.lock true || die 3 "/root/.box.lock is held — another run owns the box"
  flock -n /root/zf/.zf.lock true || die 3 "/root/zf/.zf.lock is held — another ZF job is running"
fi

# ------------------------------------------------------------------ git: both commits are what we expect
log "git fetch origin $HEAD_BRANCH $CENSUS_BRANCH"
git -C "$SRC" fetch -q origin "$HEAD_BRANCH" "$CENSUS_BRANCH" || die 4 "git fetch failed (is $CENSUS_BRANCH pushed?)"
for pair in "$SHA9:$EXPECT_HEAD:$HEAD_BRANCH" "$CSHA9:$CENSUS_HEAD:$CENSUS_BRANCH"; do
  s9=${pair%%:*}; rest=${pair#*:}; full=${rest%%:*}; br=${rest#*:}
  got="$(git -C "$SRC" rev-parse --verify -q "${s9}^{commit}")" || die 4 "$s9 is not a commit in $SRC"
  [ "$got" = "$full" ] || die 4 "$s9 resolves to $got, expected $full"
  tip="$(git -C "$SRC" rev-parse -q --verify "origin/$br" || echo '<no remote-tracking ref>')"
  if [ "$tip" = "$full" ]; then log "  origin/$br = $full (EXPECT_HEAD)"; else log "  ⓘ origin/$br is now $tip; the job measures $full as pinned"; fi
done
[ "$(git -C "$SRC" rev-parse "${EXPECT_HEAD}^{tree}")" = "$EXPECT_TREE" ] || die 4 "$SHA9's tree is not $EXPECT_TREE (33232d688's)"
[ "$(git -C "$SRC" rev-parse "${CENSUS_HEAD}^")" = "$EXPECT_HEAD" ] || die 4 "$CSHA9's parent is not $SHA9"
[ "$(git -C "$SRC" diff --shortstat "$EXPECT_HEAD" "$CENSUS_HEAD")" = "$CENSUS_SHORTSTAT" ] \
  || die 4 "$SHA9..$CSHA9 is not the census diff ($(git -C "$SRC" diff --shortstat "$EXPECT_HEAD" "$CENSUS_HEAD"))"
log "  $SHA9 tree = $EXPECT_TREE · $CSHA9 = $SHA9 +${CENSUS_SHORTSTAT}"

# ------------------------------------------------------------------ generated nsys copies (exact edits, asserted)
python3 - "$LAUNCH" "$BIN" "$NSYS_FLAGS" "$REP" <<'PY'
import hashlib, sys, os
launch, bindir, flags, rep = sys.argv[1:5]
CARGO = 'LOG="$(/usr/bin/time -v timeout 21600 cargo test --release -p lambda-vm-prover \\\n'
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
def edit(src, dst, pairs):
    text = open(src).read()
    for old, new, n in pairs:
        c = text.count(old)
        if c != n:
            sys.exit(f"EDIT REFUSED: {src}: {old!r} occurs {c} times, expected {n}")
        text = text.replace(old, new)
    open(dst, "w").write(text)
    os.chmod(dst, 0o755)
    a = open(src).read().splitlines(); b = text.splitlines()
    changed = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    print(f"GEN {dst} md5 {md5(dst)} · {changed} line(s) differ from {src} ({md5(src)})")
    return changed
L = lambda n: os.path.join(launch, n)
v10n = L("A-tree-whir.v10-nsys.sh")
c1 = edit("/root/A-tree-whir.v10.sh", v10n, [(CARGO, CARGO.replace("timeout 21600 cargo test",
         f"timeout 21600 nsys profile {flags} -o {rep} -f false cargo test"), 1)])
t17n = L("whir_tree17-nsys.sh")
c2 = edit("/root/whir_tree17.sh", t17n, [("/root/A-tree-whir.v10.sh", v10n, 2)])
c3 = edit(os.path.join(bindir, "zf-whir-arms.sh"), L("zf-whir-arms-nsys.sh"), [
    ('HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n', f"HERE={bindir}\n", 1),
    ("DRIVER=/root/whir_tree17.sh\n", f"DRIVER={t17n}\n", 1),
    ("V10=/root/A-tree-whir.v10.sh\n", f"V10={v10n}\n", 1),
    ("V10_MD5=88dd158786704e98df5a726a29df0088\n", f"V10_MD5={md5(v10n)}\n", 1),
    ("DRIVER_MD5=d243d42e4035e5f954f0be519c4aed38\n", f"DRIVER_MD5={md5(t17n)}\n", 1)])
if [c1, c2, c3] != [1, 2, 5]:
    sys.exit(f"EDIT REFUSED: changed-line counts {[c1, c2, c3]} != [1, 2, 5]")
print("GEN OK: every copy differs from its original in exactly the asserted lines")
PY
for f in "$LAUNCH"/*.sh; do bash -n "$f" || die 5 "generated $f does not parse"; done
{ grep -n "nsys profile" "$LAUNCH/A-tree-whir.v10-nsys.sh" || true; } | cut -c1-240

# ------------------------------------------------------------------ DRY
if [ "$DRY" = 1 ]; then
  DRYRED=0
  DRYD=$G6/dry
  rm -rf "$DRYD"; mkdir -p "$DRYD"
  dry_arm() {  # <label> <branch> <script> <sha9> <n> <arm> <allow other knobs 0|1>
    local out rc
    set +e
    out="$(ZF_ALLOW_OTHER_KNOBS=$7 ZF_BRANCH=$2 bash "$3" --dry-run "$4" "$5" "$6" 2>&1)"; rc=$?
    set -e
    printf '%s\n' "$out" | tail -3
    log "  dry-run $1 via $(basename "$3") rc=$rc"
    [ "$rc" = 0 ] || DRYRED=1
  }
  dry_arm P1 "$HEAD_BRANCH" "$BIN/zf-whir-arms.sh" "$SHA9" "$N0" P1:- 0
  dry_arm N "$HEAD_BRANCH" "$LAUNCH/zf-whir-arms-nsys.sh" "$SHA9" $((N0 + 1)) N:- 0
  dry_arm P2 "$HEAD_BRANCH" "$BIN/zf-whir-arms.sh" "$SHA9" $((N0 + 2)) P2:- 0
  dry_arm C "$CENSUS_BRANCH" "$BIN/zf-whir-arms.sh" "$CSHA9" $((N0 + 3)) C:LAMBDA_VM_ROW_CENSUS=1 1
  if python3 "$G6/g6_sampler.py" --selftest > "$DRYD/sampler-selftest.out" 2>&1; then log "  $(cat "$DRYD/sampler-selftest.out")"
  else log "  ✗ $(cat "$DRYD/sampler-selftest.out")"; DRYRED=1; fi
  if [ -r "$NSYSDIR/wt90nsys.sqlite" ] && [ -r /root/prof/wt90nsys-tree.log ]; then
    log "analysis smoke test on the 09-25 trace (CPU only)"
    python3 "$G6/g6_trace.py" --db "$NSYSDIR/wt90nsys.sqlite" --log /root/prof/wt90nsys-tree.log --out "$DRYD/wt90" \
      > "$DRYD/wt90.out" 2>&1 || die 14 "the analysis failed on wt90nsys (see $DRYD/wt90.out)"
    { grep -E "^(base|level0|interior|root) \||^closure|^clock|^prover thread" "$DRYD/wt90.out" || true; } | sed 's/^/    /'
    # the laptop's run of this script on its copy of the trace (= G1's g1_trace.py on it): busy / idle per window (s)
    python3 - "$DRYD/wt90.out" <<'PY'
import sys
want = {"base": (46.162, 20.238), "level0": (17.513, 6.587), "interior": (10.995, 3.405), "root": (1.602, 1.298)}
got = {}
for l in open(sys.argv[1]):
    c = [x.strip() for x in l.split("|")]
    if c and c[0] in want and len(c) >= 13:
        got[c[0]] = (float(c[11]), float(c[10]))
bad = [k for k, (b, i) in want.items() if k not in got or abs(got[k][0] - b) > 0.05 or abs(got[k][1] - i) > 0.05]
print("    G6 dry: the 09-25 partition vs the laptop's run (busy, idle within 0.05 s): " + ("MATCH" if not bad else f"DIFF in {bad}: {got}"))
PY
  else
    log "  ⓘ no wt90nsys trace on this box: the trace path is not smoke-tested"
  fi
  RECENT=""
  for t in wt1061 wt1062 wt1071 wt1072 wt1031; do [ -r "/root/prof/$t-tree.log" ] && { RECENT=/root/prof/$t-tree.log; break; }; done
  if [ -n "$RECENT" ]; then
    python3 "$G6/g6_trace.py" --log-only --log "$RECENT" --out "$DRYD/logonly" > "$DRYD/logonly.out" 2>&1 \
      || die 14 "the log-only analysis failed on $RECENT (see $DRYD/logonly.out)"
    if grep -q "PARTITION OK" "$DRYD/logonly.out"; then log "  log-only pass on $(basename "$RECENT"): PARTITION OK"
    else log "  ✗ log-only pass on $(basename "$RECENT"): the sub-phases do not partition the base"; DRYRED=1; fi
    { grep -E "^windows:|^  head |producer," "$DRYD/logonly.out" || true; } | cut -c1-240 | sed 's/^/    /'
  else
    log "  ⓘ no head-ahead tree log (wt1061/1062/1071/1072/1031) on this box: the log-only path is not smoke-tested"
  fi
  cat > "$DRYD/census-synth.log" <<'EOF'
BASE ROWS #0: CPU 4096/4096 n1 · DECODE 3000/4096 n1 · L2G 100/128 n1 · LT 100/128 n1
BASE SHAPES #0: BITWISE w21 m20 · DECODE w13 m12 · CPU[0] w100 m12 · LT[0] w20 m7 · unknown w9 m7
BASE STACK #0: tables 0..4 cells 22485504 polys 1 vars 25 · tables 4..5 cells 1152 polys 1 vars 11
ARGUE TABLE #0 0 BITWISE: vars 20 · cols 21 · argue 0.0500s
ARGUE TABLE #0 1 DECODE: vars 12 · cols 13 · argue 0.0100s
ARGUE TABLE #0 2 CPU[0]: vars 12 · cols 100 · argue 0.2000s
ARGUE TABLE #0 3 LT[0]: vars 7 · cols 20 · argue 0.0100s
ARGUE TABLE #0 4 unknown: vars 7 · cols 9 · argue 0.0050s
WHIR PROVE SPLIT #0: airs 5 · wall 1.00s || inside[Σ] challenge 0.000 · argue 0.275 (max CPU[0] 0.20) · open_groups 0.20 (2 groups) · open_prepared 0.00
BASE ROWS global: GLOBAL_MEMORY 300/512 n2 · L2G 100/128 n1
BASE SHAPES global: unknown w9 m7 · unknown w6 m8 · unknown w6 m8
BASE STACK global: tables 0..1 cells 1152 polys 1 vars 11 · tables 1..3 cells 3072 polys 1 vars 12
ARGUE TABLE global 0 unknown: vars 7 · cols 9 · argue 0.0050s
ARGUE TABLE global 1 unknown: vars 8 · cols 6 · argue 0.0040s
ARGUE TABLE global 2 unknown: vars 8 · cols 6 · argue 0.0040s
WHIR PROVE SPLIT GLOBAL (in base): airs 3 · wall 0.30s || inside[Σ] challenge 0.000 · argue 0.013 (max unknown 0.005) · open_groups 0.10 (2 groups) · open_prepared 0.00
EOF
  if python3 "$G6/g6_census.py" "$DRYD/census-synth.log" --out "$DRYD/census" > "$DRYD/census.out" 2>&1 \
     && grep -q "consistency: 0 finding" "$DRYD/census.out"; then log "  census on the synthetic log: 0 findings"
  else log "  ✗ census on the synthetic log (see $DRYD/census.out)"; DRYRED=1; fi
  hits=$({ grep -rIliE "$SECRETS" "$DRYD" || true; } | wc -l | tr -d " ")
  [ "$hits" = 0 ] || die 15 "secrets scan hit in the dry outputs"
  [ "$DRYRED" = 0 ] || die 5 "a dry step was red (see ✗ and rc lines above)"
  VERDICT_DONE=1
  echo "VERDICT: G6 DRY-RUN OK — preflight and git green, 3 launcher copies generated and parsed, 4 harness dry-runs green, sampler selftest PASS, census synthetic 0 findings"
  exit 0
fi

# ------------------------------------------------------------------ arms
mkdir -p "$OUT"
start_sampler() {  # <tag>
  python3 "$G6/g6_sampler.py" --out "$SAMP/$1.tsv" --match "$MATCH" --interval 0.05 > "$SAMP/$1.err" 2>&1 < /dev/null &
  SPID=$!
  sleep 0.5
  kill -0 "$SPID" 2>/dev/null || die 16 "the sampler for $1 died at start (see $SAMP/$1.err)"
  log "  sampler pid $SPID -> $SAMP/$1.tsv"
}
stop_sampler() {  # <tag>
  local p=$SPID n
  [ -n "$p" ] || return 0
  kill -TERM "$p" 2>/dev/null || true
  wait "$p" 2>/dev/null || true
  SPID=""
  n=$({ grep -c '^T' "$SAMP/$1.tsv" || true; })
  log "  sampler $p stopped: ${n:-0} thread samples in $SAMP/$1.tsv"
}
declare -A RC=()
run_arm() {  # <tag> <sampler 0|1> <script> <branch> <sha9> <n> <arm> <allow other knobs 0|1>
  local tag=$1 samp=$2 script=$3 br=$4 sha9=$5 n=$6 arm=$7 allow=$8 rc
  log "===== $tag ($arm) via $(basename "$script") @ $sha9 · sampler $([ "$samp" = 1 ] && echo on || echo off)"
  disk_ok "before $tag"
  card_idle || die 9 "the card is not idle before $tag"
  [ -z "$(prover_running)" ] || die 9 "a prover is running before $tag"
  if [ "$samp" = 1 ]; then start_sampler "$tag"; fi
  set +e
  ZF_ALLOW_OTHER_KNOBS=$allow ZF_BRANCH=$br bash "$script" "$sha9" "$n" "$arm" > "$G6/$tag-launch.out" 2>&1
  rc=$?
  set -e
  if [ "$samp" = 1 ]; then stop_sampler "$tag"; fi
  log "  harness rc=$rc · $({ grep -m1 '^VERDICT' "$G6/$tag-launch.out" || echo 'no VERDICT line'; } | cut -c1-220)"
  RC["$tag"]=$rc
}
head_is() { [ "$(git -C "$WT" rev-parse HEAD)" = "$1" ] || die 4 "HEAD of $WT is $(git -C "$WT" rev-parse HEAD), expected $1"; log "  HEAD $WT = $1"; }
bin_of() { { grep -hoE 'prover binary lambda_vm_prover-[0-9a-f]+ md5 [0-9a-f]{32}' "/root/zf/$1-$1/driver.log" 2>/dev/null || true; } | tail -1 | awk '{print $NF}'; }

run_arm "$TP1" 0 "$BIN/zf-whir-arms.sh" "$HEAD_BRANCH" "$SHA9" "$N0" P1:- 0
[ "${RC[$TP1]}" = 0 ] || die 11 "plain arm $TP1 failed (see $G6/$TP1-launch.out)"
head_is "$EXPECT_HEAD"
run_arm "$TN" 1 "$LAUNCH/zf-whir-arms-nsys.sh" "$HEAD_BRANCH" "$SHA9" $((N0 + 1)) N:- 0
head_is "$EXPECT_HEAD"
[ -s "$REP.nsys-rep" ] || die 13 "the nsys arm left no report at $REP.nsys-rep (harness rc ${RC[$TN]})"
run_arm "$TP2" 1 "$BIN/zf-whir-arms.sh" "$HEAD_BRANCH" "$SHA9" $((N0 + 2)) P2:- 0
[ "${RC[$TP2]}" = 0 ] || die 11 "plain arm $TP2 failed (see $G6/$TP2-launch.out)"
head_is "$EXPECT_HEAD"
B1=$(bin_of "$TP1"); BN=$(bin_of "$TN"); B2=$(bin_of "$TP2")
log "  prover binaries: $TP1 ${B1:-none} · $TN ${BN:-none} · $TP2 ${B2:-none}"
[ -n "$B1" ] && [ "$B1" = "$BN" ] && [ "$BN" = "$B2" ] || die 12 "P1/N/P2 ran different prover binaries: ${B1:-none} / ${BN:-none} / ${B2:-none}"
run_arm "$TC" 1 "$BIN/zf-whir-arms.sh" "$CENSUS_BRANCH" "$CSHA9" $((N0 + 3)) C:LAMBDA_VM_ROW_CENSUS=1 1
C_OK=1
if [ "${RC[$TC]}" != 0 ]; then C_OK=0; log "  ✗ census arm $TC failed (harness rc ${RC[$TC]}); Q2 has no numbers from this job"
else
  head_is "$CENSUS_HEAD"
  NROWS=$({ grep -c '^BASE ROWS' "/root/prof/$TC-tree.log" || true; })
  NARG=$({ grep -cE '^ARGUE TABLE (#[0-9]+|global) ' "/root/prof/$TC-tree.log" || true; })   # not the default `ARGUE TABLES #k:` lines
  log "  census lines in $TC: BASE ROWS ${NROWS:-0} · ARGUE TABLE ${NARG:-0}"
  if [ "${NROWS:-0}" -lt 1 ] || [ "${NARG:-0}" -lt 1 ]; then C_OK=0; log "  ✗ the census arm printed no census lines (knob not read?)"; fi
fi
BC=$(bin_of "$TC")
log "  census binary ${BC:-none}"

# ------------------------------------------------------------------ export, stats, analysis (CPU only)
disk_ok "before the export"
mkdir -p "$OUT/$TN" "$OUT/$TP1" "$OUT/$TP2" "$OUT/$TC"
log "nsys export of $REP"
nsys export --type sqlite --output "$REP.sqlite" "$REP.nsys-rep" > "$OUT/$TN/export.out" 2>&1 \
  || die 14 "nsys export failed for $REP (see $OUT/$TN/export.out)"
for r in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum cuda_api_sum osrt_sum; do
  nsys stats --report "$r" --format csv --output "$OUT/$TN/nsys" "$REP.sqlite" > "$OUT/$TN/stats-$r.out" 2>&1 \
    || log "  ⚠ nsys stats $r failed (see $OUT/$TN/stats-$r.out)"
done
log "g6_trace.py on $TN (trace + log + sampler)"
python3 "$G6/g6_trace.py" --db "$REP.sqlite" --log "/root/prof/$TN-tree.log" --sampler "$SAMP/$TN.tsv" --out "$OUT/$TN/g6" \
  > "$OUT/$TN/g6.out" 2>&1 || die 14 "g6_trace.py failed on $TN (see $OUT/$TN/g6.out)"
{ grep -E "^(base|level1|root) \||^closure|^base sub-phases cover|^clock|^prover thread|name rule" "$OUT/$TN/g6.out" || true; } | cut -c1-240 | sed 's/^/    /'
log "g6_trace.py --log-only on $TP1 (no sampler), $TP2 and $TC (sampler)"
python3 "$G6/g6_trace.py" --log-only --log "/root/prof/$TP1-tree.log" --out "$OUT/$TP1/g6" > "$OUT/$TP1/g6.out" 2>&1 \
  || die 14 "g6_trace.py --log-only failed on $TP1 (see $OUT/$TP1/g6.out)"
python3 "$G6/g6_trace.py" --log-only --log "/root/prof/$TP2-tree.log" --sampler "$SAMP/$TP2.tsv" --out "$OUT/$TP2/g6" \
  > "$OUT/$TP2/g6.out" 2>&1 || die 14 "g6_trace.py --log-only failed on $TP2 (see $OUT/$TP2/g6.out)"
CENSUS_LINE="census: none (C failed)"
if [ "$C_OK" = 1 ]; then
  python3 "$G6/g6_trace.py" --log-only --log "/root/prof/$TC-tree.log" --sampler "$SAMP/$TC.tsv" --out "$OUT/$TC/g6" \
    > "$OUT/$TC/g6.out" 2>&1 || log "  ⚠ g6_trace.py --log-only failed on $TC (see $OUT/$TC/g6.out)"
  log "g6_census.py on $TC"
  if python3 "$G6/g6_census.py" "/root/prof/$TC-tree.log" --out "$OUT/$TC/census" > "$OUT/$TC/census.out" 2>&1; then
    { grep -E "^  \[|^  argue per-table|^consistency" "$OUT/$TC/census.out" || true; } | cut -c1-260 | sed 's/^/    /'
    CELLS=$({ grep -E '^  \[epochs\] committed cells' "$OUT/$TC/census.out" || true; } | sed -n 's/.*= \([0-9.]*\) % of the tables.*/\1/p' | tail -1)
    STACK=$({ grep -E '^  \[epochs\] stacked polynomials' "$OUT/$TC/census.out" || true; } | sed -n 's/.*(all padding \([0-9.]*\) %).*/\1/p' | tail -1)
    ARG=$({ grep -E '^  \[epochs\] argue:' "$OUT/$TC/census.out" || true; } | sed -n 's/.*s = \([0-9.]*\) %.*/\1/p' | tail -1)
    NF=$({ grep -E '^consistency:' "$OUT/$TC/census.out" || true; } | sed -n 's/^consistency: \([0-9]*\) finding.*/\1/p' | tail -1)
    CENSUS_LINE="census (epochs): row padding ${CELLS:-?} % of the tables' cells · all padding ${STACK:-?} % of the stack · argue [E] ${ARG:-?} % · ${NF:-?} consistency finding(s)"
  else
    log "  ⚠ g6_census.py failed on $TC (see $OUT/$TC/census.out)"; CENSUS_LINE="census: the analysis failed"
  fi
fi
log "zf_summary.py over the four logs (walls, identities)"
{
  printf '# tag\tname\tknobs\tlog\n'
  printf '%s\tP1\t-\t/root/prof/%s-tree.log\n' "$TP1" "$TP1"
  printf '%s\tN\tnsys\t/root/prof/%s-tree.log\n' "$TN" "$TN"
  printf '%s\tP2\t-\t/root/prof/%s-tree.log\n' "$TP2" "$TP2"
  if [ "$C_OK" = 1 ]; then printf '%s\tC\tLAMBDA_VM_ROW_CENSUS=1\t/root/prof/%s-tree.log\n' "$TC" "$TC"; fi
} > "$G6/manifest.tsv"
set +e
python3 "$BIN/zf_summary.py" --manifest "$G6/manifest.tsv" > "$OUT/ab.md" 2>&1; ABRC=$?
set -e
{ grep -E '^identities `|^A/B identity check|^SUMMARY rc' "$OUT/ab.md" || true; } | sed 's/^/    /'

# ------------------------------------------------------------------ text copies, the secrets scan, the tarball
mkdir -p "$OUT/logs"
for t in "${TAGS[@]}"; do
  if [ -e "/root/prof/$t-tree.log" ]; then cp "/root/prof/$t-tree.log" "$OUT/logs/"; fi
  if [ -e "/root/zf/$t/summary.md" ]; then cp "/root/zf/$t/summary.md" "$OUT/logs/$t-summary.md"; fi
  if [ -e "/root/zf/$t-$t/driver.log" ]; then cp "/root/zf/$t-$t/driver.log" "$OUT/logs/$t-driver.log"; fi
  if [ -e "$G6/$t-launch.out" ]; then cp "$G6/$t-launch.out" "$OUT/logs/"; fi
done
cp "$LAUNCH"/*.sh "$OUT/logs/"
cp "$G6/manifest.tsv" "$OUT/logs/"
log "secrets scan of $OUT"
# counted with wc, never `| grep -q` (an early exit SIGPIPEs the producer and pipefail turns a hit into a miss)
NBAD=$({ find "$OUT" \( -name '*.sqlite' -o -name '*.nsys-rep' -o -name '*.qdstrm' \) -print || echo find-failed; } | wc -l | tr -d ' ')
NRAW=$({ grep -rl '^# g6_sampler' "$OUT" || true; } | wc -l | tr -d ' ')
[ "$NBAD" = 0 ] && [ "$NRAW" = 0 ] || die 15 "$NBAD report/sqlite file(s) and $NRAW raw sampler file(s) are inside $OUT"
HITS=$(grep -rIliE "$SECRETS" "$OUT" || true)
[ -z "$HITS" ] || die 15 "secret markers found in: $HITS (nothing under $OUT may be copied off)"
tar -C "$G6" -czf "$G6/g6-out.tar.gz" out
log "  clean; $OUT packed as $G6/g6-out.tar.gz ($(du -h "$G6/g6-out.tar.gz" | cut -f1))"

wall_of() { { grep -hoE 'block wall \(WHOLE RUN\) \| [0-9.]+ s' "$OUT/logs/$1-summary.md" 2>/dev/null || true; } | { grep -oE '[0-9.]+' || true; } | tail -1; }
W1=$(wall_of "$TP1"); WN=$(wall_of "$TN"); W2=$(wall_of "$TP2"); WC=$(wall_of "$TC")
OVH=$(python3 -c "import sys; a=[float(x) for x in sys.argv[1:4]]; print(f'{a[1] - (a[0] + a[2]) / 2:+.2f}')" "${W1:-nan}" "${WN:-nan}" "${W2:-nan}" 2>/dev/null || echo NA)
IDLE=$({ grep -E '^base \|' "$OUT/$TN/g6.out" || true; } | awk -F'|' '{gsub(/ /, "", $11); gsub(/ /, "", $2); print $11 " s of " $2 " s"}' | tail -1)
IDS=$({ grep -E '^identities `' "$OUT/ab.md" || true; } | sed 's/^identities //; s/ (a format change.*)//' | tr '\n' ';' | sed 's/;$//')
VERDICT_DONE=1
echo "VERDICT: G6 LEDGER RUNS DONE — walls $TP1 P1 ${W1:-NA} · $TN N ${WN:-NA} · $TP2 P2 ${W2:-NA} · $TC C ${WC:-NA} s (rc ${RC[$TP1]}/${RC[$TN]}/${RC[$TP2]}/${RC[$TC]}) · N − mean(P1,P2) ${OVH} s · base card idle (traced) ${IDLE:-NA} · ${CENSUS_LINE} · binaries P ${B1} C ${BC:-none} · identities ${IDS:-NA} (zf_summary rc $ABRC) · secrets scan clean · $G6/g6-out.tar.gz"
