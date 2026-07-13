#!/usr/bin/env bash
# =============================================================================
# Disk spill vs continuations — 8 GB budget experiment runner.
#
# Runs on the Linux bench server (needs cgroup v2 + systemd-run + GNU
# /usr/bin/time). Rationale, gotchas and the decision criteria are in
# DISK_SPILL_VS_CONTINUATIONS_8GB_PLAN.md (§8 runbook, §9 overhead, §10 8 GB).
#
# It performs, per input, under a hard cgroup memory cap with swap disabled:
#   (1) monolithic, NO spill      -> expected to OOM (baseline for WHY)
#   (2) monolithic + disk spill   -> time, peak RSS, proof size, verify
#   (3) continuation sweep 18..20 -> time, peak RSS, bundle size, verify
# and writes raw /usr/bin/time logs + a RESULTS.md matrix to $OUTDIR.
#
# NOTE: `set -e` is deliberately OFF. OOM-killed runs are an EXPECTED result we
# must record, not a reason to abort the script.
# =============================================================================
set -uo pipefail

# ----------------------------- CONFIG ----------------------------------------
# Adjust these after the box relevamiento (paths / systemd-run form / TMPDIR).
REPO="${REPO:-$HOME/lambda_vm}"
ELF="${ELF:-$REPO/executor/program_artifacts/rust/ethrex.elf}"

# Inputs to sweep: "label:absolute_path". Hard case first, then a lighter one.
# These are committed in the repo (present after a fresh clone):
#   ethrex_10_transfers (6.88M cyc, hard) / ethrex_simple_tx (1.80M) / ethrex_empty_block (0.99M).
# ethrex_3_transfers (2.95M, the "medium") is NOT committed — scp it to the box and add a
# line here if you want it in the sweep.
INPUTS=(
  "10_transfers:$REPO/executor/tests/ethrex_10_transfers.bin"
  "simple_tx:$REPO/executor/tests/ethrex_simple_tx.bin"
)

CAP="${CAP:-7500M}"                 # 8 GB budget minus OS headroom (8 GB physical box)
EPOCHS="${EPOCHS:-18 19 20}"        # continuation epoch_size_log2 sweep (20 confirms OOM)
SPILL_TMPDIR="${SPILL_TMPDIR:-/var/tmp/spill}"   # MUST be disk-backed, NOT tmpfs
OUTDIR="${OUTDIR:-$HOME/spill_vs_cont_$(date +%Y%m%d_%H%M%S)}"
TIME_REPEATS="${TIME_REPEATS:-3}"  # wall-time repeats for fitting configs (RSS is ~deterministic)
DO_BUILD="${DO_BUILD:-1}"          # 1 = build the binaries first
# How to launch a memory-capped scope. Override to `sudo systemd-run --scope`
# or `systemd-run --user --scope` depending on what the box allows (preflight tests it).
SYSTEMD_RUN="${SYSTEMD_RUN:-systemd-run --scope --quiet}"
# -----------------------------------------------------------------------------

CLI=""      # resolved after build
BENCH=""    # resolved after build (used only for epochs < 18, if ever needed)

log()  { printf '\n\033[1;36m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[FATAL] %s\033[0m\n' "$*" >&2; exit 1; }

# Locate the compiled bench binary via glob (the file with no .d extension).
find_bench() {
  local f
  for f in "$REPO"/target/release/deps/bench_continuation-*; do
    [[ -f "$f" && "$f" != *.d ]] && { echo "$f"; return; }
  done
}

# ----------------------------- PREFLIGHT -------------------------------------
preflight() {
  log "Preflight checks"

  [[ "$(stat -fc %T /sys/fs/cgroup 2>/dev/null)" == "cgroup2fs" ]] \
    || die "cgroup v2 not found (stat -fc %T /sys/fs/cgroup != cgroup2fs). A hard cap needs it."

  command -v systemd-run >/dev/null || die "systemd-run not found."
  [[ -x /usr/bin/time ]] || die "/usr/bin/time (GNU) not found — install it (needed for -v peak RSS)."
  command -v cargo >/dev/null || die "cargo not found."

  [[ -f "$ELF" ]] || die "ELF not found: $ELF"
  local entry label path
  for entry in "${INPUTS[@]}"; do
    label="${entry%%:*}"; path="${entry#*:}"
    [[ -f "$path" ]] || die "input '$label' not found: $path"
  done

  # Swap should be off so an over-budget run OOMs instead of swapping and lying.
  if [[ -n "$(swapon --show --noheadings 2>/dev/null)" ]]; then
    warn "swap is ON. MemorySwapMax=0 in the scope disables it per-run, but confirm the box has no surprises."
  fi

  # SPILL_TMPDIR must live on real disk (tmpfs pages count against the cgroup cap).
  mkdir -p "$SPILL_TMPDIR" || die "cannot create SPILL_TMPDIR=$SPILL_TMPDIR"
  local fstype; fstype="$(df --output=fstype "$SPILL_TMPDIR" 2>/dev/null | tail -1 | tr -d ' ')"
  if [[ "$fstype" == "tmpfs" || "$fstype" == "ramfs" ]]; then
    die "SPILL_TMPDIR ($SPILL_TMPDIR) is $fstype (RAM-backed) — spill would OOM. Point it at a disk-backed path."
  fi
  log "SPILL_TMPDIR=$SPILL_TMPDIR is on '$fstype' (disk-backed: OK)"

  # Self-test the capped-scope launcher with a trivial command.
  if ! $SYSTEMD_RUN -p MemoryMax="$CAP" -p MemorySwapMax=0 true 2>/dev/null; then
    die "Cannot launch a capped scope with: $SYSTEMD_RUN -p MemoryMax=$CAP -p MemorySwapMax=0
     Try SYSTEMD_RUN='sudo systemd-run --scope --quiet' or 'systemd-run --user --scope --quiet'."
  fi

  mkdir -p "$OUTDIR" || die "cannot create OUTDIR=$OUTDIR"
  log "Preflight OK. Results -> $OUTDIR (cap=$CAP, epochs='$EPOCHS')"
}

# ------------------------------- BUILD ---------------------------------------
build() {
  log "Building cli + bench_continuation with --features disk-spill (release)"
  ( cd "$REPO" && cargo build --release --features disk-spill -p cli ) \
    || die "cli build failed"
  ( cd "$REPO" && cargo build --release --features disk-spill -p lambda-vm-prover --bench bench_continuation ) \
    || die "bench build failed"

  CLI="$REPO/target/release/cli"
  [[ -x "$CLI" ]] || die "cli binary not found at $CLI"
  BENCH="$(find_bench)"
  log "cli   = $CLI"
  log "bench = ${BENCH:-<not found>}"
}

# --------------------------- METRIC HELPERS ----------------------------------
# Parse a /usr/bin/time -v logfile -> prints "RSS_KB<TAB>WALL"
parse_time() {
  local f="$1"
  local rss wall
  rss="$(grep -m1 'Maximum resident set size' "$f" 2>/dev/null | grep -oE '[0-9]+' | tail -1)"
  wall="$(grep -m1 'Elapsed (wall clock)' "$f" 2>/dev/null | sed -E 's/.*: //')"
  # TAB-separated so `record()` writes two distinct TSV columns (rss_kb, wall).
  # `run_capped`'s unquoted `$metrics` word-splits on the tab too, so its display
  # (rss=%s wall=%s) still gets two args. A space here would collapse both into one
  # TSV field, shifting every column in RESULTS.md.
  printf '%s\t%s\n' "${rss:-NA}" "${wall:-NA}"
}

# run_capped <tag> <env-assignments> <cmd...>
# Runs the command inside a memory-capped scope under /usr/bin/time -v.
# Records the time log; returns the command's exit code (137/OOM tolerated).
run_capped() {
  local tag="$1"; shift
  local envassign="$1"; shift
  local tlog="$OUTDIR/$tag.time.log"
  local olog="$OUTDIR/$tag.out.log"
  log "RUN $tag  (cap=$CAP)"
  # shellcheck disable=SC2086
  $SYSTEMD_RUN -p MemoryMax="$CAP" -p MemorySwapMax=0 \
    /usr/bin/time -v -o "$tlog" env $envassign "$@" >"$olog" 2>&1
  local rc=$?
  local metrics; metrics="$(parse_time "$tlog")"
  if [[ $rc -eq 0 ]]; then
    printf '   -> OK   rss=%s KB  wall=%s\n' $metrics
  else
    printf '   -> exit=%s (OOM/kill expected for the no-spill baseline)  rss=%s KB  wall=%s\n' "$rc" $metrics
  fi
  return $rc
}

# Record a proof-file size and a verify result into the results TSV.
# record <input> <config> <proof_path|-> <time_tag> <verify_cmd...|->
record() {
  local input="$1" config="$2" proof="$3" tag="$4"; shift 4
  local size="NA" verify="NA"
  [[ "$proof" != "-" && -f "$proof" ]] && size="$(stat -c %s "$proof" 2>/dev/null)"
  if [[ "$#" -gt 0 && "$1" != "-" ]]; then
    if "$@" >"$OUTDIR/$config.verify.log" 2>&1; then verify="PASS"; else verify="FAIL"; fi
  fi
  local m; m="$(parse_time "$OUTDIR/$tag.time.log")"
  # TSV: input  config  rss_kb  wall  proof_bytes  verify
  printf '%s\t%s\t%s\t%s\t%s\n' "$input" "$config" "$m" "$size" "$verify" >>"$OUTDIR/results.tsv"
}

# best wall over N repeats for a fitting config (RSS taken from the last run)
run_repeated() {
  local tag="$1"; shift
  local envassign="$1"; shift
  local i
  for i in $(seq 1 "$TIME_REPEATS"); do
    run_capped "${tag}.r${i}" "$envassign" "$@"
  done
  # keep the last repeat's log as the canonical <tag> log (RSS is ~stable)
  cp -f "$OUTDIR/${tag}.r${TIME_REPEATS}.time.log" "$OUTDIR/${tag}.time.log" 2>/dev/null || true
  # stash the wall seconds across repeats for the report
  local s
  for i in $(seq 1 "$TIME_REPEATS"); do
    s="$(grep -m1 'Elapsed (wall clock)' "$OUTDIR/${tag}.r${i}.time.log" 2>/dev/null | sed -E 's/.*: //')"
    [[ -n "$s" ]] && echo "$s" >>"$OUTDIR/${tag}.walls.txt"
  done
}

# ------------------------------- RUNS ----------------------------------------
run_all() {
  printf 'input\tconfig\trss_kb\twall\tproof_bytes\tverify\n' >"$OUTDIR/results.tsv"
  local entry label path

  for entry in "${INPUTS[@]}"; do
    label="${entry%%:*}"; path="${entry#*:}"
    log "===== INPUT: $label ($path) ====="

    # (1) Monolithic, NO spill. FORCE unset => sysinfo sees host RAM (not the
    # cgroup cap) => picks Ram => OOMs under the cap. That IS the baseline.
    run_capped "${label}.mono_nospill" "" \
      "$CLI" prove "$ELF" -o "$OUTDIR/${label}.mono.proof" --private-input "$path"
    record "$label" "${label}.mono_nospill" "-" "${label}.mono_nospill" -

    # (2) Monolithic + disk spill.
    run_repeated "${label}.mono_spill" "FORCE_DISK_SPILL=1 TMPDIR=$SPILL_TMPDIR" \
      "$CLI" prove "$ELF" -o "$OUTDIR/${label}.spill.proof" --private-input "$path"
    record "$label" "${label}.mono_spill" "$OUTDIR/${label}.spill.proof" "${label}.mono_spill" \
      "$CLI" verify "$OUTDIR/${label}.spill.proof" "$ELF"

    # (3) Continuation sweep.
    local k
    for k in $EPOCHS; do
      run_repeated "${label}.cont_${k}" "" \
        "$CLI" prove "$ELF" -o "$OUTDIR/${label}.cont_${k}.proof" \
        --private-input "$path" --continuations --epoch-size-log2 "$k"
      record "$label" "${label}.cont_${k}" "$OUTDIR/${label}.cont_${k}.proof" "${label}.cont_${k}" \
        "$CLI" verify "$OUTDIR/${label}.cont_${k}.proof" "$ELF" --continuations
    done
  done
}

# ------------------------------ REPORT ---------------------------------------
report() {
  local md="$OUTDIR/RESULTS.md"
  {
    echo "# Disk spill vs continuations — results (8 GB budget)"
    echo
    echo "- Host: \`$(uname -sr)\` — $(nproc) cores"
    echo "- Cap: \`MemoryMax=$CAP\`, \`MemorySwapMax=0\`  (8 GB physical → ~8 GB budget minus OS headroom)"
    echo "- Repo commit: \`$(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null)\`"
    echo "- Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "| Input | Config | Peak RSS (KB) | Wall | Proof size (bytes) | Verify |"
    echo "|---|---|---|---|---|---|"
    tail -n +2 "$OUTDIR/results.tsv" | while IFS=$'\t' read -r input config rss wall size verify; do
      echo "| $input | $config | $rss | $wall | $size | $verify |"
    done
  } >"$md"
  log "Report written: $md"
  cat "$md"
}

# ------------------------------- MAIN ----------------------------------------
main() {
  preflight
  [[ "$DO_BUILD" == "1" ]] && build || { CLI="$REPO/target/release/cli"; BENCH="$(find_bench)"; }
  [[ -x "$CLI" ]] || die "cli binary not available (set DO_BUILD=1 or build manually)."
  run_all
  report
  log "DONE. Pull the whole dir off the server: scp -r admin@<host>:$OUTDIR ."
}

main "$@"
