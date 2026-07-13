#!/usr/bin/env bash
#
# bench_recursion_cycles.sh — deterministic recursion-guest cycle + accelerator
# comparison (PR vs baseline).
#
# The recursion guest is the in-VM STARK verifier: it runs the verifier INSIDE the
# VM. For a fixed (guest ELF, input blob) its cost is fully DETERMINISTIC, so
# comparing two git refs is two EXACT integer readings — no A/B/B/A interleaving.
# For each ref we report three numbers, all read from one `execute --cycles` run of
# a single measuring CLI (MEASURE_CLI) built once from this branch:
#   * Guest cycles  — retired instructions.
#   * Keccak calls  — keccak-permutation accelerator ecalls (one cycle each, but each
#                     runs a whole permutation invisibly, so it's the companion signal).
#   * Ecsm calls    — elliptic-curve scalar-mul accelerator ecalls (same idea).
# The executor semantics are main's, so MEASURE_CLI faithfully counts ANY ref's guest
# ELF — it just feeds the blob as private input and reads the counters.
#
# Improvement convention matches scripts/bench_verify.sh:
#   NEGATIVE Δ  =  REF_A (PR) does fewer cycles/calls  =  better.
#
# Usage: scripts/bench_recursion_cycles.sh REF_A [REF_B=origin/main] [PRESET=min]
#   REF_A    ref/SHA to evaluate (the PR side).
#   REF_B    baseline ref/SHA (default origin/main).
#   PRESET   recursion-verifier preset (default min). The tool prefers
#            recursion-<PRESET>.elf and falls back to recursion.elf on refs
#            (older/main) that build a single unnamed recursion guest.
#   Env:
#     REBUILD=1            force rebuild of MEASURE_CLI and re-run of every ref
#                          (guest build + blob dump + measurement); ignore caches.
#     SYSROOT_DIR=<path>   guest-build sysroot (default $HOME/.lambda-vm-sysroot).
#     GUEST_TARGET_DIR=<p> share the RV64 guest build dir across ref worktrees
#                          (reuses build-std → big speedup for the 2nd ref's guest
#                          build). Unset = per-worktree (default, fully isolated).
#     HOST_TARGET_DIR=<p>  share the host cargo target dir for the blob-dump test
#                          build across refs. Unset = per-worktree (default).
#
# Caching: each ref's result is cached in $WORK keyed on its resolved SHA + preset,
# so re-runs are fast. Ref worktrees are kept (named by SHA) so a re-measure is a
# cargo no-op. REBUILD=1 forces everything.
#
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bench_recursion_cycles.sh REF_A [REF_B=origin/main] [PRESET=min]" >&2
  echo "  REF_A: ref or SHA to evaluate (the PR side)" >&2
  echo "  REF_B: baseline ref (default origin/main)" >&2
  echo "  PRESET: recursion verifier preset (default min)" >&2
  exit 2
fi
REF_A="$1"
REF_B="${2:-origin/main}"
PRESET="${3:-min}"
SYSROOT_DIR="${SYSROOT_DIR:-$HOME/.lambda-vm-sysroot}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

WORK="/tmp/recursion_cycles_run"
mkdir -p "$WORK"

echo "==> Refs"
git fetch origin --quiet || echo "WARNING: 'git fetch origin' failed — resolving against possibly-stale local refs." >&2
SHA_A="$(git rev-parse "$REF_A")"
SHA_B="$(git rev-parse "$REF_B")"
echo "   A (PR)       $REF_A  -> ${SHA_A:0:10}"
echo "   B (baseline) $REF_B  -> ${SHA_B:0:10}"
echo "   preset=$PRESET  work=$WORK  sysroot=$SYSROOT_DIR"

if [ ! -d "$SYSROOT_DIR/lib" ]; then
  echo "ERROR: SYSROOT_DIR=$SYSROOT_DIR does not look provisioned (no lib/). Guest builds will fail." >&2
  echo "       Provision it or point SYSROOT_DIR at an existing sysroot." >&2
  exit 1
fi

# --- 1. Build MEASURE_CLI once (release) from the current branch ---------------
# This branch is based on feat/cycles-accelerator-counts, so `cli` here has
# `execute --cycles` with the keccak/ecsm counters. Its executor is main's, so it
# counts any ref's guest ELF correctly.
HEAD_SHA="$(git rev-parse HEAD)"
MEASURE_CLI="$WORK/measure_cli"
if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$MEASURE_CLI" ] || \
   [ "$(cat "$MEASURE_CLI.sha" 2>/dev/null)" != "$HEAD_SHA" ]; then
  echo "==> Building MEASURE_CLI (cli, release) from ${HEAD_SHA:0:10} ..."
  if ! cargo build --release -p cli >"$WORK/build_measure_cli.log" 2>&1; then
    echo "ERROR: MEASURE_CLI build failed. Tail of $WORK/build_measure_cli.log:" >&2
    tail -40 "$WORK/build_measure_cli.log" >&2
    exit 1
  fi
  cp "$ROOT/target/release/cli" "$MEASURE_CLI"
  echo "$HEAD_SHA" > "$MEASURE_CLI.sha"
else
  echo "==> Reusing cached MEASURE_CLI (${HEAD_SHA:0:10})"
fi

# --- 2. Per-ref: worktree + guest build + blob dump + measurement ---------------
# Prints progress to stderr; emits the parseable result block (key=value lines) to
# stdout so the caller can capture it.
measure_ref() {
  local ref="$1" sha="$2" role="$3"
  local sha8="${sha:0:8}"
  local result="$WORK/result_${sha8}_${PRESET}.txt"

  if [ "${REBUILD:-0}" != "1" ] && [ -f "$result" ]; then
    echo "==> [$role] Reusing cached measurement: $ref ($sha8) preset=$PRESET" >&2
    cat "$result"
    return 0
  fi

  local wt="$WORK/wt_${sha8}"
  if [ ! -d "$wt" ]; then
    echo "==> [$role] Adding worktree $wt @ $sha8" >&2
    git worktree prune
    git worktree add --detach "$wt" "$sha" >/dev/null
  else
    echo "==> [$role] Reusing worktree $wt (checkout -f $sha8)" >&2
    git -C "$wt" checkout --quiet -f "$sha"
  fi

  # 2a. Build the recursion guest ELF(s) (+ empty.elf inner program).
  echo "==> [$role] make compile-recursion-elfs @ $sha8 (this can take 10-20 min the first time) ..." >&2
  local glog="$WORK/build_guest_${sha8}.log"
  if [ -n "${GUEST_TARGET_DIR:-}" ]; then
    if ! ( cd "$wt" && SYSROOT_DIR="$SYSROOT_DIR" make compile-recursion-elfs "SHARED_TARGET_DIR=$GUEST_TARGET_DIR" ) >"$glog" 2>&1; then
      echo "ERROR: [$role] 'make compile-recursion-elfs' failed for $ref ($sha8). Tail of $glog:" >&2
      tail -40 "$glog" >&2
      exit 1
    fi
  else
    if ! ( cd "$wt" && SYSROOT_DIR="$SYSROOT_DIR" make compile-recursion-elfs ) >"$glog" 2>&1; then
      echo "ERROR: [$role] 'make compile-recursion-elfs' failed for $ref ($sha8). Tail of $glog:" >&2
      tail -40 "$glog" >&2
      exit 1
    fi
  fi

  # 2b. Detect the guest ELF: prefer recursion-<PRESET>.elf, else recursion.elf.
  local artdir="$wt/executor/program_artifacts/recursion"
  local guest_elf=""
  if [ -f "$artdir/recursion-${PRESET}.elf" ]; then
    guest_elf="$artdir/recursion-${PRESET}.elf"
  elif [ -f "$artdir/recursion.elf" ]; then
    guest_elf="$artdir/recursion.elf"
  else
    echo "ERROR: [$role] no recursion guest artifact for $ref ($sha8):" >&2
    echo "       neither recursion-${PRESET}.elf nor recursion.elf found in $artdir" >&2
    ls -la "$artdir" >&2 2>/dev/null || true
    exit 1
  fi
  echo "==> [$role] guest ELF: $(basename "$guest_elf")" >&2

  # 2c. Generate this ref's own input blob via its ignored dump test.
  if ! grep -rq "fn test_dump_recursion_input" "$wt/prover/src/tests/" 2>/dev/null; then
    echo "ERROR: [$role] ref $ref ($sha8) has no 'test_dump_recursion_input' — cannot generate its input blob." >&2
    exit 1
  fi
  echo "==> [$role] dumping recursion input blob (cargo test test_dump_recursion_input) ..." >&2
  rm -f /tmp/recursion_input.bin
  local dlog="$WORK/dump_${sha8}.log"
  if [ -n "${HOST_TARGET_DIR:-}" ]; then
    if ! ( cd "$wt" && CARGO_TARGET_DIR="$HOST_TARGET_DIR" cargo test -p lambda-vm-prover --lib test_dump_recursion_input -- --ignored --nocapture ) >"$dlog" 2>&1; then
      echo "ERROR: [$role] blob-dump test failed for $ref ($sha8). Tail of $dlog:" >&2
      tail -40 "$dlog" >&2
      exit 1
    fi
  else
    if ! ( cd "$wt" && cargo test -p lambda-vm-prover --lib test_dump_recursion_input -- --ignored --nocapture ) >"$dlog" 2>&1; then
      echo "ERROR: [$role] blob-dump test failed for $ref ($sha8). Tail of $dlog:" >&2
      tail -40 "$dlog" >&2
      exit 1
    fi
  fi
  if [ ! -f /tmp/recursion_input.bin ]; then
    echo "ERROR: [$role] test_dump_recursion_input did not write /tmp/recursion_input.bin for $ref ($sha8)." >&2
    exit 1
  fi
  local blob="$WORK/blob_${sha8}_${PRESET}.bin"
  cp /tmp/recursion_input.bin "$blob"
  echo "==> [$role] blob: $(wc -c <"$blob" | tr -d '[:space:]') bytes -> $blob" >&2

  # 2d. Measure: one deterministic execute --cycles run. Time it (CI feasibility).
  echo "==> [$role] measuring: $MEASURE_CLI execute $(basename "$guest_elf") --private-input <blob> --cycles" >&2
  local t0 t1 dt out
  t0=$(date +%s)
  if ! out="$("$MEASURE_CLI" execute "$guest_elf" --private-input "$blob" --cycles 2>"$WORK/measure_${sha8}.err")"; then
    echo "ERROR: [$role] MEASURE_CLI execute failed for $ref ($sha8). Tail of stderr:" >&2
    tail -20 "$WORK/measure_${sha8}.err" >&2
    exit 1
  fi
  t1=$(date +%s); dt=$((t1 - t0))

  local cyc kec ecs
  cyc="$(printf '%s\n' "$out" | awk -F': ' '/^Cycles:/{print $2; exit}')"
  kec="$(printf '%s\n' "$out" | awk -F': ' '/^Keccak calls:/{print $2; exit}')"
  ecs="$(printf '%s\n' "$out" | awk -F': ' '/^Ecsm calls:/{print $2; exit}')"
  if [ -z "$cyc" ] || [ -z "$kec" ] || [ -z "$ecs" ]; then
    echo "ERROR: [$role] could not parse Cycles/Keccak/Ecsm from MEASURE_CLI output for $ref ($sha8):" >&2
    printf '%s\n' "$out" >&2
    exit 1
  fi
  echo "==> [$role] cycles=$cyc keccak=$kec ecsm=$ecs  (execute wall-time ${dt}s)" >&2

  {
    printf 'cycles=%s\n' "$cyc"
    printf 'keccak=%s\n' "$kec"
    printf 'ecsm=%s\n' "$ecs"
    printf 'wall=%s\n' "$dt"
    printf 'elf=%s\n' "$(basename "$guest_elf")"
  } > "$result"
  cat "$result"
}

# Baseline first, then PR (so a fresh GUEST_TARGET_DIR is warmed by the baseline).
RES_B="$(measure_ref "$REF_B" "$SHA_B" baseline)"
RES_A="$(measure_ref "$REF_A" "$SHA_A" PR)"

getv() { printf '%s\n' "$1" | awk -F= -v k="$2" '$1==k{print $2; exit}'; }
CYC_B="$(getv "$RES_B" cycles)"; KEC_B="$(getv "$RES_B" keccak)"; ECS_B="$(getv "$RES_B" ecsm)"
WALL_B="$(getv "$RES_B" wall)"; ELF_B="$(getv "$RES_B" elf)"
CYC_A="$(getv "$RES_A" cycles)"; KEC_A="$(getv "$RES_A" keccak)"; ECS_A="$(getv "$RES_A" ecsm)"
WALL_A="$(getv "$RES_A" wall)"; ELF_A="$(getv "$RES_A" elf)"

# signed integer delta (A - B); 0 prints bare, >0 gets a leading '+'
sd() { local d=$(( $1 - $2 )); if [ "$d" -gt 0 ]; then printf '+%d' "$d"; else printf '%d' "$d"; fi; }
# signed integer delta + percentage of baseline
sdp() {
  local a="$1" b="$2"
  awk -v a="$a" -v b="$b" 'BEGIN{
    d=a-b;
    pct=(b!=0)? d/b*100 : 0;
    printf("%s%d (%s%.2f%%)", (d>=0?"+":""), d, (pct>=0?"+":""), pct);
  }'
}

echo
echo "=== Recursion-guest cycle/accelerator comparison (deterministic, exact) ==="
echo "   REF_B (baseline) $REF_B  ${SHA_B:0:10}  guest=$ELF_B"
echo "   REF_A (PR)       $REF_A  ${SHA_A:0:10}  guest=$ELF_A"
echo "   preset=$PRESET   convention: - = PR fewer = better"
echo
echo "| Metric        | REF_B (baseline) | REF_A (PR) | Δ (A-B) |"
echo "|---------------|------------------|------------|---------|"
printf '| Guest cycles  | %s | %s | %s |\n' "$CYC_B" "$CYC_A" "$(sdp "$CYC_A" "$CYC_B")"
printf '| Keccak calls  | %s | %s | %s |\n' "$KEC_B" "$KEC_A" "$(sd "$KEC_A" "$KEC_B")"
printf '| Ecsm calls    | %s | %s | %s |\n' "$ECS_B" "$ECS_A" "$(sd "$ECS_A" "$ECS_B")"
echo
echo "=== RAW (machine-parseable) ==="
printf 'ref_b_sha=%s ref_b_elf=%s ref_b_cycles=%s ref_b_keccak=%s ref_b_ecsm=%s ref_b_execute_wall_s=%s\n' \
  "$SHA_B" "$ELF_B" "$CYC_B" "$KEC_B" "$ECS_B" "$WALL_B"
printf 'ref_a_sha=%s ref_a_elf=%s ref_a_cycles=%s ref_a_keccak=%s ref_a_ecsm=%s ref_a_execute_wall_s=%s\n' \
  "$SHA_A" "$ELF_A" "$CYC_A" "$KEC_A" "$ECS_A" "$WALL_A"
printf 'delta_cycles=%s delta_keccak=%s delta_ecsm=%s\n' \
  "$(( CYC_A - CYC_B ))" "$(( KEC_A - KEC_B ))" "$(( ECS_A - ECS_B ))"
