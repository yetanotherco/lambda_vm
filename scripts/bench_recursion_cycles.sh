#!/usr/bin/env bash
#
# bench_recursion_cycles.sh — deterministic recursion-guest cycle + accelerator
# comparison (PR vs baseline).
#
# The recursion guest is the in-VM STARK verifier: it runs the verifier INSIDE the
# VM. For a fixed (guest ELF, input blob) its cost is fully DETERMINISTIC, so a single
# ref is one EXACT integer reading — no A/B/B/A interleaving needed; but across runs
# neither is held fixed (fresh guest build + a freshly-dumped nondeterministic proof
# blob), so expect ~±100k cycles run-to-run. Note the two refs
# do NOT share one blob: each ref dumps its OWN input blob from its own prover (via its
# ignored dump test). So when a PR only changes guest code the delta is a clean
# guest-cycle diff, but when a PR changes the prover / proof format the delta conflates
# the guest-code change with the proof-structure change (a different blob) — read it as
# "total verifier work for each side's own proof", not an isolated guest-code delta.
#
# For each ref we report two numbers, both read from one `execute --cycles` run of a
# measuring CLI built FROM THAT REF (its own worktree, release `cli`):
#   * Guest cycles  — retired instructions.
#   * Keccak calls  — keccak-permutation accelerator ecalls (one cycle each, but each
#                     runs a whole permutation invisibly, so it's the companion signal:
#                     the verifier's Merkle/transcript hashing rides on this syscall).
# The CLI also prints an Ecsm (EC scalar-mul) count, but the STARK verifier does no
# scalar-mul, so it is structurally 0 for a recursion proof — dropped as noise, not read.
# Each ref is measured by a CLI built from THAT SAME ref — never a single shared counter
# built from the checkout (main, in CI's issue_comment flow). A shared main-built CLI
# only counts guests whose syscalls main already knows; the moment a PR guest emits a
# NEW syscall (e.g. a new accelerator ecall) the main executor aborts with
# `UnknownSyscall(...)` and the whole cycle bench fails — even though the PR itself is
# fine. Building the counter per ref makes each VM understand exactly its own guest's
# syscalls, so it is robust for PRs that add OR remove a syscall in either direction —
# mirroring the per-side build already done by scripts/bench_verify.sh. Cost: one extra
# native release `cli` build per ref; it shares HOST_TARGET_DIR with the blob-dump build
# when that is set, so most deps are already warm, and it fits the recursion step's
# existing multi-build budget (two guest builds + two blob dumps already).
#
# Improvement convention matches scripts/bench_verify.sh:
#   NEGATIVE Δ  =  REF_A (PR) does fewer cycles/calls  =  better.
#
# Usage: scripts/bench_recursion_cycles.sh REF_A [REF_B=origin/main] [PRESET=min]
#   REF_A    ref/SHA to evaluate (the PR side).
#   REF_B    baseline ref/SHA (default origin/main).
#   PRESET   recursion-verifier preset (default min): min = blowup=2, 1 query
#            (cheap diagnostic); blowup2 = blowup=2, 219 queries (realistic
#            base-layer, 128-bit); blowup4 = blowup=4, 110 queries (the other
#            base-layer point); blowup8 = blowup=8, 73 queries. Picks BOTH the
#            guest ELF (recursion-<PRESET>.elf, falling back to recursion.elf
#            on older refs) AND the dumped blob's inner-proof options (via
#            RECURSION_DUMP_PRESET). Refs predating the preset-aware dump test
#            only support PRESET=min — the script fails loudly up front rather
#            than let the guest reject the blob in-VM. Different artifact
#            names across refs (e.g. recursion.elf vs recursion-min.elf) is
#            expected — both verify under the SAME preset options.
#            `blowup4-block` isn't a build preset: it's the `continuation` guest
#            (recursion-cont-blowup4.elf) verifying a real ethrex block instead
#            of the `empty` diagnostic program — real prover minutes per ref
#            (see the blob cache below), not seconds.
#   Env:
#     REBUILD=1            force rebuild of each ref's measuring CLI and re-run of every
#                          ref (guest build + blob dump + measurement); ignore caches.
#     SYSROOT_DIR=<path>   guest-build sysroot (default $HOME/.lambda-vm-sysroot).
#     GUEST_TARGET_DIR=<p> share the RV64 guest build dir across ref worktrees
#                          (reuses build-std → big speedup for the 2nd ref's guest
#                          build). Unset = per-worktree (default, fully isolated).
#     HOST_TARGET_DIR=<p>  share the host cargo target dir for the blob-dump test
#                          build across refs. Unset = per-worktree (default).
#     PRUNE_KEEP=<n>       cap on cached ref worktrees kept under $WORK (default 10);
#                          older ones (+ their results/blobs/logs) are pruned at startup
#                          to bound disk on the long-lived bench runner.
#     BLOCK_TXS=4          PRESET=blowup4-block only: ethrex block size, reading
#                          executor/tests/ethrex_bench_<BLOCK_TXS>.bin (only _4 committed).
#     BLOCK_EPOCH_LOG2=21  PRESET=blowup4-block only: inner continuation epoch size.
#
# Caching: each ref's result is cached in $WORK keyed on its resolved SHA + preset. The
# measuring CLI is built from that same SHA, so the SHA already identifies the counter (no
# separate CLI-SHA key component). Result files are written ATOMICALLY (tmp + mv) and VALIDATED on
# read: a truncated/partial cache is discarded and re-measured, never emitted as zeros.
# Ref worktrees are kept (named by SHA) so a re-measure is a cargo no-op; the newest
# PRUNE_KEEP are retained and older ones pruned. A worktree whose guest build fails
# mid-run is removed immediately. The dumped input blob is also cached (keyed on SHA +
# preset), so re-proving blowup4-block's real ethrex block only happens once per ref.
# REBUILD=1 forces everything.
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
PRUNE_KEEP="${PRUNE_KEEP:-10}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

WORK="/tmp/recursion_cycles_run"
mkdir -p "$WORK"

# Bound the on-disk cache so the long-lived bench runner doesn't grow without limit.
# Keep the PRUNE_KEEP most-recently-used ref worktrees (each is `touch`ed on use, so
# "recent" tracks actual use, not creation) and drop older ones plus their cached
# results/blobs/logs. This preserves the reuse speedup for recent refs while capping
# disk; the big consumers are the worktrees (full checkout + build artifacts).
prune_worktree_cache() {
  local stale
  # ls -t mtime-sort is the portable way to order by recency (BSD/GNU find can't do it
  # without GNU-only -printf); names are wt_<sha8>, so word-splitting is safe here.
  # shellcheck disable=SC2012
  stale="$(ls -1dt "$WORK"/wt_* 2>/dev/null | tail -n +"$((PRUNE_KEEP + 1))" || true)"
  [ -n "$stale" ] || return 0
  local wt s8
  while IFS= read -r wt; do
    [ -n "$wt" ] || continue
    s8="$(basename "$wt")"; s8="${s8#wt_}"
    echo "==> Pruning old ref worktree $wt (keeping newest $PRUNE_KEEP)" >&2
    git worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"
    rm -f "$WORK"/result_"${s8}"_*.txt "$WORK"/blob_"${s8}"_*.bin \
          "$WORK"/build_guest_"${s8}".log "$WORK"/dump_"${s8}"*.log \
          "$WORK"/measure_"${s8}"*.err "$WORK"/measure_cli_"${s8}"* \
          "$WORK"/build_cli_"${s8}".log
  done <<< "$stale"
  git worktree prune >/dev/null 2>&1 || true
}
prune_worktree_cache

# One-time sweep of the retired single-CLI scheme's fixed-name artifacts. Before this
# script measured per ref it built one shared counter at $WORK/measure_cli (+ its .sha
# marker and build_measure_cli.log). Those are never written or read anymore, and their
# fixed names escape the per-SHA prune globs above, so on the long-lived bench runner
# they would linger forever. Drop them so the disk-bounding claim actually holds.
rm -f "$WORK"/measure_cli "$WORK"/measure_cli.sha "$WORK"/build_measure_cli.log

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

# --- 1. Measuring CLI is built PER REF (see measure_ref, step 2c2) --------------
# There is deliberately no single shared counter built here. Each ref's guest is
# executed by a `cli` built from that same ref's worktree, so the executor always knows
# exactly the syscalls its own guest emits. A main-built CLI cannot run a PR guest that
# introduces a new syscall — it aborts with `UnknownSyscall(...)` — which is precisely
# the failure this per-ref scheme replaces.

# Validate a result record (key=value lines on stdin): the three numeric keys must be
# present and integer, and elf must be non-empty. Exit 0 iff trustworthy. Used both to
# vet a cached result before reuse and to guard the final table/RAW emit, so a
# truncated/partial cache (e.g. a run killed mid-write) can never surface as bogus zeros.
# (An older cache may also carry a legacy `ecsm=` line; it's simply ignored here.)
valid_result() {
  awk -F= '
    $1=="cycles" {c=$2} $1=="keccak" {k=$2}
    $1=="wall"   {w=$2} $1=="elf"    {f=$2}
    END {
      if (c ~ /^[0-9]+$/ && k ~ /^[0-9]+$/ &&
          w ~ /^[0-9]+$/ && length(f) > 0) exit 0
      exit 1
    }'
}

# --- 2. Per-ref: worktree + guest build + blob dump + measurement ---------------
# Prints progress to stderr; emits the parseable result block (key=value lines) to
# stdout so the caller can capture it.
measure_ref() {
  local ref="$1" sha="$2" role="$3"
  local sha8="${sha:0:8}"
  # `blowup4-block`: same cache/worktree/measure plumbing, but a real ethrex
  # block through the continuation guest instead of the `min`/`blowup*`
  # presets' `empty`-program blob. BLOCK_PRESET is the underlying build
  # preset (blowup4); BLOCK_TXS/BLOCK_EPOCH_LOG2 pin the fixture and epoch
  # size to what `make recursion-profile-block-input` proves.
  local is_block=0 block_preset=""
  if [ "$PRESET" = "blowup4-block" ]; then
    is_block=1
    block_preset="blowup4"
  fi
  local block_txs="${BLOCK_TXS:-4}"
  local block_epoch_log2="${BLOCK_EPOCH_LOG2:-21}"

  # Blob cache: keyed on sha + preset (+ block fixture/epoch), persists across runs.
  local blob_key="$PRESET"
  if [ "$is_block" = 1 ]; then
    blob_key="${PRESET}_txs${block_txs}_epoch${block_epoch_log2}"
  fi
  # Key the result cache on ref SHA + blob_key (so a BLOCK_TXS/BLOCK_EPOCH_LOG2
  # override never reuses a stale measurement). The counter is now built from this same
  # ref SHA, so the SHA already identifies the counter — no separate CLI-SHA component is
  # needed. This new key shape also naturally ignores any caches written by the old
  # single-main-CLI scheme (which carried a `_m<head_sha>` suffix).
  local result="$WORK/result_${sha8}_${blob_key}.txt"
  local wt="$WORK/wt_${sha8}"

  local blob="$WORK/blob_${sha8}_${blob_key}.bin"
  local need_dump=1
  if [ "${REBUILD:-0}" != "1" ] && [ -s "$blob" ]; then
    need_dump=0
  fi

  if [ "${REBUILD:-0}" != "1" ] && [ -f "$result" ]; then
    if valid_result < "$result"; then
      echo "==> [$role] Reusing cached measurement: $ref ($sha8) preset=$PRESET" >&2
      # Mark this ref as recently used so the startup prune keeps its worktree/result.
      touch "$result" 2>/dev/null || true
      if [ -d "$wt" ]; then touch "$wt" 2>/dev/null || true; fi
      cat "$result"
      return 0
    fi
    echo "==> [$role] Cached result corrupt/incomplete ($result) — discarding and re-measuring." >&2
    rm -f "$result"
  fi

  if [ ! -d "$wt" ]; then
    echo "==> [$role] Adding worktree $wt @ $sha8" >&2
    git worktree prune
    git worktree add --detach "$wt" "$sha" >/dev/null
  else
    echo "==> [$role] Reusing worktree $wt (checkout -f $sha8)" >&2
    git -C "$wt" checkout --quiet -f "$sha"
  fi
  touch "$wt" 2>/dev/null || true

  # 2a. Build the recursion guest ELF(s) (+ empty.elf inner program), and for
  # block mode also the ethrex inner guest. GUEST_TARGET_DIR, when set, shares
  # the RV64 build dir across ref worktrees (reuses build-std).
  echo "==> [$role] make compile-recursion-elfs @ $sha8 (slow the first time) ..." >&2
  local glog="$WORK/build_guest_${sha8}.log"
  local -a make_goals=(compile-recursion-elfs)
  if [ "$is_block" = 1 ] && [ "$need_dump" = 1 ]; then
    make_goals+=(executor/program_artifacts/rust/ethrex.elf)
  fi
  local -a make_args=("${make_goals[@]}")
  if [ -n "${GUEST_TARGET_DIR:-}" ]; then
    make_args+=("SHARED_TARGET_DIR=$GUEST_TARGET_DIR")
  fi
  if ! ( cd "$wt" && SYSROOT_DIR="$SYSROOT_DIR" make "${make_args[@]}" ) >"$glog" 2>&1; then
    echo "ERROR: [$role] 'make ${make_goals[*]}' failed for $ref ($sha8). Tail of $glog:" >&2
    tail -40 "$glog" >&2
    # A failed build can leave a partial worktree; drop it so it never lingers or
    # poisons a later reuse. (The startup prune also caps total worktrees.)
    git worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"
    git worktree prune >/dev/null 2>&1 || true
    exit 1
  fi

  # 2b. Detect the guest ELF: block mode always wants recursion-cont-<preset>.elf;
  # otherwise prefer recursion-<PRESET>.elf, else recursion.elf.
  local artdir="$wt/executor/program_artifacts/recursion"
  local guest_elf=""
  if [ "$is_block" = 1 ]; then
    guest_elf="$artdir/recursion-cont-${block_preset}.elf"
    if [ ! -f "$guest_elf" ]; then
      echo "ERROR: [$role] no $guest_elf for $ref ($sha8) — ref predates the continuation guest." >&2
      exit 1
    fi
  elif [ -f "$artdir/recursion-${PRESET}.elf" ]; then
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

  # The measuring CLI is now built from THIS ref (step 2c2), not from main. The old
  # shared-from-main counter always carried the `execute --cycles` keccak/ecsm counters
  # (#807, 7dbbb1ff), so it could measure any ref; a per-ref CLI can only if THIS ref has
  # them. A ref predating #807 still builds a `cli` that runs, but prints no
  # `Keccak calls:` line — the parse in step 2d would then fail late with an opaque
  # message. Refuse up front (before the expensive blob dump), matching the other
  # "ref predates X" guards below. The default baseline origin/main always has #807, so
  # normal PR-vs-main runs never hit this; it only bites a deliberately old baseline.
  if ! grep -q "Keccak calls:" "$wt/bin/cli/src/main.rs" 2>/dev/null; then
    echo "ERROR: [$role] ref $ref ($sha8) predates the execute --cycles keccak/ecsm counters (#807, 7dbbb1ff): its CLI emits no 'Keccak calls:' line, so guest cycles/keccak are not measurable. Use a baseline at or after #807." >&2
    exit 1
  fi

  # 2c. Generate this ref's own input blob via its ignored dump test, unless a
  # cached blob covers this sha/preset already (need_dump=0). Refuse up front if
  # the ref predates a needed knob, instead of failing in-VM verification later.
  if [ "$need_dump" = 0 ]; then
    echo "==> [$role] Reusing cached recursion input blob ($blob) — skipping re-prove." >&2
  else
    if ! grep -rq "fn test_dump_recursion_input" "$wt/prover/src/tests/" 2>/dev/null; then
      echo "ERROR: [$role] ref $ref ($sha8) has no 'test_dump_recursion_input' — cannot generate its input blob." >&2
      exit 1
    fi
    if [ "$PRESET" != "min" ] && ! grep -rq "RECURSION_DUMP_PRESET" "$wt/prover/src/tests/" 2>/dev/null; then
      echo "ERROR: [$role] ref $ref ($sha8) predates the preset-aware dump test (no RECURSION_DUMP_PRESET) — only PRESET=min is measurable for it." >&2
      exit 1
    fi
    local -a dump_env=("RECURSION_DUMP_PRESET=${block_preset:-$PRESET}")
    if [ "$is_block" = 1 ]; then
      if ! grep -rq "RECURSION_DUMP_EPOCH_LOG2" "$wt/prover/src/tests/" 2>/dev/null; then
        echo "ERROR: [$role] ref $ref ($sha8) predates RECURSION_DUMP_EPOCH_LOG2 — blowup4-block is not measurable for it." >&2
        exit 1
      fi
      local block_fixture="$wt/executor/tests/ethrex_bench_${block_txs}.bin"
      if [ ! -f "$block_fixture" ]; then
        echo "ERROR: [$role] ref $ref ($sha8) is missing $block_fixture (ethrex block fixture) — blowup4-block is not measurable for it." >&2
        exit 1
      fi
      dump_env+=(
        "RECURSION_DUMP_EPOCH_LOG2=$block_epoch_log2"
        "RECURSION_DUMP_INNER_ELF=$wt/executor/program_artifacts/rust/ethrex.elf"
        "RECURSION_DUMP_INNER_INPUT=$block_fixture"
      )
    fi
    echo "==> [$role] dumping recursion input blob (cargo test test_dump_recursion_input, preset=$PRESET) ..." >&2
    rm -f /tmp/recursion_input.bin
    local dlog="$WORK/dump_${sha8}_${PRESET}.log"
    if [ -n "${HOST_TARGET_DIR:-}" ]; then
      if ! ( cd "$wt" && env "${dump_env[@]}" CARGO_TARGET_DIR="$HOST_TARGET_DIR" cargo test --release -p lambda-vm-prover --lib test_dump_recursion_input -- --ignored --nocapture ) >"$dlog" 2>&1; then
        echo "ERROR: [$role] blob-dump test failed for $ref ($sha8). Tail of $dlog:" >&2
        tail -40 "$dlog" >&2
        exit 1
      fi
    else
      if ! ( cd "$wt" && env "${dump_env[@]}" cargo test --release -p lambda-vm-prover --lib test_dump_recursion_input -- --ignored --nocapture ) >"$dlog" 2>&1; then
        echo "ERROR: [$role] blob-dump test failed for $ref ($sha8). Tail of $dlog:" >&2
        tail -40 "$dlog" >&2
        exit 1
      fi
    fi
    if [ ! -f /tmp/recursion_input.bin ]; then
      echo "ERROR: [$role] test_dump_recursion_input did not write /tmp/recursion_input.bin for $ref ($sha8)." >&2
      exit 1
    fi
    mv /tmp/recursion_input.bin "$blob"
  fi
  echo "==> [$role] blob: $(wc -c <"$blob" | tr -d '[:space:]') bytes -> $blob" >&2

  # 2c2. Build the measuring CLI FROM THIS REF's worktree (native release `cli`) and keep
  # it at a per-ref stable path. This is the crux of the per-ref design: the guest ELF
  # above may emit a syscall this ref introduced, so it must be executed by an executor
  # built from the same ref — a CLI built from another ref (e.g. main) would abort with
  # UnknownSyscall. Share HOST_TARGET_DIR (when set) with the blob-dump build so common
  # native deps are already compiled, and copy the result out so a shared target dir's
  # `cli` isn't clobbered by the other ref's build. The copy-out is ATOMIC (cp to a tmp
  # path + mv within $WORK), so a run killed mid-copy can never leave a truncated-but-
  # executable binary that the `[ -x ]` reuse check would then trust — matching the
  # atomic tmp+mv used for result files below. The per-ref binary name encodes the SHA,
  # so it doubles as its own cache (rebuilt only on REBUILD=1 or first sight).
  local measure_cli="$WORK/measure_cli_${sha8}"
  if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$measure_cli" ]; then
    echo "==> [$role] building measuring CLI (cli, release) @ $sha8 ..." >&2
    local clilog="$WORK/build_cli_${sha8}.log"
    if [ -n "${HOST_TARGET_DIR:-}" ]; then
      if ! ( cd "$wt" && CARGO_TARGET_DIR="$HOST_TARGET_DIR" cargo build --release -p cli ) >"$clilog" 2>&1; then
        echo "ERROR: [$role] cli build failed for $ref ($sha8). Tail of $clilog:" >&2
        tail -40 "$clilog" >&2
        exit 1
      fi
      cp "$HOST_TARGET_DIR/release/cli" "$measure_cli.tmp"
    else
      if ! ( cd "$wt" && cargo build --release -p cli ) >"$clilog" 2>&1; then
        echo "ERROR: [$role] cli build failed for $ref ($sha8). Tail of $clilog:" >&2
        tail -40 "$clilog" >&2
        exit 1
      fi
      cp "$wt/target/release/cli" "$measure_cli.tmp"
    fi
    mv -f "$measure_cli.tmp" "$measure_cli"
  else
    echo "==> [$role] reusing cached measuring CLI ($measure_cli)" >&2
  fi

  # 2d. Measure: one deterministic execute --cycles run. Time it (CI feasibility).
  echo "==> [$role] measuring: $measure_cli execute $(basename "$guest_elf") --private-input <blob> --cycles" >&2
  local t0 t1 dt out
  t0=$(date +%s)
  if ! out="$("$measure_cli" execute "$guest_elf" --private-input "$blob" --cycles 2>"$WORK/measure_${sha8}_${PRESET}.err")"; then
    echo "ERROR: [$role] measuring-CLI execute failed for $ref ($sha8). Tail of stderr:" >&2
    tail -20 "$WORK/measure_${sha8}_${PRESET}.err" >&2
    exit 1
  fi
  t1=$(date +%s); dt=$((t1 - t0))

  local cyc kec
  cyc="$(printf '%s\n' "$out" | awk -F': ' '/^Cycles:/{print $2; exit}')"
  kec="$(printf '%s\n' "$out" | awk -F': ' '/^Keccak calls:/{print $2; exit}')"
  # The CLI also prints an "Ecsm calls:" line; we intentionally don't read it — it is
  # structurally 0 for a recursion proof (no EC scalar-mul), so it's dropped as noise.
  if [ -z "$cyc" ] || [ -z "$kec" ]; then
    echo "ERROR: [$role] could not parse Cycles/Keccak from measuring-CLI output for $ref ($sha8):" >&2
    printf '%s\n' "$out" >&2
    exit 1
  fi
  echo "==> [$role] cycles=$cyc keccak=$kec  (execute wall-time ${dt}s)" >&2

  # Write atomically (tmp + mv) so a run killed mid-write never leaves a half file that
  # a later run would trust and parse as zeros.
  {
    printf 'cycles=%s\n' "$cyc"
    printf 'keccak=%s\n' "$kec"
    printf 'wall=%s\n' "$dt"
    printf 'elf=%s\n' "$(basename "$guest_elf")"
  } > "$result.tmp"
  mv -f "$result.tmp" "$result"
  cat "$result"
}

# Baseline first, then PR (so a fresh GUEST_TARGET_DIR is warmed by the baseline).
RES_B="$(measure_ref "$REF_B" "$SHA_B" baseline)"
RES_A="$(measure_ref "$REF_A" "$SHA_A" PR)"

# Guard the emit: refuse to print a table/RAW block from any incomplete/non-numeric
# record. A truncated result must fail loudly here, never render as a bogus zero/-100%.
if ! printf '%s\n' "$RES_B" | valid_result; then
  echo "ERROR: baseline measurement is incomplete/non-numeric — refusing to emit a bogus table." >&2
  printf '%s\n' "$RES_B" >&2
  exit 1
fi
if ! printf '%s\n' "$RES_A" | valid_result; then
  echo "ERROR: PR measurement is incomplete/non-numeric — refusing to emit a bogus table." >&2
  printf '%s\n' "$RES_A" >&2
  exit 1
fi

getv() { printf '%s\n' "$1" | awk -F= -v k="$2" '$1==k{print $2; exit}'; }
CYC_B="$(getv "$RES_B" cycles)"; KEC_B="$(getv "$RES_B" keccak)"
WALL_B="$(getv "$RES_B" wall)"; ELF_B="$(getv "$RES_B" elf)"
CYC_A="$(getv "$RES_A" cycles)"; KEC_A="$(getv "$RES_A" keccak)"
WALL_A="$(getv "$RES_A" wall)"; ELF_A="$(getv "$RES_A" elf)"

# signed integer delta (A - B); 0 prints bare, >0 gets a leading '+'
sd() { local d=$(( $1 - $2 )); if [ "$d" -gt 0 ]; then printf '+%d' "$d"; else printf '%d' "$d"; fi; }
# A single guest-cycle count rendered in millions, one decimal, e.g. 5239.7M.
mcyc() { awk -v v="$1" 'BEGIN{ printf("%.1fM", v/1e6); }'; }
# Guest-cycle delta (A - B) in signed millions (one decimal) + percentage of baseline,
# e.g. -5113.7M (-97.60%). Staying on awk's double path (no %d) is deliberate: it also
# dodges mawk's 32-bit %d truncation, which otherwise saturated a multi-billion-cycle
# delta to -2147483647 on the CI bench runner (gawk was fine, so it slipped local tests).
mcycd() {
  awk -v a="$1" -v b="$2" 'BEGIN{
    d=a-b;
    dm=d/1e6;
    pct=(b!=0)? d/b*100 : 0;
    printf("%s%.1fM (%s%.2f%%)", (dm>=0?"+":""), dm, (pct>=0?"+":""), pct);
  }'
}

# Human label for the proof regime this preset measures, so a reader can't mistake the
# single-query `min` number for the full 128-bit verifier cost. CI passes `min` plus
# the full-query regimes `blowup2`/`blowup4` (see .github/workflows/bench-verify.yml).
case "$PRESET" in
  min)     REGIME="single query (blowup=2, 1 query)" ;;
  blowup2) REGIME="128-bit (blowup=2, 219 queries — realistic base-layer)" ;;
  blowup4) REGIME="128-bit (blowup=4, 110 queries — realistic base-layer)" ;;
  blowup8) REGIME="128-bit (blowup=8, 73 queries)" ;;
  blowup4-block) REGIME="128-bit (blowup=4, 110 queries) — real ethrex block, 4 transfers" ;;
  *)       REGIME="$PRESET" ;;
esac

echo
echo "=== Recursion-guest cycle comparison — $REGIME — deterministic to ~±100k cycles ==="
echo "   REF_B (baseline) $REF_B  ${SHA_B:0:10}  guest=$ELF_B"
echo "   REF_A (PR)       $REF_A  ${SHA_A:0:10}  guest=$ELF_A"
echo
echo "| Metric        | REF_B (baseline) | REF_A (PR) | Δ (A-B) |"
echo "|---------------|------------------|------------|---------|"
# Guest cycles are shown in MILLIONS (one decimal); the exact integer counts are in
# the collapsed raw block below. Keccak stays a plain integer call count.
printf '| Guest cycles  | %s | %s | %s |\n' "$(mcyc "$CYC_B")" "$(mcyc "$CYC_A")" "$(mcycd "$CYC_A" "$CYC_B")"
printf '| Keccak calls  | %s | %s | %s |\n' "$KEC_B" "$KEC_A" "$(sd "$KEC_A" "$KEC_B")"
# One terse reproducibility caveat; the blank line before it ends the markdown table.
echo
echo "note: cycles reproduce to ~±100k (build codegen + proof nondeterminism); treat sub-100k deltas as noise, not signal."
# Exact machine-parseable counts, collapsed so they don't clutter the PR comment (the
# table above is rounded to millions; these are the exact integers). The blank lines
# around the fence are required for GitHub to render the code block inside <details>.
echo
echo "<details><summary>raw (exact integer counts)</summary>"
echo
echo '```'
printf 'ref_b_sha=%s ref_b_elf=%s ref_b_cycles=%s ref_b_keccak=%s ref_b_execute_wall_s=%s\n' \
  "$SHA_B" "$ELF_B" "$CYC_B" "$KEC_B" "$WALL_B"
printf 'ref_a_sha=%s ref_a_elf=%s ref_a_cycles=%s ref_a_keccak=%s ref_a_execute_wall_s=%s\n' \
  "$SHA_A" "$ELF_A" "$CYC_A" "$KEC_A" "$WALL_A"
printf 'delta_cycles=%s delta_keccak=%s\n' \
  "$(( CYC_A - CYC_B ))" "$(( KEC_A - KEC_B ))"
echo '```'
echo
echo "</details>"
