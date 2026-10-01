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
# native release `cli` build per ref; it shares that ref's own host target dir with the
# blob-dump build, so most deps are already warm, and it fits the recursion step's
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
#            `blowup2-block`/`blowup4-block` aren't build presets: they are the
#            `continuation` guest (recursion-cont-<blowup>.elf) verifying a real
#            ethrex block instead of the `empty` diagnostic program — real
#            prover minutes per ref (see the blob cache below), not seconds.
#   Env:
#     REBUILD=1            force rebuild of each ref's measuring CLI and re-run of every
#                          ref (guest build + blob dump + measurement); ignore caches.
#     SYSROOT_DIR=<path>   guest-build sysroot (default $HOME/.lambda-vm-sysroot).
#     GUEST_TARGET_DIR=<p> base path for the RV64 guest build dir. Each ref gets its
#                          OWN dir, `<p>_<sha8>` — NEVER one dir shared by both refs
#                          (see the cross-ref clobbering note below). Unset =
#                          per-worktree (cargo's default target/, also isolated).
#     HOST_TARGET_DIR=<p>  base path for the host cargo target dir (blob-dump test +
#                          measuring CLI). Each ref gets its OWN dir, `<p>_<sha8>` —
#                          NEVER one dir shared by both refs (see the cross-ref
#                          clobbering note below). Unset = per-worktree (cargo's
#                          default target/, also isolated).
#     PRUNE_KEEP=<n>       cap on cached ref worktrees kept under $WORK (default 10);
#                          older ones (+ their results/blobs/logs) are pruned at startup
#                          to bound disk on the long-lived bench runner.
#     GUEST_TARGET_KEEP=<n> cap on per-ref guest target dirs kept (default 3). Separate
#                          from PRUNE_KEEP and much tighter: these are GBs each, and only
#                          the current run's two refs need one (3 leaves room to re-run
#                          the same PR without a cold rebuild). See
#                          prune_ref_target_dirs for why they need their own sweep.
#     HOST_TARGET_KEEP=<n> same cap for the per-ref HOST target dirs (default 3).
#     BLOCK_TXS=20         PRESET=blowup<N>-block only: ethrex block size. Reads
#                          executor/tests/ethrex_bench_<BLOCK_TXS>.bin when present
#                          (only _4 is committed) and generates any other size via
#                          tooling/ethrex-fixtures (see resolve_block_fixture).
#     BLOCK_EPOCH_LOG2=21  PRESET=blowup<N>-block only: inner continuation epoch size.
#                          Smaller epochs mean MORE of them, and the whole bundle has to
#                          fit the guest's MAX_PRIVATE_INPUT_SIZE (512 MiB), so check the
#                          blob size if you lower it for a big block. Measured room at
#                          the defaults: an ethrex 20-tx block is 9,073,658 cycles
#                          (`cli execute --cycles`), so 2^21 is 5 epochs; the CI 4-tx
#                          blob was 70.6 MB for 2 epochs, i.e. ~35 MB/epoch, putting 20
#                          txs near 175 MB at blowup4 and ~350 MB at blowup2 — both
#                          inside the cap. (An earlier version of this comment claimed
#                          ~335 MB at blowup4 and that blowup2 would not fit; that was
#                          extrapolated from the stale ~4M cycles/transfer figure in
#                          tooling/ethrex-fixtures/README.md, which predates the
#                          ecrecover accelerator and overstates the block by ~9x.)
#
# Caching: each ref's result is cached in $WORK keyed on its resolved SHA + preset. The
# measuring CLI is built from that same SHA, so the SHA already identifies the counter (no
# separate CLI-SHA key component). Result files are written ATOMICALLY (tmp + mv) and VALIDATED on
# read: a truncated/partial cache is discarded and re-measured, never emitted as zeros.
# Ref worktrees are kept (named by SHA) so a re-measure is a cargo no-op; the newest
# PRUNE_KEEP are retained and older ones pruned. A worktree whose guest build fails
# mid-run is removed immediately. The dumped input blob is also cached (keyed on SHA +
# preset), so re-proving a blowup<N>-block real ethrex block only happens once per ref.
# REBUILD=1 forces everything.
#
# NEVER point two refs at one CARGO_TARGET_DIR, guest or host. Two worktrees are two
# distinct source roots; building both into a single target dir makes cargo consider the
# crates that did NOT change between the refs "fresh" and reuse rlibs compiled from the OTHER
# worktree, while rebuilding the ones that did — so a build that alternates refs dies
# with `multiple different versions of crate math in the dependency graph` naming both
# worktrees. That is exactly how the blowup2/blowup4 regimes silently went "unavailable"
# in CI: the FIRST preset's two builds are both first-sight and succeed, and every later
# preset returns to an already-built worktree and fails. Hence the per-ref
# `${GUEST_TARGET_DIR}_<sha8>` below: each source root owns its target dir, so build-std
# is still reused across presets AND across runs (just not across refs), and a repeated
# `make compile-recursion-elfs` for the same ref is the intended cargo no-op.
#
# On the HOST side the same sharing fails SILENTLY, which is worse: it does not error, it
# measures the wrong binary. Cargo's freshness check walks a dep-info list of paths
# RELATIVE to the invocation, so from the other worktree they all resolve and their mtimes
# are older than the artifact — cargo prints `Finished` in 0.06s and the run executes the
# binary the OTHER ref linked (the test-harness filename carries no worktree component, so
# both refs overwrite one path). The direction is fixed: the ref that only ADDS files is
# the one that gets skipped, because in the reverse order cargo finds the added file
# missing from the other worktree and rebuilds. In a /bench-verify run on the DMA PR the
# blob dump for the PR ref therefore ran the BASELINE's harness — same
# `<host_target>/release/deps/lambda_vm_prover-<hash>` path, `Finished` in 0.06s, and 546
# tests where that PR's own harness has 562. That count is the cheapest fingerprint: the
# rest of the log reads like a normal run. What executes is the other ref's WHOLE binary,
# so there is no per-crate mix to attribute. It produced a bundle that did not verify,
# surfacing as the PR's `continuation bundle must verify on host before dumping` on a PR
# whose own binary verifies that bundle fine. Hence `${HOST_TARGET_DIR}_<sha8>` too. Cost:
# one cold host build per ref — measured on the bench runner at ~25 s for the prover test
# harness plus ~10 s for the CLI, so ~35 s — still warm across presets and across runs for
# the same ref.
#
# In CI the reuse precondition is stronger than "the worktree exists": `git -C "$wt"
# checkout -f` FAILS there, because actions/checkout rebuilds $ROOT/.git every job and
# orphans the worktree registrations (`fatal: not a git repository: .../worktrees/wt_<sha8>`
# in the logs). The failure is swallowed, so a reused worktree is never refreshed and its
# file mtimes stay as old as its first checkout.
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
GUEST_TARGET_KEEP="${GUEST_TARGET_KEEP:-3}"
HOST_TARGET_KEEP="${HOST_TARGET_KEEP:-3}"
BLOCK_TXS="${BLOCK_TXS:-20}"
BLOCK_EPOCH_LOG2="${BLOCK_EPOCH_LOG2:-21}"

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
    rm -f "$WORK"/result_"${s8}"_*.txt "$WORK"/blob_"${s8}"_*.bin* \
          "$WORK"/build_guest_"${s8}".log "$WORK"/dump_"${s8}"*.log \
          "$WORK"/measure_"${s8}"*.err "$WORK"/measure_cli_"${s8}"* \
          "$WORK"/build_cli_"${s8}".log
    # The per-ref target dirs are the biggest artifacts of all (build-std + the guest
    # builds; the native deps + prover test harness + CLI on the host side) and their
    # names escape the wt_* glob above, so drop them here too or the disk-bounding claim
    # stops holding.
    if [ -n "${GUEST_TARGET_DIR:-}" ]; then
      rm -rf "${GUEST_TARGET_DIR}_${s8}"
    fi
    if [ -n "${HOST_TARGET_DIR:-}" ]; then
      rm -rf "${HOST_TARGET_DIR}_${s8}"
    fi
  done <<< "$stale"
  git worktree prune >/dev/null 2>&1 || true
}
prune_worktree_cache

# The per-ref target dirs need their OWN sweep, not just the per-worktree removal
# above, for two reasons. (1) They are not discoverable from the wt_* glob once their
# worktree is gone, and the mid-run build-failure path removes a worktree IMMEDIATELY —
# which would strand that ref's target dir forever, unreclaimable, on a long-lived
# runner. Guest builds failing is exactly the scenario this script exists to measure, so
# that is not a rare path. (2) They are the biggest thing here (build-std + the guest
# builds; the native deps + prover test harness + CLI on the host side — GBs each), and
# only the CURRENT run's two refs need one, so they deserve a tighter cap than the
# worktrees (which are cheaper and worth keeping around longer for checkout reuse).
#
# The invariant enforced is "a target dir survives only while its worktree does": a
# worktree that vanished either aged out or died mid-build, and in both cases a clean
# rebuild is what we want. This is self-healing — it also reclaims dirs orphaned by runs
# that predate this sweep.
#
# Deliberately NOT extended to the cached blobs: an orphaned blob_<sha>_<preset>.bin is
# keyed on the ref SHA, stays valid without its worktree, and represents real prover
# minutes (a 20-tx continuation prove), so dropping it would throw away an expensive and
# still-correct cache to reclaim ~300 MB.
prune_ref_target_dirs() {
  local base="$1" keep="$2" label="$3"
  [ -n "$base" ] || return 0
  local d s8 stale
  for d in "${base}"_*; do
    [ -d "$d" ] || continue
    s8="$(basename "$d")"; s8="${s8##*_}"
    if [ ! -d "$WORK/wt_${s8}" ]; then
      echo "==> Pruning orphaned $label target dir $d (no worktree $WORK/wt_${s8})" >&2
      rm -rf "$d"
    fi
  done
  # Same ls -t recency ordering as the worktree prune; names are <base>_<sha8>, so
  # word-splitting is safe. Each dir is `touch`ed after its build, so this tracks use.
  # shellcheck disable=SC2012
  stale="$(ls -1dt "${base}"_* 2>/dev/null | tail -n +"$((keep + 1))" || true)"
  [ -n "$stale" ] || return 0
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    echo "==> Pruning old $label target dir $d (keeping newest $keep)" >&2
    rm -rf "$d"
  done <<< "$stale"
}
prune_ref_target_dirs "${GUEST_TARGET_DIR:-}" "$GUEST_TARGET_KEEP" guest
prune_ref_target_dirs "${HOST_TARGET_DIR:-}" "$HOST_TARGET_KEEP" host

# One-time sweep of the retired single-CLI scheme's fixed-name artifacts. Before this
# script measured per ref it built one shared counter at $WORK/measure_cli (+ its .sha
# marker and build_measure_cli.log). Those are never written or read anymore, and their
# fixed names escape the per-SHA prune globs above, so on the long-lived bench runner
# they would linger forever. Drop them so the disk-bounding claim actually holds.
rm -f "$WORK"/measure_cli "$WORK"/measure_cli.sha "$WORK"/build_measure_cli.log

# Same for the retired single-shared-target schemes, guest and host: CI used to point
# GUEST_TARGET_DIR / HOST_TARGET_DIR at one fixed path for BOTH refs, which is exactly
# what poisoned every regime after the first (guest) and silently ran one ref's binary
# for the other (host). Nothing writes or reads them now (each ref builds into
# ${*_TARGET_DIR}_<sha8>), their names escape the per-SHA prune globs, and they are the
# largest things on disk — so reclaim them once. The CACHEDIR.TAG check keeps this an
# rm -rf of a cargo target dir and nothing else: cargo writes that file into every
# target dir it creates.
for retired in "$WORK/shared_guest_target" "$WORK/shared_host_target"; do
  if [ -f "$retired/CACHEDIR.TAG" ]; then
    echo "==> Removing retired shared target dir $retired" >&2
    rm -rf "$retired"
    # Host-only flag: a guest build produces ELFs inside its worktree, not cached copies,
    # and it failed LOUDLY when it mixed refs — nothing of unknown provenance survives it.
    case "$retired" in */shared_host_target) retired_host=1 ;; esac
  fi
done

# Dropping the retired host dir is not enough: everything a build INSIDE it produced was
# copied out and is cached under a name the sweep above does not touch, and each cache is
# reused on presence alone — `[ -x ]` for the measuring CLI, `[ -s ]` for the blob. A
# measuring CLI or an input blob that a skipped build handed over from the other ref would
# therefore survive the fix and keep feeding one more comparison. Their provenance is not
# checkable after the fact, so retire them with the dir that could have produced them: the
# cost is re-proving each ref's blob once (~50 s) and rebuilding its CLI (~10 s), against
# an advisory number measured on the wrong binary. This is a ONE-TIME branch — steady
# state keeps every cache, and the per-ref dirs mean no later build can poison one.
if [ "${retired_host:-0}" = 1 ]; then
  echo "==> Retiring artifacts copied out of the shared host dir (CLIs, blobs, results)" >&2
  rm -f "$WORK"/measure_cli_* "$WORK"/build_cli_*.log \
        "$WORK"/blob_*.bin "$WORK"/blob_*.bin.epochs "$WORK"/result_*.txt \
        "$WORK"/dump_*.log
fi

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

# Resolve the ethrex block fixture for PRESET=blowup<N>-block and echo its path. Both refs
# are handed the SAME bytes (cached in $WORK, keyed on tx count) rather than each reading
# its own worktree copy: the fixture is the WORKLOAD, so a per-ref copy would risk
# comparing two different blocks. Only ethrex_bench_4.bin is committed; other sizes are
# generated by tooling/ethrex-fixtures, which is deterministic for a given
# (n_transfers, mode) — see its README. In the CI flow scripts/bench_verify.sh has
# already generated the 20-tx fixture into the checkout, so step 2 below hits and nothing
# is rebuilt here. A ref that bumps the pinned ethrex rev makes these bytes undecodable
# for that side; the blob dump then fails loudly rather than silently benching a
# different block.
resolve_block_fixture() {
  local txs="$1"
  # The checkout copy WINS over the $WORK cache whenever it exists. $WORK lives forever on
  # the bench runner, so a cache that outranked the checkout would keep verifying an old
  # block after the committed fixture or the generator changed — and since both sections of
  # the comment say "ethrex <N>-tx block", one comment would silently be comparing two
  # different workloads. scripts/bench_verify.sh generates the 20-tx fixture into the
  # checkout earlier in the same CI job, so this is the normal path there too.
  local committed="$ROOT/executor/tests/ethrex_bench_${txs}.bin"
  if [ -f "$committed" ]; then
    printf '%s\n' "$committed"
    return 0
  fi
  local cached="$WORK/ethrex_bench_${txs}.bin"
  if [ "${REBUILD:-0}" != "1" ] && [ -s "$cached" ]; then
    printf '%s\n' "$cached"
    return 0
  fi
  echo "==> Generating missing ${txs}-tx ethrex fixture (tooling/ethrex-fixtures)" >&2
  local flog="$WORK/build_fixtures.log"
  if ! ( cd "$ROOT/tooling/ethrex-fixtures" && cargo build --release ) >"$flog" 2>&1; then
    echo "ERROR: ethrex-fixtures build failed. Tail of $flog:" >&2
    tail -40 "$flog" >&2
    exit 1
  fi
  if ! "$ROOT/tooling/ethrex-fixtures/target/release/ethrex-fixtures" \
         "$txs" "$cached.tmp" distinct >>"$flog" 2>&1; then
    echo "ERROR: ethrex-fixtures failed to generate a ${txs}-tx block. Tail of $flog:" >&2
    tail -40 "$flog" >&2
    exit 1
  fi
  mv -f "$cached.tmp" "$cached"
  printf '%s\n' "$cached"
}

# --- 2. Per-ref: worktree + guest build + blob dump + measurement ---------------
# Prints progress to stderr; emits the parseable result block (key=value lines) to
# stdout so the caller can capture it.
measure_ref() {
  local ref="$1" sha="$2" role="$3"
  local sha8="${sha:0:8}"
  # `blowup<N>-block`: same cache/worktree/measure plumbing, but a real ethrex
  # block through the continuation guest instead of the `min`/`blowup*`
  # presets' `empty`-program blob. block_preset is the underlying build preset;
  # BLOCK_TXS/BLOCK_EPOCH_LOG2 pin the fixture and epoch size.
  local is_block=0 block_preset=""
  case "$PRESET" in
    blowup2-block) is_block=1; block_preset="blowup2" ;;
    blowup4-block) is_block=1; block_preset="blowup4" ;;
  esac
  local block_txs="$BLOCK_TXS"
  local block_epoch_log2="$BLOCK_EPOCH_LOG2"

  # HOST_TARGET_DIR, like GUEST_TARGET_DIR, is a BASE path: this ref's host builds go to
  # ${HOST_TARGET_DIR}_<sha8>. Never the bare base — that is one dir for two source roots,
  # and cargo then declares the second ref fresh and hands it the first ref's binary (see
  # the header note).
  local host_target=""
  if [ -n "${HOST_TARGET_DIR:-}" ]; then
    host_target="${HOST_TARGET_DIR}_${sha8}"
  fi

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
      # The target dirs too: they age out under the much tighter *_TARGET_KEEP caps, so
      # a ref whose results are all cached would otherwise lose them and pay a cold rebuild.
      touch "$result" 2>/dev/null || true
      if [ -d "$wt" ]; then touch "$wt" 2>/dev/null || true; fi
      if [ -n "${GUEST_TARGET_DIR:-}" ] && [ -d "${GUEST_TARGET_DIR}_${sha8}" ]; then
        touch "${GUEST_TARGET_DIR}_${sha8}" 2>/dev/null || true
      fi
      if [ -n "$host_target" ] && [ -d "$host_target" ]; then
        touch "$host_target" 2>/dev/null || true
      fi
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
  # block mode also the ethrex inner guest. GUEST_TARGET_DIR, when set, is a BASE
  # path: this ref builds into ${GUEST_TARGET_DIR}_<sha8>, so build-std is reused
  # across presets and across runs for the SAME ref, never across refs.
  echo "==> [$role] make compile-recursion-elfs @ $sha8 (slow the first time) ..." >&2
  local glog="$WORK/build_guest_${sha8}.log"
  local -a make_goals=(compile-recursion-elfs)
  if [ "$is_block" = 1 ] && [ "$need_dump" = 1 ]; then
    make_goals+=(executor/program_artifacts/rust/ethrex.elf)
  fi
  local -a make_args=("${make_goals[@]}")
  if [ -n "${GUEST_TARGET_DIR:-}" ]; then
    make_args+=("SHARED_TARGET_DIR=${GUEST_TARGET_DIR}_${sha8}")
  fi
  if ! ( cd "$wt" && SYSROOT_DIR="$SYSROOT_DIR" make "${make_args[@]}" ) >"$glog" 2>&1; then
    echo "ERROR: [$role] 'make ${make_goals[*]}' failed for $ref ($sha8). Tail of $glog:" >&2
    tail -40 "$glog" >&2
    # A failed build can leave a partial worktree; drop it so it never lingers or
    # poisons a later reuse. (The startup prune also caps total worktrees.)
    git worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"
    git worktree prune >/dev/null 2>&1 || true
    # Reclaim this ref's target dirs now rather than leaving GBs behind until the next
    # run's prune_ref_target_dirs notices they have no worktree. Their contents are a
    # half-finished build anyway, so a clean rebuild is what we want next time.
    if [ -n "${GUEST_TARGET_DIR:-}" ]; then
      rm -rf "${GUEST_TARGET_DIR}_${sha8}"
    fi
    if [ -n "$host_target" ]; then
      rm -rf "$host_target"
    fi
    exit 1
  fi
  # Mark the target dirs as recently used so prune_ref_target_dirs keeps them. Guarded on
  # -d: a bare `touch` on a first build would create a FILE at that path and break cargo.
  if [ -n "${GUEST_TARGET_DIR:-}" ] && [ -d "${GUEST_TARGET_DIR}_${sha8}" ]; then
    touch "${GUEST_TARGET_DIR}_${sha8}" 2>/dev/null || true
  fi
  if [ -n "$host_target" ] && [ -d "$host_target" ]; then
    touch "$host_target" 2>/dev/null || true
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
  # "ref predates X" guards below (which likewise grep the ref's source recursively). We
  # search the whole cli source tree, not just main.rs, so relocating the counter println
  # into another module doesn't trip a false rejection; the literal stays coupled to the
  # step-2d parser (/^Keccak calls:/), so a genuine output-format change fails here AND
  # there in lockstep. The default baseline origin/main always has #807, so normal
  # PR-vs-main runs never hit this; it only bites a deliberately old baseline.
  if ! grep -rq "Keccak calls:" "$wt/bin/cli/src/" 2>/dev/null; then
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
        echo "ERROR: [$role] ref $ref ($sha8) predates RECURSION_DUMP_EPOCH_LOG2 — blowup<N>-block is not measurable for it." >&2
        exit 1
      fi
      local block_fixture
      block_fixture="$(resolve_block_fixture "$block_txs")"
      dump_env+=(
        "RECURSION_DUMP_EPOCH_LOG2=$block_epoch_log2"
        "RECURSION_DUMP_INNER_ELF=$wt/executor/program_artifacts/rust/ethrex.elf"
        "RECURSION_DUMP_INNER_INPUT=$block_fixture"
      )
    fi
    echo "==> [$role] dumping recursion input blob (cargo test test_dump_recursion_input, preset=$PRESET) ..." >&2
    rm -f /tmp/recursion_input.bin
    local dlog="$WORK/dump_${sha8}_${PRESET}.log"
    if [ -n "$host_target" ]; then
      if ! ( cd "$wt" && env "${dump_env[@]}" CARGO_TARGET_DIR="$host_target" cargo test --release -p lambda-vm-prover --lib test_dump_recursion_input -- --ignored --nocapture ) >"$dlog" 2>&1; then
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
    # Epoch count comes from the dump test's own log line. Persist it beside the blob:
    # a blob cache hit skips the dump entirely, so the log is not a reliable source at
    # report time. Empty for the non-block presets, which prove monolithically.
    awk -F': ' '/continuation epochs:/{print $NF; exit}' "$dlog" > "$blob.epochs"
  fi
  local epochs=""
  if [ -s "$blob.epochs" ]; then
    epochs="$(head -1 "$blob.epochs" | tr -d '[:space:]')"
  fi
  echo "==> [$role] blob: $(wc -c <"$blob" | tr -d '[:space:]') bytes${epochs:+, $epochs epochs} -> $blob" >&2

  # 2c2. Build the measuring CLI FROM THIS REF's worktree (native release `cli`) and keep
  # it at a per-ref stable path. This is the crux of the per-ref design: the guest ELF
  # above may emit a syscall this ref introduced, so it must be executed by an executor
  # built from the same ref — a CLI built from another ref (e.g. main) would abort with
  # UnknownSyscall, or worse count the wrong ref's cycles without saying so. Shares this
  # ref's ${HOST_TARGET_DIR}_<sha8> with the blob-dump build so common native deps are
  # already compiled, and copies the result out to its per-ref path. The copy-out is
  # ATOMIC (cp to a tmp path + mv within $WORK), so a run killed mid-copy can never leave
  # a truncated-but-executable binary that the `[ -x ]` reuse check would then trust —
  # matching the atomic tmp+mv used for result files below. The per-ref binary name
  # encodes the SHA, so it doubles as its own cache (rebuilt only on REBUILD=1 or first
  # sight).
  local measure_cli="$WORK/measure_cli_${sha8}"
  if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$measure_cli" ]; then
    echo "==> [$role] building measuring CLI (cli, release) @ $sha8 ..." >&2
    local clilog="$WORK/build_cli_${sha8}.log"
    if [ -n "$host_target" ]; then
      if ! ( cd "$wt" && CARGO_TARGET_DIR="$host_target" cargo build --release -p cli ) >"$clilog" 2>&1; then
        echo "ERROR: [$role] cli build failed for $ref ($sha8). Tail of $clilog:" >&2
        tail -40 "$clilog" >&2
        exit 1
      fi
      cp "$host_target/release/cli" "$measure_cli.tmp"
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
    # Optional: only the block presets have epochs, and caches written before this line
    # existed have none. valid_result deliberately does not require it, so a missing
    # value degrades to omitting the count rather than reporting a wrong one.
    printf 'epochs=%s\n' "$epochs"
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
EPO_B="$(getv "$RES_B" epochs)"; EPO_A="$(getv "$RES_A" epochs)"

# Epoch COUNT next to the epoch SIZE in the regime label: the size alone doesn't say how
# many epochs the bundle holds, which is what drives both its size and the verifier work.
# Show both sides when they differ — a PR that changes epoch splitting should be visible
# here, not hidden behind one number. Empty when unknown (non-block preset, or a result
# cached before `epochs=` existed): a missing count beats a wrong one.
if [ -n "$EPO_A" ] && [ -n "$EPO_B" ]; then
  if [ "$EPO_A" = "$EPO_B" ]; then
    EPOCHS_LABEL=" ($EPO_A epochs)"
  else
    EPOCHS_LABEL=" (main $EPO_B / PR $EPO_A epochs)"
  fi
else
  EPOCHS_LABEL=""
fi

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
# single-query `min` number for the full 128-bit verifier cost. CI passes `min` (cheap
# canary) and `blowup2-block` (the representative regime); `blowup2`/`blowup4`/
# `blowup4-block` stay
# available for manual runs (see .github/scripts/run_recursion_bench.sh).
case "$PRESET" in
  min)     REGIME="empty program · monolithic · blowup=2, 1 query (diagnostic — NOT a real verifier cost)" ;;
  blowup2) REGIME="empty program · monolithic · blowup=2, 219 queries (128-bit)" ;;
  blowup4) REGIME="empty program · monolithic · blowup=4, 110 queries (128-bit)" ;;
  blowup8) REGIME="empty program · monolithic · blowup=8, 73 queries (128-bit)" ;;
  blowup2-block) REGIME="ethrex ${BLOCK_TXS}-tx block · continuations, epoch 2^$BLOCK_EPOCH_LOG2$EPOCHS_LABEL · blowup=2, 219 queries (128-bit)" ;;
  blowup4-block) REGIME="ethrex ${BLOCK_TXS}-tx block · continuations, epoch 2^$BLOCK_EPOCH_LOG2$EPOCHS_LABEL · blowup=4, 110 queries (128-bit)" ;;
  *)       REGIME="$PRESET" ;;
esac

echo
# Machine anchor for the CI extractor (.github/scripts/run_recursion_bench.sh). An HTML
# comment, so unlike the visible `=== ... ===` banner it replaces it doesn't render in the
# PR comment — the heading below is the human entry point, matching bench_verify.sh.
echo "<!-- recursion-cycle-report -->"
echo "#### $REGIME"
echo
# State the measurement method, because the verifier bench above this in the same PR
# comment IS A/B/B/A with statistics and a reader will otherwise carry that framing down
# here. Nothing on this table is averaged or interleaved.
echo "_Single exact reading per ref — no ABBA: guest cycles are deterministic for a fixed"
echo "(guest ELF, input blob), so there is no machine drift to cancel._"
echo
# Same column names as bench_verify.sh's tables (main / PR / Δ) rather than REF_B / REF_A:
# one comment holds both, and two namings for the same two sides is just friction.
echo "| Metric | main | PR | Δ |"
echo "|--------|------|----|---|"
# Guest cycles are shown in MILLIONS (one decimal); the exact integer counts are in
# the collapsed raw block below. Keccak stays a plain integer call count.
printf '| **Guest cycles** | %s | %s | %s |\n' "$(mcyc "$CYC_B")" "$(mcyc "$CYC_A")" "$(mcycd "$CYC_A" "$CYC_B")"
printf '| **Keccak calls** | %s | %s | %s |\n' "$KEC_B" "$KEC_A" "$(sd "$KEC_A" "$KEC_B")"
# Which refs/guests produced the numbers, plus the reproducibility caveat. Inside a fence
# because GitHub collapses consecutive plain lines into ONE paragraph — as bare lines these
# ran together into an unreadable smear, and the fence also preserves the alignment.
echo
echo '```'
printf '  baseline  %s  %s  guest=%s\n' "$REF_B" "${SHA_B:0:10}" "$ELF_B"
printf '  PR        %s  %s  guest=%s\n' "$REF_A" "${SHA_A:0:10}" "$ELF_A"
echo "  note: cycles reproduce to ~±100k (build codegen + proof nondeterminism);"
echo "        treat sub-100k deltas as noise, not signal."
echo '```'
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
