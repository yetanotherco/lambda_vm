Guest optimization review branch: perf/guest-riscv-round12 (updated through round 15)

Purpose
-------
Build on the published round-12 guest optimizations, including their
ethrex dependencies, so a reviewer can inspect and rebuild one remote branch.
This is a review branch based on the measured #1014/noepoch/whir revision,
c303681b353026cdd9b8009c2003903a2c15c13f. It is not a rebase onto the later
noepoch/whir tip. Unrelated local prover, executor and SHA256 work is excluded.

The first commit imports the unmodified ethrex source snapshot at
797df5540c7d35cafd69b6971a74b2a49c67d1dd into vendor/ethrex, including its
licenses. Review the following optimization commit separately from that import.
Both the guest and crypto provider resolve this same vendored source by relative
path. Fixture tooling and the validation oracle retain the original git pin.

Retained changes
----------------
1. Progressive bytecode analysis, shared across Code clones, with a jump bitmap
   and a target list materialized on demand. Skip PUSH data; scan up to eight
   opcodes per loop where bounds permit. Reuse buffers for empty code.
2. Separate tracing from the ordinary JUMP frame.
3. Allocate the EVM stack directly zeroed on the heap; combine two-operand and
   DUP bounds checks while preserving error behavior.
4. Traverse resident trie nodes by reference and cursor, construct consumed
   paths only when resolving another node, and borrow flat-key-value keys.
5. Remove the duplicate child memoization walk before trie hashing.
6. Encode branch hash preimages in one pass, reserve capacity once, backfill
   prefixes and expand the 16 child writes. Encode RLP lengths/integers from
   their significant bytes.
7. Separate hash and shared-node deserialization, reserve reference tables for
   large witnesses, and deserialize BranchNode directly into its final Box.
8. Skip memory expansion arithmetic when the requested range is already covered.
9. Shorten big-endian conversion on RV64I and process nibble packing/unpacking
   in groups with word-sized shifts and masks.
10. Specialize DIV/MOD for native operands, powers of two and u32 divisors.
    Extend DIV to wider divisors using base-2^32 long division, u64 intermediates
    and quotient correction, including subtraction add-back.
11. Skip byte permutations of zero limbs in MLOAD/MSTORE conversions only.
12. Keep the short-input (<136 bytes) Keccak path. The later round-11 first-full-
    block assignment is REVERTED because it increased COPY rows.
13. Enable word-at-a-time memcmp, preserving unsigned-byte order and bounds.
14. Reuse child substate collections when the parent destination is empty.
    Transfer whole slot maps and vacant per-address slot sets rather than
    rebuilding their hash tables. Preserve union, transient overwrite and log
    append semantics when destinations are populated. Checkpoint push/revert
    and the original parent/delta swap are unchanged.
15. Construct each branch child directly in its final allocation, separating
    hash and shared-node variants. Return only Arc<Node> from the pooling
    helper and construct NodeRef at the caller, avoiding large stack temporaries.
    Archive layout, validation, pooling and partial-initialization cleanup stay
    unchanged; deserialization still goes through rkyv.
16. For jump queries at least 64 bytes beyond the analyzed prefix, test the
    preceding 32 bytes for any PUSH that could cover the destination. Four
    packed byte comparisons implement this conservative proof. Cache only
    proven-valid destinations; uncertain cases scan normally. Never advance
    the prefix cursor for a local proof. Full materialization still scans all
    bytecode. Keep bitmap allocation lazy and outside the scanner's hot body.

No validation or cryptographic checks are intentionally removed. This branch
has not yet undergone the requested security audit. Areas needing careful
review include unsafe scanning/allocation/deserialization, shared Code state,
trie cache and wire-format compatibility, assembly bounds, and DIV correction.

Exact expected results
----------------------
Fixture                     RISC-V instructions   COPY rows
ethrex_10_transfers.bin                  721,587      55,379
ethrex_bench_4.bin                       502,835      46,388
ethrex_empty_block.bin                   186,729      13,595
ethrex_simple_tx.bin                     278,051      33,961
ethrex_mainnet_25368371.bin           15,272,897     529,544
ethrex_mainnet_25453112.bin           29,043,403     961,155

Against round 14, mainnet instructions fall by 108,713 / 117,636
(0.71% / 0.40%). Each synthetic fixture saves 1,958 instructions. All COPY
calls/bytes/rows and Keccak/ECSM calls are identical to round 14.
Against published round 12 (396378ff0fa319bb920fbb26778292fd332966ba),
mainnet instructions fall by 858,191 / 1,282,769; COPY stays below round 12.
Against original #1014 (21,217,647 / 39,446,929 instructions), the cumulative
reduction is 28.02% / 26.37%. These are executed-instruction measurements,
not GPU proving-time measurements, and cover six fixtures rather than all blocks.

Reproduce from a checkout of this branch
---------------------------------------
Prerequisites: Python 3.11+, Git, Rust 1.94.0, nightly-2026-02-01 with rust-src,
LLVM clang/llvm-ar, and the LambdaVM RV64IM sysroot. For example:

  rustup toolchain install 1.94.0
  rustup toolchain install nightly-2026-02-01 --component rust-src
  make prepare-sysroot SYSROOT_DIR="$HOME/.lambda-vm-sysroot"

Set LLVM_BIN to the directory containing clang and llvm-ar when they are not
in the default locations. Set SYSROOT_DIR to your LambdaVM sysroot directory.
The repository Makefile pins the sysroot archive checksum.

With dependencies already cached, from the repository root:

  python3 tooling/guest-review/verify.py

To permit Cargo dependency downloads on a fresh machine:

  GUEST_REVIEW_ONLINE=1 python3 tooling/guest-review/verify.py

An optional --host-target /path/to/cache reuses host compilation dependencies.
The guest target directory is guest-target/ in this checkout; a new clone has
a fresh guest target. Build flags remain RV64IM, nightly-2026-02-01, fat LTO,
one codegen unit and the same target JSON/.cargo config as the measurements.

verify.py builds the actual checked-out guest, not a separately patched copy.
It verifies source/fixture hashes, all six exact instruction counts, all 160
public-output bytes against original native ethrex, and COPY/Keccak/ECSM counts.
It then runs the 35 native differential tests, four Keccak tests, regenerates
13,344 golden digests from original ethrex, executes RISC-V and memcmp checks,
and requires rejection of four invalid-input variants on each mainnet fixture.
The RISC-V validation guest must emit R15CHECK after its assertions complete.
The added checkpoint test replays 36,864 operations against unmodified ethrex
across Istanbul, Cancun and Osaka, checking nested commit/revert, access sets,
transient storage, refunds, selfdestruct/created sets and log order.
The branch-deserialization test checks 1,536 mixed nested archives against
original ethrex with three pool capacities, preserving shared Arc identity,
empty derived caches, all inline lengths (0..31), and serialized bytes.
Existing fault-injection tests check partial child cleanup on errors/unwind.
New scanner tests compare 49,152 opcode/distance/target cases plus 4,096
random-window histories with original ethrex before materializing the target
list. The RISC-V guest adds 1,440 boundary cases, including unaligned windows,
PUSH payloads, repeated queries, clones, and later full materialization.

Generated logs, ELF files and results.json are in tooling/guest-review/out/;
build outputs are ignored by Git. expected/ contains the previously measured
counts, output bytes, input/source hashes and original round-12 cumulative patches plus round-13, round-14 and round-15 incremental patches for
comparison. Do not apply those patches again to this branch.
The completed branch validation is recorded in validation/RESULTS.txt, with
machine-readable results, logs and checksums alongside it.

The validation harness and native oracle intentionally depend on the original
ethrex git revision alongside the optimized vendored crates. The production
guest and provider both use only the optimized vendored ethrex dependency tree.
