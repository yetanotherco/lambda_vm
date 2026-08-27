# HASH PINNING — `hash-blake3`

One of four sibling branches (`hash-blake3`, `hash-poseidon`, `hash-rpo`,
`hash-rpx`), all cut from the same shared base so the comparison is
like-for-like. `blake3-full-mmcs` remains the campaign trunk and is not one of
these four.

**Shared base: `25855f70`** (`feat(lfm): the leaf call sites stop serialising
felts on the algebraic path`). This supersedes the first cut at `73ee2a64`; the
base moved because items A, C and the front of B landed under all four branches
and a comparison across two different bases is not a comparison.

**This branch's hash: BLAKE3-6r** — the incumbent, and the only one of the four
with a measured block record.

## ★ Why this branch runs FIRST, and what its run is actually testing

It is not one of four data points. **It is the regression harness**, and it is
the only thing that can answer whether the shared base moved BLAKE3.

Between the block record and `25855f70` the shared base grew a shape-carrying
`WrapDigest`, three new `CommitmentHash` variants, an algebraic Merkle backend
family, an algebraic Fiat–Shamir transcript, an algebraic grinding hash, and the
first two of three leaf call sites de-serialised — twelve commits across twelve
files. Every one of them is behaviour-preserving *by test*, and the suite agrees.
**A full-suite green is not the acceptance bar.** The bar is the measured block:

| | record |
|---|---|
| wall | **104.2 min** |
| peak RSS | **358.2 GiB** |
| block proof | **36.9 MB** (38,675,976 bytes) |

★ Kept on separate lines deliberately: time and memory are separate verdicts and
are never averaged into one.

⚠ **The comparison is on those three numbers plus a passing verify and consumer
ritual — NOT on proof bytes.** Grinding draws a nonce non-deterministically, so
the proof BYTES do not reproduce run to run even at a fixed commit; the roots do.
A byte-compare against the record's proof would fail on a correct run.

⚖ ASSESSMENT: this should be close to a no-op, and that is what makes it a clean
test of the shared base rather than of the pin — see the evidence below. If it is
*not* a no-op, learning that before the algebraic three are built on top of this
base is worth a box night.

## The pin, mechanically — ONE type alias drives all four planes

✓ VERIFIED by reading each definition:

```
crypto/stark/src/config.rs
  DefaultStarkHash = Blake3StarkHash          <- THE PIN
    -> COMMITMENT_HASH = <DefaultStarkHash as StarkHash>::COMMITMENT_HASH
       -> edsl::WrapHash::production()        (prover/src/lfm/edsl.rs)
       -> statement::commitment_hash_tag      (program identity)
       -> DefaultStarkTranscript<F>           (Fiat-Shamir follows the commitment)
```

Enforcement is already in-tree and needs nothing added on this branch:
`const _: () = assert!(matches!(COMMITMENT_HASH, CommitmentHash::Blake3));`
(`crypto/stark/src/config.rs:456`), plus the `blake3-6round` lockstep assertion
immediately below it. **Re-pointing the alias without moving the blessed
artifacts is a compile error, not a wrong number.** That is why this branch's pin
commit touches no code: the mechanism it would add already exists, and a second
copy of it would be noise.

⚠ There is a `cuda` fork above the pin: a `cuda` build commits keccak, not
BLAKE3. The block runs are CPU builds, so this does not bite — but a GPU arm of
this comparison would be measuring a different hash, and the assertion above is
`#[cfg(not(feature = "cuda"))]` accordingly.

## Evidence the shared base did not move BLAKE3

| check | result | marker |
|---|---|---|
| `lfm::` suite at `25855f70`, release | 484 passed / 1 failed | ✓ VERIFIED |
| — the one failure | `lfm::epoch_tests::the_closure_rejects_a_moved_index_or_output`, the standing pre-existing exoneration | ✓ VERIFIED |
| registry drift, all six programs + the miss test | 7 passed / 0 failed, 2026-08-27 | ✓ VERIFIED |
| `commitment_hash_tag` | existing tags unchanged; the three algebraic variants took 2/3/4, additive | ✓ VERIFIED |
| `COMMITMENT_HASH` const assertion | unchanged, still `Blake3` | ✓ VERIFIED |
| BLAKE3's Merkle/transcript path | unedited — the algebraic work is SIBLING types, not a reparameterisation | ✓ VERIFIED |

⚠ None of that is the block. That is the point of the run.

## ⚠ The other three branches CANNOT run a block at this base

Stated plainly so nobody schedules a run that would silently measure BLAKE3:

✓ VERIFIED — there is **no `impl StarkHash` for any algebraic member** anywhere
in the workspace (`RpoStarkHash`, `RpxStarkHash`, `PoseidonStarkHash`,
`AlgebraicStarkHash`: zero matches). The *parts* all exist — `AlgebraicBatchBackend`,
`AlgebraicPairBackend`, `RpoTranscriptHash`, `CommitmentHash::Rpo256` — but the
configuration struct that ties them into something `DefaultStarkHash` can name
does not. Two further pieces are also open: `transcript_replay.rs` still models
the transcript as byte segments with no algebraic arm, and
`edsl::algebraic_byte_hash_unreachable()` is still a live interim panic.

So a block produced on `hash-rpo` today would be committed under **BLAKE3**,
regardless of the branch name. Those three branches get their pin commits when
those pieces land; this file will say so on each of them.

## THE RECIPE — how to run this branch's block

**Commit:** `hash-blake3` (this commit). One commit above shared base `25855f70`,
and it is documentation only — the built binary is byte-identical to the base's.

**Test:** `lfm::aggregator_tests::the_real_block_aggregates_end_to_end`

```sh
export LFM_CENSUS_ELF=/root/ethrex.elf
export LFM_CENSUS_INPUT=/root/ethrex_mainnet_25368371.bin   # 1,110,156 B, sha 61eba49b
export LFM_CENSUS_EPOCH_LOG2=24
export LAMBDA_VM_MAX_ROWS_LOG2=24
export P3_AGG_TERMINAL_BLOWUP=2
export P3_AGG_RESIDENCY=recompute
export P3_ARTIFACT_DIR=/root/p3_blake3_pin_artifacts     # ★ FRESH — see below

cargo test --release -p lambda-vm-prover --lib \
  lfm::aggregator_tests::the_real_block_aggregates_end_to_end \
  -- --ignored --exact --nocapture --test-threads=1
```

★ **`P3_ARTIFACT_DIR` must be a FRESH directory** — not the record's
`/root/p3_record_artifacts`, which now holds that run's cached bundle and wraps.
The driver *loads* them when it finds them, which would skip the base and wrap
proves entirely — precisely the legs this run exists to re-measure. This is the
record's own methodology, not an added rule: its script is headed *"P3 RECORD RUN
— one box, one process, NO CACHE, whole pipeline"*, and the same driver against a
warm cache is what produced the 49.3 min / 336.8 GiB probe, which is a different
measurement and not the bar. A fresh directory still persists the artifacts, so
an aggregation OOM does not cost the first 60 minutes again.

The two `P3_AGG_*` values are the record's, not defaults, and both are load-bearing:
- `P3_AGG_TERMINAL_BLOWUP=2` — blowup 4 OOM-killed the box three straight times.
  The query count re-derives from the same 128-bit Johnson target by construction
  (`with_blowup`), so 2 → 219 q; the FRI terminal stays at the preset's fp8.
- `P3_AGG_RESIDENCY=recompute` — trades ~2× aggregation prove time for the LDE
  peak. `Retain` at real scale OOM-killed a 483 GiB box.

Expected banner if the pin is right (the driver prints it first):
`inner blowup 4 / 110 q, wrap+aggregation blowup 4 / 110 q / fp8`, then
`aggregation TERMINAL options: blowup 2 / 219 q / fp8`.

⚠ Idle guard first: the box's own noise study puts CV at 0.06–0.25% idle but
≥2% on a single contended pair, and this is a one-shot ~1.75 h measurement.

## Comparison discipline

⚠ **The four branches are NOT four equally-grounded numbers.** BLAKE3 has a
measured block record; the other three have projections from measured chips and a
measured census. Every table must keep marking which cells are measured and which
are extrapolated, and must keep memory and time on separate lines.

Measured cells per compression, one instrument, no flags:
**RPX 325 · RPO 445 · Poseidon 621 · BLAKE3 4,946.**

Projected peak RSS is **affine**, not proportional, and must be quoted with its
unmeasured fixed term `C` named: **RPX 69.3 + 0.794·C**, **RPO 76.2 + 0.774·C**
GiB. ⚠ Counterweight, measured: BLAKE3 is the **cheapest hash per cell in time**
(39.7 ns/cell vs RPO 46.1, Poseidon 47.9) — the algebraic case is a MEMORY win
that reduces time by reducing cells, not a per-cell speedup.

Poseidon is **UNSHIPPABLE** (broken family, eprint 2026/306 and 2026/1692) and
appears only because the numbers were asked for.

Full working: `thoughts/shared/block-compression/RPO-LANE.md` and
`HASH-SWAP-DESIGN.md` on `rpo-migration`.
