# RATE-4 leaf widening — resume note

**Branch** `rate4-leaf` (worktree `/Users/maurofab/workspace/lambda_vm-rate4`), based on
`blake3-real-hash` @ `681b749c`. Signed, unpushed, **working tree clean**. This is a
complete milestone, not a checkpoint of half-done work.

| commit | what |
|---|---|
| `75d3162e` | the construction: socket + chips + instr + hash trait + callers |
| `cbf834ff` | registry re-bless |
| `85473426` | KAT re-pin + the generator |
| `240a308c` | rider: derive the socket-vs-standalone cost figures |
| `6669c997` | this note |
| `8312bf58` | the H6 gate controls, and the `m[8]` doc corrections |

## H-register — all nine done and verified

| id | disposition |
|---|---|
| H1 | framing indices derived from `NUM_LANES` (`LANE_IDX`/`OUT_PIN_IDX`/`DIGEST_IDX`/`UNREAD_IDX`). Guard `every_hash_candidate_emits_each_constraint_index_exactly_once` **green**. Confirmed count-preserving: lanes +4, unread pins −4, `NUM_CONSTRAINTS` unmoved — exactly the silent shape H1 predicted. |
| H2 | `emit_unread_input_pins` skips a slot every mode reads; `NUM_UNREAD_INPUT_PINS` derived, 8 → 4. |
| H3 | 2nd `LfmMem` receive is `is_real()` (was `Sum3` excluding `MODE_L`). |
| H4 | `leaf_lo_lane`/`leaf_hi_lane` = `4 + 2i` / `4 + 2i + 1`; felt source is `cols::leaf_felt(i)`. |
| H5 | `lanes_from_cells` is the single hybrid split; trace filler and BITWISE histogram both call it. |
| H6 | lanes 0–3 gated on full `mu`, lanes 4–11 on `digest_mu`. |
| H7 | `admits` checks `lanes_of(acc)` **and** `leaf_lanes(felts)`, in the cells the AIR reads. |
| H8 | `LfmHasher::leaf(acc, felts)` on all three arms; Test/Poseidon default `compress_out(acc, felts)`. |
| H9 | `block_len` derived as `4*(NUM_LANES+1)`; all three domains re-pinned. |

No tenth hazard found.

## ★ Two things a reviewer must look at

1. **Message layout differs from the task brief, follows the spec.** COMMIT.md §1.2 and
   `commit_ref.py::lfml_chain_row` put the tag LAST: lanes at `m[0..12]`, tag at `m[12]`.
   The brief (and §1.4.4 H6's parenthetical "m[9..13]") assumed the tag stays at `m[8]`.
   H6's substantive argument is unaffected — the free pin comes from the lane→**column**
   map (`IN0 + lane` landing on the third input cell), not from the message index.
   **§1.4.4's aside needs a doc fix.**
2. **Registry drift is narrower than the brief predicted.** Only `FriToyV0` moved. The
   brief expected all six `program_id`s to move; they don't, because the registry is
   generated under `HasherKind::Test` and `program_id` is `f(roots, log_heights, chunks,
   hasher)` — leaf hash *semantics* never enter it.

## State

* `lfm::` suite **310 passed / 19 failed**; the 19 are byte-identical to the
  `blake3-real-hash` baseline (measured in that worktree: 307/19). **Zero new
  failures**; the +3 are the tests this branch adds.
* Whole prover crate: **861 passed / 34 failed** = the 19 above plus 15 in
  `tests::prove_elfs_tests` / `tests::recursion_*`, every one of which panics
  with "run `make compile-programs-rust`" or "run `make compile-recursion-elfs`".
  Nothing outside `prover/src/lfm/` references the changed code — the only
  consumer is `bin/compute_lfm_registry.rs`.
* `FriToyV0` proves and verifies under BLAKE3 and under every hasher.
* `make lint` and `make fmt`: exit 0.
* Fresh worktrees need `make compile-programs-asm` (and the two above for the
  full crate) before the suite means anything.

## Projection (calibrated model, `lfm_census_2026-08-12/tower.py`)

Gate D1 node — fixture wrap, 1 proof, 110 queries: **124 → 78 GiB**, against the
~93 GiB budget, so it **FITS**. §1.4.1 predicted ≈81. Priced at the socket's real
width rather than the standalone chip's (§1.4.3), 119 → 75 GiB.

⚠ Two honesty caveats, both of which make the headline *less* good than it looks:

1. **The realized factor is 1.60–1.65×, not 2.0×.** The 2.0× is on leaf
   absorption alone (~75% of this node); Merkle parents and the FRI legs do not
   move. §1.4.1's ≈81 GiB already accounts for this — its "2.0× cut in ~70% of
   the cost" needs the reader to do the Amdahl step, and several downstream notes
   quote the 2.0× as if it were the node factor.
2. **78 GiB is a one-proof-VERIFY node, not the smallest aggregating one.** The
   arity-2 node is 155 GiB (fixture) / 232 GiB (real 2^21) and does not fit. The
   campaign notes already flag the aggregating node as the binding memory
   constraint, and that `tower.py`'s flat 6.5% non-hash residue is optimistic at
   higher rates — the residue tracks felts absorbed, so it does not fall with the
   compression count.

## Regenerating

```
python3 thoughts/shared/lfm-real-hash/leaf-spec/rate4_kat_gen.py           # re-pin KATs
python3 thoughts/shared/lfm-real-hash/leaf-spec/rate4_kat_gen.py --check   # staleness gate
cargo run --bin compute_lfm_registry --release                            # re-bless registry
```
