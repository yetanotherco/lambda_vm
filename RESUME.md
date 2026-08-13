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

* `lfm::` suite **308 passed / 19 failed**; the 19 are byte-identical to the
  `blake3-real-hash` baseline (measured: 307/19). Zero new failures.
* `FriToyV0` proves and verifies under BLAKE3 and under every hasher.
* Fresh worktrees need `make compile-programs-asm` before the full prover suite.

## Regenerating

```
python3 thoughts/shared/lfm-real-hash/leaf-spec/rate4_kat_gen.py           # re-pin KATs
python3 thoughts/shared/lfm-real-hash/leaf-spec/rate4_kat_gen.py --check   # staleness gate
cargo run --bin compute_lfm_registry --release                            # re-bless registry
```
