# BLAKE3 chip — implementation notes (syscall variant)

Companion to `DESIGN.md`: what the Rust implementation
(`prover/src/tables/blake3.rs` + the executor syscall) does differently from
the internal-variant design, and why. The design's §7 soundness ledger is
reproduced in the chip's module docs with per-item dispositions.

## Variant

DESIGN.md §1.1 designs the **lean internal interface** (a `Blake3` bus with a
parent-node caller). No in-circuit caller exists yet, so what is built is the
**§1.2 general syscall variant**: `Ecall` receiver + `Memw` register read of
x10 + per-dword `Memw` I/O, copied idiom-for-idiom from `keccak.rs`. The
internal bus — and with it §7 item 10 (TIMESTAMP binding in both bus tuples) —
does not exist in this variant: a row's inputs and outputs are tied by being
committed on the same row.

ABI: `x10` → 8-aligned 176-byte region, `h[32] | m[64] | t[8] |
block_len,flags[8] | out[64]` (see `BLAKE3_SYSCALL_NUMBER` docs). Syscall
number `u64::MAX - 2`.

## Deltas from the design's cell accounting

| item | DESIGN §3 | implemented | why |
|---|---|---|---|
| add2 carries | 1 committed bit each (§3 table; 6 bits/G) | **expression carry, no cell** (4 bits/G) | §4.3's own formula is the `emit_add_pair` expression form; the §3 table double-counts it. Saves 96 cells/row. |
| G block | 62 cells | **60 cells** (56 bytes + 4 bits) | above |
| I/O apparatus | none (internal variant) | +8 addr bytes, +88 ptr halfwords, +64 OLD_OUT | syscall variant |
| OLD_OUT | n/a | 64 committed bytes + 32 AreBytes | the 8 out-dword `Memw` ops need the previous memory content in their `old` field; those bytes ride only the Memw bus, so they get explicit byte checks (same aliasing argument as keccak.rs's addr bytes) |
| columns | ≈3,155 | **3,219** | |
| sends | ≈1,250 | **1,397** (832 XOR + 384 shift-AreBytes + 32 m + 32 old_out + 4 addr + 1 AND + 88 IS_HALF + 24 I/O) | |
| aux (3·⌈N/2⌉) | ≈1,875 | **2,097** | |
| **cell-equiv/compression** | ≈5,030 | **≈5,316** | +5.7% for the syscall I/O |

Against keccak-f post-#889 (72,672 cell-equiv): **≈ 1/13.7 per call**, ~6.4×
per byte (64 B vs 136 B absorbed).

## The single-dataflow rule

The compression dataflow exists once (`run_flow`), interpreted twice:
`WireFlow` (columns → constraints + bus senders) and `ValueFlow` (u32 →
trace filling + BITWISE multiplicities). Wiring divergence between prover
cells, senders and receiver multiplicities is therefore impossible by
construction; only interpretation bugs remain, and those are what the oracle
vectors + the e2e bus-balance gate check.

## Gates run

- executor ↔ oracle: the 10 pinned canonical 6-round vectors
  (`canonical_6round_vectors.json`, full-width `t` values — the counter-split
  order is load-bearing) + syscall-level tests (alignment/overflow rejection,
  input-region non-clobbering).
- `ValueFlow` ↔ executor: differential unit test.
- wire audit: every committed mixing cell written exactly once, in-range
  (unit test `wire_flow_counts`).
- e2e: `test_prove_elfs_blake3` — two chained compressions (the second
  consumes the first's output and overwrites a non-zero out region), prove +
  verify, which exercises bus balance across Ecall/Memw/ByteAlu/AreBytes/
  IsHalfword.
- the z3 gate (`z3_blake_verify.py`) proves the *design*; the transcription
  design → Rust is covered by the vectors + e2e, per the gate's own
  documentation of what it cannot see (§7 items 4, 5, 11).

## Known costs and open items

- **Always-on AIR**: `FIXED_TABLE_COUNT` 10 → 11. Every proof now carries a
  ≥4-row BLAKE3 table (~3.2k cols) even when unused. This is exactly the
  EC-campaign regression shape (PR #871, +3 near-empty AIRs → +25%); one
  near-empty table is far smaller, but a real-block ABBA is REQUIRED before
  merge.
- The proof wire format changes (one more sub-proof); old proofs do not
  verify against this branch. The recursion guest would need a rebuild
  (the in-repo recursion PoC is already non-functional, see project notes).
- `count_table_lengths` (disk-spill sizing) does not count the 23 Memw ops a
  blake3 ecall contributes; disk-spill runs of blake3-heavy workloads would
  size MEMW slightly small. Not exercised by the bench (no disk-spill).
- 7-round variant: `BLAKE3_ROUNDS` is the single knob; columns/constraints/
  sends all derive from it. Standard-BLAKE3 compatibility would also need the
  flags/t plumbed per the tree mode (out of scope here).

## The 6-round assumption (sign-off record)

The chip implements 6 rounds, not the standard 7. The z3 gate proves the chip
matches the 6-round reference; it does not and cannot prove 6 rounds are
collision-resistant. Adopting this for Merkle/Fiat–Shamir rests on the named
assumption:

> **A6R**: the BLAKE3 compression function restricted to 6 rounds is
> collision-resistant and suitable as a 2-to-1 compression for Merkle
> hashing and as a PRF for Fiat–Shamir, in the same sense the full 7-round
> function is believed to be (precedent: KangarooTwelve's reduced-round
> Keccak).

Directed for implementation by the project owner, 2026-08-05 ("trust me"
sign-off in session). Recorded in the spec (`spec/blake3.typ`, A6R section).

**External review (2026-08-06, relayed by the project owner):** the round
count was reviewed with external symmetric-cryptography experts — removing
one round (7→6) judged comfortable, removing two (7→5) explicitly not.
6 rounds is therefore the endorsed floor. Sub-6 variants are not formally
dead, but they cannot be adopted on the project's own authority — that
would need the experts to sit with the reduced margin specifically
(dedicated cryptanalytic review, not an engineering call). The 7-round
instantiation remains available as the zero-assumption / interop fallback
at ~10-12% more per merge.
