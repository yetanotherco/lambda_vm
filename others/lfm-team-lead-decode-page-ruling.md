# Team-lead ruling: DECODE and PAGE preprocessed commitments (ledger entry 7)

Ruled 2026-08-04, on wave 5's proposal. Subject to USER veto — flagged in the
session report the day it was made. If vetoed, the fallback is the in-machine
derivation and this file must record the reversal.

## The ruling

**ACCEPTED as proposed.** The five preprocessed commitments split by what they
are a function of, and each kind gets the binding that kind admits:

| Commitment | Function of | Binding |
|---|---|---|
| BITWISE | proof options only | INTERN as program constant |
| KECCAK_RC | proof options only | INTERN as program constant |
| REGISTER | previous epoch's `reg_fini` | DERIVE in-machine (Phase A calls `programs::emit_register_commitment` on the register-boundary arena — this is also what closes entry 2) |
| DECODE | the inner ELF | ARENA CELL, bound by the attestation join |
| PAGE | the inner ELF | ARENA CELL, bound by the attestation join |

"Attestation join" means: the SAME arena cell Phase A absorbs is the cell the
`program_id` fold consumes — the two-consumer join one level up, for which the
machine already has an emitter
(`machine_tests::program_id_folds_pages_in_the_production_layout`).

## Why (three legs, none of them new judgment)

1. **The alternative trips the always-stop item; the proposal doesn't.**
   Interning an ELF-dependent root makes LFM program identity a function of
   the guest ELF — one registry entry per guest program instead of one per
   epoch shape. That is the exact clause on the standing always-stop list,
   and it contradicts the phase's pin-SHAPE-not-values rule (nothing derived
   from per-proof data may be a program constant).
2. **It mirrors production's own layering.** `recursion::program_id_from_digest`
   folds `elf_digest`, `pc_start`, `decode_commitment` and every
   `(page_base, page_commitment)` — precisely the ELF-dependent roots and none
   of the options-only ones. Production already draws the line this ruling
   draws; LFM is copying an existing boundary, not inventing one.
3. **The in-machine derivation relitigates a measured decision.** Deriving
   DECODE/PAGE from ELF bytes costs a full in-machine LDE+tree per page and
   requires the ELF itself to be bound in-guest — the full-ELF keccak pass
   that sim/8 (`program_id` v2) deliberately removed, with the savings
   measured. Re-adding it needs new evidence, not a default.

## The residual risk, named plainly

`program_id`'s binding is only as strong as the consumer-side
`check_attestation` compare, which has ZERO production call sites (RESUME,
"Open items needing the USER"; PoC at
`prover/src/tests/recursion_soundness_gap_poc.rs`). This ruling makes
DECODE/PAGE **exactly as bound as `elf_digest` and `pc_start` already are — and
no more**. It adds no new weakness, but it does add two more values whose
ultimate binding rests on a ritual nothing in production performs. The
check_attestation gap therefore gets MORE load-bearing with this ruling, and
the case for the user deciding to wire it into the CLI gets stronger. That
decision stays with the user; it is not part of this ruling.

## Conditions attached (wave 6 must satisfy both)

1. **The join must be structural, not a copy** — one cell with two consumers,
   per the two-consumer rule that closed three soundness gaps this phase. A
   guard must assert it, and the guard must be falsified (run the split-cell
   forgery and watch it fail for the right reason).
2. ~~**PAGE's half is UNWITNESSED in the current fixture**
   (`num_private_input_pages = 0` — a fixture property, not a production one).
   The witness is a differently-configured real epoch (a guest with private
   input pages). Wave 6 should build that epoch and run the assembled
   verifier against it if it is cheap; if it is not cheap, the ledger keeps an
   explicit OPEN entry saying PAGE's join is design-complete but unwitnessed.~~

## Amendment (2026-08-04, after wave 6): condition (b) REVERSED

Condition (b) asked for a witness that CANNOT EXIST, and the premise behind it
("fixture property, not a production one") was wrong — wave 6 established this
by reading, three ways:

1. Private-input pages are built NON-preprocessed (`lib.rs:800-828`), so a
   guest with private input pages could never witness a PAGE preprocessed root.
2. **No continuation epoch of any guest has a PAGE sub-proof.** `prove_epoch`
   rejects page configs outright — "continuation epoch must have no PAGE
   configs (L2G bookend replaces PAGE)" (`continuation.rs:695-702`) — and both
   `build_epoch_airs` call sites pass `page_configs = &[]`. The fixture was
   matching production, not stripped down.
3. The ELF-data page roots the attestation folds are the GLOBAL proof's
   GlobalMemory AIR commitments (`continuation.rs:997-1010`), per
   `program_id_from_digest`'s own doc.

Consequence: **PAGE migrates out of the epoch verifier's scope** to the
global-proof verifier rather than closing here. The provenance classifier
panics on any preprocessed root it cannot attribute, which is the intended
handover to that future work.

Taxonomy correction to the table above: PAGE's ZERO-INIT root is options-only
(`page::zero_init_preprocessed_commitment`) and belongs in the CONSTANT
family; only the ELF-DATA page roots are ELF-dependent, and those live in the
global proof. The DECODE half of the ruling stands unchanged, conditions
satisfied (structural join + coherent-forgery falsification, wave 6 @
2c810857).
