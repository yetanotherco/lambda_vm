# Transcription audit — does the gate's model match the code it models?

Executable half: [`audit_transcription.py`](audit_transcription.py). Output is in the run
transcript [`gate.log`](gate.log).

## Why this file exists

Both earlier campaigns in this tree converged on the same conclusion about where the residual
risk sits after a green board.

`thoughts/blake3/README.md` (branch `feat/blake3-accelerator`), "Still unaudited — where to
send the next reviewer":

> Two independent reviews established that **the oracle defines the right function**, so the
> gate's UNSATs are about the right function. They did *not* audit the step after that:
> **nobody has checked the z3 gate's transcription of the oracle into constraints.** […] The
> dangerous direction is a model **stronger** than the thing it models — it yields UNSAT where
> the real object is forgeable, and a positive anchor cannot catch it, because honest inputs
> satisfy a correct model and an over-strong one equally well.

And it points at the EC campaign as the reason to take it seriously: the equivalent audit
there (`thoughts/ec-recover-opt/gate/TRANSCRIPTION-AUDIT.md`, branch `feat/ec-lincomb2`)
"found three premises the gate
asserted about the chip and never read, one of them hiding a working forgery".

So every premise the A1–A4 lemmas rely on is enumerated here as a `Premise` and **read out of
the source**, not re-derived from memory.

## The two kinds of premise

**assumed** — the model relies on something being true of the code. Failing one invalidates
whatever lemma consumes it.

**negative-space** — the model relies on something being **absent**. These are the dangerous
ones, because a reader checking the diff sees only what *is* written. Two of them here, and
both carry real weight:

- **P16 — nothing constrains `yG`'s parity.** This is A3's central premise. If any constraint
  did pin the parity, A3b's forgery would be blocked by it and A3d's "the `yG` read is
  load-bearing" verdict would be *wrong* — the gate would be claiming a fix is necessary when
  something else already covered it. Checked by enumerating **every** appearance of `cols::YG`
  in `ecsm.rs` (there are 7) and confirming each is one of five known parity-blind uses: the
  trace fill, the affine MEMW read (the fix itself), the `AreBytes` range check, the
  `Ecdas` seed/drain tuples, and the `Yg` relation's `yG²` term. A *new* appearance fails the
  audit, which is the point.
- **P17 — `YR` is not byte-checked in `ecsm.rs`.** This is contract C4-YR. `YrLtP` reads
  `YR`'s 32 columns as bytes, but the local `is_byte` list is `{X2, Q0, YG, Q1}`. So the byte
  bound is *inherited* through the `Ecdas` bus rather than emitted, and the gate is consuming
  a contract it must not silently assume. If a future commit adds `is_byte(cols::YR, 32, …)`
  the situation improves — and the audit should still notice, because RESULTS.md's contract
  list would then be stale.

## Mutation testing — a blind check is worse than a missing one

Every premise is perturbed: the source is mutated in memory and the premise's check must then
**fail**. A check that passes on mutated source is checking nothing, and reads as green.

This is not hypothetical. **P18 (instruction timestamp stride) shipped blind in its first
form.** It matched `ecsm.rs`'s comment —

```
// ts + 3 is the free 4th sub-timestamp (instruction stride is 4; xG@T, k@T+1, xR@T+2 …)
```

— and compared it against a hard-coded `4` in the model. It passed. The mutation control then
reduced the *real* stride in `trace_builder.rs` from 4 to 3, and the premise still passed: it
was checking documentation, not behaviour. It now parses the stride out of
`let timestamp = (i as u64) * 4 + 4;` and compares the **parsed** value, and the mutant is
caught.

Current state: **19/19 premises read from source, 19/19 mutations bite, 0 failures.**

## The premise table

| # | Lemma | Premise | Source | Mutation control |
|---|---|---|---|---|
| P1 | A1/A2 | `MU=666, IS_AFFINE=667, YR_SUB_P=668..684, NUM_COLUMNS=684` — and `YR_SUB_P + 16 == NUM_COLUMNS`, so the new halfwords fit exactly with no overlap | `ecsm.rs` `mod cols` | `NUM_COLUMNS → 683` |
| P2 | A1 | `debug_assert_eq!(idx, 423)`, and the header index map documents `413..420` / `421` / `422` | `ecsm.rs` | `idx → 421` |
| P3 | A4 | `ADDR_LIMB_BOUND_32B = 2^32−31`, `..._64B = 2^32−63` | `ecsm.rs:43,47` | `64B → 2^32−64` |
| P4 | A1c | `ECSM_SYSCALL_NUMBER = u64::MAX−10`, affine `= u64::MAX−11` | `execution.rs:38,47` | affine `→ u64::MAX−12` |
| P5 | A1c | the low-32-bit-word inequality is a **compile-time** assert | `execution.rs:53-58` | assert renamed away |
| P6 | A1c | the received syscall words are `xonly + IS_AFFINE·(affine − xonly)`, per word | `ecsm.rs` `syscall_word` | `coefficient: affine - xonly → 0` |
| P7 | A1/A3 | 2 `IS_AFFINE`-gated bus blocks of 4 dwords, offsets `+32 + 8i`, `yG` via `memw_read` at `ts`, `yR` via `memw_write` at `ts+3` | `ecsm.rs` | offset `+32` dropped |
| P8 | A2 | `YrLtP → (P_BYTES, YR_SUB_P, YR)`, byte-stored sum (only `KLtN` is bit-stored) | `ecsm.rs` `OverflowKind` | sum column `YR → XR` |
| P9 | A2d | all four chains share ONE `µ`-gated loop; `IS_AFFINE` does not appear inside it | `ecsm.rs` `eval` | carry bits re-gated on `IS_AFFINE` |
| P10 | A2 | 16 `µ`-gated `IsHalfword` sends on `yr_sub_p(i)` | `ecsm.rs` | sends aimed at `xr_sub_p` |
| P11 | A4 | 3 `Alu` LT senders: `xG`/`xR` vs `addr_bound_by_mode()`, `k` vs the flat bound, `LT`/result 1 | `ecsm.rs` | `xR`'s sender deleted |
| P12 | A4 | executor arm: spans `63/63/31`, the `u128` overlap guard, `yG` read at `+32`, `yR` stored at `+32` | `execution.rs` | `u128 → u64` (the pre-fix wrapping form) |
| P13 | A2e | `y_r_sub_p = (2^256 + yR − p)`, computed in the shared `compute_witness_inner` used by BOTH entry points | `witness.rs` | addend built from `x` instead of `y` |
| P14 | A3 | `CARRY_OFFSET_X2 = 8160`, `CARRY_OFFSET_YG = 16319` | `ecsm.rs:37,38` | `YG → 16320` |
| P15 | A2a | `INV_SHIFT_32 = 18446744065119617026 = 2^{−32} mod p_g` | `templates.rs:26` | last digit perturbed |
| **P16** | **A3** | **negative space:** every `cols::YG` use is parity-blind | `ecsm.rs`, all 7 sites | an unrecognised `cols::YG` use appears |
| **P17** | **A2d** | **negative space:** `YR` is not in `ecsm.rs`'s `is_byte` list ⇒ C4-YR is inherited | `ecsm.rs` | `YR` gains a local byte check |
| **P19** | **A1f** | **the COMPLETE syscall set** — every `u64::MAX − k` the `Ecall` bus can carry, parsed from source. A1f's conclusion is about which foreign syscalls the linear syscall word reaches, so a fifth syscall changes the answer | `execution.rs` | `HINT` renumbered `MAX-30 → MAX-40` |
| P18 | A4f | one instruction consumes 4 sub-timestamps; ECSM uses offsets `{0,1,2,3}`, max `== stride−1` | `trace_builder.rs:348`, `ecsm.rs` `ts_lo_plus` | stride `4 → 3` |

## What the audit does NOT cover

- **Bus wiring and lookup coverage.** Same limitation both earlier gates recorded about
  themselves (`thoughts/blake3/blake3-chip/DESIGN.md` §7 items 4/5/11, branch
  `feat/blake3-accelerator`). P7/P10/P11 check that
  the interactions are *emitted with the shape the model assumes*; they cannot check that the
  MEMW/AreBytes/IsHalfword **receivers** interpret those tuples the way the model believes.
  That is contracts C1/C2/C5 plus the e2e prove+verify tests.
- **The verifier's column-width pinning.** `NUM_COLUMNS` growing 667 → 684 is a proof-format
  change. P1 reads the constant; whether the verifier's per-table width check tracks it is
  `main`'s `#909` territory, not this campaign's.
- **Regex fragility.** These checks are textual. A refactor that preserves behaviour but
  changes formatting will fail a premise — a false alarm, but a loud and cheap one, which is
  the right failure direction for an audit. Every failure message names the premise and prints
  what it found.
