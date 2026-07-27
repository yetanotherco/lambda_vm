# lincomb2 layout lock (phase A output)

The data contract the chip / executor / guest phases build against, derived from
the **implemented and host-validated** `ecsm::lincomb2_witness`
(`crypto/ecsm/src/witness.rs`) — not from estimates. Validated by
`crypto/ecsm/src/tests/lincomb2_tests.rs`: Q matches an independent `Fp`-reference
lincomb AND k256 over 512 random + edge cases; every emitted row re-satisfies its
double/add relation and slope; the NUMS blind provably cancels; T₀ reproducible.

## Locked row count (replaces DESIGN's ±5% estimate)

Per ecrecover, ONE joint chain (today: four single-scalar chains):

| | rows |
|---|---|
| P12 precompute | 1 |
| doublings | = `len` = `max(msb u1, msb u2)+1`; mean **255.7**, max 256 |
| adds (nonzero joint digits) | mean **192.0** |
| correction | 1 |
| **total** | **mean 449.1 (Rust) / 449.7 (Python), max 471** |

DESIGN estimated 448 → confirmed (+0.2%). Cost model verdict (−74.3% EC cells
with pairing) stands on the measured count.

## Chain row schedule (what the chip telescopes)

Dense-doubling blinded Shamir/Straus, MSB-first. `acc` seeded = T₀:

```
row 0            : PRECOMPUTE   a=P1, addend=P2, op=1  → r=P12     (OFF the acc line)
for round=len-1..0:
  DOUBLE         a=acc, addend=(0,0), op=0            → r=2·acc ; acc=r
  if u1.bit|u2.bit: ADD  a=acc, addend∈{P1,P2,P12}, op=1 → r=acc+addend ; acc=r
row last         : CORRECTION   a=acc, addend=−2^len·T₀, op=1 → r=Q      (strips blind)
```

`sel ∈ {Double, AddP1, AddP2, AddP12, Precompute, Correction}` (witness enum
`JointSel`). Addend sources the chip must supply: **P1, P2, P12** (add rows),
and the **table constant −2^len·T₀** (correction). Precompute's addend is P2.

## ECSM′ column blocks (one row per ecrecover)

Names/widths from `Lincomb2Witness`. Widths: B=byte-checked, HW=U256HL 16
halfwords (IsHalfword), bit=boolean, fe=field carry.

| Block | Cols | Range authority |
|---|---|---|
| ts, addrs | ~8 | MEMW |
| xP1,yP1 | 64 B | P1=G constant for ecrecover (chip may hardcode); else membership |
| xP2,yP2 | 64 B | membership (below) + **yP2<p** |
| mem_P2 (x2,q0,c0,q1,c1) | 32+32+64fe+33+64fe | AreBytes + IsHalfword — same two convolutions ECSM proves for G |
| yP2_sub_p | 16 HW | IsHalfword — **NEW load-bearing** (yP2+p parity-flip forgery) |
| xP12,yP12 | 64 B | proven on-curve by the precompute row's relations |
| u1,u2 | 512 bit | IS_BIT ×512 |
| u1_sub_n,u2_sub_n | 16+16 HW | IsHalfword (u<N) |
| len | 1 | binds the T₀-table index; no exact-MSB lemma needed (blinding) |
| xQ,yQ | 64 B | **xQ<p, yQ<p** (below) |
| xQ_sub_p,yQ_sub_p | 16+16 HW | IsHalfword — **NEW load-bearing** (shifted keccak(pk)) |
| T₀, 2^len·T₀ | 128 B | preprocessed T₀ table (constant, indexed by len) |
| MU | 1 | IS_BIT |

## Joint chain row (ECDAS′) — one per double/add step

Today's ECDAS row (521 cols) with XG/YG → XB/YB (addend, from the Addend bus)
plus the joint selectors. Per-row convolution core (λ, xR, yR — Q0/Q1/Q2 + C0/C1/C2)
is **byte-for-byte the existing machinery** (`carries_lambda/xr/yr` reused
verbatim; L1/L2/L3/L4 of the gate port unchanged).

| Block | Cols | Notes |
|---|---|---|
| ts | 2 | chain key |
| XA,YA | 64 B | accumulator in (from Ecdas bus) |
| XB,YB | 64 B | addend; from **Addend bus** on adds, `(0,0)` on doubles (cancels — see below) |
| round | 1 B | |
| op, s1,s2,s3, d1,d2 | ~6 bit | op=double/add; s* one-hot addend; d1,d2 = this round's two scalar digit bits, carried on **both** the double and the add (two Bit streams) |
| nb | 1 bit | "an add follows me at this round" — pins the successor round `round − 1 + nb`; see correction #2 |
| XR,YR | 64 B | result out |
| LAMBDA | 32 B | |
| Q0,Q1,Q2 | 33×3 B | quotients (reused) |
| C0,C1,C2 | 64×3 fe | carries (reused) |
| MU | 1 | |

≈ **526 logic cols** (525 + `nb`, correction #2). Interactions ≈ 390 → with
AreBytes pairing ≈ 292 → ~964 committed cells/row; DESIGN §2's cost numbers hold
to within 0.1%. `CHIP-LAYOUT.md` §2 carries the full map at 529 cols (adds the
`PHASE`/`NEXT_PHASE` segment split and `S_CORR`) and §5 recomputes the cells.

**Key algebraic fact (verified against the relations, used by the tests):** on a
DOUBLE row the addend `xg`/`yg` cancels out of all three convolution relations
(λ op=0 branch uses neither; xR's `−xg` and the `(1−op)(xg−xa)` term cancel; yR
uses neither). So double rows carry addend `(0,0)` and the chip's Addend bus can
stay silent on doubles. Enables the `Sum3(s1,s2,s3)`-gated Addend receive.

## Syscall ABI (proposed; new number, keep `ecsm_mul` alive)

Little-endian throughout, mirroring ECSM (`execution.rs:424-456`, `syscalls.rs`):
- `a0` = addr to WRITE result: 65 bytes `status(1) ‖ xQ(32) ‖ yQ(32)` — OR status
  in a register and 64 bytes at a0. (Pick: status in a returned register is
  cleaner; TBD in phase B.)
- `a1` = addr of `xP1‖yP1` (64 B), `a2` = addr of `xP2‖yP2` (64 B),
  `a3` = addr of `u1‖u2` (64 B).
- Overlap/alignment guards per `ecsm_addr_ok` style.
- **Status contract**: executor computes the witness in software; on any
  `Lincomb2Error` (u=0, u≥N, P off-curve/non-canonical, P1=±P2, Q=∞) it returns
  `status≠0` and writes nothing meaningful. The guest then falls back to
  `ProjectivePoint::lincomb` (already the `unwrap_or_else` shape at
  `lib.rs:116-117`). **status≠0 is always sound** — the fallback is proven guest
  code; a lying status only wastes cycles. No trap ⇒ no crafted-tx block
  censorship (unlike today's ECSM trap, which is safe only because its inputs
  are guest-guarded).

## Corrections to DESIGN.md found while implementing

1. **Row count** 448 → measured mean 449.1, max 471. Estimate confirmed; the
   ±5% band collapses.
2. **Schedule is dense-doubling** (one double every round, always), not the
   skip-ahead variant DESIGN sketched. The add shares the double's round, and
   the round decrements only at the loop boundary. The two Bit streams
   (mult `d1`, mult `d2`) do the per-scalar counting.

   > **CORRECTED 2026-07-24 — this item previously claimed "no `nb_or` column".
   > That was wrong; DESIGN's helper column is required.** Dense doubling is
   > true, but the conclusion does not follow. Because the double and its
   > optional add share a round, the successor round is
   > `round − 1 + nb` — and on a double row `nb` ("an add follows me") is **not
   > a function of that row's other columns**. Without it `round` can stall, so
   > a prover could insert or drop doublings while every per-row relation still
   > held. It is the exact analogue of `NEXT_OP` in today's single-scalar chain
   > (`prover/src/tables/ecdas.rs:243-253` sends `round - 1 + next_op`).
   >
   > The witness could not supply it either: all four `joint_row` call sites
   > passed `next_op = 0`, and double rows were given `d1 = d2 = 0`.
   > **Fixed in `witness.rs`:** double rows now carry their round's real
   > digits, `JointStep` gained `nb` (mirrored into `EcdasStep::next_op`), and
   > `check_nb_schedule` in `tests/lincomb2_tests.rs` asserts the recurrence
   > over the whole corpus. Purely additive — `next_op` is copied straight
   > through by `build_step` and read by nothing, so no emitted math changed
   > (row stats still mean 449.1 / max 471, `cargo test -p ecsm` 26/26).
   >
   > The chip's two defining constraints, for phase D:
   > `OP · NB = 0` (deg 2) and `(1 − OP)·(NB − D1 − D2 + D1·D2) = 0` (deg 3) —
   > note the op-gating: an add row carries its round's digits (it needs them to
   > select the addend) but `nb = 0`.
   >
   > → the joint row needs **one more** bookkeeping column than this document
   > previously claimed, not two fewer. See `CHIP-LAYOUT.md` §0.1 and §2.
3. **P12 precompute is OFF the accumulator line.** DESIGN mused "seeded from T₀";
   it is not — it is a standalone chord add `P1+P2` (a=P1, g=P2). The chip's
   accumulator telescoping must special-case that the precompute row's `a` is P1
   (not the previous row's result) and the correction row's `a` is the last
   accumulator. This is the one place the clean `a = prev.r` telescoping breaks;
   flag for the L6 redo.
4. **Q=∞ and acc==addend collisions** both surface as `Lincomb2Error::ResultInfinity`
   in the witness → status≠0 → fallback. The blinding makes an interior
   `acc==addend` a discrete-log event (the assumption), so it never fires for
   honest inputs; the witness still guards it defensively.

## What the tests prove (phase-A gate, all green)

`cargo test -p ecsm --lib` → 21/21. New: T₀ on-curve+pinned+reproducible;
lincomb2 Q == two independent references over 512 cases; per-row relation/slope
re-check; blind-cancellation; degenerate-sum and bad-scalar rejection.
