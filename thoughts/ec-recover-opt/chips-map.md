# ECSM / ECDAS chip map — z3 gate input + cost census

> **UPDATE (feat/ec-arebytes-pairing):** candidate A1/A2 below LANDED — the
> single-byte AreBytes sends are now paired (ECDAS 388→290, ECSM 579→515
> interactions/row, ≈ −13.3% ECDAS committed cells). The census tables below
> describe the PRE-pairing layout the gate verified; see
> `gate/pairing-equivalence.md` for the rewrite argument and new counts.

Working doc for the EC RECOVER optimization campaign (keccak-hwsl-inline playbook:
oracle → verify existing → gate-proved rewrite → bench). Status: mapping done
2026-07-24; oracle agent running; z3 gate not started.

## The ecrecover pipeline (who does what)

1. **Guest** (`crypto/ethrex-crypto/src/lib.rs`): parses sig, computes
   `u1 = −z/r, u2 = s/r`; evaluates `pk = u1·G + u2·R` via **4 x-only ECSM
   syscalls** (`x(k·P)`, `x((k+1)·P)` for each of the two terms — the +1 query
   exists only to recover y from an x-only oracle, `solve_y` λ-linear trick at
   lib.rs:252) + one affine add. Plain RV64 guest code, proven by CPU tables.
2. **ECSM chip** (`prover/src/tables/ecsm.rs`, 667 cols, 413 constraints,
   **1 row per ecall**): reads (xG, k) from memory via MEMW, witnesses yG,
   proves curve membership `yG² ≡ xG³ + 7 mod p` (two byte-convolutions:
   `x2 = xG² − q0·p`, `yG² + µp² − xG·x2 − µ·b − q1·p = 0`), range-proves
   `xG < p`, `0 < k < N`, `xR < p` (2^256-overflow carry chains + Zero bus),
   bit-decomposes k (256 bit cols), seeds/drains the Ecdas bus, serves scalar
   bits on the Bit bus.
3. **ECDAS chip** (`prover/src/tables/ecdas.rs`, 521 cols, 200 constraints,
   **1 row per double/add step**): receives `(A, G, round, op)` on Ecdas bus,
   proves `R = 2A` (op=0) or `R = A + G` (op=1) via 3 byte-convolution mod-p
   relations (λ, xR, yR), sends `(R, round−1+next_op, next_op)` back. Rows
   telescope; ~len_k doubles + (popcount−1) adds per scalar mul ≈ 382 rows for
   a random 256-bit scalar.

Witness: `crypto/ecsm/src/witness.rs` replays double-and-add over k256
(untrusted — the chip re-proves every step).

## Column-role map

### ECSM (cols module, ecsm.rs:34)
| Cols | Role | Range authority |
|---|---|---|
| 0..8 | ts, addr_xG/k/xR (lo/hi) | MEMW consistency |
| XR 8, YR 40 (32B each) | result point | XR: bytes @store-time? NO — xR is WRITTEN, byte-checked implicitly by MEMW write path? xR_sub_p gives xR<p; **YR: NOT range-checked in ECSM** (comes back from Ecdas bus = prev ECDAS row's byte-checked yR, or = yG for k=1) |
| K 72 (256 bits) | scalar bits | IS_BIT constraints idx 1..257 |
| LEN_K 328 | MSB position | **no direct range check**; pinned via Bit-bus balance (must name a set bit; = MSB by counting, see below) + round byte-check in ECDAS for len_k−1 |
| XG 329, YG 361 | input point | xG bytes checked at memory-write time (store.rs), xG<p via XG_SUB_P; yG: is_byte here (32) |
| X2 393, Q0 425 | xG² mod p intermediate + quotient | is_byte 32+32 |
| Q1 521 (33B) | yG-relation quotient | is_byte 33 + IS_BIT(q1[32]) idx 388 |
| C0 457, C1 554 (64 each) | convolution carries | IsHalfword with offsets CARRY_OFFSET_X2=8160, _YG=16319; c[63]=0 constraints |
| XG_SUB_P 618, K_SUB_N 634, XR_SUB_P 650 (16 halfwords each) | strict-inequality witnesses | IsHalfword ×16 each; carry-bit constraints (deg 3) + overflow-required |
| MU 666 | row live flag | IS_BIT idx 0; KBitsZeroOnPadding idx 257 |

### ECDAS (cols module, ecdas.rs:32)
| Cols | Role | Range authority |
|---|---|---|
| 0..2 ts | chain key | from bus |
| XG 2, YG 34 | generator (loop-invariant) | equality with seed via Ecdas tuple matching (NOT re-checked) |
| XA 66, YA 98 | accumulator in | from bus (= prev row's byte-checked xR/yR or ECSM seed) |
| ROUND 130 | round index | is_byte 1 |
| OP 131, NEXT_OP 519 | double/add flags | IS_BIT; OP·NEXT_OP=0 (add always followed by double); NEXT_OP·(1−µ)=0 |
| XR 132, YR 164 | result point | is_byte 32+32 (**< 2^256 only, NOT < p — non-canonical reps mod p admissible mid-chain**) |
| LAMBDA 196 | slope | is_byte 32 (also < 2^256 only) |
| Q0/Q1/Q2 (33B each) | quotients for λ/xR/yR relations | is_byte 33×3 |
| C0/C1/C2 (64 each) | carries | IsHalfword ×63 each, offsets 32636/8161/16320; c[63]=0 |
| MU 520 | live flag | IS_BIT; padding: op=1, all else 0 |

### The three ECDAS relations (all with µ-gated `r·p = 3p·p` offset, `rq()` ecdas.rs:321)
- **Lambda** (deg 3): `op·(Σλ_j(xG−xA)_{i−j} + yA_i − yG_i) + (1−op)·(Σ2λ_j yA_{i−j} − 3 xA_j xA_{i−j}) + µ·(3p·p)_i − (q0·p)_i`
- **Xr** (deg 2): `(λ²)_i − xA_i − xG_i − xR_i − (1−op)(xA_i − xG_i) + rq`
- **Yr** (deg 2): `Σλ_j(xA−xR)_{i−j} − yA_i − yR_i + rq`
Each: carry recurrence `256·c_i = c_{i−1} + S_i`, c[63] = 0 ⇒ integer identity.

## Bus wiring soundness structure (what the z3 gate must cover beyond per-row)

Unlike keccak_rnd (pure per-row wiring given contracts), EC soundness lives in
**cross-row bus arguments**:

1. **Ecdas telescoping**: ECSM sends seed `[0, ts, xG,yG,xG,yG, len_k−1, 0]`,
   receives `[0, ts, xR,yR,xG,yG, −1, 0]`. Each ECDAS row receives state, sends
   successor with round' = round−1+next_op. Chain keyed by (ts, xG, yG).
   k=1 special case: seed = drain directly (round −1 ⇒ xR=xG, yR=yG echo).
2. **Bit-bus counting** (forces adds exactly at set bits, pins len_k = MSB):
   ECSM receives Bit[ts,i] with mult k_bit(i) (one per set bit); senders =
   ECDAS rows with next_op=1 (add about to consume round) + ECSM's one send at
   len_k. Any set bit above len_k ⇒ unmatched receiver ⇒ reject; len_k naming
   an unset bit ⇒ unmatched sender ⇒ reject. round strictly decreases ⇒ each
   position consumed once.
3. **Incomplete-addition edge** (add formula with A = ±G is degenerate:
   xG−xA ≡ 0 mod p): A = prefix·G at an add ⇒ prefix ≡ ±1 mod N. prefix ∈
   [2, N) after first double (prefix ≤ k < N monotone shrink) ⇒ unreachable
   for +1; −1 needs prefix = N−1 ≥ 2^255 ⇒ k ≥ 2N−2 > N ⇒ unreachable.
   **Gate must formalize this side condition** (needs k < N which IS checked).
4. **Non-canonical reps mid-chain**: xR/yR/λ only byte-checked (< 2^256 ≈
   5.4p). Relations are mod-p (quotient absorbs), so values ≡ correct mod p;
   curve membership propagates by induction from the ECSM-checked seed. Final
   xR < p checked in ECSM. Gate: prove step lemma modulo p over Z (not BV!) +
   quotient-range/no-wrap audit: max |integer LHS| vs q ≤ (2^264−1), r = 3p
   headroom, carry bounds ±~32k vs offsets, all ≪ Goldilocks wraparound —
   **the width-audit equivalent**. Keccak lesson applies: bound-necessity
   proofs need Int-mod-p, not BV (2^k invertible mod p).
5. **Splicing**: two ecalls same ts impossible (ts unique per ecall — verify
   in executor). Cross-curve id byte constant 0 today.

## Cost census (rule: every interaction ≈ 1.5 base cells of LogUp aux; split_interactions = 2 interactions/ext aux col, ext = 3 base)

### ECDAS per row — THE volume table
- Logic cells: 521
- Interactions: 1 Ecdas recv + 196 AreBytes (**all sent as [byte, 0]!**) +
  189 IsHalfword (63×3 carries) + 1 Bit + 1 Ecdas send = **388**
- Aux ≈ 582 base cells ⇒ **~1,103 committed base cells/row; bus = 53%**
- Carry machinery alone: 192 cells + 189 sends(≈284 aux) ≈ **476 cells/row = 43%**

### ECSM per row (few rows — 4/ecrecover; not the bottleneck)
- Logic 667; interactions ≈ 579 (129 AreBytes[b,0] + 174 IsHalfword + 256 Bit
  receivers + 15 MEMW + ecall + Zero + 2 Ecdas) ⇒ ~1,537 cells/row

### Per ecrecover (4 x-only scalar muls, random ~256-bit scalars)
- ECDAS rows ≈ 4 × (255 D + ~127 A) ≈ **1,528 rows ≈ 1.69M base cells**
- ECSM ≈ 4 rows ≈ 6k cells. Guest cycles for lincomb algebra + keccak(pk) extra.
- At ethrex scale (2857 tx / 60M gas block): ~4.4M ECDAS rows — dominates keccak
  by a wide margin. EC RECOVER = top bottleneck confirmed by census.

## Optimization candidates (validate AFTER gate; keccak cost rules: cells ≫ all, deg ≤ 3 hard, clean constraints > cleverness)

| # | Candidate | Mechanism | Est. win | Blast radius |
|---|---|---|---|---|
| A1 | **AreBytes pairing** | send [b_2i, b_2i+1] instead of 2× [b, 0] (contract checks both — bitwise.rs:646 ✓) | ECDAS −98 sends ⇒ −147 cells/row ≈ **−13%**; ECSM −96/row | bus repacking only; zero witness/col/constraint change; BITWISE µ recount |
| A2 | Same pairing in ECSM (yG/x2/q0/q1) | same | small (few rows) | same PR as A1 |
| B | Fuse D+A rows (add always follows double; OP·NEXT_OP=0 already forbids AA) | one row does R=2A then R'=R+G; kills per-row header dup + 2 Ecdas tuples per pair | ~−7% ECDAS cells net | new layout, witness change, moderate |
| C | GLV (secp256k1 endomorphism) | k = k1+λk2, 2-scalar Shamir, half-length scalars | ~−40% ECDAS rows | new mod-N decomposition proof in ECSM + β·x relation; crypto-design review |
| D | **lincomb2 precompile** (u1·G+u2·R in one chip pass, return x AND y) | kills the 4-query x-only dance: 1528 → ~450 rows/ecrecover | **~−70% EC cells** | new syscall + guest + executor + witness + ECSM 2-scalar redesign; the "Tier 2 representation change" of this campaign |
| E | 16-bit convolution limbs | 64→32 carries ×3 | unclear (carry range grows past IS_HALF/IS_B20) | needs paper cost model first; likely wash |

Recommended sequence: gate existing chips → A1+A2 (trivial, gate-provable,
ship like hwsl-inline) → quantify D properly (paper cost model + interface
design), C optionally folded into D (4-scalar Shamir). B only if D stalls.

## Open questions for the gate / oracle agent
- Executor semantics on invalid ECSM inputs (k=0, k≥N, xG≥p, non-residue):
  trap vs unprovable? (oracle agent pinning this)
- Is xR byte-range authority the MEMW write path (like xG/k at store time)?
  Verify — matters for the width audit. (xr_sub_p gives < p regardless, but
  the carry-chain word recomposition assumes byte-bounded xR cols.)
- ts uniqueness per ecall (splicing argument) — cite executor.
- keccak_rnd also sends AreBytes as [b, 0]? If so A1 generalizes (follow-up PR).
