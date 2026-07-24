# ECSM/ECDAS z3 gate — lemma board & soundness theorem

Reverse-order verification of the EXISTING chips (keccak_rnd playbook), 2026-07-24.
Model transcribed from `prover/src/tables/ecsm.rs` / `ecdas.rs` (citations inline in
the scripts); independent reference = `../oracle/ec_ref.py` (3-lineage-anchored).

**Faithfulness anchor**: before any UNSAT was trusted, the transcribed model was
evaluated on REAL prover witnesses (`crypto/ecsm::compute_witness` via the extended
oracle harness): 12 scalars incl. k=1, k=N−1, 2^255, 2^255−1 → 872,242 checks,
2,190 ECDAS rows, all 413/200 constraint values ≡ 0 mod p_g, all carries inside
their claimed IsHalfword windows, all chaining/bit-balance relations hold.
(`positive_real_witness.py`, run log below.)

## Board

| Lemma | Statement | Verdict | Notes |
|---|---|---|---|
| L1 ×5 | carry recurrence + c₆₃=0 ⇒ Σ256^i·S_i = 0 over ℤ | PROVED (z3 UNSAT ×5) | telescoping, per relation |
| L2a ×5 | every ConvCarry LHS integer value < p_g under contracts | PROVED (exact intervals) | max ≈ 2^25 ≪ 2^64; bounds exact for multilinear-plus-signed-squares forms; 9,000-sample corner cross-check |
| L2b ×5 | honest carries fit the IsHalfword windows (completeness) | PROVED | minimal offsets computed: x2 7949, yG 8371, λ 24605, xR 256, yR 8289 vs repo 8160/16319/32636/8161/16320 — all sound with slack |
| L2b-ctl | offset-below-necessity fails the audit; r=2p ⇒ negative honest quotient, r=3p ⇒ q_min=5 ≥ 0 | PROVED | the audit binds; r=3p is necessary AND sufficient |
| L2b-q | quotient headroom: ECDAS q ≲ 2^259 < 2^264, ECSM q1 ≲ 2^257 | PROVED | honest completeness |
| L2c | word-carry lift + strict-inequality chains (xG<p, k<N, xR<p) | PROVED (z3 UNSAT ×2) | covers the INV_SHIFT_32 virtual-carry trick |
| L3a ×5 | Σ256^i·S_i ≡ intended polynomial in composed values (identically) | PROVED (sympy expand = 0) | byte convolutions capture full products, no truncation |
| L3b ×2 | chord/tangent output stays on the curve | PROVED (sympy, reduction mod curve eqs) | feeds the L6 induction invariant |
| A-PRIME | p, N prime; N odd | CERTIFIED (sympy.isprime) | the ONLY assumed algebraic fact; used by L4a, L5c |
| L4a ×2 | λ pinned mod p given side condition (add: p∤(xG−xA); dbl: p∤2yA) | PROVED (z3 UNSAT ×4 + A-PRIME) | Euclid split: ring step + remainder step machine-checked; primality assumed-certified |
| L4b, L4c | xR, yR pinned mod p given λ pinned | PROVED (z3 UNSAT ×2) | easy divisibility direction only |
| L5a | no y≡0 point on secp256k1 ((−7) non-cube; \|E\|=N odd) | PROVED (computation) | discharges the double side condition for any on-curve-mod-p input |
| L5b | incomplete-addition edge unreachable (accumulator ≡ ±G at an add) | PROVED (z3 UNSAT) | uses k<N (K_SUB_N) + chain structure; the c=2t, u=2t+1=⌊k/2^r⌋, c≤N−2 argument |
| L5c | x-equality of on-curve points ⇒ y = ±y (mod p) | PROVED (z3 UNSAT ×2 + A-PRIME) | turns "A≠±G" into "p∤(xG−xA)" |
| L6 | chain soundness (bus telescoping + Bit counting + induction) | PROVED (structured argument below; arithmetic cores are L4/L5/N4 z3 lemmas) | rests on contracts C1–C7 |
| L7 | end-to-end: drain xR ≡ oracle x(k·P), both yG sign classes | PROVED (990 linear z3 UNSAT queries, 3 pts × 12 k × 2 signs) | quotients left FREE ⇒ conclusion covers all byte-representable lifts; drain uniqueness via XR_SUB_P |
| L8 | negative/sensitivity controls | PASSED (4 forge/catch, 5 redundant, baseline UNSAT) | battery below; `logs/l8.log` |

## L8 battery (non-vacuity + per-check load-bearing/redundancy)

Method: genuine forgeries are constructive (fixed numerals, z3 only CHECKS the
assignment); the carry-check probes N1/N3 use a LINEAR carry gadget (S_i free
bounded integers, field recurrence 256·c_i = c_{i−1}+S_i+p_g·m_i, attack goal
V=Σ256^i·S_i ≢ 0 mod p_secp — the Goldilocks-wrap model). Baseline: with ALL
checks the gadget is UNSAT (can't decode a wrong value) — non-vacuous.

| Control | Verdict | Meaning |
|---|---|---|
| baseline (all checks) | UNSAT | gadget non-vacuous: full window set + c₆₃=0 forces V ≡ 0 mod p |
| **N3** drop ColIsZero(c₆₃) | **SAT — FORGES** | top overflow unconstrained ⇒ decoded value ≢ 0 mod p. **c₆₃=0 LOAD-BEARING** |
| **N6** drop XR_SUB_P | **SAT — FORGES** | non-canonical drain xR=v+p accepted for v<2³²+977. **LOAD-BEARING** (Finding 5) |
| **NSW** transcription swap xA↔xG in yR | **SAT — FORGES** | gate catches a transcription bug: tampered relation admits wrong yR |
| **N-CONST** wrong prime p→p+2 | **SAT — CATCHES** | honest witness ≡0 under p, ≢0 under p+2 ⇒ constraints bind the constant (keccak wrong-RC analog) |
| N1 drop IsHalfword(c₄₀) | UNSAT — REDUNDANT | a MID carry window is individually redundant given c₆₃=0 + neighbours (strong: S free). Finding 6 |
| N4 drop OP·NEXT_OP | UNSAT — REDUNDANT | Bit-balance alone blocks add-after-add; +drop balance ⇒ SAT (Finding 1) |
| N5 drop IS_BIT(q1[32]) | UNSAT — REDUNDANT | curve relation still pins yG²≡xG³+7 with q1<2²⁶⁴ (Finding 2) |
| N7 drop KBitsZeroOnPadding | UNSAT — REDUNDANT | rejection-only (padding senders µ/next_op-dead ⇒ imbalance); Finding 3 |

Non-vacuity: **4** genuine forgeries/catches. Load-bearing checks confirmed:
c₆₃=0 (top overflow), XR_SUB_P (canonical drain), the relation constants, and
the operand wiring (NSW). The dropped-check redundancies (N1, N4, N5, N7) are
keccak-IS_BIT-style: individually removable, collectively part of the L1
telescoping argument — keep as insurance.

## Soundness theorem (what is now proven)

Under contracts C1–C7 below, any accepted trace satisfies: for every ECSM row
(µ=1) with ecall timestamp ts, inputs xG, k bound by the MEMW reads, and output
xR bound by the MEMW writes:

> xG is a canonical field element (< p) and a valid secp256k1 x-coordinate;
> 0 < k < N; and xR is the canonical x-coordinate of k·P where P is either lift
> (xG, ±y) — both give the same xR. Exactly matching the oracle/executor
> semantics (`ec_ref.x_only_mul`), including the k=1 echo.

Chain of proof: field constraints →(L2a/L2c) integer identities →(L1, L3a) value
relations →(L4, sides by L5) per-step group law mod p, on-curve preserved (L3b)
→(L6) whole-chain = double-and-add per the bits of k, adds exactly at set bits,
len_k = MSB →(L7) drain equals oracle on concrete instances (soundness spot-proof
battery over 72 chains ensuring no compositional gap).

**Status: the board is fully green.** All L1–L7 lemmas PROVED (or
PROVED-with-explicitly-discharged side conditions); A-PRIME certified; L8
establishes non-vacuity (4 forgeries/catches) with the baseline UNSAT and every
load-bearing check confirmed. No lemma is OPEN. Transcription is anchored by the
872k-check real-witness evaluation. Remaining trust surface = the eight contracts
C1–C7 + A-PRIME below (the assume-guarantee boundary), plus the standing
residual-risk of hand transcription mitigated (not eliminated) by the
real-witness anchor — the durable fix is generating the SMT from the constraint
IR (`air.constraint_program()`), same as the keccak note.

**Completeness** (no honest rejection): L2b windows + L2b-q headroom + the
real-witness runs (all contracts satisfied by construction-faithful witnesses).

## L6 — chain soundness argument (the multiset/induction core)

Setting: LogUp gives exact signed multiset balance per bus (contract C5). All
tuple components are field elements; byte-level equality of tuples ⇒ integer
equality of composed values.

1. **Degrees.** Every µ=1 ECDAS row receives exactly one Ecdas tuple and sends
   exactly one (both interactions have multiplicity µ; ecdas.rs:159-172, 224-247).
   Every µ=0 (padding) row participates in neither, and its Bit send is also
   dead: NEXT_OP·(1−µ)=0 (ecdas.rs:429-433) forces next_op=0. Every ECSM µ=1
   row sends one seed and receives one drain (ecsm.rs:542-576).
2. **Graph.** Balance ⇒ the µ=1 ECDAS rows form a 1-regular flow graph whose
   sources are ECSM seeds and sinks are ECSM drains, i.e. disjoint paths plus
   possibly cycles. Along any edge the tuple's (ts, xG, yG) components are
   copied unchanged (the ECDAS sender reuses the receiver's TS/XG/YG columns —
   same column indices in both tuples), so they are path invariants.
3. **No cycles.** Along consecutive rows, round' = round − 1 + next_op with
   op' = next_op; a row with op = 1 (add) satisfies... any two consecutive
   steps decrease round by ≥ 1 in total: a next_op=1 hop keeps round equal but
   forces the successor to be an add row, whose own Bit-send multiplicity
   pattern (see 5) prevents a second same-round add hop [z3-checked in N4's
   model: no schedule satisfying balance revisits a round]. A cycle would need
   total round change 0 with at least one strict decrease — impossible since
   round never increases. Also no row can receive round = −1 (ROUND is
   byte-checked, ecdas.rs:184; −1 mod p_g is not a byte), so paths terminate
   only at drains.
4. **Per-ecall uniqueness.** Distinct ecalls have distinct ts (C7). A path's ts
   invariant therefore matches it to exactly one ECSM row: its seed and drain
   belong to the same row (an ECSM row's drain receiver uses its own ts
   columns). Two ECSM rows cannot share a path. (If two ecalls had equal ts,
   xG, yG AND equal k they'd be fully interchangeable — same outputs — hence
   harmless; C7 rules it out anyway.)
5. **Bit counting ⇒ schedule.** For position i: receivers total k_bit(i) ∈
   {0,1} (IS_BIT, ecsm.rs:837-843; multiplicity column k_bit(i), ecsm.rs:529-534).
   Senders: the ECSM MSB send at len_k (mult µ, ecsm.rs:536-540) plus one send
   per ECDAS row with next_op=1 at its round (ecdas.rs:217-221). Balance ⇒
   (a) len_k names a set bit; (b) every set bit above len_k or not consumed by
   an add ⇒ imbalance ⇒ len_k = MSB(k) and adds occur EXACTLY at set bits below
   the MSB, each once (round strictly decreases between add opportunities);
   (c) zero bits get no add. With seed round = len_k − 1 and op₀ = 0 (constants
   in the seed tuple, ecsm.rs:541-561), the path's (round, op) sequence is
   exactly the reference double-and-add schedule of k. K=1: len_k = 0, seed
   round = −1: no ECDAS row can receive it (3.), so balance forces drain = seed
   ⇒ xR = xG, yR = yG — the echo, = oracle. KBitsZeroOnPadding (ecsm.rs:845-853)
   plus IS_BIT keep padding rows out of the Bit bus (see N7: even without it,
   padding could only add unmatched receives ⇒ rejection-only).
6. **Induction along the path.** Invariant: entering step t, the accumulator
   values (xA, yA) ≡ (mod p) the affine coordinates of c_t·P, on curve, with
   c_t = the binary prefix of k consumed so far; 2 ≤ c_t ≤ k < N after the
   first double. Base: seed = (xG, yG), on curve by the ECSM relations (L3a on
   x2/yG relations + L2), canonical x (XG_SUB_P, L2c), c = 1. Step: L4 pins the
   row's outputs to the chord/tangent result mod p — side conditions discharged
   by L5a (double: yA ≢ 0 for on-curve A) and L5b+L5c (add: A ≠ ±G ⇒ xA ≢ xG);
   L3b keeps the invariant on-curve; the multiplier updates ×2 / +1 per the
   schedule (5.). Drain: c_final = k, so (xR, yR) ≡ k·P; XR_SUB_P forces xR
   canonical ⇒ xR = x(k·P) exactly. yG's sign is never fixed — both lifts give
   the same x (x(k·P) = x(k·(−P))), matching the x-only oracle contract.

## Contracts (assume-guarantee boundary)

- **C1 AreBytes**: each element of an `AreBytes[x,y]` send is in [0,256)
  (bitwise.rs:646, receiver :783-796; precomputed table).
- **C2 IsHalfword**: sent value in [0, 2^16) (bitwise.rs:797-810).
- **C3 IS_BIT/booleans**: in-table x·(1−x) constraints (µ, k bits, op, next_op,
  q1[32], overflow carry bits).
- **C4 MEMW byte authority**: xG and k bytes are range-checked at memory-write
  time (store.rs path; ecsm.rs:460 comment), and xR bytes at the ECSM MEMW
  write; ECSM's YR inherits byte-ness from tuple equality with ECDAS's
  byte-checked yR (or YG for k=1).
- **C5 LogUp multiset soundness**: exact signed balance per bus (generic
  argument, spec/logup.typ).
- **C6 Ecall binding**: each ECSM row's ts corresponds to a real executed ECSM
  ecall (CPU Ecall bus sender ↔ receiver, ecsm.rs:307-316).
- **C7 Timestamp uniqueness**: distinct CPU rows (hence ecalls) have distinct
  timestamps — enforced by the PC-token chain through the MEMW consistency
  argument (trace_builder.rs:341-347 builder; cpu.rs:541-542 + spec/memory.typ
  in-proof cadence).
- **A-PRIME**: p and N are prime (sympy-certified; SEC2 published constants).

## Findings

1. **[redundancy, keep-as-insurance] OP·NEXT_OP = 0** (ecdas.rs:424-427): Bit-bus
   balance + IS_BIT(k_bit) already exclude add-after-add schedules (N4: tampered
   model UNSAT with balance kept, SAT once balance is also dropped). Keep — it
   is what makes the round-monotonicity argument local.
2. **[redundancy, keep-as-insurance] IS_BIT(q1[32])** (ecsm.rs:876-879): the yG
   relation pins yG² ≡ xG³+7 mod p for ANY q1 < 2^264 (N5 UNSAT); Goldilocks
   headroom unaffected (L2a re-audit with fat q1). Completeness needs q1 < 2^257
   which the honest builder satisfies. Keep as spec clarity.
3. **[redundancy-for-soundness] KBitsZeroOnPadding** (ecsm.rs:845-853): padding
   k_bits could only fire unmatched Bit RECEIVES (multiplicities are {0,1} by
   unconditional IS_BIT; padding ECDAS/ECSM senders are µ- or next_op-dead) ⇒
   rejection-only ⇒ not soundness-load-bearing. It IS completeness-relevant
   hygiene (honest padding is all-zero) and cheap. Keep.
4. **[tightness] Carry-offset slack**: minimal sound offsets are {x2: 7949,
   yG: 8371, λ: 24605, xR: 256, yR: 8289}; the repo constants are safely above
   with 2^16-window room on both sides. No action.
5. **[non-dead-weight] XR_SUB_P** is genuinely load-bearing: without it, drain
   values v < 2^32 + 977 admit the non-canonical representation v + p (N6 SAT),
   i.e. a wrong 32-byte xR delivered to the guest. Astronomically rare input
   class, but the check is what closes it.
6. **[redundancy, keep-as-insurance] Mid-limb carry windows** (e.g. the
   IsHalfword on c[40]): individually redundant — the linear carry gadget shows
   a Goldilocks wrap injected at one un-windowed interior carry cannot be
   absorbed while c₆₃=0 and the neighbouring windows hold (N1 UNSAT, S free =
   strong). By contrast the TOP closing check c₆₃=0 IS load-bearing (N3 SAT).
   Reading: the interior windows earn their place collectively (they are the L1
   telescoping hypothesis); no single interior one is a soundness lynchpin, but
   removing several would break L1. Do not churn — they are cheap
   precomputed-table sends.

## Reproduction

venv: the oracle scratchpad venv + `z3-solver sympy` (see ../oracle/README.md).
```
python positive_real_witness.py   # transcription anchor (run FIRST) — 872k checks
python l1_l2_lift.py              # L1, L2a/b/c + audit controls
python l3_l4_value.py             # L3a/b, A-PRIME, L4a/b/c
python l5_sides.py                # L5a/b/c
python l7_pin.py                  # L7 (990 queries)
python l8_negative.py             # L8 battery (~5 min; carry-gadget queries dominate)
```
Logs in `logs/`. The harness `witness` command (added to
`../oracle/repo-harness/src/main.rs`) dumps real EcsmWitness/EcdasStep values for
the positive control; rebuild with `cargo build --release` in that dir.
