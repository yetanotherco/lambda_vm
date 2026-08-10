# BLAKE3 compression chip — constraint-system & bus design (Phase 2)

**Status.** Model-level design + z3 equivalence gate, done **before** any Rust.
Ground truth = the phase-1 oracle (`../blake3-oracle/`, VALIDATED against 3
external anchors). Cost model = the verified one in `../keccak-verify/tier2_cost_model.md`
(a committed cell is expensive; each bus send ≈ 1.5 base cells of aux; **hard**
max constraint degree 3 *including* the ×μ gating factor).

**Verdict (numbers derived below, gate in `z3_blake_verify.py`):**
* **Layout: B — one row per compression, fully unrolled.** Chosen by arithmetic
  (≈5,030 cell-equiv vs ≈5,510 for one-row-per-round), and it deletes the
  state+message handoff bus entirely. Table below.
* **O1 (3-operand add carry): option (c) — two summed carry bits.** Cheaper than
  both options the oracle listed and stays degree ≤3 after μ-gating.
* **Rotations: rotr16/rotr8 free (byte relabel); rotr12/rotr7 inlined** as the
  μ-gated linear shift identity (no HWSL sends), saving 4 sends/G.
* **Every eval constraint is μ-gated, padding is all-zero, every constraint ≤3.**
* **≈5,030 cell-equiv per 6-round compression (≈5,810 for 7-round)** — about
  **1/15 of a keccak-f permutation** (≈77,000).

---

## 1. Scope & I/O interface

The chip implements the compression function `f` (oracle §2.4), **not** the tree.
Primary target is the **6-round internal variant** (Merkle 2-to-1 / Fiat–Shamir);
the design is `ROUNDS`-parameterised so 7-round is the same layout with one more
unrolled round.

### 1.1 Lean internal interface (the one we build first): 2-to-1 compression

Exposed on a dedicated **`Blake3` bus**. A parent-node caller supplies the two
child chaining values as the message and reads back the truncated CV.

**Receive** (multiplicity μ) — the compression inputs:

| field | words | bytes | source |
|---|---|---|---|
| `h[0..8]` chaining value / key | 8 | 32 | caller |
| `m[0..16]` message = `left_cv ‖ right_cv` | 16 | 64 | caller |
| `t_lo, t_hi` counter split | 2 | 8 | caller (t=0 for parents) |
| `block_len` | 1 | 4 | caller (64 for parents) |
| `flags` | 1 | 4 | caller (PARENT ∣ … for parents) |

**Send** (multiplicity μ) — the output `out[0..16]` (16 words = 64 B). CV-only
call sites read `out[0:8]`; the chip always produces all 16 (the XOF root needs
them, oracle §2.4).

`IV[0..4]` (v[8..11]) are **compile-time constants inlined** into the round-0
arithmetic — not columns, not on the bus.

### 1.2 General syscall / memory variant (sketched, not built here)

Same core; replaces the internal `Blake3` receive/send with the keccak I/O
idiom (`prover/src/tables/keccak.rs:160-449`): an `Ecall` receiver binding
(timestamp, syscall#), a `Memw` read of `x10` binding the state pointer, then
per-word `Memw` reads/writes of `h`,`m`,`t`,`block_len`,`flags`,`out`. Adds
~1 Ecall + ~(112+64)/8 ≈ 22 Memw interactions and the pointer-arith columns;
**orthogonal to the mixing core designed here** (open questions O5/O6 live here).

---

## 2. Row-layout decision (by arithmetic)

Per-compression work (6 rounds): each round = 8 G-functions; each G = **2
three-operand adds, 2 two-operand adds, 4 XORs, 2 free rotations (rotr16/8),
2 shift rotations (rotr12/7)**. Committed cells and bus sends per G (SSA form,
derivation in §5):

* committed: **56 byte-cells + 6 carry-bit cells** per G
* sends: **24** per G (16 ByteAlu[XOR] + 8 AreBytes for the two shift rotations)

| per compression | **A: 1 row / round (6 rows)** | **B: unrolled (1 row)** |
|---|---:|---:|
| logic committed (8 G × 6) | 2,976 | 2,976 |
| feed-forward committed | 64 | 64 |
| I/O input columns | 112 (×6 carried!) = 672 | 112 (once) |
| state+message handoff columns | +128 B/row × 6 = 768 | 0 |
| round-index / selector bookkeeping | ~18 | 0 |
| **committed total** | **≈ 3,760** | **≈ 3,150** |
| bus sends N (logic 192/round) | 1,152 + 6 handoff + 32 msg-rc ≈ 1,190 | 1,152 + 64 ff + 34 I/O = 1,250 |
| **aux = 3·⌈N/2⌉** | **≈ 1,750** | **≈ 1,875** |
| **total cell-equiv** | **≈ 5,510** | **≈ 5,030** |
| handoff bus | `Blake3Round` carries state(64B)+**msg**(64B)/row | none |
| structural cost | per-row state+msg reconstruction, permute-on-bus | pure compile-time wiring |

**Decision: B.** It wins on total cells (the handoff re-commits the 16-word
state *and* the 16-word message on every one of the 6 rows — BLAKE3, unlike
keccak, must carry the message down the rounds, which is the single biggest
extra cost of A) and it is structurally far simpler: the message schedule is a
compile-time permutation, so unrolling makes every round reference the original
16 committed message words under `permute^r` with **zero** runtime handoff. The
concentration of all sends into one row makes B's aux marginally higher, but the
committed-column saving dominates. B also removes round-index bookkeeping and the
`Blake3Round` bus wholesale. (Matches the oracle's recommendation, now with the
numbers behind it.)

Only reason to revisit A: if the ~3,150-wide single row's LDE/Merkle width ever
dominates trace area for tiny proofs — not the case here (keccak's per-row width
is already ~1,480+aux and BLAKE3 has 1/4 the rounds).

---

## 3. Column layout (Layout B)

One row = one compression call. Names group by role; counts are for `ROUNDS=6`.
"SSA word" = a fresh 4-byte committed word produced by one op.

| block | columns | count | notes |
|---|---|---:|---|
| `TIMESTAMP_0/1` | 2 | 2 | bus binding (internal variant may omit) |
| `MU` | 1 | 1 | multiplicity / gate flag |
| `H[0..8]` | 8 words | 32 | input CV bytes |
| `M[0..16]` | 16 words | 64 | input message bytes |
| `T_LO,T_HI,BLEN,FLAGS` | 4 words | 16 | counter split, block_len, flags |
| per-G logic × 48 G | see §5 | 2,976 | add/xor/shift SSA words + carry bits |
| feed-forward `OUT[0..16]` | 16 words | 64 | XOR outputs |
| **main columns total** | | **≈ 3,155** | |
| aux (LogUp) `= 3·⌈1250/2⌉` | | **1,875** | degree-3 ext columns |

Per-G committed breakdown (each of the 48 G-instances):

| sub-op | SSA output | bytes | carry bits |
|---|---|---:|---:|
| `add3` v[a]=v[a]+v[b]+mx | `A1` | 4 | 2 |
| `xor` v[d]^v[a] (→rotr16 free) | `X1` | 4 | – |
| `add2` v[c]+v[d] | `C1` | 4 | 1 |
| `xor` v[b]^v[c] | `X2` | 4 | – |
| `rotr12`(X2) | `SLLlo,SLLClo,SLLhi,SLLChi,B1` | 12 | – |
| `add3` v[a]=v[a]+v[b]+my | `A2` | 4 | 2 |
| `xor` v[d]^v[a] (→rotr8 free) | `X3` | 4 | – |
| `add2` v[c]+v[d] | `C2` | 4 | 1 |
| `xor` v[b]^v[c] | `X4` | 4 | – |
| `rotr7`(X4) | `SLLlo,SLLClo,SLLhi,SLLChi,B2` | 12 | – |
| **per G** | | **56** | **6** |

`rotr16`/`rotr8` produce **no columns** — the next consumer reads the XOR-output
bytes in relabeled order (see §4.2).

---

## 4. Constraints & bus interactions

All arithmetic reduces to the existing precomputed-BITWISE receivers
(`prover/src/tables/bitwise.rs`); all eval constraints are **μ-gated**, so
degree = (μ:1) × (body). Padding rows are all-zero and μ=0.

### 4.1 XOR — `ByteAlu[XOR]` send (per byte)

For each 32-bit XOR, 4 sends `ByteAlu[XOR, a_byte, b_byte] → out_byte`
(`bitwise.rs:903`). The lookup **simultaneously** byte-range-checks both operands
and pins `out` to the exact XOR — no separate range check. Operands may be linear
combos (the byte contract requires `sum ≤ 255`), which lets a free rotation be
read in-place. Eval constraints: none (pure lookup). Degree: n/a.

### 4.2 Rotations

* **rotr16 / rotr8 — free.** rotr16 = byte relabel `[b0,b1,b2,b3]→[b2,b3,b0,b1]`;
  rotr8 = `[b1,b2,b3,b0]` (oracle §3.1, exhaustively verified). No columns, no
  lookups, no constraints — the consumer indexes the source XOR's bytes in
  rotated order.
* **rotr12 / rotr7 — inline shift identity (chosen over HWSL sends).**
  `rotr12 = rotl20 = rotl16∘rotl4` (inner `r=4`); `rotr7 = rotl25 = rotl16∘rotl9`
  (`r=9`). For input word `X = xlo + 2^16·xhi` (halfwords `xlo,xhi`, 2 bytes each):

  **Shift identities (eval, degree 2 after ×μ):**
  ```
  μ·( xlo·2^r − SLLC_lo·2^16 − SLL_lo ) = 0
  μ·( xhi·2^r − SLLC_hi·2^16 − SLL_hi ) = 0
  ```
  **Recombine + halfword swap (eval, degree 2 after ×μ):**
  ```
  μ·( Ylo − SLL_hi − SLLC_lo ) = 0     # output low halfword  = Y[0]+256·Y[1]
  μ·( Yhi − SLL_lo − SLLC_hi ) = 0     # output high halfword = Y[2]+256·Y[3]
  ```
  **Range checks (sends):** `AreBytes` on the 8 bytes of `SLL_lo,SLLC_lo,SLL_hi,
  SLLC_hi` = 4 sends/rotation (`bitwise.rs:783`). `Y` is range-checked *free* by
  the downstream XOR that consumes it.

  Soundness (proven in `../keccak-verify/hwsl_inline_test.py` Part 2, and by the
  width audit in the gate): given `SLL_* ∈ [0,2^16)` (the tight remainder bound
  from AreBytes) and `2^16` invertible mod p, the identity **uniquely** pins
  `SLL = (xlo·2^r) mod 2^16` and `SLLC = (xlo·2^r) >> 16`; the loose 16-bit bound
  on `SLLC` suffices because it is the quotient, not the remainder. The two
  recombination sums are over non-overlapping bit ranges, so `+` = `OR` and each
  is an exact 16-bit halfword.

  **HWSL alternative, priced:** replace each shift identity with an `Hwsl` send
  (`bitwise.rs:831`). Cost/rotation: +2 Hwsl sends, same AreBytes, same columns.
  Per compression that is +4 sends/G × 48 = +192 sends → +288 aux cells (≈6%).
  Inline wins because the eval identity is free of columns/sends; it costs only
  degree budget (2 ≤ 3). **Use inline.**

### 4.3 Two-operand add — `emit_add_pair` low half (eval, degree 3 after ×μ)

`s = (a+b) mod 2^32`; one carry bit. Following `templates.rs:334`:
```
carry = (a + b − s)·2^-32           # linear expression, INV_SHIFT_32 = (2^32)^-1
μ · carry·(1 − carry) = 0            # degree (1)×(1)×(1 body)=2, ×μ = 3
```
`s`'s bytes are range-checked **free** by the next XOR that consumes `s`
(every add output in G feeds a subsequent XOR — see §5). Booleanity + `s∈[0,2^32)`
⇒ `s` unique.

### 4.4 Three-operand add — **O1 resolved: option (c), two summed carry bits**

`s = (a+b+m) mod 2^32`, carry ∈ {0,1,2}. Commit two carry **bits** `c1,c2`
(2 cells, no intermediate word):
```
μ·( a + b + m − s − 2^32·(c1+c2) ) = 0     # sum identity, linear → ×μ = degree 2
μ · c1·(1 − c1) = 0                          # ×μ = degree 3
μ · c2·(1 − c2) = 0                          # ×μ = degree 3
```
`s`'s bytes range-checked free downstream. `c1+c2 ∈ {0,1,2}` covers the carry;
`s∈[0,2^32)` + the sum identity pin `s = (a+b+m) mod 2^32` uniquely (proof in the
gate's width audit).

**Why (c):**

| O1 option | extra committed / 3-op add | degree (ungated → ×μ) | legal under ×μ? |
|---|---|---|---|
| (a) one ternary carry `k(k−1)(k−2)=0` | 1 bit | 3 → **4** | ❌ (μ-gating mandatory, §4.5) |
| (b) two chained binary adds | 1 word (4 B) + 2 AreBytes | 2 → 3 | ✅ but +4B +2 sends |
| **(c) two summed carry bits** | **2 bits** | 2 (bool) / 1 (sum) → 3 / 2 | ✅ **cheapest** |

Over a compression, (c) vs (b): saves (4B−2bit) per 3-op add × 96 three-op adds
≈ **300 committed cells + 192 AreBytes sends**. (c) is a strict refinement of the
oracle's two options.

### 4.5 μ-gating & padding — **O2 resolved: gate everything, all-zero padding**

Every eval constraint is multiplied by `μ` (the `MU` column, 1 on the real row,
0 on padding), exactly like `keccak_rnd`'s IS_BIT (`keccak_rnd.rs:914`). Padding
rows are **all-zero**:
* bus interactions carry `Multiplicity::Column(MU)` ⇒ 0 contribution on padding;
* eval constraints are `μ·(…)` ⇒ 0 on padding regardless of the (zero) cells.

This is why O1 must be (b) or (c): the ternary carry (a) is degree 3 *ungated*,
and ×μ pushes it to 4. Inlined `IV` constants are fine because the round-0 add
that consumes them is itself μ-gated (its carry expression is nonsense on an
all-zero padding row, but ×μ=0 kills it). **The μ-gating requirement is what
forecloses option (a) — this is the single tightest coupling in the design.**

### 4.6 Feed-forward (16 XORs, all `ByteAlu[XOR]`)

```
out[i]   = v[i]   ⊕ v[i+8]      i = 0..8      (v[i+8] = final state word)
out[i+8] = v[i+8] ⊕ h[i]        i = 0..8      (h = original input CV column)
```
64 sends, 64 committed output bytes (the XOR outputs), range-checked free by the
lookup. Output bytes are shipped on the `Blake3` send.

### 4.7 Range checks that are NOT free

The message `m` enters **only** through adds (never XORed), so its 64 bytes need
explicit `AreBytes` (32 sends/compression). `h` and `t/block_len/flags` all feed
an XOR (feed-forward / round-0 diagonal), so they are free. Every add/shift/xor
output feeds a downstream XOR ⇒ free.

### 4.8 Degree ledger (the hard gate)

| constraint | body degree | × μ | ≤ 3? |
|---|---:|---:|:--:|
| 2-op add carry booleanity | 2 | 3 | ✅ |
| 3-op add sum identity | 1 | 2 | ✅ |
| 3-op add carry booleanity ×2 | 2 | 3 | ✅ |
| shift identity (×2) | 1 | 2 | ✅ |
| recombine identity (×2) | 2 | 3 | ✅ |
| (rejected) ternary carry | 3 | **4** | ❌ |

Worst legal constraint = 3. **No constraint exceeds 3.**

---

## 5. Per-G dataflow, SSA + free range-checks

```
A1 = add3(v[a], v[b], mx)                 # v[a]  ; 2 carry bits ; range-checked by X1
X1 = xor(v[d], A1) ;  v[d] = rotr16(X1)   # free relabel
C1 = add2(v[c], v[d]=rotr16(X1))          # v[c]  ; 1 carry bit ; range-checked by X2
X2 = xor(v[b], C1)
B1 = rotr12(X2)                           # v[b]  ; range-checked by X4 / next round
A2 = add3(A1, B1, my)                     # v[a]  ; 2 carry bits ; range-checked by X3
X3 = xor(v[d]=rotr16(X1), A2) ; v[d]=rotr8(X3)
C2 = add2(C1, v[d]=rotr8(X3))             # v[c]  ; 1 carry bit ; range-checked by X4
X4 = xor(B1, C2)
B2 = rotr7(X4)                            # v[b]  ; range-checked next round / FF
```
Every committed add/shift word is an operand of a later XOR ⇒ its bytes are
byte-range-checked for free by that `ByteAlu` lookup. Confirmed: no add/shift
output needs its own AreBytes. (Only `m` does — §4.7.)

---

## 6. Cost & comparison

| quantity (6-round) | value |
|---|---:|
| committed main columns | ≈ 3,150 |
| bus sends N | ≈ 1,250 (832 XOR incl. 64 feed-forward + 384 shift-AreBytes + 32 msg-AreBytes + 2 I/O) |
| aux base cells (3·⌈N/2⌉) | ≈ 1,875 |
| **total cell-equiv / compression** | **≈ 5,030** |
| 7-round variant | ≈ 5,810 |
| keccak-f permutation (reference) | ≈ 77,000 |
| **BLAKE3-6r as fraction of keccak-f** | **≈ 1/15 (6.5%)** |

Dominated by the ~960 byte-XOR lookups, as the oracle predicted. Note: the
oracle's prose "¼–⅓ of a keccak permutation" is inconsistent with its own
5–6k/compression figure; the detailed count here (≈5k vs 77k) puts it at **~1/15**.

---

## 7. Soundness-critical spots a Rust implementation must NOT deviate from

1. **μ-gate every eval constraint** (carry booleanity, sum identity, shift
   identity, recombine). Un-gated ternary carry or an un-gated constraint with
   inlined IV constants breaks all-zero padding. (§4.5)
2. **3-op add = two summed carry bits with the explicit sum identity** — not a
   single ternary carry (degree 4 after gating), and the sum identity must be
   present (without it, `s` is only constrained mod nothing). (§4.4)
3. **Shift identity needs the tight `SLL ∈ [0,2^16)` AreBytes bound**; dropping it
   makes the rotation forgeable (a wrong `SLL` admits a large field `SLLC`).
   Soundness relies on `2^16` invertible mod p — a BV model cannot see this;
   verify in the field (gate width audit + `hwsl_inline_test.py`). (§4.2)
4. **Every add/shift output must actually feed a downstream XOR** (its only range
   check). If a future refactor reorders so an add output is *last* with no XOR
   consumer, add an explicit AreBytes or the carry argument is unsound. (§5)
5. **Message `m` needs explicit AreBytes** — it is never XORed. (§4.7)
6. **rotr16/rotr8 byte order** exactly `[b2,b3,b0,b1]` / `[b1,b2,b3,b0]`
   (little-endian). A wrong relabel silently corrupts. (§4.2)
7. **Message permutation `permute^r`** wired per round from the *original* 16
   `M` columns; MSG_PERMUTATION = `[2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]`. The
   trailing permute after the last round is unused (oracle §2.4). (Gate control
   `wrong_msg_index`.)
8. **IV / feed-forward / counter split** exactly per oracle §2.4:
   `v[8..12]=IV[0..4]` inlined, `v[12]=t_lo, v[13]=t_hi, v[14]=block_len,
   v[15]=flags`; `out[i]=v[i]⊕v[i+8]`, `out[i+8]=v[i+8]⊕h[i]`. (Controls
   `wrong_iv`, `drop_ff_xor`.)
9. **Non-overflow side conditions (width audit):** all add/shift field
   expressions stay `< 2^35 ≪ p`, so `≡0 mod p` ⇒ `=0` as integers; the whole
   soundness argument depends on operands being genuine ≤32-bit (byte columns)
   and carries being genuine bits.

---

## 8. Gate

`z3_blake_verify.py` — free-variable model of every column, every lookup/eval
constraint as an equation, `assert output ≠ oracle-reference`, ask z3 for a
counterexample. Reference (`bref_*`) is an independent 32-bit-BV port of
`blake3_ref.py` (RotateRight / + / ^), structurally independent of the byte-level
shift wiring. Results are appended to §9 after the run (`run.log`).
```
python3 z3_blake_verify.py            # round + wrapper + controls + audit (fast)
python3 z3_blake_verify.py --full     # + monolithic 6- and 7-round UNSAT
```

## 9. Gate results

Default run (`python3 z3_blake_verify.py`, ~2 min) — **OVERALL: PASS**:

| check | result | meaning |
|---|---|---|
| **MAIN 0** — one G-function, free inputs | **UNSAT** | the quarter-round (byte-XOR + inline rotr12/rotr7 shift identities + 2-op & 3-op adds) is correctly & tightly constrained; **covers every G, hence every round** (a round is a fixed composition of 8 G-calls). |
| **MAIN 1** — init-state + feed-forward (rounds=0) | **UNSAT** | `v` layout (`h`/IV/counter-split/block_len/flags) and `out[i]=v[i]⊕v[i+8]`, `out[i+8]=v[i+8]⊕h[i]` are correct. |
| neg `rot_wrong_amount` | **SAT** | wrong rotation amount detected. |
| neg `swap_g_operand` | **SAT** | swapped G-function operand detected. |
| neg `wrong_iv` | **SAT** | wrong IV constant detected. |
| neg `drop_ff_xor` | **SAT** | dropped feed-forward XOR detected. |
| neg `wrong_msg_index` | **SAT** | wrong message-schedule index detected (permutation is load-bearing). |
| **pos** 6-round seeds 0,1,2 (canonical vectors) | **SAT** | full 6-round pipeline reproduces the oracle's recorded output for concrete inputs. |
| **pos** 7-round (oracle-generated) | **SAT** | full 7-round pipeline reproduces the oracle's `compress(…,rounds=7)`. |
| audit: shift `SLL` 16-bit bound present | **UNSAT** | with AreBytes the shift output is pinned. |
| audit: **DROP `SLL` bound** (field neg ctrl) | **SAT** | without it the rotation is forgeable (needs `2^16` invertible mod p). |
| audit: 3-add carry booleanity present | **UNSAT** | with booleanity the sum `s` is pinned. |
| audit: **DROP carry booleanity** (field neg ctrl #4) | **SAT** | without it `s` is forgeable in the prime field. |

**The 6th team-lead control — "dropped carry booleanity" — lives in the width
audit, not the BV controls, and this is correct.** Dropping a committed carry
column's booleanity is a *field-level* soundness bug: the column becomes a full
Goldilocks element, but a *bounded-BV* model keeps the 8-bit carry + `s∈[0,2^32)`
byte-range, which still pins `s`, so BV reports UNSAT (verified: the BV version
does). Only the mod-p model exhibits the forgery — exactly the phenomenon
`../keccak-verify/hwsl_inline_test.py` Part 2 documents (`2^16`/`2^32` are zero
divisors mod `2^n`). The gate deliberately separates BV-observable logic bugs
from field-only soundness bugs; both classes fire.

**`--full`** additionally runs the heavy monolithic symbolic UNSATs (one round;
compression rounds=2 for the permutation; full 6- and 7-round). These are *bonus*
confirmations — the G-unsat + fixed-composition chaining argument + rounds=0 +
the concrete full-pipeline positive controls already establish full-compression
correctness. (The direct 6-round symbolic UNSAT is large; it is not required for
the verdict and may take a long time / be run offline.)

### What is and isn't proven
* **Proven (symbolic, all inputs):** the G quarter-round; the init-state layout;
  the feed-forward — hence, by the chaining argument, the full N-round
  compression for **both ROUNDS=6 and ROUNDS=7**.
* **Proven (concrete, external anchor):** the *entire* unrolled pipeline
  (init + 6/7 rounds + message permutation + feed-forward) reproduces the
  oracle's validated vectors.
* **Proven (field-level):** the AreBytes shift bound and the add-carry booleanity
  are each *necessary* (dropping either is a forgery mod p).
* **Assumed (assume-guarantee, not re-proven here):** the precomputed BITWISE
  table contracts themselves (ByteAlu[XOR], AreBytes) — these are existing,
  separately-audited chips (`prover/src/tables/bitwise.rs`). Same assumption the
  keccak gate makes.
* **Not modeled here:** the memory/syscall I/O variant (§1.2) — orthogonal;
  open questions O5 (counter width, already covered by the Plonky3 anchor) and
  O6 (endianness at the MEMW boundary) live there and must be pinned when that
  interface is wired.
