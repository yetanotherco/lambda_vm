# lincomb2 chip layout — ECSM′ / ECDAS′ / Addend bus (phase D de-risk)

Concrete layout spec for phase D, derived by reading the implemented witness
(`crypto/ecsm/src/witness.rs`) and the two live chips
(`prover/src/tables/{ecsm,ecdas}.rs`) line by line. **Doc only — no code changed.**

Everything below is either (a) read out of the source, or (b) a *proposal*
explicitly labelled as such. Where the witness and a design doc disagree, the
witness wins (IMPL-PLAN §0.2). Where I could not determine something, §7 says so
instead of guessing.

Baseline read: `ecsm.rs` 911 lines / 667 cols / 413 constraints; `ecdas.rs` 463
lines / 521 cols / 200 constraints; both at `bc62f00e` with the AreBytes pairing
(`42ba68ff`) already in.

---

## 0. Lead: six findings that change the plan

Ordered by how much they move the design. (0)–(3) invalidate something currently
written down; (4)–(5) are gaps nobody has costed.

Findings (4) and (5) were written against the docs and have since been confirmed
independently by the sibling phases — `phase-c-t0table`'s chip header states the
`len ≤ 256` obligation in almost the same words (§0.4), and `phase-b-executor`'s
shipped syscall already takes the shape §0.3 recommends. Where their in-flight
code settles something, I say so rather than leaving it open.

### 0.0 A register-returned status **is** expressible — correcting my own earlier claim

I previously reported that a status word returned in a register would need decode
surgery, on the grounds that ECALL decode leaves `write_register = false`
(`prover/src/tables/types.rs:781-785`). The lead folded that into IMPL-PLAN §2 and
§10 item 1 ("Decided: memory — register was tried and is not expressible").

**That conclusion is wrong, and the counter-example is already in the tree.** The
COMMIT accelerator overwrites `x10` during its ecall: `commit.rs:29` describes the
interaction as "read+write x10 register (fd=1→count)", and the trace builder emits
it at `trace_builder.rs:1225-1228` —

```rust
let memw_op = MemwOperation::new(true, reg_addr, new_value, ts, 2, true)  // is_register, NEW value
…
register_state.write(10, count, ts);
```

`write_register = false` governs the **CPU row's** register-write path. It says
nothing about what an accelerator chip may emit on the MEMW bus, and accelerators
already emit their own register reads and memory writes at their own timestamps
(ECSM's `xR` writes, `trace_builder.rs:919-932` / `ecsm.rs:432-443`). A register
status is one more MEMW send with `is_register = 1` and a new value.

`phase-b-executor` has already shipped it that way — `ecsm_lincomb2` returns the
status through `inlateout("a0")` (`syscalls/src/syscalls.rs`, new fn). **That is
correct and strictly cheaper** than the memory route: no 8-byte status word in the
output buffer, one fewer doubleword write, and the guest tests a register instead
of a load. IMPL-PLAN §2's ABI block and §10 item 1 should be updated to match what
phase B built, rather than phase B being asked to match the plan.

The chip-side consequence is in §1 (the output write is 8 doublewords, not 9, plus
one register-write MEMW send) and §4.4 (the `OK`/`STATUS` binding is unchanged —
where the status *lives* does not affect why it must be bound).

### 0.1 `layout-lock.md` correction #2 is wrong — the `nb_or` column is required

`layout-lock.md:113-117` deletes DESIGN's `nb_or` helper on the grounds that the
schedule is dense-doubling. Dense doubling is true; the conclusion does not
follow.

Read the replay (`witness.rs:825-879`): within one iteration the DOUBLE and the
optional ADD **share the same `round`**, and `round` decrements only at the loop
boundary. So the successor round is

- double row at round `r`, no add follows → next round `r − 1`
- double row at round `r`, add follows   → next round `r`
- add row at round `r`                   → next round `r − 1`

The double row's successor round therefore depends on *whether an add follows*,
which is not a function of the double row's own columns. Today's ECDAS solves the
identical problem with `NEXT_OP` and the outgoing expression
`round − 1 + next_op` (`ecdas.rs:243-253`). Deleting the equivalent column leaves
`round` free to stall, which lets a prover insert or drop doublings.

**And the witness does not currently supply it.** All four `joint_row` call sites
(`witness.rs:807`, `:830`, `:863`, `:896`) pass `next_op = 0` — the 5th positional
argument, confirmed against the signature at `witness.rs:725-740`. Likewise
`d1`/`d2` are the *real* digit bits only on the ADD row; the DOUBLE row is given
`0, 0` (`witness.rs:838-840`). So as it stands the double row carries no
information about what follows it.

**Fix (small, and it does not disturb the phase-A validation):** pass the round's
real `(d1, d2)` to the DOUBLE row as well, and add one materialized column
`nb = d1 ∨ d2 = d1 + d2 − d1·d2` (degree-2 defining constraint). Neither `d1`,
`d2` nor `nb` enters any convolution relation, so no emitted math changes and the
512-case host validation stays valid. Cost: +1 column on ECDAS′ (525 → 526).

### 0.2 DESIGN §3's justification for `yP2 < p` is incorrect

DESIGN §3 (`DESIGN.md:113-123`) says a prover submits `yP2' = yP2 + p`, which is
"same point mod p, **opposite parity as bytes**", and concludes the sign of P2
flips.

That does not hold. Negation on secp256k1 is `−(x, y) = (x, p − y)`. The value
`y + p` reduces to `y`, so `(x, y + p)` is the *same* point, not its negation —
the chip's relations are all mod p and would compute the identical Q.

Decisively: **both `y` and `p − y` are already `< p`**, so a `< p` check cannot
separate them. Whatever defends the parity/sign choice, `yP2 < p` is not it. What
actually binds the sign is that `yP2` is read from memory under MEMW and the
guest — proven CPU execution — is the parity authority (`DESIGN.md:103-111`,
guest decompression at `crypto/ethrex-crypto/src/lib.rs:98-104`).

**Recommendation: keep the check, fix the reason.** The witness already emits
`y_p2_sub_p` (`witness.rs:922`), it costs ~40 cells on a one-row-per-ecrecover
table, and it removes a non-canonical input from the width audit. But phase E
should expect its negative control to come back **UNSAT (no forgery)**, unlike
the `xQ`/`yQ` controls. Predicting that now saves phase E a confusing day.

`xQ < p` and `yQ < p` are separately and genuinely load-bearing, and DESIGN's
reasoning for *those* is right: the output bytes are written to memory and the
guest keccaks them, so a `+p`-shifted coordinate hashes to a different address.
That is exactly the existing `XR_SUB_P` / N6 argument.

### 0.3 The witness cannot prove `P1` is on-curve — the ABI must specialize or the witness must grow

`Lincomb2Witness` (`witness.rs:576-612`) carries `mem_p2` and `y_p2_sub_p` but
**no `mem_p1`, no `x_p1_sub_p`, no `y_p1_sub_p`**. `check_point(p1)` runs in the
executor (`witness.rs:786`) but emits no witness, so the chip has nothing to
constrain P1 with. `layout-lock.md:48` hints at this ("P1=G constant for
ecrecover (chip may hardcode); else membership") but the ABI at
`layout-lock.md:97` and IMPL-PLAN §2 still passes P1 in `a1` as a general point.

For ecrecover P1 is always the generator (`lib.rs:115`), so specializing is free.
**Recommended: keep `a1` in the ABI, but have ECSM′ bind it with
constant-valued MEMW reads** — `BusValue::constant(...)` for each of the 8
doublewords, exactly as the Ecall receiver already pins the syscall number
(`ecsm.rs:313-314`). That asserts "memory at `a1` contains exactly G" with **zero
columns and zero constraints**, and P1's on-curve-ness becomes a compile-time
fact. Generalizing later costs +307 columns (64 coords + 225 membership + 16
canonicalization + 2 address) and a witness extension.

**Good news on timing:** `phase-b-executor` has already frozen the ABI as
`ecsm_lincomb2(q, p1, p2, u)` with `a1` a general 64-byte `xP1‖yP1`, and that is
**compatible** — the recommendation binds the bytes at `a1` to G on the *chip*
side, so the syscall signature, the executor arm, and the guest call all stay
exactly as built. No phase-B rework. What needs deciding is only whether the chip
asserts `P1 == G` (v1, free) or the witness grows a `mem_p1` block (general,
+307 columns).

### 0.4 `len ≤ 256` must be constrained by ECSM′, and `len` does not fit a byte

`len = max(u1.bits(), u2.bits())` (`witness.rs:799`). Since `N ≈ 2^256`,
`len = 256` is the *common* case (`layout-lock.md:16` measures mean 255.7, max
256). Today's analogue `LEN_K` is a `Byte` (`ecsm.rs:47`) holding an MSB
*position* ∈ [0, 255], so it never had this problem.

**`phase-c-t0table` has since built the table and independently flagged the same
obligation.** Its header (`prover/src/tables/ec_t0.rs:32-39`) reads:

> "Rows `257..512` are padding: they keep the running `LEN` key (so every row has
> a distinct key) but carry `x = y = 0`, which is not a curve point. A lookup at
> `len > 256` therefore resolves to `(0, 0)` rather than failing. **The consumer
> chip must constrain `len ≤ 256` itself** … This mirrors KECCAK_RC, whose padding
> rows likewise carry out-of-range keys with zeroed payloads."

So this is a **hard obligation on ECSM′**, not an open question: an unconstrained
`len` silently resolves to the non-curve point `(0, 0)` and the correction row
would "add" it, producing an unblinded and unproven `Q`. The table is keyed by
`len` **directly** (real rows 0..256, `NUM_REAL_ROWS = 257`, `NUM_ROWS = 512`,
`MAX_LEN = 256`), not by `len − 1`.

**Clean resolution that costs nothing.** Store `LEN_M1 = len − 1 ∈ [0, 255]` as a
byte column (AreBytes-checked like any other byte), and key the T₀ receive with
the expression `LEN_M1 + 1` via a `LinearTerm::Constant(1)` — exactly the way
today's ECSM sends `len_k − 1` on the Ecdas seed (`ecsm.rs:560-566`). The byte
range check then bounds `len ∈ [1, 256]` for free, discharging phase C's
obligation with a range check the chip was paying for anyway. No extra constraint,
no non-byte column.

### 0.5 Trailing-zero tuple aliasing is real — the joint Bit streams need their own bus id

The fingerprint accumulator skips zero elements as an optimisation, and the code
says so explicitly (`crypto/stark/src/lookup.rs:672-676`):

> "Bus elements that are zero on this row contribute nothing — skip the F×E
> multiply. (Covers the constant(0) bus-width padding plus any variable element
> that is zero on this row…)"

Because each element is weighted by a *positional* α power
(`lookup.rs:624-625`), a tuple `[a, b, c]` and a tuple `[a, b, c, 0]` on the same
bus produce **identical fingerprints**. Tuple-width padding with `constant(0)` is
a designed-in feature, so this is not a bug — but it means arity alone never
separates two chips sharing a `BusId`.

Old ECDAS sends `Bit[ts_lo, ts_hi, round]` (`ecdas.rs:228-232`) and old ECSM
receives `Bit[ts_lo, ts_hi, i]` (`ecsm.rs:538-544`). Old and new chips coexist
until phase G (IMPL-PLAN §0.1) and can both be live in one proof. If ECDAS′ sends
`Bit[ts, round, stream]` with `stream = 0`, it aliases an old-ECDAS send exactly.

**Use a distinct bus id for the joint streams** (id 32; 29 goes to Addend), and
make the stream tag `∈ {1, 2}` rather than `{0, 1}` so a trailing zero can never
alias even within the new bus.

---

## 1. ECSM′ column map (1 row per ecrecover)

Convention: `B` = byte column (AreBytes authority), `HW` = halfword column
(IsHalfword), `bit` = boolean (IS_BIT *constraint*, no bus send), `fe` = raw
field carry (IsHalfword with offset). Offsets are cumulative and assume the
block order shown; they are a proposal, not a constraint.

Every field of `Lincomb2Witness` (`witness.rs:576-612`) maps to exactly one block.
The rightmost column names the check that bounds it.

| # | Block | Cols | Offset | Width | Range authority | Witness field |
|---|---|---:|---:|---|---|---|
| 1 | `TIMESTAMP_0/1` | 2 | 0 | packed | MEMW consistency | — |
| 2 | `ADDR_OUT_0/1` (a0) | 2 | 2 | packed | MEMW | — |
| 3 | `ADDR_P2_0/1` (a2) | 2 | 4 | packed | MEMW | — |
| 4 | `ADDR_U_0/1` (a3) | 2 | 6 | packed | MEMW | — |
| 5 | `X_P2`, `Y_P2` | 64 | 8 | B | **inherited**: byte-checked at store time (`store.rs`), same as today's xG/k (`ecsm.rs:470`) | `x_p2`, `y_p2` |
| 6 | `MEM_X2` | 32 | 72 | B | AreBytes (paired) | `mem_p2.x2` |
| 7 | `MEM_Q0` | 32 | 104 | B | AreBytes (paired) | `mem_p2.q0` |
| 8 | `MEM_C0` | 64 | 136 | fe | IsHalfword ×63 + `ColIsZero(c[63])` | `mem_p2.c0` |
| 9 | `MEM_Q1` | 33 | 200 | B | AreBytes ×32 paired + `IS_BIT(q1[32])` | `mem_p2.q1` |
| 10 | `MEM_C1` | 64 | 233 | fe | IsHalfword ×63 + `ColIsZero(c[63])` | `mem_p2.c1` |
| 11 | `Y_P2_SUB_P` | 16 | 297 | HW | IsHalfword ×16 + 7 CarryBit + 1 OverflowRequired | `y_p2_sub_p` |
| 12 | `X_P12`, `Y_P12` | 64 | 313 | B | **inherited** from the precompute row's `XR`/`YR`, byte-checked in ECDAS′ | `x_p12`, `y_p12` |
| 13 | `U1` bits | 256 | 377 | bit | `IS_BIT` ×256 + zero-on-padding | `u1` |
| 14 | `U2` bits | 256 | 633 | bit | `IS_BIT` ×256 + zero-on-padding | `u2` |
| 15 | `U1_SUB_N` | 16 | 889 | HW | IsHalfword ×16 + 8 carry constraints | `u1_sub_n` |
| 16 | `U2_SUB_N` | 16 | 905 | HW | IsHalfword ×16 + 8 carry constraints | `u2_sub_n` |
| 17 | `LEN_M1` | 1 | 921 | B | AreBytes — which *is* the `len ≤ 256` bound; T₀ receive is keyed `LEN_M1 + 1` (§0.4) | `len` (as `len − 1`) |
| 18 | `X_Q`, `Y_Q` | 64 | 922 | B | **inherited** from the correction row's `XR`/`YR` | `x_q`, `y_q` |
| 19 | `X_Q_SUB_P` | 16 | 986 | HW | IsHalfword ×16 + 8 carry constraints | `x_q_sub_p` |
| 20 | `Y_Q_SUB_P` | 16 | 1002 | HW | IsHalfword ×16 + 8 carry constraints | `y_q_sub_p` |
| 21 | `X_T0POW`, `Y_T0POW` | 64 | 1018 | B | **inherited** from the preprocessed T₀ table (constant by construction) | `x_t0_pow`, `y_t0_pow` |
| 22 | `N1`, `N2`, `N3`, `NC` (addend publish counts) | 4 | 1082 | fe | none needed — balance-forced, gate contract C5 (§3) | — (derived) |
| 23 | `STATUS` | 1 | 1086 | fe | bound to `OK` via §4.4; written to **`x10`** by a chip-emitted register MEMW (§0.0) | — (executor) |
| 24 | `S_INV` | 1 | 1087 | fe | §4.4 | — |
| 25 | `OK` | 1 | 1088 | bit | `IS_BIT`; `OK·(1−MU) = 0` | — |
| 26 | `MU` | 1 | 1089 | bit | `IS_BIT` | — |
| | **`NUM_COLUMNS`** | **1,090** | | | | |

**Fields deliberately given no columns.** `x_p1`/`y_p1` — pinned to G by
constant-valued MEMW reads (§0.3). `x_t0`/`y_t0` — T₀ is a fixed constant
(`witness.rs:482-489`); hardcode it in the constraint body via `b.const_base(...)`
the way `P_BYTES` already is (`ecsm.rs:713`). `steps` — those are ECDAS′ rows.

**1,090 vs DESIGN's ≈1,310.** The difference is P1 (−64 coords, and no
membership block was ever in the witness to cost) plus DESIGN double-counting
`xP2` canonicalization, which the witness does not emit. ECSM′ is one row per
ecrecover, so this is bookkeeping, not a win — but the count should be honest.

**Blocks reused verbatim from today's ECSM.** Rows 6–10 are a second copy of the
`X2`/`Q0`/`C0`/`Q1`/`C1` membership machinery (`ecsm.rs:50-54`, relations at
`:742-769`) applied to P2 instead of G — `membership_witness`
(`witness.rs:669-705`) explicitly "reuses the exact `x2` and `yG` convolutions".
Rows 11, 15, 16, 19, 20 are the `XG_SUB_P`/`K_SUB_N`/`XR_SUB_P` pattern
(`ecsm.rs:55-57`, `OverflowKind` `:636-677`, `carry_chain` `:796-830`): 16
halfword columns each, 16 IsHalfword sends, and 8 constraints (7 × `µ·c_i·(1−c_i)`
degree 3, plus `µ·(1−c_7)` degree 2). The eight word-carries are **virtual** —
computed as expressions with `INV_SHIFT_32`, never stored — which is why a
canonicalization block is only 16 columns.

---

## 2. ECDAS′ column map (1 row per joint step)

The λ/xR/yR convolution core ports **byte-for-byte**: `Q0/Q1/Q2` +`C0/C1/C2`,
same offsets `CARRY_OFFSET_{LAMBDA,XR,YR}` (`ecdas.rs:24-26`), same `rq()`
shifted-quotient term (`ecdas.rs:332-345`), same `conv_carry` recurrence
(`:400-418`). `witness.rs:466-469` states the same thing from the witness side and
the phase-A tests re-check every emitted row against those relations.

What changes: `XG/YG` → `XB/YB` (per-row addend), `NEXT_OP` → the joint
bookkeeping block, plus `PHASE`/`NEXT_PHASE` for the segment split (§4.2).

| # | Block | Cols | Offset | Width | Range authority | Notes |
|---|---|---:|---:|---|---|---|
| 1 | `TIMESTAMP_0/1` | 2 | 0 | packed | chain key | unchanged |
| 2 | `XB`, `YB` (addend) | 64 | 2 | B | **inherited** via Addend-bus tuple equality (§3) — *no new byte checks* | was `XG`/`YG` |
| 3 | `XA`, `YA` (accumulator in) | 64 | 66 | B | inherited via Ecdas′ tuple | unchanged |
| 4 | `ROUND` | 1 | 130 | B | AreBytes, paired with `Q0[32]` | unchanged |
| 5 | `OP` | 1 | 131 | bit | `IS_BIT` | 0 = double, 1 = add |
| 6 | `XR`, `YR` (result) | 64 | 132 | B | AreBytes ×32 paired each | unchanged |
| 7 | `LAMBDA` | 32 | 196 | B | AreBytes ×16 paired | unchanged |
| 8 | `Q0` | 33 | 228 | B | AreBytes | λ relation |
| 9 | `C0` | 64 | 261 | fe | IsHalfword ×63 + `ColIsZero` | λ carries |
| 10 | `Q1` | 33 | 325 | B | AreBytes | xR relation |
| 11 | `C1` | 64 | 358 | fe | IsHalfword ×63 + `ColIsZero` | xR carries |
| 12 | `Q2` | 33 | 422 | B | AreBytes | yR relation |
| 13 | `C2` | 64 | 455 | fe | IsHalfword ×63 + `ColIsZero` | yR carries |
| 14 | `D1`, `D2` | 2 | 519 | bit | `IS_BIT` ×2 | digit bits of **this row's** round — must now be set on DOUBLE rows too (§0.1) |
| 15 | `NB` | 1 | 521 | bit | `IS_BIT` + `NB = D1 + D2 − D1·D2` | the restored `nb_or` (§0.1) |
| 16 | `S1`, `S2`, `S3` | 3 | 522 | bit | `IS_BIT` ×3; defined from `D1`,`D2` on main-chain adds (§3) | one-hot addend {P1, P2, P12} |
| 17 | `S_CORR` | 1 | 525 | bit | `IS_BIT` | correction row consumes the T₀ constant |
| 18 | `PHASE` | 1 | 526 | B | AreBytes (pairs with `Q1[32]`) | 0 = precompute, 1 = main, 2 = correction (§4.2) |
| 19 | `NEXT_PHASE` | 1 | 527 | B | AreBytes (pairs with `Q2[32]`) | rides the outgoing tuple |
| 20 | `MU` | 1 | 528 | bit | `IS_BIT` | live flag |
| | **`NUM_COLUMNS`** | **529** | | | | |

**529 vs layout-lock's 525.** The delta is +1 `NB` (§0.1), +1 `S_CORR`, +2
`PHASE`/`NEXT_PHASE` (§4.2), less the `NEXT_OP` that goes away. +0.8% on the
volume table — immaterial to the verdict (§5), but it is a real +4 and layout-lock
should be updated rather than quietly missed.

**`XB`/`YB` are correctly absent from the AreBytes list.** Today's AreBytes loop
covers exactly `[LAMBDA, XR, YR, Q0, Q1, Q2]` (`ecdas.rs:188-195`) — `XG`/`YG` are
*not* in it, matching `chips-map.md:57` ("equality with seed via Ecdas tuple
matching (NOT re-checked)"). ECDAS′ keeps that list unchanged, so the addend costs
no new range checks. §3 justifies why that is sound for a *varying* addend.

---

## 3. Addend bus spec (bus id 29)

### 3.1 Registration — three sync points

`prover/src/tables/types.rs`, id **29** (free: `Ecdas = 28`, `Bit = 30`,
`GlobalMemory = 31`, and 29 is an unassigned gap):

1. enum variant, `types.rs:255-362` — `Addend = 29,`
2. `BusId::name()`, `types.rs:365-393` — `BusId::Addend => "Addend",`
3. `TryFrom<u64>`, `types.rs:396-425` — `29 => Ok(BusId::Addend),`

Plus id **32** for `JointBit` per §0.5 (same three points).

### 3.2 Tuple

```
Addend[id=29 | ts_lo, ts_hi, sel, x(32), y(32)]
```
`sel ∈ {1, 2, 3, 4}` — `1 = P1`, `2 = P2`, `3 = P12`, `4 = −2^len·T₀`.
Never 0, so §0.5's trailing-zero aliasing cannot bite even if the tuple is later
widened. Coordinates packed with `point_coord_busvalues` (`ecsm.rs:274`), the same
helper `ecdas_tuple` uses, so publisher and consumer pack identically.

### 3.3 Publisher — ECSM′, four sends

```rust
// sel = 1 : P1 (= G, constant-valued)     mult = Multiplicity::Column(cols::N1)
// sel = 2 : P2 (from memory)              mult = Multiplicity::Column(cols::N2)
// sel = 3 : P12 (from the precompute row) mult = Multiplicity::Column(cols::N3)
// sel = 4 : −2^len·T₀ (from the T₀ table) mult = Multiplicity::Column(cols::NC)
```
`N1`/`N2`/`N3`/`NC` are witnessed counts (§1 row 22). `NC` is additionally
constrained `= OK` (exactly one correction row per proven ecall).

### 3.4 Consumer — ECDAS′, one receive

```rust
BusInteraction::receiver(
    BusId::Addend,
    Multiplicity::Linear(vec![          // S1 + S2 + S3 + S_CORR
        LinearTerm::Column { coefficient: 1, column: cols::S1 },
        LinearTerm::Column { coefficient: 1, column: cols::S2 },
        LinearTerm::Column { coefficient: 1, column: cols::S3 },
        LinearTerm::Column { coefficient: 1, column: cols::S_CORR },
    ]),
    vec![ts_lo(), ts_hi(), sel_expr(), /* XB 32 */, /* YB 32 */],
)
```
with
```
sel_expr = 1·S1 + 2·S2 + 3·S3 + 4·S_CORR      (degree 1, linear in tuple-bound bits)
```

**Note this is `Linear`, not `Sum3`.** The brief specifies `Sum3(s1,s2,s3)`
(`lookup.rs:1350`), which covers the three scalar addends but leaves the
correction row unable to receive. `Multiplicity::Linear` (`lookup.rs:1356`) takes
the fourth term at no extra cost — still one interaction.

The **precompute row uses `sel = 2`** (its addend genuinely is P2, `witness.rs:807`
passes `p2` as the addend). So `N2` counts every P2 add *plus one* for the
precompute. That is fine: counts are witnessed and balance-forced.

### 3.5 Why balance forces correctness

The receive is keyed by `[ts, sel, x, y]` and multiplicities are non-negative
witnessed counts, so LogUp balance on bus 29 gives, per `ts`:

> for each `sel`, the number of ECDAS′ rows receiving `(sel, x, y)` equals the
> count ECSM′ published for that `sel` — **and the coordinates must match
> exactly**, because `x`/`y` are part of the keyed tuple, not a payload.

A prover who wants row `j` to add some `P*` of their choosing must make ECSM′
publish `(sel, P*)`; but ECSM′'s published coordinates are themselves pinned:
`sel = 1` to the G constant, `sel = 2` to the MEMW-bound memory bytes at `a2`,
`sel = 3` to the precompute row's output (§4.2), `sel = 4` to the preprocessed T₀
table entry keyed by `LEN_M1`. There is no free coordinate anywhere in the chain.
Counts need no range check under gate contract C5 (a negative count cannot be
represented; an inflated count leaves an unmatched send).

### 3.6 Why `XB`/`YB` need no byte checks

This is the C4 inheritance the gate already used for `YR`, extended to a varying
addend. The receiving row's `XB`/`YB` columns appear *inside the keyed tuple*, so
balance forces them equal, limb-for-limb, to the publisher's columns. It therefore
suffices that every publishable value is byte-bounded at its source:

| `sel` | source | byte-ness from |
|---|---|---|
| 1 | G | compile-time constant |
| 2 | memory at `a2` | store-time AreBytes (`store.rs`), the same authority today's `xG`/`k` rely on (`ecsm.rs:470`) |
| 3 | precompute row `XR`/`YR` | ECDAS′'s own AreBytes on `XR`/`YR` (§2 row 6) |
| 4 | T₀ table | preprocessed constant, committed |

Row 3 looks circular but is not: the precompute row range-checks its *own*
`XR`/`YR` on that row, ECSM′ receives those checked values through the phase-0
drain (§4.2), and only then republishes them. The dependency is a DAG, not a
cycle.

**Caveat carried forward from today's design:** byte-bounded means `< 2^256`, not
`< p`. Interior non-canonical representatives are admissible because every
relation is mod p with the quotient absorbing the slack — the existing argument at
`chips-map.md:93-100`. That argument must be re-run in phase E for the *addend*
(today the addend was a single loop-invariant canonical point; now it varies and
includes P12, which is an interior chip output). The width audit is the place this
could bite.

---

## 4. The four hard parts

### 4.1 Double-row cancellation — re-derived from the live `eval` body

Everything about the `Sum3`-gated (now `Linear`-gated) Addend receive being silent
on doubles rests on this, so here is the algebra straight out of
`EcdasConstraints::s_i` (`ecdas.rs:348-397`), not from a doc.

Let `op` be `cols::OP`, and let `xg(·)`, `yg(·)` denote the addend limbs (`XB`/`YB`
in ECDAS′). The three relations at limb `i`:

**Lambda** (`ecdas.rs:364-379`):
```
op·( ya(i) − yg(i) + Σ_{j≤i} lam(j)·(xg(i−j) − xa(i−j)) )
  + (1 − op)·( Σ_{j≤i} 2·lam(j)·ya(i−j) − 3·xa(j)·xa(i−j) )
  + rq(i, Q0)
```
At `op = 0` the entire first product is multiplied by zero. Every occurrence of
`xg`/`yg` in this relation lives inside that product, so the constraint value is
**independent of the addend**. The surviving `(1 − op)` branch reads only `lam`,
`ya`, `xa`; `rq` reads only `MU`, the constants `R`/`P`, and `Q0`
(`ecdas.rs:332-345`). ✓

**Xr** (`ecdas.rs:380-387`):
```
Σ_{j≤i} lam(j)·lam(i−j) − xa(i) − xg(i) − xr(i) − (1 − op)·( xa(i) − xg(i) ) + rq(i, Q1)
```
At `op = 0`, `(1 − op) = 1`:
```
= Σ lam·lam − xa(i) − xg(i) − xr(i) − xa(i) + xg(i) + rq
= Σ lam·lam − 2·xa(i) − xr(i) + rq
```
The `−xg(i)` and `+xg(i)` cancel **exactly**. ✓ (At `op = 1` the bracket vanishes
and it reduces to the chord form `xr = λ² − xa − xg`; at `op = 0` to the tangent
form `xr = λ² − 2·xa`. Both correct.)

A note on `layout-lock.md:87`, which writes the cancelling term as
`(1−op)(xg−xa)`: the source has `− (1 − op)·(xa − xg)`, which is the same thing
once the leading minus is folded in. The doc and the code agree.

**Yr** (`ecdas.rs:388-395`):
```
Σ_{j≤i} lam(j)·( xa(i−j) − xr(i−j) ) − ya(i) − yr(i) + rq(i, Q2)
```
No `xg`/`yg` occurs at all, for either value of `op`. ✓

**Conclusion — verified, not assumed.** On `op = 0` no convolution constraint
reads `XB`/`YB`. The witness agrees: the double row is built with
`&zero_pt = (0, 0)` as its addend (`witness.rs:786-789`, `:831`). So double rows
may carry `XB = YB = 0` and stay silent on the Addend bus.

**One obligation this does *not* discharge.** The addend columns are unconstrained
on double rows, so nothing stops a prover from putting garbage there instead of
zero. That is harmless for the three relations (just shown) — but it must also be
harmless for the *bus*, and it is, because the Addend receive multiplicity is
`S1 + S2 + S3 + S_CORR`, which must be 0 on a double row. That needs an explicit
constraint tying the selectors to `OP`:
```
OP = S1 + S2 + S3 + S_CORR          (degree 1)
```
i.e. adds have exactly one selector set, doubles none. Combined with `IS_BIT` on
each selector this also forbids two selectors at once. **Without this constraint
the cancellation is real but the gating is forgeable** — a prover could set
`S2 = 1` on a double row and mint a spurious Addend receive. It is one degree-1
constraint; do not omit it.

### 4.2 The two telescoping breaks — proposal: three keyed segments

The break, restated from the source. `a = prev.r` holds on every main-chain row,
but not at two places:

- **Precompute** (`witness.rs:807-822`): `a = P1`, addend `= P2`, result `= P12`.
  Entirely off the accumulator line.
- **Correction** (`witness.rs:896-911`): `a =` the last accumulator, addend
  `= −2^len·T₀`, result `= Q`.

And `round` cannot discriminate them: the precompute and correction rows are both
emitted with `round = 0` (`witness.rs:810`, `:900`), while the main loop also
produces genuine `round = 0` rows on its last iteration (`witness.rs:825`). **Any
scheme keyed on `round` is ambiguous by construction.**

**Proposal — split the chain into three separately-keyed segments** by adding
`PHASE` to the Ecdas′ tuple and `NEXT_PHASE` to the outgoing side:

```
Ecdas'[id=28 | ts_lo, ts_hi, phase, accX(32), accY(32), round, op]
```
(Note the tuple **drops** today's `genX`/`genY` — the addend now varies per row
and arrives on bus 29, so it must not be part of the accumulator state. That is a
64-element narrowing of the tuple, which costs nothing in committed cells because
LogUp aux is per *interaction*, not per element.)

| segment | ECSM′ sends (seed) | ECSM′ receives (drain) | rows |
|---|---|---|---|
| `phase = 0` precompute | `[ts, 0, xP1, yP1, round=0, op=1]` | `[ts, 0, xP12, yP12, round=0, op=1]` | exactly 1 |
| `phase = 1` main chain | `[ts, 1, xT0, yT0, round=LEN_M1, op=0]` | — (hands off to phase 2) | `len` doubles + adds |
| `phase = 2` correction | — (received from phase 1) | `[ts, 2, xQ, yQ, round=0, op=1]` | exactly 1 |

Transitions: a phase-1 row sends `NEXT_PHASE = 1` normally, and the *last* one
sends `NEXT_PHASE = 2`; the phase-2 row is the unique receiver of that tuple and
drains to ECSM′.

**Why a prover cannot forge the distinction.** `phase` is inside the keyed tuple,
so a row can only execute as phase `q` if some sender published phase `q`. The
phase-0 seed and drain are ECSM′ sends/receives with multiplicity `OK`, so
**exactly one** phase-0 row can exist per proven ecall, and its `a` is pinned to
the G constant and its result to the value ECSM′ republishes as `sel = 3`. The
phase-2 row is reachable only via a phase-1 row that chose `NEXT_PHASE = 2`, and
its drain must match ECSM′'s `[ts, 2, xQ, yQ, …]` receive, which is the same
`X_Q`/`Y_Q` the canonicalization blocks and the output MEMW write bind. So both
special rows are pinned at both ends.

**What this does *not* settle — the phase-1 counting argument.** With `PHASE`,
`NB` (§0.1) and the round bookkeeping `round' = round − 1 + NB` on doubles /
`round' = round − 1` on adds, the intended invariant is "exactly `len` doublings,
each round visited once, each set digit consumed exactly once". Proving that is
**L6, the redo IMPL-PLAN §7 calls the riskiest work in the project**, and I am not
claiming it here. Two specific obligations phase E must discharge, which the
single-scalar L6 did not have:

1. **Two interleaved streams.** The old argument counted one Bit stream against
   `k`'s set bits. Now `u1` and `u2` are counted independently on the same rows,
   and `NB = D1 ∨ D2` couples them. A prover setting `D1 = 1, D2 = 0` on a row
   whose true digits are `(0, 1)` still balances `NB`; only the *per-stream*
   JointBit balance separates them. Worth an explicit sub-lemma.
2. **`NEXT_PHASE` cannot be used to escape early.** A prover who sends
   `NEXT_PHASE = 2` before consuming all rounds shortens the chain. What forbids
   it is that the phase-2 drain is the only exit and the JointBit receivers in
   ECSM′ carry multiplicity `u1_bit(i)`/`u2_bit(i)` for **all** `i`, so any set
   digit left unconsumed leaves an unmatched receiver. That is the same shape as
   today's argument (`chips-map.md:82-87`) but now needs `round` monotonicity
   across a phase boundary.

I flag both rather than assert them.

### 4.3 The three canonicalization blocks

All three follow the existing `XG_SUB_P` / `K_SUB_N` / `XR_SUB_P` pattern exactly
(`ecsm.rs:55-57`, `OverflowKind` `:636-677`, `carry_chain` `:796-830`, emission
`:892-907`). Per block:

- **16 columns**, halfword-packed. The witness emits 32 LE *bytes*
  (`sub_witness`, `witness.rs:662-665`); the trace builder repacks them with
  `write_halfwords` (`ecsm.rs:141-147`), exactly as today.
- **16 IsHalfword sends** (`ecsm.rs:496-516` pattern).
- **8 constraints**: 7 × `µ·c_i·(1−c_i)` (degree 3) + 1 × `µ·(1−c_7)` (degree 2) —
  substitute `OK` for `µ` per §4.4. The eight word-carries stay **virtual**
  (expressions built from `INV_SHIFT_32`), so no carry columns.
- The relation proved is `p + v_sub_p = v + 2^256`, i.e. the addition must
  overflow, i.e. `v < p` strictly (`ecsm.rs:631-635`).

| block | `OverflowKind` variant | constant addend | sum source | witness field |
|---|---|---|---|---|
| `Y_P2_SUB_P` | `YP2LtP` (new) | `P_BYTES` | `Y_P2` (32 bytes) | `y_p2_sub_p` |
| `X_Q_SUB_P` | `XQLtP` (new) | `P_BYTES` | `X_Q` (32 bytes) | `x_q_sub_p` |
| `Y_Q_SUB_P` | `YQLtP` (new) | `P_BYTES` | `Y_Q` (32 bytes) | `y_q_sub_p` |

Plus `U1_SUB_N` / `U2_SUB_N` reusing `KLtN` against `N_BYTES`. Note today's
`KLtN` reads its sum from **256 individual bit columns**
(`OverflowKind::sum_is_bits`, `ecsm.rs:673-676`, consumed at `:810-815`) — both
scalar blocks must keep that variant since `U1`/`U2` are bit-decomposed for the
JointBit multiplicities.

Load-bearing status, corrected per §0.2: `X_Q_SUB_P` and `Y_Q_SUB_P` are genuine
forgery defences (output bytes are hashed by the guest). `Y_P2_SUB_P` should be
kept for width-audit hygiene but its DESIGN §3 justification does not hold, and
phase E's negative control for it will likely return UNSAT.

### 4.4 The mu-gated error row

**The problem, precisely.** The CPU sends on `Ecall` (id 19) for every ecall
(`cpu.rs:962-976`), so every syscall needs a receiver or bus 19 unbalances — the
unmatched-`Print` note at `syscalls.rs:36-40` is the cautionary tale, and ECSM's
receiver is `ecsm.rs:307-316`. On the `status ≠ 0` path there is no chain to
prove, but the receive must still fire. Today's padding trick (all-zero columns
with `µ = 0`) is unavailable because `µ = 0` would also kill the Ecall receive.

**Proposal — split the single flag into two.**

| flag | meaning | gates |
|---|---|---|
| `MU` | this row is a real ecall | Ecall receive; **all** MEMW (operand reads + status write) |
| `OK` | `status == 0`, full chain proven | JointBit receivers, Addend publishes, Ecdas′ seeds/drains, T₀ receive, and every convolution / carry constraint |

with `IS_BIT(OK)` and `OK·(1 − MU) = 0` (so `OK ⇒ MU`).

Concretely: everywhere today's ECSM body writes `b.main(0, cols::MU)` inside a
relation — the `µ·p²` and `µ·b` terms (`ecsm.rs:756-768`), the `rq()` gate
(`ecdas.rs:337`), the carry-bit gates (`ecsm.rs:897-905`) — substitute `OK`. Then
an error row sets `OK = 0` and every witness column to zero, and all relations
close at zero carries by exactly the argument the padding rows already use
(`ecsm.rs:12-16`, `ecdas.rs:10-12`).

**Executor obligation this creates (for `phase-b-executor`):** on the error path
the executor must still perform all operand reads and the status write, so the
MEMW schedule is identical on both paths and can stay gated by `MU`. A natural
implementation that early-returns before reading would desynchronise the
timestamps. Address-guard failures are a separate class and should keep trapping
(as ECSM does at `execution.rs:432-437`) — they are guest-guarded, since the
addresses are the guest's own stack arrays, never attacker bytes.

**Why it stays sound — and the constraint that makes it so.** IMPL-PLAN §2 argues
"a lying status only wastes cycles". That is right for `status ≠ 0`: the guest
falls back to `ProjectivePoint::lincomb`, which is proven CPU execution, so
over-claiming an error is merely expensive. But the converse needs enforcing —
**`status == 0` must oblige the chain proof**, or a prover sets `OK = 0`
(proving nothing), writes `status = 0`, and the guest happily reads a fabricated Q
out of memory. Nothing in the plan currently states the mechanism. Add:

```
OK · STATUS         = 0        (OK = 1  ⇒  STATUS = 0)
STATUS · S_INV      = 1 − OK   (OK = 0  ⇒  STATUS ≠ 0, witnessed inverse)
```
Two constraints and one column (`S_INV`, §1 row 24). This preserves IMPL-PLAN's
distinct per-variant error codes — `STATUS` may be any non-zero value when
`OK = 0`, and the guest only tests `!= 0`. A boolean `STATUS = 1 − OK` would also
work and drop `S_INV`, at the cost of losing the debug-distinguishable codes.

---

## 5. Cell-count check — the −74.3% verdict, recomputed

Cost rule (`chips-map.md:104`): committed base cells = logic columns +
1.5 × interactions.

### 5.1 Interaction counts, recounted from source

I recounted both live chips from `bus_interactions()` and reproduce the header
note at `chips-map.md:3-7` exactly, which is the check that the method is right.

**ECDAS today** (`ecdas.rs:152-261`): 1 Ecdas receive + 98 AreBytes (6 bases × 16
pairs at `:188-199`, plus `(ROUND, Q0[32])` and `(Q1[32], Q2[32])` at `:200-201`)
+ 189 IsHalfword (3 × 63, `:213-225`) + 1 Bit send + 1 Ecdas send = **290** ✓
(chips-map: "ECDAS 388→290").

**ECSM today** (`ecsm.rs:300-589`): 1 Ecall + 15 MEMW (3 register reads + 4 xG + 4
k + 4 xR writes) + 65 AreBytes (4 bases × 16 at `:460-464`, plus the lone
`q1[32]` at `:465-469`) + 174 IsHalfword (63 + 63 + 16 + 16 + 16, `:482-516`) + 1
Zero + 256 Bit receivers + 1 Bit send + 2 Ecdas = **515** ✓ ("ECSM 579→515").

**ECDAS′** (§2 layout): 1 Ecdas receive + **1 Addend receive** + 98 AreBytes
(unchanged — `XB`/`YB` are not checked, §3.6) + 189 IsHalfword + **2 JointBit
sends** (one per stream, mults `D1`/`D2`) + 1 Ecdas send = **292**.
DESIGN §2's 292 is confirmed. The new bit columns (`D1`,`D2`,`NB`,`S*`,`PHASE`)
cost **constraints, not interactions** — `IS_BIT` is emitted in the body
(`ecdas.rs:428-433`), not sent on a bus.

**ECSM′** (§1 layout, status in `x10` per §0.0): 1 Ecall + ~28 MEMW (4 register
reads for a0/a1/a2/a3 + 8 for P1 + 8 for P2 + 8 for u1‖u2 + 8 for the 64-byte
output + 1 register **write** of `STATUS` to x10 = 37 if P1 is read; ~28 if the
P1 read is dropped) + ~49 AreBytes (the P2 membership `x2`/`q0`/`q1` = 97 bytes →
48 pairs + 1) + 206 IsHalfword (126 membership carries + 5 × 16 canonicalization)
+ 2 Zero + 512 JointBit receivers + 4 Addend publishes + 1 T₀ receive + 4 Ecdas′
(phase-0 send/receive, phase-1 send, phase-2 receive) ≈ **807** (±9 on the P1
question). One row per ecrecover, so the ±9 moves the per-ecrecover total by
0.003%.

### 5.2 Cells per row

| row | logic | interactions | cells |
|---|---:|---:|---:|
| ECDAS today (pre-pairing) | 521 | 388 | 1,103 |
| ECDAS today (**post-`42ba68ff`**) | 521 | 290 | **956** |
| ECSM today (pre-pairing) | 667 | 579 | 1,536 |
| ECSM today (**post-`42ba68ff`**) | 667 | 515 | **1,440** |
| **ECDAS′** | 529 | 292 | **967** |
| **ECSM′** | 1,090 | 807 | **2,301** |

ECDAS′ at layout-lock's 525 columns would be 963, matching DESIGN §2 exactly; my
+4 columns (§2) put it at 967, **+0.4%**.

### 5.3 Per ecrecover

- **Today, pre-pairing**: 4 chains × 382 rows × 1,103 + 4 × 1,536 = **1.692M**
- **Today, post-pairing (the live baseline)**: 1,528 × 956 + 4 × 1,440 = **1.467M**
- **lincomb2, post-pairing**: 449.1 rows × 967 + 2,301 = **0.437M**
  (mean row count measured in phase A, `layout-lock.md:16-20`)
- lincomb2 worst case (471 rows): 0.458M

### 5.4 Does the verdict move? — the denominator does, the physics does not

| comparison | reduction |
|---|---:|
| lincomb2+pairing **vs pre-pairing baseline** (DESIGN's headline) | **−74.2%** |
| lincomb2+pairing **vs the current post-pairing baseline** | **−70.2%** |

DESIGN's absolute numbers hold — I get 0.437M against its 0.43M, and −74.2%
against its −74.3%, from an independent recount. **But every percentage in
DESIGN's verdict table is denominated against the *pre*-pairing 1.69M, and the
pairing has already shipped in `42ba68ff`.** Measured against what is on the
branch today, lincomb2's marginal win is **−70.2%**, not −74.3%. The −74.3% figure
silently re-banks a win that is already in the bank.

There is a trap here worth naming: DESIGN's "lincomb2 alone = −70.4%" is
*unpaired lincomb2 vs unpaired today*, and my −70.2% is *paired lincomb2 vs paired
today*. They are different quantities that happen to land within 0.2 points of
each other. IMPL-PLAN §9 currently reads "−70.4% standalone, −74.3% with the
AreBytes pairing already landed", which reverses the sense of both.

**Recommendation:** state the phase-H target as **−70% EC committed cells against
the post-`42ba68ff` baseline**, and keep 0.437M / 1.467M as the numbers to
reproduce. The engineering verdict is unchanged — it still clears the 2× bar by
better than 3× — but the headline should not double-count.

---

## 6. Open questions for the lead

1. **P1 general or specialized to G? (§0.3 — the one real decision left.)** The
   witness has no `mem_p1`, so a general P1 is currently *unprovable*. My
   recommendation is constant-valued MEMW binding to G for v1. Phase B's shipped
   ABI already accommodates either choice, so this is a chip-side decision only —
   but it must be made, because "general P1" is not currently an option the witness
   supports. Generalizing later is +307 ECSM′ columns plus a witness extension.

1b. **Status in register vs memory (§0.0).** Phase B shipped the register form and
   it is sound (COMMIT precedent) and cheaper. IMPL-PLAN §2 and §10 item 1 say the
   opposite and should be corrected — flagging because my earlier map is what put
   the wrong claim there.

2. **Extend the witness with double-row digit bits? (§0.1.)** Required for `NB`.
   It is additive (`d1`/`d2` on `JointSel::Double` rows, plus an `nb` field or let
   the chip derive it) and changes no emitted math, so phase A's validation
   survives. But it touches `witness.rs`, which IMPL-PLAN §0.2 calls the spec —
   worth an explicit blessing rather than an agent doing it unilaterally.

3. **Does `S_CORR` belong in the selector, or should the correction row receive
   the T₀ constant on its own bus?** I chose `sel = 4` on the Addend bus (one
   interaction, §3.4). The alternative — ECDAS′ receiving directly from the T₀
   table keyed by `len` — removes ECSM′ as a middleman but adds a second bus to
   ECDAS′, the volume table. Recommend `sel = 4`; flagging because it changes what
   phase C's table must publish to.

4. ~~**T₀ table: store `+2^i·T₀` or `−2^i·T₀`?**~~ **Resolved by phase C — no
   action.** IMPL-PLAN §10 item 3 left this open. The witness settles it
   (`witness.rs:888` adds `neg_tpow = (tpow.x, p − tpow.y)`, while
   `x_t0_pow`/`y_t0_pow` record the *un*-negated point, `:936-937`), and
   `ec_t0.rs` stores the **negation** — column `Y` holds
   `y(−2^LEN·T₀) = p − y(2^LEN·T₀)`, with `X` unchanged since `x(−P) = x(P)`, and
   `ecsm::tests::lincomb2_table_tests` asserts the match against real witness rows.
   That is the convention §3.3's `sel = 4` publishes; ECDAS′ does no implicit
   negation. Recommend IMPL-PLAN §10 item 3 be marked decided.

5. ~~**Keying the T₀ table on `len − 1`?**~~ **Resolved — but it hands phase D a
   hard obligation.** Phase C keys on `len` directly (257 real rows 0..256, padded
   to 512) and its header explicitly requires the consumer to constrain
   `len ≤ 256`, because padding rows keep live keys with a `(0, 0)` payload
   (§0.4). ECSM′ discharges it for free by carrying `LEN_M1` as a byte and keying
   the receive `LEN_M1 + 1`. **This must not be dropped** — it is the one place
   where phase C's design assumes phase D does something, and it is invisible from
   the phase-D side unless someone reads the table's header.

6. **Does the width audit survive a varying addend? (§3.6.)** Today's addend was a
   single canonical loop-invariant point; ECDAS′'s can be P12, an interior chip
   output that is only byte-bounded (`< 2^256 ≈ 5.4p`), not canonical. The
   quotient/carry headroom argument at `chips-map.md:93-100` needs re-running with
   that in mind. I did not attempt the bound arithmetic.

7. **Old and new chips both live until phase G** — beyond the bus-id aliasing in
   §0.5, is there any scenario where one proof contains both an `ecsm_mul` and an
   `ecsm_lincomb2` call? If yes, the `Ecdas` bus (id 28) is shared between old
   ECDAS and ECDAS′ and the tuples must not alias either. My §4.2 tuple drops
   `genX`/`genY` (64 elements) and inserts `phase`, so the arities differ — but per
   §0.5 differing arity is *not* sufficient. Safest is a separate id for the joint
   chain; I did not cost that.

---

## 7. What I could not verify

Stated explicitly rather than papered over.

- **The L6 counting argument.** §4.2 proposes a mechanism (`PHASE` +
  `NEXT_PHASE` + `NB` + two JointBit streams) and names two specific obligations,
  but I did not prove that the schedule is forced. That is phase E's job and
  IMPL-PLAN §11 already ranks it the riskiest work; nothing here should be read as
  discharging it.

- **Whether `yP2 < p` is load-bearing at all.** I showed DESIGN §3's stated reason
  is wrong (§0.2) and could not construct a replacement attack. I could not prove
  it is *redundant* either — that would need the full width audit of item 6. Keep
  the check; treat its necessity as open.

- **Exact ECSM′ interaction count.** The 807 in §5.1 is my count over a layout I
  proposed, not a count of existing code. The MEMW figure in particular depends on
  the final ABI (§6.1) — dropping the P1 read would take it to ~20 and ECSM′ to
  ~2,289 cells. Since ECSM′ is one row per ecrecover, a ±10% error here moves the
  per-ecrecover total by under 0.1%.

- **Whether `PHASE`/`NEXT_PHASE` can be folded into existing columns.** I added
  two byte columns for clarity. `OP` + the selectors may already carry enough
  information to encode the three phases without new columns, but I did not work
  the encoding through and would rather propose four extra columns than a clever
  encoding that turns out to be forgeable — per the standing
  clean-constraints-over-cleverness rule.

- **`bitwise.rs`'s AreBytes pairing contract.** I took `ecdas.rs:174-176`'s
  assertion (that `ARE_BYTES[X, Y]` range-checks *both* elements) and the landed
  `42ba68ff` at their word rather than re-deriving it; it is argued in
  `gate/pairing-equivalence.md`. Every paired-count figure in §5 inherits that
  assumption.
