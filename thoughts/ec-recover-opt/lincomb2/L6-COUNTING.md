# L6 — the joint-schedule counting argument

**Verdict: L6 does NOT hold for `ecdas2.rs` as written. There is a constructive
break, and it is worse than the NUMS finding — it yields an ARBITRARY CHOSEN
recovered public key, i.e. an arbitrary chosen transaction sender, with no
discrete log and no search.** The fix is two degree-2 constraints and no new
columns.

Everything else in L6 holds. §3 writes out the parts that are proved; §2 is the
break; §4 is the fix; §5 records the four mechanisms that were checked rather
than assumed and came back clean.

Reproduce: `<venv>/bin/python thoughts/ec-recover-opt/gate/l6_joint_counting.py`
(log: `gate/logs/l6.log`). Model transcribed by reading `prover/src/tables/ecdas2.rs`
and `ecsm2.rs`; convolution carries are out of scope here (that is
`WIDTH-AUDIT.md`) — this is about the *schedule*.

> **Column indices in this document are today's and will move.** `D_INV` is
> still owed and adds ~100 columns. Nothing in the argument depends on an index:
> it depends on *which constraints and which multiplicities exist*.

---

## 1. What L6 has to establish

For one ecall timestamp `ts` with `OK = 1`, the bus balance plus the per-row
constraints must force the joint chain to be exactly the honest schedule:

- exactly one doubling per round, `round = len−1 … 0`;
- an add at round `r` iff the joint digit `(u1_r, u2_r) ≠ (0,0)`, exactly once,
  consuming exactly the addend those digits select;
- every set bit of `u1` and `u2` consumed exactly once at its own round;
- no inserted, dropped or reordered rows;
- the three phases (precompute / main / correction) each occurring exactly once
  with the right endpoints.

The exact-MSB sub-lemma (old L6.5) drops: blinding makes any
`len ∈ [max_msb+1, 256]` yield the same `Q`.

## 2. THE BREAK — padding rows are live digit senders

### 2.1 The mechanism

ECDAS2 sends the per-stream digits as

```rust
// ecdas2.rs:459-470
for (stream, col) in [(1u64, cols::D1), (2u64, cols::D2)] {
    out.push(BusInteraction::sender(
        BusId::JointBit,
        Multiplicity::Column(col),          // <-- NOT MU-gated
        vec![ts_lo(), ts_hi(), packed(cols::ROUND), BusValue::constant(stream)],
    ));
}
```

and **no constraint ties `D1`/`D2` to `MU`.** Reading every occurrence of `MU`
in `Ecdas2Constraints::eval`, it appears exactly three times: idx 0 (`IS_BIT`),
idx 20 (`MU·(1−PH1−PH2)·(S2−1)`), and inside `rq()`. There is no
`D1·(1−MU) = 0`.

**The old chip has precisely this defence, for precisely this reason.**
`ecdas.rs` idx 4 is `NEXT_OP·(1−MU) = 0`, gating the multiplicity of its own
`Bit`-bus sender. ECDAS2 dropped it while adopting the same pattern.

A padding row is inert on every other bus — its `Ecdas` receive and send,
`AreBytes` and `IsHalfword` sends are all `mu()`-gated, and its `Addend` receive
is gated by `ΣS`. But its digit send is **live**.

### 2.2 The padding row is satisfiable

With `MU = 0, PH1 = 1, D1 = 1, ROUND = r` and everything else zero except
`NB = 1` (forced by idx 13), all 217 constraints hold:

| constraint | check |
|---|---|
| idx 0-10 `IS_BIT` | all columns are 0/1 |
| idx 11 `PH1·PH2` | `PH2 = 0` |
| idx 12 `OP·NB` | `OP = 0` |
| idx 13 `(1−OP)(NB−D1−D2+D1·D2)` | forces `NB = 1`; harmless, the Ecdas send is µ-dead |
| idx 14 `OP = ΣS` | `0 = 0` |
| idx 15,16 `(1−PH1)·D` | satisfied by `PH1 = 1` |
| idx 17-19 | all zero |
| idx 20 | **µ-gated — `MU = 0` kills it** |
| idx 21 | `PH2 = 0` |
| idx 22-216 conv | all limbs zero, `rq` is µ-gated ⇒ `S_i = 0`, carries 0 |

z3: **SAT**. Note also that `ROUND` is only byte-checked through a µ-gated
`AreBytes` send, so on a padding row it is entirely free.

### 2.3 The schedule-level forgery

At a round `r` where `u1`'s bit is set and `u2`'s is not, the honest chain has a
double with `(D1,D2) = (1,0)` — hence `NB = 1` — and an add consuming `P1`.

The prover instead sets the double's `D1 = D2 = 0`, so `NB = 0` and **no add
follows**, and supplies the round's `2·u1_bit(r) = 2` JointBit count from **two
phantom padding rows** carrying `D1 = 1` at `ROUND = r`. Balance is satisfied,
the chain never adds `P1` at round `r`. z3: **SAT**.

The chain then computes `Q' = (u1 − a)·G + (u2 − b)·R`, where `a`, `b` are the
dropped bit-sets, while the proof claims `u1·G + u2·R`.

### 2.4 Why it is critical: an arbitrary chosen sender

Unlike the NUMS finding — which gave only a one-parameter family and needed
~2^160 grinding to aim — here the attacker picks the *effective* multipliers
first, so no discrete log is required:

1. name the target public key `T` (a key the attacker does not hold);
2. choose `u1' = u2' = 1`, so **`R = T − G`** — a plain point subtraction;
3. set `u1 = 1 + 2^m` (bit `m` unset in `u1'` and in `u2`), `u2 = 1`, and drop
   bit `m` by the §2.3 construction;
4. back-solve the signature from `r = x(R)`, `v = parity(y(R))`:
   `z = −u1·r mod N`, `s = u2·r mod N`.

The chain computes `1·G + 1·(T − G) = T`. Verified numerically end to end:

```
   u1 = 257 (bit 8 will be dropped), u2 = 1, len = 9
   Q' == target     : True
   Q' != honest Q   : True
   guest recomputes u1: True   u2: True   lifts R: True
```

Cost: a handful of extra padding rows. Any `(u1', u2')` works; `u1' = u2' = 1`
is just the shortest to write.

## 3. What L6 does establish (given the fix)

Fix §4 first; then the following goes through. `ts` is fixed, `OK = 1`.

**(a) Phase separation.** `PHASE = PH1 + 2·PH2` with `PH1·PH2 = 0` (idx 11) so
`PHASE ∈ {0,1,2}`. Every ECDAS2 row carries `phase` unchanged from its receive to
its send (same columns in both tuples), so phase is a path invariant, and the
ECSM2 seeds/drains pin it to the constants 0/1/2.

**(b) Exactly one segment per phase.** ECSM2's six chain interactions all carry
multiplicity `OK`, which is `IS_BIT` (idx 1) — so one seed and one drain per
phase. Two ECSM2 rows cannot share a `ts` (contract C6: the `Ecall` receive
would double against the CPU's single send; C7 gives distinct ts per ecall).

**(c) No cycles, and paths end only at drains.** Along a row,
`round' = round − 1 + NB` and `op' = NB`, with `OP·NB = 0` (idx 12):

- `OP = 1` (add) ⇒ `NB = 0` ⇒ `Δround = −1`;
- `OP = 0` (double) ⇒ `Δround ∈ {0, −1}`, and `Δround = 0` forces `op' = 1`,
  i.e. the successor is an add, which then decrements.

So `round` never increases and any cycle would need every step at `Δ = 0` —
impossible, since a `Δ = 0` step is always followed by a strict decrease. The
drain tuple has `round = −1`, which no row can receive because `ROUND` is
byte-checked. Balance therefore decomposes the rows of each phase into exactly
one seed→drain path.

**(d) Phases 0 and 2 are exactly one row each.** Both seeds carry `round = 0,
op = 1`. An add has `NB = 0`, so the single row sends `round = −1, op = 0` —
exactly the drain. No second row is possible. Their selectors are pinned by idx
20 (`S2 = 1`, the precompute adds `P2`) and idx 21 (`S_CORR = 1`, the correction
adds `−2^len·T₀`), with idx 15/16 forcing `D1 = D2 = 0` on both.

**(e) The main chain is one doubling per round, plus an add iff a digit is set.**
The phase-1 seed is `(T₀, round = LEN_M1, op = 0)`. By (c) the path is
`double(L−1) [add(L−1)] double(L−2) [add(L−2)] … double(0) [add(0)]` and then
drains. Idx 13 makes `NB = D1 ∨ D2` on a doubling, so the add at round `r`
exists iff that round's joint digit is non-zero.

**(f) Digit consumption is exact.** For each `(i, stream)`, balance gives
`#{rows with ROUND = i and D = 1} = 2·u_bit(i)`. Each row's multiplicity is a
single `IS_BIT` column, so each contributes 0 or 1 and the only decomposition of
2 is `1 + 1`. With the fix, only live rows can contribute, and by (e) at most two
live rows share a round — the double and its add. Hence:

- `u_bit(i) = 1` ⇒ **both** rows exist and **both** carry `D = 1`;
- `u_bit(i) = 0` ⇒ neither does.

A set bit above `LEN_M1` has a 2× receive with no possible sender ⇒ imbalance ⇒
`len ≥ max_msb + 1`. The upper bound `len ≤ 256` is structural: the `EC_T0`
table has exactly 256 rows and no padding, so a lookup outside `[1, 256]` matches
nothing. **This is where L6.5 drops** — any larger `len` is fine, because the
extra leading doublings only double `T₀` and the keyed correction absorbs them.

**(g) The addend matches the digits.** On a live main-chain add, idx 17 gives
`S_CORR = 0`, idx 14 gives `S1+S2+S3 = 1`, and idx 18/19 give `S1+S3 = D1`,
`S2+S3 = D2`. Solving: `(1,0) ⇒ S1`, `(0,1) ⇒ S2`, `(1,1) ⇒ S3`, and `(0,0)` is
**unsatisfiable** — no spurious add. On a live doubling `OP = 0` forces every
selector to zero, so the `Addend` receive is silent. Each case machine-checked
(z3 UNSAT on the negation).

Combining (a)-(g): the chain is exactly the honest schedule, and by the Addend
balance each consumed addend is the one ECSM2 published — `G` (a constant), `P2`
(MEMW-bound), `P12` (the phase-0 drain) or the `EC_T0` constant.

## 4. The fix

```
(1 − MU)·D1 = 0
(1 − MU)·D2 = 0
```

Two degree-2 constraints, no new columns, the exact shape of `ecdas.rs` idx 4.
With them, both z3 queries of §2 go **UNSAT**.

Recommended in addition: gate the selectors too. A `MU = 0, OP = 1, S1 = 1` row
still mints a spurious `Addend` **receive**. That is harmless on its own — it
injects nothing into the chain and merely forces the publisher to inflate `N1` —
but it is the same class of hole and `(1−MU)·S* = 0` closes it for four more
degree-2 constraints. Alternatively make the multiplicities `MU·D1` / `MU·ΣS` if
the `Multiplicity` type admits a product; that is the structurally right fix and
removes the need for the constraints entirely.

**Phase E needs a negative control for this**: remove the gate, feed the §2.3
schedule, and the gate must report SAT.

## 5. Mechanisms checked, and clean

### 5.1 The 2× JointBit multiplicity is genuinely stronger than 1×

Confirmed, with a concrete counterexample at 1×. At a round where **both**
digits are set, a 1× prover splits them across the two rows — double takes
`D1 = 1, D2 = 0`, add takes `D1 = 0, D2 = 1`. Both counts balance at 1. Idx
18/19 on the add then give `S1+S3 = 0`, `S2+S3 = 1` ⇒ **`S2`**: the chain adds
`P2` where the schedule calls for `P12`. Wrong `Q`, fully satisfying. z3: SAT at
1×, UNSAT at 2×.

The `2 = 1+1`-only argument is correct because each row's multiplicity is a
single column constrained by `IS_BIT`, so no single row can contribute 2.

The companions `(1−PH1)·D1 = 0` and `(1−PH1)·D2 = 0` are load-bearing exactly as
their comment says: precompute and correction are both emitted at `round = 0`,
so without them a prover sets `D1 = 1` on both and satisfies `2·u1_bit(0)` with
no round-0 add at all. (They are *not* sufficient against §2, because a phantom
row can simply set `PH1 = 1`.)

### 5.2 The bus-28 separator holds — but not for the stated reason

`JOINT_CHAIN_ID = 1` in tuple position 0 where the old chain pins `0`. The claim
is sound. The stated justification is not the operative mechanism:

- **There is no re-alignment risk to begin with.** In `compute_fingerprints`
  (`lookup.rs:1651-1663`) `alpha_offset` advances by `num_bus_elements()` for
  every `BusValue` unconditionally. The `if result != zero` skip at `:679`
  avoids a multiply and still returns `1`, so a zero element consumes its α slot
  and contributes nothing. **Interior zeros do not shift positions.**
- **What actually separates the chains** is that the fingerprints are
  `bus_id + Σ_k v_k·α^k` and the two tuples differ in the α¹ coefficient
  (constant `0` vs constant `1`). The difference polynomial is therefore
  non-zero, and Schwartz–Zippel over the random α closes it. Lengths (133 vs 70
  elements) are irrelevant.
- **Trailing-zero aliasing is real in general** — a shorter tuple does alias a
  longer one whose extra trailing elements are all zero — which is what the
  `sel ≠ 0` note on the `Addend` bus correctly guards. It is simply not the
  mechanism at work on bus 28.

**No need for bus 33.** Worth rewording the comment on `JOINT_CHAIN_ID` so the
next reader does not inherit the wrong model of the fingerprint.

### 5.3 The phase relay

Verified. Phase-1's drain is received into `ACC_X`/`ACC_Y` and phase-2's seed is
sent from **the same columns**, so the hand-off is a literal relay; both at
multiplicity `OK`. Routing it through ECSM2 rather than along the chain is
forced, as the header says: the outgoing tuple pins the successor's `op` to
`NB`, which is 0 on the last main row, while the correction row is an add.

### 5.4 `OP = ΣS`

Verified load-bearing. On a live doubling it forces every selector to zero, so
the `Addend` receive multiplicity is 0. Without it the double-row addend
cancellation is still algebraically real, but the *gating* is forgeable — a
prover sets `S2 = 1` on a doubling and mints a spurious receive.

## 6. Findings

1. **[CRITICAL — fix before anything else] Padding rows are live digit senders**
   (§2). Arbitrary chosen recovered public key. Fix: `(1−MU)·D1 = 0`,
   `(1−MU)·D2 = 0`.
2. **[recommended] Gate the selectors too** — `(1−MU)·S* = 0`, or make the bus
   multiplicities products with `MU` (§4).
3. **[documentation] Reword the `JOINT_CHAIN_ID` comment** (§5.2). The separator
   is sound; the "trailing-zero collapse / re-alignment" reasoning is not what
   makes it sound, and it misdescribes how the fingerprint is built.
4. **[no action] 2× multiplicity confirmed**, with the 1× counterexample
   recorded as a phase-E negative control (§5.1).
5. **[phase E] Negative controls to add**: drop the §4 gate (must go SAT); use
   1× JointBit multiplicity (must go SAT); drop `(1−PH1)·D` (must go SAT); drop
   `OP = ΣS` (must go SAT).
