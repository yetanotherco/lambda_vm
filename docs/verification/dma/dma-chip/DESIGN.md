# DMA memcpy chip — constraint-system & bus design

> **Provenance, read this first.** Unlike the BLAKE3 campaign, where `DESIGN.md`
> was written *before* the Rust and the gate proved the design, this document is
> written *after* `prover/src/tables/dma.rs` (PR #874). It is the specification
> **recovered from the implementation**, and it is what `z3_dma_verify.py`
> checks. That ordering has one consequence worth stating plainly: a design
> document derived from the code cannot find a disagreement between them by
> itself. `../audit_gate_transcription.py` is what keeps this file and the Rust
> from drifting, and the oracle (`../dma-oracle/`) is what supplies an
> independent notion of "correct" that neither the code nor this file defines.

## 1. What the chip proves

One `memcpy(dst, src, n)` ecall, `n ≤ 256`, moved off the CPU trace. The guest's
strong `memcpy` symbol chunks arbitrary lengths into ecalls of at most 256
bytes; the executor performs each chunk natively; this table proves it.

The claim, stated as the oracle states it (`../dma-oracle/dma_ref.py`):

> For an ecall at timestamp `T` whose registers hold `(x10, x11, x12) =
> (dst, src, n)`, the trace contains memory operations that read every byte of
> `[src, src+n)` at `T+1` and write the same bytes to `[dst, dst+n)` at `T+2`,
> each byte exactly once, in the greedy chunking `[8]*(n/8) + [1]*(n%8)`, and
> nothing else.

Three separable obligations fall out, and the gate answers them separately:

| obligation | mechanism | gate check |
|---|---|---|
| a copied byte cannot change | the read tuple and the write tuple reference the **same** `value` columns | not a proof obligation at all — structural. Audited textually (`../audit_gate_transcription.py` §D) |
| the copy covers exactly `[src, src+n)`, once | row chaining through `DmaNext` + `Zero` end detection + the `LT` width pin | MAIN 0–3, CHAIN, CHAIN-F |
| one ecall cannot add unbounded rows | the first row proves `count < 257` on the `Alu` bus | MAIN 2c |

## 2. Row layout decision

A row copies **eight bytes while `count ≥ 8`, otherwise one byte**. The design
is cloned from `commit.rs`: recursive/streaming, one row per chunk, rows chained
by a bus rather than by a transition constraint.

Why not one row per byte: 256 rows per maximal ecall instead of 33.
Why not one row per whole copy: the row would need `n` value columns for an
unbounded `n`, and 8-byte-wide memory operations are the widest the `Memw`
table serves.

Why the 1-byte tail rather than 4/2/1 halving: a `tail` **bit** selects between
exactly two widths, so `step = 8 − 7·tail` stays linear and every constraint
stays degree 2. Halving would need a two-bit width selector and a
width-to-`w2/w4/w8` decode. The cost is at most 7 extra rows per ecall
(`n % 8` ones), against 33 rows for a maximal copy — the tail is ≤ 17% of rows
in the worst case and 0% for the 8-aligned lengths that dominate.

## 3. Column layout (32 columns)

| columns | name | packing | range-checked by |
|---|---|---|---|
| 0–1 | `timestamp` | DWordWL | the `Ecall` receiver (the CPU's own timestamp) |
| 2–3 | `src` | DWordWL | `Memw` base-address limbs (**MEMW-ADDR32**) on data rows; the register read (**REG-32**) on the head |
| 4–7 | `src_incr` | DWordHL | `IsHalfword` ×4, multiplicity `mu` |
| 8–9 | `dst` | DWordWL | as `src` |
| 10–13 | `dst_incr` | DWordHL | `IsHalfword` ×4 |
| 14–15 | `count` | DWordWL | **REG-32 on the head row only** — see §7 R1 |
| 16–19 | `count_decr` | DWordHL | `IsHalfword` ×4 |
| 20 | `first` | Bit | constraint 0 |
| 21 | `end` | Bit | constraint 1 |
| 22 | `tail` | Bit | constraint 2 |
| 23–30 | `value[8]` | Byte-ish | **nothing** — they ride only the `Memw` tuples; lanes 1–7 are forced to zero on tail rows by constraints 11–17 |
| 31 | `mu` | Bit | constraint 3 |

`src_incr`/`dst_incr`/`count_decr` are `DWordHL` (four 16-bit halfwords) and not
`DWordWL` **because they need a range check**, and `IsHalfword` is the cheapest
one available. Their halves are what makes `emit_add_pair_no_overflow`'s
`carry_1 = 0` mean "no wrap" instead of "the high word happens to be `2^32`" —
proved necessary by the width audit (§8.1).

`value[0..8]` carry no range check of their own. The argument is `keccak.rs`'s
for its address bytes: a column that is only ever *consumed by a bus tuple* is
pinned by the receiving table, and the `Memw` table decomposes and checks the
value bytes it receives. What is **not** free is the tail case: on a one-byte
row the memory tuple must be the canonical `w8 = 0` encoding, so lanes 1–7 must
be zero, hence constraints 11–17 (gate MAIN 2b, and its control shows nothing
else pins them).

## 4. Constraints (18, all degree 2)

`DmaConstraints` does not override `max_degree`, so it declares the default 2.
Every constraint below is degree 2, and the gate's encoding notes record why:
each carry is a *linear* expression in the columns (`step = 8 − 7·tail` is
linear, and `carry_0` feeds `carry_1` linearly), so booleanity on a carry and a
`boolean × column` product are both quadratic and nothing is cubic.

| idx | constraint | template |
|---|---|---|
| 0–3 | `first`, `end`, `tail`, `mu` are bits | `emit_is_bit` |
| 4 | `(first + end)·(1 − mu) = 0` | inline |
| 5–6 | `src + step = src_incr`, no `2^64` wrap on active non-terminal rows | `emit_add_pair_no_overflow(MU, END)` |
| 7–8 | `dst + step = dst_incr`, same | `emit_add_pair_no_overflow(MU, END)` |
| 9–10 | `count_decr + step = count` (**wrap permitted**) | `emit_add_pair` |
| 11–17 | `tail · value[i] = 0`, `i = 1..7` | inline |

Two asymmetries are load-bearing and must not be "tidied":

**Constraint 9–10 is the plain pair on purpose.** The terminal row holds
`count = 0` and `count_decr = 0 − 1 = 0xFFFF_FFFF_FFFF_FFFF`; a no-overflow form
would reject it. The gate's MAIN 2 is precisely the statement that this
permission is safe: **the count subtraction wraps only on the terminal row**,
because `tail` is pinned to `count < 8` so `step ≤ count` on every row with
`count ≥ 1`. Take the pin away and the wrap becomes reachable on a data row,
which is the seven-byte truncation in §8.1.

**Constraints 5–8 are the no-overflow form on purpose.** Without it a chain
could walk `src` past `2^64` and continue at low addresses, which the executor
rejects (`checked_add`) and the AIR would not — an executor/AIR divergence, and
a copy that touches unrelated memory. Gate control `drop_no_overflow_src`.

`emit_add_pair_no_overflow`'s gate is `mu − end`: terminal and padding rows
leave `carry_1` free, because their computed successor is consumed by nobody
(the `DmaNext` send has the same multiplicity).

## 5. Bus interactions (23)

| # | bus | dir | multiplicity | tuple |
|---|---|---|---|---|
| 1 | `Ecall` | recv | `first` | `[ts, DMA_LO32, DMA_HI32]` |
| 2 | `DmaNext` | send | `mu − end` | `[ts, src_incr, dst_incr, count_decr]` |
| 3 | `DmaNext` | recv | `mu − first` | `[ts, src, dst, count]` |
| 4–15 | `IsHalfword` | send | `mu` | each halfword of `count_decr`, `src_incr`, `dst_incr` |
| 16 | `Zero` | send | `mu` | `[4·65535 − Σ count_decr, end]` |
| 17–19 | `Memw` | send | `first` | register reads of x10, x11, x12 |
| 20 | `Alu` | send | `mu` | `[count, 8, LT] → tail` |
| 21 | `Alu` | send | `first` | `[count, 257, LT] → 1` |
| 22 | `Memw` | send | `mu − end` | read `src` at `T+1`, `w8 = 1 − tail`, `old == value` |
| 23 | `Memw` | send | `mu − end` | write `dst` at `T+2`, same `value` columns |

### 5.1 `DmaNext` carries the timestamp in **both** tuples

Non-negotiable, and for the reason the BLAKE3 design review found the hard way:
without a per-call binding in both halves of an internal bus, rows belonging to
two different calls can be spliced into each other's chains and the multiset
still balances. Here the timestamp is that binding, and CPU timestamps are
strictly increasing per instruction, so two DMA ecalls never share one.

### 5.2 End detection: one `Zero` lookup, four halfwords

`end = 1` iff `4·65535 − (cd₀+cd₁+cd₂+cd₃) = 0`, i.e. iff all four halfwords are
`0xFFFF`, i.e. iff `count_decr = 2^64 − 1`, i.e. (with the width pin) iff
`count = 0`. One lookup instead of four.

Two premises hold it up, and the width audit separates them:
* the `IsHalfword` bounds — a sum can only identify the all-`0xFFFF` word while
  each summand is in range. Drop them and `(0xFFFF+d, 0xFFFF−d, 0xFFFF, 0xFFFF)`
  reaches the same sum with a totally different `count_decr`, so `end` is
  claimable at a nonzero count. Since an `end` row's two `Memw` sends have
  multiplicity `mu − end = 0`, **it emits no memory operations at all** — a
  silently truncated copy with every bus balanced.
* the receiving table's **domain**. `bitwise.rs` serves `Zero[v]` only for
  `v = x + 256y + 65536z` with `x,y` bytes and `z < 16`, i.e. `v < 2^20`. The
  send's argument lies in `[0, 262140]` under the halfword bounds, comfortably
  inside; a send outside would have no partner row at all. The gate asserts
  `4·65535 < 2^20` at import.

### 5.3 The width pin

`Alu[count, 8, LT] → tail` at multiplicity `mu` is the only thing that stops the
prover choosing a convenient partition. §8.1 shows what a free `tail` buys.

### 5.4 The per-call bound

`Alu[count, 257, LT] → 1` at multiplicity `first`. `dma.rs` takes the constant
*from the executor* (`DMA_MEMCPY_MAX_BYTES as EXECUTOR_DMA_MEMCPY_MAX_BYTES`)
rather than restating it, so the bound the AIR proves cannot drift from the
bound execution enforces. That is the un-driftable form and the audit checks it
stays that way.

### 5.5 Value binding is structural, not proved

The read tuple and the write tuple are built from the same `value_columns()`.
Nothing needs to prove `read == write`, because there is one set of columns.
This is the strongest kind of argument available and also the kind a solver
cannot see, so it is audited textually (`§D` of the audit script), together with
the facts that the read carries `old == value`, that the offsets are `+1`/`+2`,
and that `w8 = 1 − tail` on both.

## 6. Overlap and the two timestamps

All reads at `T+1`, all writes at `T+2`, both as AIR constants. That is what
gives an overlapping copy snapshot (`memmove`) semantics: the memory-consistency
argument orders per-address accesses by timestamp, so every read sees
pre-ecall memory. The executor matches by copying through a fixed scratch buffer.

**Caveat the oracle records as O2:** the snapshot is per ecall, not per
`memcpy`. Chunk *k+1* reads memory chunk *k* already wrote, so a guest-level
`memcpy` of more than 256 bytes is a forward copy, not a `memmove`. That is
in-contract for `memcpy`, but it means "the DMA ecall has memmove semantics"
must not be repeated at the C level.

## 7. Soundness-critical spots a change must not touch

1. **`DmaNext` timestamp in both tuples** (§5.1). Removing it splices calls.
2. **`IsHalfword` on all twelve halfwords** (§5.2, §8.1). Each one is either an
   end-detection forgery or a wrapped address.
3. **`emit_add_pair` (plain) on `count`, `emit_add_pair_no_overflow` on
   `src`/`dst`** (§4). Swapping either direction breaks the terminal row or
   admits an address wrap.
4. **`mu − end` on both data `Memw` sends.** This is what makes a wrongly
   claimed `end` a *silent* truncation rather than an unbalanced bus, and it is
   why end detection carries the weight it does.
5. **The `Alu` width pin** (§5.3). Gone, the prover partitions at will, and
   `count = 7, tail = 0` truncates seven bytes.
6. **`tail · value[i] = 0`** (§3). Gone, a one-byte row's memory tuple carries
   seven unconstrained field elements into the `Memw` bus.
7. **The bound constant taken from the executor** (§5.4). Restating it invites
   the AIR bound and the execution bound to drift.
8. **One `first` per timestamp**, supplied by the `Ecall` receiver against the
   CPU's single send. Two heads at one timestamp would unbalance `Ecall`.
9. **The single-`end` obligation is implicit**, not a constraint: a chain with
   no terminal row has one more `DmaNext` send than receive, so the bus does not
   balance. It is worth knowing this is where termination comes from.
10. **R1 — `count`'s limb split is unconstrained on non-`first` rows.** The one
    live gap, below.

### R1 (residual): `count`'s limb split on non-head rows

`DmaNext` equates **packed** values: the receiver's `count0 + 2^32·count1` is
matched against the sender's `cd₀ + 2^16·cd₁ + 2^32·cd₂ + 2^48·cd₃` as **one
field element**. It does not say the successor split its limbs the same way.
`count`'s limbs are bounded only on `first` rows (the register read, REG-32);
`lt.rs` range-checks `lhs[1]` and `lhs[2]` — hence `count1` — but not the bare
`LHS_0` word, hence not `count0`.

So the honest statement, and what the gate proves (MAIN 3), is a disjunction:

> `successor.count == predecessor.count − width`  **or**  `successor.count > 256`

The second branch is real and reachable, and the gate exhibits it (MAIN 3b,
expected SAT): with `count1 = 2^32 − 1` and `count0 = V + 1` the packed value is
still `V` mod `p`, so the row passes `DmaNext`, while `lt.rs` sees an integer
near `2^64` and returns `tail = 0`. That row claims an **eight-byte width where
one byte remained** — up to seven bytes written past the destination end.

It is not exploitable as it stands: such a row's own `count_decr` is then near
`2^64`, and a chain must descend to `count = 0` to terminate, so it would need
~2^61 rows. But **the obstacle is trace length, not a constraint**, which is a
fragile place for soundness to live — the same shape as any "too expensive to
reach" argument that a later parameter change quietly invalidates.

Two cheap fixes, either sufficient:
* send `IsHalfword` (or an `IsWord`-equivalent) on `COUNT_0`/`COUNT_1` at
  multiplicity `mu`, so every row's count limbs are bounded, not just the head's;
* or receive `count` as `DWordHL` like `count_decr`, reusing the existing
  halfword sends.

Either makes `count ≤ 256` an invariant of every row and collapses MAIN 3's
disjunction to its first branch.

## 8. Gate

`z3_dma_verify.py`. Two layers (field-exact single/paired rows; integer and
field-exact multi-row chains with `DmaNext` as a free bijection rather than an
assumed chain), eight negative controls, a field-level width audit, and an
oracle-pinned completeness sweep over every length `0..256`.

What it cannot see, stated in its own docstring: bus **wiring**, the memory
consistency argument (hence overlap ordering), LogUp soundness, and trace
length. The first is covered by `../audit_gate_transcription.py`, the second by
the oracle's `write_before_read` mutant, the fourth by R1 above being reported
rather than assumed away.

### 8.1 The width audit — three bound-necessity results

Each is run twice, with the bound and without, so "the bound is necessary" is a
measured claim rather than an assertion. These are the concrete forgeries §3, §4
and §5 refer to:

| result | with the bound | without |
|---|---|---|
| `Σ count_decr = 4·65535 ⟺ count_decr = 2^64−1` | unsat (the identity holds) | **sat** — `(0xFFFF+d, 0xFFFF−d, 0xFFFF, 0xFFFF)` reaches the same sum with a different `count_decr`, so `end` is claimable at a nonzero count, and an `end` row emits no memory operations. A silently truncated copy. |
| `carry_1 = 0 ⟹ src + width < 2^64` | unsat (pinned) | **sat** — at `src1 = 2^32−1` the high half can be exactly `2^32`, which the `IsHalfword` pair forbids and an unbounded pair does not. The row hands on a *wrapped* address the executor's `checked_add` rejects. |
| the `LT` width pin blocks `count = 7, end = 1` | unsat | **sat** — a free `tail` takes `tail = 0`, so `step = 8`, so `count_decr = 7 − 8 = 0xFFFF…`, so `end = 1`. **Seven requested bytes silently not copied.** |

The last one is worth reading twice: `end` requires `count = step − 1`, so a free
`tail` buys exactly `count = 7` and no other value. The two constraints compose to
leave precisely one hole, which is also why an earlier draft of this audit wrote
the forgery at `count = 3` and was wrong — it is not reachable there.

## 9. Gate results

Run 2026-08-11, z3 5.0.0, full board (no `--quick`):

```
LAYER 1
  MAIN 0  row == oracle row                 -> unsat
  MAIN 1  end <=> count == 0                -> unsat
  MAIN 2  count wraps only on terminal row  -> unsat
  MAIN 2b one-byte row has zero lanes 1..7  -> unsat
  MAIN 2c one ecall asks for <= 256 bytes   -> unsat
  MAIN 3  successor honest OR count > MAX   -> unsat
  MAIN 3b R1 alias is reachable             -> sat     (the residual, exhibited)
LAYER 2
  CHAIN   2/3/4/5 rows, any balanced structure -> unsat
  CHAIN-F 2/3 rows, field-exact                -> unsat
NEGATIVE CONTROLS (all sat, 8/8)
  drop_halfword_count_decr, drop_halfword_src_incr, drop_zero_end,
  drop_lt_tail, drop_no_overflow_src, drop_tail_lane_zero,
  drop_lt_bound, drop_reg32
WIDTH AUDIT
  Zero sum identity       bounds present -> unsat   DROPPED -> sat
  no-overflow             bounds present -> unsat   DROPPED -> sat
  truncation at count=7   LT pin present -> unsat   DROPPED -> sat
COMPLETENESS
  5410 honest rows over 257 lengths, all accepted
OVERALL: PASS
```

### What is and isn't proven

**Proven.** Given the modelled contracts (`IsHalfword`, `Zero` including its
domain, `Alu[LT]` as `lt.rs`'s own constraints, MEMW-ADDR32, REG-32) and given
that bus balance means multiset equality: every satisfying assignment of one
DMA row does what the oracle says; the only bus-balanced multi-row structure at
depth ≤ 5 is a single chain tiling `[src, src+n)` exactly once with the greedy
widths; every one of the eight range checks and lookups involved is individually
necessary; and the AIR accepts every honest trace for every length `0..256`.

**Not proven.** R1 (§7). The memory consistency argument and therefore overlap
ordering. LogUp soundness. That the *Rust* implements this design — that is
`../audit_gate_transcription.py`'s 83 textual claims plus the PR's own
end-to-end prove/verify and forgery tests.
