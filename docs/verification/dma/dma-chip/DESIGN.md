# DMA memcpy chip — constraint-system & bus design

> **Provenance, read this first.** This document is written *after*
> `prover/src/tables/dma.rs` (PR #874) — unlike a design-first campaign, where the
> spec precedes the code and the gate proves the design. It is the specification
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

Why not one row per byte: 257 rows per maximal ecall (256 data + 1 terminal)
instead of 33.
Why not one row per whole copy: the row would need `n` value columns for an
unbounded `n`, and 8-byte-wide memory operations are the widest the `Memw`
table serves.

Why the 1-byte tail rather than 4/2/1 halving: a `tail` **bit** selects between
exactly two widths, so `step = 8 − 7·tail` stays linear and every constraint
stays degree 2. Halving would need a two-bit width selector and a
width-to-`w2/w4/w8` decode.

The cost, stated precisely (an earlier draft had this wrong in both directions):
a copy of `n` bytes takes `n/8` wide rows, `n % 8` tail rows and one terminal row.
So tail rows are `0%` for the 8-aligned lengths that dominate, `7/39 = 17.9%` at
`n = 255`, the longest length that *has* a tail (the maximal `n` is 256, which has none), and **`7/8 = 87.5%` at the genuine worst case `n = 7`**,
where every data row is a tail row. Against 4/2/1 halving the delta is at most
4 rows (halving needs `popcount(n % 8) ≤ 3`), not the 7 an earlier draft claimed
by comparing against a zero-tail design instead of against the alternative it was
arguing with.

## 3. Column layout (32 columns)

| columns | name | packing | range provenance |
|---|---|---|---|
| 0–1 | `timestamp` | DWordWL | the `Ecall` receiver (the CPU's own timestamp) |
| 2–3 | `src` | DWordWL | head row: **assumption A1** (see §Assumptions). Non-head rows: *derived* — the `DmaNext` link binds each 32-bit limb against the predecessor's `IsHalfword`-checked halfwords (§5.1) |
| 4–7 | `src_incr` | DWordHL | `IsHalfword` ×4, multiplicity `mu` |
| 8–9 | `dst` | DWordWL | as `src` |
| 10–13 | `dst_incr` | DWordHL | `IsHalfword` ×4 |
| 14–15 | `count` | DWordWL | head row: **assumption A2**. Non-head rows: *derived*, same mechanism as `src` |
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

`value[0..8]` carry no range check of their own, and the reason is **not** that
the receiving table checks them. `spec/memw.typ:42-45` is explicit, and the whole
sentence matters: *"Our assumptions do not explicitly cover any range checks for
the `is_register` and `value` columns, as these are not necessary for the
correctness of this chip in isolation. Still, these properties are necessary for
the consistency of the system as a whole."* So the spec does not merely decline to
check them — it says somebody must. That obligation is recorded as **A5** below. (An earlier draft cited `keccak.rs` here as authority for relying on
the receiver. That was backwards: `keccak.rs:355-378` emits four `AreBytes`
senders for its address bytes **precisely because** the receiver does not pin
them, and its comment spells out the forgery — keeping a linear combination's
field value correct while encoding non-byte values in the individual cells.)

The actual argument is narrower and specific to this chip: the `T+1` read tuple
carries `old == value` **against real memory** (§5, bus 22), so each lane is
pinned to the byte the memory argument says is at that address. Lanes are not
free field elements; they are whatever memory already held.

What that does **not** cover is the tail case: on a one-byte row the memory tuple
must be the canonical `w8 = 0` encoding, so lanes 1–7 must be zero, hence
constraints 11–17.

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
10. **The `DmaNext` tuples' element alignment** (§5.1). Both tuples are 8 bus
    elements and align pairwise; that is what makes the link a per-limb binding
    and hence what supplies the range provenance in §3 for every non-head row.
    A packing change on either side silently changes what the bus binds.

### The limb binding, and the finding that turned out not to exist

`DmaNext` does **not** compare packed 64-bit values. `Packing::num_bus_elements()`
(`crypto/stark/src/lookup.rs:227-241`) returns **2** for both `DWordWL` ("2× Direct")
and `DWordHL` ("2× Word2L"), and each element gets its own alpha power. No
`Packing` variant contains a `2³²` shift, so a 64-bit value is never one bus
element anywhere in this codebase. Both tuples are `1+1+2+2+2 = 8` elements and
align pairwise, so balance imposes two equations per value:

```
receiver.COUNT_0 == sender.cd₀ + 2¹⁶·cd₁        (low word)
receiver.COUNT_1 == sender.cd₂ + 2¹⁶·cd₃        (high word)
```

Since the sender's four `count_decr` halfwords are `IsHalfword`-checked at
multiplicity `mu`, **the receiver's limbs are 32-bit for free** — no extra range
check needed. Same for `src` and `dst`. This is what §3's "derived" entries mean,
and it is why MAIN 3 is an unconditional claim rather than a disjunction.

**Recorded because the campaign got this wrong first.** An earlier version of the
gate modelled the hop as one equation on the fully packed value, which is
strictly *weaker* than the AIR: it let the receiver re-split its limbs and so
manufactured an alias (`COUNT_1 = 2³²−1`, `COUNT_0 = V+1`, packed ≡ V mod p)
that the real bus rejects. That phantom was published here as "RESIDUAL R1 — the
one live gap", with two recommended fixes. It has been struck. Two things worth
keeping from the episode:

* **The direction of a modelling error decides what it costs.** A model weaker
  than the AIR yields false alarms but never false proofs — every UNSAT the gate
  reported survived the correction, having been proven under weaker hypotheses
  than reality supplies. A model *stronger* than the AIR is the dangerous one.
* **A proposed fix that is a no-op is evidence the gap is not there.** R1's
  second fix was "receive `count` as `DWordHL`", which under the real semantics
  changes nothing. That should have been caught at authoring time.

`../audit_gate_transcription.py` §G now asserts the element counts and the tuple
alignment directly; its absence is what let the phantom through, since the audit
previously checked only that the packing *names* appeared.

## Assumptions

Obligations on the **caller**, not checks this chip performs. Real spec chapters
render this section (`render_chip_assumptions`); its absence from an earlier draft
is why two caller obligations got mislabelled as receiver checks in §3.

| id | assumption | discharged by | status |
|---|---|---|---|
| **A1** | `src`/`dst` limbs are 32-bit words wherever a `Memw` data op fires (`mu − end == 1`) — what the gate asserts. Load-bearing only on the head row: elsewhere the `DmaNext` link derives it, which is why the gate's `memw_addr32` toggle is inert and carries no control | `spec/src/memw.toml`: `[[assumptions]] IS_WORD[base_address[i]]`. `memw.rs:257-262` justifies its own bound via *the CPU table*; DMA is a non-CPU sender, so that argument does not extend here | **not discharged locally.** Non-DMA-specific — every table sending an address depends on it |
| **A2** | the head row's `count` limbs are 32-bit words | `spec/src/memw_register.toml`: `[[assumptions]] IS_WORD[val[i]]`. MEMW_R's only range-check interaction is on the timestamp delta | **not discharged locally.** The gate's `drop_reg32` control shows what it buys: without it the `count < 257` lookup caps only a residue class |
| **A3** | `IS_WORD` on the timestamp, so `ts₀ + 2` does not carry into the high limb | `spec/src/memw.toml` timestamp assumption. In practice the CPU stride is 4 and `T = 4i+4`, so the `+1`/`+2` cannot carry — but nothing in the DMA AIR constrains it | not discharged locally; benign at the current stride |
| **A4** | two DMA ecalls never share a timestamp | CPU timestamps strictly increase per instruction | holds by construction; it is what the `DmaNext` timestamp binding relies on |
| **A5** | domain-0 memory cells hold bytes | nothing in this chip, and nothing in `memw.rs` — `spec/memw.typ:42-45` assigns it to "the consistency of the system as a whole" | **not discharged, and not DMA's to discharge.** This chip *propagates* byte-ness (§3: each lane is pinned to whatever memory held) rather than establishing it. Recorded because an obligation owned by everyone is owned by nobody, and this repo has a history of byte-decomposition range checks going missing |

A1/A2 are the **head row only**. Every other row's limbs are derived (§7 above).
Note the irony worth recording: the phantom R1 reported a gap on non-head rows,
where the bus in fact pins the limbs, while the real obligation sits on the head
row, which has no `DmaNext` receive at all.

`spec/` has a broader problem here, flagged for the spec rather than this chip:
`IS_WORD` appears across **12** chapters (`add`, `branch`, `cpu32`, `eq`, `halt`,
`load`, `lt`, `memw`, `memw_aligned`, `memw_register`, `sha256msgsched`, `store`)
**exclusively** inside `[[assumptions]]`, never as an interaction or template, and
`spec/bitwise.typ` offers no 2³² table — it exposes MSB8/MSB16/ZERO/ARE_BYTES/
IS_BYTE/IS_HALF/IS_B20/HWSL, none of which spans a word.
So the spec asserts a range obligation for nearly every address, register value
and timestamp in the VM without naming a discharger. That vacuum is what an
earlier draft of this document filled by inventing the labels "MEMW-ADDR32" and
"REG-32", and the next chip author will fill it the same way.

## Padding

Real spec chapters render this too (`render_chip_padding_table`), and the DMA
padding row is not all-zero, so it is worth writing down.

`generate_dma_trace` pads to the next power of two, minimum 4, with:

| column | value | why |
|---|---|---|
| `mu` | 0 | kills all 23 bus interactions |
| `first`, `end` | 0 | forced by constraint 4 once `mu = 0` |
| `count` | 1 | constraints 9–10 are **unconditional**, so padding must satisfy them |
| `tail` | 1 | so `step = 1` and `count_decr = count − 1 = 0` |
| `src_incr`, `dst_incr` | 1 | so the low carry is 0 rather than −1 |
| everything else | 0 | |

The gate's completeness sweep pins exactly this row (`dma_ref.padding_columns`),
and `dma_padding_row_cannot_claim_first_or_end` covers the one constraint that
stops a padding row masquerading as a copy's head or terminal row.

## 8. Gate

`z3_dma_verify.py`. Two layers (field-exact single/paired rows; integer and
field-exact multi-row chains with `DmaNext` as a free bijection rather than an
assumed chain), ten negative controls, a field-level width audit, and an
oracle-pinned completeness sweep over every length `0..256`.

What it cannot see, stated in its own docstring: bus **wiring** (covered by
`../audit_gate_transcription.py`, whose §G now includes the packing/element-count
claims whose absence let a phantom finding through), the memory consistency
argument and hence overlap ordering (covered on the model side by the oracle's
`write_before_read` mutant), and LogUp soundness (assumed). Layer 2's scope is
also bounded: it proves the tiling among groups containing exactly **one head
row**, because `ChainRow` does not model the timestamp that separates two calls.

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

Verbatim from `python3 z3_dma_verify.py` (full board, no `--quick`), 2026-08-12,
z3 5.0.0 — **pasted, not retyped**, including the `solver:` line, which an earlier
version silently dropped while the surrounding sentence claimed the block was
verbatim. If you regenerate this, paste the whole thing.

```
============================================================================
DMA memcpy chip -- z3 gate
============================================================================
  solver: z3 5.0.0
  legend: unsat = proved | sat = counterexample found | unknown = TIMED OUT (failure)

=== LAYER 1: field-exact rows ===
  MAIN 0  row == oracle row                 -> unsat   (want unsat)
  MAIN 1  end <=> count == 0                -> unsat   (want unsat)
  MAIN 2  count wraps only on terminal row  -> unsat   (want unsat)
  MAIN 2b one-byte row has zero lanes 1..7  -> unsat   (want unsat)
  MAIN 2c one ecall asks for <= 256 bytes  -> unsat   (want unsat)
  MAIN 3  successor exact + well formed     -> unsat   (want unsat)

=== LAYER 2: chain structure, DmaNext as a free bijection ===
  CHAIN   2 rows, any balanced structure   -> unsat   (want unsat)
  CHAIN   3 rows, any balanced structure   -> unsat   (want unsat)
  CHAIN   4 rows, any balanced structure   -> unsat   (want unsat)
  CHAIN   5 rows, any balanced structure   -> unsat   (want unsat)
  CHAIN-F 2 rows, field-exact              -> unsat   (want unsat)
  CHAIN-F 3 rows, field-exact              -> unsat   (want unsat)

  -- Layer 2 controls --
  positive: 2-row premise set satisfiable  -> sat   (want sat)
  positive: 3-row premise set satisfiable  -> sat   (want sat)
  positive: 4-row premise set satisfiable  -> sat   (want sat)
  positive: 2-row field-exact premise set   -> sat   (want sat)
  negative: drop `count` from the tuple       -> sat   (want sat)
  negative: drop `src` from the tuple       -> sat   (want sat)
  negative: drop `dst` from the tuple       -> sat   (want sat)

=== NEGATIVE CONTROLS -- drop one premise, expect a forgery ===
  drop_halfword_count_decr     -> sat   (want sat)
  drop_halfword_src_incr       -> sat   (want sat)
  drop_zero_end                -> sat   (want sat)
  drop_lt_tail                 -> sat   (want sat)
  drop_no_overflow_src         -> sat   (want sat)
  drop_tail_lane_zero          -> sat   (want sat)
  drop_lt_bound                -> sat   (want sat)
  drop_reg32                   -> sat   (want sat)
  drop_halfword_dst_incr       -> sat   (want sat)
  drop_no_overflow_dst         -> sat   (want sat)

=== WIDTH AUDIT -- bound necessity at the boundary (field level) ===
  Zero sum identity, bounds present        -> unsat  (want unsat)
  Zero sum identity, bounds DROPPED        -> sat    (want sat)
  no-overflow, halfword bounds present     -> unsat  (want unsat)
  no-overflow, halfword bounds DROPPED     -> sat    (want sat)
  truncation at count=7, LT pin present    -> unsat  (want unsat)
  truncation at count=7, LT pin DROPPED    -> sat    (want sat)

=== POSITIVE CONTROLS -- oracle-pinned completeness sweep ===
  PASS  5153 honest rows + 257 padding rows over 257 lengths, all accepted

============================================================================
VERDICT
============================================================================
  layer 1 (row semantics)              : True
  layer 2 (chain structure)            : True
  layer 2 controls (pos + neg)         : True
  negative controls all SAT            : True   (10/10)
  width audit (bound necessity)        : True
  completeness sweep SAT               : True

  Scope: Layer 2 proves the tiling among groups with exactly ONE head
  row. Two DMA calls are separated by the `ts` in both DmaNext tuples,
  which `ChainRow` does not model -- see `check_chain`'s docstring and
  the textual guard in ../audit_gate_transcription.py.

  OVERALL: PASS
```

### What is and isn't proven

**Proven.** Given the modelled contracts (`IsHalfword`, `Zero` including its
domain, `Alu[LT]` as `lt.rs`'s own constraints, the `DmaNext` per-limb binding of
§7) and assumptions A1–A4, and given that bus balance means multiset equality:
every satisfying assignment of one DMA row does what the oracle says; among
groups with exactly one head row, the only bus-balanced multi-row structure at
depth ≤ 5 is a single chain tiling `[src, src+n)` exactly once with the greedy
widths; every one of the **ten** range checks and lookups involved is individually
necessary, each with a named forgery; Layer 2's own premise set is satisfiable and
sensitive to each field of the bus tuple; and the AIR accepts every honest trace
for every length `0..256`.

**Not proven.** Assumptions A1–A4 (§Assumptions) — the head row's limb
canonicality and the timestamp bound, all of which are caller obligations the
spec states and no chip discharges locally. The memory consistency argument and
therefore overlap ordering for unaligned 8-byte accesses. LogUp soundness. The
multi-call case (Layer 2 models one head row). And that the *Rust* implements this
design — that is `../audit_gate_transcription.py`'s 100 textual claims plus the
PR's own end-to-end prove/verify and forgery tests.
