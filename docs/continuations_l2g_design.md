# Continuations (Approach 2) — Local-to-Global memory design

This document describes how cross-epoch memory consistency works in the
"continuations" prover (Approach 2, "prove-epoch" from the streaming spec), the
soundness mechanisms that make it safe, and the design decision (Design Y vs the
earlier Design X) for how the per-row selector is wired.

It is written to be read by a human picking this up cold.

---

## 1. Why continuations

A monolithic proof builds the trace for the **whole** execution in memory at
once; for large programs that exhausts RAM. Continuations split the execution
into fixed-size **epochs** and prove each independently, so peak memory stays
flat as program size grows.

Almost every constraint in a proof is local to its slice of cycles — *except
memory*. A load in a late epoch may read what an early epoch wrote. So the only
thing that must be stitched across epoch boundaries is **memory consistency**.

```
   one execution (e.g. 4,000,000 cycles)
   ┌───────────────────────────────────────────────┐
   │  split into epochs of N cycles                 │
   └───────────────────────────────────────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ Epoch 0 │  │ Epoch 1 │  │ Epoch 2 │  │ Epoch 3 │  each proven on its own
   │ CPU MEMW│  │ CPU MEMW│  │ CPU MEMW│  │ CPU MEMW│  (tables dropped from RAM
   │ ... L2G │  │ ... L2G │  │ ... L2G │  │ ... L2G │   after each epoch)
   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
        └────────────┴─────┬──────┴────────────┘
                           ▼
                ┌────────────────────────┐
                │   ONE global proof     │  links the epochs together
                │   (cross-epoch memory) │
                └────────────────────────┘
```

---

## 2. The pieces

A **bus** is a LogUp channel: tables *send* and *receive* tokens, and the proof
checks that everything sent is received (the bus "balances"). An unmatched token
makes the proof fail.

- **MEMW** — the actual loads/stores, driven by the CPU executing the program.
- **L2G** (local-to-global) — one row per memory cell an epoch *touches*. Two roles:
  - inside an epoch, on the **Memory bus**, it is the *bookend* — it supplies a
    cell's starting value (seed at timestamp 0) and collects its ending value.
    It **replaces the PAGE table**, which is switched off inside continuation
    epochs.
  - across epochs, on the **GlobalMemory bus**, it carries each cell's
    "where did this value come from / where is it going" claims.
- **global_memory** — the *anchors* on the GlobalMemory bus:
  - **genesis**: a cell's starting value, read from the **ELF** (preprocessed,
    so the verifier recomputes it — the prover cannot choose initial memory).
  - **finalization**: a cell's final value after the last epoch that touched it.

### A single L2G row

```
   ┌──────────┬───────────────────────────┬───────────────────────────┐
   │ address  │ init: value, epoch, time  │ fini: value, epoch, time  │
   └──────────┴───────────────────────────┴───────────────────────────┘
     which       what it was when this        what it is at this
     cell         epoch first saw it, and       epoch's end, and this
                  which epoch wrote it          epoch's number
```

Column layout (13 columns): `address_lo/hi` (32-bit), `init_value` (byte),
`init_epoch` (two 16-bit halfwords), `init_timestamp` (four halfwords),
`fini_value` (byte), `fini_timestamp_lo/hi` (32-bit), `MU` (selector).

Note: **`fini_epoch` is NOT a column** — it is supplied as a per-table constant
(see §4.2).

### Cross-epoch telescoping

For a cell touched in epochs 1, 2, 3, the GlobalMemory bus checks:

```
 global_memory      L2G(ep1)        L2G(ep2)        L2G(ep3)      global_memory
   GENESIS  ───────►  init
   (value v0,         fini  ───────►  init
    from ELF)                         fini  ───────►  init
                                                      fini ───────► FINAL
                                                                    (last value)

   each  "fini ───► init"  is one matched token:
   epoch i's fini == epoch (i+1)'s init  (same address, value, epoch, timestamp)
```

The bus balances **iff** every `fini` is consumed by the next-touching epoch's
`init`, anchored by GENESIS (the one source) and FINAL (the one sink). That
chain *is* "memory stayed consistent across epochs." Inside each epoch, ordinary
memory checking (MEMW + timestamp ordering) handles consistency; L2G only
provides the seam at the edges.

---

## 3. Soundness, by component

The skeleton above is correct but not *sound* on its own — a cheating prover
could make the buses balance while lying. Four mechanisms close the gaps.

### 3.1 Range checks on the L2G columns

Raw field columns must be forced into their intended ranges, or a prover can
stuff out-of-range junk into them.

Principle: **only check what nothing else already checks.**

- `address`, `fini_timestamp`, the value bytes — these travel on the Memory bus
  and are matched against **MEMW**, which already range-checks them (exactly how
  PAGE relied on MEMW). No extra check.
- The **cross-epoch-only** fields (`init_epoch`, `init_timestamp`) have no MEMW
  partner, so L2G checks them itself: store as 16-bit halfwords, check each with
  the `IsHalfword` lookup, and rebuild the value as `lo + 2^16·hi`. Because only
  the range-checked halfwords feed the reconstruction, no extra AIR constraint is
  needed.

The value bytes get PAGE's batched `AreBytes` check (the `init` value is a
trusted source and must be checked).

### 3.2 `fini_epoch` as a per-table constant

Inside epoch *i*'s table, **every** row's `fini_epoch` is just *i*. So it does
not need to be a per-row committed column — it is supplied to the AIR as a
constant `epoch_label`, computed by the verifier from the epoch's position.

This is *strictly more sound* than a column: the prover cannot choose it. The
genesis sentinel is `0` and real epochs are labelled `1, 2, 3, …`
(`epoch_label(i) = i + 1`), so genesis is below every real epoch.

### 3.3 Cross-epoch ordering (the subtle one)

The GlobalMemory bus only proves the tokens **match as a set** — not that they
are chained in increasing-epoch order. Without that, a cheater can make a row's
`init` and `fini` cancel each other (point `init` at its own epoch), so the row
**vanishes** from the chain — letting an epoch read a *forged* value for a cell
while a later epoch absorbs that cell's real genesis. The bus balances; the
program ran on a lie.

Fix: force every row to reference a strictly earlier source —
`init_epoch < fini_epoch`. With genesis `= 0` and 1-based epochs, genesis (`0`)
satisfies it with no special case.

How `a < b` is checked without a dedicated comparison table (the same trick
MEMW uses for timestamps): in the field, `a < b` ⟺ `b − 1 − a` is a small,
in-range number. If `a ≥ b`, that subtraction wraps to a huge field element that
fails the range check. So we range-check `fini_epoch − 1 − init_epoch` with the
`IsB20` (20-bit) lookup — reusing the bit-table already present, near-zero cost.

```
   honest:  init=2, fini=5  →  5-1-2 = 2     small  ✓ passes
   cheat:   init=5, fini=5  →  5-1-5 = -1    wraps  ✗ fails  (self-reference)
   cheat:   init=9, fini=5  →  5-1-9 = -5    wraps  ✗ fails  (future reference)
```

Strict `<` (not `≤`) is required: `≤` would permit `init_epoch == fini_epoch`,
which is exactly the self-cancel that enables the forgery. Strict `<` guarantees
a real row's `init` and `fini` epochs always differ, so a real row can never
self-cancel.

Cost: this bounds the **number** of epochs to `< 2^20` (~1M) — *not* their size.
Unreachable in practice (optimal epochs are millions of cycles → thousands of
epochs even for a billion-cycle run) and fails closed. If ever needed, widen the
gap check to 32-bit or switch to the LT table.

### 3.4 The `MU` selector

Traces are padded with blank rows to a power of two (an FFT requirement). Those
padding rows must not disturb any bus.

Originally padding was harmless because a blank row's `init` and `fini` tokens
were identical and self-cancelled. But §3.2 (constant `fini_epoch`) broke that on
the GlobalMemory bus: a padding row's `fini` now carries `epoch = the constant`
while its `init` carries `epoch = 0`, so the tokens differ and no longer cancel.

Fix: a selector column `MU` (1 on real rows, 0 on padding). Interactions gated by
`Multiplicity::Column(MU)` contribute nothing on padding rows.

`MU` is itself constrained boolean (`MU·(1−MU)=0`), and pinned to the right
rows by bus balance (a real row with `MU=0` drops its telescoping link →
imbalance).

---

## 4. Design X vs Design Y — *where* `MU` is applied

`MU` is needed to neutralize padding, but **which** interactions should it gate?

```
                       GlobalMemory   Memory     range +
                       (telescoping)  (bookend)  ordering
   Design X:               MU            MU          MU      ← MU gates everything
   Design Y (chosen):      MU           One         One      ← MU only where needed
```

### Design X (earlier)

`MU` gates **every** L2G interaction. This was the first cut — it matched the
standard table pattern (LT/MUL/MEMW each gate all their interactions with one
multiplicity column) and kept padding handling uniform.

The drawback: only the GlobalMemory bus actually *needs* `MU`. On the Memory bus,
all-zero padding still self-cancels (that token carries no epoch field); on the
range-check buses, padding sends only valid lookups (`AreBytes[0,0]`,
`IsHalfword[0]`, `IsB20[epoch_label-1]`). Gating those with `MU` was redundant —
and gating the **ordering** check with `MU` is what made "could `MU=0` skip the
ordering?" a question at all. In Design X the answer ("no") relies on a cross-bus
argument (Statement S below).

### Design Y (chosen)

`MU` gates **only the GlobalMemory bus** — the one place padding genuinely
misbehaves. The Memory bus and the range/ordering checks use `Multiplicity::One`,
so they fire on every row. Consequences:

- The ordering check fires **unconditionally** → `MU` cannot gate or skip it.
  The Design X concern disappears *by construction*, not by argument.
- `MU`'s only job is GlobalMemory padding cancellation, where it is pinned by the
  GlobalMemory bus balance + the boolean constraint.
- The boolean constraint on `MU` must live in the **global** proof's AIR
  (`l2g_global_air`), the only place `MU` is used. (Implementation guardrail: a
  test that a row with `MU=2` fails to verify catches forgetting this.)
- Padding rows now fire the range/ordering lookups, so `collect_bitwise_from_l2g`
  emits the (valid, harmless) lookups for padding too.

Design Y is both more minimal and removes the only residual soundness concern.

### Statement S (the Design X anchor, for reference)

The cross-bus argument Design X relied on, and which Design Y avoids needing for
the ordering check:

> In a continuation epoch, the only table that provides a RAM cell's seed (its
> value at timestamp 0) on the Memory bus is L2G (PAGE is off). If a cell is
> accessed by MEMW during the epoch, the memory argument requires that seed; with
> `MU = 0` the seed is absent and the Memory bus cannot balance. Therefore any
> accessed cell is forced to `MU = 1`.

S rests on three checkable facts: (1) PAGE is off in continuation epochs;
(2) MEMW enforces timestamp ordering, so a cell's access chain must bottom out at
the seed; (3) no other table provides a RAM seed (REGISTER is registers only, a
disjoint token subspace). It is the existing memory-soundness argument with L2G
playing PAGE's seed-provider role. Design Y keeps S relevant only for `MU`'s
GlobalMemory correctness, not for the ordering.

---

## 5. Adversarial review summary

Three independent adversarial reviews were run against these mechanisms; each
tried to construct a forging prover and failed:

1. **`MU` safety (Design X).** Could `MU=0` on a real row, or a non-boolean `MU`,
   skip the ordering or forge a balance? No — caught by the Memory bus (Statement
   S) and the boolean constraint.
2. **Design Y.** Is the narrow `MU` placement sound, and does it remove the
   concern? Yes — padding stays harmless (self-cancels on Memory; valid lookups
   elsewhere); the ordering becomes unconditional; the "ghost row" attack fails
   on the GlobalMemory anchors. One mandatory guardrail: the `MU` boolean
   constraint must be in `l2g_global_air`.
3. **`fini_epoch` as a constant.** Sound — strictly more so than a column. Labels
   are verifier-computed from epoch position (unforgeable); prove/verify use
   identical labels (no off-by-one); the free `init_epoch` column and
   `global_memory`'s `FINI_EPOCH` column are both pinned by bus balance.

---

## 6. Status and open items

- Implemented and tested: range checks (§3.1), `fini_epoch` constant (§3.2),
  ordering check (§3.3), the `MU` selector (§3.4).
- The committed code currently wires `MU` as in **Design X**. **Design Y** is the
  agreed wiring (§4); switching `MU` to GlobalMemory-only (and moving the `MU`
  boolean constraint into `l2g_global_air`) is the next change.
- Known soundness gap, deferred: **cross-epoch register continuity** — epoch
  `i>0`'s register init is a prover-supplied snapshot, not yet bound to epoch
  `i-1`'s fini. This is independent of the memory work above.

---

## 7. Where the code lives

- `prover/src/tables/local_to_global.rs` — L2G columns, trace generation, the
  Memory/GlobalMemory bus interactions, range checks, the ordering lookup, and
  the per-row selector.
- `prover/src/tables/global_memory.rs` — the genesis (ELF-bound) and
  finalization anchors.
- `prover/src/continuation.rs` — the epoch loop, per-epoch proofs
  (`prove_verify_epoch`), the global proof (`prove_global` / `verify_global`),
  the per-epoch AIRs (`l2g_memory_air` / `l2g_global_air`), and the
  commitment binding.
