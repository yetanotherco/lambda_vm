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
   Design X (SOUND):       MU            MU          MU      ← MU gates everything
   Design Y (UNSOUND):     MU           One         One      ← MU only on GlobalMemory
```

**Conclusion up front: Design X is sound; Design Y is *not*.** We initially
believed Y was a cleaner equivalent (and two adversarial reviews agreed). They
were wrong — Y opens a chain-truncation attack. Below is X, then Y, then the
attack and why X blocks it.

### Design X

`MU` gates **every** L2G interaction (matches the standard table pattern —
LT/MUL/MEMW each gate all their interactions with one multiplicity column).

The crucial consequence — which we first mistook for redundancy — is that gating
the **Memory bus bookend** with `MU` forces `MU = 1` on every *touched* cell:
a touched cell's MEMW accesses need the L2G seed/fini on the Memory bus (PAGE is
off), so `MU = 0` would dangle them → the epoch proof fails. This is **Statement
S** below. Forcing `MU = 1` on every touched cell forces every touching epoch
**into the global chain** — so the chain is **complete**, and cannot be truncated.

### Design Y (rejected — unsound)

`MU` gates **only the GlobalMemory bus**; the Memory bus and range/ordering checks
use `Multiplicity::One`. The intended win was that the ordering check then fires
unconditionally so `MU` can't skip it. But decoupling the Memory bookend from `MU`
**broke Statement S**: a touched cell's bookend now fires regardless of `MU`
(`Multiplicity::One`), so the epoch proof passes even with `MU = 0`. Nothing then
forces `MU = 1` on a *non-first-touch* row — and that is exploitable.

### The attack Design Y allows: orphan a touched epoch

Cell A, touched by epochs e1 then e2. Honest: genesis `v0` → e1 writes `f1` →
e2 writes `f2` → final `f2`. A cheating prover sets **`MU = 0` on e2's L2G row**
and sets `global_memory`'s finalization for A to `f1`:

```
   genesis(v0) ──► e1.init        ✓ (genesis must be consumed — forces e1 only)
   e1.fini(f1) ──► FINAL(f1)      ✓ (prover-chosen finalization absorbs it)
   e2.init / e2.fini              ✗ MU=0 — orphaned, don't fire
```

- The GlobalMemory bus **balances** (every fired token matched).
- e2's **epoch proof still passes** — in Design Y its Memory bookend is
  `Multiplicity::One`, so it fires regardless of `MU`; e2 ran internally-consistently.
- **Nothing forces `MU_e2 = 1`:** e2 isn't first-touch (genesis went to e1), and
  the finalization is a *prover column*, so it just absorbs whatever the last fired
  fini was.

Result: e2's write to A is silently dropped — A's final value is claimed `f1`
when it's really `f2`. A false statement, proven. (For a *middle* epoch, reroute
the later init to consume the earlier fini, skipping the middle one.)

The root cause is the **input/output asymmetry** of the anchors: genesis is the
*input* and is ELF-bound (fixed), but the finalization is the *output* — a prover
column. The finalization is only trustworthy if the chain is **complete** so that
the last fini is *forced* to be consumed by it. A complete chain pins the
finalization; a truncatable chain leaves it free. Design X forces completeness
(via `MU=1` on every touched cell); Design Y does not.

### Statement S (why Design X is sound, and what Y broke)

> In a continuation epoch, the only table that provides a RAM cell's seed (its
> value at timestamp 0) on the Memory bus is L2G (PAGE is off). If a cell is
> accessed by MEMW during the epoch, the memory argument requires that seed; with
> `MU = 0` the seed is absent and the Memory bus cannot balance. Therefore any
> accessed cell is forced to `MU = 1`.

S rests on three checkable facts: (1) PAGE is off in continuation epochs;
(2) MEMW enforces timestamp ordering, so a cell's access chain must bottom out at
the seed; (3) no other table provides a RAM seed (REGISTER is registers only, a
disjoint token subspace).

**S requires the Memory bookend to be `MU`-gated** — that is exactly what Design X
has and Design Y removed. So the "redundant" `MU` on the Memory bus in Design X is
in fact load-bearing: it's what forces every touched epoch into the chain, making
the chain complete and the finalization trustworthy.

### The anchoring chain (why a real access cannot be dropped at all)

`MU = 1` being forced bottoms out at the program itself:

```
   ELF ─DECODE(preprocessed)─► each row's instruction (LOAD/STORE flags) is fixed
   PC-continuity ───────────► every executed instruction is present, in order
        │
   ▼ a real load/store row has its flag = 1 (DECODE match + IsBit) ⟹ CPU sends Memw req
   ▼ MEMW must receive it (MU_READ/MU_WRITE) — dropping it ⟹ Memw-bus imbalance
   ▼ MEMW's bookend pairing needs the L2G seed/fini — in Design X (MU-gated) ⟹ MU=1
   ▼ MU=1 ⟹ the cell is in the global chain ⟹ chain complete ⟹ finalization pinned
```

This is the VM's core execution soundness (DECODE + PC-continuity + IsBit flags,
verified in `cpu.rs` / `constraints/cpu.rs`), extended one link at a time up to
cross-epoch memory. Design X keeps every link; Design Y cut the MEMW→L2G link.

### How `global_memory`'s finalization is constrained — and the parallel with `main`

The finalization is **not** checked against an external value (it's the computed
output, not a known input). It is pinned **internally** by the bus: it must consume
the last fini of each cell's chain, which (with a complete chain) is the cell's
real last-written value. This is exactly how **PAGE** works in the monolithic
prover — PAGE's `fini` is pinned by the (single, complete) Memory bus to the last
MEMW write. Design X is the faithful cross-epoch extension; Design Y silently
dropped the "chain is complete" property both rely on.

---

## 5. Adversarial review summary

1. **`MU` safety (Design X).** Could `MU=0` on a real row, or a non-boolean `MU`,
   skip the ordering or forge a balance? No — caught by the Memory bus (Statement
   S) and the boolean constraint. **Holds.**
2. **Design Y.** Two adversarial reviews concluded Y was sound (padding harmless,
   ordering unconditional, "ghost row" attack defeated). **They were wrong.** Both
   only tested *first-touch* `MU=0` (genesis dangles → caught) and added/forged
   rows; neither tested **truncating the chain at a non-first-touch row** while
   pointing the prover-controlled finalization at the truncation. That attack (§4)
   makes Y unsound. Lesson: a review that misses an attack class proves nothing
   about it — the truncation/orphan class was the gap.
3. **`fini_epoch` as a constant.** Sound — strictly more so than a column. Labels
   are verifier-computed from epoch position (unforgeable); prove/verify use
   identical labels (no off-by-one); the free `init_epoch` column and
   `global_memory`'s `FINI_EPOCH` column are pinned by bus balance **when the chain
   is complete** (Design X). Independent of the X/Y choice.

---

## 6. Status and open items

- Implemented and tested: range checks (§3.1), `fini_epoch` constant (§3.2),
  ordering check (§3.3), the `MU` selector (§3.4).
- **The committed code implements Design X** (`MU` gates every L2G interaction),
  which is the sound design. Design Y was implemented briefly, then found unsound
  (§4, the chain-truncation attack) and **reverted**. Do not re-introduce the
  Design Y wiring: gating only the GlobalMemory bus reopens the orphan attack.
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
