# The assume-guarantee boundary, re-derived for the joint chain

`RESULTS-lincomb2.md` §5 states the lincomb2 soundness theorem "under contracts
C1–C7 + A-PRIME (**unchanged from `RESULTS.md`**)" and §5's status line says the
board "rests on the same eight contracts as the original chips plus nothing
else". This pass re-derived each of the eight against the way `ecsm2.rs` /
`ecdas2.rs` / `ec_t0.rs` actually use them.

**Verdict: no contract is false, but the boundary as written is wrong in three
ways.** Two contracts are used well outside their own wording, one cites an
argument that does not exist in the document it points at, and the joint chain
depends on **three** properties no contract names at all — one of which is
demonstrably load-bearing (a working construction is included). "Plus nothing
else" should be withdrawn.

Nothing here is a chip bug. Every extension I could evaluate turns out to
*hold*; the problem is that the boundary does not say what is being relied on,
which is the failure mode that lets a later change quietly invalidate the board.

---

## 1. Lead findings

### 1.1 [NAME IT] C9 — the joint chain's compile-time curve constants are load-bearing and nothing in the proof checks them

The single-scalar chips have **no** compile-time curve point. `xG` arrives from
memory and is *proven* on-curve by the membership convolutions
(`ecsm.rs` `Relation::X2`/`Relation::Yg`, `:628-638`). The joint chain introduces three constant sources:

| constant | where | anchored to anything outside the AIR? |
|---|---|---|
| `G` = `GENERATOR_LE` | `ecsm2.rs:612-629` (P1 read), `:793-794` (P1 addend) | **yes** — the P1 read is a MEMW access, so the constant must equal the memory token the guest wrote at `a1` |
| `T₀` = `T0_X_LE`/`T0_Y_LE` | `ecsm2.rs:868-869` (phase-1 seed) | **no** |
| 256 × `−2^(j+1)·T₀` | `ec_t0.rs:131-153`, committed via `ec_t0.rs:190-209` + `prover/src/lib.rs:777-780` | **no** |

`G` is self-checking: a wrong `GENERATOR_LE` unbalances the Memw bus
(rejection), and executor-side it returns status 7 → software fallback →
correct answer. ✓ VERIFIED by reading the read at `ecsm2.rs:612-629` — the value
elements are `BusValue::constant`, so they are compared against the token.

`T₀` and the `EC_T0` rows have no such anchor. No guest write, no executor read,
no in-circuit curve check, no membership witness. They enter the proof as an AIR
constant and a preprocessed Merkle root the verifier compiles in. **If the
generator that produced them is wrong, every constraint and every bus still
balances and the chip returns a wrong `Q`.**

`c9_constants_probe.py` builds the construction, in the style of
`l6_joint_counting.py`: rebuild the full joint chain, re-derive the correction
row from a tampered addend by the group law, then *check* — not assert — all
four ECDAS2 convolution relations mod p on every row, all of idx 11..=27 on
every row, and all five buses (Ecdas telescoping, Addend, JointBit, EcT0 against
the tampered table, the result write). Two plausible generator bugs, both of
which the source itself warns about:

```
A. SIGN FLIP   table stores +2^len·T₀ (ec_t0.rs:28-41 warns: "reading y_t0_pow
               where you meant Y is a silent sign flip that still type-checks")
B. OFF-BY-ONE  row for len holds −2^(len+1)·T₀ (the ec_t0.rs:143 index)

  baseline (honest table)      61 rows, all relations + all buses OK, Q correct
  A  61 rows, all relations + all buses OK, Q_chain ≠ Q_true, canonical on-curve
  B  61 rows, all relations + all buses OK, Q_chain ≠ Q_true, canonical on-curve

  RESULT: 2/2 tampered constant sets produce an accepted, wrong Q.
```
(`logs/c9_constants.log`.)

The constants **are** correct today, and well covered:
`t0_is_on_curve_and_pinned` and `t0_derivation_matches`
(`crypto/ecsm/src/tests/lincomb2_tests.rs:316, 464`) re-run the
NUMS SHA-256 search; `every_entry_is_on_curve_and_canonical`,
`trace_rows_recompute_from_t0_by_doubling`,
`table_matches_lincomb2_witness_correction_row` and
`commitment_is_stable_and_matches_the_shipped_static_bytes`
(`prover/src/tests/ec_t0_tests.rs:96, 129, 242, 338`) check the table
independently and pin the shipped commitment. ✓ VERIFIED (test names read from
the file). Consistency between the seed and the table is *structural*, not a
coincidence: `neg_t0_pow2_points()` derives from `witness::t0()`, which is
`T0_X_LE`/`T0_Y_LE` — one source (`crypto/ecsm/src/lincomb2_table.rs:50-52`).

So this is not a live bug. The finding is that the board's contract list does
not say those tests are part of the *soundness argument*, so nothing tells a
future reader that regenerating the static commitment on a drift failure — which
`ec_t0.rs:185-189` already warns against — would launder a wrong-answer bug into
the verifier's trust anchor.

### 1.2 [NAME IT] C8 — byte-ness inheritance across a bus, and the layout that makes it valid

`WIDTH-AUDIT.md` finding #3 is right and is now bigger than it looks. Byte-ness
of the addend limbs is what the whole integer-lifting argument (L1/L2a) rests
on, and for three of the four addends it is *inherited* through tuple equality
rather than checked:

- `XB`/`YB` are deliberately not `AreBytes`-checked (`ecdas2.rs:543-544`);
- `X_P12`/`Y_P12`, `ACC_X`/`ACC_Y`, `X_Q`/`Y_Q` are not checked in ECSM2 either
  — the only `AreBytes` sends there are the membership sub-witness
  (`ecsm2.rs:692-705`).

The inheritance is valid **only** because every point tuple carries one bus
element per byte (`point_coord_busvalues`, `ecsm.rs:272-286` — 32 ×
`Packing::Direct`) so tuple equality is per-limb. ✓ VERIFIED. That function now
carries a strong "do not repack" comment (WIDTH-AUDIT finding #3 was actioned),
but it is a *comment*: nothing mechanical stops a future "shrink the Addend bus
with `Word4L`" change, and the gate's contract list does not mention it. Under
`Word4L` a receiver could satisfy the same packed value with a different
decomposition and reachable limb magnitudes run to ~2^63, against a ~2^29
breaking threshold (`WIDTH-AUDIT.md` §3.1).

There is a **second, entirely unstated invariant in the same family**: per-bus
tuple-length discipline. A shorter tuple's fingerprint equals that of a longer
tuple on the same bus whose extra *trailing* elements are all zero — the
fingerprint is `bus_id + Σ_l v_l·α^{l+1}` and the shorter tuple simply
contributes no terms past its length. Bus 28 carries two chip families with
different tuple lengths (133 old, 70 new), and what separates them is the
chain-id constant at position 0 (`ecsm.rs:616` `constant(0)` vs
`ecdas2.rs:411` `constant(1)`). ✓ VERIFIED by reading both tuple builders. That
is a real property and it holds — but it is exactly the sort of thing a length
change would break silently.

⚠ Three source comments justify the related `sel ≠ 0` / `stream ≠ 0` conventions
with a **false mechanism**: `types.rs:351-357` (`BusId::Addend`),
`types.rs:375-386` (`BusId::JointBit`) and `ecsm2.rs:110-119` all say zero
elements are "skipped by the fingerprint" so trailing-zero padding shifts
positions. It does not. ✓ VERIFIED at `lookup.rs:1655` — `alpha_offset += ...`
unconditionally — and `lookup.rs:679`, where the Linear arm returns `1` even
when it skips the multiply for a zero value. Interior zeros consume their α slot
and change nothing. `L6-COUNTING.md` §5.2 already flagged this for
`JOINT_CHAIN_ID`; the same wrong model is in two more places, and
`BusId::JointBit`'s version is doubly wrong — a `stream = 0` tag could not alias
an old-chain `Bit` send anyway, because `Bit` is bus 30 and `JointBit` is bus 33,
and the bus id enters the fingerprint at α⁰.

### 1.3 [FIX THE CITATION] C5 points at a document that contains no soundness argument

C5 reads: "*LogUp multiset soundness: exact signed balance per bus (generic
argument, `spec/logup.typ`)*."

✓ VERIFIED by reading all 146 lines of `spec/logup.typ`: it is a **protocol
description**. Notation, the four-step protocol, and three "running sum
constraint choices" with one `#aside("Justification")` — which justifies the
*bookkeeping* of choice 3, not the multiset argument. There is no theorem, no
Schwartz–Zippel step, no statement of what balance implies. Grepping the whole
of `spec/` and `crypto/stark/src/` for `Schwartz|multiset|log-derivative|Haböck`
returns exactly one file: `spec/ecsm.typ`, the EC chapter, which *cites* "the
`LogUp` multiset argument" as a contract (`:238`, `:679`). The argument is not
written down anywhere in the repo that I could find. ✗ UNCERTAIN whether it
exists outside the repo.

That matters here because the joint chain uses three forms the original chips
did not, and the lead's question — "does the argument cover them?" — cannot be
answered against the cited text at all. Against the *implementation* and the
standard log-derivative lemma, all three are fine:

| new form | where | verdict |
|---|---|---|
| `Multiplicity::Linear([{coefficient: 2, column: u_bit(i)}])` | `ecsm2.rs:775-778` | ✓ fine. Multiplicities are arbitrary base-field expressions; `emit_multiplicity` (`lookup.rs:1925-1941`) mirrors `evaluate_with` (`lookup.rs:1369-1399`) arm for arm, including `Linear`, so prover and AIR agree by construction |
| `Multiplicity::Linear` with four terms for the Addend receive | `ecdas2.rs:486-504` | ✓ fine, same reason. `Sum3` could not cover `S_CORR`; the `Linear` fallback is not a weaker object |
| key `Linear([Column{LEN_M1}, Constant(1)])` | `ecsm2.rs:822`, table side `ec_t0.rs:354-360` | ✓ fine. A key is an arbitrary linear form; both sides compute `len` and the 256 unpadded rows publish exactly `[1, 256]`, so balance forces `LEN_M1 ∈ [0, 255]` with no consumer-side range check |
| unconstrained count columns `N1`/`N2`/`N3` | `ecsm2.rs:196-202`, `:795-808` | ✓ fine **with a side condition**, below |

The count-column claim in the source ("an inflated count leaves an unmatched
send and a 'negative' one is unrepresentable") is right in substance but the
reasoning needs one more step, and that step is a **side condition C5 does not
state**: balance is an equality *in `F_p`*, not over the integers. `N2` is pinned
to `#receives mod p`; a prover setting `N2 = p − 1` would need `p − 1` receives,
impossible only because the trace has far fewer than `p` rows. The same lifting
is what makes the L6 counting argument work — `#{rows at round i with D = 1} =
2·u_bit(i)` is an `F_p` identity, and it forces the *integer* count into `{0, 2}`
only because each row contributes 0 or 1 (IS_BIT) and the total row count is
`≪ p`. Both uses are safe by an enormous margin (Goldilocks `p ≈ 1.8·10^19` vs
`≤ 2^30` rows), but the restated C5 should say so, because it is the hypothesis
that "2 = 1 + 1 is the only decomposition" leans on.

### 1.4 [RESTATE] C4 is invoked well outside its own text — and the extension holds by a mechanism C4 does not name

Confirmed, as suspected. C4's text is about `xG`/`k`/`xR` and ends with an
ECSM-specific clause ("*ECSM's YR inherits byte-ness from tuple equality with
ECDAS's byte-checked yR (or YG for k=1)*"). The joint board invokes it for
`P2`, `P12`, `xQ`, `yQ` and the T₀ constants, which fall into three different
classes:

| value | how byte-ness is really obtained | C4 covers it? |
|---|---|---|
| `X_P2`/`Y_P2` | the MEMW read binds them per byte to a memory token; every writer of a token range-checks its bytes (STORE `store.rs:235-247`, PAGE `page.rs:462-475`, L2G `local_to_global.rs:430`) | in spirit (same class as `xG`) |
| `X_P12`/`Y_P12`, `ACC_X`/`ACC_Y`, `X_Q`/`Y_Q` | inherited from the ECDAS2 row's `AreBytes`-checked `XR`/`YR` (`ecdas2.rs:552-565`) through Ecdas tuple equality — an **interior chip output**, never a memory value | **no** |
| `X_T0N`/`Y_T0N` | the `EC_T0` preprocessed table's committed byte columns | **no** — that is C9 |

The `P12` chain does discharge, and I re-verified it end to end rather than
inheriting WIDTH-AUDIT's trace: ECSM2's phase-0 drain receive is at multiplicity
`OK` (`ecsm2.rs:849-861`), so it must be matched. The only senders on bus 28 with
`chain_id = 1, phase = 0, round = −1, op = 0` are ECDAS2 sends at multiplicity
`MU` (`ecdas2.rs:616-638`) — the ECSM2 seeds carry `phase ∈ {0,1,2}` with
`round ∈ {0, LEN_M1, 0}` and `LEN_M1 ≠ −1` is forced by the `EcT0` lookup, and
across ECSM2 rows `ts` differs. So the matching row has `MU = 1`, hence its
`AreBytes` sends fire, hence its `XR`/`YR` are bytes, hence `X_P12`/`Y_P12` are.
✓ VERIFIED. `X_Q`/`Y_Q` follow the same path through the phase-2 drain, which
matters twice over: their byte-ness is what makes the `X_Q_SUB_P` word packing
(`ecsm2.rs:1119-1129`) a genuine `< p` proof, and they are the bytes the guest
keccaks.

C4 should therefore be **split**, not stretched: a global memory-token invariant
(C4) plus the inheritance rule (the new C8). As written it names neither.

**And two of C4's three clauses are wrong about the mechanism even for the old
chips**, which is why the joint chain had nothing accurate to inherit:

- "*`k` bytes are range-checked at memory-write time*" — no. `k` never rides the
  Memw bus as columns: `k_byte_busvalue` (`ecsm.rs:290-300`) reconstructs each
  byte as `Σ 2^j·k_bit`, so byte-ness is in-chip IS_BIT (C3), not C4. Identical
  for `u1`/`u2` via `scalar_byte` (`ecsm2.rs:505-514`). Harmless, but it means
  C4 has been over-claiming its own scope from the start.
- "*and `xR` bytes at the ECSM MEMW write*" — no. ✓ VERIFIED: **not one of the
  three MEMW variants contains an `AreBytes` send** (`grep -c AreBytes` over
  `memw.rs`, `memw_aligned.rs`, `memw_register.rs` → `0, 0, 0`). A MEMW write
  range-checks nothing; it transports whatever the sender supplies. ECSM's own
  `AreBytes` sends cover `X2`, `Q0`, `YG`, `Q1` only (`ecsm.rs:470-478`) — `XR`
  is not among them. So `xR` inherits byte-ness by *exactly* the mechanism C4's
  next clause grants only to `yR`: tuple equality with the ECDAS drain.

So C4's only true content, in either chip family, is the memory-token invariant
applied to one value — `xG` then, `P2` now. Everything else it claims belongs to
C3 (in-chip IS_BIT) or C8 (inheritance).

**On the transcription audit's flag** (`ecsm2.rs:631-649`, `P2` from the 8 MEMW
dword reads): agreed that it is an unstated instance, with one refinement. The
*mechanism* is not new — `operand_dword` (`ecsm2.rs:488-497`) emits 8
`Packing::Direct` elements per doubleword, byte for byte, exactly as the old
chip's `dword_bytes` (`ecsm.rs:259-262`) does for `xG`. What is new is only the
value. The defect is structural: C4 is written as an **enumeration of values**
rather than as an invariant, so every value the next chip binds is "unstated" by
construction. Restating it as the invariant fixes this instance and all future
ones at once.

### 1.5 [RESTATE — it is stronger than "assumed"] C7 carries four buses now, and for ECSM2 it is discharged in-chip

Every joint bus is keyed by `ts`: `Ecdas` (28), `Addend` (29), `EcT0` — no, that
one is keyed by `len` only, correctly, since it is a pure function lookup —
`JointBit` (33). So `ts` really is doing the cross-call separation for three
buses instead of two.

**But C7 need not be assumed for these chips.** Every ECSM2 row that
participates in anything has `MU = 1` (`OK·(1 − MU) = 0`, idx 2, plus idx
517..=521 killing the raw-column multiplicities on `OK = 0` rows), and every
`MU = 1` row performs a combined read+write of register `x10` at its own `ts`
(`ecsm2.rs:568-582`). Two ECSM2 rows sharing a `ts` would be two accesses to the
same address at the same timestamp; the memory token chain at that address is a
total order with strictly increasing timestamps, enforced by
`old_timestamp[i] < timestamp` (`memw.rs:757-830`, ALU LT) or
`IS_HALF[ts − old_ts − 1]` (`memw_register.rs:286-301`). ✓ VERIFIED. So
`ts`-uniqueness *for lincomb2 ecalls* follows from the memory argument, and C7
degrades from an assumption to a consequence.

Independently, C6 alone already forces it: the CPU sends exactly one `Ecall`
tuple `[ts, 0, rv1]` per ecall (`cpu.rs:982-998`) and ECSM2 receives one at
multiplicity `MU` with the syscall constant (`ecsm2.rs:553-562`), so two `MU = 1`
rows at one `ts` mean two receives against one send.

Two problems with the *evidence* C7 cites:

- ✓ VERIFIED **`trace_builder.rs:341-347` is the honest builder, not a
  constraint.** It is `let timestamp = (i as u64) * 4 + 4;` — it says the prover
  *chooses* distinct timestamps, which proves nothing against a malicious
  prover. Citing it alongside the in-proof mechanism blurs the two.
- ✓ VERIFIED **`cpu.rs:541-542` has drifted.** Those lines are now a blank line
  and a comment about `rv1/rv2/arg2`. The PC-token cadence actually lives at
  `cpu.rs:822-892` (the inline-PC `Memory` sender/receiver pair: consume at
  `ts − 3`, emit at `ts + 1`, forcing `ts' = ts + 4` along the chain) and
  `cpu.rs:560-574` (the padding-row comment). Worth noting that the cadence
  comment at `cpu.rs:827` cites `docs/cpu-rework-deviations.md`, **which does not
  exist anywhere in the repo** — I searched for it by name. So the C7 evidence
  trail has one stale line reference and one dangling document.
- `spec/memory.typ` does contain the load-bearing part — temporal integrity,
  "the newly emitted token must have a strictly greater timestamp than the
  consumed token" (`:102-110`), plus the design note that same-timestamp accesses
  to the same address are impossible (`:57-58`, `:67-75`). ✓ VERIFIED. It does
  *not* contain a PC-cadence argument; that is a chip property, not a spec one.

**On `PHASE` and the chain id: they do real work, `ts` is not doing it alone.**
`ts` separates *calls*; within one call `phase` (constant `0/1/2` in every seed
and drain, relayed unchanged along every ECDAS2 row) is what stops the three
segments merging — notably the phase-1 drain and the phase-2 seed are sent from
the *same* `ACC_X`/`ACC_Y` columns (`ecsm2.rs:877-907`) and would cancel each
other outright if `phase` were not in the tuple. The chain id separates the two
chip families sharing bus 28. ✓ VERIFIED by reading both tuple builders.

---

## 2. The board

| # | restated for the joint chain | status | evidence | verdict |
|---|---|---|---|---|
| **C1** AreBytes | every element of an `AreBytes[x, y]` send is in `[0,256)`, for **both** slots, over the paired layout | **assumed** (preprocessed table) — and it holds in the stronger usage | table is 2^20 rows indexed `x + 256y + 65536z` with `X`/`Y` columns equal to those bytes (`bitwise.rs:98, 117-152`); receiver keys on `(X, Y)` (`bitwise.rs:783-796`) — every byte pair exists, no non-byte does | ✓ holds. Pairing changed the number of sends, not the per-element guarantee |
| **C2** IsHalfword | sent value ∈ `[0, 2^16)` | **assumed** (same table) | receiver `[X + 256·Y]` (`bitwise.rs:797-810`) over all byte pairs = exactly `[0, 2^16)` | ✓ holds |
| **C3** IS_BIT/booleans | now: ECSM2 `MU`, `OK`, 512 scalar bits, `mem_q1[32]`, 35 overflow carry bits; ECDAS2 `MU`, `OP`, `NB`, `D1`, `D2`, `S1..S3`, `S_CORR`, `PH1`, `PH2` | **discharged** in-chip, list extended | `ecsm2.rs:1148-1153, 1181-1189, 1227-1231, 1243-1247`; `ecdas2.rs:850-870` | ✓ holds; C3 is a note about in-chip constraints, not an external assumption — say so |
| **C4** MEMW byte authority | **global**: every value element of a memory token is in `[0,256)`, maintained by every writer | **assumed** (whole-VM invariant) | STORE `store.rs:235-247`; PAGE `page.rs:462-475`; L2G `local_to_global.rs:430`; MEMW binds per byte via `Packing::Direct` (`memw.rs:567-590`) | ✓ holds for `P2`. **Text does not cover** `P12`/`xQ`/`yQ`/T₀ — see §1.4 |
| **C5** LogUp multiset soundness | balance **in `F_p`** per distinct fingerprint, w.h.p. over the two challenges; integer conclusions need contributing totals `≪ p` | **assumed**, and the **cited argument does not exist in the cited file** | `spec/logup.typ` is protocol-only (§1.3); implementation supports every new form (`lookup.rs:1328-1362, 1369-1399, 1925-1959`) | ✓ the three new forms are within it; ✗ the citation is empty; side condition unstated |
| **C6** Ecall binding | each `MU = 1` ECSM2 row ↔ exactly one executed lincomb2 ecall, both directions | **discharged** by balance | CPU send `cpu.rs:982-998`; ECSM2 receive `ecsm2.rs:553-562`; syscall numbers differ from ECSM (`execution.rs:39, 47`) | ✓ holds, and is now bidirectional (an unmatched CPU send would also reject) |
| **C7** Timestamp uniqueness | distinct lincomb2 ecalls have distinct `ts`; `ts` + `phase` + chain id separate the four keyed buses | **discharged** for ECSM2 (was: assumed) | x10 access `ecsm2.rs:568-582` + temporal integrity `memw.rs:757-830`, `memw_register.rs:286-301`, `spec/memory.typ:102-110`; and C6 independently | ✓ holds, **stronger than stated**; two of its three citations are stale/misleading (§1.5) |
| **A-PRIME** | `p`, `N` prime; `N` odd | **assumed** (sympy-certified) | unchanged | ✓ holds. Still load-bearing: L4a needs `p` prime; L5a (no `y ≡ 0` point) still discharges the doubling side condition and still consumes `N` odd. L5c is no longer used — L5b′ replaced it |
| **C8** *(new)* byte-ness inheritance | a receiver inherits limb range checks from a sender **only** while every point coordinate rides as 32 × `Packing::Direct`, and tuples on one bus never differ by trailing zeros | **assumed**, unstated | `ecsm.rs:272-286`; `lookup.rs:1655, 679`; bus-28 length 133 vs 70 separated by the chain-id constant | ✓ holds today; **must be named** (§1.2) |
| **C9** *(new)* EC constants | `T₀` is the intended on-curve NUMS point, and the 256 `EC_T0` rows are `−2^(j+1)·T₀`, bound by the verifier's compiled-in preprocessed commitment | **assumed**, unstated, **demonstrably load-bearing** | probe: 2/2 tampered tables accepted with a wrong `Q` (`c9_constants_probe.py`, `logs/c9_constants.log`); discharge is test-time only (§1.1) | ✓ holds today; **must be named** |
| **C10** *(new)* caller status discipline | the caller treats **every** non-zero status identically (full software fallback), so the prover's free choice of `STATUS` cannot change the program's output | **discharged** by the caller | `crypto/ethrex-crypto/src/lib.rs:172-178` (`if status != OK { return None }` → `unwrap_or_else(ProjectivePoint::lincomb)`), documented at `:146-162` and `syscalls.rs:41-44` | ✓ holds; worth naming because it is chip-external and a future caller could break it |

---

## 3. Proposed restatements

Replacements for the contract block at `RESULTS.md:164-181`, to be referenced
(not re-copied) from `RESULTS-lincomb2.md`:

- **C1 AreBytes** — every element of an `AreBytes[x, y]` send lies in `[0,256)`.
  Both slots; single-byte checks pass `y = 0`. Rests on the BITWISE preprocessed
  table being committed and containing all `(x, y)` byte pairs
  (`bitwise.rs:98, 117-152`, receiver `:783-796`).
- **C2 IsHalfword** — a sent value lies in `[0, 2^16)` (`bitwise.rs:797-810`),
  same table.
- **C3 IS_BIT/booleans** — *(not an assumption; a pointer.)* Every column used as
  a bus multiplicity or a one-hot selector carries an in-table `x·(1−x)`.
  Joint-chain list: ECSM2 `MU`, `OK`, `u1[0..256]`, `u2[0..256]`, `mem_q1[32]`,
  the 35 overflow carry bits; ECDAS2 `MU`, `OP`, `NB`, `D1`, `D2`, `S1`, `S2`,
  `S3`, `S_CORR`, `PH1`, `PH2`.
- **C4 Memory-token byte authority** — every *value* element carried on the
  memory-token bus is in `[0,256)`, because every chip that emits a token
  range-checks the bytes it writes (STORE, PAGE, L2G, and each accelerator for
  its own output). Consumers therefore inherit byte-ness for anything they bind
  to a token per byte. **This covers memory-resident operands only** — for the
  joint chain, `X_P2`/`Y_P2`. Interior values (`P12`, `ACC`, `xQ`/`yQ`) are
  covered by C8, not by this.
- **C5 LogUp multiset soundness** — for random challenges sampled after the
  trace commitment, per-bus balance implies, with probability
  `1 − O(max tuple length / |𝔾|)`, that for **each distinct fingerprint** the
  signed sum of multiplicities is zero **in `F_p`**, and that two interactions
  with equal fingerprints have equal `(bus_id, value-vector)`. Multiplicities and
  tuple elements may be arbitrary linear forms over columns and constants;
  coefficient-2 multiplicities, four-term sums and constant-offset keys are
  covered. *Side condition, used by every counting argument:* an `F_p` balance
  lifts to an integer statement only when the total contribution to a single
  fingerprint is bounded well below `p` — here by (rows in the table) × (max
  per-row multiplicity), i.e. `≤ 2^31 ≪ 2^64`. **The generic argument is not
  written down in `spec/logup.typ`, which is a protocol description; this
  contract currently has no citable proof in-tree.**
- **C6 Ecall binding** — the CPU sends exactly one `Ecall[ts, 0, syscall]` per
  executed ecall, so a chip's `MU`-gated receive at a given syscall constant
  corresponds one-to-one with real ecalls of that kind, in both directions.
- **C7 Timestamp uniqueness** — *(now derived, for these chips.)* Distinct
  accesses to one address carry distinct timestamps, because the per-address
  memory-token chain is totally ordered and every MEMW variant enforces
  `old_ts < ts` (`memw.rs:757-830`; `memw_register.rs:286-301`;
  `spec/memory.typ:102-110`). Since every live ECSM2 row read-writes `x10` at its
  own `ts`, distinct lincomb2 ecalls have distinct `ts`. Separation of the joint
  buses is then `ts` (across calls) × `phase` (across the three segments of one
  call) × chain id (across the two chip families on bus 28) × `stream`/`sel`
  tags. *Do not cite `trace_builder.rs` for this — that is the honest builder.*
- **C8 Bus-carried range inheritance** *(new)* — a receiver may inherit a limb
  range check from a sender's checked columns iff the two tuples place those
  limbs in the same positions as **individual** bus elements
  (`Packing::Direct`, one element per byte — `ecsm.rs:272-286`), and no two
  interactions on the same bus differ only by trailing zero elements. Repacking a
  point coordinate (`Word4L`) or shortening a tuple to a zero-padded prefix of
  another on the same bus invalidates every inheritance and, via `WIDTH-AUDIT`
  §2, the integer-lifting argument itself.
- **C9 EC constants and preprocessed tables** *(new)* — the compile-time
  constants the chips assert rather than prove are the intended ones: `T₀`
  (`witness.rs:490-497`) is the on-curve NUMS point of `T0.md`, and the `EC_T0`
  table's 256 rows are `−2^(j+1)·T₀` under a preprocessed commitment the verifier
  computes from its own compiled-in constants (`ec_t0.rs:190-274`,
  `prover/src/lib.rs:777-780`). `G` is exempt: it is anchored to guest memory by
  the P1 MEMW read, so a wrong value can only reject or fall back.
- **C10 Caller status discipline** *(new)* — the calling program treats every
  non-zero status word as "the accelerator produced nothing" and re-computes in
  software, so a prover's free choice of `STATUS` on an `OK = 0` row costs cycles
  and cannot change the program's output.
- **A-PRIME** — `p` and `N` prime (hence `N` odd). Unchanged, and still
  load-bearing for the joint chain via L4a (`p`) and L5a (`N` odd). L5c's use is
  retired with L5b.

Two smaller edits worth making while the file is open:

1. Withdraw "*plus nothing else*" from `RESULTS-lincomb2.md` §5 and replace the
   §5 preamble with "under C1–C10 + A-PRIME (C1–C7 + A-PRIME as restated in
   `CONTRACT-BOUNDARY.md`; C8–C10 are new to the joint chain)".
2. Fix the three false "zeros are skipped by the fingerprint" justifications
   (`types.rs:351-357`, `types.rs:375-386`, `ecsm2.rs:110-119`) — same correction
   `L6-COUNTING.md` §5.2 already made for `JOINT_CHAIN_ID`.

---

## 4. What I could not determine

- **Whether the generic LogUp argument exists anywhere.** I established it is not
  in `spec/logup.typ` and not elsewhere in `spec/` or `crypto/stark/src/`. It may
  exist in a design note outside the repo. If it does not, C5 is the largest
  unproven object the entire EC board rests on, and it is shared with every other
  chip.
- **Whether the global C4 invariant actually holds for every token writer.** I
  verified STORE, PAGE and L2G range-check their bytes. `commit.rs` contains no
  `AreBytes` and `keccak.rs` one; I did not trace whether those chips' written
  bytes are byte-bounded by other means (bit decomposition, etc.). It does not
  affect the joint chain's *own* operands unless a guest can point `a2` at
  accelerator-written memory, which for ecrecover it does not — but the restated
  C4 is a whole-VM claim and I have only spot-checked it.
- **Completeness of the honest carry windows for the varying addend.**
  `WIDTH-AUDIT.md` §4 is explicit that its table is a measurement over 6,346
  rows, not a proof. Unchanged by this pass; completeness-only.
- **The `PH*`/`S*` modelling gap** already recorded in `RESULTS-lincomb2.md` §7:
  those are derived by `phase_bits`/`selector_bits` rather than being witness
  fields, and my probe reproduces that mapping from the same source, so it shares
  the gap.
- **Whether `LEN_M1` needs `ROUND`-style byte-ness on an `OK = 0` row.** It does
  not, as far as I traced — every consumer of `LEN_M1` (`EcT0` send, phase-1 seed
  round) is `OK`-gated — but I did not enumerate the ECSM2 column set
  exhaustively for other free-on-error-row columns. That enumeration belongs to
  the transcription audit, not here.
- **Cross-epoch behaviour.** All of the above assumes one proof. Whether `ts`
  uniqueness survives continuations (L2G/GlobalMemory boundaries) for
  accelerator ecalls is untouched by this pass.

---

## 5. Reproduction

```sh
cd /Users/maurofab/workspace/lambda_vm
python3 thoughts/ec-recover-opt/gate/c9_constants_probe.py     # 2/2 forgeries
```
Pure stdlib (no venv, no z3); reference machinery is `oracle/ec_ref.py` +
`oracle/lincomb2_ref.py`. Log: `logs/c9_constants.log`.
