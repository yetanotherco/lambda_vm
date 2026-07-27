# lincomb2 — implementation plan (phases B–H)

Execution plan for DESIGN.md §8 steps (3)–(6). Phase A (witness + T₀) is
**DONE** and committed (`bc62f00e`); its measured outputs are locked in
`layout-lock.md` and are the contract every phase below builds against.

This document is written to be **agent-consumable**: each phase is a
self-contained brief with a touch list, an acceptance gate, and explicit
"do not touch" boundaries, so phases can be delegated in parallel where the
dependency graph allows.

> ## ⚠️ DESIGN §4's NUMS argument is BROKEN — do not ratify the assumption
>
> The blinding does **not** close the incomplete-addition edge, because the
> prover chooses P2 (= `lift_x(r)`, free signature bytes). Setting
> **P2 = μ·T₀** cancels the T₀ coefficient and makes the collision condition
> solvable with **no knowledge of dlog(T₀)** — with P1 = G and u1 = 1 it is
> `μ·(c2 − 1) ≡ −2^j (mod N)`, one modular inversion. That is *cheaper* than
> the attack DESIGN describes against the **unblinded** chain: the blind made
> the edge easier to aim at, not harder.
>
> Verified 5/5 (len ∈ {8,12,16,32,256}): `acc == addend` with equal y at the
> targeted round ⇒ `λ` unconstrained ⇒ forgeable row, and the packaged
> `(z,v,r,s)` is well-formed (the guest's own decomposition reproduces u1/u2
> and lifts R back to P2). Repro `oracle/nums_blinding_probe.py`, writeup
> `FINDING-nums-blinding.log`, independently re-run.
>
> **Phase A is unaffected** — the honest witness refuses every case
> (`ResultInfinity` → status ≠ 0 → fallback). The hole is on the
> malicious-prover side, where the row is hand-crafted. It is a **chip**
> obligation.
>
> **FIXED — `D_INV·(xB − xA) ≡ 1 (mod p)` has LANDED**, gated by `ΣS` (the
> same expression that counts the Addend receive — strictly better than the
> `OP` gating originally proposed, and lemma (a1) proves the two coincide on
> live rows), covering all five addend-consuming row types including the
> correction row, never doubles. ECDAS2: 529 → **658 cols**, 217 → **288
> constraints**; cells/row 967 → 1240 ⇒ measured **−61.9%** against the
> 1.467M post-pairing baseline.
>
> **Consequence: lincomb2 rests on no cryptographic assumption that the
> original chips did not.** The T₀-dlog assumption is neither necessary nor
> sufficient — **do not sign it off.** Keep the blind only for the `len`
> simplification (L6.5 drops), which is convenience, not soundness, and say
> so wherever it is described.

## Status (2026-07-24, uncommitted working tree)

| Phase | State | Gate |
|---|---|---|
| A witness + T₀ | **committed** `bc62f00e` | — |
| A′ `nb` schedule fix | **done** | `-p ecsm` 26/0 |
| B syscall + executor | **done** | `-p executor lincomb2` 18/0 |
| C T₀ table (+ `LEN_M1`) | **done** | `--lib ec_t0` 17/0 |
| D0 oracle anchors | **done** | 611 lincomb + 1,536 small-scalar + 489 Wycheproof + 40/40 + 69,431 rows field-by-field, **0 failures** |
| E0 spec chapter | **done** | `spec/ecsm.typ`, compiles standalone + in `ebook.typ` |
| D chips (+ both soundness fixes) | **done** | ECDAS2 658 cols / 288 constraints; `--lib lincomb2` green |
| E z3 gate | **done, unconditional** | `gate/RESULTS-lincomb2.md`: all lemmas UNSAT, **7 forgeries SAT**, 0 live holes |
| G guest switch | **done** | `ethrex-crypto` 13/0; both real blocks prove **and verify**; **−78,823 / −78,493 guest cycles per ecrecover** (two blocks, agreeing to 0.4%) |
| H bench | **harness ready, NOT run** | `BENCH.md` has the server commands; benches run on the server, never locally |

**Two "silently inert" traps, both now guarded — check these first if a result
looks wrong.** (a) The guest falls back to pure-Rust `lincomb` on non-zero
status and returns **the same answer**, so output equality cannot detect a run
where the precompile never fired. `test_ethrex_block_uses_the_lincomb2_
accelerator` counts ECSM2 rows with `MU=1, OK=1, STATUS=0` and requires exactly
one per transfer; on the bench side only the **cell slope** can tell (a slope
near baseline, or a mean/worst ratio near 1.0, means you measured the
fallback). (b) Five `pr_main.yaml` cache keys hashed `executor/programs/rust/**`
and `syscalls/**` but **not** `crypto/ethrex-crypto/**`, which the ethrex guest
depends on by path — CI would have restored a stale `ethrex.elf` and kept
proving the old x-only path. Fixed (cache-key only).

**Measured cycle win is below DESIGN's prediction**: ~78.5k/ecrecover vs the
predicted 100–150k (~21% short of the low end). Quote the measurement.

Whole-tree gates re-run independently at this point, not taken on report:
**`make lint` clean** (fmt + all four clippy passes) and
**`--lib prove_elfs` 101 passed / 0 failed** — real end-to-end proofs still
prove and verify with the EC_T0 AIR in the machine. Five recursion-ELF
failures elsewhere in the prover suite are pre-existing and unrelated
(`executor/program_artifacts/recursion/*.elf` was never built in this tree;
`make compile-recursion-elfs`).

---

## 0. Invariants that hold for every phase

1. **The old path stays alive until phase G.** `ecsm_mul` (syscall),
   `compute_witness` (single-scalar witness), the ECSM/ECDAS chips as they
   exist today, and the guest's 4-query `lincomb2_with_oracle` all keep
   working and keep their tests green. lincomb2 lands *beside* them.
   Nothing is deleted until the new path is proven end-to-end.
2. **`crypto/ecsm/src/witness.rs::lincomb2_witness` is the spec.** It is
   host-validated against two independent references over 512+ cases. If a
   chip/executor disagrees with it, the chip/executor is wrong — do not
   "fix" the witness to match an implementation.
3. **No constraint-body micro-optimization.** Per the standing rule
   (`feedback-clean-constraints-over-cleverness`), bodies are the spec;
   cleverness needs a measured bench win or it gets reverted. Clones in
   bodies are free.
4. **`make lint` (not per-package clippy) + `cargo fmt` before any push.**
5. **Every phase reports honestly**: if a gate is red, say so with the
   output. A phase is not "done" because it compiles.

---

## 1. Dependency graph

```
A (DONE: witness + T0)
├── B  syscall + executor arm ──────────┐
├── C  T0 preprocessed table chip ──────┤
├── D0 oracle extension (paper/python) ─┤
└── E0 spec chapter + NUMS assumption   │
                                        ▼
                          D  chips (ECSM' / ECDAS' / Addend bus)
                             + trace-builder collectors
                                        │
                                        ▼
                          E  z3 gate port + L6 redo + negative controls
                                        │
                                        ▼
                          G  guest switch (delete solve_y + 4-query)
                                        │
                                        ▼
                          H  bench + EC-share measurement
```

**Parallel now (no interdependency): B, C, D0, E0.**
D is the long pole and needs B's log format + C's table + D0's oracle.

---

## 2. Phase B — syscall + executor arm

**Goal.** A new `ecsm_lincomb2` syscall the guest can call, executed in
software by the executor, emitting a log the trace builder will later
consume. No chip work; the proof side does not yet constrain it.

**ABI (decided; `layout-lock.md:91-107` left this TBD).**
Status is returned **in a register**:

```
a7 = u64::MAX - 11        (next free; continues the negative-ECALL convention)
a0 = addr to write xQ(32) ‖ yQ(32), 64 B LE   →  a0 holds STATUS on return
a1 = addr of xP1 ‖ yP1 (64 B LE)
a2 = addr of xP2 ‖ yP2 (64 B LE)
a3 = addr of u1  ‖ u2  (64 B LE)
```

**This decision flip-flopped once; here is the resolution, so it does not
get relitigated.** An intermediate draft moved the status into memory on the
grounds that ECALL decode leaves `write_register = false`
(`prover/src/tables/types.rs`, `Instruction::EcallEbreak` arm). That flag is
real, **but it governs the CPU row's write path only** — it says nothing
about what an accelerator may emit on the MEMW bus. The counter-example is
already in the tree: **COMMIT writes x10 during its ecall**
(`prover/src/tables/commit.rs:29` "Sender: Memw bus — read+write x10
register", implemented at `trace_builder.rs:1225-1228` via
`MemwOperation::new(true, reg_addr, new_value, ts, 2, true)` +
`register_state.write(10, count, ts)`).

The register form is therefore both expressible and strictly cheaper: no
status word in the buffer, one fewer dword write, and the guest tests a
register instead of doing a load.

**The error path must still perform every operand read and the status
write.** Do not early-return before the loads: skipping them desynchronises
the MEMW timestamps and makes the trace unprovable.

**Status contract (soundness-critical — get this right).**
`status ≠ 0` ⇒ the guest takes its pure-Rust `ProjectivePoint::lincomb`
fallback. This is **always sound**: the fallback is proven guest code, so a
lying status can only waste cycles, never forge a result. Therefore:

- **Do NOT trap** on degenerate input. Today's `ecsm_mul` traps, which is
  acceptable only because its inputs are guest-guarded; a trap here would
  let a crafted transaction censor an entire block.
- Map every `Lincomb2Error` variant (`ScalarIsZero`, `ScalarOutOfRange`,
  `PointNotOnCurve`, `PointNotCanonical`, `SumDegenerate`,
  `ResultInfinity`) to a distinct non-zero status value — distinct so the
  bench/debug path can tell them apart; the guest only tests `!= 0`.
- On `status ≠ 0`, write **only** the status word; leave the 64-byte Q
  region untouched (there is no witness to prove).

**There is no precompile log type — do not invent one.** Verified: no
`EcsmLog`, no event enum, no side-channel `Vec` on the executor. ECALLs
repurpose the ordinary per-instruction `Log` (`executor/src/vm/logs.rs:9-28`):
`src1_val` = a7 (set generically at `execution.rs:471`), `src2_val`/`dst_val`
= two spare address slots. **The prover re-derives the witness itself** —
`trace_builder.rs:851-852` calls `ecsm::compute_witness(...)` and recovers
operand addresses from `register_state.read(10/11/12)` (`:839-841`). lincomb2
follows the same precedent: recover all four addresses from register state in
the trace builder; the `Log` struct does not change.

**Touch list.** `syscalls/src/syscalls.rs` (+1 `pub fn`, riscv64 + non-riscv64
stub pair, mirroring `ecsm_mul` at `:168-188`); the syscall-number enum
(`executor/src/vm/instruction/execution.rs:10-19`, `TryFrom` `:38-51`,
`Accelerator` `:54-58`, `accelerator()` `:60-74` — exhaustive match, so a new
variant is a compile error there by design); the ecall dispatch `:356-359`;
a new arm mirroring `SyscallNumbers::Ecsm` at `:424-456`; and **`bin/cli/src/
main.rs:367` `accelerator_of()` + its tests `:1100-1131`** — a third sync
table the DESIGN touch list missed, and one that is *not* compile-forced, so
it goes stale silently.

Address guards: `ecsm_addr_ok` (`:96-99`) + `load_u256_le`/`store_u256_le`
(`:77-94`). Note the overlap guard's **real** rationale (`:438-444`): xG and k
are read at adjacent proof timestamps, so an overlap makes the **MEMW access
chain unprovable** — it is about trace provability, not arithmetic. Reason
about the four regions' timestamp schedule rather than copying the
`abs_diff < 32` check.

**Build note.** `lambda-vm-syscalls` is **excluded from the workspace**
(`Cargo.toml:11`, bare-metal riscv — `#[global_allocator]`/`no_mangle` won't
link on host), so a plain workspace `cargo test` does not build it.

**Ecall-bus obligation (phase D solves it; phase B must not preclude it).**
The CPU sends on the `Ecall` bus (id 19) for *every* ecall, and each syscall
needs a matching receiver or **the bus unbalances** — the unmatched-`Print`
note at `syscalls.rs:36-40` is the cautionary tale; ECSM's receiver is
`ecsm.rs:307-316`. This must hold on the **error path too**, where there is no
chain to prove. Hence the executor always writes a status word whether or not
it succeeded, so the error path stays expressible as a mu-gated row that
proves the receive and the status write and nothing else. Soundness is
unaffected: `status ≠ 0` only sends the guest to its proven fallback, so a
prover may always *under*-claim; only `status == 0` obliges a full chain proof.

**Acceptance gate.**
- Executor unit tests: one happy path asserting Q equals `lincomb2_witness`'s
  Q (and k256's), plus one test per `Lincomb2Error` variant asserting
  `status != 0` **and** that the output buffer is untouched.
- Guard tests: out-of-bounds, unaligned, and each overlapping-region case
  rejected without panicking the executor.
- A guest→executor round trip through the existing ECSM test harness style.
- `make lint` clean.

**Do not touch.** `ecsm_mul`, `compute_witness`, any file under
`prover/src/tables/`, the guest's `lib.rs`.

---

## 3. Phase C — T₀ preprocessed constant table

**Goal.** A preprocessed (committed-once, verifier-known) table of
`2^len · T₀` for `len ∈ [0, 256]`, which the correction row indexes by `len`
to strip the NUMS blind. Follows the keccak round-constant chip precedent
(`keccak_rc.rs`).

**Content.** Row `i` = `(i, x(2^i·T₀), y(2^i·T₀))` — or the negation
`−2^i·T₀` directly, since the correction row *adds* `−2^len·T₀`. Prefer
storing the negation: it removes a per-row negation from the chip and the
table is constant either way. **Decide once, document it in the chip
header, and make the witness/test assert the same convention** —
`witness.rs` already emits the correction addend, so read what it emits
(`JointSel::Correction`) and match it exactly rather than assuming.

T₀ itself is pinned and reproducible: `ecsm::t0()`, derivation in `T0.md`,
independently reproduced by the `t0_derivation_matches` dev test and the
Python oracle.

**Acceptance gate.** A test that recomputes the whole table from `ecsm::t0()`
by repeated doubling and compares against the committed constants; a test
that the table's commitment is stable across runs; on-curve check for every
entry.

**Do not touch.** Anything outside the new table chip + its registration.

---

## 4. Phase D0 — oracle extension (parallel, cheap)

`thoughts/ec-recover-opt/oracle/lincomb2_ref.py` already exists from phase A.
Extend the oracle set so phase E has anchors ready:

- ≥500 random lincomb differentials vs an independent implementation.
- Wycheproof ECDSA-verify secp256k1 vectors (they exercise the `sR − zG`
  lincomb shape).
- Re-run the existing 40-signature 3-way ecrecover differential
  (`bonus_ecrecover.py`) against the lincomb2 path.
- Small-joint-scalar unrollings (u1, u2 ∈ [1, 16]) — the L7 anchor.

**Acceptance gate.** All differentials green, logged to
`thoughts/ec-recover-opt/lincomb2/` like `oracle_lincomb2.log`.

---

## 5. Phase E0 — spec chapter + the NUMS assumption

No EC spec chapter exists today (`spec/src/ecsm.toml` references are dead —
confirm before writing). Write one, and in it state the **named
computational assumption** the design introduces:

> No efficient prover can produce a known linear relation on `dlog(T₀)`.

This is dlog-class — strictly within what ECDSA/ecrecover already assume for
the chain being proven — and it replaces gate lemma L5b, which does *not*
survive the joint chain (DESIGN §4 gives a constructive attack on the
unblinded version). Precedent for a named assumption in the spec: blake3's
6-round assumption.

**DO NOT RATIFY THIS AS WRITTEN — see the banner at the top of this
document.** The assumption is **necessary but not sufficient**: the reduction
*to* it does not close, because the prover chooses P2 and can set it to a
known multiple of T₀. Signing it off would buy strictly less than DESIGN
attributes to it, and would leave the chip accepting proofs of false
statements for one scalar multiplication of work.

The spec chapter should therefore state **both** the assumption **and** that
the reduction to it is incomplete, plus whichever non-degeneracy mechanism is
actually adopted (working assumption: the unconditional witnessed inverse, in
which case the assumption is not load-bearing for soundness at all and the
blind survives only as a convenience that drops the exact-MSB lemma L6.5).

---

## 6. Phase D — chips (the long pole)

**Goal.** ECSM′ (1 row/ecrecover) + ECDAS′ (1 row/joint step) + the Addend
bus, proving exactly what `lincomb2_witness` emits. This is the largest chip
job since the tables were written.

**Design B (locked; Design A rejected, DESIGN §2.1).** The addend is
*received* per-add on a new **Addend bus** rather than carried in every row.
ECSM′ publishes `{1: P1, 2: P2, 3: P12}` with witnessed count
multiplicities; ECDAS′ receives keyed `[ts, sel]` with
`sel = s1 + 2·s2 + 3·s3`, multiplicity `Sum3(s1,s2,s3)` (supported —
`lookup.rs:1336-1349`). Balance forces correctness; counts need no range
check under gate contract C5. XB/YB inherit byte-ness from the publisher's
checked columns via tuple equality (the C4 inheritance the gate already used
for YR) — **no new byte checks for the addend**.

**Reuse verbatim.** The three convolution relations (λ, xR, yR — Q0/Q1/Q2 +
C0/C1/C2) are byte-for-byte today's ECDAS machinery; `carries_lambda/xr/yr`
are reused unchanged by the witness already. Gate lemmas L1/L2/L3/L4 port
as-is.

**The four things that are genuinely new** (and where the bugs will be):

1. **Addend-bus binding** — the pending digit bits ride the accumulator
   tuple exactly as `op` does today.
2. **Double rows carry addend `(0,0)`.** Verified algebraic fact: on a
   double the addend cancels out of all three relations (λ's op=0 branch
   uses neither coordinate; xR's `−xg` cancels against `(1−op)(xg−xa)`; yR
   uses neither). So the Addend receive is gated by `Sum3(s1,s2,s3)` and
   stays silent on doubles. **Re-derive this before relying on it.**
3. **Telescoping breaks in exactly two places** (`layout-lock.md:118-123`):
   the **precompute row's `a` is P1**, not the previous row's result (it is
   a standalone chord add `P1 + P2`, *off* the accumulator line), and the
   **correction row's `a`** is the last accumulator. Everywhere else
   `a = prev.r` holds. This is the single most likely source of a silent
   soundness hole — treat it as such.
4. **The three NEW load-bearing canonicalization checks**: `yP2 < p`,
   `xQ < p`, `yQ < p`. Without `yP2 < p` a prover submits `yP2 + p` (fits in
   32 bytes) — same point mod p, **opposite parity as bytes**, flipping the
   sign of Q. Without `xQ/yQ < p` the guest keccaks a shifted coordinate to
   a different address. Same class as the gate's N6 finding. These are not
   optional hygiene; each one is a forgery if dropped, and phase E has a
   negative control for each.

**Schedule the chip must accept** (`layout-lock.md:25-39`): dense-doubling
blinded Shamir/Straus, MSB-first, acc seeded at T₀ — one double every round
*always* (no skip-ahead, so **no `nb_or` column**; DESIGN's sketch was wrong
here), the add sharing the double's round, then the correction row.

**Row budget to hit** (measured, not estimated): mean 449.1 rows per
ecrecover. **For capacity use 514, not `layout-lock.md`'s 471** — 471 is only
the max over *random* scalars. The worst case over the valid input domain is
`(u1, u2) = (2^255, 2^255 − 1)`: both in `[1, N)`, complementary bit patterns
⇒ every one of the 256 rounds has an add ⇒ `1 + 256 + 256 + 1 = 514`. Cheap
for a submitter to construct deliberately, and verified independently. (The
all-ones case `(N−1, N−1)` gives only 449 — complementarity maximises, not
popcount.) Mean 449.1 still governs the cost model; 514 governs padding
bounds, per-ecall row allowances and any rows-per-call assertion. Worth an
explicit test that a 514-row ecrecover proves and verifies. Column targets are now **ECSM′ 1,090** and **ECDAS′ 529** per
the layout pass — not DESIGN's 1,310/525. The ECSM′ delta is P1 (bound to G,
zero columns) plus a DESIGN double-count of `xP2` canonicalization the
witness never emits; ECDAS′ gains `NB`, `S_CORR`, `PHASE`/`NEXT_PHASE` and
loses `NEXT_OP`.

### 6.1 Findings from the layout pass — read `CHIP-LAYOUT.md` before coding

The full column maps, bus specs and derivations are in
`thoughts/ec-recover-opt/lincomb2/CHIP-LAYOUT.md`. The load-bearing
corrections, several of which contradict DESIGN or `layout-lock.md`:

1. **`nb` IS required — and it is now IMPLEMENTED in the witness.**
   `layout-lock.md` correction #2 claimed the opposite and has been fixed in
   place. Dense doubling is true, but the double and its add *share a
   round*, so a double row's successor round depends on whether an add
   follows — not a function of its own columns. Today's ECDAS solves this
   with `NEXT_OP` + `round − 1 + next_op` (`ecdas.rs:243-253`). Without the
   analog, `round` can stall and a prover can insert or drop doublings.

   **The defining constraint is op-gated — an ungated version is wrong** (it
   fails on add rows, which carry their round's real digits to select the
   addend but have `nb = 0`, because the row after an add is always the next
   round's double):
   ```
   OP · NB = 0                                (degree 2)
   (1 − OP) · (NB − D1 − D2 + D1·D2) = 0      (degree 3)
   ```
   Degree 3 is within budget (`EcdasConstraints::max_degree()` is already 3
   for the λ relation). Both forms are asserted on every emitted row by
   `check_nb_schedule`, so phase D can lift them straight out of the test.

   **Semantics, stated precisely** (an off-by-one here breaks the
   recurrence): `nb` is the OR of **this row's own round's** digits —
   equivalently "the next emitted row is an add at my round" — *not* the next
   round's digits. This is the same predicate as today's `EcdasStep::next_op`
   ("1 ⇒ next row adds at this round"), so **phase D can reuse the existing
   `NEXT_OP` column and the `round − 1 + next_op` outgoing expression
   unchanged**; the witness asserts the two never diverge. The chain still
   drains at the sentinel round `−1` for free, since a round-0 iteration's
   last row always has `nb = 0`.

   Verified additive: `s.next_op` occurred exactly once in the pre-change
   file (a plain struct-field copy at `build_step`'s tail, feeding no
   numerator, quotient or carry array), and the row statistics are
   bit-identical to phase A's lock (mean 449.1 / max 471 over 512 cases).
   `cargo test --release -p ecsm --lib` → **26 passed, 0 failed**
   (21 phase-A + 4 phase-C + 1 new), independently re-run.
2. **`OP = S1 + S2 + S3 + S_CORR` is missing from every doc.** The
   double-row addend cancellation is real (re-derived from the live `eval`
   body: λ's `xg`/`yg` sit inside the `op·(…)` product; xR's `−xg(i)`
   cancels the `+xg(i)` from `−(1−op)(xa−xg)`; yR contains neither), **but
   without this gating constraint it is forgeable** — a prover sets `S2 = 1`
   on a double row and mints a spurious Addend receive.
3. **The error row needs `OK·STATUS = 0` and `STATUS·S_INV = 1 − OK`.**
   Split `MU` (real ecall ⇒ Ecall receive + all MEMW) from `OK` (chain
   proven ⇒ every relation and chain bus), with `OK·(1−MU) = 0`. "A lying
   status only wastes cycles" holds for `status ≠ 0`, but **`status == 0`
   must *oblige* the proof** — otherwise a prover sets `OK = 0`, writes
   `status = 0`, and the guest reads a fabricated Q. Two constraints, one
   column, distinct error codes preserved.
4. **`round` cannot discriminate the telescoping breaks.** Precompute and
   correction are both emitted with `round = 0` (`witness.rs:810`, `:900`)
   and the main loop also produces genuine round-0 rows. Proposed fix: a
   `PHASE` element in the Ecdas′ tuple splitting the chain into three
   separately-keyed segments (0 = precompute, 1 = main, 2 = correction),
   each pinned at both ends by ECSM′ with multiplicity `OK`. Bonus: this
   drops `genX`/`genY` (64 elements) from the tuple.
5. **JointBit needs its own bus id (33); Addend keeps 29.** (32 was the
   original proposal but phase C claimed it for `EcT0`; `Bit = 30`,
   `GlobalMemory = 31`.) Bus elements
   that are zero on a row are **skipped** (`crypto/stark/src/lookup.rs:676-679`,
   a deliberate optimization) and positions are α-weighted, so trailing-zero
   padding is invisible: `[a,b,c]` and `[a,b,c,0]` have identical
   fingerprints, and arity never separates chips. Since the old chips live
   alongside the new ones until phase G, a `Bit[ts, round, stream=0]` send
   would alias an old-ECDAS send exactly. Use a stream tag ∈ {1,2}, never 0.
6. **Addend multiplicity is `Multiplicity::Linear`, not `Sum3`.** `Sum3`
   covers the three scalar addends but leaves the correction row unable to
   receive. Still one interaction. Tuple `[ts, sel, x(32), y(32)]`,
   `sel ∈ {1,2,3,4}` — never 0, per finding 5. Precompute reuses `sel = 2`
   (its addend genuinely is P2).
7. **`len ≤ 256` — fix APPROVED, NOT YET LANDED.** As of this writing
   `ec_t0.rs` still has `NUM_REAL_ROWS = 257` / `NUM_ROWS = 512` and still
   carries the consumer-obligation text at `:30-35`. **Verify the code
   before relying on the resolution below.** The hazard is real: the table
   pads rows
   257..511 with `x = y = 0`, so an unconstrained `len` would resolve to the
   off-curve `(0,0)` and the correction row would add it. The fix landed on
   the *table* side, not the consumer side (my original note put it in the
   wrong phase): the table now stores `LEN_M1 ∈ [0,255]` in 256 real rows
   with **no padding**, and keys the receive with `LEN_M1 + 1` via
   `LinearTerm::Constant(1)`. The byte bound *is* the range check, so phase D
   inherits no obligation and must **not** re-add a redundant consumer check.
   (Safe because `len = max(msb u1, msb u2) + 1` with both scalars non-zero
   ⇒ `len ∈ [1, 256]`, so `len = 0` is unreachable and no row is wasted.)
8. **DESIGN §3's justification for `yP2 < p` is wrong** — see §11 risk 7.

**Touch list** (verified refs). `prover/src/tables/ecsm.rs` (911 lines, 667
cols, `cols` `:34-110`, `bus_interactions()` `:300`, `EcsmConstraints` `:704`,
`eval` `:839`) and `ecdas.rs` (463 lines, 521 cols, `cols` `:32-64`,
`bus_interactions()` `:152`, `eval` `:427`) — **prefer new sibling modules**
so the old chips stay byte-identical and their tests stay green.

- **Registration: there is no table enum.** `VmAirs` is a plain struct, one
  field per table — `prover/src/lib.rs`: struct `:487-517`, `air_trace_pairs()`
  `:521-592` (prover), `air_refs()` `:595+` (verifier), construction
  `:760-761`, struct literal `:866-867`, imports `:55-56`.
- **AIR constructors live in `prover/src/test_utils.rs`**, despite the name —
  `create_ecsm_air` `:928-937`, `create_ecdas_air` `:940-949` are the
  production ones `lib.rs` calls.
- **Addend bus**: `BusId` in `prover/src/tables/types.rs` — **id 29 is free**
  (`Ecdas = 28`, `Bit = 30`, `GlobalMemory = 31`). Three sync points: enum
  `:255`, `name()` `:365-393`, `TryFrom<u64>` `:396-425`.
- **Collectors**: `trace_builder.rs` — clone `collect_ecsm_ops` `:821-949`
  (MEMW schedule contract in its doc comment `:821-827`); hook in
  `collect_ops_from_cpu` `:536-713` at `:648-655`; classification bit
  `cpu.rs:189-190`, `:236-237`, `:355`. `collect_ops_from_cpu` returns a
  **10-tuple** destructured at `:2764`, `:2818`, `:2960`, `:3003`, `:4264`,
  `:4282`, `:4341`, `:4355` — all need the extra element. Trace plumbing:
  `Traces` fields `:2718-2722`, gen closures + rayon `:3363-3365`, `:3417`,
  `:3444`, `:3478`, `:3545`, cell accounting `:3805-3806`/`:3912-3913` (main),
  `:3953-3954`/`:4042-4043` (aux).
- `collect_bitwise_from_ecsm` `:2285` / `_ecdas` `:2325` (wired at
  `:3078-3079`) must **mirror `bus_interactions()` exactly** — sends and
  multiplicities, including the paired `AreBytes` from `42ba68ff`. The
  comments at `:2290`/`:2329` say so; a mismatch here is a silent bus break.
- ~~Pinned per-table column/interaction counts in `trace_builder_tests.rs`~~
  — **wrong**: those assertions live inside a keccak-specific test module,
  not a general registry, and ECSM/ECDAS have no entries there at all. The
  EC-table precedent is to pin layout and bus shape in the table's *own*
  test file (see `ec_t0_tests::layout_is_as_documented` /
  `bus_interaction_shape`). Follow that.

**T₀ sign gotcha for the correction row.** The table stores the **negation**
`−2^i·T₀`, matching what `lincomb2_witness` passes as the correction row's
addend (`witness.rs:888` `neg_tpow`, used at `:896-898`; `build_step` writes
the addend to `x_g`/`y_g` at `:364-365`). So the lookup wires straight into
the addend columns with **no in-circuit modular negation**. But the witness
*also* records `x_t0_pow`/`y_t0_pow` (`:936-937`), which hold the
**positive** `2^len·T₀` — the opposite convention. `x` is shared
(`x(−P) = x(P)`); `y` is negated. If phase D binds the addend from the
table, `y_t0_pow` is redundant. Do not mix the two.

**Acceptance gate.**
- Trace-level: for ≥100 random ecrecover inputs, the generated trace
  satisfies every constraint (the debug trace validator), and the chip's Q
  equals `lincomb2_witness`'s Q.
- Bus balance: all buses balance, including the new Addend bus
  (`DEBUG_BUS_TRACKER=1`).
- A real end-to-end proof of a guest calling the new syscall **verifies**.
- The old ECSM/ECDAS tests still pass untouched.

---

## 7. Phase E — z3 soundness gate

Port the existing gate (`thoughts/ec-recover-opt/gate/`, L1–L8 all green on
the current chips) to the new layout. Structure and harness
(`gate_common.py`, `positive_real_witness.py`) carry over.

- **Ports unchanged**: L1, L2 (recompute offsets), L3a/L3b, L4, L5a.
- **Replaced**: ~~L5b → the NUMS reduction argument~~ — **this cannot be
  discharged as specified** (see the banner). L5b is instead replaced by the
  **non-degeneracy check** on add rows, which is unconditional and needs no
  assumption. Prove: the check is imposed on exactly the rows that consume an
  addend (all three scalar addends *and* the correction row), never on
  doubles, and that it is not satisfiable when `xA = xB`.
- **New negative control, derived from a working attack**: ablate the
  non-degeneracy check and feed the gate the construction from
  `oracle/nums_blinding_probe.py` — **it must go SAT**. This is the strongest
  control in the suite because the forgery is real, not hypothesised.
  Structure the check so it can be cleanly ablated for this test.
- **Redone (riskiest work in the whole project)**: **L6** — the joint
  schedule: two Bit streams, Addend-bus binding via tuple-carried digit
  bits, per-scalar counting. Two interleaved counting arguments. The
  exact-MSB sub-lemma (old L6.5) **drops** — blinding makes any
  `len ∈ [max_msb+1, 256]` yield the correct Q, since extra leading doubles
  just double T₀ and the keyed correction absorbs them.
- **Redone**: L7 — small-joint-scalar unrollings vs the extended oracle.
- **New negative controls (each must SAT, i.e. forgery reappears when the
  check is removed)**: drop `yP2_SUB_P` (the §3 parity-flip attack); drop
  `xQ`/`yQ` canonicalization; addend-selector tamper; status-contract
  mismatch; correction-table wrong-`len`.
- **Anchor**: re-run the real-witness positive evaluation (the 872k-check
  anchor) against a real lincomb2 trace.

**Acceptance gate.** All lemmas UNSAT (sound), all negative controls SAT
(the gate can actually see forgeries), results written to
`gate/RESULTS.md` in the existing format.

---

## 8. Phase G — guest switch

Now, and only now, the deletion. `crypto/ethrex-crypto/src/lib.rs`:

- **The call site does not change.** `lib.rs:116-117` is already
  `ecsm_lincomb2(...).unwrap_or_else(|| ProjectivePoint::lincomb(...))` —
  exactly the shape the status contract needs. Only the *body* of
  `ecsm_lincomb2` changes: four oracle queries + `solve_y` → one syscall,
  `status != 0` ⇒ `None` ⇒ existing fallback.
- **Delete**: `lincomb2_with_oracle` (~lib.rs:195-250), `solve_y`
  (~:252-272), and `scalar_near_edge` if it has no other caller — check;
  keep `affine_xy`/`point_from_xy` if still used.
- Keep the guest-side `(r, v)` decompression exactly as is: **the guest is
  and remains the parity authority** (proven CPU execution). The chip's new
  obligations are membership + canonicalization only.

**Acceptance gate.** ethrex-crypto tests green; the 40-signature differential
green; a real block proves and verifies; guest cycle count measured (expect
~100–150k fewer cycles/ecrecover from the removed round-trips and inversion).

---

## 9. Phase H — bench

Extend `thoughts/ec-recover-opt/gen_ec_bench.sh` with an ecrecover-loop
guest. Measure (a) EC table rows/cells before vs after, (b) the EC share of
total prover cells on a real block — the multiplier that converts −74.3% EC
cells into a whole-prover number.

**The always-on cost of EC_T0 — RESOLVED, and it is negligible per proof.**
The concern was that every proof carries a 256×66 table even with no EC ops.
It does not: `field_elements_by_table` counts EC_T0 as
`EC_T0_COLS - EC_T0_PRECOMPUTED` = 66 − 65 = **1 committed column** (the
multiplicity) × 256 rows — the same treatment `KECCAK_RC` gets
(`trace_builder.rs:4364` and `:4315`). The other 65 columns are constants
fixed by the **preprocessed commitment**, committed once rather than per
proof. That is why a per-table breakdown reports ~1k cells for EC_T0 rather
than ~17k, and it is correct accounting, not a measurement bug.

**What is NOT resolved by that**: the extra AIR still costs the recursion
guest per-AIR opening work, which is a different axis from committed cells
and remains worth watching. `include_halt` is the precedent for a
conditionally-included AIR if it ever matters.

**Benches run on the bench server, not locally** — hand the user the
command; do not run or schedule them here.

**Target to confirm: −61.9% EC cells at the mean** (0.559M vs the 1.467M
post-`42ba68ff` baseline, at 449.1 rows), and **−56.4% at the 514-row worst
case** (0.640M). Report both; do not quote only the flattering one. If the
measured number lands materially below either, say so plainly rather than
reframing the target.

**Three baselines are in circulation and two of them are wrong. Quote
neither:**

| figure | why it is wrong |
|---|---|
| DESIGN's **−74.3%** | denominated against the **pre-pairing** 1.69M baseline, but the AreBytes pairing already shipped (`42ba68ff`) — it re-banks a banked win |
| an earlier draft of this section's **−70.2% / 0.437M** | post-pairing but **pre-`D_INV`** — written before the non-degeneracy relation that closes the degenerate-add forgery, which costs ~129 cols and ~96 interactions/row |
| **−61.9% / −56.4%** | ✅ current: measured against the live baseline with `D_INV` in |

The headline moved from ~−70% to −61.9% because that is what unconditional
soundness cost. It is the right trade and should be stated as one: the design
originally bought its extra ~8 points by resting on a T₀-dlog assumption that
turned out **not to hold** (see the banner). Also note DESIGN's "−70.4%
standalone" is *unpaired vs unpaired* while the old −70.2% was *paired vs
paired* — two different quantities that happened to land 0.2 points apart;
never treat that coincidence as confirmation.

Verdict unchanged: better than **2.2×** the 2× bar, with no computational
assumption.

---

## 10. Open decisions

| # | Decision | Status |
|---|---|---|
| 1 | Status in register vs memory | **Decided: register** (§2) — the `write_register = false` objection was mistaken; COMMIT sets the precedent |
| 1b | P1 general vs bound to G | **Decided: bound to G** (§6) — the witness has no `mem_p1`, so a general P1 is not provable today. Zero-cost via constant MEMW reads; ABI stays general |
| 2 | NUMS assumption named in spec | **Needs user sign-off** (§5) |
| 3 | T₀ table stores `+2^i·T₀` or `−2^i·T₀` | Phase C decides, must match witness |
| 4 | New chip modules vs in-place rework | Recommend new modules (§6) |
| 5 | GLV (4-way, −41% further) | **Deferred** until lincomb2 benches (DESIGN §7a) |

## 11. Known risks

1. **L6 redo** — two interleaved counting arguments; the riskiest proof work.
2. **Telescoping special cases** (precompute off-line, correction row) — the
   likeliest silent soundness hole.
3. **Preprocessed-table plumbing** for T₀ assumed cheap on the keccak_rc
   precedent; unverified until phase C.
4. **Row-shape estimate** was ±5%; phase A collapsed the band (449.1
   measured vs 448 estimated), so the cost verdict now rests on measurement.
5. **Chip constraint tests hardcode constraint indices.**
   `prover/src/tests/ecsm_tests.rs:16-21` pins `IDX_KBITS_ZERO = 257`,
   `IDX_X2_CONV0 = 258`, … into the single `eval` body. Any edit to a chip
   reshuffles them. Expect churn there, and do not "fix" a failing index by
   renumbering until you know which constraint actually moved.
6. **Ecall-bus balance on the error path** (§2) — a syscall with no matching
   receiver unbalances bus 19. Phase D must handle the `status ≠ 0` case,
   which has no chain to prove. See §6.1 finding 3 for the two constraints
   that make it *sound* as well as balanced.
7. **`yP2 < p` may not be load-bearing at all, and DESIGN's reason for it is
   wrong.** DESIGN §3 argues `yP2 + p` is "opposite parity as bytes" and
   flips P2's sign. But negation is `(x, p − y)`, not `(x, y + p)`; and
   `y + p ≡ y (mod p)` is the *same* point, so Q is unchanged. Decisively,
   **both `y` and `p − y` are already `< p`**, so a `< p` test cannot
   separate them — it cannot be the parity defence. What actually binds the
   sign is MEMW plus the guest being parity authority
   (`crypto/ethrex-crypto/src/lib.rs:98-104`). Keep the check (the witness
   emits it, ~40 cells, good width-audit hygiene), but **expect phase E's
   negative control for it to return UNSAT, not SAT** — and do not "fix" the
   gate when it does. `xQ < p` and `yQ < p` remain genuinely load-bearing:
   the output bytes get keccak'd, which is exactly the XR_SUB_P/N6 argument.
8. **~~The width audit has not been re-run for a *varying* addend.~~ —
   DONE, and the widths hold** (`WIDTH-AUDIT.md`). Worst case 2^24.4 against
   2^64, i.e. **~2^39 of headroom**; z3 UNSAT ×36 on both steps with the
   addend limbs free bytes, plus 36,000 corner samples and 6,346 real rows
   cross-checked against the prover's own quotients/carries (38,076
   comparisons, 0 mismatches). **No new constraint, and `P12` does NOT need
   canonicalization** (priced anyway at ~80 cells; recommendation: skip —
   the only freedom a missing `< p` leaves is the encoding `v + p`, which
   denotes the same field element and never leaves the chip; contrast
   `xQ/yQ`, which do leave and get hashed).

   **Correction to an earlier note in this plan:** `chips-map.md:93-100` was
   cited as having derived the bounds assuming a *canonical* addend. It does
   not — item 4 there is titled "Non-canonical reps mid-chain" and already
   concludes the relations are mod-p with the quotient absorbing. That
   reasoning generalizes unchanged; the genuine gap was coverage of an
   addend that *varies* or is chip-produced, which the audit adds. Don't
   "fix" a note that is already correct.

   `D_INV` was audited too: smallest width of the four relations, **needs no
   new `CARRY_OFFSET_*` constant** (honest carries `[-581, 6041]` fit inside
   `CARRY_OFFSET_XR`'s window), quotient 258 bits with `r = 3p` unchanged,
   degree 3 when `op`-gated — the ≤3 budget is untouched.

   *Limitation stated honestly by the audit*: the completeness table is a
   measurement plus a structural-invariance argument, not a closed-form
   worst case. Completeness-only, so a miss costs an unprovable honest
   witness, never a forgery.
9. **Byte-ness inheritance depends on the Addend/Ecdas bus staying
   one-element-per-byte.** `point_coord_busvalues` (`ecsm.rs:272`) emits 32
   separate `Packing::Direct` elements, which is what makes tuple equality
   per-limb and lets a receiver inherit byte-ness without re-checking. Repack
   it (e.g. `Word4L`, to shrink the bus) and a receiver can satisfy the same
   packed value with a different decomposition — its limbs carry no range
   check, and a single ~2^29 non-byte limb breaks the integer identity. The
   margin against *malformed* limbs is small even though the margin against
   *byte* limbs is astronomical. Now documented at the helper. Two agents
   independently hit this hazard (phase C rejected `Word4L` for the EC_T0
   table on the same grounds), which is why it is recorded as a standing
   invariant rather than a one-off note.

## 12. Corrections already folded in

Recorded so they are not rediscovered: `execution.rs:424-456` is **not** stale
— the real path is `executor/src/vm/instruction/execution.rs`, the DESIGN doc
just omitted the directory. DESIGN's `lookup.rs:1336-1349` is off by ~2 lines
(`crypto/stark/src/lookup.rs:1328-1360+`; `Sum3` `:1350`, `Linear` `:1356`) —
the claim holds, only the numbers were wrong. DESIGN's "`spec/src/ecsm.toml`
references confirmed dead" is **correct** (`git ls-files spec` finds no EC
source; `spec/book.typ:51` has keccak, no EC chapter), but two in-repo comments
still point at it — `prover/src/tables/ecsm.rs:9` and
`trace_builder.rs:907` — to be fixed when the phase-E0 chapter lands.
