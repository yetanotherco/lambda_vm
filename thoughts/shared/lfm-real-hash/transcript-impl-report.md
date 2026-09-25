# The compress-chain transcript (option B1) — implementation report

**Status:** GREEN with one named deviation (§7); post-review fixes D1-D4 applied (§9). **Date:** 2026-08-11.
**Ground:** worktree `lambda_vm-blake3-impl`, branch `blake3-real-hash`, parent
`2957c3f9`, uncommitted. **Spec:** `transcript-spec/TRANSCRIPT.md`, treated as
binding; every departure is named in §7 rather than silently taken.

Claims are ✓ EXECUTED (ran it, output quoted), ✓ VERIFIED (read the code, cited)
or ✗ OPEN.

---

## 0. Board

| item | result |
|---|---|
| spec KATs — per-op, 6 and 7 rounds | ✓ EXECUTED, PASS |
| spec KAT — end-to-end `FriToyV0`-preamble transcript, 6 and 7 rounds | ✓ EXECUTED, PASS |
| the crate anchor: step == `blake3::hash(state‖operand‖"LFMT")[..16]` @7r | ✓ EXECUTED, PASS |
| M1–M7 pre-committed controls | ✓ EXECUTED, all 7 fire as predicted |
| `TrivialV0` proves + verifies under BLAKE3 | ✓ EXECUTED, PASS |
| `FriToyV0` proves + verifies under BLAKE3 | ✗ **BLOCKED — by O1, not by the sponge.** §7 |
| transcript program proves + verifies under Test / Poseidon / BLAKE3 | ✓ EXECUTED, 3/3 |
| cost claims 16,527 / 369,103 @7r | ✓ EXECUTED, both exact |
| full `lfm::` suite | 291 pass / 19 fail — the 19 are the pre-existing `fibonacci.elf` set, identical at 6r and 7r |
| `make fmt` + `make lint` (4 feature combos) | ✓ EXECUTED, clean |
| `cargo clippy --features blake3-6round` | ✓ EXECUTED, clean |
| keccak wrap path | untouched — the whole diff is inside `prover/src/lfm/` |

---

## 1. What was built

The Fiat–Shamir sponge is now a **compress chain over one cell**, for every
hasher, and no permute socket exists or ever will.

```
absorb(c)        state ← T(state, c)                       1 step
absorb2(c0, c1)  state ← T(T(state, c0), c1)               2 steps
squeeze()        out = state ; state ← T(state, SQ(i))     1 step
```

`T` is an ordinary `LFM_HASH` two-to-one row in the **transcript domain**;
`SQ(i) = [SQUEEZE_MARK, i, 0, 0]` with `SQUEEZE_MARK = "SQZ0"` LE. Squeeze
outputs before advancing, mirroring the construction it replaced so the diff
stays reviewable.

The domain is carried by `m[8] = MODE_C·"LFMC" + MODE_T·"LFMT"` — a linear form
over two **preprocessed** columns. It costs no witness columns, no range checks,
and no degree (`m[8]` went from degree 0 to degree 1 inside an `add3` operand
whose body is degree 1 either way; the arm's max degree is still 3).

### File:line map

| what | where |
|---|---|
| `MODE_T` column, `NUM_SELECTORS = 3`, `PREP_WIDTH` 11→12 | `prover/src/lfm/layout.rs:81-113` |
| `HashMode::Transcript` + `is_two_to_one()` | `prover/src/lfm/instr.rs:47-80` |
| `LfmBuilder::transcript_step` / `two_to_one` | `prover/src/lfm/builder.rs:290-316` |
| `SpongeVar` — the chain, `SQUEEZE_MARK` | `prover/src/lfm/edsl.rs:14-131` |
| `HostSponge` — the host mirror, hasher-parameterised | `prover/src/lfm/fixture.rs:46-131` |
| `LfmHasher::transcript` / `transcript_out` (+ `HasherKind` dispatch) | `prover/src/lfm/hash.rs:64-95`, `:238-256` |
| `TAG_LFMT`, `socket_digest_rounds_tagged`, `transcript_digest`, `tag_for_mode` | `prover/src/lfm/blake3_socket.rs:194-300` |
| BLAKE3 `transcript`/`transcript_out`/`step` | `prover/src/lfm/blake3_socket.rs:370-420` |
| `MU_COLUMNS = (MODE_C, MODE_T)` | `prover/src/lfm/blake3_socket.rs:445-455` |
| `TAG_SELECTOR` + `message_word_ref` | `prover/src/lfm/blake3_socket.rs:505-537` |
| the idx 0–5 / MU changes | `prover/src/lfm/blake3_socket.rs:915-975` |
| `WordRef::ModeSelected` + `word_expr` arm + `rotr_bytes` | `prover/src/lfm/blake3_chip.rs:365-407`, `:1017-1042` |
| Test + Poseidon mode-sum widening | `prover/src/lfm/chips.rs:718-765`, `:800-812` |
| `Sum3`→`selector_sum` on the `LfmMem` receives | `prover/src/lfm/chips.rs:614-632` |
| compiler one-hot emission by name | `prover/src/lfm/compiler.rs:326-352` |
| validator one-hot over 3 selectors | `prover/src/lfm/validator.rs:290-311` |
| executor two-to-one arm | `prover/src/lfm/executor.rs:370-425` |
| trace filler: per-row domain tag | `prover/src/lfm/trace.rs:182-203`, `:236-256` |
| `TrivialV0`'s third compress | `prover/src/lfm/programs.rs:62` |
| `permute_coverage_program_source` (`#[cfg(test)]`, unregistered) | `prover/src/lfm/programs.rs:74-108` |
| KAT vectors (generated from the spec JSON) | `prover/src/lfm/transcript_kats.rs` |
| transcript tests (17) | `prover/src/lfm/transcript_tests.rs` |
| M1–M7 + the two F3.4 milestone tests | `prover/src/lfm/blake3_socket_tests.rs:1068-1290`, `:1490-1600` |

---

## 2. KAT results — ✓ EXECUTED

Every vector in `transcript_kats.json` reproduces. The Rust table
(`transcript_kats.rs`) is **rendered from the spec's JSON**, not hand-copied, so
it cannot drift from the oracle's reference.

| | check | evidence |
|---|---|---|
| K1 | 6 per-op step vectors, at 6 **and** 7 rounds | `every_step_vector_reproduces_at_both_round_counts` |
| K1′ | the compiled-in entry point matches its own round count's vector | `the_compiled_step_matches_its_round_counts_vectors` |
| K1″ | **the crate anchor**: `blake3::hash(state ‖ operand ‖ "LFMT")[..16]` @7r, message re-derived byte-level | `seven_rounds_is_blake3_of_the_transcript_message` |
| K2 | end-to-end `FriToyV0`-preamble transcript, state after all 10 recorded ops + 3 ext challenges + 4 query-bit vectors, at 6 **and** 7 rounds | `the_end_to_end_vector_reproduces_at_{six,seven}_rounds` |
| K2′ | the same, through `HostSponge` (the mirror property, made checkable) | `the_host_sponge_reproduces_the_end_to_end_vector` |
| K2″ | the same, through the **machine** (`SpongeVar` → `LFM_HASH`, executed under BLAKE3) | `the_machine_reproduces_the_end_to_end_vector` |
| K3 | transcript step ≠ Merkle parent on the same cells, both round counts | `a_transcript_step_is_not_a_merkle_parent` (+ honest control: they differ **only** in the tag) |
| K4 | the squeeze counter is load-bearing | `the_squeeze_counter_is_load_bearing` (+ honest control: squeeze 0 agrees either way, so the test is not passing on noise) |
| K5 | absorb order is load-bearing | `absorb_order_is_load_bearing` (+ honest control: same order ⇒ same state) |
| K6 | the preamble costs 11 compressions, all `Transcript` rows | `the_preamble_costs_eleven_transcript_steps` |

`transcript_tests`: **17 passed, 0 failed** at 7 rounds; **17 passed, 0 failed**
under `--features blake3-6round`.

The `blake3-6round` build is not merely lint-clean: the per-op and end-to-end
vectors are pinned at *both* round counts from a single build (the reference
takes `rounds` as an argument), and the compiled-in path is separately checked
against whichever vector its knob selects.

---

## 3. M1–M7 conformance checklist — ✓ EXECUTED, all 7

Stated in the spec §5.3 before the chip existed, so these are inherited
obligations. Every one is paired with an honest-path assertion.

| | spec statement | expected | implemented as | result |
|---|---|---|---|---|
| **M1** | `m[8]` pinned to `TAG_LFMC` while `MODE_T = 1` | SAT (a transcript row computing the Merkle tag) | a `MODE_T` row whose entire witness is the `"LFMC"` computation | **rejected** ✓; honest control (same row, own domain) accepted |
| **M2** | mirror: `TAG_LFMT` while `MODE_C = 1` | SAT | a `MODE_C` row whose witness is the `"LFMT"` computation | **rejected** ✓; honest control accepted |
| **M3** | `MODE_C = MODE_T = 1` on one row | UNSAT via idx 4 | set both, evaluate | **violates exactly idx 4** ✓; clearing it restores acceptance |
| **M4** | `MODE_C = MODE_T = 0` with `MU = 1` | UNSAT — `MU` *is* their sum | `MU_COLUMNS == (MODE_C, MODE_T)` asserted structurally; a garbage row with no mode set is padding | vacuous with no mode ✓; **fails** the moment a mode is restored (so the vacuity is not the set accepting anything) |
| **M5** | drop the mode-sum booleanity ⇒ modes arbitrary ⇒ `m[8]` prover-chosen | **SAT** | see below | **SAT, and it fires** ⚠ |
| **M6** | `MODE_T` as a MAIN column ⇒ prover-chosen | **SAT** | same row as M5 | **SAT, and it fires** ⚠ |
| **M7** | generalised capacity form idx 0–3 | UNSAT present / SAT dropped | tamper each `S8+k` on a **transcript** row | each violates **exactly** constraint `k` ✓; transcript capacity == compress capacity == IV |

### ⚠ M5/M6 fired, and the finding is sharper than the spec anticipated

Test: `blake3_socket_tests::m5_m6_the_mode_columns_must_be_preprocessed_or_the_tag_is_prover_chosen`.

Constraint idx 4 pins the mode **sum** to a bit — it does **not** pin each
selector to a bit. So a row with `MODE_C = x`, `MODE_T = 1 − x` satisfies it for
*every* field element `x`, and `m[8] = x·"LFMC" + (1−x)·"LFMT"`. Solving for `x`
makes that **any 32-bit value the prover likes**. The test picks the tag
`"XXXX"`, derives the `x` that produces it, builds the full honest witness under
that forged domain, and the constraint set **accepts the row with zero
violations**.

This is not a defect introduced here — it is exactly what M5/M6 were
pre-committed to demonstrate, and it is the same shape as the pre-existing
`MU = MODE_C` argument. Two mechanisms close it, and the test asserts both
rather than asserting them in prose:

1. **The mode columns are preprocessed** — `MODE_C`, `MODE_T` and `MODE_P` are
   all `< PREP_WIDTH`, so a prover supplies none of them; their values are fixed
   by the row's position in a trace whose commitment is folded into
   `lfm_program_id`.
2. **The admission validator rejects a non-one-hot selector** — the test tampers
   a real program's hash group with the same fractional `x` and asserts
   `validate` returns `NonOneHotSelector { chip: "LFM_HASH" }`, with an honest
   control that the untouched program is admissible.

**Consequence for review:** the domain separation rests on the preprocessed
binding plus the registrar, *not* on the AIR alone. That was already the design
(the spec §3.3 says "exactly-one-of stays the registrar's job"), but M5/M6 turn
it from a sentence into an executed demonstration, and it should be read as a
standing requirement on any future change that makes a mode selector a main
column.

---

## 4. The programs

### `TrivialV0` — F3.4 retired for this entry

Its raw `b.permute` became a third `compress`
(`programs.rs:60-62`). It now **proves and verifies under BLAKE3**
(`the_trivial_program_proves_and_verifies_under_blake3`), which it could not
before. Public output moved from `[d1, permuted_cell, m]` to `[d1, d2, m]` —
still three words; no test asserted the old shape (? INFERRED → ✓ VERIFIED by
the suite being green).

Permute coverage moved to `programs::permute_coverage_program_source`, a
`#[cfg(test)]` fixture that is **not** a registry entry: two chained permutations
so an output cell is also an input cell. `a_permute_row_is_refused_under_blake3`
now points at it (and keeps its honest control under `Test`).

### Cost claims — ✓ EXECUTED, both exact

`transcript_tests::the_programs_cost_what_option_b_priced_them_at`. The per-row
price comes from the census (`main_cols + 3·aux_cols`), not from a literal, and
the row counts are asserted separately — a product that came out right for two
wrong reasons is the failure mode.

| program | rows | price @7r | total | spec predicted |
|---|---|---:|---:|---:|
| `TrivialV0` | 3 compress + 0 transcript | 5,509 | **16,527** | 16,527 ✓ |
| `FriToyV0` | 56 compress + **11 transcript** | 5,509 | **369,103** | 369,103 ✓ |

The transcript's share is exactly the spec's 11 compressions (K6).

---

## 5. Regression — the other two hashers

B1 changed the sponge for **all** hashers, so `Test` and `Poseidon` had to move
and stay green.

- `transcript_tests::the_transcript_proves_and_verifies_under_every_hasher` —
  the preamble program proves and verifies under **Test, Poseidon and BLAKE3**.
- `transcript_tests::the_machine_and_the_host_chain_agree_under_every_hasher` —
  `SpongeVar` and `HostSponge` produce identical challenges under all three.
- The full `lfm::` suite has **no new failures** (§6).

⚠ **Recorded rather than assumed:** under `Test` and `Poseidon` a transcript step
*is* a Merkle parent — those hashers have one domain, so the trait default
(`LfmHasher::transcript_out` = `compress_out`) does not separate them. That is
documented at the default (`hash.rs:64-83`) as a deliberate weakening with the
reason (neither is a production hash) and the standing requirement that a future
production candidate must override it.

---

## 6. Test and lint status

**Full `lfm::` suite, release:** `290 passed; 19 failed; 7 ignored`.

All 19 failures are the pre-existing `executor/program_artifacts/recursion/fibonacci.elf`
fixture set — 15 fail directly on the missing ELF and 4 (`arena_filler_reads_real_committed_roots`,
`l2g_binding_holds_on_the_real_bundle`, `l2g_binding_proves_and_verifies`,
`tampered_l2g_binding_rejects`) fail downstream of the resulting one-epoch
fixture. **✓ EXECUTED: the failure set is byte-identical at 7 rounds and under
`--features blake3-6round`** (`diff` of the two sorted lists is empty), so
nothing in this change is round-count-sensitive.

Baseline was 276 passed / 19 failed; the +14 is this change's new tests.

| gate | result |
|---|---|
| `make fmt` | clean |
| `make lint` (4 feature combos: default, no-default+debug-checks, disk-spill, cuda) | **clean** |
| `cargo clippy -p lambda-vm-prover --all-targets --features blake3-6round` | **clean** |
| `lfm::blake3_socket_tests` | 35 pass @7r, 35 pass @6r |
| `lfm::transcript_tests` | 17 pass @7r, 17 pass @6r |

**Keccak wrap path untouched.** ✓ VERIFIED: the entire diff is 19 files under
`prover/src/lfm/` plus `thoughts/blake3/socket-kats/SOCKET.md`. Nothing in
`crypto/`, `syscalls/`, `executor/` or `prover/src/tables/`; `keccak_adapter.rs`,
`keccak_host.rs`, `transcript_replay.rs` and `wrap_tests.rs` are unmodified, and
`wrap_tests` is green.

### Registry re-bless — once, as planned

All six `program_id`s moved (`PREP_WIDTH` 11→12 moves the `LFM_HASH`
preprocessed root, which every entry binds):

| entry | old (first 8 bytes) | new |
|---|---|---|
| `TrivialV0` | `9f0537f570afe0ef` | `998428afa2a39d25` |
| `FriToyV0` | `3b4e718c02077762` | `e527cd58fc9b0c15` |
| `KeccakChainV0` | `eb591de10644b164` | `d5c294dd849e92b5` |
| `KeccakSpongeV0` | `1d90d7b5eb540778` | `421d582b159474ac` |
| `TranscriptReplayV0` | `26033a9e4101fae8` | `3371ec2badbd4c6f` |
| `StatementReplayV0` | `af842fd9b9fe6ebe` | `9f7e67a8d92c7387` |

One `log_heights` entry moved: `FriToyV0`'s `LFM_CONST` group 4→5 (16→32 rows),
which is the interned `SQ(i)` constants. Regenerated with
`cargo run --release --bin compute_lfm_registry`, pasted whole; the drift tests
pass.

### Tag tables — NOT touched by this work; one staleness reported instead

The tag-table pass was already done by the oracle before this build started:
`SOCKET.md` §2.4 and `ORACLE.md` §2.3 both carry `"LFMT" = 0x544D464C`, mark
`"LFMP"` **RETIRED UNUSED** with the reason recorded, note the O5/`"LFML"`
ratification, and put a superseded banner on `SOCKET.md` §7's rejected permute
sketch.

⚠ **I edited both files before that instruction reached me, and have reverted
those edits.** What I had done: bumped the `"LFMT"` status word from *specified*
to *built* in each table, and rewritten `SOCKET.md` §2.2's `m[8]` row. Both are
backed out; the oracle's pass is byte-for-byte intact (`git diff` on `SOCKET.md`
is now exactly that pass, and the only `gate-oracle/` file I ever opened was
`ORACLE.md`, now reverted).

**✓ VERIFIED — the tag constants agree, so there is no inconsistency in the
values.** The implementation uses `TAG_LFMC = 0x434D464C` and
`TAG_LFMT = 0x544D464C`, matching both tables exactly, and `"LFMP"` remains
retired-not-deleted so `0x504D464C` cannot be reallocated into an unanalysed
domain.

**✓ CLOSED (2026-08-11, superseded within the hour).** I had reported two stale
`m[8]` framing rows here — `SOCKET.md` §2.2 and `ORACLE.md` §2.2 still describing
`m[8]` as the bare constant `0x434D464C`, where the built chip has
`MODE_C·TAG_LFMC + MODE_T·TAG_LFMT`. **The oracle's re-transcription pass fixed
both on disk about five minutes after this report was written.** The item is
closed, not outstanding; it is left in the record because the sequence — report
rather than edit, then the owner fixes it — is the one that worked.

### Pinning instruments — untouched, and the DRIFT is expected

`artifact_pin.py`, `artifact_pin.json`, `chip_model.py`, `gate.py` and
`CHIP-GATE.md` were not opened by this work. `artifact_pin.py --check` will now
report DRIFT and exit 1 because `blake3_socket.rs` changed; that is the
instrument refusing to vouch for a chip it has not been re-transcribed against,
and re-pinning is the oracle's move, not this one's.

---

## 7. Deviations from the spec — named, with reasons

### 7.1 ⚠ `FriToyV0` does NOT prove under BLAKE3 — blocked by **O1**, not by the sponge

This is the one milestone in the brief that was not reached, and the reason is
structural rather than a shortfall in the implementation.

**✓ EXECUTED.** `execute(fri_toy_program(), fixture_arenas(), Blake3)` returns
`HasherRejected("BLAKE3 compress input lane is not a u32 (SOCKET.md obligation O1)")`.
Measured cause: **124 of the fixture's 128 committed column values are ≥ 2^32.**

`FriToyV0` hashes **FRI data** — Merkle leaves over LDE evaluations
(`compress(row_even, row_odd)`) and folded ext values — which are arbitrary
Goldilocks elements by construction. The BLAKE3 socket's inputs must be
`u32`-laned (obligation O1, pre-existing and unrelated to the transcript). No
choice of fixture polynomial changes this: the *evaluations* of a low-degree
polynomial over a coset are arbitrary mod `p`.

**What B1 did deliver here:** the transcript was one of two blockers and it is
gone — the chain runs on the compress socket under every hasher, and
`FriToyV0` now contains **zero** permute instructions. The remaining blocker is
O1 alone.

**Closing it is a different change:** field elements would have to reach the hash
through a committed `u32`-half decomposition — the shape
`transcript_replay::felt_be_halves` already uses for keccak leaves — which moves
`FriToyV0`'s arena layout and its program identity. That is a decision about the
**leaf convention**, adjacent to obligation O5, and I did not take it
unilaterally.

**Left as a tripwire, not a silence:**
`blake3_socket_tests::fri_toy_is_still_blocked_by_o1_and_no_longer_by_the_sponge`
asserts (a) no permute remains, (b) the fixture values are not u32-laned, (c) the
refusal is specifically O1 — so a refusal for any *other* reason is a regression
— and (d) the honest control that the same program and arenas still run under
`Test`. Its doc says in as many words that when O1 is closed the test must be
replaced by a prove+verify.

### 7.2 `MODE_T` sits at index 8, not appended after the multiplicities

The spec fixes `PREP_WIDTH` 11→12 but not the placement. I first appended
`MODE_T` at index 11 to minimise churn; that was **wrong** and the admission
validator caught it: `one_hot` reads the selectors as a contiguous span
(`NUM_SELECTORS` from `MODE_C`), so a selector parked past the mults would have
been **outside the one-hot check and silently unchecked**. `MODE_T` is now index
8 and `MULT0..2` shifted to 9..11; `layout::hash::NUM_SELECTORS = 3` replaces the
hard-coded `2` at the call site. The reason is recorded at the constant.

### 7.3 `SQ(i)` is an interned program constant, not a packed word

The spec says `SQ(i)` is "a constant cell … a program constant either way". The
implementation uses `LfmBuilder::digest_const`, so each distinct index costs one
`LFM_CONST` row and nothing else. (An earlier draft used `pack_word`, which would
have added an `LFM_LANES` row per squeeze — the same value, not the same cost.)

### 7.4 `WordRef::byte` / `rotr_bytes` panic on `ModeSelected`

A mode-selected word has no byte decomposition without witnessing one. Since the
whole reason the tag lives in `m[8]` is that message words reach `add3` and
nothing byte-granular, both byte-level accessors `unreachable!` on it rather than
silently acquiring four uncommitted columns. Not in the spec; it is the shape the
new variant needs to be safe.

### 7.5 Incidental DRY

`rotr16`/`rotr8` were byte-identical in `blake3_chip::WireFlow` and
`blake3_socket::SocketWire`; adding a third `WordRef` variant would have meant a
third copy in each. They now call one `WordRef::rotr_bytes`. Wire-identical
(the socket and probe suites are green at both round counts).

---

## 8. What is still open

| item | status |
|---|---|
| `FriToyV0` under BLAKE3 | ✗ OPEN — needs the O1 leaf-convention decision (§7.1) |
| gate extension: `chip_model.py` `MODE_T` role + a `WordRef`-equivalent for the mode-selected tag + `gate.py` B0a/B0b mode-sum widened to `MODE_C + MODE_T + MODE_P` | ✗ OPEN — spec §5.2. Not attempted here: `gate-oracle/` is the oracle's instrument and this build touched none of it. **What the gate needs from the chip is all exposed**: `cols::MODE_T`, `cols::MU_COLUMNS = (MODE_C, MODE_T)`, `TAG_SELECTOR` (the `(column, tag)` pairs verbatim), `tag_for_mode`, and unchanged constraint indices (0–3 capacity, 4 mode-sum, 5 `MODE_P` pin) |
| the two stale `m[8]` framing rows in `SOCKET.md` §2.2 / `ORACLE.md` §2.2 | ✓ CLOSED by the oracle's re-transcription pass |
| ⚠ `others/lfm-hash-matrix-scope.md` cites the pre-B1 rate | ✗ OPEN — §9 below; flagged for the lead rather than edited |
| production hasher overriding `LfmHasher::transcript_out` | ✗ OPEN by design — BLAKE3 does; a future candidate that does not is shipping an unseparated transcript (`hash.rs:64-83`) |
| squeeze-run bound revisit at `k = 2^16` | recorded at `SpongeVar` (`edsl.rs:44-70`); today's max run is `NUM_QUERIES = 4` |

---

## 9. Post-review fixes (b1-verify.md D1–D4)

The adversarial review found **no soundness defect**. Four items came back to
this workstream; all four are done. No soundness-relevant code changed — D1 is a
cost model, D2 is prose, D4 is filler discipline.

### D1 — MEDIUM. `LFM_HASH_RATE_FELTS` was derived from the deleted duplex. FIXED, and the projection gets WORSE.

The review is right and my report missed it: `8` was "2 of 3 state cells", the
rate of the construction B1 deleted. It is a **live** constant driving the epoch
verifier's permutation-axis projection, and it is quoted in the hash decision
record.

**⚠ This is not a number swap. Two of the model's premises broke with it, and
the corrected projection is materially worse.**

| | before | after |
|---|---|---|
| `LFM_HASH_RATE_FELTS` | `8` (literal) | `HASH_DIGEST_FELTS` = **4** (derived) |
| candidate/keccak rate ceiling | 17/8 = **2.125×** | 17/4 = **4.25×** |
| FRI-layer leaf (6 felts) | 1 block — rate-INVARIANT | **2 blocks — rate-SENSITIVE** |
| decomposition | "only the leaf term moves" | "only ABSORPTION moves" |

1. **The constant is now derived, not remembered** (`epoch_verify.rs:428-456`).
   The chain absorbs one cell per step, so the rate *is*
   `hash::HASH_DIGEST_FELTS`, and it is written as that constant so it cannot
   outlive its derivation a second time.
2. **The lever moved, and it is a worse one.** Under the duplex the rate
   followed from `HASH_STATE_FELTS = 12`, so widening the state bought
   throughput freely. Under the chain it follows from `HASH_DIGEST_FELTS = 4`,
   which is *the same constant the socket's 64-bit collision bound rests on* —
   **throughput and collision resistance are no longer independent knobs.** That
   belongs in the hash decision record.
3. **The FRI-leaf term became rate-sensitive.** The old model folded it into the
   invariant remainder on the premise "a layer leaf fits any rate ≥ 6" — true at
   8, false at 4. New `FRI_LEAF_FELTS` + `fri_leaf_permutations_at_rate`
   (`epoch_verify.rs:450-510`); `query_permutations_at_rate` now sums four terms
   and its doc states the real rule: **absorption is rate-sensitive, compression
   is not** (Merkle parents of both kinds compress and do not move).
   At rate 17 the new term reduces to `num_committed()`, so the rate-17
   differential against the byte-side `query_permutations` is preserved
   unchanged — that check still passes by construction.
4. **The broken assertion is gone, not re-asserted.** `epoch_verify_tests.rs`'s
   `6 <= LFM_HASH_RATE_FELTS` is replaced by
   `blocks_at_rate(6, 17) == 1` / `blocks_at_rate(6, 4) == 2`, and the
   decomposition assert now reads *"only ABSORPTION may move with the rate"*.
   The printed banner reports absorbed/compressed instead of leaves/paths, the
   ceiling is computed from the constants rather than written `2.125`, and it
   carries a ⚠ line naming the change.

**✗ The corrected epoch ratio is NOT computable in this environment.** The
consuming block lives inside `the_assembled_epoch_verifier_runs`, one of the 19
`fibonacci.elf`-blocked tests — which is precisely how the constant outlived its
derivation. So I added an **ELF-free** test,
`epoch_verify_tests::the_candidate_rate_model_is_derived_not_remembered`, which
executes here (✓ PASS at both round counts) and pins the correction itself: the
constant's derivation, that a 6-felt leaf is 1 block at 17 / 1 at the old 8 /
**2** at 4, that the new FRI term reduces to `num_committed()` at keccak's rate
and doubles at the candidate's, and that a terminal-only shape contributes zero
at every rate. **That test would have caught the original defect.**

Illustrative magnitude of the term the old model could not express at all
(ELF-free arithmetic, blowup 8 / 73 queries, **not** the epoch total):

| `log2_lde` | committed layers | FRI-leaf perms @17 | @8 (old) | @4 (new) | added |
|---:|---:|---:|---:|---:|---:|
| 16 | 9 | 657 | 657 | 1,314 | **+657** |
| 20 | 13 | 949 | 949 | 1,898 | **+949** |
| 22 | 15 | 1,095 | 1,095 | 2,190 | **+1,095** |

Per sub-proof. The trace-group leaf term worsens separately, from a ≤2.125×
multiplier to ≤4.25×.

**⚠ FLAGGED, NOT EDITED — `others/lfm-hash-matrix-scope.md` is stale in three
places** (the lead's message offered either; I flagged because the corrected
epoch number cannot be produced here, so any banner I wrote would announce a
wrong number without supplying the right one):

- **:128** — *"the `LFM_HASH` sponge is 'state = 3 cells (rate 2, capacity 1)'
  … **8 felts per permutation**"*. The cited `edsl.rs:16-17` no longer says that.
- **:130** — *"a candidate behind socket 2 pays **2.125×** as many
  permutations"*. Now 4.25× on the absorption term.
- **:228 — the one that matters most.** It argues Miden's BlakeG figures
  transfer directly because *"State 12, rate 8, digest 4 is exactly our frozen
  `LFM_HASH` contract"*, and concludes *"every field-native candidate shares ONE
  permutation count"*. **The rate-8 half of that identification is gone**, so
  the transfer argument needs re-examining, not just the number. `:1132`'s
  ⚠ conservatism note is unaffected (it is about padding, and holds at any rate).

### D2 — LOW. Two stale `PREP_WIDTH = 11` prose sites. FIXED.

- `statement.rs:43` — the load-bearing sentence justifying why `lfm_program_id`
  folds the hasher tag in. Rewritten to say what is actually true and stable:
  the preprocessed group is the **instruction** group, which no candidate
  changes, so every hasher commits the same width (12 since `MODE_T`).
- `poseidon_chip_tests.rs:546` — the prose 380 lines below the assertion I had
  already updated to 12.

✓ EXECUTED: `grep -rn "is 11" prover/src/lfm/` is now empty.

### D3 — my report amended.

§6's tag-table item and §8's open-items table now record the two `m[8]` framing
rows as **closed by the oracle's re-transcription**, which landed minutes after I
wrote the original claim. The record is no longer self-contradictory.

### D4 — APPLIED. The filler reads the row, like the Poseidon one.

It was genuinely small, and it closes a real gap rather than only a stylistic
one: `m[8]` is a linear form over `MODE_C`/`MODE_T`, so a filler that takes the
tag as an argument can be handed a domain the row's own selectors contradict.
`chip_trace` populates the preprocessed columns *before* calling `fill`, so the
row already carries them.

- `fill_socket_witness(row)` now derives the tag via `tag_from_row`, which panics
  if the row selects neither two-to-one domain (`blake3_socket.rs:810-855`).
- `fill_socket_witness_tagged(row, tag)` is retained `pub(crate)` for the M1/M2
  controls, which must build a row whose witness and mode columns deliberately
  disagree — production can no longer construct that.
- ✓ VERIFIED asymmetry, stated rather than hidden: the BITWISE **histogram**
  (`trace.rs:182-203`) still routes the domain through `tag_for_mode`, because it
  runs before any trace row exists. Filler and histogram must agree, and a
  disagreement unbalances the `ByteAlu` bus — which the socket's prove+verify
  tests cover.

### One more stale site, not in the review: `blake3_probe.rs`'s rate 8

✓ VERIFIED and **corrected as comments only — the arithmetic is right and
untouched.** The probe hard-codes rate `8` in three places. That is BLAKE3's own
rate (its socket absorbs two cells of message per compression) and B1 did not
change it, so every number the probe prints is still correct. But its gloss read
*"@ rate 8 (blake and field-native)"* — the two coincided only while the sponge
was a duplex. The gloss now names BLAKE3 alone and points at
`LFM_HASH_RATE_FELTS` for the field-native chain's 4.

### Verification of this pass

| gate | result |
|---|---|
| full `lfm::` suite | **291 passed / 19 failed** — failure set `diff`-identical to before these fixes (the `fibonacci.elf` 19); +1 is the new ELF-free rate-model test |
| `lfm::blake3_socket_tests` | 35 pass (D4 touched every filler call) |
| `lfm::transcript_tests` | 17 pass @7r, 17 pass @6r |
| `the_candidate_rate_model_is_derived_not_remembered` | pass @7r and @6r |
| `make fmt` + `make lint` (4 combos) | clean |
| `clippy --features blake3-6round` | clean |
