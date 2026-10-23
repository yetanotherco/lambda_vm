# Phase 2 — adversarial verification of the `LFM_HASH` BLAKE3 arm

**Reviewer:** F9 (adversarial). **Date:** 2026-08-10.
**Target:** uncommitted change set in `/Users/maurofab/workspace/lambda_vm-blake3-impl`,
branch `blake3-real-hash`, HEAD `65025095`.
**Report under review:** `thoughts/shared/lfm-real-hash/phase2-report.md`.
**Spec:** `gate-oracle/chip_model.py` + `ORACLE.md` §7.

> **⚠ READ THE ADDENDUM FIRST.** The worktree was being edited while this review ran, and
> `phase2-report.md` was itself updated at 20:58 — *after* I first read it. The implementer
> has since frozen the tree and confirmed my pinned hashes are final. The addendum below
> records what that changed: **D4, D5 and D8 are WITHDRAWN**, D3's stated *cause* was
> **wrong** and is corrected, and D1 reduces to one precise residual.

---

## Verdict table

| # | target | verdict |
|---|---|---|
| 1 | executor change | **CONFIRMED-SOUND** |
| 2 | O1 (host reject + chip binding + add2 mod-p algebra) | **CONFIRMED-SOUND** (algebra machine-checked), with **D2** — the load-bearing half is untested |
| 3 | MU-gating, `MODE_P = 0`, #915 admission bounding | **CONFIRMED-SOUND**, with **D5** (a claim, not the code, is wrong) |
| 4 | bus balance / census | **CONFIRMED-SOUND** |
| 5 | claim verification by execution | **CONFIRMED on the final tree**, one count refuted — see **D1**, **D3** |
| 6 | hygiene sweep | **CONFIRMED-SOUND**, with **D4** |
| 7a | knob aliasing — no second rounds knob survives | **CONFIRMED**, with **D9** (invariant unpinned) |
| 7b | `canonical_expected_out` is not vacuously selected | **CONFIRMED-SOUND** |
| 8a | add2 expression-carry equivalence (implementer's challenge) | **CONFIRMED** (machine-checked); premise stale, see **D7** |
| 8b | no permute row can reach the trace filler (implementer's challenge) | **CONFIRMED-SOUND** |
| 8c | tautology sweep — no test reads the knob where it means a fixed count | **CONFIRMED** clean, now enforced (**D9 closed**) |
| 9 | D2/D9 fixes + zero-delta model census | **VERIFIED by execution**; surfaces **D10** |

---

## Addendum 3 — D10 fixed in-tree; two out-of-tree sites remain

Fixed at hashes `fd19f4c5…` (`blake3_socket.rs`) / `540233bb…` (`blake3_socket_tests.rs`),
other eleven files unchanged, scope still 12 M + 3 ??.

**The change is doc-only, verified rather than taken.** `NUM_CONSTRAINTS = 26 + 16 * NUM_G`
and `CORE_IDX = 26` are intact, and all three moved anchors land exactly where claimed:
`:267` `panic!(`, `:820` the idx 6–13 loop, `:880` the add2 loop.

**The rewritten argument is correct on every leg.** I checked it as adversarially as the
original, because a garbled correction to a soundness argument is worse than the wrong one
it replaces:

1. *"the core reads the same linear form as the message word, so `IN_lane` and `m[lane]` are
   the same field element by construction"* — ✓ `message_word_ref(i)` for `i < 8` returns
   `word_cols(cols::lane_byte(i, 0))`, the very columns the identity's right-hand side sums.
2. *"the textbook alias … is unconstructible here, not merely prevented"* — ✓.
3. *"(It is real for a chip that derives the message bytes by reduction mod 2^32 instead of
   by a checked decomposition … it is not what the `AreBytes` sends buy.)"* — ✓ and this
   parenthetical is the right call: it keeps the *design* justification for choosing a
   checked decomposition while detaching it from the sends, which is exactly the conflation
   that caused D10.
4. *"`m` reaches `add3` and nothing else, never an XOR"* — ✓ (`blake3_socket.rs:428` is the
   only site; they independently confirmed `blake3_chip.rs:327, :333`).
5. The round-0 solve-for-any-`s` argument is reproduced correctly, including that `a`, `b`
   are compile-time constants, that `s` is byte-bounded by the consuming XOR, and that the
   other three `MB` bytes can be zeroed because nothing bounds them. ✓

Their observation that the **standalone chip had it right all along** also checks out:
`blake3_chip.rs:52` reads *"all 64 `m` bytes keep their explicit `AreBytes` (they are never
XORed)"*. So D10 was a regression in the socket's prose, not a gap in the design, and the
chip's framing is the one the out-of-tree sites should converge to.

### ⚠ CORRECTION TO MY OWN FINDING — D10 is ONE site, not three

I wrote that D10 hit `ORACLE.md` §7 O1 as well. **That was wrong, and I made the claim
without re-reading the file** — the exact failure mode my own claim-verification discipline
exists to prevent. The implementer caught it. Verified now, properly:

- `grep` for the collision story across the whole of `ORACLE.md` returns **nothing**.
- §7 O1 says only that the host *"must **reject** an out-of-range lane, not silently reduce,
  or host and chip disagree about what was proved"* — a correct statement about the host
  side, with no collision claim.
- `ORACLE.md` BLOCK 1 already carries the **correct** argument, and reached it
  independently of both of us: *"Without the `AreBytes`, the byte columns are full field
  elements … and **the prover chooses the message that gets hashed** — every load
  authenticated through `compress` becomes forgeable."* It even records the load-bearing
  fact I derived: *"The message enters `f` only through `add3` — it is never XORed — so
  those 32 bytes needed an explicit `AreBytes` regardless."*

**So D10's remaining scope is a single site:** `gate-oracle/chip_model.py`'s
`emit_lane_bytes` docstring, lines 152–156 — *"the only thing standing between a
prover-hinted Merkle sibling and a chosen collision … so `v` and `v + 2^32` hash alike."*

The scoping note survives and matters more now that it is the only item: that file still
**enforces** `self.c.are_bytes(*word)`, two sends per lane, so this is a comment fix, not a
re-derivation, and it must not become a reason to defer the gate re-run.

Worth remembering: of the four documents discussing this, three — `ORACLE.md` BLOCK 1,
`blake3_chip.rs:52`, and now `blake3_socket.rs` — independently reach the
"`m` is never XORed, so these sends are its only bound" reasoning. The two that drifted to
the collision story were the socket module doc and the model docstring: **the two closest to
the new code.** That the drift happened twice, in exactly those two places, is the
transferable lesson.

### One clause added after I closed — reviewed, correct

The implementer added a closing paragraph to the O1 bullet and flagged it as unreviewed
rather than letting it ride on a closed verdict. Correct instinct, and it is reviewed now.
Final hash **`89856eb4…`** (`blake3_socket_tests.rs` unchanged at `540233bb…`);
`NUM_CONSTRAINTS` and `CORE_IDX` sit at `:779`/`:782`, exactly +7 from `:772`/`:775` — the
new paragraph's 6 lines plus a blank — so doc-only is confirmed arithmetically, not
asserted. **44 passed / 0 failed / 2 ignored** re-run at that hash.

The clause states the mechanism positively: *"what the sends do is **transfer a bound onto
the lane**. Without them the identity is satisfiable for every felt `IN_lane` — put the whole
value in `MB[0]` — so it bounds nothing. With them the four bytes sum to less than `2^32`,
so it is satisfiable exactly when `IN_lane < 2^32`, and then the decomposition is unique."*

Both directions check, computationally:

- **Without**: `MB[0] = x`, rest zero, gives `Σ = x` for *any* felt `x`. The identity bounds
  nothing. ✓
- **With**: `Σ ≤ 255·(1 + 2^8 + 2^16 + 2^24) = 4294967295 = 2^32 − 1`, so `Σ < 2^32 ≪ p` and
  cannot wrap; the identity is satisfiable exactly when `canonical(IN_lane) < 2^32`, and the
  four-byte base-256 representation of such a value is unique. ✓

It is also consistent with the paragraph above it rather than a competing story: the sends
bound the *bytes*, the identity transfers that bound to the *lane*, and because `m[lane]` is
the same linear form the message word is bounded by the same step. "Only range check on `m`"
and "transfers a bound onto the lane" are one mechanism seen from two ends.

*Minor, offered rather than filed:* `ORACLE.md` BLOCK 1's intermediate step —
*"one linear equation in four unknowns leaves three of them free"* — is inherited from the
standalone chip, where `m` has its own columns and it is exactly right. In the **socket** the
lane identity pins `m[lane]` to `IN_lane`, so the three spare byte degrees of freedom buy
the prover nothing; the operative freedom is `IN_lane` itself being an unbounded
prover-hinted felt. The **conclusion is correct either way** — this is a routing nit, not a
second D10, and I flag it only because I have just been burned for over-claiming.

**Important scoping: the gate's theorem is NOT affected.** I checked that
`emit_lane_bytes` still *enforces* the check — `self.c.are_bytes(*word)`, two sends per lane
— so what is wrong in the model is the prose explaining why, not the constraint being
gated. This is a comment fix on the oracle side, not a re-derivation.

Freshest mtimes, for D7: `run-gate.log` **20:09:33**, `chip_model.py` **20:40:46**,
`gate.py` **21:16:59**, `ORACLE.md` **21:27:24**. The recorded PASS is now stale against all
three.

**D7 and the census staleness both survived the 21:27 `ORACLE.md` edit** — I re-checked
rather than assuming the edit swept them up. §3.2 still reads main **3,533 / 3,037**,
cell-equiv **5,606 / 4,822**, and a 7-round breakdown of `add2` **560** + `I/O+MU` **13**,
i.e. the carry-column model's figures, against the executed **3,436 / 2,956** and
**5,509 / 4,741** with `add2` 448 and prefix 28. And `run-gate.log` is still 20:09:33: the
gate has not been re-run.

---

## Addendum 2 — D2 and D9 closed; one NEW finding (D10)

Tree re-opened and edited to close D2/D9. New hashes, re-verified by me and identical
before and after every run below:

```
9d91954dd243b35601ae787ea43a2f1729d800675f208226bff49bfb2c44fafa  blake3_socket.rs
cac6348a339a5f129f4f21cf12253796d984ae1ccb492cbaacc1a753dbf32058  blake3_socket_tests.rs
```
(aggregate of all `prover/src/lfm/*.rs`: `62b13a25…`, unchanged across all three suites.)

| claim | measured |
|---|---|
| `lfm::blake3` at 7r | **44 passed, 0 failed, 2 ignored** ✓ |
| `lfm::blake3` at 6r | **44 passed, 0 failed, 2 ignored** ✓ |
| full `lfm::` | **263 passed, 19 failed** — the identical pre-existing set ✓ |

**D9 — CLOSED, and closed better than I asked.** `const _: () = assert!(SOCKET_ROUNDS == BLAKE3_ROUNDS)`
at `blake3_socket.rs:132`, plus `assert_eq!(NUM_G, blake3_chip::NUM_G)` in the layout test.
The second is the one with teeth: it ties the socket's layout to the standalone probe's, so
re-introducing a `cfg` pair fails even if someone edits the alias to match.

**D2 — CLOSED, and building it produced a genuine correction to my framing.** I proposed two
witnesses; only one behaves as I claimed, and the implementer found this by asserting "no
violations" and getting `[26, 89, 155, 197, 254, 296, 323]`.

Root cause, which I verified independently: **the lane bytes ARE the message bytes.**
`message_word_ref(i)` for `i < 8` returns `word_cols(cols::lane_byte(i, 0))` — the very
columns the lane-decomposition constraint reads. So absorbing an alias carry into `MB[3]`
satisfies the lane identity *and* moves message word `m[0]` by `2^32`, which the add3 sum
identity rejects. `the_lane_range_check_is_load_bearing_on_its_own` now pins all three cases
and, more usefully, pins *which mechanism* catches each:

| witness | caught by | pinned as |
|---|---|---|
| (a) `MB[0] += 256, MB[1] −= 1` — weighted sum preserved exactly | **only `AreBytes`** | eval set asserted *silent*, then rejected at proof level |
| (b) `IN0 += 2^32`, bytes untouched | the lane identity | `violations` contains lane index 6 |
| (c) `IN0 += 2^32`, `MB[3] += 256` | the **mixing core** | index 6 explicitly *absent*, all violations ≥ `CORE_IDX` |

Case (a)'s "eval set is silent" assertion doubles as the honest control for (d): if it ever
starts failing, the proof-level half has silently become a duplicate of
`the_lane_decomposition_binds_the_felt_to_its_bytes`. That is the right shape.

(d)'s reasoning is sound too: the shuffle leaves `IN0` untouched, so the `LfmMem` receive
token is unchanged and the rejection can only come from the range check. I confirmed the
message bytes have **no** `ByteAlu[XOR]` consumer — `m` is passed only into `add3`
(`blake3_socket.rs:428`), never into `xor` — so the lane `AreBytes` pair is genuinely their
only range check.

### D10 — the recorded justification for O1 names a hazard that cannot occur (severity MEDIUM, docs/argument)

Prosecuting the implementer's discovery one step further turns up something neither of us
had. **The chip is correct; the *reason* written down for why it is correct is wrong**, in
three places at once:

- `blake3_socket.rs` module doc, O1 bullet: *"If the four message bytes of a lane were taken
  by reduction mod 2^32, then `v` and `v + 2^32` would hash alike: a free, prover-chosen
  collision."*
- `ORACLE.md` §7 O1 and `chip_model.py`'s `emit_lane_bytes` docstring: *"without the
  AreBytes the prover picks the bytes — so `v` and `v + 2^32` hash alike."*

**They cannot hash alike.** The lane identity forces `IN0 = Σ MB[k]·2^{8k}` and the mixing
core reads *the same linear form* as `m[0]`, so `IN0` and `m[0]` are equal as field elements
**by construction**. Move the lane and you move the message word with it — which is exactly
what case (c) demonstrates empirically. There is no configuration in which two different
`IN0` values feed the same message. The stated attack is unconstructible, and the
implementer's failed assertion is the proof.

**What the sends actually buy is stronger.** The lane `AreBytes` are the *only* bound on
`m[0..8]`, and the mixing core's field identities need that bound to be exact. Concretely,
in round 0 the first add3's other operands are compile-time constants (`input_h` and
`input_v12` return `WordRef::Const`), and its output `s` is byte-bounded because the `X1`
XOR consumes it. So the constraint is `μ·(a + b + m − s − 2^32·(c1+c2)) = 0` with `a, b`
constant, `s ∈ [0, 2^32)`, `c1+c2 ∈ {0,1,2}`. Drop the lane `AreBytes` and `m` becomes an
unbounded field element, so a prover who wants a chosen `s` simply solves

```
m ≡ s + 2^32·k − a − b   (mod p)
```

sets `MB[0][0] = m` with the other three bytes zero (the identity is satisfied, nothing
range-checks them), and hints the sibling cell `IN0 = m` — which `edsl::merkle_walk` lets it
choose. **The first add3's output, and hence the whole compression, becomes prover-chosen.**
That is a real soundness break, it needs no aliasing story, and it is what the sends prevent.

Why this matters rather than being pedantry: the next person to audit this arm will read the
O1 bullet, try to build the `v` / `v + 2^32` collision, fail exactly as the implementer did,
and may conclude the range check is redundant. The correct one-line statement is *"the lane
`AreBytes` are the only bound on the message words, and the add3 exactness argument needs
`m < 2^32`."* Recommend rewriting the O1 justification in all three places; the constraint
system needs no change.

## Addendum — after the freeze (supersedes parts of D1/D3/D4/D5/D8)

The implementer froze the tree, confirmed **all four pinned hashes are the final state**,
and pointed out that `phase2-report.md` was updated at **20:58**, after the round-flip wave
and after I first read it. I re-read the updated report and re-audited. Net effect:

| finding | status after re-check |
|---|---|
| **D4** — "default build changes, report implies it doesn't" | **WITHDRAWN.** The 20:58 report states it outright: `blake3.rs:63-85` *"**`BLAKE3_ROUNDS` flipped to 7**"*, `blake3_chip.rs:720` *"`output_words()` follows `BLAKE3_ROUNDS` (was hardwired 6-round)"*, and §5.3 *"it drives both chips through the single `BLAKE3_ROUNDS`"*. My objection was against the pre-flip report. |
| **D8** — "report omits O5 and the collision bound" | **WITHDRAWN.** The 20:58 report carries a full ORACLE §7 obligation table with **O5 marked ✗ OPEN** (`:258`), a dedicated paragraph (`:263`) and the 64-bit birthday note (`:274-275`). |
| **D5** — "'every eval constraint × MU' is false" | **WITHDRAWN.** I read a phrase out of its cell. In §3's row 14 the *left* column states the model's requirement; the *right* column — the one describing the Rust — says *"every BLOCK 1–5 constraint gated"*, which is exactly right, and §3.2 declares idx 14–21 separately as BLOCK-0 framing. The report is internally consistent; the code is unchanged and still correct (ungated is strictly stronger). |
| **D3** — the two transient failures | **Conclusion stands, my stated CAUSE was WRONG.** See below. |
| **D1** — moving target | Reduces to **one precise, mechanical residual**. See below. |
| D2, D6, D7, D9 | **Unchanged and still open.** |

### D3, corrected — they were real assertion failures, not half-applied edits

I hypothesised the two extra failures were an inconsistent mid-write snapshot. **That was
wrong**, and the implementer's account is better than mine: no file was ever mid-write;
both were genuine assertion failures caused by the round flip landing before the dependent
expectations were updated.

1. `blake3_probe::the_hosted_chip_proves_and_verifies` asserted a hardcoded `1_248`
   BITWISE feed — the **6-round** figure; at 7 rounds it is 1,440.
2. `blake3::tests::six_rounds_is_not_the_blake3_crate` read `BLAKE3_ROUNDS`, which had just
   become 7, so it compared 7-round output against `blake3::hash`, they matched, and **the
   negative control had become a tautology.**

Both are fixed in the frozen tree, and §5.3 of the updated report now discloses the
tautology trap by name. This *strengthens* the process point rather than weakening it: a
reviewer sampling an uncommitted tree measured, as real failures, bugs the author had
already found and fixed — and one of them was a control silently ceasing to discriminate,
which is the single worst failure mode for a test suite of this kind. Good catch by the
implementer; my job was to notice it independently and I only got as far as "these two are
new", not "and here is why".

### D1, reduced — every `blake3_socket.rs` line reference in the report is off by +17

The 20:58 refresh updated the counts and the `blake3.rs` / `blake3_chip.rs` /
`blake3_probe.rs` references, but **not** the socket ones — `blake3_socket.rs` gained 17
lines of module doc at 20:44 and its line numbers were never re-derived. Checked
mechanically, nine for nine:

| report says | actually at that line | intended construct is at line + 17 |
|---|---|---|
| `:215` (permute panic) | `// The host-side hasher` | `panic!(` |
| `:350` (`message_word_ref`) | `pub const fn out_byte(…)` | `fn message_word_ref(…)` |
| `:532-545` (XOR sends) | a doc line | `for xw in &wires.xors {` |
| `:557-570` (lane `AreBytes`) | `byte_bus_value(xw.b.byte(b))` | `for lane in 0..cols::NUM_LANES {` |
| `:766` (idx 6–13) | `// idx 4: mode sum-boolean` | `for lane in 0..cols::NUM_LANES {` |
| `:787` (idx 22–25) | `b.emit_base(6 + lane, …)` | `for i in 0..OUT_WINDOW {` |
| `:802-824` (add3) | a comment | `for aw in &wires.add3s {` |
| `:826-834` (add2) | `let sum_id = …` | `for aw in &wires.add2s {` |
| `:840-876` (rot) | a comment | `for rw in &wires.rots {` |

Fix is one `sed`: add 17 to every `blake3_socket.rs:` reference in `phase2-report.md`.
Everything else in the 20:58 report checks out against the frozen tree.

---

**Bottom line: I found no soundness defect and no regression to the existing machine.**
Everything I found is either a process problem (D1, D3), a test-coverage gap on the one
claim that matters most (D2), or a report/claim inaccuracy (D4, D5, D7, D8). The arm
itself holds up under every attack I could construct.

---

## The tree I actually verified

The review is pinned to these hashes, which were **identical before and after** every
test run reported below:

```
f4a61d76100c17438ba29bffc43e5421e4fc5e59ef1c255df735853a2bd88134  prover/src/lfm/blake3_socket.rs
675148529a70404528721ea2d279599482faf1f985c3f864505043e7e16b7280  prover/src/lfm/blake3.rs
8030b0b3c3deae0a6a329dffefb8b5d3655c087dbc3d79ebac417f3154b03f67  prover/src/lfm/blake3_chip.rs
8c14a9057c4ff48e03aa687387e84fd8d91351b3da3006d0f1d05406ebb83349  prover/src/lfm/blake3_probe.rs
d2ecfa5c15d6256196661278df4754558f531a7739d3fef9654e9c7e0cb2cf9a  prover/src/lfm/trace.rs
1eb4b8aa573a884e0c46ccc83a7b9c7772b4ce56dfc436d8742d9dc70c848d89  prover/src/lfm/hash.rs
a1f4d19d2a0183b171c8226990919259a11857cb4a83dbe394f7005ea9a1c8ee  prover/src/lfm/executor.rs
4a00ec8933a6eb6d0f7f55cea2a6e5662355c0e102d7e73673a3a59565402f02  prover/src/lfm/chips.rs
```

Scope matches the brief: **12 modified + 3 new**, nothing outside
(`Cargo.lock`, `prover/Cargo.toml`, `airs.rs`, `blake3.rs`, `blake3_chip.rs`,
`blake3_probe.rs`, `chips.rs`, `executor.rs`, `hash.rs`, `mod.rs`,
`poseidon_chip_tests.rs`, `trace.rs`; new `blake3_socket{,_kats,_tests}.rs`).
Note `mod.rs` and `Cargo.lock` are modified but absent from the report's §2 file map.

---

## Target 1 — the executor change: CONFIRMED-SOUND

**Claim:** *"Test/Poseidon behaviour is unchanged by construction — the default
`compress_out` IS the old expression."* **✓ CONFIRMED**, and the claim is exactly right.

Old (`git diff`): `let out_state = hasher.permute(state);` for **both** modes, where
`executor.rs:376-385` had already built `state = [a ‖ b ‖ hasher.compress_iv()]` on the
Compress arm.

New (`executor.rs:398-409`): Compress goes through `hasher.compress_out(&a, &b)`; the
trait default (`hash.rs:49-57`) is `state[0..4]=a; state[4..8]=b; state[8..12]=self.compress_iv(); self.permute(state)`
— the same expression, reconstructed from the same `compress_iv()`.

I checked every implementor rather than trusting the default:

- `TestPermutation` (`hash.rs:100-117`) overrides only `permute` + `compress_iv`.
- `PoseidonGoldilocks` (`poseidon.rs:591-627`) overrides only `permute` + `compress_iv`.
- Neither overrides `compress`, `compress_out` or `admits` ⇒ both take the defaults
  ⇒ bit-identical output to the old inline expression.
- `HasherKind`'s explicit delegation (`hash.rs:200-215`) routes `Test`/`Poseidon` to
  those same defaults.

**The Permute arm is untouched** (`executor.rs:408` is still `hasher.permute(state)`),
so the wrap/keccak role-1 path is unaffected. The only new behaviour on that arm is the
`admits` guard at `executor.rs:395-397`, whose default (`hash.rs:72-76`) is `Ok(())` for
every hasher that does not override it.

The frozen six `LfmMem` tuples are **byte-identical**: `chips.rs` moved the `vec![...]`
body verbatim from `bus_interactions()` into `lfm_mem_interactions()` (the diff hunk
touches only the signature; the six `BusInteraction`s are pure context lines), and
`bus_interactions(kind)` returns exactly that list for `Test`/`Poseidon`. Pinned by
`poseidon_chip_tests.rs:234-239`.

---

## Target 2 — O1: CONFIRMED-SOUND (algebra machine-checked), with a test gap

### (a) Host side — rejects, never reduces. ✓ VERIFIED

`lanes_of` (`blake3_socket.rs:198-205`) is the single lane boundary and it uses
`u32::try_from(GoldilocksField::canonical(...)).ok()?` — `try_from`, so ≥ 2^32 yields
`None`. There is **no** `as u32`, no `& 0xFFFF_FFFF` and no `% (1<<32)` anywhere in the
module. `admits` (`:266-282`) turns `None` into `Err`, and `executor.rs:395-397` turns
that into `LfmExecError::HasherRejected`.

The witness filler (`blake3_socket.rs:658-659`, the `trace.rs:244` arm) also goes through
`lanes_of` and `.expect(...)`s — a panic, not a truncation. Same for `trace.rs:185-196`.
Reaching either means the executor and the filler disagreed; neither can silently reduce.

### (b) Chip side — the binding is unique. ✓ VERIFIED

`blake3_socket.rs:783-788` emits, per lane `j ∈ 0..8`,
`MU · (IN_j − Σₖ MB[j][k]·2^{8k}) = 0`, and `:557-568` sends
`AreBytes(MB[j][0], MB[j][1])` and `AreBytes(MB[j][2], MB[j][3])` — all four bytes
covered, `Multiplicity::Column(MU)`. The receiving table
(`tables/bitwise.rs:343-372`, `AreBytes` receiver at `:784`) enumerates exactly
`x, y ∈ [0,256)`, so the bound is the tight one.

Bytes < 256 ⇒ `Σ ≤ 4294967295 < 2^32 ≪ p` (computed), so the identity cannot wrap:
the felt equals that integer exactly, hence `< 2^32`, and base-256 representation is
unique. Both halves are present and both are needed.

### (c) The add2 expression-carry deviation — mod-p algebra. ✓ VERIFIED BY COMPUTATION

`blake3_socket.rs:826-834` (numbering per the report; now `:843-851`) emits only
`MU · c · (1 − c) = 0` with `c := (A + B − s) · 2^{−32}`.

- `INV_SHIFT_32 = 18446744065119617026` **is** `2^{−32} mod p` — I recomputed it.
- `A, B, s` are `word_expr` recompositions of byte columns whose range checks I traced to
  a real consumer (add2's operands are a previous `add2`/`IV` const and a `rotr` relabel
  of a `ByteAlu[XOR]` output; its own output `s` is consumed as an XOR operand, including
  in the last round via the feed-forward). So all three are in `[0, 2^32)`.
- Integer range of `A + B − s` is `[−4294967295, 8589934590]`. Over that range the field
  value `0` has exactly one integer preimage (`0`) and the field value `2^32` has exactly
  one (`4294967296`). **A negative difference cannot alias `2^32 mod p`.**

Same check for the neighbours, all clean:

| identity | integer range | multiples of `p` in range |
|---|---|---|
| add3 `a+b+m−s−2^32(c1+c2)` | `[−12884901887, 12884901885]` | `{0}` only |
| rot `xlo·2^r − sllc·2^16 − sll`, r=4 | `[−4294967295, 1048560]` | `{0}` only |
| rot, r=9 | `[−4294967295, 33553920]` | `{0}` only |

So every "field identity" in the arm is an exact integer identity, and add3's
`s` is uniquely pinned because `c1+c2 ∈ {0,1,2}` is the true carry range.

### D2 — DEFECT (test coverage, severity MEDIUM)

**`blake3_socket.rs:778-782`** states plainly: *"NEITHER ALONE SUFFICES — without the
sends the bytes are free field elements and this identity holds for arbitrary byte
strings."* **The `AreBytes` half is never exercised adversarially.**

Every negative control breaks the *linear identity*, which is the half that is not
load-bearing for O1:

- `the_lane_decomposition_binds_the_felt_to_its_bytes` (`blake3_socket_tests.rs:874-893`)
  bumps one byte (identity breaks), then does `IN0 += 2^32` **leaving the bytes alone**
  (identity breaks).
- `tampering_with_the_witness_is_not_accepted` (`:1075-1092`): a lane byte `+1`, an add3
  carry `+1`, a digest byte `+1`, a padding row marked real — all identity/mode breaks.

**The attack that is actually O1 is not in the suite:** set `IN0 = v + 2^32` *and*
`MB[0][0] = v + 2^32` (or, cheaper, `MB[0][0] += 256`, `MB[0][1] -= 1`). The linear
identity is preserved by construction; the only thing standing between that witness and
an accepted proof is the `AreBytes` lookup having no matching table row. That is the
statement the module docs and ORACLE §7 O1 rest on, and it is asserted rather than
executed.

**I did not run it** — the brief says do not modify the worktree, and the test would have
to live in the crate. **Recommend adding it before the lead commits**; it is ~10 lines in
`tampering_with_the_witness_is_not_accepted` and it is the single highest-value control in
the arm. My reading says it will pass (unmatched send ⇒ LogUp imbalance ⇒ reject), but
"my reading says" is exactly the standard this control exists to replace.

---

## Target 3 — MU-gating and `MODE_P = 0`: CONFIRMED-SOUND

**MU is preprocessed and prover-unchosen. ✓ VERIFIED, three ways.**

1. `cols::MU = MODE_C = layout::hash::MODE_C = 6`, and `layout::hash::PREP_WIDTH = 11`,
   so MU sits inside the preprocessed prefix.
2. `compiler.rs:329-350` builds the hash group's rows from `Instr::Hash`'s `mode`
   (`Compress → (1,0)`, `Permute → (0,1)`) into
   `ColumnGroup::from_rows(layout::hash::PREP_WIDTH, hash_rows)`.
3. `airs.rs:426-434` builds the hash AIR with
   `.with_preprocessed(roots[5], layout::hash::PREP_WIDTH)` (`airs.rs:331-349`), so the
   column is under a committed root that `lfm_program_id` binds.

**Even if it were not preprocessed, the AIR bounds it.** `blake3_socket.rs:766-774`:
idx 4 is `(MODE_C + MODE_P)·(1 − MODE_C − MODE_P) = 0` and idx 5 is `MODE_P = 0`;
together they force `MODE_C ∈ {0,1}` on **every** row. This matters more than the report
says, because MU is the multiplicity of ~1,382 new BITWISE sends — a field-negative MU
would be the #915 forgery shape again. It is closed in the AIR, not just at admission.

**`MODE_P = 0` genuinely bites.** ✓ EXECUTED
(`a_permute_marked_row_violates_the_air`, `padding_is_satisfied_and_a_real_marked_empty_row_is_not`).
Belt and braces with `admits` refusing at execution (`a_permute_row_is_refused_under_blake3`,
which has its own honest control under `Test`).

**Padding rows satisfy everything.** `chip_trace` (`trace.rs:60-75`) fills only
`0..real_rows` and copies the (zero-padded) group prefix for all rows, so a padding row
has `MODE_C = MODE_P = 0` and all-zero values: idx 0–3 reduce to `S = 0` ✓, idx 4/5 ✓,
idx 6–13 and 22–25 are mu-gated ✓, idx 14–21 read zero `OUT` lanes ✓, the whole mixing
core is mu-gated ✓, and every BITWISE send has multiplicity 0 ✓.

**Every new BITWISE send carries `Multiplicity::Column(MU)`** — all three groups,
`blake3_socket.rs:536, 551, 561`. ✓

**#915 (commit `3638b825`) coverage.** `validator.rs:384-403` lists the bounded columns
per chip; for `LFM_HASH` that is `[MULT0, MULT1, MULT2]` — **not** `MODE_C`. So check 9
does *not* bound the new sends' multiplicity. That is fine, and I confirmed why: the new
sends are gated by MU, which the AIR itself pins to `{0,1}` (above). The #915 attack shape
(a committed group multiplicity holding `p − 1`) is unreachable here. **No gap.**

### D5 — inaccurate claim (severity LOW, docs only)

Report §3 item 14 asserts *"every eval constraint × MU"*. **Constraint idx 14–21 are
ungated** (`blake3_socket.rs:794-797` — `b.emit_base(14 + j, out)`, no `mu` factor). The
code is right (ungated is strictly stronger, and padding rows have `OUT = 0` so
completeness holds); the blanket claim is not, and §3.2 lists idx 14–21 without noting
the exception. Fix the sentence, not the code.

---

## Target 4 — bus balance and census: CONFIRMED-SOUND

`bus_interactions(kind)` has exactly **two** production callers and both thread the same
`hasher`: the census at `airs.rs:189` and the AIR at `airs.rs:429`. Same function, same
argument ⇒ same list in the same order, by construction. (The other three hits are tests.)

The trace's actual sends are not a separate list — the prover generates them from the
AIR's declared interactions. What must agree is the shared BITWISE table's multiplicity
histogram, and both sides come from the *same* dataflow:
`bitwise_interactions()` (`:527-571`) walks `socket_wires()`, `bitwise_ops_for()`
(`:576-612`) walks `socket_values()`, and both are `run_flow(_, FLOW)` with one shared
`FlowConfig`. Group for group:

| group | sender tuple | histogram op |
|---|---|---|
| XOR, 4/word | `(XOR, a.byte(b), b.byte(b), out[b])` | `byte_op(ByteAluXor, x>>8b, y>>8b)` |
| rot, 4/rotation | `AreBytes(pair[0], pair[1])` over `sll_lo/sllc_lo/sll_hi/sllc_hi` | `byte_op(AreBytes, hw&0xFF, hw>>8)` |
| lanes, 2/lane | `AreBytes(lane_byte(l,2p), lane_byte(l,2p+1))` | `byte_op(AreBytes, lane>>16p, lane>>(16p+8))` |

Constant operands (`WordRef::Const`, i.e. `m[8..16]` and the initial state) become
`BusValue::constant` on the sender side and the same literal `u32` on the histogram side.
`fill_socket_witness` writes exactly the 60 cells per G-block that the senders read
(4+4+4+4+16+12+12 = 56 bytes + 4 carries), plus the 32 lane bytes and 16 output bytes —
I checked the arithmetic against `cols::G_SIZE = 60` and `the_layout_assigns_every_column_exactly_once`.

Empirically, `the_blake3_socket_proves_and_verifies` passing **is** the bus-balance check:
a census/AIR/trace mismatch cannot produce an accepted proof.

The frozen six are byte-identical for Test/Poseidon before and after (see Target 1), and
`S8` is read by no bus tuple at all — the six read `IN0`, `IN0+4`, `IN0+8`, `OUT0`,
`OUT0+4`, `OUT0+8` only (`chips.rs:612-644`).

---

## Target 5 — claim verification by execution: PARTLY REFUTED

### ✓ CONFIRMED

| claim | measured |
|---|---|
| `lfm::blake3` at default (7r) | **43 passed, 0 failed, 2 ignored** |
| `lfm::blake3` at 6r (`--features blake3-6round`) | **43 passed, 0 failed, 2 ignored** |
| the two `#[ignore]`d tests hide nothing | ✓ — both are pre-existing at HEAD (`git show HEAD:…blake3_probe.rs` has `#[ignore]` at 550 and 768). They are cost-model *reporting* tests (`the_blake_column…`, the two-term RSS matrix) that `println!` projections; `#[ignore]`d for runtime, same as `wrap_tests::the_wrap_census_at_blowup_8`. Neither asserts a soundness property. |
| honest-path controls | ✓ `the_blake3_socket_proves_and_verifies` and `tampering_with_the_witness_is_not_accepted` both pass, as do the O1 pair (`an_out_of_range_lane_is_rejected_rather_than_reduced`, `the_lane_decomposition_binds_the_felt_to_its_bytes`) |
| the 7-round external anchors | ✓ `seven_rounds_is_the_blake3_crate`, `seven_rounds_is_blake3_of_the_domain_separated_message`, and the discriminator `six_rounds_is_not_the_blake3_crate` all pass |

| full `lfm::` suite, final tree | **262 passed, 19 failed, 7 ignored** — the 19 are exactly the report's set (`epoch_tests` ×7, `epoch_verify_tests` ×6, `logup_tests` ×1, `machine_tests` ×5), **no BLAKE3 failures** ✓ |

### ✗ REFUTED: "40 passed" / "259 passed"

**The report's passing counts do not describe the current tree.** `lfm::blake3` is **43**,
not 40, and the full suite passes **262**, not 259 — three tests were added after the
report was written. The **19-failure claim is CONFIRMED**; only the pass counts moved.
See D1/D3.

---

## Target 6 — hygiene: CONFIRMED-SOUND

- **`Blake3Permutation::permute` panics — unreachable, including from a malicious proof.**
  ✓ VERIFIED by tracing callers, not by trusting the comment. The only production call is
  `executor.rs:408`, guarded by `admits` at `:395-397`. The verifier path
  (`proof.rs:192-204`, `verify_against` → `LfmAirs::new_with_hasher`) uses `hasher` only
  to select `num_columns(kind)`, `bus_interactions(kind)` and `HashConstraints{kind}` —
  **it never calls `permute`, `compress` or `compress_out`.** So no proof, honest or
  forged, can reach the panic; a verifier is never in the same call graph. (`edsl.rs:30,41`
  are the *builder* emitting `Instr::Permute`, not the hasher; `fixture.rs`/`programs.rs`
  name `TestPermutation` explicitly.) Residual: the method is `pub` on a `pub` trait, so a
  library consumer calling it directly panics. Documented at the definition; low severity.
- **No debug leftovers** in the new files: zero `println!`/`dbg!`/`eprintln!`/`TODO`/
  `FIXME`/`todo!`/`unimplemented!`.
- **`blake3` crate is dev-only.** ✓ `prover/Cargo.toml` `[dev-dependencies]`, and the only
  mention of `blake3::hash` outside test modules is a doc comment (`hash.rs:153`). It
  cannot enter the production dependency graph.

### D4 — the default build DOES change with the feature off (severity LOW)

`blake3.rs` changed `BLAKE3_ROUNDS` from a hard `6` to `#[cfg(not(feature = "blake3-6round"))] = 7`.
That is not just the socket's knob: `blake3_chip` (`LFM_BLAKE3`) reads the same constant,
so with the feature **off** its `NUM_G` goes 48 → 56, its width and constraint count move
(769 → 897), and `Blake3Operation::output_words` now computes a 7-round compression where
it computed a 6-round one. The report's §5.3 frames the knob as the socket's and its file
map does not flag the change to the existing chip.

**No production impact** — I checked: `LFM_BLAKE3` is **not** among the 14 registered
chips (`airs.rs:50-66`), so no program digest and no preprocessed root moves. But
"does the default build change at all when the feature is off?" is **yes**, and the report
implies no. `blake3_compress_6round` correctly still pins 6 regardless of the feature.

---

## Target 7a — the knob aliasing: CONFIRMED (one knob), but unpinned

**There is exactly one rounds knob in the tree.** A full sweep of `prover/src` for
`ROUNDS` / `rounds: usize` / `blake3-6round` finds a single `#[cfg(feature = ...)]` pair,
`blake3.rs:82-85`, defining `BLAKE3_ROUNDS`. Everything downstream derives from it with no
branch of its own:

```
blake3.rs:82-85     BLAKE3_ROUNDS  = STANDARD(7) | SIX(6)      ← the ONLY cfg
  blake3_chip.rs:101   NUM_G = BLAKE3_ROUNDS * 8
  blake3_chip.rs:445   run_flow(_, FlowConfig::full(BLAKE3_ROUNDS))
  blake3_chip.rs:580   ValueFlow::compute → FlowConfig::full(BLAKE3_ROUNDS)
  blake3_chip.rs:729   Blake3Operation::output_words(_, BLAKE3_ROUNDS)
  blake3_socket.rs:120 SOCKET_ROUNDS = BLAKE3_ROUNDS             ← a plain alias
    blake3_socket.rs:123 NUM_G = SOCKET_ROUNDS * 8
    blake3_socket.rs:150 FLOW.rounds = SOCKET_ROUNDS
    blake3_socket.rs:184 socket_digest → SOCKET_ROUNDS
```

The three deliberate **non**-knob uses are correct and are what make the tautology fix
real: `blake3_compress_6round` → `BLAKE3_SIX_ROUNDS` (`:115`), the 6-round socket-shaped
check → `BLAKE3_SIX_ROUNDS` (`:770`), and the crate anchors → `BLAKE3_STANDARD_ROUNDS`
(`:664`, `:739`). So `six_rounds_is_not_the_blake3_crate` cannot become a tautology when
the knob is flipped — I confirmed it passes at **both** round counts.

`blake3-6round` is declared only at `prover/Cargo.toml:20` and is enabled by no crate, no
Makefile target and no CI workflow, so cargo feature unification cannot switch it on
implicitly.

### D9 — the single-knob invariant is enforced by one line and nothing else (severity LOW)

Nothing asserts `blake3_socket::SOCKET_ROUNDS == blake3_chip::BLAKE3_ROUNDS`. The existing
assertions are each internally consistent —
`blake3_socket_tests.rs:142` (`NUM_G == 8 * SOCKET_ROUNDS`) and
`blake3_probe.rs:370` (`cols::OUT - cols::G == 60 * NUM_G`) — and would all still pass if
the two chips were compiled for different round counts.

That is not hypothetical: **the tree had exactly that shape until wave 2.** `SOCKET_ROUNDS`
was its own `#[cfg(feature = "blake3-6round")]` pair before 20:44; wave 2 collapsed it to
the alias. Re-introducing the pair is a one-line regression that no test catches, and its
consequence is precisely the "silent pricing lie" — `blake3_probe`'s matrix would compare a
7-round socket against a 6-round standalone chip and the report's "hosting is 3.6% cheaper"
would be measuring two different hash functions.

Cheapest fix: a `const { assert!(SOCKET_ROUNDS == super::blake3::BLAKE3_ROUNDS) }` next to
the alias, or one line in `the_built_layout_matches_the_prediction`.

## Target 7b — `canonical_expected_out` selection: CONFIRMED-SOUND

Not vacuous in either direction, because three *independent* statements cover it and two of
them are knob-**independent** (they run whichever way the build is compiled):

| test (`blake3.rs`) | what it pins | knob-dependent? |
|---|---|---|
| `the_compression_matches_the_canonical_vectors_at_seven_rounds` (`:655-670`) | `CANONICAL_OUT_7ROUND` == `blake3_compress_rounds(…, BLAKE3_STANDARD_ROUNDS)`, all 10 | **no** |
| `the_six_and_seven_round_vector_tables_differ_everywhere` (`:674-680`) | `assert_ne!(v.out, CANONICAL_OUT_7ROUND[i])`, all 10 | **no** |
| `canonical_expected_out_follows_the_round_knob` (`:686-698`) | the accessor == the expected table **and** == `blake3_compress_rounds(…, BLAKE3_ROUNDS)` | yes |

The third is what makes selection non-vacuous: the chosen branch is checked against a
*computation* at the compiled round count, not merely against the table it just selected.
A wrong branch returns the other table, which the second test proves differs on every
vector, so the equality fails. And the second test is the explicit anti-vacuity control the
lead asked about — a generation bug that emitted the 6-round outputs twice is caught even
though it would leave the first test passing.

Both branches were **executed**: all three tests are inside the 43 that passed at 7 rounds
*and* the 43 that passed at 6 rounds. `blake3_probe.rs:461` is the consumer
(`the_hosted_chip_proves_and_verifies` asserts the chip's `OUT` columns against
`canonical_expected_out(row)`), and it passes at both counts too.

The socket's own two-table selections (`blake3_socket_tests.rs:396`, `:815`, choosing
`digest_7` vs `digest_6`) are the same shape and were likewise exercised at both counts,
and the `other_round_count` framing control (`:552`) derives its wrong count from the knob
(`if SOCKET_ROUNDS == 7 { 6 } else { 7 }`), so it stays discriminating either way.

## Prosecuting the implementer's two claims

They asked me to attack (a) the add2 equivalence and (b) the claim that no permute row can
reach the trace filler. Both survive.

### (a) "chip_model.py witnesses the carry as a column, the chip derives it — provably equivalent"

**The equivalence holds** — I machine-checked it (Target 2c): the model's pair asserts
`∃ carry ∈ {0,1}. A + B = s + 2^32·carry`; the chip asserts `(A + B − s)·2^{−32} ∈ {0,1}`,
i.e. `A + B − s ∈ {0, 2^32}` in `F_p`. With `A, B, s` byte-bound below `2^32`, the reachable
integer range is `[−4294967295, 8589934590]`, in which the field values `0` and `2^32` have
**exactly one integer preimage each**. The existential is eliminated because its witness is
determined. Same statement, one fewer column, same degree 3.

**But the premise is stale, and the direction of fit has inverted.** `chip_model.py` on disk
(mtime **20:40**) no longer witnesses the carry as a column. Its `emit_add2` docstring now
reads *"CHIP COLUMNS: s[0..4] bytes. **NO carry column.** CHIP CONSTRAINT (mu-gated), the
only one — `blake3_socket.rs:826-834`"*, and explicitly *"the gate must certify the chip
that EXISTS, not a stronger cousin, so **the model follows the chip**."* So:

- The report's §3 row 6 (⚠ DEVIATION) and §3.1's recommendation to *"re-express
  `emit_add2` before the Phase-4 gate"* are **already done — by the oracle side, not you.**
- `run-gate.log` is **20:09**, which **predates** both `chip_model.py` and `gate.py` (both
  20:40). **The recorded green verdict does not certify the model now on disk.**
- The model's own line reference (`:826-834`) is stale by the same +17 as the report's.

That is D7, and it is the one thing here that needs a *re-run*, not an edit: the gate must
be re-executed against the 20:40 model before task #4 can claim anything. Note the model is
honest about the seam — it flags `2^{−32}` as having no faithful BV counterpart and defers
the "only reachable roots" side condition to the field audit `WA7`. **I independently
discharged that side condition** (the table above), so the equivalence is not resting on the
gate to begin with.

**Executed, independently: the model and the chip now agree to zero.** I ran
`SocketChip(...).build()` at both round counts myself:

| | model main | model sends | model aux/3 | model cell-equiv | chip |
|---|---:|---:|---:|---:|---|
| 6 rounds | 2,956 | 1,190 | 595 | 4,741 | **identical on all four** |
| 7 rounds | 3,436 | 1,382 | 691 | 5,509 | **identical on all four** |

Not "small explainable deltas" — **zero**. The old −81/−97 was the carry column (112 at 7r)
net of the prefix accounting (+15), and both are gone: the model's block breakdown now reads
`add2` 448 (was 560) and `frozen_socket_prefix(IN/S/OUT)` 28 (was "I/O+MU 13").

**Consequence the lead needs: `ORACLE.md` §3.2 is stale.** Its census table still carries
main 3,533 / 3,037 and cell-equiv 5,606 / 4,822, its 7-round breakdown still says `add2` 560
and `I/O+MU 13`, and **its whole reconciliation against the standalone chip is computed from
those numbers**. If "expected census targets from the gated model" were taken from that
table, they are superseded by the four figures above. `ORACLE.md` is 20:16, i.e. also older
than the 20:40 model. It is the oracle side's file; I have not touched it.

(`gate.py` has since moved again — mtime 21:16:59 — so the 20:09 `run-gate.log` is now
stale against both the model *and* the gate.)

### (b) "no path where a permute row reaches the trace filler" — CONFIRMED

Traced rather than assumed:

- **The mode is program-derived, not record-derived.** `trace.rs:132-138` builds
  `hash_modes` by filtering `program.instrs` for `Instr::Hash { mode }` — the same source
  `compiler.rs:329-350` uses to write the preprocessed `MODE_C`/`MODE_P`. The two cannot
  disagree.
- **Only two production callers** of `build_traces_with_hasher`: `trace.rs:116` (the
  `build_traces` wrapper, which passes `HasherKind::default()` = `Test`) and `proof.rs:90`
  inside `lfm_prove_with_hasher`, which passes the *same* `hasher` it called `execute` with.
  (The two other hits are doc comments.) So on the prove path, `admits` has already rejected
  any `Permute` row before `records.hash` exists, and the filler cannot see one.
- **Three independent fallbacks if someone hand-built the mismatch** (e.g. calling
  `build_traces_with_hasher(prog, records_from_Test, Blake3)`):
  1. `trace.rs:185-196` runs **first**, mapping `lanes_of(...).expect(...)` over *every*
     hash record — a permute row's capacity lanes are arbitrary felts, so it panics there,
     prover-side, before any witness is written.
  2. If the lanes happened to be `u32`, `fill_socket_witness` would write a BLAKE3 witness
     onto a row whose preprocessed `MODE_P = 1`, which violates AIR idx 5 — unprovable.
  3. `MU = MODE_C = 0` on such a row, so every BLAKE3 constraint and every BITWISE send is
     vacuous anyway.

  Worst case is a prover-side panic or a rejected proof. **No path produces an accepted
  proof**, and none reaches `Blake3Permutation::permute`'s panic.

## The tautology sweep they asked me to run against them

The trap class they identified — *a test that reads `BLAKE3_ROUNDS` when it means a fixed
count silently stops discriminating at the default* — is the right thing to audit, so I ran
it wider than the two files they named. **Clean: no surviving tautology.**

Every `BLAKE3_ROUNDS` read in test code falls into one of two safe shapes: a *parameterised
prediction* (`predicted_main(BLAKE3_ROUNDS)`, `predicted_interactions(…)`,
`predicted_cells(…)`, `predicted_bitwise(…)` — `blake3_probe.rs:376, 379, 382, 488, 494`) or
a *branch selector* (`if BLAKE3_ROUNDS == 6`, `if BLAKE3_ROUNDS == BLAKE3_STANDARD_ROUNDS` —
`blake3_probe.rs:414, 811`, `blake3.rs:688`). The one bare use, `blake3.rs:696`, is the
accessor-vs-primitive cross-check where "the compiled count" is exactly what is meant.

Everything that means a **fixed** count now names the constant, each with a comment saying
why: `BLAKE3_SIX_ROUNDS` at `blake3.rs:115` (`blake3_compress_6round`), `:495` (the
`CANONICAL` conventions struct) and `:770` (the 6-round socket-shaped check);
`BLAKE3_STANDARD_ROUNDS` at `:664` and `:739` (the crate anchors).

Extending to `blake3_socket_tests.rs`, which they did not name: `SOCKET_ROUNDS` appears only
as a prediction argument, as a branch selector (`:396`, `:815`), as the honest framing
(`:249`), and — the one worth checking — at `:552` as
`rounds: if SOCKET_ROUNDS == 7 { 6 } else { 7 }`, the `other_round_count` negative control,
which stays a *different* count either way. The explicit-7 KAT rows (`:387`, `:429`) and the
explicit-6 row (`:381`) hardcode their counts rather than reading the knob, which is why
`seven_rounds_is_blake3_of_the_domain_separated_message` still passes under
`--features blake3-6round`.

**D9 is the residue of this class**: the sweep is clean *today*, but nothing enforces it.
See below.

## D1 — the review target moved during the review (severity HIGH, process)

**This is the finding the lead most needs.** `phase2-report.md` describes a tree that no
longer exists. Recorded mtimes:

```
18:22–18:24  hash.rs, executor.rs, chips.rs, airs.rs, poseidon_chip_tests.rs
19:27        mod.rs
20:22:43     blake3_socket_tests.rs, blake3_socket_kats.rs, trace.rs
20:37:05     blake3_chip.rs            ← after my first test run started
20:44:58     blake3_socket.rs  (878 → 895 lines)
20:45:08     blake3.rs, blake3_probe.rs
```

Consequences:

1. **The report's §2 file:line map is off.** e.g. `blake3_socket.rs:215` (the `permute`
   panic) is now `:232`; `:826-834` (add2) is now `:843-851`; the file is 895 lines, not
   the 878 the report states.
2. **Test counts moved**: 40 → 43.
3. **The implementer's "the socket arm is unchanged in wave 2" claim: ✓ VERIFIED.**
   `blake3_socket.rs` is untracked so `git diff` cannot show it; I tested it structurally
   instead. All of the +17 lines land **before** line 113, so every declaration from there
   on should sit at exactly its old offset + 17. It does, at all eight anchors I checked:

   | line | declaration found |
   |---|---|
   | 299 | `pub mod cols {` |
   | 367 | `fn message_word_ref(i: usize) -> WordRef {` |
   | 544 | `pub fn bitwise_interactions() -> Vec<BusInteraction> {` |
   | 593 | `pub fn bitwise_ops_for(rows: &[([u32; 4], [u32; 4])]) -> Vec<BitwiseOperation> {` |
   | 656 | `pub fn fill_socket_witness(row: &mut [FE]) {` |
   | 737 | `pub const NUM_CONSTRAINTS: usize = 26 + 16 * NUM_G;` |
   | 750 | `pub fn eval<B: ConstraintBuilder<F, E>>(b: &mut B) {` |
   | 843 | `for aw in &wires.add2s {` |

   Combined with re-reading the entire `eval()` body on the final tree (byte-identical to
   what I analysed: same `NUM_CONSTRAINTS`, same `CORE_IDX = 26`, same 26 framing
   constraints, same core loops), wave 2's socket-arm delta is **the O5/128-bit module-doc
   block plus the `SOCKET_ROUNDS` alias, and nothing else**. Layout, senders, histogram
   mirror, trace filler and constraints are untouched. **The analysis above stands for the
   pinned hashes.**
4. The lead has since confirmed the tree is idle and wave 2 is final; scope re-checked
   after that confirmation is still **12 M + 3 ??** at the same hashes, with wave 2 visible
   as the larger per-file deltas (`blake3.rs` +272 vs +107 before, `blake3_probe.rs` +142
   vs +19, `blake3_chip.rs` +158 vs +134).

**Recommendation: do not commit against the report's numbers.** Have the implementer
regenerate §2's file map and §6's counts against the final tree, or commit first and let
the report describe the commit. The *findings* below need no re-run — every test result in
this document was measured after wave 2 landed.

## D3 — "19 failures, all pre-existing": CONFIRMED, but only after a rebuild

My full `lfm::` run against the **20:37 intermediate** tree gave **260 passed / 21 failed**,
not the report's 259 / 19. The two extras were BLAKE3's own:
`blake3::tests::six_rounds_is_not_the_blake3_crate` and
`blake3_probe::the_hosted_chip_proves_and_verifies`.

**These were artifacts of compiling a half-applied edit, not regressions.** Decisive
evidence: re-running `lfm::blake3` on the final tree (hashes verified unchanged
immediately before and after) gives **43 passed / 0 failed**, with both of those tests
listed as `ok`. The 20:37–20:45 wave was the round-knob unification landing across
`blake3.rs` / `blake3_chip.rs` / `blake3_probe.rs` / `blake3_socket.rs`, and I sampled it
mid-flight.

**Settled by a clean re-run.** I re-ran the full `lfm::` suite against the frozen final
tree, with the aggregate hash of every `prover/src/lfm/*.rs` (`df061a67…`) verified
identical immediately before and after: **262 passed, 19 failed, 7 ignored**, and the 19
group exactly as the report says — `epoch_tests` ×7, `epoch_verify_tests` ×6,
`logup_tests` ×1, `machine_tests` ×5, none touching `LFM_HASH`. **The report's
"19 failed, all pre-existing" is CONFIRMED.** Only its pass counts are stale (262 vs 259).

**Lesson for the record:** the report's "no new test failures" was true of the tree its
author had, but a reviewer sampling the same worktree minutes later measured two new
failures. Uncommitted review targets need a freeze or a commit.

---

## Additional findings outside the numbered targets

### D6 — underconstrained-but-unread columns (severity INFO, no soundness impact)

With `FLOW.out_window = 4` the truncated feed-forward reads only `v[0..4]` and `v[8..12]`.
In the **last** round, the four diagonal G's write their `b` slot to `v[4..8]` and their
`d` slot to `v[12..16]` — neither is read by anything. The `d` words are `ByteAlu[XOR]`
outputs so their bytes stay pinned, but the `b` words are `rot` outputs `Y`, and `Y`'s four
byte columns are constrained only by the two half-sums (`Ylo = SLL_hi + SLLC_lo`,
`Yhi = SLL_lo + SLLC_hi`). Their individual bytes get their range check "free from the XOR
that consumes them" (`chip_model.py:emit_rotr`) — and in the last round there is no
consumer. So 16 byte columns carry 2 free degrees of freedom each.

**Not a soundness issue** — nothing reads them, so no digest, bus token or public value can
move. It is exactly the waste that `chip_model.py`'s optional `tail_truncate` (ORACLE §3.3,
report §3 item 16) would remove, and the model has the same shape, so the chip is
conformant. Worth knowing before someone "optimises" `Y`'s constraints on the assumption
they are tight.

### D7 — the gate no longer certifies an independently-derived model (severity MEDIUM, process)

The report's §3.1 headline — *"the one deviation: `emit_add2`'s carry"*, with a
recommendation to re-express the model *before* Phase 4 — is **stale, and the fix went the
wrong way round**. On disk right now:

- `chip_model.py` (mtime **20:40**) already models the expression-carry form. Its docstring
  reads *"CHIP COLUMNS: s[0..4] bytes. **NO carry column.** CHIP CONSTRAINT (mu-gated), the
  only one — `blake3_socket.rs:826-834`"* and *"the model follows the chip"*.
- `run-gate.log` is **20:09** — it **predates** both `chip_model.py` and `gate.py` (both
  20:40). **The recorded gate verdict does not cover the model now on disk.**

So the §3.1 action item is already done, but the spec was retro-fitted to the
implementation and the gate has not been re-run since. Two things follow: (i) the report's
§3 conformance table is against a superseded revision (it says so, but the implication is
understated); (ii) whoever picks up task #4 ("z3 gate on the real chip") must **re-run the
board** — the green log in the directory is not evidence for the current model. The
model's own `emit_add2` docstring is honest about this and points at `WA7` as the field
audit that discharges the aliasing side condition; I independently confirmed that side
condition holds (Target 2c), so the direction of fit is a process problem, not a
correctness one.

### D8 — the report omits O5 and the 64-bit collision bound (severity MEDIUM, disclosure)

`ORACLE.md` §7 lists **O5 — leaf/parent domain separation — as ✗ OPEN, needs a decision**,
and states plainly that the socket's 128-bit digest gives **64-bit collision resistance**.
`phase2-report.md` discusses O1, O2 and O3 and never mentions O4, O5 or the collision
bound. A reader of the report alone would conclude the obligation set is discharged.

The implementer evidently agreed: the 20:44 edit added exactly this to the module doc
(`blake3_socket.rs:59-78`, *"✗ OPEN — O5: leaf/parent domain separation is NOT decided"*
plus the birthday-bound note). **The code is now honest; the report is not.** If the lead
commits from the report, the open obligation is invisible. Recommend a §5.4 entry, and it
should probably become a task alongside #8.

(O4 — the `keccak_host` one-felt-one-u32 little-endian convention — **is** satisfied:
`word_of`/`lanes_of`/`set_word_bytes` are all LE per-lane, and `lanes_big_endian` is a live
negative control. Just uncalled-out.)

---

## What I could not falsify

For the record, the attacks I constructed and that the arm survived:

- Silent reduction of an out-of-range lane anywhere on the host path — no `as u32`, no
  mask, no modulus exists.
- A surviving second rounds knob letting the machine's hash and the chip it is priced
  against describe different functions — one `cfg` pair in the tree, everything else
  derived (D9 notes it is unpinned, not broken).
- A vacuous `canonical_expected_out` branch silently unpinning the chip's `OUT` columns —
  two knob-independent controls plus a primitive cross-check close it.
- `A + B − s` negative aliasing to `2^32 mod p` in the expression-carry add2 — ruled out by
  exhaustive range arithmetic.
- Choosing MU per row to zero out the BITWISE sends or make them negative — MU is
  preprocessed *and* AIR-pinned to `{0,1}`.
- Smuggling a `permute` row under BLAKE3 — refused at execution and unsatisfiable in the
  AIR, independently.
- A census/AIR mismatch making the declared and sent interaction lists differ — one
  function, one argument, two call sites.
- Reaching the `permute` panic from a verifier — the verifier never calls the hasher.
- A behavioural change to Test/Poseidon from the `compress_out` refactor — the default is
  the old expression and neither implementor overrides it.

---

## Appendix — commands run

All in `/Users/maurofab/workspace/lambda_vm-blake3-impl`, one cargo invocation at a time:

```
cargo test --release -p lambda-vm-prover lfm::blake3
    → 43 passed, 0 failed, 2 ignored          (final tree, default 7 rounds)

cargo test --release -p lambda-vm-prover --features blake3-6round lfm::blake3
    → 43 passed, 0 failed, 2 ignored          (final tree, 6 rounds)

cargo test --release -p lambda-vm-prover lfm::
    → 262 passed, 19 failed, 7 ignored        (final tree; aggregate source hash
                                               df061a67… verified unchanged
                                               before and after the run)
```

An earlier `lfm::` run against the 20:37 intermediate tree gave 260 / 21 — the two extra
failures were the half-applied round-knob edit, see D3.

Field-arithmetic checks were done in Python against `p = 2^64 − 2^32 + 1`:
`INV_SHIFT_32` recomputed as `2^{−32} mod p`; integer preimage ranges enumerated for the
add2, add3 and rotation identities (Target 2c).
