# Proving the LFM machine with a real hash — BLAKE3

**Status:** plan, not implementation. Nothing here has been built.
**Date:** 2026-08-10. **Target branch:** `pr915` (worktree `/Users/maurofab/workspace/lambda_vm-pr915`).
**Author's ground rules:** every factual claim about the tree is marked ✓ VERIFIED (I read
the code and cite `file:line`), ? INFERRED (derived, arithmetic shown), or ESTIMATE
(labelled, with its basis). Line numbers are as of the `pr915` worktree read on 2026-08-10.

---

## 0. The short version

Three things need saying before the phases, because two of them change what the goal *is*.

**(a) The 6-round reference problem is much closer to solved than the brief assumes.**
The oracle, the official-crate cross-check, the recorded 6-round vectors and the z3 gate
all exist — in **git**, on `feat/blake3-accelerator` / `spike/blake3-recovered`, not in
the working tree (which has decayed to `__pycache__` and a venv). And the ten canonical
6-round vectors are already transcribed into `prover/src/lfm/blake3.rs:151-342` **with a
negative control that breaks one convention at a time** (`blake3.rs:463-514`), and the
chip's own `OUT` columns are asserted against them (`blake3_probe.rs:403-421`). Phase 1
is therefore *restoration plus a second independent source*, not construction. Effort: S.

**(b) "BLAKE3 as the machine's real hash" names two different sockets, and the cost
numbers in the PR body belong to the one the brief is not asking about.** There are
three hash roles in this system; `HasherKind`/`TestPermutation` is role 2, the measured
2.75 B / 1.1 B epoch figures are role 1. Swapping `TestPermutation` for BLAKE3 changes
the wrap's cost by **exactly zero**, because the assembled epoch verifier emits no
`Instr::Hash` at all. §1.1 and §6 work this through. This is not a reason to abandon the
goal — it is a reason to state the goal as "make socket 2 cryptographically real and
bound", which is achievable now, rather than "make the wrap cost 2.75 B", which is a
production-side migration.

**(c) The A6R assumption buys about 5% of the epoch bill, and standard 7-round BLAKE3
removes it entirely.** Working the chip's own column and interaction budget forward from
6 to 7 rounds (§7, arithmetic shown) gives ≈ +15.5% per compression and ≈ +5.5% on the
whole epoch column. Against that, 7-round is bit-compatible with published BLAKE3, so
the `blake3` crate becomes a *direct external KAT* and A6R disappears. **My
recommendation is to build the chip round-parameterised and instantiate 7-round first**,
keeping 6-round as the measured performance variant behind an explicit signed assumption.
The user's "as long as it works with Blake 6r, or blake, it's fine" permits this, and it
is the cheaper path to a defensible result.

**Start here:** §9.

---

## 1. Ground truth — what exists today

### 1.1 There are three hash roles, and they are not interchangeable

This taxonomy is the single most important thing in this document. The scoping report
already found two of them and says so in its headline: *"The machine already has a hash
swap surface, and it is NOT the socket keccak is plugged into"*
(`others/lfm-hash-matrix-scope.md:14-56`, ✓ VERIFIED by reading).

| | role 1 — the **inner** hash | role 2 — the **program** hash | role 3 — the **outer** hash |
|---|---|---|---|
| What it is | the hash the *proof being verified* was committed under | the hash an LFM *program* calls via `Instr::Hash` | the hash the LFM prover commits its own traces under |
| Today | keccak (production RV64 proofs) | `TestPermutation` behind `LFM_HASH` | keccak (the `stark` framework) |
| In-machine chip | `LFM_KECCAK` + hosted `KECCAK_RND`/`KECCAK_RC`/`BITWISE` (`airs.rs:51-66`) | `LFM_HASH`, chip slot 5 (`airs.rs:427-435`) | none — it is outside the machine |
| Gadgets | `edsl::keccak_merkle_walk`, `edsl::keccak256` | `edsl::merkle_walk`, `edsl::SpongeVar` (`edsl.rs:16-79`) | — |
| Digest | 2 machine cells / 8 felts / 32 bytes | 1 machine cell / 4 felts (`word.rs:1-9`) | `Commitment = [u8; 32]` |
| Selected by | the inner proof's own construction | `HasherKind` (`hash.rs:101-109`) | the framework, not swappable here |
| Cost measured | 11.17 B cells / epoch verify | **0 permutations in the wrap** | n/a |

`edsl.rs:137-143` states the non-interchangeability outright: *"`merkle_walk` compresses
with `LFM_HASH`/`TestPermutation`, the deliberately non-cryptographic Milestone-C
placeholder, so it can only ever authenticate the Milestone-C fixture tree. Production
trees are keccak throughout."* ✓ VERIFIED.

**The wrap emits no `Instr::Hash`.** ✓ VERIFIED independently of the review: grepping
`b.permute(` / `b.compress(` across `builder.rs`, `edsl.rs`, `programs.rs`, `epoch.rs`,
`epoch_verify.rs`, `fri.rs`, `sub_proof.rs`, `transcript_replay.rs`, `statement_replay.rs`
returns callers in `edsl.rs:30,41,76` (the library) and `programs.rs:44-46` and
`programs.rs:576-623` (`trivial_program`, `fri_toy_program`) — and **nothing in the epoch
verifier's own modules**. `wrap_tests.rs:26-28` says the same thing from the other side.

So role 2's only current consumers are two toy programs, both of which *are* in the
registry (`registry.rs:27-42`: `TrivialV0`, `FriToyV0`). That is exactly the F3.4
disclosure: `FriToyV0` is billed as "the Milestone-C FRI-opening verifier" while its
Merkle authentication and its Fiat–Shamir sponge are both cryptographically vacuous.
**Making role 2 real is what retires that disclosure**, and it is a well-sized,
self-contained project. It is not what makes the wrap cheaper.

### 1.2 What `LFM_BLAKE3` is today, and why it is unregistered

✓ VERIFIED by reading `blake3.rs`, `blake3_chip.rs`, `blake3_probe.rs` and `mod.rs`:

- `lfm/blake3.rs` is the **primitive**: `blake3_compress_6round(h, m, t, block_len, flags)
  -> [u32; 16]`, a byte-for-byte vendoring of #903 at head `89aeeb8c`
  (`blake3.rs:1-13`), plus the ten `CANONICAL_VECTORS` and four convention tests.
  `BLAKE3_ROUNDS = 6` (`blake3.rs:56`); the loop permutes the schedule when
  `r < ROUNDS - 1` (`blake3.rs:119`), which means **setting that constant to 7 yields
  exactly standard BLAKE3's compression function `f`** — no other edit. ? INFERRED from
  reading the loop against the BLAKE3 spec; it is the property the whole 7-round fallback
  rests on and should be pinned by a test, not assumed (Phase 1, step 4).
- `lfm/blake3_chip.rs` is the **chip**: 3,072 columns of which 16 are preprocessed
  (`blake3_chip.rs:148,154-159,221`), 1,259 bus interactions, one row per compression,
  769 constraints at degree 3. Its I/O side was re-expressed on `LfmMem` word tokens
  (7 reads + 4 writes) in place of #903's syscall `Ecall`/`Memw` shape, and the header
  argues each dropped range check (`blake3_chip.rs:34-61`).
- `lfm/blake3_probe.rs` proves and verifies it standalone against the **production**
  `BITWISE` table, at 4,946 base-field-equivalent cells per compression
  (`blake3_probe.rs:327-356`), with five tamper-rejection tests.
- **Registration status:** `blake3` and `blake3_chip` appear at `mod.rs:19-20` and
  `blake3_probe` at `mod.rs:74`, and `grep -rn blake3 prover/src crypto executor` returns
  those four files and nothing else. ✓ VERIFIED — they are absent from `LFM_CHIP_NAMES`,
  `LfmAirs`, `LfmTraces`, `Instr`, the compiler and the executor.

**Why unregistered, precisely.** `NUM_LFM_CHIPS = 14` (`airs.rs:50`) and `lfm_program_id`
iterates `0..NUM_LFM_CHIPS` folding each slot's root and log-height into a keccak preimage
(`statement.rs:49-53`). Adding a 15th chip class changes the loop bound and therefore
**every registered program's digest**, which invalidates all six registry entries and
every attestation that folded one. That is the "registration moves every program digest"
consequence, stated exactly (`blake3_chip.rs:72-77`). It is a re-blessing, not a bug —
but it is a decision with a blast radius, and §3 shows it is **avoidable**.

### 1.3 The `HasherKind` swap surface as it stands

✓ VERIFIED:

- `HasherKind` has exactly two variants, `Test` (the `#[default]`) and `Poseidon`
  (`hash.rs:101-109`). **There is no `Blake3` variant.** `lfm_prove_with_hasher(...,
  HasherKind::Blake3)` does not compile today.
- The contract behind it is `LfmHasher` (`hash.rs:27-44`): `permute([FE; 12]) -> [FE; 12]`,
  `compress_iv() -> LfmWord`, and a defaulted `compress(a, b)` that permutes `a ‖ b ‖ IV`
  and truncates to the first cell. `HASH_STATE_FELTS = 12`, `HASH_DIGEST_FELTS = 4`
  (`hash.rs:19-21`).
- One `hasher` value reaches the executor, the trace filler and the AIR set through a
  single function, `lfm_prove_with_hasher` (`proof.rs:61-78`) — that is the agreement
  mechanism the brief refers to.
- `hash::num_columns(kind)` and `HashConstraints::num_constraints(kind)` are two-arm
  matches (`chips.rs:583-588`, `chips.rs:652-657`). `hash::bus_interactions()` takes **no
  hasher argument** and returns 6 `LfmMem` interactions (3 receivers over `IN_ADDR0..2`,
  3 senders over `OUT_ADDR0..2`, `chips.rs:590-623`).
- The shared value prefix `IN0..11`, `S8..11`, `OUT0..11` is 28 columns and is frozen at
  fixed offsets in **every** layout, "which is why they keep their offsets in EVERY layout
  — a candidate appends its witness columns after them rather than reflowing the prefix"
  (`chips.rs:487-491`). Poseidon appends 584 columns after it, reaching 612 + 11.
- `layout::hash::PREP_WIDTH = 11` (`layout.rs:81-94`), and because it is 11 under both
  hashers **the preprocessed roots and hence `lfm_program_id` are hasher-independent by
  construction** (`airs.rs:375-380`).

**What is missing to make BLAKE3 selectable end to end,** enumerated against those facts:
a `HasherKind::Blake3` variant and its `LfmHasher` impl; an arm in `num_columns` and
`num_constraints`; a BLAKE3 arm in `HashConstraints::eval`; a witness-filling arm in
`trace.rs` (which currently special-cases `hasher == HasherKind::Poseidon` at
`trace.rs:222`); and — the one structural change — `hash::bus_interactions()` must become
**hasher-dependent**, because BLAKE3 needs 1,248 `BITWISE` lookups per permutation that
Poseidon and `TestPermutation` do not. That is a signature change with three call sites
(`airs.rs:189`, `airs.rs:429`, and the census). See §3.

### 1.4 The binding gap (the brief calls it F3-2; the findings file numbers it **F3.3**)

✓ VERIFIED, and worth restating precisely because it is the soundness-gating item:

- `LfmRegistryEntry` has fields `kind`, `blowup_factor`, `roots`, `log_heights`,
  `keccak_rnd_chunks`, `program_id` — **no hasher** (`registry.rs:52-60`).
- `lfm_program_id`'s preimage is the tag, the machine version, the preset tag, then per
  slot `(index, root, log_height)`, then the chunk count — **no hasher**
  (`statement.rs:40-56`).
- `lfm_verify` resolves the registry entry and calls `verify_against`
  (`proof.rs:135-150`), which hardwires `HasherKind::default()` (`proof.rs:172-181`).

So today the *only* thing separating a Poseidon-proved trace from a Test-built AIR set is
that `hash::num_columns` differs (11 + 28 = 39 vs 11 + 612 = 623), which the framework rejects as a width
mismatch. That is a coincidence of layout, not a binding — and a BLAKE3 arm is exactly the
kind of third candidate that could collide with an existing width. **Fix before, not
after, adding the third arm.**

### 1.5 The 6-round reference material — where it actually is

✓ VERIFIED by `git log --all --diff-filter=A -- 'thoughts/blake3/*'`:

| artifact | added in | what it is |
|---|---|---|
| `thoughts/blake3/blake3-oracle/blake3_ref.py` | `3b9b8137` | the round-parameterised Python oracle |
| `thoughts/blake3/blake3-oracle/ORACLE.md`, `test_oracle.py` | `3b9b8137` | its documentation and tests |
| `thoughts/blake3/blake3-chip/DESIGN.md`, `z3_blake_verify.py` | `3b9b8137` | the gate-proved chip design and its z3 gate |
| `thoughts/blake3/blake3-oracle/official_test_vectors.json` | `19ed761b` | the **official crate** vectors — the external anchor |
| `thoughts/blake3/blake3-oracle/canonical_6round_vectors.json` | `19ed761b` | the ten 6-round vectors |
| `thoughts/blake3/ground-truth/{Cargo.toml,src/main.rs}` | `19ed761b` | a Rust project that links the real `blake3` crate |
| `thoughts/blake3/{TRANSCRIPTION,GATE-TRANSCRIPTION}-AUDIT.md` | `8fec369e` | two transcription audits and the corrections they forced |
| `thoughts/blake3/blake3-chip/IMPLEMENTATION.md` | `35038501` | #903's implementation notes, where A6R is named |
| `spec/blake3.typ` | `2e0f0b41`, `a7a8bdd5`, `783c5a95` | the chip spec page and the A6R section |

These live on `feat/blake3-accelerator` (and `origin/feat/blake3-accelerator`), **not on
`main` and not on `pr915`.** ✓ VERIFIED: `git ls-files thoughts/blake3` on `main` returns
nothing, and the working-tree directory now contains only
`blake3-oracle/__pycache__/blake3_ref.cpython-314.pyc`,
`blake3-chip/__pycache__/z3_blake_verify.cpython-314.pyc`, a venv, and
`ground-truth/target/`. The `.pyc` files are the compiled form of the two deleted
sources — recoverable, but `git show` is the honest route.

The provenance chain the primitive currently rests on (`blake3.rs:15-38`, ✓ VERIFIED as an
accurate self-description): official crate vectors pin the oracle **at 7 rounds**, so the
G-function, message schedule, counter split and feed-forward are externally validated;
only the round count is varied; the oracle at `rounds = 6` emitted the ten vectors. The
module says outright that this is "weaker than a direct KAT and is recorded as such."

---

## 2. Phase 1 — a trustworthy 6-round reference, and the KATs it pins

**Goal:** a re-runnable, two-source derivation of the 6-round vectors, plus a KAT layer
for the *socket instantiation* that `CANONICAL_VECTORS` does not cover.
**Effort:** S (1–2 days). **Risk:** LOW. **Blocks:** everything else.

### 2.1 What is already pinned, and what is not

✓ VERIFIED — do not redo this work:

- The primitive reproduces all ten vectors (`blake3.rs:431-440`).
- A *parameterised* control at canonical parameters equals the port
  (`blake3.rs:445-454`), so the negative controls differ in exactly one convention.
- Four conventions each break the vectors when perturbed alone (`blake3.rs:462-514`):
  `rotr12 → rotr13`, `rotr16 ↔ rotr8`, message schedule transposed, and **7 rounds**.
  The last one is the round-count discriminator, and it is already there.
- The counter halves are not interchangeable (`blake3.rs:519-538`).
- **The chip is pinned to the vectors, not merely to the primitive.**
  `the_hosted_chip_proves_and_verifies` asserts `expected == CANONICAL_VECTORS[row].out`
  and then checks every one of the 64 `OUT` byte columns against it
  (`blake3_probe.rs:403-421`). This closes the obvious "the chip is only checked against
  the same Rust that produced it" worry.

**Not pinned, and this is the real gap:** `CANONICAL_VECTORS` pins `f(h, m, t, block_len,
flags)`. It says nothing about *how the socket calls it* — which flags, where the two
input digest cells land in `m`, what `t` is, how the 16-word output becomes one digest
cell. Every one of those is a fresh way to be wrong, and rule 9's whole point
(`others/lfm-standing-decisions.md:121-136`) is that a right constant plus a wrong framing
is the normal failure. §5 fixes the framing; this phase must pin it.

### 2.2 Steps

1. **Restore the artifacts into the working tree.** `git show 3b9b8137:<path>` and
   `git show 19ed761b:<path>` for the eight files in §1.5's table, into
   `thoughts/blake3/`. Do not resurrect them from the `.pyc` files — the git blobs are
   authoritative and the audits in `8fec369e` apply to them.
2. **Re-run the first link.** `test_oracle.py` against `official_test_vectors.json` at
   `rounds = 7`. This is the only external anchor in the chain; if it does not run green
   the chain is broken and nothing downstream means anything.
3. **Add a second, independently derived 6-round source.** The Python oracle is in-repo
   and was itself recovered from transcripts, so one source is thin. The best second
   source is **upstream BLAKE3's own `reference_impl/reference_impl.rs`** with its round
   loop parameterised — external code, a minimal and reviewable diff, a different author
   and a different language from the Python oracle. Vendor it under
   `thoughts/blake3/reference-impl/` with the diff visible. Run it at 7 rounds against the
   official vectors (proving the parameterisation is inert), then at 6.
   **Acceptance:** both sources, at `rounds = 6`, reproduce all ten
   `CANONICAL_VECTORS` byte for byte. If they disagree, stop — the vectors in
   `blake3.rs:151-342` are wrong and everything built on them is wrong.
4. **Pin the 7-round claim as a test, not a comment.** Add a test that instantiates the
   parameterised control (`blake3.rs:375-428`) at `rounds = 7` and checks it against the
   **`blake3` crate**, via `blake3::hash()` of a ≤ 64-byte message with
   `h = IV, t = 0, block_len = len, flags = CHUNK_START|CHUNK_END|ROOT`. This is a direct
   external KAT of `f` with no oracle in the middle, and it is what makes the 7-round
   fallback assumption-free. ? INFERRED that the public `hash()` API suffices for a
   single-chunk message — verify against the crate's docs before writing the test rather
   than assuming the flag values.
5. **Pin the socket framing** (depends on §5's decision): once `compress(a, b)` is defined
   in terms of `(h, m, t, block_len, flags)`, add vectors for **that function**, generated
   by both sources, plus a negative control per framing degree of freedom (swap `a`/`b`,
   change the flags byte, move the truncation window). At 7 rounds this KAT can be
   `blake3::hash(a ‖ b)` from the crate directly — a further reason to prefer 7.
6. **Wire the gate into CI or delete the claim.** Today nothing re-derives the vectors.
   Either add a job that runs steps 2–4, or state plainly in `blake3.rs` that the chain is
   a one-time historical derivation. Rule 8 (`a search that ERRORS looks exactly like a
   search that found nothing`) argues for the job.

### 2.3 The 6r-vs-full tradeoff, stated once

| | 6-round | 7-round (standard) |
|---|---|---|
| Reference for the primitive | oracle + reference-impl at `rounds = 6`; **no library, no published vector** | the `blake3` crate directly; published vectors |
| Reference for the 2-to-1 socket | must be generated by the same two sources | `blake3::hash(a ‖ b)` — a library call |
| Security | assumption **A6R**, named and unratified (`blake3.rs:40-42`) | standard BLAKE3; no new assumption |
| Interop | none — nothing else computes it | bit-compatible with BLAKE3 parent merges |
| Cost per compression | 4,946 base-equiv ✓ MEASURED | ≈ 5,714 ? INFERRED (§7) |
| Cost on the epoch column | 2.752 B ✓ MEASURED | ≈ 2.902 B ? INFERRED (+5.5%) |

---

## 3. Phase 2 — make BLAKE3 a first-class selectable hasher

**Goal:** `lfm_prove_with_hasher(program, artifacts, arenas, options, HasherKind::Blake3)`
proves, and the matching verify accepts. **Effort:** L (the chip re-expression is the
bulk). **Risk:** MEDIUM. **Depends on:** Phase 1 and §5's mapping decision.

### 3.1 Two routes, and the recommendation

**Route A — host BLAKE3 *behind* the frozen `LFM_HASH` socket. ★ RECOMMENDED.**

Add `HasherKind::Blake3` and give `LFM_HASH` a BLAKE3 layout the same way Poseidon got
one: keep `PREP_WIDTH = 11` and the frozen 28-column shared value prefix, append the
BLAKE3 witness columns after it, and reuse `blake3_chip`'s mixing core.

What this costs:
- `hash::bus_interactions()` gains a `hasher` parameter (3 call sites: `airs.rs:189`,
  `airs.rs:429`, and `lfm_chip_census_with_hasher`). Under `Blake3` it returns the 6
  `LfmMem` tuples **plus** the 1,248 `BITWISE` lookups. Aux columns go from 3 to ≈ 627.
- `chips::hash` grows a `blake3_cols` module and an `eval_blake3` arm, sharing
  `blake3_chip::run_flow`/`WireFlow`/`ValueFlow` so the single-dataflow rule survives
  (`blake3_chip.rs:63-70` — this property is why the sender list and the witness cannot
  drift, and it is worth preserving on sight).
- The `LFM_HASH` chip's constraint count and degree change; degree stays 3 (the BLAKE3
  chip is already degree 3, `blake3_probe.rs:361`).

What this **buys**, and it is the decisive argument:
- `NUM_LFM_CHIPS` stays 14. `PREP_WIDTH` stays 11. **No root moves and no program digest
  moves** (`airs.rs:375-380`, `statement.rs:49-53`). The six registry entries survive
  untouched; only programs that actually opt into `HasherKind::Blake3` get a different
  identity, and after §4 they get it *deliberately*.
- The frozen `LFM_HASH` bus contract — 2 cells in, 1 cell out — is honoured, so every
  existing `edsl::merkle_walk` / `SpongeVar` caller works unchanged. That is exactly what
  `hash.rs:1-8` promises the swap surface is for.
- `blake3_chip.rs`/`blake3_probe.rs` stay as the measurement probe and the standalone
  falsification harness. Their 4,946 number remains a real, separately-proved datum.

**Route B — register `LFM_BLAKE3` as chip class 15.**

`NUM_LFM_CHIPS` 14 → 15, a new `LFM_CHIP_NAMES` entry, a new `LfmAirs` field, a new
`LfmTraces` field, a new instruction and a compiler lowering. Every registered program's
digest moves; all six entries must be regenerated and re-blessed; every attestation that
folded an old `program_id` is invalidated. Every LFM proof — including programs with no
BLAKE3 at all — carries a padded `LFM_BLAKE3` instance, which is the fixed-machine
principle working as designed (`airs.rs:34-49`) but is still 4 rows × 3,056 columns of
nothing.

Route B is only necessary if a single proof must use BLAKE3 **and** a different
`LFM_HASH` permutation simultaneously. Nothing in the roadmap wants that.

**Recommendation: Route A.** Route B's only advantage is that `blake3_chip.rs` could be
registered nearly as-is; Route A's I/O re-expression (from 7-in/4-out syscall-shaped words
to `LFM_HASH`'s 3-in/3-out) is real work, but it is the *same kind* of work
`blake3_chip.rs` already did once when it re-expressed #903's `Ecall`/`Memw` side onto
`LfmMem` — and the header documenting that swap (`blake3_chip.rs:13-32`) is a ready-made
template for doing it again.

### 3.2 The `LfmHasher` contract problem — read this before writing code

This is the sharpest technical issue in the whole plan and it is easy to miss.

`LfmHasher::permute` is typed `[FE; 12] -> [FE; 12]` — **arbitrary** Goldilocks elements.
BLAKE3 operates on 32-bit words. A Goldilocks felt is up to ~64 bits. So a BLAKE3
`permute` cannot honour that signature on its whole domain without deciding how a 64-bit
felt becomes BLAKE3 input, and the naive answer is unsound: `Σ byteₖ·256ᵏ = v` over the
field does **not** pin the byte string, because `v` and `v + p` both satisfy it — so
without a `< p` argument the prover chooses what gets absorbed and Fiat–Shamir breaks.
That is not my analysis; it is recorded verbatim at
`others/lfm-hash-matrix-scope.md:1440-1452`, and it is why `felt_be_halves` routes through
`bit_dec`, whose contract enforces canonicity (`transcript_replay.rs:735-736`). ✓ VERIFIED.

Two ways out:

- **(i) u32 lanes — restrict the domain. ★ RECOMMENDED.** Every state felt carries a
  `u32`. The state is 12 × 32 = 384 bits = 48 bytes, which fits **one** 64-byte BLAKE3
  block with room to spare (`m[0..12] = state`, `m[12..16] = 0`) — so one permutation is
  one compression, and a digest cell is 4 × `u32` = 128 bits, exactly the machine's
  declared "128-bit target" (`word.rs:1-9`). The map `u32 → FE` is injective with no
  canonicity argument at all, and the chip's existing byte decomposition already
  range-checks each lane. This is the same four-`u32`-lanes-per-machine-word convention
  `LFM_KECCAK` and `blake3_chip` already use (`layout.rs:96-102`, `blake3_chip.rs:8-11`).
  **Cost:** a program that wants to absorb an arbitrary felt must split it into two `u32`
  halves first — the byteswap gadget, in-machine.
- **(ii) Felt-absorbing with an in-chip canonicity gate.** The chip receives full 64-bit
  felts and decomposes them to bytes inside its own constraints, with a borrow-chain
  `< p` gate per absorbed felt (ESTIMATE ≈ 20 base-equiv per felt, so ≈ 240 per
  permutation). This keeps `permute` total on `[FE; 12]` and deletes the byteswap gadget
  from callers. **Cost:** the 12-felt state is 96 bytes, which does **not** fit one
  64-byte block — either two compressions per permutation or a restructured state.

Route (i) is simpler, sounder-by-construction, and one-compression-per-permutation. Take
it. Then **the trait's contract must be made explicit**: `LfmHasher::permute` becomes
documented as partial for lane-restricted hashers, and the `HasherKind::Blake3` impl must
reject (not silently reduce) an out-of-range lane, so the host and the chip agree on the
domain. Silently reducing is the bug that would make a host-side `assert` pass while the
chip proves something else.

### 3.3 Steps

1. Decide §5 first — the mapping is an input to the layout, not an output.
2. `hash.rs`: add `HasherKind::Blake3`; implement `LfmHasher` for it via a
   `Blake3Permutation` struct; make the partial-domain contract explicit in the trait doc
   and enforce it in the impl.
3. `chips.rs`: `blake3_cols` module appended after `SHARED_VALUE_COLUMNS`; `eval_blake3`
   sharing `blake3_chip`'s `run_flow`; arms in `num_columns` and `num_constraints`;
   `bus_interactions(hasher)`.
4. `airs.rs`, `trace.rs`: thread the hasher through the three call sites and add the
   witness-filling arm beside the Poseidon one (`trace.rs:76-113`, `trace.rs:222`).
5. `executor.rs`: nothing structural — `Instr::Hash` already dispatches through
   `&impl LfmHasher` (`executor.rs:363-399`).
6. Tests, in this order: primitive KAT (Phase 1) → a `blake3_chip_tests`-style constraint
   test mirroring `poseidon_chip_tests.rs` → prove+verify of `TrivialV0` under
   `HasherKind::Blake3` → the five tamper-rejection analogues from `blake3_probe.rs`
   (rule 2: an execute-only test proves nothing about the chip).

---

## 4. Phase 3 — bind the hasher into `lfm_program_id` and the registry

**Goal:** a BLAKE3-backed machine has a distinct, pinned program digest, and `lfm_verify`
reads the hasher from the registry entry instead of defaulting.
**Effort:** S. **Risk:** LOW mechanically, but this is the **soundness-gating** step.
**Do it BEFORE Phase 2 lands**, not after — see §1.4.

Steps:

1. Give `HasherKind` a stable `u8` discriminant with an explicit `as_tag()`, so the wire
   value never follows enum declaration order.
2. `statement.rs`: fold the tag into `lfm_program_id`'s preimage — after
   `LFM_PRESET_TAG`, before the per-slot loop (`statement.rs:45-55`). This moves all six
   existing digests **once**, which is a deliberate re-blessing and must be called out in
   the PR body.
3. `registry.rs`: add `hasher: HasherKind` to `LfmRegistryEntry` and `LfmArtifacts`; thread
   it through `build_artifacts`; regenerate via `compute_lfm_registry`.
4. `proof.rs`: `lfm_verify` passes `entry.hasher` to `verify_against_with_hasher` instead
   of `verify_against`'s `HasherKind::default()` (`proof.rs:141-149`, `172-181`).
5. Add the test that makes it real: a proof produced under one hasher must be **rejected**
   when verified under an entry naming another — and, per the honest-control rule, an
   accompanying test that the matched pair still **verifies**. A rejection test alone
   passes just as well if the fix rejects everything.
6. While in `compute_lfm_registry`: it never calls `validate()` (finding F3.2,
   `compute_lfm_registry.rs:28-66`). Same file, same PR, one line — take it.

**Note on Route A's interaction with this phase.** Under Route A the roots do *not* move
with the hasher, so after step 2 the hasher tag is the *only* thing distinguishing a
BLAKE3 machine's digest from a Test machine's. That makes step 2 load-bearing rather than
belt-and-braces, and it is the reason it cannot be deferred.

---

## 5. Phase 4 — the digest→felt mapping decision

**Goal:** one written, signed decision. **Effort:** XS to write, but it gates Phases 1.5
and 2. **Risk:** HIGH if got wrong, and it is not recoverable by testing.

The problem, stated exactly as the measurement report leaves it
(`others/lfm-hash-matrix-scope.md:1454-1460`, ✓ VERIFIED): *"A blake output word is 32
bits, so a felt built from 8 output bytes is a 64-bit value reduced mod `p` and the map is
not injective. How a blake digest becomes felts — truncate to four `u32`s, reduce,
domain-separate — changes the security argument, the digest width, and the token count."*

There are two directions and they are **not** symmetric:

**Input side (felt → BLAKE3).** Covered by §3.2. Under option (i) it is the identity on
`u32`-valued lanes and needs no argument. Under option (ii) it needs a per-felt `< p`
gate, and omitting that gate breaks Fiat–Shamir.

**Output side (BLAKE3 → felt).** BLAKE3 emits 8 `u32` words of chaining value. The
options:

| option | digest | injective? | notes |
|---|---|---|---|
| **A. Truncate to 4 `u32`s, one per lane** ★ | 128 bits, 1 cell | yes, on the truncated image | matches `word.rs`'s declared 128-bit target and `HASH_DIGEST_FELTS = 4`; no reduction anywhere; collision resistance is 64 bits |
| B. All 8 `u32`s, 2 cells | 256 bits, 2 cells | yes | breaks the frozen `LFM_HASH` contract (1 cell out) and doubles the token count; this is keccak's shape, i.e. socket 1's |
| C. Pack 8 output bytes into one felt, reduce mod `p` | 256 bits, 4 cells | **no** | the non-injective case the report flags; needs a rejection or canonicalisation argument; avoid |
| D. 4 `u32`s + domain separation in `flags` | 128 bits, 1 cell | yes | A, plus a distinct flag byte per use (leaf / parent / sponge) |

**Recommendation: D — option A with domain separation.** It preserves the frozen 1-cell
digest contract, is injective without any reduction argument, matches the machine's own
stated security target, and the domain byte costs nothing (it is a constant in the
constraints, not a column). The 64-bit collision bound is the honest consequence of a
128-bit digest and must be written down next to the decision; if the ecosystem target is
128-bit *collision* resistance rather than 128-bit *security level*, option B and a
2-cell digest is the answer and the `LFM_HASH` contract has to be reopened. **That is the
question to put to the user, and it is the only one in this phase.**

---

## 6. Phase 5 — the end goal, and the cost reconciliation

**This is where the brief's framing needs correcting, so the reconciliation comes first.**

### 6.1 Reconciling 2.75 B, 1.1 B, and 4,946

All three numbers are correct and they are about different things. ✓ VERIFIED against
`blake3_probe.rs:327-356` and `blake3_probe.rs:549-733`, and the derivation in
`others/lfm-hash-matrix-scope.md:1400-1460`.

- **4,946 base-field-equivalent cells per compression** — MEASURED, standalone, via a real
  prove+verify against the production `BITWISE` table. `MAIN_COLUMNS + 3 × aux =
  3,056 + 3 × 630`. This is a property of the chip and is solid.
- **2.752 B cells per epoch verify** — the **role 1** column: what the epoch verifier would
  cost *if the inner proofs it verifies were BLAKE3-committed instead of keccak-committed*.
  It is `hash + residue + BITWISE` at P = 192,000 permutations, where the hash term is
  195,593 × 4,946 ≈ 967 M and the rest is the measured keccak-shaped residue. Against
  keccak's measured 11.166 B that is 4.06×.
- **≈ 1.097 B** — the same column for an **unbuilt** felt-absorbing variant. The delta is
  almost entirely the **byteswap gadget**: the felt→byte serialization that exists only
  because the chip consumes `u32` lanes and `felt_be_halves` is what produces them. That
  gadget is 95.84% of the residue in padding-aware cells, and it is **upstream** of the
  chip's input format, so hosting the chip cannot delete it
  (`others/lfm-agent-status.log:223`, ✓ VERIFIED as the recorded finding). The variant
  adds ≈ 156 cells/compression (an ESTIMATE: 8 absorbed felts × ~20 for the canonicity
  gate) and deletes the gadget, landing at 1.097 B / 10.2×.
- **What none of these measure: role 2.** `TestPermutation` costs 37 base-equiv per
  permutation (`others/lfm-hash-matrix-scope.md:106-121`), and **the wrap performs zero of
  them** (§1.1). Replacing it with BLAKE3 multiplies zero by ~135.

**So: swapping `TestPermutation → BLAKE3` does not move the wrap's cost at all.** The
wrap's 11.17 B is keccak, hosted, verifying keccak-committed inner proofs, and it stays
11.17 B. Anyone reading "we proved the wrap with a real hash" should understand that the
wrap's *own* hashing work was, and remains, keccak.

### 6.2 What the end goal should be instead — three rungs

**E1 — socket 2 becomes real and bound.** Phases 1–4 complete. `TrivialV0` and `FriToyV0`
prove and verify under `HasherKind::Blake3`, with distinct registry entries and distinct
program digests. The F3.4 disclosure is retired: `FriToyV0`'s Merkle authentication and
Fiat–Shamir sponge become cryptographically meaningful. **This is achievable now and is
what I would ship.**

**E1.5 — a BLAKE3 Merkle fixture wrap.** Replace the Milestone-C fixture tree that
`edsl::merkle_walk` authenticates with a real BLAKE3 tree, and prove+verify the resulting
program. This exercises the whole path end to end — executor, trace, AIR, registry, and
the §5 mapping — under a real hash, with no production migration. **This is the honest
"prove a wrap with a real hash" milestone**, and it is a much better demo than E1.

**E2 — the epoch verifier over BLAKE3-committed inner proofs.** This is where 2.752 B (or
1.097 B) lives. It requires the *production* RV64 prover's Merkle and Fiat–Shamir hash to
be BLAKE3 — i.e. the ecosystem hash decision, landed, in `stark/`. Out of LFM's control
and far larger than everything above combined. Scope it separately; do not fold it into
this plan's deliverable.

### 6.3 What changes in the wrap tests

- `wrap_tests.rs:26-28` already says the module "cannot see the hash". After E1 that
  remains true and should be **strengthened**, not deleted: add an assertion that the
  epoch program's `LFM_HASH` group is empty, so the fact is enforced rather than narrated.
  If a future epoch-verifier change starts emitting `Instr::Hash`, that assertion is what
  makes it visible.
- `the_wrap_census_at_blowup_8` and `the_blake_column_and_the_residue_split` are both
  `#[ignore]`d and stay so.
- For E1.5, a new `blake3_wrap_tests` module mirroring `wrap_tests`' structure.

### 6.4 Memory and box requirements

✓ VERIFIED from the PR body and the scoping report: a production-shaped wrap (73 queries)
needs 290–350 GiB under keccak and fits a single 124 GiB box under any candidate hash
(blake ≈ 71 GiB, Poseidon ≈ 7 GiB). **Those figures are role 1.** E1 and E1.5 are
ordinary LFM proves at blowup 2 and run wherever the existing wrap tests run; they need no
special box. E2 does, and by then the hash change is what makes it fit.

---

## 7. Phase 6 — the A6R decision

**A6R** is the named, unratified assumption that the 6-round BLAKE3 internal variant is
collision resistant (`blake3.rs:40-42`, from #903's `IMPLEMENTATION.md`). Today it costs
nothing, because `LFM_BLAKE3` is unregistered and unreachable. **The moment BLAKE3 becomes
a selectable hasher with a registry entry, A6R becomes a live soundness surface for every
program that selects it.** That is a protocol decision, not an engineering one, and it
needs a signature.

**What is already on the record** (✓ VERIFIED, `git show a7a8bdd5`, `783c5a95`):

> *External review (2026-08).* The round-count choice was reviewed with external
> symmetric-cryptography experts consulted by the project: removing *one* round
> (7 → 6) was judged comfortable; removing *two* (7 → 5) was explicitly not.
> Accordingly, 6 rounds is the endorsed floor. Variants below 6 rounds are not
> formally ruled out, but they are not available on the project's own authority:
> adopting one would require the external experts to study the reduced-round
> margin specifically — a dedicated cryptanalytic review, not an engineering or
> configuration decision.

(Corrected 2026-08-10: an earlier revision of this plan paraphrased the last sentence as
"variants below 6 rounds are out of scope and MUST NOT be instantiated" — a strengthening
of the source. The text above is the actual wording; see A6R-signoff.md §1.1.)

and, in the same commit, the alternative:

> *The assumption-free alternative.* The chip design is round-parameterised; a 7-round
> instantiation (standard BLAKE3 compression, bit-compatible with official parent-node
> merges) costs roughly 10–12% more per merge end-to-end and requires no assumption beyond
> standard BLAKE3.

**What that costs in this machine.** ? INFERRED — arithmetic over the chip's own verified
budget (`blake3_probe.rs:336-352`), shown so it can be rechecked. Going 6 → 7 rounds means
`NUM_G` 48 → 56:

```
main columns   3,056 + 8 G-blocks x 60 cells                    = 3,536
BITWISE XOR    56 G x 16 + 64 feed-forward   (was 48x16+64=832)  =   960
shift halfwords 56 G x 8                     (was 48x8   =384)   =   448
message bytes                                (unchanged)         =    32
LfmMem tokens                                (unchanged)         =    11
interactions   960 + 448 + 32 + 11           (was 1,259)         = 1,451
aux            ceil(1451 / 2)                (was 630)           =   726
base-equiv     3,536 + 3 x 726               (was 4,946)         = 5,714   (+15.5%)

epoch column   2.752 B - 967 M + (195,593 x 5,714 = 1.118 B)     = 2.902 B (+5.5%)
               vs keccak 11.166 B                                = 3.85x   (was 4.06x)
```

The +15.5%-per-compression figure is consistent with the spec's "10–12% per merge
end-to-end" once the non-hash terms are included, which is a useful cross-check on both.

**The decision to sign, in one line:** *A6R buys roughly 5.5% of the epoch column. Is a
named, unratified, non-interoperable assumption worth 5.5%?*

**My recommendation: no — build round-parameterised, instantiate 7-round first.** Reasons,
in order of weight: (1) the reference problem dissolves — the `blake3` crate becomes a
direct external KAT for both the primitive *and* the 2-to-1 socket framing, satisfying
rule 9 in its intended form rather than one step removed; (2) no assumption to sign,
ratify, or re-litigate at audit; (3) bit-compatibility with published BLAKE3 parent merges
is worth something on its own; (4) 5.5% is inside the noise of the decisions still open
above it (the felt-absorbing variant is a 2.5× lever on the same column — an order of
magnitude more leverage than the round count). Keep 6-round as a measured variant behind
`BLAKE3_ROUNDS`, gated on an explicit signed A6R, and switch to it if and when the 5.5%
matters. The user's stated tolerance — "as long as it works with Blake 6r, or blake, it's
fine" — permits this.

**If the user signs A6R anyway**, the record needs: the assumption named in
`SOUNDNESS.md` (not only in `blake3.rs`), the registry entry's doc comment saying which
hash it rests on, and a note that sub-6-round variants are **not available on the
project's own authority** and would need a dedicated external cryptanalytic review
(the spec's actual wording — see §7's quote).

---

## 8. Phases, effort, risk, order

| # | Phase | Effort | Risk | Depends on |
|---|---|---|---|---|
| 4 | §5 digest→felt mapping decision | XS (a decision) | **HIGH if wrong** | user sign-off |
| 6 | §7 A6R decision / round count | XS (a decision) | **HIGH if wrong** | user sign-off |
| 1 | §2 reference + KATs | S, 1–2 d | LOW | git restore |
| 3 | §4 bind hasher into digest + registry | S, 1–2 d | LOW mechanically, soundness-gating | — |
| 2 | §3 `HasherKind::Blake3`, Route A | **L, 1–2 w** | MEDIUM | 1, 3, 4, 6 |
| 5a | §6.2 E1 — registered BLAKE3 toy programs | S | LOW | 2 |
| 5b | §6.2 E1.5 — BLAKE3 Merkle fixture wrap | M | MEDIUM | 5a |
| — | §6.2 E2 — BLAKE3-committed inner proofs | **XL** | — | the ecosystem hash decision |

**Ordering constraint worth flagging:** Phase 3 (binding) is listed after Phase 1 but must
**land before** Phase 2, because Route A deliberately keeps the roots hasher-independent —
which makes the digest tag the only separator between a BLAKE3 machine and a Test machine.
Adding the third hasher first would create exactly the collision F3.3 warns about.

**Two decisions block the largest phase.** Phases 4 and 6 are one paragraph of writing
each and gate ~2 weeks of work. Get them signed before starting Phase 2.

---

## 9. Start here

**Step 1, today, ~30 minutes, no decisions required:**

```
git show 3b9b8137:thoughts/blake3/blake3-oracle/blake3_ref.py
git show 3b9b8137:thoughts/blake3/blake3-oracle/ORACLE.md
git show 3b9b8137:thoughts/blake3/blake3-oracle/test_oracle.py
git show 3b9b8137:thoughts/blake3/blake3-chip/DESIGN.md
git show 3b9b8137:thoughts/blake3/blake3-chip/z3_blake_verify.py
git show 19ed761b:thoughts/blake3/blake3-oracle/official_test_vectors.json
git show 19ed761b:thoughts/blake3/blake3-oracle/canonical_6round_vectors.json
git show 19ed761b:thoughts/blake3/ground-truth/Cargo.toml
git show 19ed761b:thoughts/blake3/ground-truth/src/main.rs
```

into `thoughts/blake3/`, then run `test_oracle.py` against `official_test_vectors.json` at
`rounds = 7`.

That single run either confirms or breaks the only external anchor the entire 6-round
chain hangs from. Everything else in this plan — the chip, the 4,946 measurement, the
2.752 B column, A6R itself — is downstream of it. If it does not go green, nothing below
it is worth starting.

**Step 2, in parallel, requires no code:** put §5's mapping question and §7's round-count
question to the user as two yes/no decisions. They gate the largest phase and each is one
paragraph.
