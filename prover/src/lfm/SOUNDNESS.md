# LFM write-once memory: the soundness argument (Phase 0b)

Status: **for review**. This is the document a reviewer is invited to reject. If the argument
below does not convince, the fallback is the VM's timestamped memory argument (known-sound,
priced: 37% of MEMW's columns are timestamps plus 8 `<`-lookups per row) or in-circuit
well-formedness checks (uniqueness, per-selector booleanity, mult bounds — a material repricing).

## 1. Setting

An LFM **program** is a set of *instruction column groups*: per-chip preprocessed matrices holding
addresses, opcode selectors and multiplicities. They are committed once (interpolate → LDE →
row-pair Merkle) and their roots are pinned in `LFM_REGISTRY`, a drift-tested Rust constant table.
At verify time the roots are resolved from the registry and the framework rejects any proof whose
preprocessed commitment differs (`verifier.rs` equality check; on the prover side a mismatching
trace fails with `PrecomputedCommitmentMismatch`). The **main** (witness) columns carry values
only.

Memory is not a table. A cell is a word `(v0..v3)` at an address `a`; the producing instruction's
chip **sends** the token `(a, v0, v1, v2, v3)` on the `LfmMem` bus with multiplicity `mult(a)` — a
*preprocessed* column — and each consuming instruction's chip **receives** the same token once,
with multiplicity gated by its (also preprocessed) `is_real`/selector columns. The LogUp argument
the whole prover already runs enforces, per bus, with soundness error `O(D/|E|)` over the
challenges `z, α` (`E` = the degree-3 extension, `D` = total interaction count):

> **(B) Balance.** The multiset of sent tokens with multiplicity equals the multiset of received
> tokens with multiplicity.

### 1.1 The framework premises the machine inherits

(B) is not a fact about the LFM prover; it is an assumption about the *outer* verifier that checks
the LFM proof (`lfm/proof.rs` → `Verifier::multi_verify_views`), which is ordinary, unmodified
framework code. Four of its checks are load-bearing here and the machine defends none of them
independently: **per-column opening-width pinning** (`trace_opening_widths_well_formed`), which
pins each query opening's precomputed/main/aux split against the AIR rather than only their sum;
**`ood_blocks_well_formed`**, which derives the OOD table's shape from the AIR; **the
precomputed-root equality check** cited above, which is what delivers §2's registry premise; and
**the composition part-count check**, which fixes `num_composition_parts` from the AIR's degree
bound instead of reading it off the proof.

The width pin is the one worth spelling out, because it is what makes (B) hold with error
`O(D/|E|)` rather than not at all. Every LFM chip is `with_preprocessed` with a non-empty aux trace
whose root is absorbed *after* the LogUp challenges `z, α`. If only the sum of the opening widths is
pinned, a prover may declare one value column into the aux group and choose it with the challenges
in hand; the bus check is a single aggregate scalar equality per proof, so one free extension
element solves it for an arbitrary perturbation — and (B) is gone while §2's whole custody chain
still passes, because that chain binds the *addressing* columns and the break is in the value
columns' commitment timing. **The machine's soundness is therefore only as good as the base tree's
version of these checks**: on a base predating one of them the theorem of §3 has no (B) to consume
and a wrap proof certifies nothing, no matter how much of §2–§6 holds. Item 8 of §7 is the
reviewer's version of this.

## 2. What is vouched, and by whom

| obligation | enforced by | mechanism |
|---|---|---|
| per-op algebra (`out = a·b + c`, …) | AIR | transition constraints on value columns |
| bit booleanity where a witness bit exists | AIR | degree-2 constraint |
| selector sum-boolean per ALU row | AIR | degree-2 constraint (belt over suspenders) |
| token balance (B) | AIR | LogUp, framework-emitted |
| **(U) uniqueness** — no address written twice | registrar | admission validator check 1 |
| **(A) acyclicity** — operand addr < destination addr | registrar | check 2 (dense emission order gives it by construction; re-checked) |
| **(M) mult-equality** — `mult(a)` = number of emitted reads of `a` | registrar | check 3 |
| **(S) selector one-hot-ness** | registrar | check 4 |
| **(P) padding rows all-zero** (mult = 0, is_real = 0) | registrar | check 5 |
| arena discipline | registrar | check 6 |

"Registrar" means: the release-mode validator (`lfm/validator.rs`) ran on this exact program
before its digest entered `LFM_REGISTRY`, and the proof's preprocessed roots equal the registry's.
The chain of custody is: validator ⇒ digest ⇒ registry constant ⇒ drift tests on every PR ⇒
root-equality check at prove and verify time. **There is no runtime off-switch and there must
never be one; the registry check *is* the soundness argument's first premise.**

This is the industry-standard trust shape for this machine class: SP1 v4 relies on the same
premise but checks it only in dev builds (its validator omits the double-write check entirely);
Risc0 makes it structural (write destinations are program text, so uniqueness is syntactic). We
run the full checklist, in release, at admission — strictly more than either reference.

## 3. The claim

> **Theorem.** Assume (U), (A), (M), (S), (P) hold for the program (registrar) and (B) holds for
> the proof (LogUp). Then in any accepted execution, every read of address `a` observes the unique
> value written at `a`.

**Argument.** By (U) each address has at most one producing instruction, so "the write at `a`" is
well-defined; let `W(a) = (a, w0..w3)` be its token, sent with multiplicity `mult(a)` (the
preprocessed column — the prover cannot vary it). By (M), `mult(a)` equals the number of program
reads of `a`. By (P), padding rows contribute no tokens (their multiplicities are preprocessed
zeros).

Consider the multiset equation (B) on `LfmMem`. Every receive token is generated by some real
instruction row whose address operand columns are preprocessed, so the *addresses and counts* of
all receives are program text; only the value lanes are witness. Fix an address `a`. The sends at
address `a` are exactly `mult(a)` copies of `W(a)` (one writer, (U)). The receives at address `a`
are exactly the program's reads of `a` — `mult(a)` of them, by (M) — each carrying the value lanes
the reading chip's row exhibits. Balance of the full multiset then forces the sub-multisets at
each address to match (tokens include the address, and the fingerprint separates distinct tuples
except with the LogUp soundness error), so each of the `mult(a)` receive tokens equals `W(a)`:
every read observes the written value.

Two degenerate cases are closed by the remaining premises. If a value could feed its own
producing row (`a := f(a)`), the send and receive would cancel *within* the row for **any** value
— balance holds vacuously and the value is unconstrained. (A) excludes this: every operand
address is strictly below its destination, so the read-token's address refers to an
earlier-produced cell, and the dataflow relation is a DAG; induction over addresses in ascending
order grounds every value in constants, hints, or hash outputs. If selectors were not one-hot, one
row could emit tokens under two op semantics at once; (S) excludes it beyond the in-AIR
sum-boolean.

## 4. What the AIR must still get right (per-chip obligations)

The argument above reduces chip soundness to: *each chip's constraints must force the value lanes
of every token it sends to be the correct function of the value lanes of the tokens it receives,
on every row where its (preprocessed) multiplicities are nonzero.* Concretely: base ops constrain
lane 0 and send `(a, out, 0, 0, 0)` with constant-zero high lanes in the tuple (a base cell
cannot smuggle extension lanes); ext ops likewise pin lane 3 ≡ 0; `MulBase` additionally
constrains the shared B-columns to zero on its rows so the received token matches a base writer's;
`BitDec`'s canonicity gadget (`G/Z/GINV`) forces the 64 bit columns to recompose to the *canonical*
representative — without it, bits summing to `v + p` would satisfy the linear recomposition and
two distinct bit-vectors could both "be" `v`.

## 5. Arenas

`Hint` rows send unconstrained words into memory (their chip has no constraints by design). The
**arena rule** restores soundness at the program level: every arena-sourced value must be
transitively authenticated by a hash the machine itself performs (Merkle openings are absorbed
into hashed paths; anything transcript-derived is never hinted). This is a *program-review*
obligation, enforced at emitter review, exactly like the reference systems' hint discipline — the
machine-level theorem above is indifferent to hint values; it only guarantees reads see what was
hinted.

## 6. Transcript replay (R1d)

`edsl::TranscriptReplay` reproduces the production `DefaultTranscript` inside the machine. Three
things about it are worth a reviewer's attention.

### 6.1 Absorbed data is hinted, and that is correct

The replay absorbs arena-supplied words. That is not a breach of the arena rule (§5). The rule bans
*hinting a challenge*; it does not ban hinting the data a challenge is derived FROM — in
Fiat–Shamir that data is precisely the untrusted proof material, and binding it is the entire
point. Every challenge the machine uses is computed by `LFM_KECCAK` rows from the absorbed
segment, never read from an arena. The obligation that remains is the ordinary one: whatever is
absorbed must also be the thing the rest of the program checks against.

### 6.2 The canonicity guard is a constraint, not a witness

`sample_field_element` must reject candidates ≥ `p`. Since `p = (2^32 − 1)·2^32 + 1`, a candidate
`hi·2^32 + lo` with canonical `u32` halves is out of range **iff** `hi = 2^32 − 1 ∧ lo ≠ 0` — the
same predicate `BitDec` already uses for 64-bit canonicity (§4), over the same split.

The guard emits one instruction: `div(lo, (2^32 − 1) − hi)`. `LFM_BALU` constrains division as
`SEL_DIV·(B·OUT − A) = 0`, so with `B = 0` it reads `A = 0` and leaves `OUT` free. The division is
therefore provable exactly when `hi ≠ 2^32 − 1` or `lo = 0`. Nothing is hinted and nothing needs
verifying — it is the same assert-via-division mechanism `assert_eq` is built from. (An earlier
plan used an `is_zero` gadget with a hinted-and-verified inverse; the division subsumes it.)

`machine_tests::canonicity_guard_rejects_an_out_of_range_candidate_in_the_proof` exhibits a
coherent forgery at candidate `p` — every bus balances and the mul-add's own constraint is
satisfied — and confirms it is rejected; neutralising `emit_base(3, …)` makes that forgery
ACCEPTED, which is what pins the guard on this one constraint.

### 6.3 Zero rejection: a completeness restriction, and why it is not a parameter

The production sampler *loops* on an out-of-range candidate. The number of candidates a draw
consumes is therefore data-dependent, and so is every later draw's position in the output buffer.
A straight-line machine has exactly one shape, so it cannot follow that. The emitted program
encodes the **no-rejection schedule** and is unprovable for any transcript that ever rejects.

This costs completeness only, never soundness. The emitted relation is a strict subset of the real
one: challenge values are pinned by constraints to the no-rejection schedule, so a transcript that
would have rejected yields *no* LFM proof rather than a wrong one. An honest prover sees it as a
loud `LfmExecError::DivByZero`, not a silent divergence.

The bound. A candidate is uniform over `2^64` values and `2^64 − p = 2^32 − 1` of them are out of
range, so `q = (2^32 − 1)/2^64 ≈ 2^−32` per candidate. Only `sample_field_element` draws are
exposed: `sample_u64` at a power-of-two bound has `threshold = 0`, so it accepts its first
candidate unconditionally and contributes nothing. Every verifier challenge is a cubic-extension
element, i.e. three independent base draws, so with `E` extension draws the union bound gives

> `P[the program cannot prove this proof] ≤ 3E · (2^32 − 1)/2^64`

(`reject_probability_per_proof` in `transcript_replay.rs`, which takes the BASE draw count `3E`).

The verified per-proof draw schedule, for a multi-proof over `T` tables, is

> `E = 2` (LogUp `z, α` — shared transcript, drawn before the per-table forks)
> `  + 2` (bus-balance replay, on a forked transcript)
> `  + Σ_t (3 + L_t)` — per table `β`, `z_OOD`, `γ`, then `L_t` FRI fold challenges,
>   with `L_t = max(log2(trace_length_t) − 7, 0)`, **independent of the blowup factor**.

`β` and `γ` are one draw each no matter how many terms they batch (both expand to powers), which
is what keeps `E` small. At `T = 24` with tables at their row cap (`L_t = 12`), `E = 364`, so
`3E = 1,092` base candidates and `P ≈ 2.5·10^−7`. At a larger `T ≈ 60`, `E = 904` and
`P ≈ 6.3·10^−7`.

`T = 24` is **measured, not assumed**: reading a real two-epoch continuation proof
(`machine_tests::arena_filler_reads_real_committed_roots`) gives 24 sub-proofs for an
intermediate epoch and 25 for the final one, the extra being HALT. It was an honest hedge when
this section was written; it no longer needs to be.

**State it as `< 10^−6` per proof at production shapes**, growing by `≈ 1.05·10^−8` per additional
table — each table contributes `3 + L_t ≈ 15` extension draws, so the per-table increment is 15×
the `≈ 7·10^−10` an individual extension draw costs. (Do not quote the per-draw figure as the
per-table one; the two differ by that factor of 15.)

Headroom is large: 1% failure needs `≈ 4.3·10^7` base candidates (`≈ 2^25.4`) and 50% needs `2^31`,
four-plus orders of magnitude beyond any realistic verifier. Every figure above is pinned by
`machine_tests::zero_rejection_completeness_bound` rather than merely asserted here.

**One host/machine divergence worth recording.** The verifier does not check that the
prover-supplied `trace_length` is a power of two. A malicious proof could therefore hand
`sample_u64` a non-power-of-two bound, making `threshold` nonzero and putting even a query draw on
the rejection path — the one circumstance in which a `u64` draw could matter to this bound. Such a
proof is simply unprovable in the machine, which is the safe direction (unprovable = rejected), but
it is a case where the host transcript and the emitted program diverge rather than agree.

**Do not record this as "k-rejection is an emitter parameter later".** It is not. Supporting even
one rejection requires the downstream schedule to branch, which in a straight-line machine means
either a program per rejection pattern (`2^draws` of them, and program identity would become
proof-dependent, breaking the registry premise in §2) or a production transcript change to
constant-consumption sampling. The realistic route, if the bound ever stops being acceptable, is
the latter — make the production sampler consume a fixed number of candidates per draw — and that
is a change to `crypto`, not to this emitter.

Timing note for whoever picks that up: the ecosystem hash migration already has to rebuild the
transcript (a field-native sponge replaces the keccak chain), and constant-consumption sampling is
a design constraint to carry into that rebuild rather than a separate migration. Fixing it there
costs nothing extra and removes this restriction for every future machine; retrofitting it onto the
current transcript would be a second proof-breaking change for no other benefit.

### 6.4 The algebraic transcript: the rebuild §6.3 anticipated, and it removes that restriction

§6.3's closing note said the ecosystem hash migration would have to rebuild the transcript as a
field-native sponge, and that constant-consumption sampling was a design constraint to carry into
that rebuild rather than a separate migration. That rebuild is
`lfm::algebraic_transcript::AlgebraicTranscript`, and the constraint was carried.

For an algebraic configuration, `CANDIDATES_PER_COORDINATE = Some(1)` — **and it is guaranteed
rather than probabilistic.** A squeeze returns four felts that are canonical *by construction*
(they are field elements, not bytes reinterpreted), so the `u64`s carved out of their 32 canonical
bytes are always in range. There is no rejection to schedule around: not "rejects with probability
2^−32", but *cannot reject*.

This is strictly stronger than the BLAKE3 configuration, which needs two candidates because its
digest bytes are arbitrary and a single miss has nowhere to go (`transcript_hash.rs`), and it is
stronger than keccak's `None`, which is the data-dependent loop §6.3 exists to talk about.

**So §6.3's completeness restriction does not apply to an algebraic configuration.** The emitted
program's no-rejection schedule is not a subset of the real relation there — it *is* the real
relation, because the production sampler cannot take the other branch. The `q ≈ 2^−32` per-candidate
bound and the `2^draws` program-explosion argument are both about the byte configurations and
should not be quoted against an algebraic one.

`algebraic_transcript::tests::the_host_transcript_and_the_machine_replay_derive_the_same_challenges`
is the differential that pins host and machine to the same challenge stream, under every tenant.
`sampling_is_constant_consumption` pins the one-cell-per-draw property directly.

### 6.5 Grinding reuses the LEAF domain — a recorded weakening, and why it does not bite

**What the socket separates.** `LFM_HASH` pins a per-mode capacity for `MODE_C` (Merkle parent),
`MODE_T` (transcript step) and `MODE_L` (leaf), so those three are different functions and a row
cannot claim a domain it does not carry (`chips::hash::emit_socket_prefix`; the AIR rejects a row
carrying another mode's capacity).

**What grinding does.** An algebraic configuration's `StarkHash::Transcript` — the hash the
proof-of-work runs on — reuses the **leaf** construction and the leaf domain
(`algebraic_commit::AlgebraicDigest`), rather than taking a fourth domain of its own.

**Why the machine cannot do otherwise, which is the actual reason.** The per-mode capacities are
pinned by the *preprocessed* mode selectors, so the only capacities an emitted program can produce
are the three the chip pins (plus whatever a `MODE_P` row carries as program data). A
grinding-specific domain would need a new mode, i.e. a change to the frozen `LFM_HASH` tuple
contract. Grinding hashes a byte string — `state ‖ nonce`, 40 bytes, five felts — which *is* data,
exactly what a leaf is, so the leaf construction is both the natural reading and the only one the
verifier can emit with an existing mode. This is a constraint, not a preference.

**Why the reuse is not exploitable.** Domain separation matters where a verifier ACCEPTS a digest
the prover supplies, because there the prover chooses which domain's output to present. Grinding
is not that: **the verifier RECOMPUTES the grinding digest itself, over a preimage the protocol
fixes** — the transcript state at that point, concatenated with the nonce — and then tests its
leading zeros. The prover supplies only the nonce, and every other input is already bound. There
is therefore **no substitution surface**: a leaf digest cannot be presented in place of a grinding
digest, whatever it might collide with, because no digest is presented at all.

⚠ Recorded as a weakening nonetheless, because it *is* one relative to the socket's design intent
— three domains carry four uses — and because the argument above depends on grinding staying a
recompute-and-compare check. **If a future change ever has a verifier accept a grinding digest
rather than recompute it, this subsection is the one that has to be revisited**, and at that point
the fix is a fourth mode rather than a fourth constant.

### 6.6 The APPEND CALL BOUNDARY is part of the message

**The rule.** A machine-side transcript replay must reproduce the host's `append_bytes` calls
**one for one** — same number of calls, same split — not merely the concatenation of their bytes.

**Why the byte arm hides it.** A byte transcript absorbs into one flat segment: the sponge sees a
byte stream, so splitting one 64-byte absorb into two 32-byte ones, or coalescing two into one, is
invisible. Emitters written against a byte hash therefore coalesce freely, and nothing in the code
records that the boundary ever mattered. An **algebraic** transcript length-prefixes every
`append_bytes` call, so the boundary is *in* the message: a coalesced replay absorbs a different
message and derives different challenges.

**How it fails.** Not as a wrong answer. The replay's challenge diverges from the host's, the leg
then inverts a difference that should have been non-zero, and the executor reports `DivByZero` at
some address — thousands of instructions from the cause, naming nothing. ✓ VERIFIED: six live
instances were found this way on the migration, in the LFM statement, the global statement and the
per-leg absorbs; each was a coalescing that was correct under BLAKE3.

⚖ ASSESSMENT — **what class this is.** The length prefix is a soundness feature *of the algebraic
transcript*: it makes the absorbed stream unambiguous, so a prover cannot re-split a message to
land on another message's state. A replay that coalesces does not break that — it absorbs a
different, still-unambiguous message — so the defect presents as **completeness** (the wrap program
cannot prove) rather than as a forgery. It is recorded here because the *rule* is what keeps it
that way: the moment a replay is allowed to differ from the host by "the same bytes, different
calls", the transcript's injectivity stops being the thing the two sides agree on.

**The general form, which is what to carry forward.** *When the byte hash makes a distinction
irrelevant, the code stops expressing it, and the algebraic hash needs it back.* ✓ VERIFIED three
instances of that shape: this one; the Fiat–Shamir transcript OBJECT being separable from the
commitment configuration (§6.7, axis 2); and the `LFM_HASH` socket permutation being consulted at
all (§6.7, axis 3).

### 6.7 THE HASH IS NAMED, NEVER IMPLIED — three axes, and two deliberate carve-outs

**The rule.** Every prove, verify, execute and commitment-building call on the block path names the
pin (`hash_pin::BlockStarkHash`, `hash_pin::BlockTranscript` / `block_transcript`,
`hash_pin::BLOCK_HASHER`). None reaches a workspace default alias.

**Three orthogonal axes.** They must agree and nothing in the type system makes them:

1. **The commitment configuration** — what the HOST commits under (`BlockStarkHash`). Reaching
   `stark::config::DefaultStarkHash`, or the `stark::prover::Prover` / `stark::verifier::Verifier`
   aliases (which are `GenericProver`/`GenericVerifier` *at* that default), pins BLAKE3 whatever
   `H` the surrounding code passes.
2. **The Fiat–Shamir transcript OBJECT** — built by the caller and handed to `multi_prove`, so its
   type is not forced to match (1). ⚠ **This is the dangerous axis**: a half-flip is
   self-consistent between prover and verifier and therefore **silent**, where (1) and (3) fail
   loudly.
3. **The `LFM_HASH` socket permutation** — which permutation the MACHINE's `Instr::Hash` rows
   compute, passed per call to `execute` / `lfm_prove_with_hasher`. ★ Under a byte hash the
   emitter's Merkle work lowers to the dedicated KECCAK / `LFM_BLAKE3` chips and emits no
   `Instr::Hash` at all, so this argument is never consulted and a toy permutation is free and
   correct — which is exactly why it stayed unnamed until an algebraic pin made it load-bearing.

✓ VERIFIED: eleven sites were carrying an implied hash and were closed by enumerating every call
site rather than by search; two of them were on axis 3.

⛔ **TWO DELIBERATE CARVE-OUTS. Do not "fix" these — they are pinned KECCAK on every arm.**

- `programs::emit_program_id` and its host counterpart `recursion::program_id_from_digest`. A
  `program_id` identifies a program to CONSUMERS; it is not part of the proof system's commitment
  layer. Following the configured hash would make the attestation join disagree with every host
  consumer of a `program_id`, and the disagreement would surface as a consumer-side compare
  failing rather than as an unprovable program. ✓ VERIFIED both name their hash explicitly.
- `statement::elf_digest` (`prover/src/statement.rs:30`), same class: it names `Keccak256`
  directly. ✓ VERIFIED.

⚠ A naive sweep — "replace every `edsl::keccak256`" — produces a wrong proof at exactly these two
sites, because grinding (`epoch::emit_grinding_check`) shares the same `ByteString` type and DOES
follow the configuration.

**A consequence for widths.** Because the id stays a two-cell keccak digest while a commitment root
becomes one algebraic cell, a schema holding both carries two different root widths. Anything that
counts published words must read `proof_arena::lanes_per_root` for a COMMITMENT and keep the
literal two for the ID. `epoch_tests::the_arena_writer_and_the_machine_reader_agree_on_a_roots_width`
gates the first; the second is stated at `aggregator_tests::WrapPublicLayout`.

### 6.8 A FALSE DOC CLAIM RECRUITS THE NEXT CALLER — treat it as a defect, not a typo

**The rule.** A helper whose doc asserts a capability its body does not have is a defect of the
same class as the code errors above, and is worse than no doc at all: no doc makes a caller read
the body, a wrong doc makes them skip it.

✓ VERIFIED two instances on this migration, both structurally unreachable at the time and both
therefore invisible to every test:

- `edsl::digest_halves` — "Hash-agnostic: it is two `Unpack`s." It indexes `d[1]`, and an
  algebraic `WrapDigest` is ONE cell whose second slot REPEATS the first
  (`WrapDigest::from_cell`), so on that arm it would have returned `lo ‖ lo`: eight plausible
  halves, four of them a duplicate, with nothing to notice. Now carries an emit-time width assert.
- `edsl::parent_stream` — "Hash-agnostic … under either hash", on the one function that calls
  `digest_halves`. Two false claims on a helper pair, which is the ordinary way this spreads.

⚖ ASSESSMENT of severity: neither was reachable, so neither was a live bug — and that is exactly
what makes the class worth naming. Unreachable-but-wrong survives every test suite indefinitely
and is discharged only when a future caller takes the claim at face value, at which point it is a
live bug in someone else's change.

**The cheap instrument.** Grep the hash-adjacent helpers for generality phrases — "hash-agnostic",
"either arm", "both hashes", "on both arms" — and check each against its body. That sweep found
the two above and confirmed six others sound.

★ **What the sound ones have in common is the fix.** `ByteWrapHash::hash_bytes` says "Both hashes
take the SAME packing" and is correct, because it hangs on `ByteWrapHash`, a type whose only two
inhabitants ARE the byte hashes — the claim cannot over-reach because the type will not let it.
`epoch::RootCells::byte_halves` says "correct on both arms" and is correct because its body has an
explicit algebraic branch. `transcript_replay::sample` names the algebraic arm and explains why its
behaviour there is a DEFINITION rather than a fidelity claim. **So the durable remedy is the same
one the width defects want: put the distinction in a TYPE, and the doc cannot lie about it.**

## 7. Reviewer checklist (reject if any fails)

0. For an algebraic configuration: is grinding still a RECOMPUTE-and-compare check (§6.5)? If a
   verifier ever accepts a prover-supplied grinding digest, the leaf-domain reuse stops being
   safe and needs a fourth socket mode.
1. Is the validator actually on the only path into `LFM_REGISTRY`, in release builds, with no
   env-var or feature bypass?
2. Do the drift tests pin the registry on every PR (not merge-queue-only)?
3. Does every chip keep the one sign convention (writes = senders `Column(mult)`, reads =
   receivers `Column(is_real)`, no `Negated` forms)?
4. Are all address/selector/mult columns actually in the preprocessed group of every chip
   (no witness-supplied addressing anywhere)?
5. Does the BitDec canonicity constraint cover the full 64-bit range for `p = 2^64 − 2^32 + 1`
   (top-32-all-ones ⇒ bottom-32-zero)?
6. Is the LogUp soundness error budget (`O(D/|E|)`, `|E| ≈ 2^192`) acceptable at the machine's
   interaction counts (≤ 2^25 per epoch)?
7. Does every `TranscriptReplay` challenge reach the program through machine keccak rows rather
   than an arena, and is the zero-rejection completeness bound (§6.3) acceptable at this
   program's actual draw count?
8. Is the base tree at or past every framework verifier fix the inherited premises of §1.1 name —
   currently per-column opening-width pinning (`trace_opening_widths_well_formed`, #909 /
   `6949ceb9`)? On an older base (B) is not delivered, and nothing below §1.1 can recover it.
9. For an algebraic configuration: does every machine-side transcript replay match the host's
   `append_bytes` calls ONE FOR ONE, not just byte for byte (§6.6)? A coalesced absorb is correct
   under a byte hash and derives a different challenge under an algebraic one.
10. Does every prove / verify / execute call on the block path NAME the pin rather than reach a
    default alias, on all three axes (§6.7)? And are the two keccak carve-outs — `program_id` and
    `elf_digest` — still pinned rather than following the configuration?
11. Does every hash-adjacent helper's doc match its body (§6.8)? A "hash-agnostic" claim on a
    function that indexes a second digest cell is a defect even when nothing reaches it.

## 8. Proven bits at the aggregation-era presets (the η re-tune, applied)

Every security claim in this stack is quoted in PROVEN bits under the
Johnson/proximity-gaps bound (1 − √ρ regime) — never the capacity conjecture.
Two consequences of the adopted η re-tune, both derivation-level (no runtime
parameter moved):

1. **Per-table / per-proof base.** At blowup 4 / 110 queries the presets'
   proven soundness re-derives from ≈ 95.5 bits to ≈ 114–117 bits at the SAME
   query count: the slack parameter η in the Johnson-bound proximity term was
   set conservatively, and re-optimizing it against the actual domain sizes
   tightens bits-per-query with zero prover or verifier cost. The security
   audit of 2026-08-15 carries the derivation; the numbers here are its
   published range.

2. **The batched class-split floor tracks the re-tune 1:1.** The batched
   format's additional loss (≈ 3.3 bits of class split inside the measured
   ≈ 92.2-bit floor against the ≈ 95.5 per-table base) comes from the |D₀|²
   union lift in ε_C — a term in DOMAIN SIZE, independent of η. An additive
   bits penalty that does not contain η moves with the base unchanged:
   post-re-tune the batched floor is ≈ (114–117) − 3.3 ≈ 111–114 proven bits.
   This settles the design review's unresolved point ("does the batched floor
   move with η") in the affirmative, from the ε_C derivation's own structure.

Query headroom above the presets is priced in the aggregation census's query
sweep (blowup 4, terminal 2^8): every program boundary — the wrap's 2^18, the
global wrap's 2^20, the aggregator's 2^21 — holds through q = 121
(≈ 125–128 proven bits post-re-tune); q ≥ 124 additionally requires the
global proof's terminal at 2^8 to keep the global wrap under its boundary.
Raising q is a preset decision with registry consequences (new program
identities), not part of this note.
