# D0 — LFM proof under the machine's native BLAKE3 scheme

**Design record.** Scoping pass, read-only; no builds run.
**Ground:** worktree `/Users/maurofab/workspace/lambda_vm-blake3-impl`, branch
`blake3-real-hash` @ `2a8552f2`. **Date:** 2026-08-12.

**Decision this implements:** the LFM machine's own proof (`lfm_prove`,
`prover/src/lfm/proof.rs`) moves from a standard STARK `MultiProof` over
`DefaultTranscript` + keccak256 Merkle commitments to the machine's **native**
scheme under `HasherKind::Blake3` — LFMC Merkle parents, LFML leaves, LFMT
compress-chain Fiat–Shamir (ratified option B, form B1). Purpose: a future LFM
program verifying an LFM proof recomputes the proof's hashes with the machine's
cheap blake3 chips instead of the hosted keccak family.

Claims are ✓ VERIFIED (read the code, cited) / ? INFERRED / ✗ UNVERIFIED.

> **Provenance note.** Two delegated sweeps (registry/pinned-digest; GPU
> hash-dependence) stalled without returning to the author. Their load-bearing
> claims were re-derived independently against the source; every citation below
> was read directly.

---

## 0. Verdict

The switch is tractable and **does not require a new proof format**. Three
structural facts make it so, and three hazards make the ordering
non-negotiable.

**Why it is tractable.** The transcript is already injectable; the Merkle
backend is pinned in exactly one file with everything beneath it already
generic; and `word::pack_digest` already defines an `LfmWord` → `[u8;32]`
embedding, so the rkyv wire format never moves.

**Why ordering matters.** Three separate places will *silently* stamp a Blake3
label on keccak-derived data. All three type-check. None fails loudly. They
must be guarded **before** any Blake3 commitment path exists, not after.

**The one genuine blocker.** `LFML` hashes exactly four felts. Production LFM
AIRs have arbitrary column counts, and no ratified spec covers a wide leaf.
That is a spec task, not a coding task, and it comes first.

---

## 1. Host-side precedent (Q1)

**There is a reusable host-side LFM-native commitment layer** — not test-local
helpers. It lives in `pub mod fixture` (`prover/src/lfm/mod.rs:33`), i.e. it is
production-visible, not `#[cfg(test)]`.

✓ VERIFIED, `prover/src/lfm/fixture.rs`:

| Role | Function | Location and shape |
|---|---|---|
| **LFMT B1 transcript** | `HostSponge` | `:60-136` — state = one cell (`:61,:80`); `absorb` = `hasher.transcript(state,c)` (`:101-103`); `absorb2` (`:105-108`); `absorb_felts` = leaf-encode then absorb (`:113-116`); `squeeze_cell` outputs *then* advances with `SQ(i)` (`:119-125`); `squeeze_operand` (`:87-94`); `squeeze_ext` = lanes 0–2 (`:127-130`); `squeeze_index(n)` = low `n` bits of lane 0 (`:132-135`) |
| **LFML leaves + LFMC parent** | `host_leaf_hash_pair` | `:145-147` — `hasher.compress(&hasher.leaf(c0), &hasher.leaf(c1))` |
| **LFMC Merkle parents** | `HostTree` | `:157-190` — parents `:170`, `root` `:177`, `open` `:182-189` |

All three are `HasherKind`-parameterised, so they already run under Test /
Poseidon / Blake3. The primitives underneath sit in `blake3_socket.rs`:
`TAG_LFMC:227`, `TAG_LFML:241`, `TAG_LFMT:253`,
`socket_digest_rounds_tagged:294`, `transcript_digest:330`, `leaf_digest:414`,
`lanes_of:431`, `word_of:443`, `Blake3Permutation` impl `:456-520`.

The **guest** side already exists in LFM-native form: `edsl::leaf_hash_pair`
(`:167-171`) and `edsl::merkle_walk` (`:177-190`) operate on one-cell digests,
distinct from the keccak twins (`keccak_merkle_walk:267`,
`KeccakDigest = [Cell;2]:196`). A tower verifier therefore walks **half as many
cells per Merkle level** as the keccak path — the recursion-tower payoff is
already built.

**The limitation, stated by the file itself** (`fixture.rs:9-12`): this is *not*
the production proof format — "`crypto/stark` hardcodes keccak at its Merkle
layer; the measured 26-site migration seam is deliberately not touched here."
That seam has roughly doubled: **58** references to the four backend aliases
inside `crypto/stark/src` (`commitment.rs`, `config.rs`, `prover.rs`,
`verifier.rs`, `gpu_lde.rs`, `fri/mod.rs`, `tests/commitment_tests.rs`).

`HostSponge`/`HostTree` are the right **reference**, not the right
**implementation**: they assume fixed 4-column rows, two-row leaves, fixed
depth.

---

## 2. Prove-path genericity (Q2)

**Injectable today — the transcript.**
`Prover::multi_prove(… transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send) …)`
— `prover.rs:3032-3044`; mirrored at `verifier.rs:1219-1231`. ✓ VERIFIED
`prover.rs` never names `DefaultTranscript` (grep: zero hits). A B1 impl slots
in with no signature change.

**Hardwired today — Merkle backend, FRI layer commitment, node type.**
`crypto/stark/src/config.rs:10-24` pins all three backends to keccak;
`Commitment = [u8;32]` at `:16-17`. `IsStarkProver` (`prover.rs:807-814`)
carries **no** hash parameter, and its default method bodies name
`BatchedMerkleTree<E>` concretely (`:823, :879, :898`).

**But everything beneath the alias layer is already generic** — this is what
makes the migration a parameterization rather than a rewrite:

- `IsMerkleTreeBackend` with associated `Node`/`Data` — `crypto/crypto/src/merkle_tree/traits.rs:8-29`
- `Proof<T: PartialEq + Eq>` — `merkle_tree/proof.rs:22-24`
- `verify_merkle_path_from_leaf_hash<B>` / `verify_merkle_path<B>` — `proof.rs:31-38, 57-65`, already turbofished at `verifier.rs:587, 671, 723`
- the keccak backends are themselves generic over `D: Digest, const NUM_BYTES` — `field_element_vector.rs:98-132, 135-203`

### Does the 128-bit `LfmWord` force a distinct proof format?

**No — and the tower-leg recomputation story survives.**

`pack_digest` (`word.rs:44-50`) maps an `LfmWord` to `[u8;32]` as four canonical
LE u64 lanes; `unpack_digest` is the inverse (`:53-61`). A Blake3 digest cell
has all four lanes `< 2^32` (`word_of`, `blake3_socket.rs:443`), so it embeds
with 16 bytes of zero padding.

Keeping `Node = [u8;32]` leaves `StarkProof`'s commitment fields
(`proof/stark.rs:47, 88, 89, 91, 94, 102, 106`) and the rkyv derives
(`:30-37, 52-59, 73-80, 128-135`) **byte-identical** — no wire-format bump, no
disturbance to the in-place rkyv verify path. Padding costs proof size only:
parent hashing operates on **cells**, never on the padded bytes, so the guest
recomputes exactly `LFMC(cell_l, cell_r)` with no padding in the preimage. The
tower leg is unaffected.

⚠ **Two riders on that choice.**

1. `unpack_digest` reduces mod p and does not bound lanes (`word.rs:52-61`), so
   many distinct 32-byte strings decode to one node — **node malleability**.
   Decode strictly via `lanes_of` (`blake3_socket.rs:431-438`), which rejects
   rather than reduces.
2. `Node = [u8;32]` is precisely what makes a Blake3 backend type-check against
   a keccak GPU kernel — see hazard **H3** (§4).

### Recommended shape

? INFERRED — design proposal, not compile-verified. A `StarkHash` config trait
carrying the three backends plus the node type; add it as a generic parameter
on `IsStarkProver`/`IsStarkVerifier` and on a `GenericProver`/`GenericVerifier`;
keep `pub type Prover<F,E,PI> = GenericProver<F,E,PI,KeccakStarkHash>`. Every
existing RV64 call site resolves unchanged, including the bare
`Prover::multi_prove` at `proof.rs:140`.

A defaulted parameter on the *trait* will not work — `H` would be uninferable at
the call site; it has to ride on the concrete type via the alias.

---

## 3. Transcript mapping (Q3)

`IsTranscript` has five methods (`is_transcript.rs:7-26`); `IsStarkTranscript`
adds `sample_z_ood*` (`:28-90`). Call census ✓ VERIFIED by grep
(prover / verifier): `append_bytes` 4/8, `append_field_element` 2/4,
`sample_field_element` 3/5, `sample_u64` 1/1, `state` 1/1, `sample_z_ood*` 1/1.

| `IsTranscript` method | B1 op | Verdict |
|---|---|---|
| `append_field_element` | `absorb_felts` (`fixture.rs:113-116`) | ✓ direct |
| `sample_u64(bound)` | `squeeze_bits(n)` (`fixture.rs:132-135`) | ✓ **exact** |
| `sample_field_element` | `squeeze_ext` (`fixture.rs:127-130`) | ✓ shape, ⚠ semantics |
| `append_bytes(&[u8])` | — | ⚠ B1 has no byte-level absorb |
| `state() -> [u8;32]` | — | ✗ **no equivalent** |
| `sample_z_ood*` | default body over `sample_field_element` | ✓ inherited |

`sample_u64` maps **exactly**, which is worth stating because it looks like it
should not. The only call is `sample_u64(domain_size >> 1)` with `domain_size` a
power of two (`prover.rs:2132-2134`, `verifier.rs:138-140`). For a power-of-two
bound, `upper_bound.wrapping_neg() % upper_bound == 0`, so the rejection loop at
`default_transcript.rs:136-145` accepts the first candidate and returns its low
`log2` bits — precisely `squeeze_bits`.

### Four items with no B1 equivalent

1. **Grinding / PoW — the hard gap.** `transcript.state()` seeds
   `grinding::generate_nonce` (`prover.rs:2093`) and `is_valid_nonce`
   (`verifier.rs:1668`); both hash Keccak256 unconditionally
   (`grinding.rs:1, 72, 87`). It is **live**: `MIN_PROOF_OPTIONS` sets
   `grinding_factor: 1` (`prover/src/recursion.rs:39-45`). A tower guest
   recomputing keccak PoW defeats the purpose of the switch.
   **Recommend `grinding_factor: 0` for LFM proofs and scoping grinding out
   explicitly**, rather than re-specifying PoW over the LFM hash. Note
   `options.rs:114` asserts `security_bits > grinding_factor` — confirm 0 is
   admissible before relying on it.
2. **Rejection sampling is loop-shaped; the machine cannot hold it.**
   `sample_field_element` for E calls the base sampler three times
   (`extensions_goldilocks.rs:575-581`), each an unbounded `loop`
   (`goldilocks.rs:548-555`). The eDSL fully unrolls — "nothing loop-shaped
   reaches the machine" (TRANSCRIPT.md §1.1, citing `edsl.rs:1-4`). The B1 impl
   must use lane-direct `squeeze_ext`, whose u32 lanes are canonical by
   construction, and must **not** reuse `sample_field_element_from`.
3. ⚠ **Challenge entropy drops to 96 bits.** A squeezed cell is four u32 lanes
   = 128 bits (`word_of`, `blake3_socket.rs:443`); `squeeze_ext` takes lanes
   0–2. `DefaultTranscript` yields three near-full Goldilocks coordinates
   (~192 bits). TRANSCRIPT.md §4.1 analyses the 128-bit state and its ~64-bit
   collision bound but **does not** analyse per-challenge entropy at production
   query counts. Needs a decision before it is a security claim.
4. **`append_bytes` needs an encoding.** `absorb_lfm_statement`
   (`statement.rs:79-89`) feeds raw byte strings — tags, `program_id`, LE
   integers. B1 absorbs cells. A padding-and-length-bound byte→cell convention
   must be specified, not improvised.

Also: TRANSCRIPT.md §4.2(b) warns the maximum squeeze run **is** `NUM_QUERIES`;
at the `Blowup2` preset's 219 queries the run is ~219 (≈7 bits loss). Within the
doc's own `k < 2^16` guidance, but record it rather than assume it.

---

## 4. The three silent-mislabel hazards

These are why the ordering is non-negotiable. All three type-check; none fails
loudly.

### H1 — artifacts stamp Blake3 on keccak roots

✓ VERIFIED `build_artifacts_with_hasher` (`registry.rs:128-171`) computes every
root via `commit_group(g, options)` (`:150`) plus
`keccak_rc::preprocessed_commitment` (`:155`) and
`bitwise::preprocessed_commitment` (`:157`) — all keccak (`commit.rs:56` →
`commit_columns:21-46` → `commit_bit_reversed`, `commitment.rs:140-155`). It
then stamps `hasher` into `program_id` and the returned `LfmArtifacts`
(`:163-170`).

Under `HasherKind::Blake3` today you get artifacts *naming* Blake3 whose roots
were built with keccak. Harmless while the commitment hash is not part of the
claim — **silently wrong the instant a Blake3 commitment path exists.**

The doc comment at `registry.rs:112-116` actively asserts the currently-true
reasoning ("its preprocessed width is the same under every candidate … so no
commitment moves with it"). That sentence becomes **false** and must move in the
same commit as the guard.

### H2 — the cross-hasher test sweep does not cover Blake3

✓ VERIFIED `machine_tests.rs:2427-2430`:
`const ALL_HASHERS: [HasherKind; 2] = [Test, Poseidon]` — while `HasherKind` has
**three** variants (`hash.rs:196-212`). Its own doc at `:2424-2426` claims
exhaustiveness: *"Every `HasherKind` there is. Not derived — a new candidate must
be added here by hand, which is the point."* Blake3 was added and this was not.

The two tests it drives — the digest-binding test (`:2456`) and the cross-hasher
reject test (`:2503`) — are exactly the ones that would catch hasher confusion,
and the Blake3 arm currently sits outside both.

### H3 — the GPU tree path takes a backend parameter it does not honour

✓ VERIFIED `try_expand_leaf_and_tree_row_major_keep<F, E, B>`
(`gpu_lde.rs:680-695`) is bounded `B: IsMerkleTreeBackend<Node = [u8; 32]>` but
its body unconditionally calls the keccak device kernel
`math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep` (`:720`). `B` is a
**type-level label only**.

A Blake3 backend satisfying `Node = [u8;32]` — exactly what §2 recommends —
would compile silently and yield **keccak trees typed as Blake3**. The §2
node-type recommendation is what opens this, so it must ship with the guard.
Same shape applies to `try_expand_split_trees_row_major_keep` (`:779`),
`try_expand_leaf_and_tree_ext3_row_major_keep` (`:863`), `..._keep_dev`
(`:1554`).

---

## 5. GPU staging

✓ VERIFIED **no BLAKE3 exists anywhere in CUDA** — `grep -ril blake3
crypto/math-cuda/` returns nothing. Kernel sources: `arith.cu, barycentric.cu,
constraint_interp.cu, deep.cu, ext3.cuh, fri.cu, goldilocks.cuh, inverse.cu,
keccak.cu, logup.cu, ntt.cu`. Only `keccak.cu` is a hash.

**Survives (hash-agnostic):** LDE/NTT (`ntt.cu`), constraint composition
(`constraint_interp.cu`), barycentric (`barycentric.cu`), DEEP (`deep.cu`), FRI
fold arithmetic (`fri.cu`), LogUp (`logup.cu`), inverse/arith.

**Dies (all `keccak.cu`):** `keccak256_leaves_base_batched:152`,
`..._base_row_pair_batched:196`, `..._ext3_batched:237`,
`keccak_comp_poly_leaves_ext3:277`, `keccak_fri_leaves_ext3:326`,
`keccak_merkle_level:394`, `keccak_merkle_tail:408`,
`keccak256_leaves_base_row_major_row_pair:473`, `..._range:511`. Rust wrappers
in `crypto/math-cuda/src/merkle.rs`: `keccak_leaves_base:33`,
`keccak_leaves_ext3:83`, `build_merkle_tree_on_device:316`,
`build_comp_poly_tree_from_slabs_dev:494`,
`build_comp_poly_tree_from_evals_ext3_keep:544`,
`build_fri_layer_tree_from_evals_ext3:564`. Note tree *building* is on-device
too, not only leaf hashing.

### The tree-less R1 entry the short-term staging needs

✓ VERIFIED `try_expand_columns_batched<F, E>(columns, blowup_factor, weights)
-> Option<()>` at `gpu_lde.rs:430-434`. It expands columns in place, takes **no
backend parameter** and builds **no Merkle tree** — GPU does the LDE, host does
leaves and tree. This is the correct entry for the accept-CPU-trees phase, and
it is immune to H3 by construction.

### `device_only_gate` must be forced false

⚠ `device_only_gate` (`gpu_lde.rs:189-212`) is entirely hash-agnostic — field
tower, thresholds, `!is_preprocessed`, contiguous offsets, uniform zerofier.
That is the hazard, not the relief: it would still evaluate **true** under
blake3, but device-only residency drops the host trace, and the module doc
(`:175-180`) says a violated precondition hits a `host_trace_empty` **hard
abort**. CPU leaf hashing needs the LDE on the host.

**Follow-up (not step 1):** blake3 leaf + `merkle_level`/`merkle_tail` kernels.
`merkle_gather_paths:433` (`gather_merkle_paths_dev`, `merkle.rs:358`) is
already hash-agnostic and reusable once a device tree exists again.

---

## 6. Change list (Q6)

Each step independently verifiable. `cargo test --release` throughout — proving
tests crawl otherwise.

| # | Change | Test oracle |
|---|---|---|
| **0** | **Guards first, before any Blake3 commitment exists.** H1: make `build_artifacts_with_hasher` reject or assert when `hasher` disagrees with the commitment hash actually used, and correct `registry.rs:112-116`. H2: add `Blake3` to `ALL_HASHERS` (`machine_tests.rs:2427`). H3: remove the unused `B` parameter from the GPU tree entries, or bind it to a keccak-only marker so a Blake3 backend cannot be passed | Existing suite green; H2's two tests (`:2456`, `:2503`) now exercise the Blake3 arm and must pass unchanged |
| **1** | **Spec, no code.** `commit-spec/COMMIT.md` covering the three things no ratified doc covers: wide-leaf construction, byte→cell absorb encoding, node embedding + strict decode | Python KATs in the style of `leaf_kats.py` / `transcript_kats.py`, written before any Rust — the discipline LEAF.md and TRANSCRIPT.md both followed |
| **2** | `StarkHash` config trait + `GenericProver`/`GenericVerifier`, keccak instance only. Pure refactor | Full existing suite + a cross-version verify (the king gate) |
| **3** | `LfmBlake3` leaf + pair backends implementing `IsMerkleTreeBackend`, `Node = [u8;32]` packed, LFML leaves / LFMC parents, strict decode via `lanes_of` | Step-1 KATs + host parity against `HostTree` (`fixture.rs:157-190`) |
| **4** | B1 `IsStarkTranscript` impl (`LfmTranscript`); `grinding_factor: 0` for LFM options | Op-for-op parity against `HostSponge` + transcript KATs |
| **5** | Generalize `absorb_lfm_statement` (`statement.rs:74`) and `replay_transcript_phase_a_view` (`prover/src/lib.rs:989-992`) from `&mut DefaultTranscript<E>` to `&mut impl IsTranscript<E>` | Compile-level; existing `lfm::` suite unchanged |
| **6** | Wire `lfm_prove` / `verify_against` (`proof.rs:133, 207`) | Prove+verify round trip, `TrivialV0` then `FriToyV0` under Blake3 |
| **7** | Registry: hasher-aware `resolve` key; **add** Blake3 rows, never flip the default | `compute_lfm_registry` + drift test; Test rows must still resolve and verify |
| **8** | GPU staging: route LFM prove to `try_expand_columns_batched` (`gpu_lde.rs:430`), force `device_only_gate` false under the blake3 config | `crypto/math-cuda/tests/{keccak_leaves,merkle_root_parity,fri_layer_tree}.rs` still green for keccak |
| **9** | *Follow-up:* blake3 device kernels, restore device-resident trees | Parity tests mirrored for blake3 |

### On the registry (step 7)

✓ VERIFIED all six entries (`registry.rs:194, 278, 362, 446, 530, 614`) are
`blowup_factor: 2`, `hasher: HasherKind::Test`, with inline literal roots and
program_ids; **no Blake3 entry exists**.

`resolve` (`:176-186`) keys on `(kind, blowup_factor)` **only** — so a Blake3
row cannot coexist with a Test row until the key includes hasher. Regeneration
is `cargo run --bin compute_lfm_registry --release` (`:5`) under the standing
policy at `:6-8`: *"a drift failure is investigated, never re-blessed to silence
the test."*

**Add rows; never flip the default.** That preserves the Test entries as the
honest control the campaign depends on. Every root moves (H1's cause), so this
is a strictly larger re-bless than Phase 3's, which moved six program_ids but
**no root**.

Keep `lfm_program_id` (`statement.rs:50-68`) on keccak for now — it is a
host/consumer artifact the tower guest does not recompute, and `statement.rs:6-7`
already reserves `_V2` for the ecosystem migration.

---

## 7. Soundness register

| # | Point | Pinned by |
|---|---|---|
| **S1 ★** | **Wide-leaf / opening-width binding.** LEAF.md §1.4 specifies only the toy shape (2 rows × 4 cols → 2 LFML + 1 LFMC). Production AIRs vary in width, and the leaf hash streams `evaluations ‖ evaluations_sym` **with no length prefix or separator** — `verifier.rs:204-206` states this verbatim; `:207-213` records it *was* exploitable (a prover could pick columns after challenges they must precede) and is closed today by an explicit I3 width check, not by the hash. Any LFML chain must bind width itself or preserve that check | **Nothing.** Same class as the standing `main↔aux` open item |
| **S2** | Node malleability on decode — use `lanes_of` (`blake3_socket.rs:431-438`), not `unpack_digest` (`word.rs:52-61`) | Nothing; add a KAT |
| **S3** | Extension packing — aux openings are `FieldElement<E>` (`verifier.rs:666`), 3 base felts each, against a 4-felt LFML cell | Nothing |
| **S4** | Transcript domain separation — and the explicit correction that constraint idx 4 is *not* what makes selectors one-hot | TRANSCRIPT.md §2, §3.3; controls M5/M6/M8 |
| **S5** | Leaf domain separation — O5 retired, enforced by the tag | LEAF.md §4; M9/M10 |
| **S6** | Tree arity/padding — `HostTree::build` asserts power-of-two leaves (`fixture.rs:164`) and pads nothing; `build_from_hashed_leaves` off that shape unchecked | Nothing |
| **S7** | Grinding — scope out explicitly (§3, item 1) | Nothing |
| **S8** | 96-bit challenge entropy; squeeze runs scale with query count | TRANSCRIPT.md §4.1/§4.2 cover state collision and run length, **not** per-challenge entropy |

S1 is the gating item. S2, S3, S6 and S7 are all "pinned by nothing" and belong
in step 1's spec.

---

## 8. Scope

PLAN.md §6.2 classifies this rung as **E2** and says it "requires the
*production* RV64 prover's Merkle and Fiat–Shamir hash to be BLAKE3 … Out of
LFM's control and far larger than everything above combined."

The parameterization in step 2 is precisely what makes an **LFM-only E2**
possible while the RV64 path keeps keccak untouched. That is the central
architectural claim of this design and the reason the work is tractable at all.

It is nonetheless materially larger than **P5** as tracked in ORCHESTRATION.md
("prove+verify a wrap under BLAKE3 (swap TestPermutation)"), which concerns the
machine's *chips* (role 2). This concerns the machine's *own commitments*
(role 1 for LFM). They are different axes and should be tracked as separate
rungs, not folded together.
