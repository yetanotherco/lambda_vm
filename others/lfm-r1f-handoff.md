# R1f handoff — keccak-emitter → successor

Written 2026-07-30. R1f is PARTIAL: (b) and half of (a) are done and committed;
(c) and (d) are not started. Handing off on context, per the standing decisions'
"quality over completion".

**State: `cargo test -p lambda-vm-prover --lib lfm` green, `make lint` 0,
everything committed** — `feat/lfm` at `2d4aa350` plus the doc/verification
slice. Only `others/` is untracked. R1a–R1e are all closed and green.

---

## 1. What is DONE

### 1b. Real proof bytes — `lfm/proof_fixture.rs`
A two-epoch continuation proof in the guest's wire format, produced by the SAME
encoder the guest's blob comes from (`prove_continuation` →
`encode_continuation_guest_input`, both already `pub`). No new format, no new
visibility.

The existing dump path (`test_dump_recursion_input`) is `#[ignore]`d, driven by
five env vars, and writes a fixed `/tmp` path — unusable from a deterministic
test — so only its two encoder calls were reused.

**Epoch size is measured, not guessed**: the `fibonacci` guest gives ONE epoch at
`log2` 6/8/10 and TWO at 4, so it runs 17–64 cycles. `FIXTURE_EPOCH_LOG2 = 4`,
preset `min`. Blob: 310,212 B at one epoch, 587,188 B at two. The cache lives in
`temp_dir`, NOT the repo — a checked-in binary drifts from the encoder silently,
so the GENERATION path is what a cold run exercises.

Test: `continuation_fixture_generates_two_epochs`.

### 1a (half). Arena filler — `lfm/proof_arena.rs`
Reads an epoch's main-trace Merkle roots out of the archived blob in place and
packs them into arena halves. Tests:
`arena_filler_reads_real_committed_roots`,
`supplied_preprocessed_roots_are_embedded_in_the_blob`.

**Measured on the real proof**: epoch 0 = 24 sub-proofs / 8-byte public output;
epoch 1 = 25 / 0-byte. Confirms `T_epoch = counts + (10 final | 9 intermediate)
+ pages + 1`, and confirms `T = 24` for SOUNDNESS §6.3 (now marked measured
rather than assumed).

**NOT done**: openings and sibling-path extraction. The API is located —
`query_list_len()`, `query(i) -> FriDecommitmentView`, `deep_poly_openings_len()`
at `crypto/stark/src/proof/view.rs:409-423` — so this is mechanical, not
exploratory.

---

## 2. What REMAINS — (c) and (d)

### ★ `edsl::merkle_walk` CANNOT be used. Build `keccak_merkle_walk`.
The existing walk calls `LfmBuilder::compress` → the `LFM_HASH` chiplet running
`TestPermutation`, the deliberately non-cryptographic Milestone-C placeholder. It
authenticates the Milestone-C fixture tree because that tree used the same
placeholder. **Production trees are keccak throughout**, so no amount of correct
path-walking reproduces a production root. This was the leg's original spec
instruction and it is wrong; both prerequisites for the replacement already exist
(R1c/R1d keccak256 over byte streams, R1e slice a big-endian rendering).

### The conventions, read from source
**Leaf** (`crypto/stark/src/commitment.rs`, `ROWS_PER_LEAF = 2`, line 42):

```
leaf(i) = keccak( col_0[br(2i)] ‖ col_1[br(2i)] ‖ … ‖ col_0[br(2i+1)] ‖ … )
```

Every element via `write_bytes_be` (8 bytes base, 24 ext). `br` is a bit-reversal
of the row index — a host-side arena-filler concern, not the machine's. One path
authenticates a value and its symmetric counterpart, which is why the pair is the
leaf.

**Parent** (`crypto/crypto/src/merkle_tree/backends/field_element.rs:41`):
`keccak(left ‖ right)`, 64 bytes, **no domain separation, no ordering flag**.

### Shape and cost
- Per level: TWO `select`s (a digest is two machine words and both must swap on
  the same bit), then `keccak256` over 16 halves. 64 bytes fits inside one
  136-byte rate block ⇒ one permutation per level.
- The LEAF is the expensive part and **byteswapping dominates it, not hashing**:
  `2 · cols` elements each needing `felt_be_halves` (1 `BitDec` + 64 `BALU`). For
  a 50-column table ≈ 100 `BitDec` + 6.4k `BALU` against only ~6 permutations.
  **Measure this — it is the input to whether a byteswap chiplet is worth
  proposing.** It is not avoidable by pre-swapping in the arena: opened values are
  consumed as field elements by the FRI algebra AND as bytes by the leaf hash, so
  something must connect the two representations.
- Root comparison: `assert_word_eq_lanes` with the root's unpack hoisted, as
  `fri_toy_program` already does per query.

### (d) Tamper vectors
Wrong sibling, wrong index bits, wrong leaf → all must reject.

---

## 3. Non-obvious decisions and WHY

- **★ Archived accessors are METHODS, not relaxed field visibility.** rkyv mirrors
  the source field's visibility onto the archived struct, so making
  `ContinuationProof::epochs` `pub(crate)` would have opened the OWNED type at the
  same time — silently becoming the route the team lead had explicitly rejected.
  `impl ArchivedContinuationProof { pub(crate) fn num_epochs / epoch_proof /
  epoch_public_output }` exposes only the path `verify_continuation_archived`
  already traverses. **If you need anything else off an epoch, add a method there;
  do not touch the field.** Visibility on the OWNED type is a different question
  and needs a ruling.
- **The fixture is BYTES because that is what production is.** The guest never
  holds a `ContinuationProof`; it reads a blob zero-copy. A reader over bytes is
  the direct analogue, and divergence between the two is a meaningful signal.
- **Pack each field into its OWN halves.** An arena is a vector of words, not a
  byte stream. Concatenating fields then packing lets any field of
  non-multiple-of-four length shift everything behind it — silently, since the
  halves count still comes out right. This cost real debugging time in R1e.
- **Shape-static values are program CONSTANTS, never arena reads** (table counts,
  page-range list, `num_private_input_pages`). A program reading them from an
  arena claims to verify a shape it was not compiled for.

## 4. Preprocessed roots — the ruling, and what it still owes

Team lead's ruling: static roots (BITWISE, KECCAK_RC) are shape-static ⇒ program
constants, which is already how `LfmAirs` treats them. Supplied roots come from
the blob.

**Verified, partially.** `ContinuationGuestInput` carries `decode_commitment` and
`page_commitments` as `pub` fields, and the fixture's DECODE root is present and
nonzero. **Caveat: this fixture has ZERO page commitments** (fibonacci touches no
data pages), so the page path is present-but-unexercised — do not treat it as
tested.

**Refinement the ruling did not cover: REGISTER is DERIVED, not supplied.**
`EpochProof` (`continuation.rs:394`) carries `reg_fini: Vec<u32>` — the register
FILE — and the verifier derives the next epoch's REGISTER root from it. The data
is in the blob, but a derivation step sits between it and the root. Budget for it.

**Also from `EpochProof`: `runtime_page_ranges` is ALWAYS EMPTY for continuation
epochs** (PAGE tables are skipped; the comment at line 401 says so). R1e's
`epoch_statement_shape()` uses two ranges, which is fine for a synthetic shape but
means the REAL statement has `R = 0`, so its length is `207 + L` and the Phase-A
shift is `(3 + L) mod 4` with no `16R` term.

## 5. Method rules (non-negotiable — these caught every real bug this phase)

1. **Falsify every new mechanism.** Break it, watch the RIGHT test fail, revert.
   If nothing fails, or the wrong thing fails, the TEST is wrong.
2. **Execute-only tests prove nothing about chips.** Only prove+verify sees them.
3. **Scrutinise the oracle** as hard as the thing under test. Here the best oracle
   is the real proof's own committed root — use it rather than recomputing a leaf
   host-side and comparing against yourself.
4. **Soundness claims need coherent forgeries**, not trace tampering.
5. **A deferral's safety argument is itself a claim needing evidence.** Twice this
   phase a "surely it's fine" premise was false: a mask that looked cosmetic was
   pinning arena bytes past a length prefix, and a remembered public-output length
   was simply wrong.

## 6. Process

- Append one line to `others/lfm-agent-status.log` per slice; commit each green
  slice yourself (`git -c user.name="Mauro Toscano" -c
  user.email="maurotoscano2@gmail.com"`), no AI attribution, never commit red.
- `others/lfm-standing-decisions.md` lists what is pre-authorised — read it before
  stopping to ask.
- `others/lfm-target-shape.md` has the epoch composition and the chaining
  obligations that come next (R1g).
- `make lint` from the repo root is the gate; `cargo fmt --check` is not enough.
