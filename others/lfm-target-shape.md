# The target: recursing CONTINUATION EPOCHS

User-confirmed 2026-07-29. Every Phase R track should build against this shape,
not against a monolithic proof.

## What the machine must verify

**One continuation EPOCH proof**, and later the **global proof** that ties
epochs together. Not `prove()`/`verify()`'s monolithic shape — that path exists
but is not the target, and building for it would silently miss AIRs (see below).

Sub-proof count per epoch (`prover/src/continuation.rs`, verified earlier this
phase):

```
T_epoch = table_counts.total()              # 14 split-table families, chunked
        + (10 if final_epoch else 9)        # FIXED_TABLE_COUNT, minus HALT
                                            # on intermediate epochs
        + page_configs.len()                # one PAGE AIR per touched page
        + 1                                 # the epoch-local L2G table
```

The global proof carries one L2G sub-proof per epoch plus GLOBAL_MEMORY.

## Consequence 1 — there are 28 AIRs, not 25

`l2g_global_air`, `l2g_memory_air`, `global_memory_air` are private fns in
`continuation.rs` and appear in NONE of the four hand-maintained 25-item
enumerations, none of which asserts a count. They are exactly the AIRs a
continuation proof adds. Any artifact, coverage claim or constraint-evaluation
leg scoped to "the 25" is complete for monolithic proofs and quietly incomplete
for the target. `l2g_memory_air` is the one with real constraints
(`L2gMemoryConstraints`); the other two are `EmptyConstraints` but still need
shape + meta + max_degree.

## Consequence 2 — the statement is the ContinuationEpoch variant

`absorb_statement(StatementKind::ContinuationEpoch { epoch_label })` with
`CONTINUATION_EPOCH_TAG = b"LAMBDAVM_CONTINUATION_EPOCH_V2"` (30 bytes, ≡ 2 mod
4 — one of the two misalignment points R1e handles), and the trailing
`epoch_label` u64 the monolithic variant lacks. R1e is already building this
variant; do not "simplify" it to the monolithic tag.

## Consequence 3 — chaining is part of the statement, not an extra

Verifying epochs in isolation is not verifying a continuation. The chaining
obligations, all of which the RV64 guest already performs and the machine will
have to emit:

- epoch *i*'s `reg_fini` feeds epoch *i+1*'s REGISTER root — a DERIVATION, not a
  comparison, and nothing is supplied (see below);
- L2G root equality between each epoch's own L2G commitment and the
  corresponding sub-proof in the global proof;
- the attestation fold `program_id ‖ concatenated public_output`.

### ★ The REGISTER derivation IS the binding — decided, not a TODO (R1g)

The first obligation is often written as "compare `reg_fini` against the next
epoch's supplied REGISTER root". There is no supplied root and no comparison.
The chaining loop carries `register_init = epoch.reg_fini()` forward and
`build_epoch_airs` (`continuation.rs:636`) *constructs* the next epoch's
preprocessed commitment from it via
`register::compute_precomputed_commitment_with_fini`. Lie about `reg_fini` and
the constructed commitment no longer matches the one the proof was made against,
so the proof fails. The binding is structural.

This settles the long-carried Phase-0 item **"wire the REGISTER verify-side
supply route"**. `VmAirs::new` does have a `register_preprocessed:
Option<(Commitment, usize)>` parameter that every verify caller passes `None`
to, so the plumbing looks like an unfinished route. It is not unfinished — it
must stay unwired. **Computing the commitment from `reg_fini` is what ties the
VALUES to the commitment.** Supply the root instead and `reg_fini` has no
remaining role, so a prover can offer a root consistent with a `reg_fini` it
never honoured, and the cross-epoch chain that `reg_fini` carries goes
unenforced. The in-guest RV64 verifier's per-epoch recomputation is therefore
load-bearing, not wasteful.

Consequence for the machine: it must EMIT that derivation — 3 columns × 128
rows, an inverse FFT and an LDE FFT each, then a full Merkle tree build. Its
output is exactly the preprocessed root Phase A absorbs, which is why Phase A
cannot be replayed over a real epoch without it. Cost is negligible: 255
permutations at blowup 2 against ~1.4M for the epoch verify (~0.02%), 1023
against ~460k at blowup 8 (~0.2%). See the sizing note below before quoting any
ratio.

### ★ The derivation, BUILT and MEASURED (R1g(i))

First-hand, from `machine_tests::register_derivation_cost` and
`the_register_derivation_matches_production` on branch
`feat/lfm-register-derivation`. Everything in this subsection is measured or
asserted by a test unless marked otherwise.

**The permutation prediction was exactly right.** 255 / 511 / 1023 at blowup
2 / 4 / 8, matching `2·leaves − 1` with `leaves = 128·blowup / ROWS_PER_LEAF`,
i.e. `128·blowup − 1`. So was the noise figure: **0.0182%** of an epoch's
hashing at blowup 2 and **0.2224%** at blowup 8, against the ~1.4M / ~460k
baselines above. A leaf is 48 bytes (three columns × a row pair) and a parent
64, so each is exactly one rate block and the permutation count is the node
count with nothing else in it.

**The FFT half of the prediction was wrong in an interesting direction.** The
text above says "3 columns × 128 rows, an inverse FFT and an LDE FFT each".
Measured, the transform is **8.5% of the derivation's own arithmetic at blowup
8** (18,176 `LFM_BALU` rows of 214,784) and the remaining **91.5% is byte
swapping** — turning each extended field element into the big-endian bytes the
leaf hash absorbs, 64 rows per value via `felt_be_halves`. Anyone budgeting this
leg from "it is two FFTs" is budgeting the small half. Two corrections feed
that:

- **Only two of the three columns need a transform.** OFFSET holds the register
  word addresses, which are fixed, so its extension is a program CONSTANT — the
  shape-static principle paying out for once. It is computed at program-build
  time by production's own `interpolate_fft`/`evaluate_polynomial_on_lde_domain`
  and interned, which also means the three columns reach one tree by two
  different routes and a matching root pins the emitter against the function it
  emits.
- **The constant column is not free at leaf time.** Its values are byte-swapped
  into the leaf like any other, one bit decomposition and 64 ALU rows each.
  Pre-computing those halves as constants would drop 32,768 rows at blowup 8 —
  and save **zero committed cells**, because `layout::padded_rows` rounds the
  group to a power of two and 214,784 and 182,016 both land on 2^18. Measured,
  not estimated; the same holds at blowup 2 and 4. Left undone deliberately.

**Answer to the gadget question: no second hashing gadget.** The tree build
needs no gadget `keccak_merkle_walk` does not already contain. What it needed
was for the PARENT step to stop being welded to the walk's `Select`:
`edsl::keccak_hash_pair` is now that step, `keccak_merkle_walk` calls it after
its select, and `keccak_merkle_tree_root` calls it with no select at all — a
tree knows every child's side when the program is emitted. The leaf gadget is
`keccak_leaf_hash`, reused unchanged. Applying the sizing rule to the
alternative: a `Select` is **17 main cells against a permutation's 36,256**, so
the two selects a walk step carries would add 0.09% to each parent (0.03% over
the whole tree) — the case against routing the tree through the walk is
structural, not economic. A walk visits one node per level; a tree visits `2^k`.

For the FRI leg the shape of the answer transfers but the answer does not, and
this leg did not check which: a verifier RECEIVES layer roots from the proof and
authenticates openings against them, so FRI plausibly wants the WALK with a
different leaf width rather than a build — and `keccak_leaf_hash` is already
parameterized by leaf width. Unverified; the FRI leg should settle it rather
than inherit this sentence.

**What this leg does not establish.** The two arenas are unbound. In the
assembled verifier `R_{i+1}` is the vector the next epoch reads as its INIT and
the published root is what that epoch's Phase A absorbs; until those joins exist
a prover may supply any pair and get the honestly-derived root for it. Same
standing caveat as the L2G binding.

**A fourth member of the degenerate-parameter family, demonstrated.** A real
register file is mostly zeros — the fixture's epoch-0 boundary is **3/67 nonzero
INIT, 10/67 nonzero FINI, 9 rows differing**. Deliberately dropping row 40 from
the emitted columns PASSED the differential against the real fixture and was
caught only by a synthetic file with all 67 rows distinct. The falsification is
recorded because it is the rule's clearest instance so far: the real data is not
merely a weak witness here, it is blind over 57 of 67 rows.

### Sizing rule — compare against the WHOLE leg, never a sample of it

Two gadget-sizing errors this phase produced ratios that were arithmetically
correct and pointed the wrong way:

1. **Rows are not a cost unit across chips.** An `LFM_BALU` row is 4
   non-preprocessed columns; one keccak permutation expands into 24
   `KECCAK_RND` rounds of 1480. Compare CELLS. (This killed the byteswap-chiplet
   proposal: 322 cells vs 36,256, no crossover at any table width.)
2. **A sample of a leg is not the leg.** The REGISTER tree was first sized
   against R1f's opening program — 22 permutations — giving "12–46× the opening
   leg" and the conclusion that it was expensive. But R1f was ONE query on ONE
   table, roughly `1/(queries × tables)` of the epoch's opening work. Against the
   whole epoch verify the same gadget is ~0.02%. Same number, opposite decision.

Baseline to size against, from *Scale* below: **~1.4M keccak permutations per
epoch verify at blowup 2 / 219 queries, ~460k at blowup 8 / 73 queries.**

## Consequence 4 — page-parameterized constraints are an identity risk

PAGE's captured constraint program folds `page_base` into IR constants, so it is
not one static blob. Size is negligible; IDENTITY is not — if constraint
artifacts vary with the workload's page set, a machine program embedding
constraint evaluation would too, and page bases are arbitrary addresses rather
than a small ladder. Registry entries must not become workload-dependent
(SOUNDNESS §2). The eventual fix is promoting page base to a runtime uniform —
it is already authenticated via supplied page roots. Tracked as a machine-side
design item; Phase 0 measures and reports it.

## The shape-static principle (R1e, and it generalises)

**Shape-static values are program CONSTANTS, never arena reads.** The table
counts, the page-range list and `num_private_input_pages` determine how many
sub-proofs Phase A absorbs and what the AIR layout is. A program that read them
from an arena would be claiming to verify a shape it was not compiled for — the
prover would choose the shape, which is exactly the property the registry exists
to deny. Only genuinely per-proof data (ELF digest, public output, epoch label,
roots, openings) comes from arenas.

Corollary, and the tension to watch: every shape-static constant is part of
program identity, so each distinct shape is a distinct registry entry. That is
correct and cheap for a small ladder of shapes; it is what makes Consequence 4
(page bases folded into constraint constants) a real problem rather than a
theoretical one, since page bases are not a small ladder. When the constant set
stops being enumerable, the answer is the runtime-uniform promotion — a value
supplied at verify time and authenticated, not a constant.

## Next-row PRUNING is program text, not arena data (constraint leg)

Same class as the shape-static principle, and it bites the other way round.

The verifier opens every trace column at `z` but prunes the `g·z` block down to
each AIR's DECLARED `trace_ood_next_row_columns`, reconstructing **ZERO** for
every column outside that set (`ood::OodLayout::reconstruct_full`). So a machine
that hinted a value into an undeclared next-row slot would evaluate constraints
over a frame **no verifier can produce** — accepting proofs the real verifier
rejects. The declared set is AIR shape, it is carried in `AirShape`, and the
zeros belong in the emitted program as the pooled zero constant.

`lfm::constraints::hint_ood_frame` does this, and
`pruned_next_row_columns_are_program_zeros` pins it by asserting, column by
column on CPU's real shape, that a slot is the pooled zero cell **iff** the AIR
does not declare it. The cheap consequence is that a frame costs
`width + (steps − 1)·|next_row_columns|` arena words instead of `steps · width`;
the expensive consequence is the one above.

The generalisation for any leg that touches openings: **when the verifier
reconstructs a value rather than reading it, the machine must reconstruct it
too — from program text.** Reading it from an arena hands the prover a degree of
freedom the protocol does not give them, and it is invisible in a differential
test that feeds both sides the same frame.

## A degenerate parameter hides implementations from every real test

Third member of the same family, and the most general.

**When every production instance shares one value of a parameter, a differential
over production data cannot distinguish implementations that differ only off
that value.** The real data does not exercise the difference, so the wrong
implementation passes — not by luck, but structurally. The synthetic case is the
only witness there can be.

Two instances so far, both in the constraint/DEEP legs:

- **`step_size = 1`.** `build_pruned_trace_term_coeffs` walks column-major, so
  along a fixed row the γ exponent advances by the block's ROW COUNT, not by one.
  Every one of the 28 production AIRs has `step_size = 1` and a single next row,
  collapsing both strides to one. A per-row Horner in plain γ therefore passes
  the real-proof differential against the production reconstruction, the census,
  and every other test in the suite — and is wrong for the first AIR that widens
  a step. Witnessed by `the_coefficient_exponent_formula_holds_at_a_wider_step`,
  which builds a `step_size = 2` layout through the verifier's own
  `build_pruned_trace_term_coeffs`.
- **`end_exemptions = 0`.** Every production constraint emits through
  `RowDomain::ALL`, so a consumer that ignored the field entirely would pass the
  whole production sweep. `crypto/stark`'s `ExemptConstraints` exists for exactly
  this reason, and its author says so: "so that *always zero in production*
  cannot decay into *never tested*".

**The falsification needs two halves.** Showing the synthetic case passes proves
nothing on its own — a test that only checks the right answer passes against
both implementations if the wrong one happens to agree there too. It must also
assert that the DEGENERATE reading gives a DIFFERENT answer at the synthetic
value. Without that second half the test passes against the wrong emitter, which
is the failure mode it exists to prevent.

How to find these: for each parameter a leg consumes, ask what values production
actually takes. If the answer is "one", that parameter needs a synthetic witness
before the leg can be called tested.

## Alignment is a property of the cursor, not the field (R1e)

A 32-byte root is self-aligned and still lands misaligned if it inherits an odd
byte cursor. The epoch statement is `207 + |public_output| + 16·ranges` bytes,
`≡ 3 (mod 4)`, so EVERY subsequent absorb — all of Phase A included — is spliced
at shift 3. Machine-checked by
`epoch_statement_ends_three_bytes_past_a_boundary`. Anyone reasoning about a
field's alignment in isolation will get this wrong.

## Scale, for sizing decisions

A small ethrex block is 1–2 epochs plus the global proof, so the target is one
epoch verify (≈7.3M machine instructions unfused) plus chaining — not a fleet.
Keccak permutations per epoch verify: ~1.4M at blowup 2 / 219 queries, ~460k at
blowup 8 / 73 queries — which is what makes chunking mandatory and the
blowup/topology choice worth revisiting before the wrap run.
