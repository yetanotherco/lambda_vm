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

- epoch *i*'s `reg_fini` equals epoch *i+1*'s supplied REGISTER root
  (`build_epoch_airs` is the single prove/verify source of truth; the verifier
  derives `register_init` from the entry point for epoch 0 and from the previous
  epoch's `reg_fini` thereafter);
- L2G root equality between each epoch's own L2G commitment and the
  corresponding sub-proof in the global proof;
- the attestation fold `program_id ‖ concatenated public_output`.

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
