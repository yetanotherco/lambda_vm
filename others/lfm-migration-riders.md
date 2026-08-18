# Riders to carry into the ecosystem hash migration

Things that are cheap-to-free if they ride the transcript/hash rebuild (which
is already proof-breaking and already owed), and not worth a proof-breaking
change on their own. Each entry: what, why it helps, what it costs today.

## 1. Constant-consumption challenge sampling

**What:** make the production `sample_field_element` consume a FIXED number of
candidates per draw instead of looping on rejection.

**Why:** a straight-line machine cannot follow a data-dependent consumption
schedule, so the LFM transcript replay encodes the no-rejection schedule and is
unprovable for a transcript that ever rejects (`SOUNDNESS.md` §6.3). With
constant consumption the restriction disappears for every future machine.

**Cost today:** completeness only, bounded `< 10^-6` per proof at production
draw counts. Acceptable — hence a rider, not a fix.

## 2. One-byte pad at the end of the statement encoding

**What:** pad the continuation-epoch statement so its length is `≡ 0 (mod 4)`.

**Why:** the encoding is `207 + L + 16R` bytes (not 223 — an arithmetic slip in
the first report, now machine-checked by
`epoch_statement_cursor_is_three_plus_output_len`). Every subsequent absorb
inherits the resulting cursor — including all of Phase A, whose roots are
individually 32-byte-aligned but land misaligned because they inherit the
statement's cursor. (Alignment is a property of the CURSOR, not of the field:
this was initially mis-analysed as "Phase A needs no splice", which is true in
isolation and false in context.)

**⚠ Second correction — the shift is NOT unconditionally 3.** It is
`(3 + L) mod 4`, where `L = |public_output|`. The earlier claim of "≡ 3" quietly
assumed `L ≡ 0 (mod 4)`, which is false in general: `public_output` is collected
one byte per COMMIT operation (`trace_builder`), so `L` is whatever the workload
produced. Consequences: the Phase-A splice cost is WORKLOAD-DEPENDENT, and it is
**zero** whenever `L ≡ 1 (mod 4)` — roughly one workload in four pays nothing at
all. A pad that fixes the cursor would make the cost zero and, more usefully,
*predictable*, which is the stronger argument for the rider.

**⚠ Third correction (2026-07-30, measured on a real fixture) — the `16R` term
is not live.** `runtime_page_ranges` is ALWAYS EMPTY for continuation epochs
(PAGE tables are skipped; the struct comment says so). So the real encoding is
`207 + L`, with `R = 0`, and the shift is `(3 + L) mod 4` full stop. R1e's
synthetic test shape uses `R = 2`, which is a legitimate test shape but means
any arithmetic above quoting `16R` is computed over a term the real statement
does not have. The rider's conclusion is unchanged — the shift still depends on
`L`, and a pad still makes it predictable — but do not read `16R` as live.

**Cost today:** 2 roots × 8 halves × T tables spliced, whenever the inherited
shift is nonzero — at T = 24 that is 384 `BitDec` + ~13k `BALU` rows per proof,
and zero for the ~1-in-4 workloads whose output length lands the cursor on a
boundary. Against a ≈7.3M-instruction
epoch verify that is ~0.2% of instructions; the `BitDec` rows are wide, so
call it low single-digit percent of the machine's fixed trace floor. Real, but
nowhere near worth a proof-breaking change by itself.

**Note:** the encoding is already versioned by its domain tag
(`LAMBDAVM_CONTINUATION_EPOCH_V2`), so a pad is a tag bump — exactly the kind of
change a migration absorbs for free.

## Rule for adding to this list

An entry belongs here if (a) it costs the machine real work today, (b) fixing it
requires a proof-breaking or production-semantics change, and (c) the migration
has to touch that code anyway. If (c) is false it is a normal PR, not a rider.
