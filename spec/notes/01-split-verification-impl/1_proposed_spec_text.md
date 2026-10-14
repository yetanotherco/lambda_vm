3MI spec · spec/main 41c3af9a6 · proposals from the Lambda VM implementation

# Proposed text for the open parts of the Field VM and Recursion chapters

For every section the spec leaves as TODO or as an implementation detail, and for the gaps we hit building the uniform tower, this page gives what we propose, why (with what we measured or tested), and Typst text ready to paste into the chapter.

The text follows what is built and tested in the Lambda VM: a uniform recursion tower over RPX, from an Ethereum block (a 10-transfer ethrex test block), at ~111 bits (blowup 4, Q = 80, grinding 32), one program id at every level. Names in `monospace` are the implementation's; the Typst uses the chapters' own notation where it exists. The story behind each proposal is in `0_input_for_3MI.md`.

## Sections the spec leaves open

### 1 · The verification chapter TODO

verifier.typ:1 — "TODO: sigma protocol, layout the various steps; Fiat–Shamir transformation"

**Proposal**

Lay out the verifier as the ordered sequence of transcript steps. The recursion chapter needs exactly this list: it fixes the record (section 2) and the cost of each split half.

**Why**

Every soundness gap we found in the split was a value one half assumed and the other never produced. Writing the steps once, with what each absorbs and samples, makes the record derivable instead of discovered. The order below is the one our verifier and its in-VM emitter both follow.

*Paste into verifier.typ*

``` typst
= Verification <verification>

The verifier replays the prover's Fiat–Shamir transcript. Each step absorbs
prover messages and then samples challenges; a challenge is sampled only after
everything it must depend on has been absorbed.

+ *Statement.* Absorb the instance: the public input and output, the public
  Field VM cells and the shared regions.
+ *Shape and commitments.* Absorb the shape (the number of tables and each
  table's trace length), then the preprocessed roots, which include the
  program's `DECODE` and `FIELD_VM_DECODE` roots, then the main-trace roots.
+ *LogUp challenges.* Sample $z$ and $alpha$, shared by every bus.
+ *Auxiliary commitments.* Absorb the auxiliary-trace roots.
+ *Bus contributions.* Absorb each table's claimed LogUp contribution.
+ *Composition challenges.* Sample one $beta_t$ per table $t$; the composition
  polynomial of $t$ is $sum_i beta_t^i dot C_(t,i)$ over its constraints.
+ *Composition commitments.* Absorb the composition roots.
+ *Out-of-domain points.* Sample one point $z_t$ per table, rejecting points in
  the trace or evaluation domain.
+ *Out-of-domain claims.* Absorb, per table, the trace values at $z_t$ and at
  $omega z_t$ (only the columns read at the next row), then the composition parts.
+ *DEEP challenges.* Sample one $gamma_t$ per table; if an FRI instance batches
  several tables, sample $alpha_"fri"$ to combine them.
+ *FRI commit phase.* Per FRI instance and folding layer, sample the folding
  challenge $zeta_j$ and absorb the layer root; absorb the final polynomial.
+ *Proof of work.* Check the grinding nonce against the transcript state, then
  absorb it.
+ *Queries.* Sample $Q$ query indices as bits of the largest evaluation domain.
+ *Checks.* At $z_t$: each table's constraints against its composition claim.
  Per query: every opened leaf against its root; the DEEP value of each table
  from its openings and claims; the FRI fold along the query path; the final
  polynomial. Last, the LogUp balance: the contributions sum to the term the
  public input fixes.

#aside("Who does what in a split verifier")[
  Steps 1–13 and the Merkle checks are hashing and bit work, for $v_"bin"$; the
  constraint, DEEP, FRI-fold and balance checks are field arithmetic, for
  $v_"fld"$. Section @split-verification:record lists what crosses.
]
```

### 2 · The communication record TODO

recursion.typ:329–334 — "primarily consists of the Fiat-Shamir-derived challenges … update the communication record overview once @verification is complete."

**Proposal**

Define the record as every value one half assumes and the other produces, in a fixed order, with a count computed from the proof shape alone. Challenges are only part of it.

**Why**

- With the record defined as "whatever the cut makes cross", seven tamper cases verified forged proofs before each fix: a moved constant, a decomposed hint, three truncated digest lanes, non-canonical query bits and another guest program.
- We rebuilt the split so that only named values cross. On a tower level at Q = 1 the crossing went from 2,674 cells (91% without a meaning) to 2,319 named cells. At Q = 80 it went from 32,220 to 6,129. Cycles per level fell 1.2% at Q = 1 and 6.8% at Q = 80 (6.48 M → 6.04 M cycles on the fixture tower, three fewer tables, 3.4 GB less memory), and the levels keep one program id.
- The counts below are checked against the emitted verifier every time it is built.

| Entry | Producer → consumer | Type | Count per verified proof |
|----|----|----|----|
| Fiat–Shamir challenges, in transcript order | v_bin → v_fld | extension, canonical | 2 + 3·T + \[α_fri\] + Σ ζ |
| Query bits | v_bin → v_fld | base, 0/1 | Q · log₂\|D_max\| |
| Register LDE | v_fld → v_bin | base, canonical | 2 · \|LDE\| per REGISTER table |
| Statement words (hinted ELF digest and public output) | v_fld → v_bin | base, canonical | 4 + ⌈L/8⌉ |

T = number of tables, ζ = folding challenges over the FRI instances, L = public-output length in bytes. The register LDE and statement words are the values v_bin hashes that v_fld computes; values both halves can compute from the shared input are computed on both and are not record.

*Paste into recursion.typ, replaces the paragraph at line 329 and the TODO at 334*

``` typst
== The communication record <split-verification:record>

The record holds every value one verification half assumes and the other
produces. For each entry the producing half verifies it and the consuming
half reads it; nothing else crosses. Values either half can compute from the
shared input alone are computed by both and are not part of the record.

For a verified proof of $T$ tables, $Q$ queries and FRI folding challenges
$zeta_1, ..., zeta_k$, the record is, in this order:

#table(columns: 4,
  [*Entry*], [*From → to*], [*Type*], [*Count*],
  [Fiat–Shamir challenges, transcript order (@verification)], [$v_"bin" -> v_"fld"$],
    [extension, canonical], [$2 + 3T + [alpha_"fri"] + k$],
  [Query bits, low to high], [$v_"bin" -> v_"fld"$], [bit], [$Q dot log_2 |D_max|$],
  [Register LDE, leaf order], [$v_"fld" -> v_"bin"$], [base, canonical],
    [$2 |D_"REGISTER"|$],
  [Hinted statement as 8-byte words], [$v_"fld" -> v_"bin"$], [base, canonical],
    [$4 + ceil(L / 8)$],
)

The count is a function of the proof shape, so a verifier that knows the
shape knows the record layout. A recursive verifier that verifies two
proofs (@split-recursion) concatenates their records in verification order.

#aside("Why not only challenges")[
  The consuming half must not trust any value it did not compute. A record
  that lists only the challenges leaves every other crossing value
  unconstrained in one half; in our implementation each omitted kind of value
  admitted a forged proof.
]
```

### 3 · Split verification in one proof, and split proving TODO

recursion.typ:145–239 — "Split verification": the two halves share a record and are proven as one program, `prove(product(verify_0, verify_1))`; recursion.typ:345–346 — the input, private input and record "are made available to the VMs by means of `PAGE` tables"; recursion.typ:240–241 — "= Split proving \#et("todo")"

**Proposal**

Two separate additions:

- **Split verification:** say how the two VMs share data when both halves are proven in one proof. This is the construction Robin and Erik proposed: a 24-byte pseudo-`PAGE` that binds RISC-V memory bytes to Field VM cells. It adds a second direction, `BACK`, and canonical limbs. The spec says the input and record reach the VMs through `PAGE` tables, but not how the Field VM reads them.
- **Split proving (the TODO):** proving the halves separately, possibly in other proof systems. State the requirement we found: a binding between the two proofs whose cost does not grow with what crosses.

**Why**

- The one-proof construction is the one behind the uniform tower; Robin indicated the side-by-side Field VM and Lambda VM case "can still be a single proof".
- `BACK` is how the record travels from v_bin to v_fld.
- Without the limb bound, bytes encoding v + p pass as v; there is a test for it.
- What was tried for split proving: the Field VM half as a STARK over Goldilocks and the whole hash half as a Flock proof, two proofs per step. Every shared value was published in both, and the next level checked they matched and verified both proofs. Most of the shared values are the opened values and Merkle siblings of the proof being verified: the Field VM folds them, and only the hash half can authenticate them. They cannot be cut, so the published interface grows with each level (112,804 → 972,491 cells), and it did not converge. A random-linear-combination binding instead costs ~20 billion constraints in Flock. The full account is in `0_input_for_3MI.md`, §12.

*Paste into recursion.typ, as a new subsection of "Split verification" after "Communication"*

``` typst
== Proving both halves in one proof <split-verification:one-proof>

When both halves run on VMs proven over the same field, the product
$product(verify_0, verify_1)$ is proven as a single proof: one transcript, one
set of commitments, and the LogUp buses of both VMs balanced together. The
halves share the input and the record through _pseudo-`PAGE`_ regions of the
binary VM's memory.

A region is a sequence of 24-byte cells. Cell $k$ holds three little-endian
64-bit limbs $(l_0, l_1, l_2)$, the coordinates of an extension element in the
basis $(1, X, X^2)$, and is mapped to the Field VM address $"FADDR"_k$. Per byte,
the region takes the place of the memory system's page: it sends the initial
token and receives the final token of that byte. Per cell, it sends
$("FADDR"_k, l_0, l_1, l_2)$ to `FIELD_VM_MEM`, with the limbs read from

- the *initial* bytes for an `IN` region: values present before $v_"bin"$ runs
  (the shared input and the record entries $v_"fld"$ produces), and
- the *final* bytes for a `BACK` region: values $v_"bin"$ writes for $v_"fld"$
  (the record entries $v_"bin"$ produces).

The region constrains every limb to be below $p$, so the bytes of $v + p$ do
not stand for $v$. It does not range-check bytes: the binary program reads
every byte of an `IN` cell and writes every byte of a `BACK` cell, so each byte
meets a range-checked memory token.

`FADDR` is preprocessed and its root is part of the program commitment.
Regions are page-aligned and lie in private-input pages. The statement absorbs
the number of regions and each region's base, direction, cell count and
address map; the binary program's public output begins with its program
digest and each region's base and cell count, which the verifier compares.
```

*Paste into recursion.typ, replaces the TODO at line 241*

``` typst
= Split proving <split-proving>

Split proving proves the two halves separately, possibly in different proof
systems (e.g. $v_"bin"$ in a binary field). The proofs then cannot share a
commitment, so the values that cross between the halves (the record, and the
proof data both halves read) must be bound across the two proofs.

Publishing those values in both proofs does not converge under recursion: each
level's interface contains the previous level's proof data, and it grows with
every level. Split proving requires a binding of the crossing values whose
cost does not grow with them and is cheap to compute in both fields. Until
such a binding is specified, split verification in one proof
(@split-verification:one-proof) is the construction that converges.
```

### 4 · Soundness of the recursion: the level counter Open

recursion.typ:131–137 — "soundness of this construction is not yet proven … one might be able to prevent this … by including a recursion-level counter"

**Proposal**

Make the counter part of the construction instead of a suggestion. Also say how the bit b is constrained and how both branches coexist in one program.

**Why**

We built it this way. The final verifier accepts level k and rejects k + 1. Ten tampers are rejected: b = 2, b flipped both ways, a wrong comm(x) (twice), a forged child level, a wrong comm(y), another program of the same shape, a moved output, and shared precomputed trees.

*Paste into recursion.typ, after the verify\* definition*

``` typst
== Recursion level <split-recursion:level>

Every recursive proof states its level. $v_"fld"$ asserts $b^2 = b$. Every
assertion of the base verifier is multiplied by $1 - b$ and every assertion of
the recursive verifier by $b$: both run, and the inactive one constrains
nothing. With $b = 1$ the binary half checks that the child's output is
$(comm(x), comm(y), ell_"child")$, and publishes

$ ell = 1 + b dot ell_"child". $

The final verifier checks $comm(x)$, $comm(y)$ and the level it expects; a
proof of level $k$ is rejected at any other level.
```

### 5 · Which value a hint takes Implementation detail

field_vm.typ:146–147 — "provides no information on which value exactly should be hinted … leave this as an implementation detail"

**Proposal**

Give executors one default rule, so that two executors of the same program agree on its execution.

**Why**

This is what our executor does. It covers every hint the generated verifier uses: inversions are solved linearly, and frame pointers and dereferenced destinations come from the host.

*Paste into field_vm.typ, after the instruction-notation paragraph*

``` typst
#aside("Choosing hinted values")[
  An executor chooses hinted values in this order:
  + an output hint on a register argument $d = #`imm`_0 dot #`reg` + #`imm`_1$
    with $#`imm`_0 != 0$ is solved for `reg`;
  + a single input hint that enters the constraint linearly (as in `INV`) is
    solved for its register;
  + any other hint (a new frame pointer, a `d` that dereferences memory, an
    output hint with $#`imm`_0 = 0$) is supplied by the host.
  An output hint on `ZERO` is invalid: $next("ZERO")$ is already fixed by the
  result. An instruction whose `d` register the next instruction input-hints
  is rejected by the executor.
]
```

### 6 · Restricting the design after observed use Future

field_vm.typ:350–356 — "Depending on observed use, in the future, we can restrict this design…"

**Proposal**

Record what a generated STARK verifier actually uses, so the parameters can be chosen from it.

**Why**

All measured on the Field VM half of a real epoch verifier:

- **N = 2 registers:** 2.92 M → 2.39 M cells (−18%).
- **d = 3 at blowup 2:** proving goes from 4.53 s to 2.82 s, but the proof grows from 3.9 MB to 7.3 MB.
- **A register-free row variant:** only 13% of rows could use it, so we discarded it.
- **A third memory argument:** no net gain.

*Paste into field_vm.typ, appended to "Notes and potential optimizations"*

``` typst
- Observed on a generated STARK verifier: it works mostly through memory, so
  $N = 2$ general-purpose registers suffice and reduce the trace by about a
  fifth against $N = 5$. $d = 3$ at blowup 2 proves faster at the price of a
  larger proof; the parameters $(d, N, t)$ should be part of the program's
  identity rather than fixed globally.
```

## Gaps the spec does not cover yet

### 7 · The MEM table Gap

field_vm.typ:16–27 — "a committed table containing the memory as well as the multiplicities"

**Why**

With LogUp alone, a prover can commit two rows with one address and serve different values to different reads. Robin confirmed both the chip and the backfilling.

*Paste into field_vm.typ, new section after the ISA*

``` typst
== The memory table

`MEM` has one row per address: `ADDR`, an extension `VALUE`, a multiplicity
`MU` and a public flag `PUB`. It receives $(#`ADDR`, #`VALUE`)$ on
`FIELD_VM_MEM` with multiplicity `MU`; an address with non-zero extension
coefficients does not balance. Addresses are unique by construction:
$#`ADDR`_0 = 0$ and $next(#`ADDR`) = #`ADDR` + 1$.

`MEM` is read-only in the proof and write-once in execution: the prover
backfills it, an instruction whose `d` dereferences an undefined cell defines
it, and reading an undefined cell from `a`, `b` or `c` is an error.
```

### 8 · Binding public cells Gap

**Why**

The inputs and constants in `MEM` must be fixed by the statement. Shared data goes through the pseudo-PAGE (section 3); constants and published values need their own path.

*Paste into field_vm.typ, after the memory table*

``` typst
Rows with $#`PUB` = 1$ send $(#`ADDR`, #`VALUE`)$ on `FIELD_VM_PUBLIC`. The
verifier absorbs the public cells into the transcript, requires their
addresses to be strictly increasing and canonical, and closes the bus itself
against the term those cells determine.
```

### 9 · Canonical encodings of what crosses Gap

**Why**

Each of these accepted a forged proof before its fix:

- x and x + p decompose to different query bits;
- a four-element RPX digest truncated to a three-coordinate cell;
- a base value carried with non-zero extension coordinates.

*Paste into recursion.typ, in the record section*

``` typst
Every record entry has one canonical encoding, enforced by its producer and
checked where it crosses:
- bits are 0 or 1, and a decomposition of a field element reconstructs a value
  strictly below $p$;
- limbs crossing in a pseudo-`PAGE` are below $p$;
- a base-field entry has zero extension coordinates;
- a digest crosses whole: all of its elements, never truncated to fit a cell.
```

### 10 · Recursive instance commitment and the commitment domain Gap

recursion.typ:359–394 vs memory.typ:25–33 and streaming.typ:117–124

**Why**

The recursion chapter says stdout is backed by `PAGE` tables. The memory chapter calls the commitment domain unpaged, and the streaming chapter forbids `PAGE` on unpaged domains. We bind stdout through the COMMIT bus, whose term the verifier recomputes. With batched trace commitments (section 11) there is no per-table `PAGE` root to open anyway.

*Paste into recursion.typ, replaces the PAGE-opening check at 370–394*

``` typst
$v_"bin"$ writes $comm(x)$, $comm(y)$ and its level to `stdout`, the
commitment domain, which is unpaged: its bytes reach the verifier as public
output, whose bus term the verifier recomputes. With $b = 1$, $v_"bin"$ checks
that the child's public output, after its header, is
$(comm(x), comm(y), ell_"child")$.
```

### 11 · Batched trace commitments and one FRI Gap

recursion.typ:347–349 — "the prover commits to each of these tables individually"

**Why**

- With one tree and one FRI per table, verifying an epoch of 2²¹ cycles costs 409,013 hash permutations, and recursion diverges (35.5× even with a native v_bin).
- With one tree per phase and one FRI it costs about 57,000, and recursion converges.
- On a mainnet block, the same change gives −12% prove time, −73% proof size and −75% verify time.
- Program tables keep their own trees, so the program id is still a list of roots.

*Paste into recursion.typ, replaces 347–349*

``` typst
Tables whose content the program fixes (`DECODE`, `FIELD_VM_DECODE` and the
other program-dependent preprocessed tables) are committed individually;
their roots identify the program. Trace tables are committed in one tree per
phase (main, auxiliary, composition) over all heights, and their low-degree
tests combined into one FRI instance, once the shape (table set and heights)
has been absorbed.
```

### 12 · Program identity and the two commitment functions Gap

**Why**

The spec uses one commitment notation for both instances and programs. A split program is identified by both halves and by the layout of what they share.

*Paste into recursion.typ, in "Recursive instance verification"*

``` typst
A split program's identity is the hash of a domain tag, its binary
executable's digest and entry point, and the roots of its program-dependent
tables: `DECODE`, the executable's data pages, the `FIELD_VM_DECODE` chunks and
the pseudo-`PAGE` address maps. Instances and programs are committed with
separate domain tags:
$ comm(x) = H("instance" || "id" || |"out"| || "out"), quad
  comm(y) = H("program" || "id"). $
```

### 13 · Field VM parameters left unnamed Gap

**Why**

- **The extension polynomial:** it has to be fixed for proofs and programs to be portable.
- **DECODE packing:** the X² coefficient was always zero, so we pack base-field values instead.
- **Segments:** they avoid padding a long run to the next power of two.

*Paste into field_vm.typ*

``` typst
- The extension field is $FF_p [w] \/ (w^3 - 2)$.
- The decoding tuple may be packed as base-field values: LogUp weights each
  value with its own power of the challenge, so packing into extension
  elements is an optimization, not a requirement.
- An execution may be proven in power-of-two segments. Each segment has public
  first and last states; a segment that is not the last repeats the next
  segment's first step in its last row, so the hand-over is constrained, and
  only the last segment pads with the halt state.
```

### 14 · When recursion converges Gap

recursion.typ:80–85 — "the proof size typically shrinks"

**Why**

Measured cost of verifying one epoch, as a multiple of the epoch:

- 337× per table with an interpreted v_bin;
- 35.5× with a native v_bin;
- 11.5× with batched commitments;
- 2.38× with no keccak in the guest;
- 0.615× at 2²³.

The tower level is at 0.973 × 2²³ at Q = 80. The node that verifies two proofs is the binding case.

*Paste into recursion.typ, after the shrinking argument*

``` typst
Recursion converges when the cycles of the recursive verifier, for the proof
shape it verifies and at the target security, fit one epoch:
$ C_"verify"(L) < 2^L. $
$C_"verify"$ includes every hash permutation, the guest's overhead, the
public input and, for a node that verifies two proofs, both. "Typically
shrinks" is not a guarantee: in our measurements the condition required
batched commitments, no binary-hash tables in the recursive guest, a
compiled $v_"bin"$, and epochs of $2^23$ cycles.
```

Prepared by the Lambda VM team, October 2026, against spec/main 41c3af9a6 (Field VM \#971, Recursion \#943). Every number is measured in our implementation unless stated otherwise.
