# BLAKE3 in-circuit cost model — handover notes for the crypto team's split work

Verified against `blake3-real-hash` (2026-08-13); every figure code-cited in the full
report (session record). Context: the team owns the algebraic/hash split; this is what
the machine-side measurements say about where the cost lives and what moves it.

## The numbers (cells per compression = main + 3·⌈interactions/2⌉)

| | standalone LFM_BLAKE3 (probe) | **LFM_HASH socket arm (what runs)** | #903 syscall chip |
|---|---|---|---|
| main cols @6r | 3,056 | **2,964** | 3,219 |
| interactions @6r | 1,259 | 1,190 | 1,397 |
| **cells/compression @6r** | 4,946 | **4,749** | 5,316 |
| cells/compression @7r | 5,714 | 5,517 | n/a |

One row per compression, fully unrolled (48 G-blocks side by side). ⚠ Stale docs:
`blake3_probe.rs:392` + `phase2-report.md` say 4,741/5,509 — off by 8 (the option-C
canonicity block); the pinned tests carry the correct 4,749/5,517.

## Where the cost lives

- **The G core is 97.0%** of the socket's bill: 48 G × (60 main + 24 interactions) =
  96 base-equiv cells per G.
- **The representation is 8-bit limbs, and byte cells are 92.3% of main.** There is no
  separate range-check family — bytes are bound by the XOR lookup itself (operands and
  output), with explicit AreBytes sends only where no XOR consumes a value.
- What's already FREE and should stay free in any redesign: rotr16/rotr8 (pure byte
  relabel — the payoff of byte limbs), the message permutation (index bookkeeping, no
  copies), v-state columns (aliased, zero dedicated), add2 carries (an expression),
  the m[8] tag (a linear form over preprocessed selectors).
- The socket contract's overhead is negligible: the LFM_HASH bus = 6 of 1,190
  interactions (0.5%); 16 of 28 shared columns are dead weight (0.54% of main).

## What moves the number (the design question for the split circuit)

1. **Reshaping does NOT**: cells/compression is invariant under packing (round-per-row
   re-adds state/message carry columns; aspect ratio changes, the bill doesn't).
2. **Floor of the CURRENT primitive (byte-pair XOR table): ≈ 3,981 (−16%)** — via a
   ternary add3 carry (degree-4, only legal on an ungated dedicated chip) and halfword
   shift witnesses (needs a u16 range table).
3. **The real lever: the XOR primitive.** 64.8% of main and 64.5% of interactions are
   byte-XOR-forced. A **16-bit limb design over a 16-bit XOR lookup** lands ≈ **2,200
   cells/compression (−54%)** — the one representation change that matters, and it is a
   table-cost conversation (2^32-entry XOR table vs today's byte-pair table), i.e.
   exactly the kind of trade a dedicated split-out hash circuit can make and the
   general machine cannot.

## Prior art for the split glue: SP1 v6's deferred shards (surveyed, code-verified)

How SP1 moves precompile work (incl. keccak) into separate shards and binds them soundly
— the direct prior art for any split design:

- **SP1 does not SOLVE the cross-proof-challenge problem — it SIDESTEPS it.** Their
  LogUp challenges are per-shard, sampled after that shard's commitment; the cross-shard
  bus is a **challenge-free, group-homomorphic multiset accumulator** (hash each message
  to an EC point, negate sends, sum) — not a Schwartz–Zippel fingerprint, so nothing is
  adaptively choosable and shard proving ORDER is irrelevant. That property is what any
  port must preserve. Load-bearing caveat: the verifier's chip-cluster whitelist is what
  stops a shard from simply omitting the accumulator chip.
- **The glue is NOT a cross-shard LogUp** (that's a dead enum in v6). Every boundary-
  crossing event (syscall dispatch, memory init/finalize) is emitted twice — send in one
  shard, receive in the other — hashed to a septic-extension elliptic-curve point
  (domain-separated by kind, negated on send), summed per-shard into a public
  `global_cumulative_sum`, and the whole-proof check is one equation: Σ over shards +
  the program's memory-image digest = the fixed identity point. Aggregation re-sums it
  in-circuit; the root asserts 14 felt equalities.
- **Measured glue cost:** the Global chip is 241 columns, of which the in-circuit
  **Poseidon2 hash-to-curve is 74%**; per deferred keccak permutation the glue is
  ~12.8k cells ≈ **17% of the precompile shard**. Precompiles that stay in-shard
  ("retained": sha256, poseidon2, bn254/bls fp, u256) pay zero glue.
- **★ The design tension for OUR split:** SP1's cross-shard binding itself rests on
  Poseidon2 (the hash-to-curve). Under the no-algebraic-hash posture that motivates our
  whole blake3 direction, copying this glue re-imports the assumption we're escaping.
  A blake-consistent split needs either (a) a different accumulator (e.g. an EC digest
  with a non-algebraic hash-to-curve — costs more in-circuit), (b) LogUp-style
  cross-proof accounting with shared challenges (the commit-all-then-prove barrier our
  epoch-local FS design deliberately avoids — see streaming-proving-vs-zisk), or (c)
  accepting Poseidon2 in the GLUE only, with the assumption scoped and documented.
  This is the first decision the split design must make.
- Also transferable: SP1's per-syscall shard-sizing (row thresholds per cost table),
  the "shard is transparent to the state chain" trick (non-execution shards pin
  timestamp/pc so contiguity passes through), and the verifier's cluster allowlist
  (a shard cannot omit the glue chip).

## The backwards target: required cells/compression per fleet budget

Block 25368371, arity-4 tower, throughput anchor MEASURED (481M cells / 7.17s on one
5090). "Residue" = the machine's felt-marshalling arithmetic around the hash calls —
tracks felts absorbed, untouched by ANY hash lever (calibrated at 439 cells/felt from
the measured chip census). `--` = the residue ALONE exceeds the budget: no hash chip,
however cheap, fits at that fleet size. Fleet budgets: 4/8/16/32/64 GPUs = 3.22/6.44/
12.89/25.78/51.56 B cells/block. Reproduction: bench_cache/hash_split_2026-08-13/handover.py.

Best configuration family (2^23 epochs, batching ON):

| inner preset | RATE | comps/block | residue | required @64 GPU |
|---|---|---|---|---|
| blowup2/219q | 4 | 19.0M | 35.4 B | 849 |
| blowup2/219q | 8 | 10.2M | 35.4 B | 1,579 |
| blowup4/110q | 4 | 14.0M | 26.2 B | 1,816 |
| blowup4/110q | 8 | 7.5M | 26.2 B | 3,392 |

All 4-32 GPU cells are `--` in every configuration measured. Verified candidates to
compare against: socket 4,749 @6r / **5,517 @7r (the DEFAULT build)**; standalone chip
4,946/5,714; #903 accelerator 5,316; plausible floor within the byte-decomposed family
~4,060.

**The one-line conclusion for the table: per-compression cost is not the binding term.**
The required figures only leave `--` once batching lands AND epochs are large, and even
then they sit below every buildable candidate — what binds is the residue, i.e. the
machine's own arithmetic around the hash, which responds only to reducing FELTS
ABSORBED (narrower inner tables, fewer queries, single-row leaves — ROWS_PER_LEAF=2
verified at commitment.rs:42, halving it halves the dominant term) and to row-shape
(D12: verification cost ∝ width, invariant proving cost — tower node 104→69 GiB @RATE 8).

## Boundary conditions from the campaign (decided/measured)

- Commitment digests stay **256-bit** (128-bit truncation = 64-bit collision bound,
  below the 128-bit security floor — R2).
- **6 rounds** is the chosen variant (Mauro), −16% vs 7r; A6R ratification formally owed.
- The circuit's WIDTH is paid twice in recursion: the tower re-absorbs the hash chip's
  own trace at every layer (57% of a tower node's leaf bill today) — column count
  matters more than row count for the split circuit's downstream cost.
- RATE-4 leaf construction (in implementation) doubles felts-per-compression on the
  absorption side — orthogonal to per-compression cost; both compose.
