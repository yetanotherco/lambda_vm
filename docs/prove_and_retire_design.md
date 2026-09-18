# Prove-and-retire prover (Approach 1) — how it works and how to run it

This is the design and usage document for the prove-and-retire prover —
Approach 1 of the streaming spec (`spec/streaming.typ` at `624998db`), the
sibling of [Continuations design](./continuations_design.md), which is the
spec's Approach 2 ("prove-epoch"). Both are "streaming" in the spec's sense of
bounding the prover's memory; this one does it by re-walking the execution and
retiring tables, continuations by splitting it into epochs. It
covers what the prover does in each of its passes, the two proof formats it can
produce and when each is the right one, every knob, what verifies with what,
the numbers measured on a mainnet ethrex block, where the code lives, and the
correctness rule that every future change to the walk has to respect.

It is written to be read by a human picking this up cold.

## 1. The problem and the idea

A proof of an execution is a LogUp over many tables — the CPU, the memory
tables, the ALU chips, the preprocessed tables (BITWISE, DECODE), the
accelerators, one table per page. The monolithic prover builds every trace,
commits every one (LDE + Merkle), samples **one** LogUp challenge `(z, α)` that
ties all tables together through the bus, and only then proves each table. Until
every round-1 root exists nothing can be dropped, so the peak is the sum of all
tables: **108 GB** for a mainnet ethrex block of 30.5 M cycles.

Approach 1 accepts redoing work in exchange for not retaining it. The execution
is **walked several times**; each pass proves what it can from a table and
drops the table. What crosses from one pass to the next is small: roots,
challenges, an accumulated codeword.

## 2. The passes

"Walk" means re-executing the program and rebuilding the tables chunk by chunk
(`prover/src/pass.rs`, `trace_builder::walk_and_emit_chunks`). Tables fall into three
groups, not two. Most chunked tables (CPU, MEMW, MEMW_A, MEMW_R, LOAD, CPU32,
BRANCH, EQ, BYTEWISE, STORE — the list is `CHUNKED_KINDS`) are handed to the
pass's visitor as soon as a chunk fills. Four more — **LT, MUL, DVRM and
SHIFT** — are chunked tables in the finished proof but are *not* handed over
during the walk: later derivations keep appending to them (MEMW and HINT feed
LT, DVRM feeds LT and MUL, CPU32 feeds SHIFT, MUL and DVRM), and the finished
run concatenates each source whole, so closing them early would cut their
chunks somewhere the monolithic build does not — which would move their roots
and cost the byte-identical property of §3. Their op lists are held to the end
and chunked in `pass::finish`. The tables that cannot be chunked at all — the
preprocessed ones, the accumulators (KECCAK, ECSM, …), REGISTER, HALT, one PAGE
per page — are the *residents*, handed over when the walk ends.

So what the walk holds is one chunk per table of the first group, plus the
residents, plus the op lists of the second group and of BITWISE. The last term
grows with the execution rather than with `k`: on the ethrex block it is a low
single-digit percentage of the peak, but a workload that sends many wide memory
accesses down the general MEMW path (up to eight LT ops each, against one for
the aligned path) grows it considerably faster. §10 lists it as the known
asymptotic gap.

| pass | what it does | keeps | drops |
|---|---|---|---|
| **1 Commit** (walk) | per chunk: trace → LDE main → Merkle → root, in pipelined batches of *k* tables; the walk never waits for a batch | 245 roots, the resident traces | traces, LDEs, trees |
| **2 Challenge** | commits the residents (in parallel), absorbs the statement and every root in AIR order, samples **one `(z, α)`**, builds the AIRs once | `(z, α)`, transcript, AIRs | — |
| **3 LogUp** (walk) | per table: aux (LogUp), rounds 2–3 (β, z_ood, γ), composition. **Per-table variant**: continues with its own FRI and openings → one `StarkProof` per table. **Batched variant**: stops at the DEEP codeword | per-table proofs, or DEEP codewords folded per height | everything else |
| **4 Fold** (batched, inside pass 3) | each DEEP codeword is multiplied by a coefficient drawn from the shared transcript and added to the accumulator of its height; then one FRI + grinding + query indices per height group | 13 group FRIs (ethrex), the fold order | codewords |
| **5 Open** (walk, batched) | the query indices of a group are only known after its FRI, and its tables are gone: each table is **rebuilt** (round 1, and round 2 unless `A1_KEEP_COMPOSITION`) and opened at its group's indices | openings → `BatchedProof` | — |

The residents are proved / folded / opened as one parallel batch after each
walk; one at a time they left the machine idle.

## 3. The two variants

### Per-table (`--through logup`)

Ends at pass 3 and emits the same `MultiProof` the monolithic prover emits —
on ethrex, 245 tables with roots **byte for byte identical** to the monolithic
proof — so the existing verifier (`prover::verify`) checks it unchanged. This is
the drop-in: same proof, same verifier, one fifth of the memory, 1.27× the time.

### Batched (`--through batched`)

Continues with passes 4 and 5. Folding every DEEP codeword of one height into a
single codeword leaves **13 FRIs instead of 245** on ethrex. It is a new proof
format (`logup_phase::BatchedProof`) with its own verifier
(`prover::batched_verifier::verify`, §5). It costs 1.96× the monolithic time
(1.75× with the knob below) and buys a proof that is 57% smaller in the same
encoding and verifies in half the time with 40% fewer hashes — which is what
recursion pays for. Choose it when the proof will be verified inside a guest.

### `A1_KEEP_COMPOSITION=1` (batched only)

Pass 5 rebuilds each table to open it; the constraint evaluation (round 2) is
the costliest part of that rebuild and its output — the composition parts over
the LDE domain — was already computed in pass 3-4 and dropped. With the knob
the fold pass keeps them and pass 5 only rebuilds round 1 and re-commits the
kept parts: **−24 s for +8.6 GB** on ethrex. Off by default because it is the
memory-for-time trade the approach otherwise avoids.

## 4. Running it

```sh
# The Rust ELFs in the repo may predate a syscall the branch decodes: rebuild them.
SYSROOT_DIR=$HOME/.lambda-vm-sysroot make compile-programs-rust
cargo build --release -p cli --features jemalloc-stats   # jemalloc-stats prints the peak heap

E=executor/program_artifacts/rust/ethrex.elf
I=executor/tests/ethrex_mainnet_25368371.bin

# Per-table: the monolithic proof format, verified with the existing verifier.
./target/release/cli trace-build $E --private-input $I --prove-and-retire --through logup --verify

# Batched: one FRI per height, verified with the batched verifier.
./target/release/cli trace-build $E --private-input $I --prove-and-retire --through batched --verify
A1_KEEP_COMPOSITION=1 ./target/release/cli trace-build $E --private-input $I --prove-and-retire --through batched --verify

# Stages, for measuring one pass at a time: walk | commit | challenge | logup | batched
# --output <path> writes the per-table proof (the batched one is not serialized yet).
```

Each run prints the pass timings, `Trace build (prove-and-retire): N tables, T s`,
`Peak heap: M MB` and, with `--verify`, `A1 proof verifies: 245 tables` or
`Batched proof verifies: 245 tables in 13 groups`.

| knob | default | what it does |
|---|---|---|
| `A1_TABLE_PARALLELISM` | cores / 6, at most 16 | tables in flight per batch. The sweep recorded on `pass::table_parallelism` is flat from 1 to 16 (21550 MB) and steps at 32 (26640 MB, +5.0 GB) for 1% of time |
| `A1_PIPELINE=0` | pipelined | run each batch inline instead of on the consumer thread (diagnostic) |
| `A1_KEEP_COMPOSITION=1` | off | keep the composition parts for the Open pass (§3) |
| `A1_INFLIGHT` | 0 | full batches allowed to wait in the channel beyond the one being processed |
| `LAMBDA_STREAM_LDE` | off | `1`/`true` retires each table's main LDE after the Round 1 commit and rebuilds it on demand, and frees the leaf half of every committed Merkle tree; `auto` decides from the peak-RAM estimate (needs `disk-spill`). Off by default, and it changes this approach's memory profile too — §6's numbers were taken with it off |
| `_RJEM_MALLOC_CONF` | — | jemalloc options for experiments; the CLI's own setting is §6 |

Verification on its own: `cli verify <proof> <elf>` for a per-table proof
written with `--output`. The `hash-metrics` feature that prints a verify's
keccak count comes from #987, which is **not on this branch** — the verify-hash
column in §6 was measured on a tree that has it, and cannot be reproduced here
until this branch is rebased onto it.

## 5. What verifies, and with what

| variant | verifier | evidence |
|---|---|---|
| per-table | `prover::verify` (unchanged) | ethrex: "A1 proof verifies: 245 tables"; tables byte-identical to the monolithic proof (`examples/cmp_proofs.rs`) |
| batched | `prover::batched_verifier::verify` + `stark::batched_verifier::verify_batched` | ethrex: "Batched proof verifies: 245 tables in 13 groups"; `a_tampered_batched_proof_is_rejected` |

The batched verifier replays the transcript from the proof — statement, roots
in AIR order, `(z, α)`, the fold coefficients in the order the proof records,
per group the FRI challenges, grinding and query indices — then, per table,
runs the ordinary verifier's steps on a view of the table's data with the FRI
left empty (out-of-domain consistency, authentication of the openings at the
group's indices, reconstruction of its DEEP value there), and per group sums
those values with the coefficients and verifies the group's FRI from that first
layer down to the final polynomial. On the VM side it rebuilds the AIRs from the
layout the proof declares, checks the preprocessed roots against the AIRs'
constants and the LogUp bus balance against the public output. Its tamper test
changes one thing at a time — an out-of-domain value, an opening, the fold
order, a final-polynomial coefficient, a query index, the public output, the
layout — and requires each to be rejected while the untouched proof passes.

Still to be reviewed by someone who did not write it: the soundness of the
fold coefficient (sampled per table from the shared seed after absorbing that
table's round-3 data) and of the absorption order.

## 6. Numbers (ethrex block 25368371, 30.5 M cycles, 96 cores, blowup 2)

Taken with `LAMBDA_STREAM_LDE` off and `A1_KEEP_COMPOSITION` off except where
the row names it. `trace-build` hard-codes blowup 2 and has no `--blowup`
flag, so the batched path cannot yet be measured at the blowup-4 posture.

| | peak heap | prove | vs monolithic | proof | verify | verify hashes |
|---|---|---|---|---|---|---|
| monolithic | 107.7 GB | 88.1 s | 1× | 437 MB (838 CBOR) | 6.0 s | 19.8 M |
| **A1 per-table** | **22.9 GB** | **112 s** | **1.27×** | identical | 6.0 s | 19.8 M |
| A1 batched | 31.3 GB | 173 s | 1.96× | 360 MB CBOR (−57%) | 3.1 s | **11.9 M** |
| A1 batched + `KEEP_COMPOSITION` | 39.0 GB | 154 s | 1.75× | same | 3.0 s | 11.9 M |
| continuations 2^22 (CI bench) | 46.6 GB | 110.6 s | 1.26× | 727 MB | 11.5 s | 33.1 M |
| continuations 2^21 | 29.0 GB | 120.7 s | 1.37× | 1 045 MB | 17.8 s | 51.9 M |
| continuations 2^20 | 18.7 GB | 142.2 s | 1.61× | 1 671 MB | 30.4 s | 90.5 M |

Verify hashes are the keccak-256 finalizes the verifier does (grinding
excluded), the proxy for the recursion guest's cost. Every epoch of a
continuation carries its own fixed tables and its own FRIs, hence 1.7×–4.6× the
hashes of a single proof; the batched proof needs 0.36× of the CI bench's.

Where the time goes: prove-and-retire per-table = monolithic − trace build + round 1 twice +
pass 2 + pipeline edges, almost to the second; the profile and the core
utilization are the monolithic prover's. What is left without a protocol
change: chunking KECCAK_RND (the 3.8 s of pass 2).

Two things that were not the approach but decided its speed: jemalloc purges
every extent of 8 MiB or more the moment it is freed (`extent.c`,
`extent_may_force_decay`), which made every chunk re-fault its pages — the CLI
disables that arena's decay and purges it every 10 s
(`keep_large_buffers_warm`, Linux only, −13%); and the resident tables were
processed one at a time after each walk (−10% once batched).

## 7. Against the spec

- *Commit*, *Challenge*, *LogUp re-execution*, *FRI*, *Open*: as written.
- The spec has the Commit phase already accumulating "FRI polynomials"; nothing
  FRI-able exists before the LogUp challenge (aux and composition need `(z, α)`),
  so batching starts in the re-execution pass.
- One batch polynomial **per height**, not one in total: the fold squares the
  coset offset each layer, so codewords of different lengths do not line up
  without a mixed-height commitment (#951's direction).
- Holding each table's whole Merkle tree from the fold pass to the Open pass
  measured +9 GB for −4 s and was not kept; keeping the composition parts
  (`A1_KEEP_COMPOSITION`) is the trade that pays. The spec's Open optimization
  proper — keep the internal nodes, drop the leaves — *is* implemented
  (`MerkleTree::drop_leaves`, `get_proof_by_pos_with_leaf_sibling`,
  `TableCommit::retire_leaves`) and ships behind `LAMBDA_STREAM_LDE`.
- The per-table variant is not in the spec; it is what makes the approach a
  drop-in. Distribution across workers is not attempted; the pipelined batch
  worker is the seam for it.

## 8. Code map

```
prover/src/pass.rs                 Visitor (one table per call), Batched (channel → consumer thread,
                                   batches of k), Resident, the walk driver (walk / finish)
prover/src/commit_phase.rs         pass 1; Precomputed (the ELF-only preprocessed commitments, beside the walk)
prover/src/challenge_phase.rs      pass 2: residents in parallel, roots in AIR order, (z, α), the AIRs
prover/src/logup_phase.rs          run (per-table) · run_batched (3-4) · run_open (5) · assemble_vm_proof ·
                                   assemble_batched_proof · BatchedProof
prover/src/batched_verifier.rs     the VM half of the batched verifier
crypto/stark/src/batched_verifier.rs   the STARK half: rounds 2-3 per table, openings, accumulated DEEP,
                                   one FRI per group
crypto/stark/src/prover.rs         round_1_from_trace, rounds_2_and_3, deep_for_table, fold_coefficient,
                                   batch_fri, open_for_table / open_for_table_kept
prover/src/streaming.rs            AirOrder (the AIR order every pass and the verifier agree on)
prover/src/tables/trace_builder.rs walk_and_emit_chunks, WalkLeftover::finalize (what the shared tables owe)
bin/cli/src/main.rs                trace-build --prove-and-retire, keep_large_buffers_warm; examples/cmp_proofs.rs
```

## 9. The rule every change to the walk must respect

Several tables are fed by others: LT gets a row for every MEMW timestamp check,
for every DVRM `|r| < |d|` and for every HINT range check; BITWISE gets the
lookups of every chip including CPU32; MUL and DVRM get the CPU's derived ops
and the CPU32 dispatch; SHIFT gets the CPU32 dispatch. The monolithic build
derives all of that from complete op lists. The walk retires chunks before
those lists are complete, so **whatever a retired chunk owes another table has
to be derived when the chunk closes**, and the tail's share in
`WalkLeftover::finalize`, in the monolithic build's order. Five such gaps kept
the ethrex proof from verifying while every small-program test passed; two
tests now pin the pattern — `bitwise_multiplicities_match_the_ordinary_build`
(cell-by-cell diff of the walk's BITWISE against the ordinary build) and
`a1_verifies_with_many_chunks` (small chunks of every kind on a Rust program
with memory and ALU, proved and verified) — and `--verify` on a real block is
the last word. When adding a table or a derived lookup, `grep` every producer
of it in the monolithic build and check the walk has each.

## 10. Not done

- Chunking KECCAK_RND (protocol change; the remaining ~4 s of pass 2).
- Serializing `BatchedProof` to disk (`--output` covers the per-table proof).
- Distributing retirement batches across workers.
- An independent soundness review of the batched fold (§5).
- Bounding the op lists the walk holds whole (§2): LT, MUL, DVRM and SHIFT
  are excluded from `CHUNKED_KINDS` because their chunk boundaries are not
  knowable until the run ends, so residency is O(cycles) rather than O(chunk)
  for that term. BITWISE's lookup list is the cheaper half of the same problem
  and has no ordering constraint — a histogram is commutative, so it could be
  folded per segment without moving any root.
