# LFM epoch-verifier emission: where 89 GiB goes (the "emitter audit")

> **⚠ MEASURED OUTCOME (2026-08-12, box, `97124d18`; full table
> `~/workspace/lambda_vm_bench_cache/lfm_census_2026-08-12/pc_emitter_memory_results.md`):**
> the 219q OOM point now emits — **89.06 GiB OOM → 58.36 GiB exit 0**. Attribution at
> q=96 (uncensored): **Win 2 (flat-append builder) is the ENTIRE memory win** (−15.74 GiB,
> −35.8%); **Win 1 (drop read_counts) is ~0 bytes of peak** — §1g summed the map and the
> row intermediate as co-resident, but the phases are SEQUENTIAL and the emitter's peak
> dominates — Win 1 is a WALL-TIME lever instead (−30%, it stops hashing ~446M
> addresses). The with_capacity item was deliberately not implemented (the §1g virtual-
> allocation note is correct; measured ~0). §3's caveat stands: emission no longer walls,
> P-b remains load-bearing for provability at 219q.

Delegated audit, 2026-08-12. Worktree `/Users/maurofab/workspace/lambda_vm-blake3-impl`
@ `2a8552f2`. Read-only in the tree; type sizes measured on a faithful standalone
replication of the `Instr` enum (scratchpad `rustc`, not a project build). This is the
document task #29 and PLAN.md's P-c entries cite; CENSUS.md Part 2 §2 carries the census
agent's independent (and partially superseded — see its ⚠ boxes) read.

## 0. Measured type sizes

`FE` = 8 B (`math/src/field/element.rs:50-52`, `goldilocks.rs:73`). `Instr` = **80 B**
align 8 (replicated from `prover/src/lfm/instr.rs:178-252`); largest variant `Hash`
(72 B of u64 arrays + HashMode); `KeccakOperands` boxed 432 B. 271M × 80 B = **21.7 GB —
only ~24% of the observed 89.1 GiB. The instruction vector is NOT the dominant term.**

## 1. What dominates

1a. `compile()` does NOT copy the instruction stream — REFUTED suspect
(`compiler.rs:136-223` destructures and moves; no clone).

1b. **`emit_column_groups` (`compiler.rs:231-425`) builds a second, FATTER
materialization**: ten `Vec<Vec<FE>>` — one heap allocation per instruction — all ten
alive until the struct literal at :413-424 consumes them. Measured actual capacities:
`vec![…]` + `.extend(sels)` + `.push(mult)` lands a 10-wide BALU row at **cap 18**
(144 B heap + 24 B header = 168 B) — **80% waste**; XALU cap 20; BITDEC 130-wide = 1,064 B.
An ALU instruction costs 80 B as an `Instr` and ~168-184 B as a retained row Vec.

1c. The mix engine: `felt_be_halves` (`transcript_replay.rs:743-761`) = 1 `BitDec` +
64 `BaseAlu` per leaf felt (const pool makes weights free); both leaf paths reach it
(`edsl.rs:235-246` once per value; `sub_proof.rs:245-269` 3× per ext value). ? INFERRED
mix ≈ 95% BaseAlu / 1.5% BitDec; conclusion insensitive (168→184 B at the extreme).

1d. A `BitDec` is a ~2.2 KB instruction: 80 B in the Vec + 1,024 B `bits` Vec (retained
for program life, `builder.rs:273-280`) + 1,040 B for its 130-wide row + 24 B header.

1e. ★ **`read_counts` (~18.3 GB) held alive across `emit_column_groups` by scope**
(`compiler.rs:137-143`; drained via `remove` at :155, asserted empty at :207 — but
HashMap does not shrink on removal; `emit_column_groups` is called at :213 inside the
scope). ~2 addrs/instruction → ~534M addresses → 16 B/entry + control at 7/8 load →
2^30 buckets ≈ 18.3 GB (+~27 GB transient at the last rehash). `written: vec![false]`
+0.5 GB.

1f. Arena schema: ~1.5% of instructions, <1 GB — not a factor.

1g. Budget at 219q (? INFERRED arithmetic over verified unit costs): Vec<Instr> 21.7 +
BitDec bits 4.4 + read_counts 18.3 + written 0.5 + **row intermediate ~47** + flat BALU
~20.6 + flat BITDEC ~4.4 → **peak ~99-102 GB inside `from_rows(balu_rows)`**. The
89.1 GiB OOM lands in that window (exact death point ambiguous from RSS alone).
Allocator caveat: the Vec<Instr> power-of-two capacity (2^29 slots = 42.9 GB) is virtual
until touched; glibc realloc uses mremap (no copy spike). The test binary does NOT use
jemalloc (`#[global_allocator]` only in `bin/cli/src/main.rs:11`).

## 2. Materialization

Everything is built into one `LfmBuilder.instrs` Vec (`builder.rs:97-104`, pushed at 13
sites, handed out whole by `finish()` :490-498); the 219×25 loop (`epoch_tests.rs:
1284-1341` → `epoch_verify.rs:198`, per-query loop :314-366) appends to the same
builder; nothing is ever freed. **But the query body is already streaming-shaped** —
only `fri_terminal.push` (8 B/query) escapes the iteration.

## 3. Could emission stream?

- Machine is straight-line ✓ (`instr.rs:1-8`; eleven data-op variants, no
  branch/jump/halt).
- Every consumer is a single forward scan ✓: compile pass 1 (:157),
  emit_column_groups (:243), execute (`executor.rs:227`), build_traces
  (`trace.rs:132-139`), validate (`validator.rs`, 5 passes). No consumer indexes instrs.
- The program digest commits the MATRICES, not the stream ✓
  (`registry.rs:151-203` reads only `program.groups`; `commit.rs:56`;
  `statement.rs:50-68`) — so committed bytes are identical under any emitter shape.
- Blockers: LFM_HINT group is lossy (`compiler.rs:384-386` drops arena/index → need a
  33 MB side-stream); multiplicity backfill needs an addr→(chip,row) side table
  (~4.3 GB vs the 21.7 GB stream it replaces; in-row slot recoverable since multi-write
  outputs are consecutive); groups are per-CHIP not per-leg (append rows per leg, free
  that leg's Instrs — cannot commit-and-free per leg).
- ⚠ Emission is not the only wall: even streamed, `execute` needs
  `memory: Vec<Option<LfmWord>>` ≈ 21 GB + records ~10.4 GB, and LFM_BALU pads to 2^28
  rows at 219q. Fixing emission does not make 219 queries provable — P-b remains
  load-bearing.

## 4. Verdict: (b), with two nearly-free wins first

★ Win 1 — one line, ~18.8 GB: `drop(read_counts); drop(written);` before
`compiler.rs:213`.

★ Win 2 — local, ~27 GB: replace `ColumnGroup::from_rows(width, rows: Vec<Vec<FE>>)`
(`compiler.rs:38-52`) with a flat-append `ColumnGroupBuilder { width, real_rows,
data: Vec<FE> }` — removes headers, malloc chunk overhead, the 80% capacity waste, and
~271M malloc/free pairs (~50 lines, compiler.rs-local).

Together: projected peak ~99-102 → **~53-56 GB**.

Streaming seams, named: (1) `builder.rs:98` `instrs` field + `finish()` — replace with
ten flat group matrices + read_counts + addr→(chip,row) table + hint side-stream; every
`push` becomes `emit_row`. (2) `compiler.rs:136` — passes merge into the builder.
(3) `compiler.rs:84` `LfmProgram.instrs` consumers: executor = the hard one (10-way merge
by destination address, monotone within a chip, + hint side-stream); trace.rs trivial
(recover hash modes from the one-hot MODE_* columns); validator checks re-expressed
(note `Instr::writes()/reads()` allocate a fresh Vec per call ~3×/instruction — ~800M
transient allocations — want SmallVec regardless; `check_multiplicities` builds a second
full ~18 GB HashMap that should be a dense Vec). (4) `epoch_verify.rs:198/:314` — already
streaming-correct.

Nothing about soundness, the AIR set, bus topology, or program_id moves — committed
matrices are bit-identical. Work concentrates in builder.rs + compiler.rs (mechanical)
and executor.rs (the one genuinely new algorithm).
