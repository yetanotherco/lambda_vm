# VM Scaling Optimizations: Research Report & Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Solve lambda_vm's scaling bottlenecks by adopting proven techniques from OpenVM and ZisK, enabling proving of programs with millions of cycles.

**Architecture:** Execution-level segmentation with per-segment proving, "no-CPU" instruction routing to eliminate the CPU table bottleneck, and a recursive aggregation tree to compose segment proofs into a single proof.

**Tech Stack:** Rust, Goldilocks field (existing), STARK prover (existing), Poseidon2 for Merkle trees (new for continuations)

---

## Part 1: Research Findings

### 1.1 OpenVM Architecture (Whitepaper, April 2026)

**Field & PCS:** BabyBear (31-bit), quartic extension F[x]/(x^4-11). v1: FRI-based `TwoAdicFriPcs` from Plonky3. v2: SWIRL — multilinear PCS using WHIR + sumcheck + ZeroCheck + LogUp-GKR with "Stacked Reduction."

**FRI parameters (100-bit security):** log_blowup=1, 193 queries, 20 bits PoW grinding. Max constraint degree = 3. Hash: Poseidon2 (width=16, rate=8, digest_width=8).

**No-CPU Design (KEY INSIGHT):** OpenVM has NO central CPU chip. Instead:
- **Execution bus** carries `(pc, t)` messages. Each instruction executor chip sends `(pc_to, t_to)` and receives `(pc_from, t_from)`, constraining `t_from < t_to`.
- A **connector chip** (width=5, height=2) seeds `(pc_0, 1)` on the send side and `(pc_final, t_final)` on the receive side. Records `(initial_pc, final_pc, exit_code, is_terminate)`.
- The execution trace is **distributed across all instruction executor chips** — no single chip needs a row per cycle.
- This eliminates the "CPU table grows with total cycles" bottleneck that lambda_vm currently has.

**Adapter + Core Pattern:** Each chip splits into:
- **AdapterAir**: Handles memory reads/writes and bus interactions (system-level).
- **CoreAir**: Handles instruction-specific arithmetic constraints.
- Trace columns concatenated: `[adapter_cols | core_cols]`. This allows reuse of memory-access logic.

**RV32IM Extension Chips (~15 chips):** base_alu (ADD/SUB/AND/OR/XOR), branch_eq (BEQ/BNE), branch_lt (BLT/BGE/BLTU/BGEU), less_than (SLT/SLTU), shift (SLL/SRL/SRA), mul, mulh, divrem, jal_lui, jalr, auipc, loadstore (LW/SW), load_sign_extend (LB/LH/LBU/LHU), hintstore.

**System Chips:** VmConnector, ProgramChip, VolatileBoundary/PersistentBoundary, MemoryMerkle, Poseidon2Periphery, AccessAdapters (N=2,4,8,16,32), RangeChecker.

**Memory Bus:** Offline memory checking [BEG+94]. Messages `(addr_space, ptr, data, t)`. Two modes:
- **Volatile** (single-segment): VolatileBoundaryChip sorts all touched (addr_space, pointer) pairs, enforces monotonic timestamps.
- **Persistent** (continuations): PersistentBoundaryChip + MemoryMerkleChip. Poseidon2 Merkle tree for state commitment. Merkle chip verifies inclusion proofs for accessed addresses.
- **Access Adapters** for width alignment: chips for block sizes 2,4,8,16,32 split/merge memory accesses.

**Continuations (KEY INSIGHT):**
- **Segmentation thresholds** (checked every 1000 instructions): (1) any chip's padded height > max_trace_height (default 2^22), (2) total cells > 1.2B, (3) total interactions > ~2^31. Checkpoint mechanism rolls back to last safe state.
- Each segment is `Segment { instret_start, num_insns, trace_heights }`.
- Each segment proof is **self-contained** — no inter-segment communication during proving.
- Boundary state committed via: (1) Program ROM as cached trace, (2) `pc_0`/`pc_final` from connector chip, (3) **Merkle roots** of initial/final memory states from boundary chip.
- **Aggregation tree:** App VM Segments → Leaf Verifiers (verify multiple segments, check boundary consistency) → Internal Verifiers (recursive) → Root Verifier → Halo2 SNARK → EVM.
- **Optimization:** Boundary chip only adds messages for addresses accessed in the segment. Merkle multi-proofs combine multiple leaf inclusions with shared witness data.

**Three-Phase Execution Pipeline:**
1. *Pure execution*: Fast state computation (150 MHz CPU, 3.8 GHz with AOT in v2.0)
2. *Metered execution*: Determines segment boundaries
3. *Preflight execution*: Generates minimal records for trace generation (embarrassingly parallel)

**Performance:** v1.4.0: <$0.0003/tx in 15s on GPU. v2.0: 11.4 MHz single 5090, 139 MHz on 16x 5090 cluster.

### 1.2 ZisK Architecture

**Field & PCS:** Goldilocks (p = 2^64 - 2^32 + 1, same as lambda_vm). FRI-based PCS. Hash: Poseidon2 for Merkle commitments.

**Execution Speed:** AOT compilation (RISC-V → native x86_64) achieves 1.5GHz — only 3-4 x86 instructions per RISC-V instruction.

**Minimal Trace Architecture (KEY INSIGHT):** The fast binary produces only:
- **Memory Read Log:** Sequential list of every value read from memory.
- **Register Checkpoints:** CPU register snapshots at intervals (e.g., every 1M cycles).
This avoids I/O bottlenecks and enables parallel witness generation.

**Parallel Witness ("Memoryless Re-execution"):** Workers re-execute their assigned segment using local register state from checkpoints and values from the global Memory Read Log instead of actual memory. Each worker independently generates full witness for its chunk.

**State Machines (PIL2-defined):**

| State Machine | Purpose |
|---|---|
| **main** | Primary CPU execution; plans segments, maps opcodes to rows |
| **arith** | Arithmetic ops; has range tables and FrequentOps tables |
| **binary** | Bitwise ops; split into basic/extension, both have frops variants |
| **mem** | Memory consistency; sorted address ordering, alignment, range checks (22 source files) |
| **rom** | Read-only memory / program ROM |
| **frequent-ops** | Precomputed lookup tables for common operations |
| Precompiles | keccakf, sha256f, poseidon2, blake2, arith_eq, big_int, dma |

**FrequentOps Virtual Tables (KEY INSIGHT):**
- 40 opcodes precomputed for small operands (a,b < 386): 148,996 entries per opcode.
- Covers: SignExtend, Add/Sub/Mul/Div/Rem (all variants), Sll/Sra/Srl, Eq/Lt/Le, And/Or/Xor.
- **Virtual table** mechanism: table is used for lookups but columns are NOT committed. Only multiplicities collected. Saves commitment + FRI work.

**VADCOPs (Variable-Degree Composite Proofs):** Allow splitting a large trace that would exceed row limits into multiple smaller traces with different row counts, then aggregating. Eliminates need for zkCounters (pre-counting operations to fit fixed-size tables).

**Memory Model:** Sorted by (address, step). Range checks for address/step deltas via LogUp lookups (22-bit, 16-bit, 24-bit). `MemPreviousSegment` struct tracks last `(addr, step, value)` from preceding segment for cross-segment continuity.

**PIL2 Standard Library:** std_lookup (LogUp), std_permutation (sum or product), std_connection (wiring), std_range_check (U8/U16/specified), std_virtual_table (uncommitted lookups), std_sum (LogUp bus), std_prod (grand product).

**Proof Aggregation:** Per-airgroup STARK proofs → recursion/aggregation/composition → STARK-to-SNARK wrapping (Groth16 for on-chain verification).

**Distributed Proving:** gRPC coordinator-worker model. Three phases: Partial Contributions → Prove (global challenge + partial proofs) → Aggregation (first finisher becomes aggregator). Supports Docker + GPU (`--preallocate`, `--max-streams`).

**GPU Proving:** 24x RTX 5090 proves 99.74% of Ethereum blocks under 12s (avg 6.56s). Venus hardware backend: CUDA Graphs + FPGA kernels for Goldilocks NTT/Poseidon2/Merkle/FRI.

### 1.3 Lambda VM Current State

**Architecture:** RV64IM, Goldilocks field (64-bit), cubic extension, 12+ tables with LogUp.

**Current Table Decomposition (after recent MEMW split):**

| Table    | Main cols | Bus interactions | Eff. width | Max rows/chunk |
|----------|-----------|-----------------|------------|----------------|
| CPU      | 74        | 40              | 194        | 2^19           |
| MEMW     | 49        | 26              | 127        | 2^19           |
| MEMW_A   | 29        | 20              | 89         | 2^19           |
| DVRM     | 34        | 34              | 136        | 2^19           |
| MUL      | 26        | 16              | 74         | 2^20           |
| SHIFT    | 27        | 15              | 72         | 2^20           |
| LT       | 15        | 9               | 42         | 2^21           |
| LOAD     | 18        | 5               | 33         | 2^21           |
| BRANCH   | 14        | 6               | 32         | 2^21           |
| MEMW_R   | 10        | 7               | 31         | 2^21           |
| Bitwise  | 22        | (precomputed)   | —          | 2^20 (fixed)   |

**Chunking exists but is not segmentation:** Tables are split into chunks when they exceed `max_rows`, but all chunks are proved in a single `multi_prove` call. This is per-table overflow handling, not execution segmentation.

**What's Already Done Well:**
- Sequential per-table proving (P3.1) — memory-efficient, one table at a time
- Evaluation-form STARK (Phases 1-3 complete)
- LogUp interaction batching (83→42 term columns, 67→55 aux columns)
- MEMW split into 3 specialized tables (MEMW, MEMW_A, MEMW_R)
- Fused coset LDE, buffer pool reuse, twiddle caching
- 3-layer butterfly FFT fusion

**Current Scaling Bottlenecks:**

1. **CPU table is the primary bottleneck:** 74 cols + 40 interactions = 194 effective width. Every instruction generates exactly one CPU row. A program with 1M cycles = 1M CPU rows = ~2 chunks at 2^19. The CPU table dominates proving cost.

2. **No execution segmentation:** Cannot split a 10M-cycle program into independent segments proved in parallel on different machines. The entire program must be proved by one prover instance.

3. **No proof aggregation:** Cannot recursively combine segment proofs. Every chunk is part of one monolithic multi-proof.

4. **Bitwise table fixed at 2^20:** Always committed regardless of how many lookups are needed. Wastes commitment/FRI resources for small programs.

5. **Memory footprint:** Even with sequential proving, the prover must hold all trace data in memory during trace generation (all operation vectors before chunking).

---

## Part 2: Optimization Opportunities (Ranked by Impact)

### Tier 1: Architectural (Highest Impact, Highest Effort)

#### O1: Execution Segmentation (Continuations)
**From:** OpenVM §5, ZisK distributed proving
**Impact:** Enables horizontal scaling. A 10M-cycle program becomes 10 independent 1M-cycle segment proofs that can be proved in parallel.
**Approach:**
- During execution, emit register checkpoints every N cycles (configurable, e.g., 2^19).
- Each segment proves cycles [i*N, (i+1)*N) independently.
- Boundary state: `(pc, registers, memory_merkle_root)` at segment start/end.
- Memory consistency across segments via Merkle tree of memory state.
- Requires Poseidon2 (or similar ZK-friendly hash) for in-circuit Merkle verification.
**Prerequisites:** Boundary chip, Poseidon2 chip, memory Merkle tree.
**Estimated effort:** Large (4-6 weeks).

#### O2: No-CPU Architecture (Eliminate CPU Table)
**From:** OpenVM §4.5
**Impact:** Eliminates the widest table (194 eff. width). Proving cost distributed across instruction-specific chips.
**Approach:**
- Replace CPU table with an **execution bus** `(pc, timestamp)`.
- Each instruction chip (ADD, BRANCH, LOAD, etc.) handles its own `(pc_from, t_from)` → `(pc_to, t_to)` transitions.
- Connector chip seeds initial/final `(pc, t)`.
- Each instruction chip only has rows for instructions it handles — a program that's 90% ADD has a large ADD chip but small BRANCH/MUL chips.
**Risk:** Major refactor. All constraints and trace generation need restructuring.
**Estimated effort:** Very large (6-8 weeks). Consider as Phase 2 after segmentation works.

#### O3: Recursive Proof Aggregation
**From:** OpenVM §5.3-5.4
**Impact:** Enables composing segment proofs into a single proof. Required for on-chain verification.
**Approach:**
- Build a STARK verifier circuit (verify lambda_vm proofs inside lambda_vm).
- Aggregation tree: Segments → Leaf Verifier → Internal Verifier → Root Verifier.
- Leaf verifier checks boundary state consistency between consecutive segments.
**Prerequisites:** O1 (segmentation), STARK verifier as a guest program or native circuit.
**Estimated effort:** Very large (8+ weeks).

### Tier 2: Medium Impact, Medium Effort

#### O4: Dynamic Bitwise Table
**From:** Performance plan (existing), similar to ZisK's on-demand tables
**Impact:** Small programs don't pay the 2^20-row bitwise commitment cost. Programs that need few bitwise ops save significant proving time.
**Approach:**
- Build bitwise table lazily from actual lookups instead of precomputing all 2^20 rows.
- Only include rows that are actually looked up + enough padding for power-of-2.
- For programs with <2^16 unique bitwise lookups, this saves ~16x commitment work.
**Risk:** Low — purely an optimization, no protocol changes.
**Estimated effort:** Small (1 week).

#### O5: Parallel Trace Generation
**From:** ZisK's "memoryless re-execution"
**Impact:** Trace generation is currently sequential (phases 0-5). Parallelizing the heaviest phases could cut trace gen time by 2-4x.
**Approach:**
- Phase 1 (CPU ops from logs) is embarrassingly parallel — each log entry is independent.
- Phase 2 (MEMW/LOAD/LT ops) has memory state dependencies but can be parallelized by address range.
- Phase 5 (generate traces) can use `par_iter` over chunks.
**Risk:** Memory state tracking in Phase 2 is the hard part.
**Estimated effort:** Medium (2 weeks).

#### O6: CPU Table Width Reduction
**From:** Observation that CPU has 74 main + 40 bus = 194 effective width
**Impact:** Even without eliminating CPU, reducing its width saves proportional proving cost.
**Approach:**
- Factor out instruction-specific columns into dedicated sub-tables (like the MEMW split).
- CPU keeps: `pc`, `opcode`, `timestamp`, `operand_routing`, and bus interactions to instruction-specific chips.
- E.g., ALU-specific columns (result decomposition, flags) move to an ALU chip.
- Target: CPU at ~30-40 main cols + ~15 interactions = ~75 effective width (2.6x reduction).
**Risk:** Medium — requires careful bus interaction redesign.
**Estimated effort:** Medium-large (3-4 weeks).

#### O7: Streaming Trace Generation (Reduce Peak Memory)
**From:** ZisK's minimal trace, OpenVM's metered execution
**Impact:** Currently all operation vectors are held in memory before chunking. For large programs this is GBs of data.
**Approach:**
- Stream operations directly into chunk-sized trace tables instead of collecting all ops first.
- Process logs in a single pass, emitting filled trace tables as they complete.
- Prove each chunk immediately (or queue for proving) instead of holding all chunks.
**Risk:** Requires restructuring the phased collection in trace_builder.rs.
**Estimated effort:** Medium (2-3 weeks).

#### O7b: Virtual Tables for Lookups (Uncommitted Lookup Tables)
**From:** ZisK's `std_virtual_table.pil`
**Impact:** Lookup receiver tables (like bitwise) collect multiplicities but the precomputed columns are NOT committed. Only multiplicities go through commitment/FRI. This saves ~50% of commitment work for preprocessed tables.
**Approach:**
- For preprocessed tables (bitwise, decode, page, register), the precomputed columns are agreed upon between prover and verifier.
- Currently we commit precomputed + multiplicity columns. Instead, only commit multiplicity columns and hardcode the precomputed data in the AIR/verifier.
- This is partially done already (preprocessed tables have `precomputed_commitment`), but the full "virtual table" approach would skip FRI openings on precomputed columns entirely.
**Risk:** Low — requires verifier changes to reconstruct precomputed openings.
**Estimated effort:** Medium (2 weeks).

### Tier 3: Lower Effort, Incremental Impact

#### O8: Memory Read Log for Re-execution
**From:** ZisK's AOT + memory read log
**Impact:** Faster re-execution for segmented proving. Instead of re-executing from scratch, use logged memory reads.
**Prerequisite:** O1 (segmentation).
**Estimated effort:** Small (1 week) after O1 is done.

#### O9: Instruction-Frequency-Aware Table Sizing
**From:** Observation that different programs have very different instruction mixes
**Impact:** Programs heavy on branches but light on MUL waste resources on empty MUL chunks.
**Approach:**
- After execution, count operations per type and allocate chunk sizes proportionally.
- Skip generating empty tables (currently even 0-op tables generate a 4-row minimum).
**Risk:** Verifier must handle variable table counts (already does via TableCounts).
**Estimated effort:** Small (1 week).

---

## Part 3: Recommended Implementation Order

```
Phase 1 (Immediate, 2-4 weeks):
  O4: Dynamic Bitwise Table         — low risk, immediate win
  O6: CPU Table Width Reduction      — reduces the #1 bottleneck
  O9: Instruction-Frequency Sizing   — easy win

Phase 2 (Short-term, 4-8 weeks):
  O5: Parallel Trace Generation      — CPU-bound improvement
  O7: Streaming Trace Generation     — memory-bound improvement
  O1: Execution Segmentation (design + prototype)

Phase 3 (Medium-term, 8-16 weeks):
  O1: Execution Segmentation (full implementation)
  O2: No-CPU Architecture (design)
  O3: Recursive Proof Aggregation (design)

Phase 4 (Long-term):
  O2: No-CPU Architecture (full)
  O3: Recursive Proof Aggregation (full)
  O8: Memory Read Log
```

---

## Part 4: Detailed Implementation Tasks (Phase 1)

### Task 1: Dynamic Bitwise Table (O4)

**Context:** The bitwise table is currently always 2^20 rows = 1,048,576 rows with 22 columns. It's a precomputed lookup table covering all possible (a, b, op) → result for 8-bit inputs. For small programs that only use a few hundred distinct bitwise lookups, this is massively wasteful.

**Files:**
- Modify: `prover/src/tables/bitwise.rs` — add `generate_dynamic_bitwise_trace`
- Modify: `prover/src/tables/trace_builder.rs` — collect actual bitwise lookups, generate dynamic table
- Modify: `prover/src/lib.rs` — update `VmAirs::new` to handle dynamic bitwise sizing
- Test: `prover/src/tests/bitwise_bus_tests.rs` — add dynamic bitwise tests

**Current behavior** (`prover/src/tables/bitwise.rs`): `generate_bitwise_trace()` generates all 2^20 rows unconditionally. The table is treated as preprocessed with a hardcoded commitment.

**New behavior:** When the number of unique bitwise lookups is small, generate only the needed rows (padded to next power of 2). When lookups cover most of the table, fall back to full precomputed table.

- [ ] **Step 1: Write failing test for dynamic bitwise trace generation**

In `prover/src/tests/bitwise_bus_tests.rs`, add a test that generates a bitwise trace from a small set of operations and verifies the trace only contains those operations (plus padding):

```rust
#[test]
fn test_dynamic_bitwise_trace_small() {
    // Only 3 distinct lookups
    let lookups: Vec<(u8, u8, u8)> = vec![
        (0x0F, 0xF0, bitwise::OP_AND),
        (0xFF, 0x00, bitwise::OP_OR),
        (0xAA, 0x55, bitwise::OP_XOR),
    ];
    let trace = bitwise::generate_dynamic_bitwise_trace(&lookups);
    // Should be padded to next power of 2 (>= 4 rows), NOT 2^20
    assert!(trace.num_rows() <= 8, "Dynamic trace should be small, got {} rows", trace.num_rows());
    assert!(trace.num_rows() >= 4, "Minimum trace size is 4 rows");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p lambda_vm_prover test_dynamic_bitwise_trace_small -- --nocapture`
Expected: FAIL — `generate_dynamic_bitwise_trace` doesn't exist yet.

- [ ] **Step 3: Implement `generate_dynamic_bitwise_trace`**

In `prover/src/tables/bitwise.rs`, add:

```rust
/// Generate a bitwise trace containing only the rows needed for the given lookups.
/// Each lookup is (a, b, op_type) where op_type is OP_AND/OP_OR/OP_XOR.
/// The trace is padded to the next power of 2 (minimum 4 rows).
/// Rows are sorted by (op, a, b) to match the preprocessed table's ordering.
pub fn generate_dynamic_bitwise_trace(
    lookups: &[(u8, u8, u8)],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    use std::collections::BTreeSet;

    // Collect unique (op, a, b) tuples, sorted
    let mut unique: BTreeSet<(u8, u8, u8)> = BTreeSet::new();
    for &(a, b, op) in lookups {
        unique.insert((op, a, b));
    }

    let n_rows = unique.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); n_rows * cols::NUM_COLUMNS];

    for (row_idx, &(op, a, b)) in unique.iter().enumerate() {
        let offset = row_idx * cols::NUM_COLUMNS;
        fill_bitwise_row(&mut data, offset, a, b, op);
    }
    // Padding rows are all-zero (valid: 0 AND 0 = 0, with multiplicity 0)

    TraceTable::new_main(data, cols::NUM_COLUMNS)
}
```

The `fill_bitwise_row` helper extracts the existing row-fill logic from `generate_bitwise_trace`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p lambda_vm_prover test_dynamic_bitwise_trace_small -- --nocapture`
Expected: PASS

- [ ] **Step 5: Wire dynamic bitwise into trace builder**

In `prover/src/tables/trace_builder.rs`, modify `from_elf_and_logs` to collect unique bitwise lookups during Phase 4 and choose dynamic vs full table:

```rust
// In Phase 5 trace generation:
let bitwise = if bitwise_lookups.len() < (1 << 16) {
    // Dynamic: only include needed rows
    bitwise::generate_dynamic_bitwise_trace(&bitwise_lookups)
} else {
    // Full precomputed table (too many unique lookups to benefit from dynamic)
    bitwise::generate_bitwise_trace()
};
```

- [ ] **Step 6: Update VmAirs to handle dynamic bitwise (non-preprocessed)**

In `prover/src/lib.rs`, when the bitwise table is dynamic, it should NOT use a precomputed commitment. Add a `bitwise_is_dynamic: bool` flag to `VmAirs` and conditionally skip the preprocessed commitment.

- [ ] **Step 7: Add integration test with prove_and_verify using dynamic bitwise**

```rust
#[test]
fn test_prove_verify_dynamic_bitwise() {
    // Use a small program that only needs a few bitwise ops
    let elf_bytes = asm_elf_bytes("test_bitwise_8");
    let result = crate::prove_and_verify(&elf_bytes);
    assert!(result.is_ok());
    assert!(result.unwrap());
}
```

- [ ] **Step 8: Run full test suite**

Run: `cargo test -p lambda_vm_prover -- --nocapture`
Expected: All tests pass.

- [ ] **Step 9: Commit**

```bash
git add prover/src/tables/bitwise.rs prover/src/tables/trace_builder.rs prover/src/lib.rs prover/src/tests/bitwise_bus_tests.rs
git commit -m "feat: dynamic bitwise table — only generate needed rows for small programs"
```

---

### Task 2: CPU Table Width Reduction — Factor Out ALU Columns (O6, Part 1)

**Context:** The CPU table has 74 main columns and 40 bus interactions (194 effective width). Many columns are instruction-specific. The strategy: move instruction-specific computation columns out of CPU into dedicated sub-tables that CPU routes to via bus interactions. This follows the same pattern used for MEMW → MEMW_A/MEMW_R split.

**Analysis of CPU columns that could be factored out:**

Looking at `prover/src/tables/cpu.rs` and `prover/src/constraints/cpu.rs`:
- Sign extension columns (sign bits, extension bytes)
- Result decomposition columns for ALU ops
- Immediate value decomposition
- Condition evaluation columns for branches

**This is a design task first.** Before implementing, we need to:

- [ ] **Step 1: Audit CPU columns and identify factoring candidates**

Read `prover/src/tables/cpu.rs` columns module and categorize each column as:
- **Core** (needed for every instruction): pc, opcode, timestamp, operands, etc.
- **ALU-specific**: result decomposition, overflow flags, sign bits
- **Memory-specific**: already factored out to MEMW/MEMW_A/MEMW_R
- **Branch-specific**: already factored to BRANCH table
- **Other**: instruction-specific columns that could move to dedicated chips

Document the classification and propose which columns to factor out.

- [ ] **Step 2: Design the factoring — which new sub-tables, what bus interactions**

For each group of columns being factored out:
- Define the new table's column layout
- Define the bus interaction between CPU and the new table
- Estimate the new effective widths
- Verify the total effective width decreases (accounting for new bus interactions on both sides)

- [ ] **Step 3: Document the design in a markdown file**

Save to `docs/superpowers/plans/2026-04-13-cpu-table-split-design.md`.

- [ ] **Step 4: Implement the highest-value factoring first (sign/immediate columns)**

This will be a separate follow-up task once the design is approved.

---

### Task 3: Execution Segmentation Design (O1)

**Context:** This is the most impactful optimization. The goal is to split a long execution into independent segments that can be proved in parallel.

**Design decisions needed:**

1. **Segment sizing:** Fixed cycle count per segment (e.g., 2^19 cycles)? Or adaptive based on memory usage?

2. **Boundary state:** What state is committed at segment boundaries?
   - Minimum: `(pc, registers[0..31], memory_merkle_root)`
   - OpenVM approach: Merkle tree of all memory, boundary chip adds messages only for accessed addresses

3. **Memory Merkle tree hash:** Need a ZK-friendly hash for in-circuit verification. Options:
   - Poseidon2 over Goldilocks (fast in-circuit, slower out-of-circuit)
   - Keep Keccak256 for out-of-circuit tree, only prove inclusion via bus

4. **Cross-segment memory consistency:** Two approaches:
   - **OpenVM style:** Each segment commits Merkle roots of initial/final memory. Aggregation circuit checks `final_root[i] == initial_root[i+1]`.
   - **Sorted memory style:** Sort all memory accesses by address, then by timestamp. Verify ordering across segments.

5. **Aggregation strategy:** How are segment proofs composed?
   - Recursive STARK verification (requires STARK verifier circuit)
   - Halo2 wrapping (requires Halo2 integration)
   - Deferred: just prove segments, compose later

**This task produces a design document, not code.**

- [ ] **Step 1: Analyze execution patterns in current test programs**

Profile how execution distributes across tables for `bench_32k`, `fib_iterative_372k`, and `all_instructions_64`. Count cycles, memory accesses, register accesses per segment window.

- [ ] **Step 2: Design the segmentation protocol**

Define:
- Segment boundary state format
- How the executor emits segment metadata
- How the prover generates per-segment traces
- How segment proofs relate to each other (public values linking)
- What changes in the verifier

- [ ] **Step 3: Design the boundary chip**

Define:
- Column layout
- Bus interactions (memory bus send/receive for initial/final state)
- Constraints
- How Merkle proofs are handled (as hint data? as auxiliary columns?)

- [ ] **Step 4: Estimate resource impact**

For a 1M-cycle program split into 4 segments:
- Per-segment table sizes (smaller = faster individual proofs)
- Boundary chip overhead (Merkle proof columns)
- Total proving cost vs monolithic (should be lower due to parallelism)
- Memory usage per segment prover

- [ ] **Step 5: Document the complete design**

Save to `docs/superpowers/plans/2026-04-13-execution-segmentation-design.md`.

---

### Task 4: Instruction-Frequency-Aware Table Sizing (O9)

**Context:** Currently, even tables with 0 operations generate a minimum trace (4 rows). For programs that don't use MUL/DVRM/SHIFT, these empty tables still cost commitment + FRI resources.

**Files:**
- Modify: `prover/src/tables/trace_builder.rs` — skip empty tables
- Modify: `prover/src/lib.rs` — handle TableCounts with 0 for unused tables
- Test: `prover/src/tests/prove_elfs_tests.rs` — test with programs that skip tables

- [ ] **Step 1: Analyze which tables can be empty**

A program that only does ADD and memory access doesn't need MUL, DVRM, or SHIFT tables. Currently `TableCounts::validate()` rejects count=0 for any table. We need to decide which tables are truly required vs optional.

Required (every program): CPU, MEMW or MEMW_A (at least one memory table), DECODE, HALT, REGISTER
Optional: MUL, DVRM, SHIFT, BRANCH, LOAD, LT (if no MEMW timestamp checks)

- [ ] **Step 2: Write test for program with no MUL/DVRM operations**

```rust
#[test]
fn test_prove_verify_no_mul_dvrm() {
    // test_add_8.elf only uses ADD — no MUL or DVRM
    let elf_bytes = asm_elf_bytes("test_add_8");
    let result = crate::prove_and_verify(&elf_bytes);
    assert!(result.is_ok());
    assert!(result.unwrap());
}
```

- [ ] **Step 3: Modify trace builder to skip empty operation tables**

In `chunk_and_generate`, when `ops` is empty AND the table is optional, return an empty `Vec` instead of generating a dummy table:

```rust
fn chunk_and_generate_optional<T>(
    ops: &[T],
    max_rows: usize,
    generate: impl Fn(&[T]) -> TraceTable<GoldilocksField, GoldilocksExtension>,
) -> Vec<TraceTable<GoldilocksField, GoldilocksExtension>> {
    if ops.is_empty() {
        vec![] // Skip entirely for optional tables
    } else {
        ops.chunks(max_rows).map(generate).collect()
    }
}
```

- [ ] **Step 4: Update TableCounts::validate to allow 0 for optional tables**

```rust
pub fn validate(&self) -> Result<(), Error> {
    // Required tables must have at least 1 chunk
    let required = [("cpu", self.cpu)];
    for (name, count) in required {
        if count == 0 {
            return Err(Error::InvalidTableCounts(format!("{name} count is 0")));
        }
    }
    // Optional tables can be 0
    Ok(())
}
```

- [ ] **Step 5: Update VmAirs to conditionally create AIRs for optional tables**

Skip creating AIR + trace pairs for tables with 0 chunks. Update `air_trace_pairs()` and `air_refs()`.

- [ ] **Step 6: Run full test suite**

Run: `cargo test -p lambda_vm_prover -- --nocapture`
Expected: All tests pass, including existing tests (which still generate all tables).

- [ ] **Step 7: Commit**

```bash
git add prover/src/tables/trace_builder.rs prover/src/lib.rs prover/src/tests/
git commit -m "feat: skip empty optional tables — no MUL/DVRM/SHIFT overhead when unused"
```

---

## Part 5: Key Architectural Decisions Needed

Before proceeding with Phase 2+, the team needs to decide:

1. **Segmentation vs No-CPU: which first?**
   - Segmentation (O1) is more impactful for scaling (enables parallelism) but requires Merkle tree machinery.
   - No-CPU (O2) reduces per-proof cost but is a massive refactor.
   - **Recommendation:** Segmentation first. It's orthogonal to the table structure and provides immediate horizontal scaling.

2. **In-circuit hash function for Merkle trees:**
   - Poseidon2 is the industry standard (OpenVM, SP1, Plonky3 all use it).
   - Implementing Poseidon2 as a chip in lambda_vm is significant work.
   - Alternative: Use the existing Keccak-based Merkle tree for out-of-circuit work, and defer in-circuit Merkle verification to the aggregation phase.

3. **Field migration (Goldilocks → BabyBear):**
   - OpenVM/SP1 use BabyBear (31-bit). ZisK uses Goldilocks (64-bit).
   - BabyBear is 2x narrower → less memory per element, but requires degree-4 extension (vs cubic for Goldilocks).
   - **Recommendation:** Stay on Goldilocks for now. Field migration is a full rewrite with no partial migration path. ZisK proves it works well at scale.

4. **LogUp-GKR (eliminate aux columns entirely):**
   - OpenVM's SWIRL uses LogUp-GKR, which proves lookup arguments via GKR protocol instead of auxiliary columns.
   - This would eliminate all aux columns (currently ~55 across tables), saving ~40% of commitment work.
   - **Recommendation:** Investigate as Phase 3 optimization after segmentation is working.

5. **OpenVM segmentation reference values** (for calibrating our own thresholds):
   - Max trace height per chip: 2^22 (~4M rows)
   - Max total cells across all chips: 1.2B field elements
   - Max total interactions: ~2^31
   - Check interval: every 1000 instructions
   - Checkpoint/rollback mechanism to ensure no chip overshoots

6. **ZisK FrequentOps pattern** (precomputed results for small operands):
   - 40 opcodes precomputed for a,b < 386 → 148,996 entries per opcode
   - Virtual (uncommitted) tables save commitment work
   - Could be adopted for lambda_vm's most common operations (ADD, SUB, AND, OR, XOR with small immediates)
