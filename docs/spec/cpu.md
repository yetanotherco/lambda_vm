# CPU Chip

The  chip coordinates memory accesses and dispatches to other chips for arithmetic and logical operations. It bases its decisions on the entry of the `DECODE` table ([decode]) corresponding the current program counter (PC).

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `Timestamp` | A preprocessed timestamp to coordinate the memory argument. Since we have at most 3 non-disjoint memory accesses (`(rs1, rs2, rd)`, `(rs1, pc, pc)`, `MEMORY`) a maximum of 4 slots is enough. |
| `pc` | `DWordWL` | The program counter |
| `rs1` | `Byte` | Source register 1 index |
| `rs2` | `Byte` | Source register 2 index |
| `rd` | `Byte` | Destination register index |
| `read_register1` | `Bit` | Whether to read from `rs1` (1) or to place a 0 in `rv1` (0) |
| `read_register2` | `Bit` | Whether to read from `rs2` (1) or to place a 0 in `rv2` (0) |
| `write_register` | `Bit` | Whether to write back to the destination register |
| `imm` | `DWordWL` | The fully extended 64-bit version of the immediate |
| `half_instruction_length` | `Byte` | Half the number of bytes consumed by this instruction, commonly used to indicate whether the instruction is of C type, i.e., whether it is 2 bytes long (= 1) instead of 4 (= 2) |
| `word_instr` | `Bit` | Whether the instruction is a \*W instruction, requiring the inputs and outputs to be (sign) extended |
| `ALU` | `Bit` | Whether to use the ALU for this instruction |
| `alu_flags` | `Byte` | The ALU operation + flags (interpreting things as signed/unsigned, choosing the MUL/DVRM output, ...) to pass to the ALU |
| `ADD` | `Bit` | Addition fast-path bypassing the ALU |
| `SUB` | `Bit` | Subtraction fast-path bypassing the ALU |
| `MEMORY` | `Bit` | Whether this instruction touches memory (LOAD/STORE) |
| `mem_flags` | `Byte` | The flags to pass for MEMORY operations (LOAD vs STORE, number of bytes touched, signed) |
| `BRANCH` | `Bit` | Whether this instruction is a conditional branch (BLT, BEQ) |
| `ECALL` | `Bit` | Whether this instruction is an ECALL |

### Output

| Name | Type | Description |
|------|------|-------------|
| `next_pc` | `DWordWL` | The program counter for the next instruction |
| `rvd` | `DWordWL` | The value to (maybe) be written back to rvd |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `prev_pc_timestamp_borrow` | `Bit` | The borrow bit for computing the previous timestamp the PC was accessed |
| `pc_double_read` | `Bit` | Whether the PC is being read as a general purpose register (`rs1`) this cycle |
| `rv1` | `DWordWL` | The value of register `rs1` |
| `rv2` | `DWordWL` | The value of register `rs2` |
| `arg2` | `DWordWL` | A multiplexed version of `rv2` and `imm`, to be used as second argument to ALU calls |
| `res` | `DWordHL` | The ALU result |
| `branch_cond` | `Bit` | Whether a branch is taken: the branch condition evaluates to true, or we are doing an unconditional jump |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `JALR` | `Bit` | Read whether our BRANCH corresponds to a JAL(R) instruction from `mem_flags`, as `MEMORY` and `BRANCH` are mutually exclusive |
| `packed_decode` | `BaseField` | A packed representation of all bit flags and register indices obtained from the decoding |

**Definition of `JALR`:**
```
JALR := mem_flags
```

**Definition of `packed_decode`:**
```
packed_decode := 2^0 * read_register1 + 2^1 * read_register2 + 2^2 * write_register + 2^3 * word_instr + 2^4 * ALU + 2^5 * ADD + 2^6 * SUB + 2^7 * MEMORY + 2^8 * BRANCH + 2^9 * ECALL + 2^10 * rs1 + 2^18 * rs2 + 2^26 * rd + 2^34 * half_instruction_length + 2^42 * alu_flags + 2^50 * mem_flags
```

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `CPU-A1` |  | `MEMORY` and `BRANCH` are mutually exclusive |
| `CPU-A2` |  | When `MEMORY + BRANCH = 0`, either `read_register2 = 0` or `imm = 0` should be enforced by the decoding. This is needed for `arg2`. |
| `CPU-A3` |  | $#`!MEMORY` => #`IS_BIT<mem_flags>`$ |

Additionally, the following constraints can be used to provide defense-in-depth validation of the assumptions.

| Tag | Description |
|-----|-------------|
| `CPU-C1` | not (`MEMORY` and `BRANCH`) |
| | _polynomial:_ `MEMORY * BRANCH = 0` |
| `CPU-C2` | (1 - `MEMORY` - `BRANCH`) => (`read_register2` = 0 or `imm[i]` = 0) |
| | _polynomial:_ `(1 - MEMORY - BRANCH) * read_register2 * (imm[0] + imm[1]) = 0` |
| `CPU-C3` | 1 - MEMORY ⇒ `IS_BIT<mem_flags>` |

## Constraints

First, we perform a decoding lookup for the current PC. Instructions having the `word_instr` flag set are not decoded here, as they are delegated to the `CPU32` chip. In that case, we ensure that the current row of the CPU cannot have any other observable effects.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-C4` | `DECODE[pc, imm, packed_decode]` | 1 - word_instr |
| `CPU-C5` | `word_instr` => `MEMORY = 0` |  |
| | _polynomial:_ `word_instr * MEMORY = 0` | |
| `CPU-C6` | `word_instr` => `BRANCH = 0` |  |
| | _polynomial:_ `word_instr * BRANCH = 0` | |
| `CPU-C7` | `word_instr` => `ECALL = 0` |  |
| | _polynomial:_ `word_instr * ECALL = 0` | |
| `CPU-C8` | `word_instr` => `read_register1 = 0` |  |
| | _polynomial:_ `word_instr * read_register1 = 0` | |
| `CPU-C9` | `word_instr` => `read_register2 = 0` |  |
| | _polynomial:_ `word_instr * read_register2 = 0` | |
| `CPU-C10` | `word_instr` => `write_register = 0` |  |
| | _polynomial:_ `word_instr * write_register = 0` | |
| `CPU-C11` | `CPU32[half_instruction_length; timestamp, pc]` | word_instr |

### Range checks

We constrain all columns to have the appropriate ranges. All values in `packed_decode` need to be checked to ensure the packing is correct for the interaction. In contrast, we know ahead of time that decoding will ensure proper range checks for `pc` and `imm`. Similarly, since `next_pc` will propagate through the memory argument and be looked up in the instruction decoding on the next cycle, it is forced to be in the correct range; the final value for `next_pc` is similarly fixed by the memory finalization. For the auxiliary columns, we need to check the limbs of `res`, since `rv1` and `rv2` are enforced by the memory argument, and `rvd` is correct by the correctness of the dependent chips. The ranges of the other auxiliary columns are enforced through later constraints.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CR12` |  | `IS_BIT<read_register1>` |  |
| `CPU-CR13` |  | `IS_BIT<read_register2>` |  |
| `CPU-CR14` |  | `IS_BIT<write_register>` |  |
| `CPU-CR15` |  | `IS_BYTE<half_instruction_length>` |  |
| `CPU-CR16` |  | `IS_BIT<word_instr>` |  |
| `CPU-CR17` |  | `IS_BIT<ALU>` |  |
| `CPU-CR18` |  | `IS_BYTE<alu_flags>` |  |
| `CPU-CR19` |  | `IS_BIT<ADD>` |  |
| `CPU-CR20` |  | `IS_BIT<SUB>` |  |
| `CPU-CR21` |  | `IS_BIT<MEMORY>` |  |
| `CPU-CR22` |  | `IS_BYTE<mem_flags>` |  |
| `CPU-CR23` |  | `IS_BIT<BRANCH>` |  |
| `CPU-CR24` |  | `IS_BIT<ECALL>` |  |
| `CPU-CR25` |  | `IS_BYTE<rs1>` |  |
| `CPU-CR26` |  | `IS_BYTE<rs2>` |  |
| `CPU-CR27` |  | `IS_BYTE<rd>` |  |
| `CPU-CR28.i` | i ∈ [0, 3] | `IS_HALF[res[i]]` | 1 |

### ALU

The ALU functionality is then obtained through delegation to the `ALU` signature, backed by the various ALU chips, or by using the appropriate template. For the pure ALU path, `arg2` is computed as `rv2 + imm`, which relies on [cpu:a:arg2]-multiplex to be either `rv2` or `imm`, depending on the instruction. The other contributions for `arg2` are specific to the (mutually exclusive, [cpu:a:mem]-branch-mutex) `MEMORY` and `BRANCH` flags: - For the `MEMORY` path, we want the output of the ALU to be ``rv1` + `imm``, as that is the address at which the memory access occurs. - For the `BRANCH` path, we want the ALU output to reflect the branch condition (or just be inactive for JALR).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CA29.i` | i ∈ [0, 1] | `arg2` = `MEMORY` dot `imm` + `BRANCH` dot `rv2` + (1 - `MEMORY` - `BRANCH`) dot (`rv2` + `imm`) |  |
| | | _polynomial:_ `arg2[i] - MEMORY * imm[i] - BRANCH * rv2[i] - (1 - MEMORY - BRANCH) * (rv2 + imm)[i] = 0` | |
| `CPU-CA30` |  | ADD ⇒ `ADD<res::DWordWL; rv1, arg2>` |  |
| `CPU-CA31` |  | SUB ⇒ `SUB<res::DWordWL; rv1, arg2>` |  |
| `CPU-CA32` |  | `ALU[res::DWordWL; rv1, arg2, alu_flags]` | ALU |

### Memory<cpu:memory>

Note that since registers need no byte-addressing, we store them in the memory argument with `Word` limbs, simultaneously ensuring that register reads are properly range checked as long as all writes are. The `pc` register behaves very predictably with respect to its timestamps and when it is being read, so for performance reasons, we inline its memory interactions directly into the  chip.

Potentially overlapping memory accesses are ensured to have disjoint timestamps. One consequence of that is that `next_pc` is written at `timestamp + 1` to ensure the access is disjoint with the `pc` read into `rv1` as part of the `AUIPC` instruction (see [cpu:c:read_rv1] and [decode]:decoding-overview). Constraints regarding whether `pc_double_read` corresponds to an `AUIPC` instruction are not necessary, as regardless of its value, the old timestamp is guaranteed smaller than the new timestamp, and the integrity of the memory argument therefore ensures the correctness of this bit.

The memory interaction itself is handled by the `MEMORY` signature, which will read the `mem_flags` argument to perform either a `LOAD` or a `STORE`. We refer to the previous section's description of `arg2` for how the address is computed.

The value to (potentially) be written back to `rd` is stored in `rvd`, which can either come from the ALU --- in case of an ALU operation or a JALR branch --- or from the MEMORY interaction.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CM33` |  | `MEMW[[rv1[0], rv1[1], 0, 0, 0, 0, 0, 0]; 1, 2::DWordWL * rs1, [rv1[0], rv1[1], 0, 0, 0, 0, 0, 0], timestamp + 0::DWordWL, 1, 0, 0]` | read_register1 |
| `CPU-CM34.i` | i ∈ [0, 1] | `!read_register1` => `rv1[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register1) * rv1[i] = 0` | |
| `CPU-CM35` |  | `MEMW[[rv2[0], rv2[1], 0, 0, 0, 0, 0, 0]; 1, 2::DWordWL * rs2, [rv2[0], rv2[1], 0, 0, 0, 0, 0, 0], timestamp + 1::DWordWL, 1, 0, 0]` | read_register2 |
| `CPU-CM36.i` | i ∈ [0, 1] | `!read_register2` => `rv2[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register2) * rv2[i] = 0` | |
| `CPU-CM37` |  | `MEMW[1, 2::DWordWL * rd, [rvd[0], rvd[1], 0, 0, 0, 0, 0, 0], timestamp + 2::DWordWL, 1, 0, 0]` | write_register |
| `CPU-CM38` |  | `MEMOP[rvd; timestamp, res::DWordWL, rv2, mem_flags]` | MEMORY |
| `CPU-CM39.i` | i ∈ [0, 1] | `!MEMORY` and `!BRANCH` => `rvd` = `res` |  |
| | | _polynomial:_ `(1 - MEMORY - BRANCH) * (rvd[i] - (res::DWordWL)[i]) = 0` | |
| `CPU-CM40` |  | `IS_BIT<pc_double_read>` |  |
| `CPU-CM41` |  | `IS_BIT<prev_pc_timestamp_borrow>` |  |
| `CPU-CM42.i` | i ∈ [0, 1] | `memory[1, [2 * 255 + i, 0], [(timestamp[0] - 3 * (1 - pc_double_read)) + 2^32 * prev_pc_timestamp_borrow, timestamp[1] - prev_pc_timestamp_borrow], pc[i]]` | 1 |
| `CPU-CM43.i` | i ∈ [0, 1] | `memory[1, [2 * 255 + i, 0], timestamp + 1::DWordWL, next_pc[i]]` | -1 |

### Branching

A branch is expressed by having the `BRANCH` flag set to 1. Since `BRANCH` and `MEMORY` are mutually exclusive ([cpu:a:mem]-branch-mutex), we can repurpose the `mem_flags` field to indicate a JALR instruction. When JALR is not set, we have a conditional branch that is decided upon by the result of the ALU instructions, as set in the `res` variable. As such, we can set `branch_cond` appropriately as multiplicity flag for the `BRANCH` chip.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CB44` | `branch_cond` = `BRANCH` and (`JALR` or `res`) |  |
| | _polynomial:_ `branch_cond - BRANCH * JALR - BRANCH * (1 - JALR) * res[0] = 0` | |
| `CPU-CB45` | `BRANCH[next_pc; pc, imm, rv1, JALR]` | branch_cond |
| `CPU-CB46` | 1 - branch_cond ⇒ `ADD<next_pc; pc, [2 * half_instruction_length, 0]>` |  |
| `CPU-CB47` | BRANCH ⇒ `ADD<rvd; pc, [2 * half_instruction_length, 0]>` |  |

### System

The interactions with the wider system go through the `ECALL` interface. Since we treat `EBREAK` instructions as unprovable traps, we avoid emitting `DECODE` rows for these, and do not need any further handling in the CPU.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CS48` | `ECALL[timestamp, rv1]` | ECALL |

## Padding

The CPU can be padded with the following values, which have a corresponding row in the DECODE table, at the _odd_ address 1, only reachable through a HALT ecall.

| Column | Padding value |
|--------|---------------|
| `pc` | `1` |
| `rs1` | `0` |
| `rs2` | `0` |
| `rd` | `0` |
| `read_register1` | `0` |
| `read_register2` | `0` |
| `write_register` | `0` |
| `imm` | `0` |
| `half_instruction_length` | `2` |
| `word_instr` | `0` |
| `ALU` | `0` |
| `alu_flags` | `0` |
| `ADD` | `0` |
| `SUB` | `0` |
| `MEMORY` | `0` |
| `mem_flags` | `0` |
| `BRANCH` | `0` |
| `ECALL` | `0` |
| `next_pc` | `1` |
| `rvd` | `0` |
| `prev_pc_timestamp_borrow` | `0` |
| `pc_double_read` | `0` |
| `rv1` | `0` |
| `rv2` | `0` |
| `arg2` | `0` |
| `res` | `0` |
| `branch_cond` | `0` |

This approach minimizes the number of dependent lookups, increasing only multiplicities in the `DECODE` table and the `IS_BYTE` and `IS_HALF` lookups.