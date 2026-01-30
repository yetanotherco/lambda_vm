# CPU Chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `Timestamp` | A preprocessed timestamp to coordinate the memory argument. Since we have at most 3 non-disjoint memory accesses (`(rs1, rs2, rd)`, `(rs1, pc, pc)`, `(LOAD)` or `(STORE)`) a maximum of 4 slots is enough. |
| `pc` | `DWordWL` | The program counter |
| `rs1` | `Byte` | Source register 1 index |
| `rs2` | `Byte` | Source register 2 index |
| `rd` | `Byte` | Destination register index |
| `read_register1` | `Bit` | Whether to read from `rs1` (1) or to place a 0 in `rv1` (0) |
| `read_register2` | `Bit` | Whether to read from `rs2` (1) or to place a 0 in `rv2` (0) |
| `write_register` | `Bit` | Whether to write back to the destination register |
| `memory_2bytes` | `Bit` | Whether the memory access (read or write) touches exactly 2 bytes |
| `memory_4bytes` | `Bit` | Whether the memory access (read or write) touches exactly 4 bytes |
| `memory_8bytes` | `Bit` | Whether the memory access (read or write) touches exactly 8 bytes |
| `c_type_instruction` | `Bit` | Whether the instruction is of C type, i.e., whether it is 2 bytes long instead of 4 |
| `imm` | `DWordWL` | The fully extended 64-bit version of the immediate |
| `signed` | `Bit` | Indicates whether we're dealing with a signed or unsigned instruction |
| `mp_selector` | `Bit` | Multi-purpose selector used by different ALU operations for different purposes. Currently, it is used     - by the `MUL` chip to select between `MUL`/`MULH` and `MULH[S]U`, and     - as flag for inverting the condition of conditional branches (see `branch_cond`)     - as direction (left or right) for `SHIFT` |
| `muldiv_selector` | `Bit` | Selects which output of `MUL` (lo/hi) or `DIV` (quo/rem) is wanted |
| `word_instr` | `Bit` | Whether the instruction is a \*W instruction, requiring the inputs and outputs to be (sign) extended |
| `ADD` | `Bit` | One-hot ALU selector flag |
| `SUB` | `Bit` | One-hot ALU selector flag |
| `SLT` | `Bit` | One-hot ALU selector flag |
| `AND` | `Bit` | One-hot ALU selector flag |
| `OR` | `Bit` | One-hot ALU selector flag |
| `XOR` | `Bit` | One-hot ALU selector flag |
| `SHIFT` | `Bit` | One-hot ALU selector flag |
| `JALR` | `Bit` | One-hot ALU selector flag |
| `BEQ` | `Bit` | One-hot ALU selector flag |
| `BLT` | `Bit` | One-hot ALU selector flag |
| `LOAD` | `Bit` | One-hot ALU selector flag |
| `STORE` | `Bit` | One-hot ALU selector flag |
| `MUL` | `Bit` | One-hot ALU selector flag |
| `DIVREM` | `Bit` | One-hot ALU selector flag |
| `ECALL` | `Bit` | One-hot ALU selector flag |
| `EBREAK` | `Bit` | One-hot ALU selector flag |

### Output

| Name | Type | Description |
|------|------|-------------|
| `next_pc` | `DWordWL` | The program counter for the next instruction |
| `rvd` | `DWordWL` | The value to (maybe) be written back to rvd |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `rv1` | `DWordWHH` | The value of register `rs1` |
| `rv2` | `DWordWHH` | The value of register `rs2` |
| `rv1_sign_bit` | `Bit` | The sign bit of `rv1` if seen as a 32-bit word |
| `arg1` | `DWordBL` | The extended version of `rv1`, depending on `word_instr` |
| `arg2_sign_bit` | `Bit` | The sign bit of `arg2` if seen as a 32-bit word |
| `arg2` | `DWordBL` | A multiplexed version of `rv2` and `imm`, to be used as second argument to ALU calls |
| `res_sign_bit` | `Bit` | The sign bit of `res`, if seen as a 32-bit word |
| `res` | `DWordBL` | The ALU result |
| `is_equal` | `Bit` | Whether `rv1` and `arg2` are equal |
| `branch_cond` | `Bit` | Whether a branch is taken, i.e., the branch condition |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `packed_decode` | `BaseField` | A packed representation of all bit flags and register indices obtained from the decoding |
| `pad` | `Bit` | When no flags are set, we must be in a padding row. |

**Definition of `packed_decode`:**
```
packed_decode := 2^0 * read_register1 + 2^1 * read_register2 + 2^2 * write_register + 2^3 * memory_2bytes + 2^4 * memory_4bytes + 2^5 * memory_8bytes + 2^6 * c_type_instruction + 2^7 * signed + 2^8 * mp_selector + 2^9 * muldiv_selector + 2^10 * word_instr + 2^11 * ADD + 2^12 * SUB + 2^13 * SLT + 2^14 * AND + 2^15 * OR + 2^16 * XOR + 2^17 * SHIFT + 2^18 * JALR + 2^19 * BEQ + 2^20 * BLT + 2^21 * LOAD + 2^22 * STORE + 2^23 * MUL + 2^24 * DIVREM + 2^25 * ECALL + 2^26 * EBREAK + 2^27 * rs1 + 2^35 * rs2 + 2^43 * rd
```

**Definition of `pad`:**
```
pad := 1 - ADD - SUB - SLT - AND - OR - XOR - SHIFT - JALR - BEQ - BLT - LOAD - STORE - MUL - DIVREM - ECALL - EBREAK
```

The `CPU` chip is comprised of  variables that are expressed using  columns:

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `CPU-A1` |  | At most one ALU selector flag is 1 by the decoding, and every other flag is 0. |
| `CPU-A2` |  | When `STORE + LOAD + BEQ + BLT = 0`, either `rs2 = 0` or `imm = 0` should be enforced by the decoding. This is needed for `arg2`. |

## Constraints

First, we perform a decoding lookup for the current PC.

| Tag | Description |
|-----|-------------|
| `CPU-C1` | `DECODE[pc, imm, packed_decode]` |

> **Note:** All casts for interactions will have to be reviewed once other chip interfaces stabilise

### Range checks

> **Note:** Make sure we argue for every column here

> **Note:** is `rvd` still sufficiently constrained? (can also be done through the memory argument like `pc`?)

We constrain all columns to have the appropriate ranges. The flags and register indices looked up from the decoding need to be checked, as they are communicated through the interaction in a packed form. In contrast, we know ahead of time that decoding will ensure proper range checks for `pc` and `imm`. Similarly, since `next_pc` will propagate through the memory argument and be looked up in the instruction decoding on the next cycle, it is forced to be in the correct range. For the auxiliary columns, we need to check the limbs of `arg1`, `arg2`, and `res`. The ranges of the other auxiliary columns are enforced through later constraints.

| Tag | Range | Description |
|-----|-------|-------------|
| `CPU-CR2` |  | `IS_BIT<read_register1>` |
| `CPU-CR3` |  | `IS_BIT<read_register2>` |
| `CPU-CR4` |  | `IS_BIT<write_register>` |
| `CPU-CR5` |  | `IS_BIT<memory_2bytes>` |
| `CPU-CR6` |  | `IS_BIT<memory_4bytes>` |
| `CPU-CR7` |  | `IS_BIT<memory_8bytes>` |
| `CPU-CR8` |  | `IS_BIT<c_type_instruction>` |
| `CPU-CR9` |  | `IS_BIT<signed>` |
| `CPU-CR10` |  | `IS_BIT<mp_selector>` |
| `CPU-CR11` |  | `IS_BIT<muldiv_selector>` |
| `CPU-CR12` |  | `IS_BIT<word_instr>` |
| `CPU-CR13` |  | `IS_BIT<ADD>` |
| `CPU-CR14` |  | `IS_BIT<SUB>` |
| `CPU-CR15` |  | `IS_BIT<SLT>` |
| `CPU-CR16` |  | `IS_BIT<AND>` |
| `CPU-CR17` |  | `IS_BIT<OR>` |
| `CPU-CR18` |  | `IS_BIT<XOR>` |
| `CPU-CR19` |  | `IS_BIT<SHIFT>` |
| `CPU-CR20` |  | `IS_BIT<JALR>` |
| `CPU-CR21` |  | `IS_BIT<BEQ>` |
| `CPU-CR22` |  | `IS_BIT<BLT>` |
| `CPU-CR23` |  | `IS_BIT<LOAD>` |
| `CPU-CR24` |  | `IS_BIT<STORE>` |
| `CPU-CR25` |  | `IS_BIT<MUL>` |
| `CPU-CR26` |  | `IS_BIT<DIVREM>` |
| `CPU-CR27` |  | `IS_BIT<ECALL>` |
| `CPU-CR28` |  | `IS_BIT<EBREAK>` |
| `CPU-CR29` |  | `IS_BYTE[rs1]` |
| `CPU-CR30` |  | `IS_BYTE[rs2]` |
| `CPU-CR31` |  | `IS_BYTE[rd]` |
| `CPU-CR32.i` | i ∈ [0, 7] | `IS_BYTE[arg1[i]]` |
| `CPU-CR33.i` | i ∈ [0, 7] | `IS_BYTE[arg2[i]]` |
| `CPU-CR34.i` | i ∈ [0, 7] | `IS_BYTE[res[i]]` |

### ALU

The ALU functionality is then obtained through judicious dispatching to the corresponding chips.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CA35` |  | ADD + LOAD + STORE ⇒ `ADD<res::DWordWL; arg1::DWordWL, arg2::DWordWL>` |  |
| `CPU-CA36` |  | SUB + BEQ ⇒ `SUB<res::DWordWL; arg1::DWordWL, arg2::DWordWL>` |  |
| `CPU-CA37` |  | `LT[res[0]; arg1::DWordWL, arg2::DWordWL, signed]` | SLT + BLT |
| `CPU-CA38.i` | i ∈ [1, 7] | `SLT` + `BLT` => `res[i]` = 0 |  |
| | | _polynomial:_ `(SLT + BLT) * res[i] = 0` | |
| `CPU-CA39.i` | i ∈ [0, 7] | `AND_BYTE[res[i]; arg1[i], arg2[i]]` | AND |
| `CPU-CA40.i` | i ∈ [0, 7] | `OR_BYTE[res[i]; arg1[i], arg2[i]]` | OR |
| `CPU-CA41.i` | i ∈ [0, 7] | `XOR_BYTE[res[i]; arg1[i], arg2[i]]` | XOR |
| `CPU-CA42` |  | `SHIFT[res::DWordHL; arg1::DWordHL, arg2[0], mp_selector, signed, word_instr]` | SHIFT |
| `CPU-CA43` |  | JALR ⇒ `ADD<res::DWordWL; pc, (2 * c_type_instruction + 4 * (1 - c_type_instruction))::DWordWL>` |  |
| `CPU-CA44` |  | `MUL[res; arg1, signed, arg2, mp_selector, muldiv_selector]` | MUL |
| `CPU-CA45` |  | `DVRM[res; arg1, arg2, signed, muldiv_selector]` | DIVREM |

### Memory

The interactions with the memory, both for register loading and storing, as for `LOAD` and `STORE` instructions are handled. Note that since registers need no byte-addressing, we store them in the memory argument with `Word` limbs. The timestamps are ensured to be disjoint for disjoint memory locations. One consequence of that is that `next_pc` is written at `timestamp + 1` to ensure the access is disjoint with the `pc` read into `rv1` as part of the `AUIPC` instruction.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CM46` |  | `MEMW[rv1; 1, 2 * rs1, rv1, timestamp + 0, 1, 0, 0]` | read_register1 |
| `CPU-CM47.i` | i ∈ [0, 2] | `!read_register1` => `rv1[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register1) * rv1[i] = 0` | |
| `CPU-CM48` |  | `MEMW[rv2; 1, 2 * rs2, rv2, timestamp + 1, 1, 0, 0]` | read_register2 |
| `CPU-CM49.i` | i ∈ [0, 2] | `!read_register2` => `rv2[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register2) * rv2[i] = 0` | |
| `CPU-CM50` |  | `MEMW[1, 2 * rd, rvd, timestamp + 2, 1, 0, 0]` | write_register |
| `CPU-CM51` |  | `LOAD[rvd; 0, res, timestamp + 0, memory_2bytes, memory_4bytes, memory_8bytes, signed]` | LOAD |
| `CPU-CM52` |  | `MEMW[0, res, rv2, timestamp + 1, memory_2bytes, memory_4bytes, memory_8bytes]` | STORE |
| `CPU-CM53` |  | `MEMW[pc; 1, 2 * 255, next_pc, timestamp + 1, 1, 0, 0]` | 1 - pad |

### System

The interactions with the wider system.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CS54` | `!EBREAK` |  |
| | _polynomial:_ `1 - EBREAK = 0` | |
| `CPU-CS55` | `ECALL[timestamp, rv1::DWordWL]` | ECALL |

### Input and output to the ALU

We constrain `arg1`, `arg2` and `rvd` to correspond to the wanted values, including the appropriate sign/zero extension, depending on `word_instr`.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CE56` | (`rv1_sign_bit` or `arg2_sign_bit` or `res_sign_bit`) => `word_instr` |  |
| | _polynomial:_ `(rv1_sign_bit + arg2_sign_bit + res_sign_bit) * (1 - word_instr) = 0` | |
| `CPU-CE57` | `MSB16[rv1_sign_bit; rv1[1]]` | word_instr |
| `CPU-CE58` | `arg1[:4]` = `rv1[:2]` |  |
| | _polynomial:_ `(arg1::DWordWL)[0] - (rv1::DWordWL)[0] = 0` | |
| `CPU-CE59` | `arg1[4:]` = `rv1[2]` dot (1 - `word_instr`) + (2^(32) - 1) dot `rv1_sign_bit` dot `signed` |  |
| | _polynomial:_ `(arg1::DWordWL)[1] - (1 - word_instr) * rv1[2] - signed * rv1_sign_bit * (2^32 - 1) = 0` | |
| `CPU-CE60` | `MSB16[arg2_sign_bit; rv2[1]]` | word_instr |
| `CPU-CE61` | `arg2[:4]` = (1 - `STORE` - `LOAD`) dot `rv2[:2]` + (1 - `BEQ` - `BLT`) dot `imm[0]` |  |
| | _polynomial:_ `(arg2::DWordWL)[0] - (1 - STORE - LOAD) * (rv2::DWordWL)[0] - (1 - BEQ - BLT) * imm[0] = 0` | |
| `CPU-CE62` | `arg2[4:]` = (1 - `STORE` - `LOAD`) dot ((1 - `word_instr`) dot `rv2[2]` + `signed` dot `arg2_sign_bit` dot (2^(32) - 1)) + (1 - `BEQ` - `BLT`) dot `imm[1]` |  |
| | _polynomial:_ `(arg2::DWordWL)[1] - (1 - STORE - LOAD) * (1 - word_instr) * rv2[2] - (1 - STORE - LOAD) * signed * arg2_sign_bit * (2^32 - 1) - (1 - BEQ - BLT) * imm[1] = 0` | |
| `CPU-CE63` | `MSB8[res_sign_bit; res[3]]` | word_instr |
| `CPU-CE64` | `!LOAD` => `rvd[0]` = `res[:4]` |  |
| | _polynomial:_ `(1 - LOAD) * (rvd[0] - (res::DWordWL)[0]) = 0` | |
| `CPU-CE65` | `!LOAD` => `rvd[1]` = (1 - `word_instr`) dot `res[4:]` + `res_sign_bit` dot (2^(32) - 1) |  |
| | _polynomial:_ `(1 - LOAD) * (rvd[1] - (1 - word_instr) * (res::DWordWL)[1] - res_sign_bit * (2^32 - 1)) = 0` | |

### Other constraints

> **Note:** proper ref to IsZero/IsEqual

For [cpu:c:is_equal], refer to the logic of IsZero or IsEqual, in combination with the subtraction of [cpu:c:sub].

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CO66` | `ZERO[is_equal; res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7]]` | BEQ |
| `CPU-CO67` | `branch_cond` = `JALR` or (`BLT` and (`res` xor `invert`)) or (`BEQ` and (`is_equal` xor `invert`)) |  |
| | _polynomial:_ `-branch_cond + JALR + res[0] * (1 - mp_selector) * BLT + (1 - res[0]) * mp_selector * BLT + is_equal * (1 - mp_selector) * BEQ + (1 - is_equal) * mp_selector * BEQ = 0` | |
| `CPU-CO68` | `BRANCH[next_pc; pc, imm[0], arg1::DWordWL, JALR]` | branch_cond |
| `CPU-CO69` | `ADD<next_pc; pc, (2 * c_type_instruction + 4 * (1 - c_type_instruction))::DWordWL>` |  |

> **Note:** Document the choice to not have a multiplicity column here for padding

## Padding

The CPU can be padded with the following values, which have a corresponding row in the DECODE table, at the _odd_ address 1, only reachable through a HALT ecall.

This approach minimizes the number of dependent lookups, increasing only multiplicities in the DECODE table and the IS_BYTE lookup.