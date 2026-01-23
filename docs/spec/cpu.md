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

## Assumptions

| Ref | Range | Description |
|-----|-------|-------------|
| `cpu:a:one-hot` |  | At most one ALU selector flag is 1 by the decoding, and every other flag is 0. |
| `cpu:a:arg2-multiplex` |  | When `STORE + LOAD + BEQ + BLT = 0`, either `rs2 = 0` or `imm = 0` should be enforced by the decoding. This is needed for `arg2`. |

## Constraints

### decode

| Ref | Kind | Description |
|-----|------|-------------|
| `1` | interaction | `DECODE[pc, imm, packed_decode]` |

### range

| Ref | Kind | Range | Description |
|-----|------|-------|-------------|
| `cpu:c:range_read_register1` | template |  | `IS_BIT<read_register1>` |
| `cpu:c:range_read_register2` | template |  | `IS_BIT<read_register2>` |
| `cpu:c:range_write_register` | template |  | `IS_BIT<write_register>` |
| `cpu:c:range_memory_2bytes` | template |  | `IS_BIT<memory_2bytes>` |
| `cpu:c:range_memory_4bytes` | template |  | `IS_BIT<memory_4bytes>` |
| `cpu:c:range_memory_8bytes` | template |  | `IS_BIT<memory_8bytes>` |
| `cpu:c:range_c_type_instruction` | template |  | `IS_BIT<c_type_instruction>` |
| `cpu:c:range_signed` | template |  | `IS_BIT<signed>` |
| `cpu:c:range_mp_selector` | template |  | `IS_BIT<mp_selector>` |
| `cpu:c:range_muldiv_selector` | template |  | `IS_BIT<muldiv_selector>` |
| `cpu:c:range_word_instr` | template |  | `IS_BIT<word_instr>` |
| `cpu:c:range_ADD` | template |  | `IS_BIT<ADD>` |
| `cpu:c:range_SUB` | template |  | `IS_BIT<SUB>` |
| `cpu:c:range_SLT` | template |  | `IS_BIT<SLT>` |
| `cpu:c:range_AND` | template |  | `IS_BIT<AND>` |
| `cpu:c:range_OR` | template |  | `IS_BIT<OR>` |
| `cpu:c:range_XOR` | template |  | `IS_BIT<XOR>` |
| `cpu:c:range_SHIFT` | template |  | `IS_BIT<SHIFT>` |
| `cpu:c:range_JALR` | template |  | `IS_BIT<JALR>` |
| `cpu:c:range_BEQ` | template |  | `IS_BIT<BEQ>` |
| `cpu:c:range_BLT` | template |  | `IS_BIT<BLT>` |
| `cpu:c:range_LOAD` | template |  | `IS_BIT<LOAD>` |
| `cpu:c:range_STORE` | template |  | `IS_BIT<STORE>` |
| `cpu:c:range_MUL` | template |  | `IS_BIT<MUL>` |
| `cpu:c:range_DIVREM` | template |  | `IS_BIT<DIVREM>` |
| `cpu:c:range_ECALL` | template |  | `IS_BIT<ECALL>` |
| `cpu:c:range_EBREAK` | template |  | `IS_BIT<EBREAK>` |
| `R28` | interaction |  | `IS_BYTE[rs1]` |
| `R29` | interaction |  | `IS_BYTE[rs2]` |
| `R30` | interaction |  | `IS_BYTE[rd]` |
| `R31` | interaction | i ∈ [0, 7] | `IS_BYTE[arg1[i]]` |
| `R32` | interaction | i ∈ [0, 7] | `IS_BYTE[arg2[i]]` |
| `R33` | interaction | i ∈ [0, 7] | `IS_BYTE[res[i]]` |

### alu

| Ref | Kind | Range | Description | Multiplicity |
|-----|------|-------|-------------|--------------|
| `A1` | template |  | ADD + LOAD + STORE ⇒ `ADD<res::DWordWL; arg1::DWordWL, arg2::DWordWL>` |  |
| `cpu:c:sub` | template |  | SUB + BEQ ⇒ `SUB<res::DWordWL; arg1::DWordWL, arg2::DWordWL>` |  |
| `A3` | interaction |  | `LT[res[0]; arg1::DWordHHW, arg2::DWordHHW, signed]` | SLT + BLT |
| `A4` | arith | i ∈ [1, 7] | `SLT` + `BLT` => `res[i]` = 0 |  |
| | | _polynomial:_ `(SLT + BLT) * res[i] = 0` | |
| `A5` | interaction | i ∈ [0, 7] | `AND_BYTE[res[i]; arg1[i], arg2[i]]` | AND |
| `A6` | interaction | i ∈ [0, 7] | `OR_BYTE[res[i]; arg1[i], arg2[i]]` | OR |
| `A7` | interaction | i ∈ [0, 7] | `XOR_BYTE[res[i]; arg1[i], arg2[i]]` | XOR |
| `A8` | interaction |  | `SHIFT[res::DWordHL; arg1::DWordHL, arg2[0], mp_selector, signed, word_instr]` | SHIFT |
| `A9` | template |  | JALR ⇒ `ADD<res::DWordWL; pc, (2 * c_type_instruction + 4 * (1 - c_type_instruction))::DWordWL>` |  |
| `A10` | interaction |  | `MUL[res; arg1, signed, arg2, mp_selector, muldiv_selector]` | MUL |
| `A11` | interaction |  | `DVRM[res; arg1, arg2, signed, muldiv_selector]` | DIVREM |

### mem

| Ref | Kind | Range | Description | Multiplicity |
|-----|------|-------|-------------|--------------|
| `M1` | interaction |  | `MEMW[rv1; 1, 2 * rs1, rv1, timestamp + 0, 1, 0, 0]` | read_register1 |
| `M2` | arith | i ∈ [0, 2] | `!read_register1` => `rv1[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register1) * rv1[i] = 0` | |
| `M3` | interaction |  | `MEMW[rv2; 1, 2 * rs2, rv2, timestamp + 1, 1, 0, 0]` | read_register2 |
| `M4` | arith | i ∈ [0, 2] | `!read_register2` => `rv2[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register2) * rv2[i] = 0` | |
| `M5` | interaction |  | `MEMW[1, 2 * rd, rvd, timestamp + 2, 1, 0, 0]` | write_register |
| `M6` | interaction |  | `LOAD[rvd; 0, res, timestamp + 0, memory_2bytes, memory_4bytes, memory_8bytes, signed]` | LOAD |
| `M7` | interaction |  | `MEMW[0, res, rv2, timestamp + 1, memory_2bytes, memory_4bytes, memory_8bytes]` | STORE |
| `M8` | interaction |  | `MEMW[pc; 1, 2 * 255, next_pc, timestamp + 1, 1, 0, 0]` | 1 - pad |

### sys

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `cpu:c:ebreak_traps` | arith | `!EBREAK` |  |
| | | _polynomial:_ `1 - EBREAK = 0` | |
| | | _note:_ We treat `EBREAK` as an unprovable trap | |
| `S2` | interaction | `ECALL[rvd; rv1, pc, timestamp, rv2]` | ECALL |

### ext

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `E1` | arith | (`rv1_sign_bit` or `arg2_sign_bit` or `res_sign_bit`) => `word_instr` |  |
| | | _polynomial:_ `(rv1_sign_bit + arg2_sign_bit + res_sign_bit) * (1 - word_instr) = 0` | |
| `E2` | interaction | `MSB16[rv1_sign_bit; rv1[1]]` | word_instr |
| `E3` | arith | `arg1[:4]` = `rv1[:2]` |  |
| | | _polynomial:_ `(arg1::DWordWL)[0] - (rv1::DWordWL)[0] = 0` | |
| `E4` | arith | `arg1[4:]` = `rv1[2]` dot (1 - `word_instr`) + (2^(32) - 1) dot `rv1_sign_bit` dot `signed` |  |
| | | _polynomial:_ `(arg1::DWordWL)[1] - (1 - word_instr) * rv1[2] - signed * rv1_sign_bit * (2^32 - 1) = 0` | |
| `E5` | interaction | `MSB16[arg2_sign_bit; rv2[1]]` | word_instr |
| `E6` | arith | `arg2[:4]` = (1 - `STORE` - `LOAD`) dot `rv2[:2]` + (1 - `BEQ` - `BLT`) dot `imm[0]` |  |
| | | _polynomial:_ `(arg2::DWordWL)[0] - (1 - STORE - LOAD) * (rv2::DWordWL)[0] - (1 - BEQ - BLT) * imm[0] = 0` | |
| `E7` | arith | `arg2[4:]` = (1 - `STORE` - `LOAD`) dot ((1 - `word_instr`) dot `rv2[2]` + `signed` dot `arg2_sign_bit` dot (2^(32) - 1)) + (1 - `BEQ` - `BLT`) dot `imm[1]` |  |
| | | _polynomial:_ `(arg2::DWordWL)[1] - (1 - STORE - LOAD) * (1 - word_instr) * rv2[2] - (1 - STORE - LOAD) * signed * arg2_sign_bit * (2^32 - 1) - (1 - BEQ - BLT) * imm[1] = 0` | |
| `E8` | interaction | `MSB8[res_sign_bit; res[3]]` | word_instr |
| `E9` | arith | `!LOAD` => `rvd[0]` = `res[:4]` |  |
| | | _polynomial:_ `(1 - LOAD) * (rvd[0] - (res::DWordWL)[0]) = 0` | |
| `E10` | arith | `!LOAD` => `rvd[1]` = (1 - `word_instr`) dot `res[4:]` + `res_sign_bit` dot (2^(32) - 1) |  |
| | | _polynomial:_ `(1 - LOAD) * (rvd[1] - (1 - word_instr) * (res::DWordWL)[1] - res_sign_bit * (2^32 - 1)) = 0` | |
| | | _note:_ _Sign_ extend the output if it wasn't a `LOAD`. Only `LOAD` has both `write_register = 1` and `rvd ≠ res`. `LOAD` and `word_instr` are disjoint | |

### misc

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `cpu:c:is_equal` | interaction | `ZERO[is_equal; res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7]]` | BEQ |
| `O2` | arith | `branch_cond` = `JALR` or (`BLT` and (`res` xor `invert`)) or (`BEQ` and (`is_equal` xor `invert`)) |  |
| | | _polynomial:_ `-branch_cond + JALR + res[0] * (1 - mp_selector) * BLT + (1 - res[0]) * mp_selector * BLT + is_equal * (1 - mp_selector) * BEQ + (1 - is_equal) * mp_selector * BEQ = 0` | |
| | | _note:_ where `invert` is represented by `mp_selector` | |
| `O3` | interaction | `BRANCH[next_pc; pc, imm[0], arg1::DWordWL, JALR]` | branch_cond |
| `O4` | template | `ADD<next_pc; pc, (2 * c_type_instruction + 4 * (1 - c_type_instruction))::DWordWL>` |  |
