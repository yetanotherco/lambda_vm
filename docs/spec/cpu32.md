# CPU32 Chip

The  chip is used to delegate the 32-bit instructions of the RV64I instruction set from the main CPU table ([cpu]). All 32-bit instructions are ALU-only instructions, so the BRANCH, MEMORY and ECALL paths need no elaboration. The timestamp and PC have already been read by the CPU table at this point, and need no further checking; the PC for the next instruction will also already be handled by CPU.

The structure follows the regular ALU path, with some extra variables and constraints to contain the required sign extensions.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | The timestamp for the CPU row |
| `pc` | `DWordWL` | The PC at which the instruction occurs |

### Output

| Name | Type | Description |
|------|------|-------------|
| `half_instruction_length` | `Byte` | The length of this instruction |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `rs1` | `Byte` | Source register 1 |
| `read_register1` | `Bit` | Whether to read from `rs1` or not |
| `rv1` | `DWordWHH` | The value in register `rs1` |
| `rv1_sign` | `Bit` | The sign bit of the lower word of `rv1` |
| `arg1` | `DWordWL` | The sign-extended version of `rv1` |
| `rs2` | `Byte` | Source register 2 |
| `read_register2` | `Bit` | Whether to read from `rs2` |
| `rv2` | `DWordWHH` | The value in register `rs2` |
| `rv2_sign` | `Bit` | The sign bit of the lower word of `rv2` |
| `imm` | `DWordWL` | The fully sign-extended immediate to use |
| `arg2` | `DWordWL` | Either the sign-extended version of `rv2` or all of `imm` |
| `res` | `DWordHL` | The ALU result |
| `res_sign` | `Bit` | The sign bit of the lower word of `res` |
| `rd` | `Byte` | Destination register |
| `write_register` | `Bit` | Whether to write back to `rd` |
| `rvd` | `DWordWL` | The value to write back to `rd`, the sign-extended version of `res` |
| `ALU` | `Bit` | Whether the full ALU is active |
| `alu_flags` | `Byte` | The ALU operation + flags |
| `ADD` | `Bit` | Whether the full ALU is active |
| `SUB` | `Bit` | Whether the full ALU is active |
| `signed` | `Bit` | Whether the instruction is signed or not. Extracted from `alu_flags`, used to determine the extension for the inputs |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `packed_decode` | `BaseField` | The packed representation of all flags and information from the decode table |

**Definition of `packed_decode`:**
```
packed_decode := 2^0 * read_register1 + 2^1 * read_register2 + 2^2 * write_register + 2^3 * 1 + 2^4 * ALU + 2^5 * ADD + 2^6 * SUB + 2^10 * rs1 + 2^18 * rs2 + 2^26 * rd + 2^34 * half_instruction_length + 2^42 * alu_flags
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `CPU32-A1.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |
| `CPU32-A2.i` | i ∈ [0, 1] | `IS_WORD[pc[i]]` |
| `CPU32-A3` |  | `read_register2 = 0` or `imm = 0`, enforced by decoding. |

Some of the assumptions can be checked with only arithmetic constraints, so we provide these below.

| Tag | Description |
|-----|-------------|
| `CPU32-C1` | `read_register2` = 0 or `imm = 0` |
| | _polynomial:_ `read_register2 * (imm[0] + imm[1]) = 0` |

## Constraints

Most constraints correspond to those already present in the CPU, and we present them here first, including some updates to the range checking corresponding to the differing types. We also need to make sure that for padding rows (`mu = 0`), no side effects can occur.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU32-C2` | `DECODE[pc, imm, packed_decode]` | μ |

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU32-CR3` |  | `IS_BIT<μ>` |  |
| `CPU32-CR4` |  | `IS_BIT<read_register1>` |  |
| `CPU32-CR5` |  | `IS_BIT<read_register2>` |  |
| `CPU32-CR6` |  | `IS_BIT<write_register>` |  |
| `CPU32-CR7` |  | `IS_BYTE<half_instruction_length>` |  |
| `CPU32-CR8` |  | `IS_BIT<ALU>` |  |
| `CPU32-CR9` |  | `IS_BYTE<alu_flags>` |  |
| `CPU32-CR10` |  | `IS_BIT<ADD>` |  |
| `CPU32-CR11` |  | `IS_BIT<SUB>` |  |
| `CPU32-CR12` |  | `IS_BYTE<rs1>` |  |
| `CPU32-CR13` |  | `IS_BYTE<rs2>` |  |
| `CPU32-CR14` |  | `IS_BYTE<rd>` |  |
| `CPU32-CR15.i` | i ∈ [0, 1] | `IS_HALF[rv1[i]]` | μ |
| `CPU32-CR16.i` | i ∈ [0, 1] | `IS_HALF[rv2[i]]` | μ |
| `CPU32-CR17.i` | i ∈ [0, 3] | `IS_HALF[res[i]]` | μ |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU32-CA18` | ADD ⇒ `ADD<res::DWordWL; arg1, arg2>` |  |
| `CPU32-CA19` | SUB ⇒ `SUB<res::DWordWL; arg1, arg2>` |  |
| `CPU32-CA20` | `ALU[res::DWordWL; arg1, arg2, alu_flags]` | ALU |

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU32-CM21` |  | `MEMW[[(rv1::DWordWL)[0], rv1[2], 0, 0, 0, 0, 0, 0]; 1, 2::DWordWL * rs1, [(rv1::DWordWL)[0], rv1[2], 0, 0, 0, 0, 0, 0], timestamp + 0::DWordWL, 1, 0, 0]` | read_register1 |
| `CPU32-CM22.i` | i ∈ [0, 2] | `!read_register1` => `rv1[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register1) * rv1[i] = 0` | |
| `CPU32-CM23` |  | `MEMW[[(rv2::DWordWL)[0], rv2[2], 0, 0, 0, 0, 0, 0]; 1, 2::DWordWL * rs2, [(rv2::DWordWL)[0], rv2[2], 0, 0, 0, 0, 0, 0], timestamp + 1::DWordWL, 1, 0, 0]` | read_register2 |
| `CPU32-CM24.i` | i ∈ [0, 2] | `!read_register2` => `rv2[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register2) * rv2[i] = 0` | |
| `CPU32-CM25` |  | `MEMW[1, 2::DWordWL * rd, [rvd[0], rvd[1], 0, 0, 0, 0, 0, 0], timestamp + 2::DWordWL, 1, 0, 0]` | write_register |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU32-C26` | `!μ` => `read_register1 = 0` |  |
| | _polynomial:_ `(1 - μ) * read_register1 = 0` | |
| `CPU32-C27` | `!μ` => `read_register2 = 0` |  |
| | _polynomial:_ `(1 - μ) * read_register2 = 0` | |
| `CPU32-C28` | `!μ` => `write_register = 0` |  |
| | _polynomial:_ `(1 - μ) * write_register = 0` | |
| `CPU32-C29` | `CPU32[half_instruction_length; timestamp, pc]` | -μ |

Then, we have the constraints corresponding to the sign-extension and definition of `arg1`, `arg2` and `rd`. This includes a step where we extract the `signed` bit from the `alu_flags`, as this determines whether to sign extend the inputs or not.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU32-C30` | `signed` != 0 => `μ` = 1 |  |
| | _polynomial:_ `signed * (1 - μ) = 0` | |
| `CPU32-C31` | `BYTE_ALU[32 * signed; ⧼AND⧽, 32, alu_flags]` | μ |
| `CPU32-C32` | `SIGN<rv1_sign; rv1[1], signed>` |  |
| `CPU32-C33` | `arg1[0]` = `rv1[:2]` |  |
| | _polynomial:_ `arg1[0] - (rv1::DWordWL)[0] = 0` | |
| `CPU32-C34` | `arg1[1]` = (2^(32) - 1) dot `rv1_sign` |  |
| | _polynomial:_ `arg1[1] - (2^32 - 1) * rv1_sign = 0` | |
| `CPU32-C35` | `SIGN<rv2_sign; rv2[1], signed>` |  |
| `CPU32-C36` | `arg2[0]` = `rv2[:2]` + `imm[0]` |  |
| | _polynomial:_ `arg2[0] - (rv2::DWordWL)[0] - imm[0] = 0` | |
| `CPU32-C37` | `arg2[1]` = (2^(32) - 1) dot `rv2_sign` + `imm[1]` |  |
| | _polynomial:_ `arg2[1] - (2^32 - 1) * rv2_sign - imm[1] = 0` | |
| `CPU32-C38` | `SIGN<res_sign; res[1], μ>` |  |
| `CPU32-C39` | `rvd[0]` = `res[:2]` |  |
| | _polynomial:_ `rvd[0] - (res::DWordWL)[0] = 0` | |
| `CPU32-C40` | `rvd[1]` = (2^(32) - 1) dot `res_sign` |  |
| | _polynomial:_ `rvd[1] - (2^32 - 1) * res_sign = 0` | |

## Padding

The table can be padded with the following values:

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `pc` | `0` |
| `half_instruction_length` | `2` |
| `rs1` | `0` |
| `read_register1` | `0` |
| `rv1` | `0` |
| `rv1_sign` | `0` |
| `arg1` | `0` |
| `rs2` | `0` |
| `read_register2` | `0` |
| `rv2` | `0` |
| `rv2_sign` | `0` |
| `imm` | `0` |
| `arg2` | `0` |
| `res` | `0` |
| `res_sign` | `0` |
| `rd` | `0` |
| `write_register` | `0` |
| `rvd` | `0` |
| `ALU` | `0` |
| `alu_flags` | `0` |
| `ADD` | `0` |
| `SUB` | `0` |
| `signed` | `0` |
| `μ` | `0` |