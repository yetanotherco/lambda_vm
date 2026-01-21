# DECODE Chip

## Columns

### Output

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | value of the program counter this instruction is associated with. |
| `rs1` | `Byte` | index of source register 1. |
| `rs2` | `Byte` | index of source register 2. |
| `rd` | `Byte` | index of destination register. |
| `write_register` | `Bit` | whether the result should be written to `rd` ($=0$ for memory write and when $`rd` = `x0`$. |
| `mem_2B` | `Bit` | whether the memory access (read or write) touches exactly $2$ bytes. |
| `mem_4B` | `Bit` | whether the memory access (read or write) touches exactly $4$ bytes. |
| `mem_8B` | `Bit` | whether the memory access (read or write) touches exactly $8$ bytes. |
| `c_type` | `Bit` | Whether the instruction is of type `C`, i.e., whether it is $2$ bytes long instead of $4$. |
| `imm` | `DWordWL` | the *fully extended (!)* 64-bit version of the immediate. |
| `signed` | `Bit` | selector used to indicate signed or unsigned input interpretation. |
| `mp_selector` | `Bit` | Multi-purpose selector used by the CPU to to configure several ALU operations in different ways.            See the `CPU` chip for more details. |
| `muldiv_selector` | `Bit` | selects which output of `MUL` (lo/hi) or `DVRM` (quo/rem) is wanted. |
| `word_instr` | `Bit` | Whether the instruction is a `*W` instruction, requiring the inputs and outputs to be (sign) extended. |
| `ADD` | `Bit` | ALU selector flag |
| `SUB` | `Bit` | ALU selector flag |
| `SLT` | `Bit` | ALU selector flag |
| `AND` | `Bit` | ALU selector flag |
| `OR` | `Bit` | ALU selector flag |
| `XOR` | `Bit` | ALU selector flag |
| `SHIFT` | `Bit` | ALU selector flag |
| `JALR` | `Bit` | ALU selector flag |
| `BEQ` | `Bit` | ALU selector flag |
| `BLT` | `Bit` | ALU selector flag |
| `LOAD` | `Bit` | ALU selector flag |
| `STORE` | `Bit` | ALU selector flag |
| `MUL` | `Bit` | ALU selector flag |
| `DIVREM` | `Bit` | ALU selector flag |
| `ECALL` | `Bit` | ALU selector flag |
| `EBREAK` | `Bit` | ALU selector flag |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` | The multiplicity with which this instruction is looked up in the `CPU` table. |
