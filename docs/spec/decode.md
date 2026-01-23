# DECODE Chip

## Columns

### Output

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | value of the program counter this instruction is associated with. |
| `packed_decode` | `BaseField` | Ordered concatenation of several small variables. The `decode (uncompressed)` section explains the purpose of each variable.\ A list of each variable and the bit(-range) in which it is located:\ [0] `read_register1`, \ [1] `read_register2`, \ [2] `write_register`, \ [3] `memory_2bytes`, \ [4] `memory_4bytes`, \ [5] `memory_8bytes`, \ [6] `c_type`, \ [7] `signed`, \ [8] `mp_selector`, \ [9] `muldiv_selector`, \ [10] `word_instr`, \ [11] `ADD`, \ [12] `SUB`, \ [13] `SLT`, \ [14] `AND`, \ [15] `OR`, \ [16] `XOR`, \ [17] `SHIFT`, \ [18] `JALR`, \ [19] `BEQ`, \ [20] `BLT`, \ [21] `LOAD`, \ [22] `STORE`, \ [23] `MUL`, \ [24] `DIVREM`, \ [25] `ECALL`, \ [26] `EBREAK`; \ [27:35] `rs1`, \ [35:43] `rs2`, \ [43:51] `rd`, \ the remaining bits are set to zero.  |
| `imm` | `DWordWL` | the *fully extended (!)* 64-bit version of the immediate. |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` | The multiplicity with which this instruction is looked up in the `CPU` table. |
