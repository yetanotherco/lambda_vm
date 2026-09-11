# DECODE Table

All `RV64IMC` instruction are to be decoded to a format that can be interpreted by the VM. This section outlines the decoding table being used in the VM. For reasons of efficiency, data in this table is significantly compressed. Since reasoning about this compressed form is needlessly complex, the `decode (uncompressed)` section presents the same table in uncompressed form, and explains how to decode `RV64IM` assembly instructions to it. Instructions on how to compress the uncompressed table to form the compressed decode table, can be derived from the `packed_decode` variable provided below.

## Variables

The  table is comprised of  variables that are expressed using  columns:

### Output

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | value of the program counter this instruction is associated with. |
| `packed_decode` | `BaseField` | Ordered concatenation of several small variables. The `decode (uncompressed)` section explains the purpose of each variable.\ A list of each variable and the bit(-range) in which it is located:\ [0] `read_register1`, \ [1] `read_register2`, \ [2] `write_register`, \ [3] `word_instr`, \ [4] `ALU`, \ [5] `ADD`, \ [6] `SUB`, \ [7] `MEMORY`, \ [8] `BRANCH`, \ [9] `ECALL`, \ [10:17] `rs1`, \ [18:25] `rs2`, \ [26:33] `rd`, \ [34:41] `half_instruction_length`, \ [42:49] `alu_flags`, \ [50:57] `mem_flags`, \ the remaining bits are set to zero.  |
| `imm` | `DWordWL` | the *fully extended (!)* 64-bit version of the immediate. |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` | The multiplicity with which this instruction is looked up in the `CPU` table. |

## Padding

The  table must be padded to a length that is a power of two. Empty rows with the following content can be added to achieve this:

| Column | Padding value |
|--------|---------------|
| `pc` | `1` |
| `packed_decode` | `0` |
| `imm` | `0` |
| `μ` | `0` |

This is simultaneously the row that is used for padding rows in the CPU, if the multiplicity is nonzero, so we need to ensure that this table has at least one row of padding.

## Decoding<decode:decoding-overview>

For the purposes of explaining decoding, we decompress 's `packed_decode` variable into its constituent variables. Note that the below table is _not_ used in practice: it is solely used for the purposes of this explanation. The construction of the `alu_flags` and `mem_flags` columns is given here through virtual columns.

### Output

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | value of the program counter this instruction is associated with. |
| `rs1` | `Byte` | index of source register 1. |
| `rs2` | `Byte` | index of source register 2. |
| `rd` | `Byte` | index of destination register. |
| `read_register1` | `Bit` | whether to load the contents of address `rs1` (1) or `0` (0) into `rv1`. |
| `read_register2` | `Bit` | whether to load the contents of address `rs2` (1) or `0` (0) into `rv2`. |
| `write_register` | `Bit` | whether the result should be written to `rd` ($=0$ for memory write and when $`rd` = `x0`)$. |
| `imm` | `DWordWL` | the *fully extended (!)* 64-bit version of the immediate. |
| `word_instr` | `Bit` | Whether the instruction is a `*W` instruction, requiring the inputs and outputs to be (sign) extended. |
| `ALU` | `Bit` | Enable the ALU |
| `ADD` | `Bit` | ALU does an ADD |
| `SUB` | `Bit` | ALU does a SUB |
| `BRANCH` | `Bit` | The instruction is a branch |
| `MEMORY` | `Bit` | The instruction is a memory access |
| `ECALL` | `Bit` | Perform an ECALL |
| `half_instruction_length` | `Byte` | Half of how many bytes this instruction takes up in the program |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `alu_op` | `B4` | Operation selector value for the ALU |
| `signed` | `Bit` | selector used to indicate signed or unsigned input interpretation. |
| `signed2` | `Bit` | A second signed bit, useful for MUL instructions |
| `muldiv_selector` | `Bit` | selects which output of `MUL` (lo/hi) or `DVRM` (quo/rem) is wanted. |
| `invert` | `Bit` | Instructs the EQ or LT chip to invert its result, or inverts the direction of the SHIFT chip (right instead of left) |
| `memory_op` | `Bit` | Selects whether to LOAD (0) or STORE (1) |
| `mem_2B` | `Bit` | whether the memory access (read or write) touches exactly $2$ bytes. |
| `mem_4B` | `Bit` | whether the memory access (read or write) touches exactly $4$ bytes. |
| `mem_8B` | `Bit` | whether the memory access (read or write) touches exactly $8$ bytes. |
| `mem_signed` | `Bit` | Whether the memory operation is a signed one, this is distinct from `signed` to enable the `JALR` flag to alias `mem_flags` |
| `JALR` | `Bit` | The branch is a JAL(R) |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `alu_flags` | `Byte` | The combined ALU flags |
| `mem_flags` | `Byte` | The combined memory flags (or JALR when BRANCHing) |

**Definition of `alu_flags`:**
```
alu_flags := alu_op + 32 * signed + 64 * (signed2 + invert) + 128 * muldiv_selector
```

**Definition of `mem_flags`:**
```
mem_flags := JALR + memory_op + 2 * mem_signed + 4 * mem_2B + 8 * mem_4B + 16 * mem_8B
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` | The multiplicity with which this instruction is looked up in the `CPU` table. |

First, we provide a mapping from an an ALU operation "descriptor" to the numerical value as used for the `alu_op` column. This is the table used to find the value for the ) notation when performing `ALU` or `BYTE_ALU` interactions.

table(columns: (auto, auto), stroke: 0pt, inset: (right: .5em), align: (left, left), table.header[*Descriptor*][*value*], table.hline(stroke: 1.5pt))[ *AND*][0][ *OR*][1][ *XOR*][2][ *EQ*][3][ *LT*][4][ *SHIFT*][5][ *SHIFTW*][6][ *MUL*][7][ *DIVREM*][8]

We will illustrate how each instruction should be expressed in this (uncompressed) decoding table. The columns of the accompanying table represent the following: - *`operation`*: the assembly operation being encoded. - *`alu`*: Set to the descriptor of the ALU operation to be used for `alu_op`. If listed as `ADD` or `SUB`, the corresponding flag should be set, otherwise set `ALU = 1` when this column is not empty. - *`w_instr`*, *`signed`*: whether to set the `word_instr` and `signed` flags, respectively. - *other*: the other flags that should be set or variables that should be given specific values.

For the purpose of brevity and readability, the table uses the following rules-of-thumb: + `rd`, `rs1`, `rs2`, and `imm` are mapped to the values provided by the instruction; when a value is not specified by an instruction it defaults to `0`. + `read_register1`, `read_register2` and `write_register` are set to `1` when respectively ``rs1` != 0`, ``rs2` != 0`, or  ``rd` != 0`.

Further clarification is provided in the notes following the table.

/// Add a reference to one or more notes following this table.

super("[" + refs.pos().map(r => ref(r)).join(",") + "]") }

show figure: set block(breakable: true)

figure(table( columns: (auto, auto, auto, auto, 1fr, auto), stroke: 0pt, inset: (right: .5em), align: (left, right, center, center, left, right), fill: (_, y) => // Overlay a low-opacity fill color to distinguish the different rows better if calc.odd(y) and y <= lines.len() { color.rgb(0, 0, 100, 20) } else { color.rgb(255, 255, 255, 20) }, table.header([*Operation*], [*alu*], [*`w_instr`*], [*`signed`*], [*other*], []), table.hline(stroke: 1.5pt), table.vline(x: 1, start: 1, end: lines.len() + 1, stroke: .5pt), ..lines.flatten(), table.hline(stroke: 1.5pt), table.footer([*Operation*], [*alu*], [*`w_instr`*], [*`signed`*], [*other*]), )) }

// OP-IMM ([`ADDI[W]   rd, rs1, imm`], [`ADD`], [`[W]`], [], [], []), ([`SLTI[U]   rd, rs1, imm`], [`LT`], [], [.not`[U]`], [], []), ([`ANDI      rd, rs1, imm`], [`AND`], [], [], [], []), ([`ORI       rd, rs1, imm`], [`OR`],   [], [], [], []), ([`XORI      rd, rs1, imm`], [`XOR`], [], [], [], []), ([`SLLI[W]   rd, rs1, imm`], [`SHIFT[W]`], [`[W]`], [], [], []), ([`SRLI[W]   rd, rs1, imm`], [`SHIFT[W]`], [`[W]`], [], [`invert`], []), ([`SRAI[W]   rd, rs1, imm`], [`SHIFT[W]`], [`[W]`], [1], [`invert`], []), // OP ([`ADD[W]    rd, rs1, rs2`], [`ADD`], [`[W]`], [], [], []), ([`SUB[W]    rd, rs1, rs2`], [`SUB`], [`[W]`], [], [], []), ([`SLT[U]    rd, rs1, rs2`], [`LT`], [], [.not`[U]`], [], []), ([`AND       rd, rs1, rs2`], [`AND`], [], [], [], []), ([`OR        rd, rs1, rs2`], [`OR`], [], [], [], []), ([`XOR       rd, rs1, rs2`], [`XOR`], [], [], [], []), ([`SLL[W]    rd, rs1, rs2`], [`SHIFT[W]`], [`[W]`], [], [], []), ([`SRL[W]    rd, rs1, rs2`], [`SHIFT[W]`], [`[W]`], [], [`invert`], []), ([`SRA[W]    rd, rs1, rs2`], [`SHIFT[W]`], [`[W]`], [1], [`invert`], []), // OP - M ([`MUL[W]    rd, rs1, rs2`], [`MUL`], [`[W]`], [1], [`signed2`], []), ([`MULH      rd, rs1, rs2`], [`MUL`], [], [1], [`signed2`, `muldiv_selector`], []), ([`MULHU     rd, rs1, rs2`], [`MUL`], [], [], [`muldiv_selector`], []), ([`MULHSU    rd, rs1, rs2`], [`MUL`], [], [1], [`muldiv_selector`], []), ([`DIV[U][W] rd, rs1, rs2`], [`DIVREM`], [`[W]`], [.not`[U]`], [], []), ([`REM[U][W] rd, rs1, rs2`], [`DIVREM`], [`[W]`], [.not`[U]`], [`muldiv_selector`], []), // LUI/AUIPC ([`LUI       rd, imm`], [`ADD`], [], [], [], []), ([`AUIPC     rd, imm`], [`ADD`], [], [], [`rs1 := x255`], []), ([`JAL       rd, imm`], [], [], [], [`BRANCH`, `JALR`, `rs1 := x255`], []), // Branching ([`JALR      rd, rs1, imm`], [], [], [], [`BRANCH`, `JALR`], []), ([`BEQ      rs1, rs2, imm`], [`EQ`], [], [], [`BRANCH`], []), ([`BNE      rs1, rs2, imm`], [`EQ`], [], [], [`BRANCH`, `invert`], []), ([`BLT[U]   rs1, rs2, imm`], [`LT`], [], [.not`[U]`], [`BRANCH`], []), ([`BGE[U]   rs1, rs2, imm`], [`LT`], [], [.not`[U]`], [`BRANCH`, `invert`], []), // LOAD ([`LD        rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_8B`], []), ([`LW[U]     rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_signed := `.not`[U]`, `mem_4B`], []), ([`LH[U]     rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_signed := `.not`[U]`, `mem_2B`], []), ([`LB[U]     rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_signed := `.not`[U]`], []), // STORE ([`SD       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`, `mem_8B`], []), ([`SW       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`, `mem_4B`], []), ([`SH       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`, `mem_2B`], []), ([`SB       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`], []), // ECALL/EBREAK ([`ECALL`], [], [], [], [`ECALL`, ``rs1` := `x17``], []), // FENCE ([`FENCE`], [`ADD`], [], [], [], []),

Note that the above table has no entry for the `EBREAK` instruction. We treat `EBREAK` as an unprovable trap, and its absence from the table enables this by having no valid decoding available for when the instruction is encountered.

### C-type instructions

The `RV64C` extension for compressed instructions specifies that \~50% of all instructions can be represented using a 16-bit instruction (rather than 32-bits), saving \~25% in code size. This execution of assembly code is _not_ agnostic to an instruction's compression state; after executing a compressed instruction, the `pc` should be incremented by `2` rather than `4`. As such, we provide the `half_instruction_length` column that *must take on the value `1` for compressed instructions and `2` for regular instructions*. It is represented as half the number of bytes in the instruction to make misaligned instructions lengths unrepresentable. Additionally, having the variable opens the door for future optimizations involving "fused" instructions, where common sequences of instructions are merged into a single decoded version and need only a single CPU row to prove.

// Construct a note that can be referenced through `lbl`

show figure: (it) => align(left, []) [ ] }

### Notes

We note the following about the above decoding table:

enum.item( referenceable_note( "note_word_instr", [`word_instr`: `[W]` indicates that ``word_instr` = 1` for the `W`-variant of the operation, and `0` for the non-`W`-variant. Similarly, `SHIFT[W]` indicates the `SHIFTW` operation for the `W`-variant, and `SHIFT` otherwise.] ), enum.item( referenceable_note( "note_signed", [`signed`: .not`[U]` indicates that ``signed` = 1` for the *non-`U`*-variant of the operation, and `0` for the `U`-variant.] ), enum.item( referenceable_note( "note-lui", [`LUI`: this operation loads the 20-bit `imm` in the upper bits of `rd`. Observe that this can be represented using `ADDI rd, x0, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[12:32]` of `imm` and extending it to 64 bits.*] ), enum.item( referenceable_note( "note-auipc", [`AUIPC`: this operation adds the 20-bit immediate to the upper bits of `pc` and stores the result in `rd`. Given that the `pc` is stored in `x255`, this operation can be represented using `ADDI rd, x255, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[12:32]` of `imm` and extending it to 64 bits.*] ), enum.item( referenceable_note( "note-jal", [`JAL`: this operation stores ``pc` + `2 * half_instruction_length`` in `rd` and adds two times the sign-extended 20-bit immediate to the `pc`. Note that this can be represented using `JALR rd, x255, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[1:21]` of `imm` and extending it to 64 bits; the least significant bit should always be 0.*] ), enum.item( referenceable_note( "note-ecall", [`ECALL`: "On RISC-V a system call has its own instruction: `ECALL`. [...] A7 [= register `x17`] contains the system call number." [[source]] ] ), enum.item( referenceable_note( "note-fence", [`FENCE`: currently, the VM interprets this operation as `ADDI x0 x0 0`; a no-op.]