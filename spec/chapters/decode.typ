#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_padding_table,
)
#import "/expr.typ": expr_to_math
#import "/meta.typ": stripe_tables

#let config = load_config()
#let chip = load_chip("src/decode.toml", config)

#let decode = raw(chip.name)

All `RV64IMC` instruction are to be decoded to a format that can be interpreted by the VM.
This section outlines the decoding table being used in the VM.
For reasons of efficiency, data in this table is significantly compressed.
Since reasoning about this compressed form is needlessly complex, the `decode (uncompressed)` section presents the same table in uncompressed form, and explains how to decode `RV64IM` assembly instructions to it.
Instructions on how to compress the uncompressed table to form the compressed decode table, can be derived from the `packed_decode` variable provided below.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The #decode table is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_variable_table(chip, config)

= Padding
The #decode table must be padded to a length that is a power of two.
Empty rows with the following content can be added to achieve this:

#render_chip_padding_table(chip, config)

This is simultaneously the row that is used for padding rows in the CPU,
if the multiplicity is nonzero,
so we need to ensure that this table has at least one row of padding.

= Decoding<decode:decoding-overview>
For the purposes of explaining decoding, we decompress #decode's `packed_decode` variable into its constituent variables.
Note that the below table is _not_ used in practice: it is solely used for the purposes of this explanation.
The construction of the `alu_flags` and `mem_flags` columns is given here through virtual columns.

#let config = load_config()
#let uncompressed_chip = load_chip("src/decode_uncompressed.toml", config)

#render_chip_variable_table(uncompressed_chip, config)

First, we provide a mapping from an an ALU operation "descriptor" to the numerical value as used for the `alu_op` column.
This is the table used to find the value for the #expr_to_math(("opsel", "OPERATION")) notation when performing `ALU` or `BYTE_ALU` interactions.

#figure(
  table(columns: (auto, auto),
        stroke: 0pt,
        inset: (right: .5em),
        align: (left, left), table.header[*Descriptor*][*value*], table.hline(stroke: 1.5pt))[
    *AND*][0][
    *OR*][1][
    *XOR*][2][
    *EQ*][3][
    *LT*][4][
    *SHIFT*][5][
    *SHIFTW*][6][
    *MUL*][7][
    *DIVREM*][8]
)

We will illustrate how each instruction should be expressed in this (uncompressed) decoding table.
The columns of the accompanying table represent the following:
- *`operation`*: the assembly operation being encoded.
- *`alu`*: Set to the descriptor of the ALU operation to be used for `alu_op`.
  If listed as `ADD` or `SUB`, the corresponding flag should be set, otherwise set `ALU = 1` when this column is not empty.
- *`w_instr`*, *`signed`*: whether to set the `word_instr` and `signed` flags, respectively.
- *other*: the other flags that should be set or variables that should be given specific values.

For the purpose of brevity and readability, the table uses the following rules-of-thumb:
+ `rd`, `rs1`, `rs2`, and `imm` are mapped to the values provided by the instruction;
  when a value is not specified by an instruction it defaults to $0$.
+ `read_register1`, `read_register2` and `write_register` are set to $1$ when respectively $#`rs1` != 0$, $#`rs2` != 0$, or  $#`rd` != 0$.

Further clarification is provided in the notes following the table.

/// Add a reference to one or more notes following this table.
#let ref_note(..refs) = {
  super("[" + refs.pos().map(r => ref(r)).join(",") + "]")
}

#let decoding_table(lines) = {
  show figure: set block(breakable: true)
  show: stripe_tables

  figure(table(
    columns: (auto, auto, auto, auto, 1fr, auto),
    stroke: 0pt,
    inset: (right: .5em),
    align: (left, right, center, center, left, right),
    table.header([*Operation*], [*alu*], [*`w_instr`*], [*`signed`*], [*other*], []),
    table.hline(stroke: 1.5pt),
    table.vline(x: 1, start: 1, end: lines.len() + 1, stroke: .5pt),
    ..lines.flatten(),
    table.hline(stroke: 1.5pt),
    table.footer([*Operation*], [*alu*], [*`w_instr`*], [*`signed`*], [*other*]),
  ))
}

#let decoding = (
    // OP-IMM
  ([`ADDI[W]   rd, rs1, imm`], [`ADD`], [`[W]`], [], [], [#ref_note(<note_word_instr>)]),
  ([`SLTI[U]   rd, rs1, imm`], [`LT`], [], [#sym.not`[U]`], [], [#ref_note(<note_signed>)]),
  ([`ANDI      rd, rs1, imm`], [`AND`], [], [], [], []),
  ([`ORI       rd, rs1, imm`], [`OR`],   [], [], [], []),
  ([`XORI      rd, rs1, imm`], [`XOR`], [], [], [], []),
  ([`SLLI[W]   rd, rs1, imm`], [`SHIFT[W]`], [`[W]`], [], [], [#ref_note(<note_word_instr>)]),
  ([`SRLI[W]   rd, rs1, imm`], [`SHIFT[W]`], [`[W]`], [], [`invert`], [#ref_note(<note_word_instr>)]),
  ([`SRAI[W]   rd, rs1, imm`], [`SHIFT[W]`], [`[W]`], [1], [`invert`], [#ref_note(<note_word_instr>)]),
  // OP
  ([`ADD[W]    rd, rs1, rs2`], [`ADD`], [`[W]`], [], [], [#ref_note(<note_word_instr>)]),
  ([`SUB[W]    rd, rs1, rs2`], [`SUB`], [`[W]`], [], [], [#ref_note(<note_word_instr>)]),
  ([`SLT[U]    rd, rs1, rs2`], [`LT`], [], [#sym.not`[U]`], [], [#ref_note(<note_signed>)]),
  ([`AND       rd, rs1, rs2`], [`AND`], [], [], [], []),
  ([`OR        rd, rs1, rs2`], [`OR`], [], [], [], []),
  ([`XOR       rd, rs1, rs2`], [`XOR`], [], [], [], []),
  ([`SLL[W]    rd, rs1, rs2`], [`SHIFT[W]`], [`[W]`], [], [], [#ref_note(<note_word_instr>)]),
  ([`SRL[W]    rd, rs1, rs2`], [`SHIFT[W]`], [`[W]`], [], [`invert`], [#ref_note(<note_word_instr>)]),
  ([`SRA[W]    rd, rs1, rs2`], [`SHIFT[W]`], [`[W]`], [1], [`invert`], [#ref_note(<note_word_instr>)]),
  // OP - M
  ([`MUL[W]    rd, rs1, rs2`], [`MUL`], [`[W]`], [1], [`signed2`], [#ref_note(<note_word_instr>)]),
  ([`MULH      rd, rs1, rs2`], [`MUL`], [], [1], [`signed2`, `muldiv_selector`], []),
  ([`MULHU     rd, rs1, rs2`], [`MUL`], [], [], [`muldiv_selector`], []),
  ([`MULHSU    rd, rs1, rs2`], [`MUL`], [], [1], [`muldiv_selector`], []),
  ([`DIV[U][W] rd, rs1, rs2`], [`DIVREM`], [`[W]`], [#sym.not`[U]`], [], [#ref_note(<note_word_instr>, <note_signed>)]),
  ([`REM[U][W] rd, rs1, rs2`], [`DIVREM`], [`[W]`], [#sym.not`[U]`], [`muldiv_selector`], [#ref_note(<note_word_instr>, <note_signed>)]),
  // LUI/AUIPC
  ([`LUI       rd, imm`], [`ADD`], [], [], [], [#ref_note(<note-lui>)]),
  ([`AUIPC     rd, imm`], [`ADD`], [], [], [`rs1 := x255`], [#ref_note(<note-auipc>)]),
  ([`JAL       rd, imm`], [], [], [], [`BRANCH`, `JALR`, `rs1 := x255`], [#ref_note(<note-jal>)]),
  // Branching
  ([`JALR      rd, rs1, imm`], [], [], [], [`BRANCH`, `JALR`], []),
  ([`BEQ      rs1, rs2, imm`], [`EQ`], [], [], [`BRANCH`], []),
  ([`BNE      rs1, rs2, imm`], [`EQ`], [], [], [`BRANCH`, `invert`], []),
  ([`BLT[U]   rs1, rs2, imm`], [`LT`], [], [#sym.not`[U]`], [`BRANCH`], [#ref_note(<note_signed>)]),
  ([`BGE[U]   rs1, rs2, imm`], [`LT`], [], [#sym.not`[U]`], [`BRANCH`, `invert`], [#ref_note(<note_signed>)]),
  // LOAD
  ([`LD        rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_8B`], []),
  ([`LW[U]     rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_signed := `#sym.not`[U]`, `mem_4B`], [#ref_note(<note_signed>)]),
  ([`LH[U]     rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_signed := `#sym.not`[U]`, `mem_2B`], [#ref_note(<note_signed>)]),
  ([`LB[U]     rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_signed := `#sym.not`[U]`], [#ref_note(<note_signed>)]),
  // STORE
  ([`SD       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`, `mem_8B`], []),
  ([`SW       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`, `mem_4B`], []),
  ([`SH       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`, `mem_2B`], []),
  ([`SB       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`], []),
  // ECALL/EBREAK
  ([`ECALL`], [], [], [], [`ECALL`, $#`rs1` := #`x17`$], [#ref_note(<note-ecall>)]),
  // FENCE
  ([`FENCE`], [`ADD`], [], [], [], [#ref_note(<note-fence>)]),
)

#decoding_table(decoding)

Note that the above table has no entry for the `EBREAK` instruction.
We treat `EBREAK` as an unprovable trap, and its absence from the table enables
this by having no valid decoding available for when the instruction is encountered.

== C-type instructions
The `RV64C` extension for compressed instructions specifies that \~50% of all instructions can be represented using a 16-bit instruction (rather than 32-bits), saving \~25% in code size.
This execution of assembly code is _not_ agnostic to an instruction's compression state; after executing a compressed instruction, the `pc` should be incremented by $2$ rather than $4$.
As such, we provide the `half_instruction_length` column that *must take on the value $1$ for compressed instructions and $2$ for regular instructions*.
It is represented as half the number of bytes in the instruction to make misaligned instructions lengths unrepresentable.
Additionally, having the variable opens the door for future optimizations involving "fused" instructions, where common sequences
of instructions are merged into a single decoded version and need only a single CPU row to prove.

// Construct a note that can be referenced through `lbl`
#let referenceable_note(lbl, note) = {
  show figure: (it) => align(left, [#it])
  [#figure(kind: "note", supplement: [], [#note]) #label(lbl)]
}

== Notes
We note the following about the above decoding table:
#enum(numbering: "[1]",
  enum.item(
    referenceable_note(
      "note_word_instr",
      [`word_instr`: `[W]` indicates that $#`word_instr` = 1$ for the `W`-variant of the operation, and $0$ for the non-`W`-variant. Similarly, `SHIFT[W]` indicates the `SHIFTW` operation for the `W`-variant, and `SHIFT` otherwise.]
    )
  ),
  enum.item(
    referenceable_note(
      "note_signed",
      [`signed`: #sym.not`[U]` indicates that $#`signed` = 1$ for the *non-`U`*-variant of the operation, and $0$ for the `U`-variant.]
    )
  ),
  enum.item(
    referenceable_note(
      "note-lui",
      [`LUI`: this operation loads the 20-bit `imm` in the upper bits of `rd`.
      Observe that this can be represented using `ADDI rd, x0, imm`.
      As such, *we expect the decoding to take care of writing the immediate in bit range $[12:32]$ of `imm` and extending it to 64 bits.*]
    )
  ),
  enum.item(
    referenceable_note(
      "note-auipc",
      [`AUIPC`: this operation adds the 20-bit immediate to the upper bits of `pc` and stores the result in `rd`. 
      Given that the `pc` is stored in `x255`, this operation can be represented using `ADDI rd, x255, imm`.
      As such, *we expect the decoding to take care of writing the immediate in bit range $[12:32]$ of `imm` and extending it to 64 bits.*]
    )
  ),
  enum.item(
    referenceable_note(
      "note-jal",
      [`JAL`: this operation stores $#`pc` + #`2 * half_instruction_length`$ in `rd` and adds two times the sign-extended 20-bit immediate to the `pc`.
      Note that this can be represented using `JALR rd, x255, imm`.
      As such, *we expect the decoding to take care of writing the immediate in bit range $[1:21]$ of `imm` and extending it to 64 bits; the least significant bit should always be 0.*]
    )
  ),
  enum.item(
    referenceable_note(
      "note-ecall",
      [`ECALL`:
      "On RISC-V a system call has its own instruction: `ECALL`. [...] A7 [= register `x17`] contains the system call number." #link("https://libriscv.no/docs/concepts/syscalls/#the-risc-v-system-call-abi")[[source]]
      ]
    )
  ),
  enum.item(
    referenceable_note(
      "note-fence",
      [`FENCE`: currently, the VM interprets this operation as `ADDI x0 x0 0`; a no-op.]
    )
  )
)
