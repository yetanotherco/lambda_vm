#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
)

#let config = load_config()
#let chip = load_chip("src/decode_uncompressed.toml", config)
#show: book-page.with(title: "DECODE chip")

#let decode = raw(chip.name)

= #decode table (uncompressed)
This section outlines how to decode `RV64IM` assembly to a VM-readable table.
Note that this table is not used in practice: it is solely used to explain how one is to fill the (compressed) #decode table.

== Columns
For the purposes of this explanation, we decompress the (uncompressed) #decode table's `packed_decode` variable into its constituent variables: 
#render_chip_column_table(chip, config)

== Decoding
The below table illustrates how each instruction should be expressed in the decoding table.
The columns of the table represent the following:
- *`operation`*: the assembly operation being encoded,
- *`op-flag`*: which of the "`ALU` selector flags" operation flags to set. Each operation sets exactly one.
- *`w_reg`*, *`w_instr`*, *`signed`*: whether to set the `write_register`, `word_instr` or `signed` flag, respectively,
- *other*: the other flags that should be set or variables that should be given specific values.

For the purpose of brevity and readability, the table uses the following rules-of-thumb:
- `rd`, `rs1`, `rs2`, and `imm` are mapped to the values provided by the instruction.
  When an instruction does not provide a value, it is set to $0$.
- When a flag's cell is empty / is not listed in `other`, it is set to $0$.
- $#`rd` eq.not 0$ indicates that $#`write_register` = 1$ when $#`rd` eq.not 0$ and $0$ otherwise.
- `[W]` indicates that $#`write_register` = 0$ for the `W`-variant of the operation, and $0$ for the non-`W`-variant.
- #sym.not`[U]` indicates that $#`signed` = 1$ for the *non-`U`*-variant of the operation, and $0$ for the `U`-variant.
- *The `c_type` flag is set independently of the below table.* (see next paragraph)

Further clarification is provided in the notes following the table.

=== C-type instructions
The `RV64C` extension for compressed instructions specifies that \~50% of all instructions can be represented using a 16-bit instruction (rather than 32-bits), saving \~25% in code size.
This execution of assembly code is _not_ agnostic to an instruction's compression state; after executing a compressed instruction, the `pc` should be incremented by $2$ rather than $4$.
To indicate an instruction is provided in compressed form, the `c_type` flag is introduced.
*This flag should be set to $1$ whenever the decoded instruction is provided in compressed form and $0$ otherwise.*

/// Add a reference to one or more notes following this table.
#let ref_note(..refs) = {
  super("[" + refs.pos().map(r => ref(r)).join(",") + "]")
}

#let decoding_table(lines) = {
  show figure: set block(breakable: true)

  figure(table(
    columns: (auto, auto, 40pt, 40pt, 40pt, 1fr, 15pt),
    stroke: 0pt,
    inset: (right: .5em),
    align: (left, right, center, center, center, left, right),
    fill: (_, y) =>
      if calc.odd(y) and y <= lines.len() { luma(245) }
      else { white },
    table.header([*Operation*], [*op-flag*], [*`w_reg`*], [*`w_instr`*], [*`signed`*], [*other*], []),
    table.hline(stroke: 1.5pt),
    table.vline(x: 1, start: 1, end: lines.len() + 1, stroke: .5pt),
    ..lines.flatten(),
    table.hline(stroke: 1.5pt),
    table.footer([*Operation*], [*op-flag*], [*`w_reg`*], [*`w_instr`*], [*`signed`*], [*other*]),
    ),  
    caption: [Decoding table]
  )
}

#let decoding = (
    // OP-IMM
  ([`ADDI[W]   rd, rs1, imm`], [`ADD`], [$#`rd` eq.not 0$], [`[W]`], [], [], []),
  ([`SLTI[U]   rd, rs1, imm`], [`SLT`], [$#`rd` eq.not 0$], [], [#sym.not`[U]`], [], []),
  ([`ANDI      rd, rs1, imm`], [`AND`], [$#`rd` eq.not 0$], [], [], [], []),
  ([`ORI       rd, rs1, imm`], [`OR`],  [$#`rd` eq.not 0$],  [], [], [], []),
  ([`XORI      rd, rs1, imm`], [`XOR`], [$#`rd` eq.not 0$], [], [], [], []),
  ([`SLLI[W]   rd, rs1, imm`], [`SHIFT`], [$#`rd` eq.not 0$], [`[W]`], [], [], []),
  ([`SRLI[W]   rd, rs1, imm`], [`SHIFT`], [$#`rd` eq.not 0$], [`[W]`], [], [`mp_selector`], []),
  ([`SRAI[W]   rd, rs1, imm`], [`SHIFT`], [$#`rd` eq.not 0$], [`[W]`], [1], [`mp_selector`], []),
  // OP
  ([`ADD[W]    rd, rs1, rs2`], [`ADD`], [$#`rd` eq.not 0$], [`[W]`], [], [], []),
  ([`SUB[W]    rd, rs1, rs2`], [`SUB`], [$#`rd` eq.not 0$], [`[W]`], [], [], []),
  ([`SLT[U]    rd, rs1, rs2`], [`SLT`], [$#`rd` eq.not 0$], [], [#sym.not`[U]`], [], []),
  ([`AND       rd, rs1, rs2`], [`AND`], [$#`rd` eq.not 0$], [], [], [], []),
  ([`OR        rd, rs1, rs2`], [`OR`], [$#`rd` eq.not 0$], [], [], [], []),
  ([`XOR       rd, rs1, rs2`], [`XOR`], [$#`rd` eq.not 0$], [], [], [], []),
  ([`SLL[W]    rd, rs1, rs2`], [`SHIFT`], [$#`rd` eq.not 0$], [`[W]`], [], [], []),
  ([`SRL[W]    rd, rs1, rs2`], [`SHIFT`], [$#`rd` eq.not 0$], [`[W]`], [], [`mp_selector`], []),
  ([`SRA[W]    rd, rs1, rs2`], [`SHIFT`], [$#`rd` eq.not 0$], [`[W]`], [1], [`mp_selector`], []),
  // OP - M
  ([`MUL[W]    rd, rs1, rs2`], [`MUL`], [$#`rd` eq.not 0$], [`[W]`], [1], [`mp_selector`], []),
  ([`MULH      rd, rs1, rs2`], [`MUL`], [$#`rd` eq.not 0$], [], [1], [`mp_selector`, `muldiv_selector`], []),
  ([`MULHU     rd, rs1, rs2`], [`MUL`], [$#`rd` eq.not 0$], [], [], [`muldiv_selector`], []),
  ([`MULHSU    rd, rs1, rs2`], [`MUL`], [$#`rd` eq.not 0$], [], [1], [`muldiv_selector`], []),
  ([`DIV[U][W] rd, rs1, rs2`], [`DIVREM`], [$#`rd` eq.not 0$], [`[W]`], [#sym.not`[U]`], [], []),
  ([`REM[U][W] rd, rs1, rs2`], [`DIVREM`], [$#`rd` eq.not 0$], [`[W]`], [#sym.not`[U]`], [`muldiv_selector`], []),
  // LUI/AUIPC
  ([`LUI       rd, imm`], [`ADD`], [$#`rd` eq.not 0$], [], [], [], [#ref_note(<note-lui>)]),
  ([`AUIPC     rd, imm`], [`ADD`], [$#`rd` eq.not 0$], [], [], [`rs1 := x255`], [#ref_note(<note-auipc>)]),
  ([`JAL       rd, imm`], [`JALR`], [$#`rd` eq.not 0$], [], [], [`rs1 := x255`], [#ref_note(<note-jal>)]),
  // Branching
  ([`JALR      rd, rs1, imm`], [`JALR`], [$#`rd` eq.not 0$], [], [], [], []),
  ([`BEQ      rs1, rs2, imm`], [`BEQ`], [], [], [], [], []),
  ([`BNE      rs1, rs2, imm`], [`BEQ`], [], [], [], [`mp_selector`], []),
  ([`BLT[U]   rs1, rs2, imm`], [`BLT`], [], [], [#sym.not`[U]`], [], []),
  ([`BGE[U]   rs1, rs2, imm`], [`BLT`], [], [], [#sym.not`[U]`], [`mp_selector`], []),
  // LOAD
  ([`LD        rd, rs1, imm`], [`LOAD`], [], [], [], [`mem_2B`, `mem_4B`, `mem_8B`], []),
  ([`LW[U]     rd, rs1, imm`], [`LOAD`], [], [], [#sym.not`[U]`], [`mem_2B`, `mem_4B`], []),
  ([`LH[U]     rd, rs1, imm`], [`LOAD`], [], [], [#sym.not`[U]`], [`mem_2B`], []),
  ([`LB[U]     rd, rs1, imm`], [`LOAD`], [], [], [#sym.not`[U]`], [], []),
  // STORE
  ([`SD       rs1, rs2, imm`], [`STORE`], [], [], [], [`mem_2B`, `mem_4B`, `mem_8B`], []),
  ([`SW       rs1, rs2, imm`], [`STORE`], [], [], [], [`mem_2B`, `mem_4B`], []),
  ([`SH       rs1, rs2, imm`], [`STORE`], [], [], [], [`mem_2B`], []),
  ([`SB       rs1, rs2, imm`], [`STORE`], [], [], [], [], []),
  // ECALL/EBREAK
  ([`ECALL`], [`ECALL`], [1], [], [], [$#`rs1` := #`x17`$, $#`rs2` := #`x11`$, $#`rd` := #`x10`$], [#ref_note(<note-ecall>)]),
  ([`EBREAK`], [`EBREAK`], [], [], [], [], []),
  // FENCE
  ([`FENCE`], [`ADD`], [], [], [], [], [#ref_note(<note-fence>)]),
)


#pagebreak()
#decoding_table(decoding)

// Construct a note that can be referenced through `lbl`
#let referenceable_note(lbl, note) = {
  show figure: (it) => align(left, [#it])
  [#figure(kind: "note", supplement: [], [#note]) #label(lbl)]
}

=== Notes
We note the following about the above decoding table:
#enum(numbering: "[1]",
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
      [`JAL`: this operation stores `pc + 4` in `rd` and adds two times the sign-extended 20-bit immediate to the `pc`.
      Note that this can be represented using `JALR rd, x255, imm`.
      As such, *we expect the decoding to take care of writing the immediate in bit range $[1:13]$ of `imm` and extending it to 64 bits; the least significant bit should always be 0.*]
    )
  ),
  enum.item(
    referenceable_note(
      "note-ecall",
      [`ECALL`:
      "On RISC-V a system call has its own instruction: `ECALL`. A system call can have up to 7 arguments and has 1 return value. The arguments are in registers A0-A6, in that order, and the return value is written into A0 before giving back control to the guest. A7 contains the system call number." #link("https://libriscv.no/docs/concepts/syscalls/#the-risc-v-system-call-abi")[[source]]
      As such,
      - syscall number in A7 (= register `x17`)
      - first syscall argument in A1 (= register `x11`)
      - syscall output in A0 (= register `x10`)]
    )
  ),
  enum.item(
    referenceable_note(
      "note-fence",
      [`FENCE`: currently, the VM interprets this operation as `ADDI x0 x0 0`; a no-op.]
    )
  )
)
