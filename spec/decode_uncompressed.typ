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
This table is not used in practice: it is solely used to explain how one is to fill the (compressed) #decode table.

== Columns
For the purposes of this explanation, we decompress the `packed_decode` column into its constituent variables: 
#render_chip_column_table(chip, config)

== Decoding
The following table illustrates how each instruction should be expressed in the decoding table.
For the purpose of brevity, some values are not explicitly specified.
Here,
+ the corresponding element of the instruction (in the case of `rs1`, `rs2`, `rd`), or 
+ set to $0$ otherwise.

// Subheader for the following table
#let subheader(title, common) = {
  (
    table.header(
      level: 2,
      table.cell(colspan:2, [#emph(title)]),
      if common != [] {
        [_Common: #(common)_]
      } else {
        []
      }
    ),
    table.hline(stroke: .5pt)
  )
}

#table(
  columns: (130pt, auto, auto),
  stroke: 0pt,
  inset: (right: 1em),
  align: left + bottom,
  table.header([*Operation*], [*Op-flag*], [*Other*]),
  table.hline(stroke: 1.5pt),

  ..subheader("OP-IMM", [$#`rs2`:=0$, $#`write_register` := (#`rd` eq.not 0)$]),
  [`ADDI[W] rd, rs1, imm`], [`ADD`], [$#`word_instr` := #`[W]`$],
  [`SLTI[U] rd, rs1, imm`], [`SLT`], [$#`signed` := not#`[U]`$],
  [`ANDI    rd, rs1, imm`], [`AND`], [],
  [`ORI     rd, rs1, imm`], [`OR`], [],
  [`XORI    rd, rs1, imm`], [`XOR`], [],
  [`SLLI[W] rd, rs1, imm`], [`SHIFT`], [$#`word_instr` := #`[W]`$],
  [`SRLI[W] rd, rs1, imm`], [`SHIFT`], [$#`word_instr` := #`[W]`$, $#`mp_selector` := #`1`$],
  [`SRAI[W] rd, rs1, imm`], [`SHIFT`], [$#`word_instr` := #`[W]`$, $#`mp_selector` := #`1`$, $#`signed` := #`1`$],

  ..subheader("LUI/AUIPC", []),
  [`LUI   rd, imm`], [#sym.arrow.double.r], [`ADDI rd, x0, imm`],
  [`AUIPC rd, imm`], [#sym.arrow.double.r#footnote("The program counter (pc) is stored in register 255.")<pc-index-255>], [`ADDI rd, 255, imm`],

  ..subheader("OP", [$#`imm`:=0$, $#`write_register` := (#`rd` eq.not 0)$]),
  [`ADD[W] rd, rs1, rs2`], [`ADD`], [$#`word_instr` := #`[W]`$],
  [`SUB[W] rd, rs1, rs2`], [`SUB`], [$#`word_instr` := #`[W]`$],
  [`SLT[U] rd, rs1, rs2`], [`SLT`], [$#`signed` := not#`[U]`$],
  [`AND    rd, rs1, rs2`], [`AND`], [],
  [`OR     rd, rs1, rs2`], [`OR`], [],
  [`XOR    rd, rs1, rs2`], [`XOR`], [],
  [`SLL[W] rd, rs1, rs2`], [`SHIFT`], [$#`word_instr` := #`[W]`$],
  [`SRL[W] rd, rs1, rs2`], [`SHIFT`], [$#`word_instr` := #`[W]`$, $#`mp_selector` := #`1`$],
  [`SRA[W] rd, rs1, rs2`], [`SHIFT`], [$#`word_instr` := #`[W]`$, $#`mp_selector` := #`1`$, $#`signed` := #`1`$],

  ..subheader("BRANCHING (unconditional)", [$#`write_register` := #`rd` eq.not 0$]),
  [`JAL   rd, imm`], [#sym.arrow.double.r#footnote(<pc-index-255>)], [`JALR rd, 255, imm`],
  [`JALR  rd, rs1, imm`], [`JALR`], [],

  ..subheader("BRANCHING (conditional)", [$#`rd` := 0$, $#`write_register` := 0$]),
  [`BEQ    rs1, rs2, imm`], [`BEQ`], [],
  [`BNE    rs1, rs2, imm`], [`BEQ`], [`mp_selector`],
  [`BLT[U] rs1, rs2, imm`], [`BLT`], [$#`signed` := not#`[U]`$],
  [`BGE[U] rs1, rs2, imm`], [`BLT`], [$#`signed` := not#`[U]`$, $#`mp_selector` := 1$],

  ..subheader("LOAD", [$#`rs2` := 0$]),
  [`LD    rd, rs1, imm`], [`LOAD`], [$#`mem_2b` := 1$, $#`mem_4b` := 1$, $#`mem_8b` := 1$],
  [`LW[U] rd, rs1, imm`], [`LOAD`], [$#`signed` := not#`[U]`$, $#`mem_2b` := 1$, $#`mem_4b` := 1$],
  [`LH[U] rd, rs1, imm`], [`LOAD`], [$#`signed` := not#`[U]`$, $#`mem_2b` := 1$],
  [`LB[U] rd, rs1, imm`], [`LOAD`], [$#`signed` := not#`[U]`$],

  ..subheader("STORE", [$#`rd` := 0$]),
  [`SD    rs1, rs2, imm`], [`STORE`], [`mem_2b`, `mem_4b`, `mem_8b`],
  [`SW    rs1, rs2, imm`], [`STORE`], [`mem_2b`, `mem_4b`],
  [`SH    rs1, rs2, imm`], [`STORE`], [`mem_2b`],
  [`SB    rs1, rs2, imm`], [`STORE`], [],

  ..subheader("SYSTEM", []),
  [`ECALL`], [`ECALL`], [`write_register`, $#`rs1` := #`x17`$, $#`rs2` := #`x11`$, $#`rd` := #`x10`$],
  [`EBREAK`], [`EBREAK`], [],

  ..subheader("OP (M-extension)", [$#`imm` := 0$, $#`write_register` := #`rd` eq.not 0$]),
  [`MUL[W]    rd, rs1, rs2`], [`MUL`], [$#`word_instr` := #`[W]`$, `signed := 1`, `mp_selector`],
  [`MULH      rd, rs1, rs2`], [`MUL`], [`muldiv_selector`, `signed`, `mp_selector`],
  [`MULHU     rd, rs1, rs2`], [`MUL`], [`muldiv_selector`],
  [`MULHSU    rd, rs1, rs2`], [`MUL`], [`muldiv_selector`, `signed`],
  [`DIV[U][W] rd, rs1, rs2`], [`DIVREM`], [$#`word_instr` := #`[W]`$, $#`signed` := not#`[U]`$],
  [`REM[U][W] rd, rs1, rs2`], [`DIVREM`], [$#`word_instr` := #`[W]`$, $#`signed` := not#`[U]`$, `muldiv_selector := 1`],

  ..subheader("MISC", []),
  [`FENCE`], [#sym.arrow.double.r#footnote("Note: this is a no-op")], [`ADDI 0, x0, 0`],
)

=== Notes
- LUI/AUIPC: these operations load/add `imm` in/to the upper bits of `rd`/`pc`. As such, *we expect the decoding to take care of writing the immediate in the upper bits of `imm`*.
- ECALL:
  "On RISC-V a system call has its own instruction: `ECALL`. A system call can have up to 7 arguments and has 1 return value. The arguments are in registers A0-A6, in that order, and the return value is written into A0 before giving back control to the guest. A7 contains the system call number." #link("https://libriscv.no/docs/concepts/syscalls/#the-risc-v-system-call-abi")[[source]]
  As such,
  - syscall number in A7 (= register `x17`)
  - first syscall argument in A1 (= register `x11`)
  - syscall output in A0 (= register `x10`)
