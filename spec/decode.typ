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
#let chip = load_chip("src/decode.toml", config)
#show: book-page.with(title: "DECODE chip")

#let decode = raw(chip.name)

= #decode chip

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The #decode chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

#pagebreak()
== Decoding
We specify how each instruction should be expressed in the decoding table. Unspecified values are either 
+ the corresponding element of the instruction (e.g. `rs1`), or 
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
  columns: (110pt, auto, auto),
  stroke: 0pt,
  inset: (right: 1em),
  align: left + bottom,
  table.header([*Operation*], [*Op-flag*], [*Other*]),
  table.hline(stroke: 1.5pt),

  ..subheader("OP-IMM", [$#`rs2`:=0$, $#`write_register` := (#`rd` eq.not 0)$]),
  table.hline(stroke: .5pt),
  [`ADDI  rd, rs1, imm`], [`ADD`], [],
  [`ADDIW rd, rs1, imm`], [`ADD`], [`word_instr`],
  [`SLTI  rd, rs1, imm`], [`SLT`], [`signed`],
  [`SLTIU rd, rs1, imm`], [`SLT`], [],
  [`ANDI  rd, rs1, imm`], [`AND`], [],
  [`ORI   rd, rs1, imm`], [`OR`], [],
  [`XORI  rd, rs1, imm`], [`XOR`], [],
  [`SLLI  rd, rs1, imm`], [`SHIFT`], [],
  [`SLLIW rd, rs1, imm`], [`SHIFT`], [`word_instr`],
  [`SRLI  rd, rs1, imm`], [`SHIFT`], [`mp_selector`],
  [`SRLIW rd, rs1, imm`], [`SHIFT`], [`mp_selector`, `word_instr`],
  [`SRAI  rd, rs1, imm`], [`SHIFT`], [`mp_selector`, `signed`],
  [`SRAIW rd, rs1, imm`], [`SHIFT`], [`mp_selector`, `signed`, `word_instr`],

  ..subheader("LUI/AUIPC", []),
  [`LUI   rd, imm`], [#sym.arrow.double.r], [`ADDI rd, x0, imm`],
  [`AUIPC rd, imm`], [#sym.arrow.double.r#footnote("The program counter (pc) is stored in register 255.")<pc-index-255>], [`ADDI rd, 255, imm`],

  ..subheader("OP", [$#`imm`:=0$, $#`write_register` := (#`rd` eq.not 0)$]),
  [`ADD   rd, rs1, rs2`], [`ADD`], [],
  [`ADDW  rd, rs1, rs2`], [`ADD`], [`word_instr`],
  [`SUB   rd, rs1, rs2`], [`SUB`], [],
  [`SUBW  rd, rs1, rs2`], [`SUB`], [`word_instr`],
  [`SLT   rd, rs1, rs2`], [`SLT`], [`signed`],
  [`SLTU  rd, rs1, rs2`], [`SLT`], [],
  [`AND   rd, rs1, rs2`], [`AND`], [],
  [`OR    rd, rs1, rs2`], [`OR`], [],
  [`XOR   rd, rs1, rs2`], [`XOR`], [],
  [`SLL   rd, rs1, rs2`], [`SHIFT`], [],
  [`SLLW  rd, rs1, rs2`], [`SHIFT`], [`word_instr`],
  [`SRL   rd, rs1, rs2`], [`SHIFT`], [`mp_selector`],
  [`SRLW  rd, rs1, rs2`], [`SHIFT`], [`mp_selector`, `word_instr`],
  [`SRA   rd, rs1, rs2`], [`SHIFT`], [`mp_selector`, `signed`],
  [`SRAW  rd, rs1, rs2`], [`SHIFT`], [`mp_selector`, `signed`, `word_instr`],

  ..subheader("BRANCHING (unconditional)", []),
  [`JAL   rd, imm`], [#sym.arrow.double.r#footnote(<pc-index-255>)], [`JALR rd, 255, imm`],
  [`JALR  rd, rs1, imm`], [`JALR`], [$#`write_register` := #`rd` eq.not 0$],

  ..subheader("BRANCHING (conditional)", [$#`write_register` := 0$]),
  [`BEQ   rs1, rs2, imm`], [`BEQ`], [],
  [`BNE   rs1, rs2, imm`], [`BEQ`], [`mp_selector`],
  [`BLT   rs1, rs2, imm`], [`BLT`], [`signed`],
  [`BLTU  rs1, rs2, imm`], [`BLT`], [],
  [`BGE   rs1, rs2, imm`], [`BLT`], [`signed`, `mp_selector`],
  [`BGEU  rs1, rs2, imm`], [`BLT`], [`mp_selector`],
  [`BGT   rs1, rs2, imm`], [#sym.arrow.double.r #footnote["BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively", RISC-V unprivileged ISA, page 32.] <bgt-bgtu-ble-bleu>], [`BLT `  *`rs2, rs1`*,` imm`],
  [`BGTU  rs1, rs2, imm`], [#sym.arrow.double.r  #footnote(<bgt-bgtu-ble-bleu>)], [`BLTU`  *`rs2, rs1`*,` imm`],
  [`BLE   rs1, rs2, imm`], [#sym.arrow.double.r  #footnote(<bgt-bgtu-ble-bleu>)], [`BGE `  *`rs2, rs1`*,` imm`],
  [`BLEU  rs1, rs2, imm`], [#sym.arrow.double.r  #footnote(<bgt-bgtu-ble-bleu>)], [`BGEU`  *`rs2, rs1`*,` imm`],

  ..subheader("LOAD", []),
  [`LD    rd, rs1, imm`], [`LOAD`], [`memory_2bytes`, `memory_4bytes`, `memory_8bytes`],
  [`LW    rd, rs1, imm`], [`LOAD`], [`memory_2bytes`, `memory_4bytes`, `signed`],
  [`LWU   rd, rs1, imm`], [`LOAD`], [`memory_2bytes`, `memory_4bytes`],
  [`LH    rd, rs1, imm`], [`LOAD`], [`memory_2bytes`, `signed`],
  [`LHU   rd, rs1, imm`], [`LOAD`], [`memory_2bytes`],
  [`LB    rd, rs1, imm`], [`LOAD`], [`signed`],
  [`LBU   rd, rs1, imm`], [`LOAD`], [],

  ..subheader("STORE", []),
  [`SD    rs1, rs2, imm`], [`STORE`], [`memory_2bytes`, `memory_4bytes`, `memory_8bytes`],
  [`SW    rs1, rs2, imm`], [`STORE`], [`memory_2bytes`, `memory_4bytes`],
  [`SH    rs1, rs2, imm`], [`STORE`], [`memory_2bytes`],
  [`SB    rs1, rs2, imm`], [`STORE`], [],

  ..subheader("SYSTEM", []),
  [`ECALL`], [`ECALL`], [`write_register`, $#`rs1` := #`x17`$, $#`rs2` := #`x11`$, $#`rd` := #`x10`$],
  [`EBREAK`], [`EBREAK`], [],

  ..subheader("OP (M-extension)", [$#`imm` := 0$, $#`write_register` := #`rd` eq.not 0$]),
  [`MUL    rd, rs1, rs2`], [`MUL`], [`signed`, `mp_selector`],
  [`MULW   rd, rs1, rs2`], [`MUL`], [`signed`, `mp_selector`, `word_instr`],
  [`MULH   rd, rs1, rs2`], [`MUL`], [`signed`, `mp_selector`, `muldiv_selector`],
  [`MULHU  rd, rs1, rs2`], [`MUL`], [`muldiv_selector`],
  [`MULHSU rd, rs1, rs2`], [`MUL`], [`signed`, `muldiv_selector`],
  [`DIV    rd, rs1, rs2`], [`DIVREM`], [`signed`],
  [`DIVW   rd, rs1, rs2`], [`DIVREM`], [`signed`, `word_instr`],
  [`DIVU   rd, rs1, rs2`], [`DIVREM`], [],
  [`DIVUW  rd, rs1, rs2`], [`DIVREM`], [`word_instr`],
  [`REM    rd, rs1, rs2`], [`DIVREM`], [`signed`, `muldiv_selector`],
  [`REMW   rd, rs1, rs2`], [`DIVREM`], [`signed`, `muldiv_selector`, `word_instr`],
  [`REMU   rd, rs1, rs2`], [`DIVREM`], [`muldiv_selector`],
  [`REMUW  rd, rs1, rs2`], [`DIVREM`], [`muldiv_selector`, `word_instr`],

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
