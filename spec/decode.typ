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

== Decoding
We specify how each instruction should be expressed in the decoding table. Unspecified values are either 
+ the corresponding element of the instruction (e.g. `rs1`), or 
+ set to $0$ otherwise.

=== OP-IMM
In general:
+ $#`rs2` = 0$
+ $#`write_register` = min(1, #`rd`)$
#table(
  columns: (auto, auto, auto),
  stroke: 0pt,
  inset: (right: 1em),
  table.header([*Operation*], [*Op-flag*], [*Other*]),
  table.hline(stroke: 1.5pt),
  [`ADDI  rd, rs1, imm`], [`ADD`], [],
  [`SLTI  rd, rs1, imm`], [`SLT`], [`signed`],
  [`SLTIU rd, rs1, imm`], [`SLT`], [],
  [`ANDI  rd, rs1, imm`], [`AND`], [],
  [`ORI   rd, rs1, imm`], [`OR`], [],
  [`XORI  rd, rs1, imm`], [`XOR`], [],
  [`SLLI  rd, rs1, imm`], [`SHIFT`], [],
  [`SRLI  rd, rs1, imm`], [`SHIFT`], [`mp_selector` ],
  [`SRAI  rd, rs1, imm`], [`SHIFT`], [`mp_selector`, `signed`],
)

=== LUI / AUIPC
Note: these operations load/add `imm` in/to the upper bits of `rd`/`pc`. 
As such, *we expect the decoding to take care of writing the immediate in the upper bits of `imm`*.

- `LUI rd, imm` #sym.arrow.double.r `ADDI rd, x0, imm`
- `AUIPC rd, imm` #sym.arrow.double.r `ADDI rd, 255, imm`
  - Note: PC index $= 255$

=== OP
In general:
+ $#`imm` = 0$
+ $#`write_register` = min(1, #`rd`)$
#table(
  columns: (auto, auto, auto),
  stroke: 0pt,
  inset: (right: 1em),
  table.header([*Operation*], [*Op-flag*], [*Other*]),
  table.hline(stroke: 1.5pt),
  [`ADD  rd, rs1, rs2`], [`ADD`], [],
  [`SUB  rd, rs1, rs2`], [`SUB`], [],
  [`SLT  rd, rs1, rs2`], [`SLT`], [`signed`],
  [`SLTU rd, rs1, rs2`], [`SLT`], [],
  [`AND  rd, rs1, rs2`], [`AND`], [],
  [`OR   rd, rs1, rs2`], [`OR`], [],
  [`XOR  rd, rs1, rs2`], [`XOR`], [],
  [`SLL  rd, rs1, rs2`], [`SHIFT`], [],
  [`SRL  rd, rs1, rs2`], [`SHIFT`], [`mp_selector` ],
  [`SRA  rd, rs1, rs2`], [`SHIFT`], [`mp_selector`, `signed`],
)

=== BRANCH
- `JAL rd, imm` #sym.arrow.double.r `JALR rd, 255, imm`
  - Note: PC index $= 255$

// TODO: JALR

In general:
+ $#`imm` = 0$
+ $#`write_register` = 0$
#table(
  columns: (auto, auto, auto),
  stroke: 0pt,
  inset: (right: 1em),
  table.header([*Operation*], [*Op-flag*], [*Other*]),
  table.hline(stroke: 1.5pt),
  [`BEQ   pc, rs1, rs2, imm`], [`BEQ`], [],
  [`BNE   pc, rs1, rs2, imm`], [`BEQ`], [`mp_selector`],
  [`BLT   pc, rs1, rs2, imm`], [`BLT`], [`signed`],
  [`BLTU  pc, rs1, rs2, imm`], [`BLT`], [],
  [`BGE   pc, rs1, rs2, imm`], [`BLT`], [`signed`, `mp_selector`],
  [`BGEU  pc, rs1, rs2, imm`], [`BLT`], [`mp_selector`],
)
*Note*: "BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively", RISC-V unprivileged ISA, page 32. In other words, these four operations are pseudo-instructions.


=== LOAD
#table(
  columns: (auto, auto, auto),
  stroke: 0pt,
  inset: (right: 1em),
  table.header([*Operation*], [*Op-flag*], [*Other*]),
  table.hline(stroke: 1.5pt),
  [`LOAD rd, rs1, imm, width, sign_extend`], [`LOAD`], [
    $#`memory_2bytes` := #`width` >= 2$\
    $#`memory_4bytes` := #`width` >= 4$\
    $#`memory_8bytes` := #`width` >= 8$\
    $#`signed` := #`sign_extend`$
  ],
)

=== STORE
#table(
  columns: (auto, auto, auto),
  stroke: 0pt,
  inset: (right: 1em),
  table.header([*Operation*], [*Op-flag*], [*Other*]),
  table.hline(stroke: 1.5pt),
  [`STORE rs1, rs2, imm, width`], [`STORE`], [
    $#`memory_2bytes` := #`width` >= 2$\
    $#`memory_4bytes` := #`width` >= 4$\
    $#`memory_8bytes` := #`width` >= 8$
  ],
)


=== MISC-MEM
- `FENCE` #sym.arrow.double.r `ADDI 0, x0, 0`
  - Note: this is a NOP

=== System
#table(
  columns: (auto, auto, auto),
  stroke: 0pt,
  inset: (right: 1em),
  table.header([*Operation*], [*Op-flag*], [*Other*]),
  table.hline(stroke: 1.5pt),
  [`ECALL`], [`ECALL`], [`write_register`, $#`rs1` := 17$, $#`rs2` := 11$, $#`rd` := 10$],
  [`EBREAK`], [`EBREAK`], [],
)

Note for `ECALL`: 
*“On RISC-V a system call has its own instruction: `ECALL`. A system call can have up to 7 arguments and has 1 return value. The arguments are in registers A0-A6, in that order, and the return value is written into A0 before giving back control to the guest. A7 contains the system call number.”* [[src](https://libriscv.no/docs/concepts/syscalls/#the-risc-v-system-call-abi)]
- syscall number in A7 (= register `x17`)
- first syscall argument in A1 (= register `x11`)
- syscall output in A0 (= register `x10`)

=== OP (M - extension)
#table(
  columns: (auto, auto, auto),
  stroke: 0pt,
  inset: (right: 1em),
  table.header([*Operation*], [*Op-flag*], [*Other*]),
  table.hline(stroke: 1.5pt),
  [`MUL    rd, rs1, rs2`], [`MUL`], [`signed`, `mp_selector`],
  [`MULH   rd, rs1, rs2`], [`MUL`], [`signed`, `mp_selector`, `muldiv_selector`],
  [`MULHU  rd, rs1, rs2`], [`MUL`], [`muldiv_selector`],
  [`MULHSU rd, rs1, rs2`], [`MUL`], [`signed`, `muldiv_selector`],
)

#table(
  columns: (auto, auto, auto),
  stroke: 0pt,
  inset: (right: 1em),
  table.header([*Operation*], [*Op-flag*], [*Other*]),
  table.hline(stroke: 1.5pt),
  [`DIV  rd, rs1, rs2`], [`DIVREM`], [`signed`],
  [`DIVU rd, rs1, rs2`], [`DIVREM`], [],
  [`REM  rd, rs1, rs2`], [`DIVREM`], [`signed`, `muldiv_selector`],
  [`REMU rd, rs1, rs2`], [`DIVREM`], [`muldiv_selector`],
)