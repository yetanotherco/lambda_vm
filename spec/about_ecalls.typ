#import "/book.typ": book-page, aside
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_assumptions,
  render_chip_padding_table,
)

#let config = load_config()

#show: book-page("about_ecalls.typ")

ECALLs provide system-level functionalities to the guest program.

When `ECALL` is executed, it is assumed that:
- register `A7` contains the system call number
  #footnote([The RISC-V system call ABI; libriscv.no, #link("https://web.archive.org/web/20260128152107/https://libriscv.no/docs/concepts/syscalls/#the-risc-v-system-call-abi")[[src]]]),
- the arguments are located in registers `A0`-`A6`, and
- the return value is written to `A0`,
where `A0`-`A7` are symbolic names for the registers `x10`-`x17`
#footnote([RISC-V - Register sets; en.wikipedia.org, #link("https://web.archive.org/web/20260209053447/https://en.wikipedia.org/wiki/RISC-V#Register_sets")[[src]]]).

= ECALL number overview

We provide a list of supported ECALL numbers.
Negative numbers (represented as 2s complement 64-bit numbers), are used for our own custom accelerators/extensions.

/ 64: `write` (@commit)
/ 93: `exit` (@halt)
/ -1: `SHA256` (@sha256)
/ -2: `KECCAK` (@keccak)
/ -3: `ECSM` (@ecsm)