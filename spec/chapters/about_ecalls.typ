#import "/meta.typ": aside
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

/ 64: `write` (@copy)
/ 93: `exit` (@halt)
/ -1: `SHA256` (@sha256)
/ -2: `KECCAK` (@keccak)
/ -11: `ECSM`/`secp256k1` (@ecsm)
/ -12: `ECSM`/`secp256r1` (@ecsm)
/ -20: `FEXT_LOAD` (@fext)
/ -21: `FEXT_FMA` (@fext)
/ -22: `FEXT_ZERO` (@fext)
/ -30: `memcpy`/`memmove` (@copy)
/ -31: `memset` (@copy)

== Committing to values

In order to make a claim about a public value to the verifier, a guest program can _commit_ to it.
In this VM, this is achieved by writing to `stdout`, with a write syscall to file descriptor 1.
Values are committed by letting the verifier initialize and finalize the global memory argument
(see @memory and @streaming), with the claimed commitments in its own domain separated part of memory,
with domain separator value 2.#footnote[
  In order to make sure the verifier can properly finalize the committed values, the last epoch can "bring forward"
  all commitments from earlier epochs, similar to padded values, in the `L2G` table.
  Then the contribution of the commitments only consists of the tuples `(2, address, last_epoch_index, value)`, which is entirely known to the verifier.
]
In doing this, we enforce that all values being committed match the claimed commitment.
The verifier should additionally check that register 254 contains the total number of bytes committed.
The technical details on how the copy to memory domain 2 is achieved can be found in @copy, where `write` is implemented.
