#import "/meta.typ": aside
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  compute_nr_interactions,
  render_constraint_table,
  render_chip_assumptions,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/halt.toml", config)
#let halt = raw(chip.name)

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #halt chip leverages #nr_variables variable, spanning #nr_columns columns and leverages #nr_interactions interactions:
#render_chip_variable_table(chip, config)

= Assumptions
It is assumed the input is range checked:
#render_chip_assumptions(chip, config)

= Constraints
The #halt chip:
+ makes sure register `x10` (containing the exit code) equals $0$ (@halt:c:read_zero_exit_code),
+ writes $0$ to all other registers (@halt:c:zeroize_registers_lo/@halt:c:zeroize_registers_hi), and
+ sets `pc` equal to $1$ (@halt:c:consume_pc, @halt:c:emit_pc).
Note that the writes performed by all these interactions --- except for the `pc` --- are accompanied by the timestamp $2^32-1$; the maximum timestamp.
This prevents any other operation involving memory from being executed hereafter.
The `pc` is consumed and re-emitted at the same timestamp to enable padding rows for the CPU.
This means that the verifier will have to know the final timestamp at which a CPU padding `pc` was written
to be able to balance the final LogUp.
#render_constraint_table(chip, config, groups: "all")

#aside("Note on register clean up",
[
  Observe that --- in its current state --- this solution puts the burden of verifying the register cleanup on the verifier inside of the lookup argument.
  Alternatively, one could add 31 lookups to the "memory" table to remove the _known_ final tokens for the registers there.
])

== Lookup
In this VM, halting is considered equivalent to executing a `sys_exit`.
Hence, this chip responds to `ECALL`s with system call number 93.
#footnote([RISC-V GNU-toolchain, `unistd.h`; version 2026-01-23, #link("https://github.com/riscv-collab/riscv-gnu-toolchain/blob/2026.01.23/linux-headers/include/asm-generic/unistd.h#L258")[[src]]])
The HALT chip therefore contributes the following interaction to the lookup-argument:
#render_constraint_table(chip, config, groups: "lookup")

= Padding
This chip should only contain a single row.
Given that $2^0 = 1$, this chip does not need to be padded.
As such, no padding is defined.
