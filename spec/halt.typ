#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_column_table,
  render_chip_padding_table,
  render_constraint_table,
  total_nr_instantiated_columns,
  total_nr_variables,
)

#let config = load_config()
#let chip = load_chip("src/halt.toml", config)
#let halt = raw(chip.name)

#show: book-page.with(title: "Halt chip")

= #halt chip

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The #halt chip leverages #nr_variables variables, spanning #nr_columns columns:
#render_chip_column_table(chip, config)

== Assumptions
It is assumed the input is range checked:
#render_chip_assumptions(chip, config)

== Constraints
The #halt chip:
+ makes sure register `x10` (containing the exit code) equals $0$ (@halt:c:read_zero_exit_code),
+ writes $0$ to all other registers (@halt:c:zeroize_registers_lo/@halt:c:zeroize_registers_hi), and
+ sets `pc` equal to $1$ (@halt:c:pc).
#render_constraint_table(chip, config, groups: "all")

=== Lookup
The HALT chip contributes the following interaction to the lookup-argument:
#render_constraint_table(chip, config, groups: "lookup")

*Note*: #link("https://github.com/riscv-collab/riscv-gnu-toolchain/blob/master/linux-headers/include/asm-generic/unistd.h#L258")[$93$ is the system call number corresponding to `sys_exit`.]

== Padding
This chip should only contain a single row.
Given that $2^0 = 1$, this chip does not need to be padded.
As such, no padding is defined.
