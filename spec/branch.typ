#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/branch.toml", config)

#show: book-page("branch.typ")

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `BRANCH` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

== Assumptions

#render_chip_assumptions(chip, config)

== Constraints

#rj[Check correspondence with CPU for passing in `offset` as word or dword]
We constrain `next_pc` to be $#`base_address` + #`offset`$,
where `base_address` equals `pc` when $#`JALR` = 0$ and `register` otherwise.

The range checks on `unmasked_low_byte` and `next_pc_low[0]` are performed implicitly by the `AND_BYTE` lookup.
#render_constraint_table(chip, config, groups: "all")

This chip contributes the following to the lookup argument.
#render_constraint_table(chip, config, groups: "output")

== Padding

The table can be padded to the next power of two with the following value assignments:

#render_chip_padding_table(chip, config)
