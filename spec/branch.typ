#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
)

#let config = load_config()
#let chip = load_chip("src/branch.toml", config)

#show: book-page.with(title: "BRANCH chip")

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `BRANCH` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

== Constraints

We constrain `next_pc` to be `base_address + offset`,
where `base_address` is `pc` when `JALR = 0` and `register` otherwise.
#render_constraint_table(chip, config, groups: "all")

This chip contributes the following to the lookup argument.
#render_constraint_table(chip, config, groups: "output")

