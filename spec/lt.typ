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
#let chip = load_chip("src/lt.toml", config)

#show: book-page.with(title: "LT chip")

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `LT` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

== Assumptions
We assume the inputs `lhs`, `rhs` and `signed` are appropriately range checked.
#render_chip_assumptions(chip, config)

== Constraints
We first constrain that all variables correspond to their definition.
#rj[Explain formulae properly, including sign bit logic and how overflow only matters if signs differ]

#render_constraint_table(chip, config, groups: "defs")

And then we constrain the subtraction.

#render_constraint_table(chip, config, groups: "sub")

The chip contributes the following to the lookup argument.

#render_constraint_table(chip, config, groups: "output")
