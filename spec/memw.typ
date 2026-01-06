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
#let chip = load_chip("src/memw.toml", config)

#show: book-page.with(title: "MEMW chip")

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `MEMW` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

== Assumptions

#render_chip_assumptions(chip, config)

== Constraints

#render_constraint_table(chip, config, groups: "consistency")

The chip adds the following tuples to the lookup argument,
to effectuate that part of the memory argument.
#render_constraint_table(chip, config, groups: "memory")

This chip contributes the following to the lookup argument.
#render_constraint_table(chip, config, groups: "output")



