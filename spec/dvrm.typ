#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_assumptions
)

#let config = load_config()
#let chip = load_chip("src/dvrm.toml", config)

#show: book-page.with(title: "DVRM chip")

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `DVRM` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

== Assumptions
#render_chip_assumptions(chip, config)

== Constraints
#render_constraint_table(chip, config, groups:("defs", ))

#render_constraint_table(chip, config, groups:("div_by_zero", ))

#render_constraint_table(chip, config, groups:("overflow", ))

#render_constraint_table(chip, config, groups:("equality", ))

#render_constraint_table(chip, config, groups:("abs_diff", ))

#render_constraint_table(chip, config, groups:("output", ))