#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_assumptions,
)

#let config = load_config()
#let chip = load_chip("src/mul.toml", config)

#show: book-page.with(title: "MUL chip")


#outline()

= Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `MUL` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)


= Assumptions
#render_chip_assumptions(chip, config)


= Constraints
#render_constraint_table(chip, config, groups: "def")
#render_constraint_table(chip, config, groups: "prod")
*Note*: by the definition of `raw_product`, all components of the sum are of degree at most three.
#render_constraint_table(chip, config, groups: "lookup")