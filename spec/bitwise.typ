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
#let chip = load_chip("src/bitwise.toml", config)

#let bitwise = raw(chip.name)

#show: book-page.with(title: "BRANCH chip")

= #bitwise chip

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_precomputed = ("input", "output").map(c => chip.variables.at(c)).flatten().len()

The #bitwise chip is comprised of #nr_variables variables that are expressed using #nr_columns columns.
Of these, the _input_ and _output_ variables (#nr_precomputed in total) are precomputed.
#render_chip_column_table(chip, config)

*Note*: This table contains one row for every possible value of `(X, Y, Z)`.
As such, it has length $2^5 dot 2^5 dot 2^4 = 2^(20)$.

== Lookup
This chip adds the following interactions to the lookup:
#render_constraint_table(chip, config)