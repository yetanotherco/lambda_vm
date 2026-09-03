#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  compute_nr_interactions,
  render_constraint_table,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/eq.toml", config)
#let eq = raw(chip.name)

The #eq chip is an ALU chip that compares two values and outputs a bit indicating whether they are equal or not.
It optionally inverts the result if the `invert` flag is set.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #eq chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Assumptions

#render_chip_assumptions(chip, config)

= Constraints

#render_constraint_table(chip, config)

= Padding

The chip can be padded with the following values:
#render_chip_padding_table(chip, config)
