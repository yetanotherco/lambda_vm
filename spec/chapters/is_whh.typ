#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  compute_nr_interactions,
  render_chip_variable_table,
  render_constraint_table,
  set_nr_interactions,
  total_nr_instantiated_columns,
  total_nr_variables,
)

#let config = load_config()
#let chip = load_chip("src/is_whh.toml", config)

#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

#let is_whh = raw(chip.name)

#is_whh is a constraint template that is used to assert that a variable of three columns
consists of a `Word` limb and two `Half` limbs.
Using this template is space-reducing when a chip calls it several times,
where for each call it is expected that the `Word` limb and the first `Half`
limb remain constant.

= Variables
The #is_whh chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Constraints
#render_constraint_table(chip, config)
