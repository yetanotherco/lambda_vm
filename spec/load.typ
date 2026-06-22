#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_variable_table,
  render_chip_padding_table,
  render_constraint_table,
  compute_nr_interactions,
  total_nr_instantiated_columns,
  total_nr_variables,
)

#let config = load_config()
#let chip = load_chip("src/load.toml", config)
#let load = raw(chip.name)

The #load chip provides functionality to read values from memory and sign-extend them where appropriate.
It delegates low-level memory handling to the `MEMW` chip (@memw).

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #load chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Assumptions
#render_chip_assumptions(chip, config)

= Constraints
The chip delegates the actual memory interaction to the `MEMW` chip,
and ensures correctness of the requested sign/zero extension.
The output `res` is correctly range-checked as long as the memory contents are.

#render_constraint_table(chip, config, groups: "all")

The chip contributes the following to the lookup argument.

#render_constraint_table(chip, config, groups: "output")

= Padding

The table can be padded to the next power of two with the following value assignments:

#render_chip_padding_table(chip, config)
