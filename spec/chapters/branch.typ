#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_variable_table,
  compute_nr_interactions,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/branch.toml", config)
#let branch = raw(chip.name)

The #branch chip computes the target address of a branching instruction.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #branch chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Assumptions

#render_chip_assumptions(chip, config)

Some of the assumptions can be checked with only arithmetic constraints, so we
provide these below.

#render_constraint_table(chip, config, groups: "assumptions")

= Constraints

We constrain `next_pc` to be $#`base_address` + #`offset`$,
where `base_address` equals `pc` when $#`JALR` = 0$ and `register` otherwise.

The range checks on `unmasked_low_byte` and `next_pc_low[0]` are performed implicitly by the `AND_BYTE` lookup.
#render_constraint_table(chip, config, groups: "all")

This chip contributes the following to the lookup argument.
#render_constraint_table(chip, config, groups: "output")

= Padding

The table can be padded to the next power of two with the following value assignments:

#render_chip_padding_table(chip, config)
