#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, render_constraint_table, compute_nr_interactions, total_nr_variables, total_nr_instantiated_columns

#let config = load_config()
#let chip = load_chip("src/is_byte.toml", config)
#let is_byte = raw(chip.name)

#is_byte is a constraint template that is used to assert that a variable lies in the range $[0, 255]$ under the condition that `cond` is non-zero. Note: when `cond` is omitted, it defaults to $1$.

When a chip leverages this template twice or more, implementors are encouraged to merge pairs of #is_byte interactions with identical conditions into `ARE_BYTES` interactions; the #is_byte template is included for convenience of notation, and to complete the specification of chips that use an odd number of #is_byte range checks.

= Variables
#let nr_interactions = compute_nr_interactions(chip)

The #is_byte template leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Constraints
#render_constraint_table(chip, config)
