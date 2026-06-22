#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, total_nr_variables, render_chip_assumptions, render_constraint_table, compute_nr_interactions,

#let config = load_config()
#let chip = load_chip("src/sign.toml", config)

#let nr_variables = total_nr_variables(chip)
#let nr_interactions = compute_nr_interactions(chip)

#let sign = raw(chip.name)

#sign is a constraint template that is used to extract a `Half`word's sign.
It constrains that `sign` is set to `1` when both `X`'s most significant bit and `signed` are $1$, and $0$ otherwise.

= Variables
The #sign template introduces #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Assumptions
The #sign template operates on the following assumptions:
#render_chip_assumptions(chip, config)

If `sign` is set to $1$, `X` will be range-checked to be a halfword, and hence proving may fail if this is not ensured.

= Constraints
It takes only two constraints to compute the `sign` of `X`, given whether `X` represents a `signed` value or not. 
When $#`signed` = 1$, the sign of `X` is equal to its most significant bit. 
This value is extracted in @sign:c:sign_if_signed.
If `X` is unsigned (i.e., $#`signed` = 0$), its sign is always $0$.
This is constrained by @sign:c:sign_if_unsigned.
#render_constraint_table(chip, config)
