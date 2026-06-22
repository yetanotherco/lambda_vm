#import "/meta.typ": et
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, render_chip_assumptions, render_constraint_table, set_nr_interactions, compute_nr_interactions,

#let config = load_config()
#let chip = load_chip("src/add.toml", config)

#set_nr_interactions(chip, name: "SUB")
#let nr_interactions = compute_nr_interactions(chip)

#let add = raw(chip.name)
#let sub = raw("SUB")

#add is a constraint template that is used to assert that $#`sum` equiv #`lhs` + #`rhs` (mod 2^64)$, under the condition that `cond` is non-zero.
For ease of notation, we moreover introduce the #sub constraint template
$
#`SUB<diff; lhs, rhs>` := #`ADD<lhs; rhs, diff>`,
$
in both conditional and unconditional versions.
It constrains that $#`diff` equiv #`lhs` - #`rhs` (mod 2^64)$ when the expression `cond` is non-zero.

= Variables
This template introduces #nr_interactions interaction(s).
#render_chip_variable_table(chip, config)

= Assumptions
#render_chip_assumptions(chip, config)

= Constraints
This template introduces the following constraints
#render_constraint_table(chip, config)
