#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, render_constraint_table, compute_nr_interactions,

#let config = load_config()
#let chip = load_chip("src/reg.toml", config)
#show: book-page(chip.name)

#let nr_interactions = compute_nr_interactions(chip)

#let reg = raw(chip.name)

#reg is a constraint template 

= Variables
This template introduces #nr_interactions interaction(s).
#render_chip_variable_table(chip, config)

= Constraints
We constrain this equality using two constraints:
#render_constraint_table(chip, config)
