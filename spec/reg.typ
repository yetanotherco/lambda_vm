#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, render_constraint_table, compute_nr_interactions,

#let config = load_config()
#let reg_read = load_chip("src/reg_read.toml", config)
#let reg_write = load_chip("src/reg_write.toml", config)
#let read = raw(reg_read.name)
#let write = raw(reg_write.name)

#show: book-page("REG")


= #read
#read is a constraint template 

== Variables
#let nr_interactions = compute_nr_interactions(reg_read)

This template introduces #nr_interactions interaction(s).
#render_chip_variable_table(reg_read, config)

== Constraints
We constrain this equality using two constraints:
#render_constraint_table(reg_read, config)


= #write
#write is a constraint template 

== Variables
#let nr_interactions = compute_nr_interactions(reg_write)

This template introduces #nr_interactions interaction(s).
#render_chip_variable_table(reg_write, config)

== Constraints
We constrain this equality using two constraints:
#render_constraint_table(reg_write, config)
