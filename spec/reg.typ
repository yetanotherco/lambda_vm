#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, render_constraint_table, set_nr_interactions,

#let config = load_config()
#let reg_read = load_chip("src/reg_read.toml", config)
#let reg_write = load_chip("src/reg_write.toml", config)
#let read = raw(reg_read.name)
#let write = raw(reg_write.name)

We provide the #read and #write templates.
These templates act as short hand notation for `MEMW` interactions pertaining to operations in the `register` domain ($= 1$, see @memory).

The necessity for two templates follows from the existence of two subtly different `MEMW` signatures (see @signatures): there is both a read-write and an write-only interaction.
Here, the former is wrapped by #read, while the #write template encapsulates the latter.

It is recommended to utilize these two templates in favor of direct `MEMW` interactions to specify register updates, as these
+ take care of the crucial register-number to register-address conversion, and
+ provide significantly more concise notation,
improving both readability and correctness of the specification.

= #read
The #read template encapsulates register read-write interactions.
Note: when `cond` is omitted, it defaults to 1.

== Variables
#set_nr_interactions(reg_read)
#render_chip_variable_table(reg_read, config)

== Constraints
#render_constraint_table(reg_read, config)

#pagebreak(weak: true)

= #write
The #write template encapsulates register write-only interactions.
Note: when `cond` is omitted, it defaults to 1.

== Variables
#set_nr_interactions(reg_write)
#render_chip_variable_table(reg_write, config)

== Constraints
#render_constraint_table(reg_write, config)
