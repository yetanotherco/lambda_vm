#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, render_constraint_table, set_nr_interactions,

#let config = load_config()
#let reg_read = load_chip("src/reg_read.toml", config)
#let reg_write = load_chip("src/reg_write.toml", config)
#let read = raw(reg_read.name)
#let write = raw(reg_write.name)

#show: book-page("REG")

We provide the #read and #write templates.
These templates act as short hand notation for `MEMW` interactions pertaining to operations in the `register` domain ($= 1$, see @memory).
Interactions on this domain can denote more concisely since
+ the register address can be represented using a one-column `Byte` rather than a two-column `DWordWL`, and
+ registers accesses (almost) always have a width of 2.
Moreover, the factor 2 that must be included in converting from register-number to the register-address has led to several hard-to-catch errors in the past; introducing this abstraction will reduce the likelihood of these mistakes.

The necessity for two templates follows from the existence of two subtly different `MEMW` signatures (see @signatures): there is both a read-write and an write-only interaction.
Here, the former is wrapped by #read, while the #write template encapsulates the latter.

For both templates hold: when not provided, `cond` defaults to `1`.

= #read

== Variables
#set_nr_interactions(reg_read)
#render_chip_variable_table(reg_read, config)

== Constraints
#render_constraint_table(reg_read, config)

#pagebreak(weak: true)

= #write

== Variables
#set_nr_interactions(reg_write)
#render_chip_variable_table(reg_write, config)

== Constraints
#render_constraint_table(reg_write, config)
