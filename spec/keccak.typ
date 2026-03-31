#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/keccak.toml", config)

#show: book-page(chip.name)
#let keccak = raw(chip.name)

The #keccak chip applies the keccak permutation.

= Core chip
== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The #keccak chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_variable_table(chip, config)

== Assumptions

// #render_chip_assumptions(chip, config)

== Constraints
In this VM, we assign syscall number -2 to the #keccak accelerator.
The chip therefore contributes the following interaction to the lookup-argument:
#render_constraint_table(chip, config, groups: "output")

The address containing the state to be permuted are passed in as argument `A0 = x10`.
This address is read into `addr`, from which `state_ptr` --- the collection of pointers to all lanes of the state --- is derived.
The state is then read into `input_state`, while the `output_state` is written back to the indicated address.
#render_constraint_table(chip, config, groups: "mem")

Lastly, the input state is pushed to the Keccak-round function, while the output after 24 rounds is taken off the bus.
#render_constraint_table(chip, config, groups: "round")

== Padding

The table can be padded to the next power of two with the following value assignments:

#render_chip_padding_table(chip, config)

= Round chip
#let round_chip = load_chip("src/keccak_round.toml", config)
#let keccak_rnd = raw(round_chip.name)

== Columns
#let nr_variables = total_nr_variables(round_chip)
#let nr_columns = total_nr_instantiated_columns(round_chip, config)

The #keccak_rnd chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_variable_table(round_chip, config)


== Constraints

#render_constraint_table(round_chip, config)