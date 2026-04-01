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

The #keccak chip applies the keccak permutation $kappa$ to a given memory range.

This permutation $kappa: FF_2^1600 -> FF_2^1600$ operates on 1600 bits and is composed of 24 applications of round-permutation $Lambda: FF_2^1600 times NN -> FF_2^1600$, where the additional parameter is the round constant.
$Lambda$ is defined as the composition $iota compose chi compose pi compose rho compose theta$, where only $iota$ depends on the round constant.
#footnote("More details on the KECCAK permutation: FIPS 202, NIST, " + link("https://csrc.nist.gov/pubs/fips/202/final"))

The keccak accelerator comprises two chips: a core chip that interacts with the memory, and a round chip that applies the round permutation.


= Core chip
== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The #keccak chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_variable_table(chip, config)

== Constraints
In this VM, we assign syscall number -2 to the #keccak accelerator.
The chip therefore contributes the following interaction to the lookup-argument:
#render_constraint_table(chip, config, groups: "output")

The address containing the state to be permuted is passed in as argument `A0 = x10`.
The following constraints describe that this address is read into `addr` (@keccak:c:read_addr), from which `state_ptr` --- the collection of pointers to all lanes of the state --- is derived (@keccak:c:state_ptr).
The state is then read into `input_state`, while the `output_state` is written back to the indicated address (@keccack:c:load_store_state).
#render_constraint_table(chip, config, groups: "mem")

Lastly, the input state is pushed to the Keccak-round function, while the output after 24 rounds is taken off the bus:
#render_constraint_table(chip, config, groups: "round")

== Padding
The #keccak table can be padded to the next power of two with the following value assignments:
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