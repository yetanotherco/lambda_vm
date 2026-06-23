#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  compute_nr_interactions,
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

The #keccak chip applies the keccak permutation $kappa$ to a given memory range;
other aspects of keccak hashing (such as repeated permutation invocation, 
input padding and state initialization) fall outside the scope of this accelerator.

This permutation $kappa: FF_2^1600 -> FF_2^1600$ operates on 1600 bits and is composed of 24 applications of round-permutation $Lambda: FF_2^1600 times NN -> FF_2^1600$, where the additional parameter is the round constant.
$Lambda$ is defined as the composition $iota compose chi compose pi compose rho compose theta$, where only $iota$ depends on the round constant.
#footnote("More details on the KECCAK permutation: FIPS 202, NIST, " + link("https://csrc.nist.gov/pubs/fips/202/final"))

The keccak accelerator comprises two chips: a core chip that interacts with the memory --- loading the input and writing the output, and a round chip that applies the round permutation.


= Core chip
== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #keccak chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

== Constraints
In this VM, we assign syscall number -2 to the #keccak accelerator.
The chip therefore contributes the following interaction to the lookup-argument:
#render_constraint_table(chip, config, groups: "output")

The address containing the state to be permuted is passed in as argument `A0 = x10`.
The following constraints describe that this address is read into `state_ptr[0][0]` (@keccak:c:read_state_ptr), from which full `state_ptr` --- the collection of pointers to all lanes of the state --- is derived (@keccak:c:state_ptr); @keccak:c:range_state_ptr is included to satisfy @add:a:lhs respectively @add:a:sum.
The state is then read into `input_state`, while the `output_state` is written back to the indicated address (@keccak:c:load_store_state).
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
#let nr_interactions = compute_nr_interactions(round_chip)

The #keccak_rnd chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(round_chip, config)

#strong("Note on " + raw("start") + ".")
`start` contains the state to which the permutation should be applied.
Its three-dimensional array mimics the specification's three-dimensional state
#footnote("FIPS 202, NIST, Section 3.1 (" + link("https://csrc.nist.gov/pubs/fips/202/final") + ")")
and orders the bits as prescribed.
#footnote("FIPS 202, NIST, Section B.1, Algorithm 10 (" + link("https://csrc.nist.gov/pubs/fips/202/final") + ")")

#strong("Note on " + raw("rnc") + " and " + raw("rbc") + ".")
Rho rotates every lane by a rotation offset in $[0, 64)$.
These offsets are identical for every round.
#footnote("FIPS 202, NIST, page 13, Table 2 (" + link("https://csrc.nist.gov/pubs/fips/202/final") + ")")
We decompose each offset in three components: the lower nibble (4 bits) are represented by `rnc`, while the upper two bits are represented by as `Bit`s in `rbc`.
That is, $#`rho_offset[x][y]` = #`rnc[x][y]` + 16 dot #`rbc[x][y][0]` + 32 dot #`rbc[x][y][1]`$.


== Constraints

The following constraints ensure that `theta` captures the state after applying the first subpermutation of the round-permutation: $theta$.
Note here that `Cxz_left` and `Cxz_right` do have to be range-checked; it cannot be assumed that this implicitly follows from @keccak:c:Dxz combined with `rotated_Cxz`'s definition.
#render_constraint_table(round_chip, config, groups: "theta")

Next, we constrain that `rho` captures the state after applying subpermutation $rho$.
Note here as well that `rot_left` and `rot_right` do have to be range-checked; it cannot be assumed that this implicitly follows from later constraints.
#render_constraint_table(round_chip, config, groups: "rho")

Observe that the lane-permutation performed by $pi$ is absorbed in `pi`'s definition.
The next permutation that is constrained in $chi$:
#render_constraint_table(round_chip, config, groups: "chi")

Lastly, the round constants are added to one of the lanes in the state.
`iota` contains the updated lane.
In the definition of `out`, the output of `chi` and `iota` is combined to construct the output of the permutation.
#render_constraint_table(round_chip, config, groups: "iota")

Lastly, the round chip contributes the following interactions to the lookup:
#render_constraint_table(round_chip, config, groups: "io")

== Notes/potential optimizations
- one does not have to range check `state_ptr[0][0]` since it is read from memory.
  Moreover, it could be represented as a `DWordWL`. 
  All this would save 2 columns and 4 `IS_HALF` checks.
- step $rho$ does not need to be applied to `state[0][0]`; its has a zero-shift. This saves 16 columns and 4 `HWSL` interactions.
- when the output of `HWSL` are `Byte`s mapped as `Half`s, we find that out of every four output bytes, at least one is zero. 
  Since `rnc` is constant, @keccak:c:rho_rotation makes those zero-bytes show up in `rot_left` and `rot_right` at constant locations.
  This means 96 columns can be removed from the chip at no cost.
  Likewise, 96 `IS_BYTE` interactions can be dropped from @keccak:c:range_rot_left and @keccak:c:range_rot_right.
  - the shift-constants are equivalent to $1 mod 16$ for $(#`x`, #`y`) = (1, 0)$ and $-1 mod 16$ for $(2, 3)$. This means that for those lanes it suffices to constrain `rot_left`/`rot_right` as `Bit`s rather than `Byte`s, saving an additional 8 `IS_BYTE` interactions.
- $#`rc[2]` = #`rc[4]` = #`rc[5]` = #`rc[6]` = 0$. As such, those elements need not be stored in `rc`, and need not be XORed into the state in the $iota$-step. This saves 8 columns and 4 `XOR_BYTE` interactions.
- when executed in large volumnes, `KECCAK_RND` could benefit from having a three-way XOR lookup table. With this in place, the 80 interactions in @keccak:c:theta_cxz_start and @keccak:c:theta_cxz could be dropped.
  Likewise, 80 columns could be removed from the chip (a \~5% savings).

= Round constant lookup
#let rc_chip = load_chip("src/keccak_rc.toml", config)
#let keccak_rc = raw(rc_chip.name)

== Columns
#let nr_variables = total_nr_variables(rc_chip)
#let nr_columns = total_nr_instantiated_columns(rc_chip, config)

We provide the round constants through a short precomputed lookup table: #keccak_rc.
#render_chip_variable_table(rc_chip, config)
#render_constraint_table(rc_chip, config)