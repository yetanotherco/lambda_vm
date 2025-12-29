#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_assumptions,
)

#let config = load_config()
#let chip = load_chip("src/shift.toml", config)

#show: book-page.with(title: "SHIFT chip")


#outline()

= Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `SHIFT` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

// == Assumptions
// #render_chip_assumptions(chip, config)

= Constraints
== Definitions
For starters, constrain the auxiliary variables `bit_shift`, `limb_shift_odd`, `limb_shift_even`, `limb_shift_1`, `limb_shift_2` and `limb_shift_3` according to their definitions.
#render_constraint_table(chip, config, groups: "defs")
*Implementation note*: one could remove the $1/512$ factor listed in the definitions of `limb_shift_1`, `limb_shift_2` and `limb_shift_3`.
While this would mean that the three are no longer `Bit`s, this fact does not impact the correctness of the chip: all constraints using these variables only require the variables to be _non-zero_, not specifically $1$.

== Left shifting
Shifting (both left and right) is achieved in a two-step process: first, shift `in` by `bit_shift`, and then shift the limbs in `bit_shift` the required number of full limbs to form the `shifted` output.
Since left shifting does not need to concern itself with the signedness of `in`, it is slightly more straightforward, and therefore treated first.

#render_constraint_table(chip, config, groups: "intra_limb_left_shifting")

When $floor.l #`bit_shift`/16 floor.r = 0 mod 4$, `bit_shifted` already contains the output.
As such, it suffices to set require `bit_shifted` and `shifted`.
This holds irrespective of whether we are shifting `left` or `right`.
#render_constraint_table(chip, config, groups: "limb_shifting")

When $floor.l #`bit_shift`/16 floor.r eq.not 0 mod 4$, the limbs in `bit_shifted` need to be shifted over by a multiple of 16 bits to form `shifted`.
This exact number is indicated by the variables `limb_shift_1`, `limb_shift_2`, and `limb_shift_3`. 
By construction, exactly one of the three is $1$, while the other two are $0$.
In the following, each case is given its own separate set of constraints.
Here, the limbs in `shifted` are constrained to match those of `bit_shift`, shifted over by the number specified by `limb_shift_x`.
#render_constraint_table(chip, config, groups: "limb_left_shifting")

To complete the output, the lower bits of `shifted` are set to zero, based on the number of limbs `bit_shifted` is shifted over to form `shifted`.
#render_constraint_table(chip, config, groups: "limb_left_shifting_zero")

== Right shifting

#render_constraint_table(chip, config, groups: "intra_limb_right_shifting")
#render_constraint_table(chip, config, groups: "limb_right_shifting")
#render_constraint_table(chip, config, groups: "limb_right_shifting_extension")
