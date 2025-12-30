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

= Assumptions
#render_chip_assumptions(chip, config)

= Constraints
== Definitions
Constrain the auxiliary variables `bit_shift`, `limb_shift_0`, `limb_shift_1`, `limb_shift_2` and `limb_shift_3` according to their definitions.
#render_constraint_table(chip, config, groups: "defs")

== Left shifting
#render_constraint_table(chip, config, groups: "intra_limb_left_shifting")
#render_constraint_table(chip, config, groups: "limb_shifting")
#render_constraint_table(chip, config, groups: "limb_left_shifting")
#render_constraint_table(chip, config, groups: "limb_left_shifting_zero")

== Right shifting
#render_constraint_table(chip, config, groups: "intra_limb_right_shifting")
#render_constraint_table(chip, config, groups: "limb_right_shifting")
#render_constraint_table(chip, config, groups: "limb_right_shifting_extension")

== Lookup
#render_constraint_table(chip, config, groups: "lookups")