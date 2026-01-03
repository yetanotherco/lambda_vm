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


= Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `SHIFT` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

= Assumptions
#render_chip_assumptions(chip, config)

= About
This chip is designed to enforce that 
$ 
#`out` := cases(
  #`in` #`<<` #`s` " if" #`direction` = 0,
  #`in` #`>>` #`s` " if" #`direction` = 1 and #`signed` = 0,
  #`in` #`>>>` #`s` "if" #`direction` = 1 and #`signed` = 1,
) 
$
where
$ 
#`s` := cases(
  #`shift` mod 32 "if" #`word_instr` = 1,
  #`shift` mod 64 "if" #`word_instr` = 0,
) 
$
Here, `<<` and `>>` denote the _logical_ left and right shift operations, while `>>>` denotes the _arithmetic_ right shift operation.

Note that, while they share many similarities, these six different operations are sufficiently different that the resulting compact design is rather complex.
Pay close attention as we work through the constraints put in place to enforce `out` is the correct value. 


= Constraints
#render_constraint_table(chip, config, groups: "defs")

== Left shift
Left shifting, when `bit_shift != 0`.
#render_constraint_table(chip, config, groups: "intra_limb_left_shift")

== Right shift
Right shifting, when `bit_shift != 0`.
#render_constraint_table(chip, config, groups: "logical_right")

== Full-limb shifting
#render_constraint_table(chip, config, groups: "limb_shifting")

== Lookups
#render_constraint_table(chip, config, groups: "lookups")
