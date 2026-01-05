#import "/book.typ": book-page, et
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

#let shift = raw(chip.name)

#show: book-page.with(title: "SHIFT chip")

= #shift chip

== Interface
The #shift chip has the following interface:
#block(radius: 5pt, width: 100%, inset: 1.5em, fill: luma(240), 
```
// param in: the value being shifted
// param shift: the number of bits to shift `in` by
// param direction: whether to shift left (0) or right (1) 
// param signed: whether to interpret `in` as a signed (1) or unsigned (0) integer
// param word_instr: whether to execute the SLL/SR* (0) or SLLW/SR*W (1) instruction
// out shifted: the resulting value
SHIFT[shifted: DWord; in: DWord, shift: Byte, direction: Bit, signed: Bit, word_instr: Bit]
```
)
In other words, the #shift chip is designed to constrain that 
$ 
#`shifted` := cases(
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

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `SHIFT` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

== Assumptions
#render_chip_assumptions(chip, config)

== Constraints
First, we constraint `bit_shift` based on whether we are left or right-shifting.
#render_constraint_table(chip, config, groups: "bit_shift")

Next, we apply shift the limbs of `in` left and right, storing them in `X` and `Y` respectively.
When `right`-shifting and `bit_shift = 0`, the output is incorrect.
As such, we override `Y[i] := in[i]` and `X[i] := 0`.

The case of `left`-shifting and `bit_shift = 0` will be used for padding rows.
To prevent unnecessary lookups in padding rows, we also override `X[i] := in[i]` and `Y[i] := 0` in this case.
#render_constraint_table(chip, config, groups: "intra_limb_shift")

=== Full-limb shifting
Lastly, `X` and `Y` are combined in the right way to form the limbs of `output`.
#render_constraint_table(chip, config, groups: "limb_shifting")

=== Miscellaneous 
To make sure `left` is actually a `Bit`, we introduce constraint @shift:c:direction_implies_mu. 
Moreover, @shift:c:is_negative_if_signed is included to compute if `in` is negative.
Since `in` cannot be negative in the unsigned case, @shift:c:is_negative_if_unsigned constrains that `is_negative` will be $0$ in that case.
#render_constraint_table(chip, config, groups: "left_flag")
#render_constraint_table(chip, config, groups: "is_negative")

=== Lookups
This chip adds the following interaction to the lookup.
#render_constraint_table(chip, config, groups: "lookups")
