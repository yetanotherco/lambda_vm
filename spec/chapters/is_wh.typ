#import "/meta.typ": aside
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  compute_nr_interactions,
  render_constraint_table,
  render_chip_assumptions,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/is_wh.toml", config)
#let is_wh = raw(chip.name)

Several chips load/write vast amounts of data from/to memory as part of their execution.
This quickly involves many memory interactions, as each interaction operates on
at most eight bytes at a time.
For each of these interactions, the address of the memory section being accessed
has to be stored on the chip in some way.
Given that many of these loads/writes operate on contiguous memory, these address 
values typically only differ in their lowest limb, but exceptions exist.
As a result of these exceptions, the values of the upper limbs still need to 
be range checked for every single address.

This #is_wh is introduced to reduce the range-checking pressure on these chips.
Rather than range checking all 64 bits locally, chips now only need to check the
bottom `Half` limb, and defer the check for the top 48 bits to this chip.
With several range checks for the same 48 bits coming in, the overhead incurred
on this chip is limited.
As a bonus, the calling chips can now often represent an address with a
`DWordWHH` rather than a `DWordHL`.
Hence, why the input of this chip are `Word`-`Half` pairs rather than `Half` triples.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #is_wh chip leverages #nr_variables variables, spanning #nr_columns columns and leverages #nr_interactions interactions:
#render_chip_variable_table(chip, config)

= Constraints
#render_constraint_table(chip, config, groups: "all")

= Padding

The table can be padded to the next power of two with the following value assignments:

#render_chip_padding_table(chip, config)

