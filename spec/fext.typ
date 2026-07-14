#import "/book.typ": book-page, aside
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

#show: book-page("fext.typ")

#let config = load_config()
#let loadchip = load_chip("src/fext_load.toml", config)
#let load = raw(loadchip.name)
#let fmachip = load_chip("src/fext_fma.toml", config)
#let fma = raw(fmachip.name)

We introduce a set of chips for faster processing of numbers mod the native goldilocks prime,
or a degree three extension field thereof.
Our approach is to off an arithmetic black box, consisting of the *TODO* chips,
that operates on a separate memory domain, and the #load chip to bridge the gap
from normal byte-addressed RAM memory to this separate field-storage.
As noted in @memory, we reserve the domain separator values $3$, $4$ and $5$ for field-storage.

= The #load chip
#let nr_variables = total_nr_variables(loadchip)
#let nr_columns = total_nr_instantiated_columns(loadchip, config)
#let nr_interactions = compute_nr_interactions(loadchip)

We use the #load chip to load the three composing coefficients from registers A1-A3 (in little-endian),
verify that all of them are in the correct range for a field element,
and then write them as field elements into field-storage.
We do this using #nr_variables variables spanning #nr_columns columns and #nr_interactions interactions.

== Variables

#render_chip_variable_table(loadchip, config)

== Constraints

#render_constraint_table(loadchip, config)

== Padding

#render_chip_padding_table(loadchip, config)

= The #fma chip
#let nr_variables = total_nr_variables(fmachip)
#let nr_columns = total_nr_instantiated_columns(fmachip, config)
#let nr_interactions = compute_nr_interactions(fmachip)

To compute a fused multiply-add (FMA) operation `output = a * b + c`, we first load
everything from memory, passing the input (ABB) addresses in A0-A2 and the output address in A3.
Then we constrain the extension field operation, and write back to (ABB) memory.

The extension field is expressed through three constant columns, $alpha$, $beta$ and $gamma$,
such that the defining polynomial is $X^3 - alpha X^2 - beta X - gamma = 0$, or alternatively,
$X^3 = alpha X^2 + beta X + gamma$.

== Variables

We express this chip using #nr_variables variables spanning #nr_columns columns and #nr_interactions interactions.

#render_chip_variable_table(fmachip, config)

== Constraints

#render_constraint_table(fmachip, config)

== Padding

#render_chip_padding_table(fmachip, config)
