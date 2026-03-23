#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_assumptions,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/regw.toml", config)

#show: book-page(chip.name)

#let reg = raw(chip.name)

The #reg chip is used to read and write register locations.
It introduces the `old` value and last-accessed timestamps of the registers internally, in order to satisfy the design of the memory argument (@memory).

= Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The #reg chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

#let stackrel(top, bottom) = {
 $mat(delim: #none, top; bottom)$
}

= Assumptions
The following range checks are assumed to be performed/enforced outside of this chip:
#render_chip_assumptions(chip, config)

= Constraints
One of the primary tests this chip performs, is verify that $#`old_timestamp`<#`timestamp`$.
This is achieved by ensuring that adding $#`timestamp_diff` := #`old_timestamp` - #`timestamp`$ to `timestamp` overflows.
This is asserted by means of the following constraints:
#render_constraint_table(chip, config, groups: "sub")

With $#`old_timestamp`<#`timestamp`$ asserted, `old` is read from the register (@regw:c:read_old) and `val` is written back (@regw:c:write_val).
#render_constraint_table(chip, config, groups: "interactions")

This chip can either just write ($#`μ_write` = 1$), or read&written in the same cycle ($#`μ_read` = 1$).
However, it must be asserted that at most one of these two options is selected:
#render_constraint_table(chip, config, groups: "multiplicities")

Lastly, this chip contributes the following interactions to the logup:
#render_constraint_table(chip, config, groups: "output")

= Padding
The table can be padded to the next power of two with the following value assignments:

#render_chip_padding_table(chip, config)

= Notes
- Given that most accesses are to "hot" register (i.e., registers that are accessed often), `timestamp_diff` will rarely be large.
  This might allow `timestamp_diff` to be reduced to `Half` in a fast-path version of this chip.
  It could even be that `old_timestamp[1]` can be dropped. Things would then be computed as
  - `carry[0] = (old_timestamp[0] + timestamp_diff - timestamp[0]) / 2^32`
  - `IS_HALF[timestamp_diff-1]` to ensure diff is at least 1
  - `old_timestamp[1] = timestamp[1] - carry[0]`
  - `carry[1] = 0` to ensure old_ts < ts
  (-4 col)
- Most register accesses both read&write (and not just read). The fast-path chip could assume this to always be the case, allowing the two multiplicities to be merged. (-1 col)