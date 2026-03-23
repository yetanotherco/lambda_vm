#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_padding_table
)

#let config = load_config()
#let chip = load_chip("src/memw.toml", config)

#show: book-page(chip.name)

#let memw = raw(chip.name)

The #memw chip is used to read and write memory locations (both RAM and registers)
in chunks of 1, 2, 4 or 8 values.
It introduces the old value and last-accessed timestamps of memory addresses internally,
in order to satisfy the design of the memory argument (@memory).

= Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `MEMW` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

= Assumptions

#render_chip_assumptions(chip, config)

Our assumptions do not explicitly cover any range checks for the `is_register` and `value` columns,
as these are not necessary for the correctness of this chip in isolation.
Still, these properties are necessary for the consistency of the system as a whole, and therefore
we document it here, keeping the type information as a reading help.

= Constraints

Depending on the values of `write2`, `write4` and `write8`, the addresses following `base_address` need to be constructed.
Rather than computing these in full (which would require the later addresses to be instantiated), 
it suffices to know the `carry`: the bit indicating whether $#`base_address`_0 + t >= 2^32$, i.e., whether adding $t in [1, 7]$ to `base_address` requires a carry from the lower to the upper limb.
Note that it is safe for the prover to chose these bits: additions for which this bit is not correctly set
will yield an address where either the lower or upper limb is out of bounds.
As such, the constructed address will not match any existing memory tokens, 
which are only initialized for correctly formatted and range-checked doublewords (see @memory).

#render_constraint_table(chip, config, groups: "consistency")

As long as `timestamp` is properly range-checked, the presence of `old_timestamp`
in the memory argument automatically ensures it is appropriately range checked
(this assumes no external entities provide negative multiplicities without range checking the timestamp).
This ensures the assumptions for `LT` are satisfied.

There is no need to check that the additions do not overflow,
as our address calculations are not performed modulo $2^64$ here,
and any overflow will result in an address without matching initialization.

The chip adds the following tuples to the lookup argument,
to effectuate that part of the memory argument.
#render_constraint_table(chip, config, groups: "memory")

This chip contributes the following to the lookup argument:
#render_constraint_table(chip, config, groups: "output")

= Read-size aligned fast path

#let alignedchip = load_chip("src/memw_aligned.toml", config)
#let aligned = raw(alignedchip.name)

When a memory access happens at an address with proper alignment for its access size
(i.e., adding the access size to `base_address`'s lowest limb does not overflow), 
and all accessed elements were last accessed at the same timestamp, we can 
instead use the #aligned chip to save on total column count.
The saving comes from only requiring a single old timestamp to be stored,
as well as being able to guarantee that all values of `add_limb_overflow` would be zero.
A minor extra cost is introduced in the form of a check that the alignment is indeed correct,
and the corresponding decomposition of the `base_address`.

Further logic remains essentially the same, so we briefly present the relevant tables for this chip.
#let nr_variables = total_nr_variables(alignedchip)
#let nr_columns = total_nr_instantiated_columns(alignedchip, config)

The #aligned chip only needs #nr_variables variables, expressed through #nr_columns columns.
#render_chip_column_table(alignedchip, config)
#render_chip_assumptions(alignedchip, config)
#render_constraint_table(alignedchip, config)

= Register fast-path

#let config = load_config()
#let register_chip = load_chip("src/memw_register.toml", config)
#let reg = raw(register_chip.name)

Given that i) there are significantly fewer registers than memory addresses, and ii) registers are accessed far more frequently, a fast-path is devised.

== Columns
#let nr_variables = total_nr_variables(register_chip)
#let nr_columns = total_nr_instantiated_columns(register_chip, config)

The #reg chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(register_chip, config)

== Assumptions
The following range checks are assumed to be performed/enforced outside of this chip:
#render_chip_assumptions(register_chip, config)

== Constraints
One of the primary tests this chip performs, is verify that $#`old_timestamp`<#`timestamp`$.
This is achieved by ensuring that adding $#`timestamp_diff` := #`old_timestamp` - #`timestamp`$ to `timestamp` overflows.
This is asserted by means of the following constraints:
#render_constraint_table(register_chip, config, groups: "sub")

With $#`old_timestamp`<#`timestamp`$ asserted, `old` is read from the register (@regw:c:read_old) and `val` is written back (@regw:c:write_val).
#render_constraint_table(register_chip, config, groups: "interactions")

This chip can either just write ($#`μ_write` = 1$), or read&written in the same cycle ($#`μ_read` = 1$).
However, it must be asserted that at most one of these two options is selected:
#render_constraint_table(register_chip, config, groups: "multiplicities")

Lastly, this chip contributes the following interactions to the logup:
#render_constraint_table(register_chip, config, groups: "output")

== Padding
The table can be padded to the next power of two with the following value assignments:

#render_chip_padding_table(register_chip, config)

== Notes
- Given that most accesses are to "hot" register (i.e., registers that are accessed often), `timestamp_diff` will rarely be large.
  This might allow `timestamp_diff` to be reduced to `Half` in a fast-path version of this chip.
  It could even be that `old_timestamp[1]` can be dropped. Things would then be computed as
  - `carry[0] = (old_timestamp[0] + timestamp_diff - timestamp[0]) / 2^32`
  - `IS_HALF[timestamp_diff-1]` to ensure diff is at least 1
  - `old_timestamp[1] = timestamp[1] - carry[0]`
  - `carry[1] = 0` to ensure old_ts < ts
  (-4 col)
- Most register accesses both read&write (and not just read). The fast-path chip could assume this to always be the case, allowing the two multiplicities to be merged. (-1 col)

= Future optimization ideas

- `MEMB` chip that does a one-byte write to remove old_timestamp from here (uncertain tradeoffs)
- Additional fast path for registers? (Always guaranteed same timestamp, alignment could be an assumption, always only two values)
- Adding `μ_sum`/`w2`/`w4`/`write8` multiplicities to the `IS_HALF` lookups may make some GKR things faster if there are known zeroes.
