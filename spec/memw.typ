#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  compute_nr_interactions,
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

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_memw_interactions = compute_nr_interactions(chip)

The #memw chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_memw_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Assumptions

#render_chip_assumptions(chip, config)

Some of the assumptions can be checked with only arithmetic constraints, so we
provide these below.

#render_constraint_table(chip, config, groups: "assumptions")

Our assumptions do not explicitly cover any range checks for the `value` column,
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

= Padding
The table can be padded to the next power of two with the following value assignments:
#render_chip_padding_table(chip, config)

= Read-size aligned fast path

#let alignedchip = load_chip("src/memw_aligned.toml", config)
#let aligned = raw(alignedchip.name)
#let nr_aligned_interactions = compute_nr_interactions(alignedchip)

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

The #aligned chip only needs #nr_variables variables, expressed through #nr_columns columns; it leverages #nr_aligned_interactions interactions.
#render_chip_variable_table(alignedchip, config)
#render_chip_assumptions(alignedchip, config)

Some of the assumptions can be checked with only arithmetic constraints, so we
provide these below.

#render_constraint_table(alignedchip, config, groups: "assumptions")

#render_constraint_table(alignedchip, config)

== Padding
The table can be padded to the next power of two with the following value assignments:
#render_chip_padding_table(alignedchip, config)

= Register fast-path

#let config = load_config()
#let register_chip = load_chip("src/memw_register.toml", config)
#let reg = raw(register_chip.name)

The #reg chip provides a fast-path for accessing registers.
This fast-path leverages that registers
+ can be addressed using a `Byte`, rather than a full `DWord`,
+ are constantly accessed, i.e., $#`timestamp` - #`old_timestamp`$ is small, and
+ have a fixed access pattern
to achieve a footprint that is significantly smaller than both #memw and #aligned.

Note: as a result of hard optimization, this chip can only be used for register accesses for which 
$#`timestamp` - #`old_timestamp` in [1, 2^16]$.
If this does not apply to your access, you should fall back to using `MEMW_A`.

Note moreover that this chip does not guard against misaligned register access faults: to access register with a given `address`, one must provide $2 dot #`address`$ in the lookup. 

== Variables
#let nr_variables = total_nr_variables(register_chip)
#let nr_columns = total_nr_instantiated_columns(register_chip, config)
#let nr_memw_r_interactions = compute_nr_interactions(register_chip)

The #reg chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_memw_r_interactions interactions:
#render_chip_variable_table(register_chip, config)

== Assumptions
The following range checks are assumed to be performed/enforced outside of this chip:
#render_chip_assumptions(register_chip, config)

== Constraints
Since most registers are frequently accessed, the difference between `timestamp` and `old_timestamp` is small most of the times.

Verifying that $#`timestamp` > #`old_timestamp`$ now simplifies to verifying that $#`timestamp` - #`old_timestamp` > 0$.
For most accesses, this value will be small enough to fit in a `Half`.
This chip thus enforces this by means of the following constraint:
#render_constraint_table(register_chip, config, groups: "diff")

With $#`old_timestamp`<#`timestamp`$ asserted, `old` is read from the register (@regw:c:read_old) and `val` is written back (@regw:c:write_val).
#render_constraint_table(register_chip, config, groups: "interactions")

This chip can either just write ($#`μ_write` = 1$), or both read and write ($#`μ_read` = 1$) in the same cycle.
It must be asserted that at most one of these two options is selected:
#render_constraint_table(register_chip, config, groups: "multiplicities")

Lastly, this chip contributes the following interactions to the logup:
#render_constraint_table(register_chip, config, groups: "output")

== Padding
The table can be padded to the next power of two with the following value assignments:
#render_chip_padding_table(register_chip, config)

= Notes/optimizations
The following ideas may prove to be optimizations for the #memw/#aligned/#reg chip:
- `MEMB` chip that does a one-byte write to remove old_timestamp from here (uncertain tradeoffs)
- Adding `μ_sum`/`w2`/`w4`/`write8` multiplicities to the `IS_HALF` lookups may make some GKR things faster if there are known zeroes.
- For the register fast-path, one may upgrade the `IS_HALF` check to an `IS_B20` check for extended range at the cost of looking through a larger table.
