#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
)

#let config = load_config()
#let chip = load_chip("src/memw.toml", config)

#show: book-page.with(title: "MEMW chip")

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `MEMW` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

== Assumptions

#render_chip_assumptions(chip, config)

Our assumptions do not explicitly cover any range checks for the `is_register` and `value` columns,
as these are not necessary for the correctness of this chip in isolation.
These properties are necessary for the consistency of the system as a whole, and therefore
we document it here, keeping the type information as a reading help.

== Constraints

#render_constraint_table(chip, config, groups: "consistency")

As long as `timestamp` is properly range-checked, the presence of `old_timestamp`
in the memory argument automatically ensures appropriate range checking
(as long as no external entities provide negative multiplicities without range checking the timestamp).
This ensures the assumptions for `LT` are satisfied.

We additionally check that the address does not overflow
for more significant bytes of the access.
#render_constraint_table(chip, config, groups: "overflow")

The chip adds the following tuples to the lookup argument,
to effectuate that part of the memory argument.
#render_constraint_table(chip, config, groups: "memory")

This chip contributes the following to the lookup argument.
#render_constraint_table(chip, config, groups: "output")


== Future optimization ideas

- Fast path for aligned memory access where all bytes have the same old timestamp
- MEMB chip that deals does a one-byte write to remove old_timestamp from here (uncertain tradeoffs)
- Compute `base_address[1] + 1` once and have high words of `address_add` as Words
- Improve overflow trapping somehow so we don't need `LT` (could tie into previous one by checking carry bit of the +1)
- Adding `μ_sum`/`w2`/`w4`/`write8` multiplicities to the `IS_HALFWORD` lookups may make some GKR things faster if there are known zeroes.
