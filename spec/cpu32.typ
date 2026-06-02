#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_variable_table,
  render_chip_padding_table,
  render_constraint_table,
  compute_nr_interactions,
  total_nr_instantiated_columns,
  total_nr_variables,
)

#let config = load_config()
#let chip = load_chip("src/cpu32.toml", config)

#show: book-page(chip.name)
#let cpu32 = raw(chip.name)

The #cpu32 chip is used to delegate the 32-bit instructions of the RV64I instruction set
from the main CPU table (@cpu).
All 32-bit instructions are ALU-only instructions, so the BRANCH, MEMORY and ECALL paths need no elaboration.
The timestamp and PC have already been read by the CPU table at this point, and need no further checking;
the PC for the next instruction will also already be handled by CPU.

The structure follows the regular ALU path, with some extra variables and constraints to contain the required sign extensions.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #cpu32 chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Assumptions

#render_chip_assumptions(chip, config)

Some of the assumptions can be checked with only arithmetic constraints, so we
provide these below.

#render_constraint_table(chip, config, groups: "assumptions")

= Constraints

Most constraints correspond to those already present in the CPU, and we present them here first,
including some updates to the range checking corresponding to the differing types.

#render_constraint_table(chip, config, groups: ("decode", "range", "alu", "mem", "logup"))

Then, we have the constraints corresponding to the sign-extension and definition of `arg1`, `arg2` and `rd`.
This includes a step where we extract the `signed` bit from the `alu_flags`, as this determines
whether to sign extend the inputs or not.

#render_constraint_table(chip, config, groups: "ext")

= Padding

The table can be padded with the following values:
#render_chip_padding_table(chip, config)

