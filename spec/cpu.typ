#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  compute_nr_interactions,
  render_constraint_table,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/cpu.toml", config)

#show: book-page(chip.name)
#let cpu = raw(chip.name)

The #cpu chip coordinates memory accesses and dispatches to other chips for arithmetic and logical operations.
It bases its decisions on the entry of the `DECODE` table (@decode) corresponding the the current program counter (PC).

= Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #cpu chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_column_table(chip, config)

= Assumptions
#render_chip_assumptions(chip, config)

= Constraints
First, we perform a decoding lookup for the current PC.

#render_constraint_table(chip, config, groups: "decode")

== Range checks

We constrain all columns to have the appropriate ranges.
The flags and register indices looked up from the decoding need to be checked,
as they are communicated through the interaction in a packed form.
In contrast, we know ahead of time that decoding will ensure proper range checks for `pc` and `imm`.
Similarly, since `next_pc` will propagate through the memory argument and be looked up
in the instruction decoding on the next cycle, it is forced to be in the correct range.#rj[is this true, do we need this elsewhere for chip assumptions?]
For the auxiliary columns, we need to check the limbs of `arg1`, `arg2`, and `res`.
The ranges of the other auxiliary columns are enforced through later constraints.
#rj[Make sure we argue for every column here]
#rj[is `rvd` still sufficiently constrained? (can also be done through the memory argument like `pc`?)]

#render_constraint_table(chip, config, groups: "range")

== ALU

The ALU functionality is then obtained through judicious dispatching to the corresponding chips.

#render_constraint_table(chip, config, groups: "alu")

== Memory

The interactions with the memory, both for register loading and storing, as for `LOAD` and `STORE` instructions are handled.
Note that since registers need no byte-addressing, we store them in the memory argument with `Word` limbs.
The timestamps are ensured to be disjoint for disjoint memory locations.
One consequence of that is that `next_pc` is written at `timestamp + 1`
to ensure the access is disjoint with the `pc` read into `rv1` as part of the `AUIPC` instruction.

#render_constraint_table(chip, config, groups: "mem")

== System

The interactions with the wider system.

#render_constraint_table(chip, config, groups: "sys")

== Input and output to the ALU

We constrain `arg1`, `arg2` and `rvd` to correspond to the wanted values,
including the appropriate sign/zero extension, depending on `word_instr`.

#render_constraint_table(chip, config, groups: "ext")

== Other constraints
For @cpu:c:is_equal, note that @cpu:c:sub sets `res` to be the difference between `arg1` and `arg2` whenever `BEQ` is $1$.
Given that this difference is $0$ when both are equal, @cpu:c:is_equal ensures `is_equal` is set to $1$ if and only if $#`arg1` = #`arg2`$ and `BEQ` is set.

#render_constraint_table(chip, config, groups: "misc")

#rj[Document the choice to not have a multiplicity column here for padding]

= Padding

The CPU can be padded with the following values, which have a corresponding row
in the DECODE table, at the _odd_ address 1, only reachable through a HALT ecall.

#render_chip_padding_table(chip, config)

This approach minimizes the number of dependent lookups, increasing only multiplicities in the DECODE table and the IS_BYTE lookup.
