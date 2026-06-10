#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_variable_table,
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
It bases its decisions on the entry of the `DECODE` table (@decode) corresponding the current program counter (PC).

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #cpu chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Assumptions
#render_chip_assumptions(chip, config)

Additionally, the following constraints can be used to provide defense-in-depth
validation of the assumptions.

#render_constraint_table(chip, config, groups: "assumptions")

= Constraints

First, we perform a decoding lookup for the current PC.
Instructions having the `word_instr` flag set are not decoded here, as they are delegated to the `CPU32` chip.
In that case, we ensure that the current row of the CPU cannot have any other observable effects.

#render_constraint_table(chip, config, groups: "decode")

== Range checks

We constrain all columns to have the appropriate ranges.
All values in `packed_decode` need to be checked to ensure
the packing is correct for the interaction.
In contrast, we know ahead of time that decoding will ensure proper range checks for `pc` and `imm`.
Similarly, since `next_pc` will propagate through the memory argument and be looked up
in the instruction decoding on the next cycle, it is forced to be in the correct range;
the final value for `next_pc` is similarly fixed by the memory finalization.
For the auxiliary columns, we need to check the limbs of `res`, since
`rv1` and `rv2` are enforced by the memory argument, and `rvd` is correct by the correctness of the dependent chips.
The ranges of the other auxiliary columns are enforced through later constraints.

#render_constraint_table(chip, config, groups: "range")

== ALU

The ALU functionality is then obtained through delegation to the `ALU` signature, backed by the various ALU chips,
or by using the appropriate template.
For the pure ALU path, `arg2` is computed as `rv2 + imm`, which relies on @cpu:a:arg2-multiplex to
be either `rv2` or `imm`, depending on the instruction.
The other contributions for `arg2` are specific to the (mutually exclusive, @cpu:a:mem-branch-mutex)
`MEMORY` and `BRANCH` flags:
- For the `MEMORY` path, we want the output of the ALU to be $#`rv1` + #`imm`$, as that is the
  address at which the memory access occurs.
- For the `BRANCH` path, we want the ALU output to reflect the branch condition (or just be inactive for JALR).

#render_constraint_table(chip, config, groups: "alu")

== Memory<cpu:memory>

Note that since registers need no byte-addressing, we store them in the memory argument with `Word` limbs,
simultaneously ensuring that register reads are properly range checked as long as all writes are.
The `pc` register behaves very predictably with respect to its timestamps and when it is being read,
so for performance reasons, we inline its memory interactions directly into the #cpu chip.

Potentially overlapping memory accesses are ensured to have disjoint timestamps.
One consequence of that is that `next_pc` is written at `timestamp + 1`
to ensure the access is disjoint with the `pc` read into `rv1` as part of the `AUIPC` instruction (see @cpu:c:read_rv1 and @decode:decoding-overview).
Constraints regarding whether `pc_double_read` corresponds to an `AUIPC` instruction are not necessary,
as regardless of its value, the old timestamp is guaranteed smaller than the new timestamp,
and the integrity of the memory argument therefore ensures the correctness of this bit.

The memory interaction itself is handled by the `MEMORY` signature,
which will read the `mem_flags` argument to perform either a `LOAD` or a `STORE`.
We refer to the previous section's description of `arg2` for how
the address is computed.

The value to (potentially) be written back to `rd` is stored in `rvd`,
which can either come from the ALU --- in case of an ALU operation or a JALR branch ---
or from the MEMORY interaction.

#render_constraint_table(chip, config, groups: "mem")

== Branching

A branch is expressed by having the `BRANCH` flag set to 1.
Since `BRANCH` and `MEMORY` are mutually exclusive (@cpu:a:mem-branch-mutex),
we can repurpose the `mem_flags` field to indicate a JALR instruction.
When JALR is not set, we have a conditional branch that is decided upon by the result of the ALU instructions,
as set in the `res` variable.
As such, we can set `branch_cond` appropriately as multiplicity flag for the `BRANCH` chip.

#render_constraint_table(chip, config, groups: "branch")

== System

The interactions with the wider system go through the `ECALL` interface.
Since we treat `EBREAK` instructions as unprovable traps, we avoid emitting `DECODE` rows
for these, and do not need any further handling in the CPU.

#render_constraint_table(chip, config, groups: "sys")


= Padding

The CPU can be padded with the following values, which have a corresponding row
in the DECODE table, at the _odd_ address 1, only reachable through a HALT ecall.

#render_chip_padding_table(chip, config)

This approach minimizes the number of dependent lookups, increasing only multiplicities in the `DECODE` table and the `IS_BYTE` and `IS_HALF` lookups.
