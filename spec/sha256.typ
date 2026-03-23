#import "/book.typ": book-page, aside, rj
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

#show: book-page("sha256.typ")

#let sha256chip = load_chip("src/sha256.toml", config)
#let sha256msgschedchip = load_chip("src/sha256msgsched.toml", config)
#let sha256roundchip = load_chip("src/sha256round.toml", config)
#let rotxorchip = load_chip("src/rotxor.toml", config)
#let sha256 = raw(sha256chip.name)
#let sha256msgsched = raw(sha256msgschedchip.name)
#let sha256round = raw(sha256roundchip.name)
#let rotxor = raw(rotxorchip.name)


The base #sha256 chip provides the `ECALL` interface, interacts with memory and then delegates to the #sha256msgsched and #sha256round chips
to perform the message schedule and the compression rounds, respectively.
The `SHA2_M` interaction signature is used to represent the output of the message schedule.
The `SHA2_K` interaction signature is used to represent the `k` constants.
It could either be instantiated with a (short) precomputed table, or through hardcoded LogUp contributions in this chip.
Additionally, we introduce a #rotxor chip that takes as input `a`, `r0`, `r1`, `r2` (pre-split into high bit and low nibble) and a bit `last_rot` and computes
$
  cases(
    (a >>> r_0) xor (a >>> r_1) xor (a >>> r_2) quad "if" #`last_rot`,
    (a >>> r_0) xor (a >>> r_1) xor (a >> r_2) quad "if" #`!last_rot`
  ),
$
where we let $>>>$ denote right rotation and $>>$ logical shift right.

= #sha256 chip

== Columns
#let nr_variables = total_nr_variables(sha256chip)
#let nr_columns = total_nr_instantiated_columns(sha256chip, config)

The #sha256 chip leverages #nr_variables variables, spanning #nr_columns columns:
#render_chip_column_table(sha256chip, config)

== Constraints

The first responsibility of the chip is to read the current state and message chunk from memory,
passed as arguments through pointers.
Since the memory ranges could overlap, we read the chunk first, before reading and writing the state at the next timestamp.
The state is passed in argument `A0 = x10`, and the chunk as `A1 = x11`.
Note that following the SHA256 spec, this state and the chunks are read and written as big-endian.
#render_constraint_table(sha256chip, config, groups: "memory")

Then we prepare the message schedule, by emitting the input chunk and then invoking the message schedule for every remaining index.
We additionally provide a constant multiplicity that takes into account the number of times this word will be read
by the message schedule and the compression rounds combined.
#render_constraint_table(sha256chip, config, groups: "sched")

And finally, we provide the boundaries for the #sha256round chip and the
final addition of the compression to the old state.
Observe that we embed the addition into the upper 32 bits of a double word,
in order to satisfy and use the `ADD` chip.
#render_constraint_table(sha256chip, config, groups: "compress")

In this VM, we assign syscall number -1 to the #sha256 accelerator.
The chip therefore contributes the following interaction to the lookup-argument:
#render_constraint_table(sha256chip, config, groups: "lookup")

== Padding

#render_chip_padding_table(sha256chip, config)

= #sha256msgsched chip

== Columns

#let nr_variables = total_nr_variables(sha256msgschedchip)
#let nr_columns = total_nr_instantiated_columns(sha256msgschedchip, config)

The #sha256msgsched chip leverages #nr_variables variables, spanning #nr_columns columns:
#render_chip_column_table(sha256msgschedchip, config)

== Assumptions

#render_chip_assumptions(sha256msgschedchip, config)

== Constraints

First, we gather the dependencies from earlier in the message schedule.

#render_constraint_table(sha256msgschedchip, config, groups: "lookback")

Then, we calculate the result.
It suffices to check that the carry of adding four range-checked words
into a range-checked word is not too big, following the logic from @add.
#render_constraint_table(sha256msgschedchip, config, groups: "calc")

Finally, we contribute to the LogUp.
#render_constraint_table(sha256msgschedchip, config, groups: "output")

= #sha256round chip

== Columns

#let nr_variables = total_nr_variables(sha256roundchip)
#let nr_columns = total_nr_instantiated_columns(sha256roundchip, config)

The #sha256round chip leverages #nr_variables variables, spanning #nr_columns columns:
#render_chip_column_table(sha256roundchip, config)

== Assumptions

#render_chip_assumptions(sha256roundchip, config)

== Constraints

First, we compute the necessary intermediate values.
#let bitand = math.class("binary", math.amp)
To compute `maj`, observe that $ (a bitand b) xor (a bitand c) xor (b bitand c) = (a bitand b) xor (c bitand (a xor b)), $
by distribution.
Additionally, since for this form, $(a bitand b)$ and $(a xor b)$ are disjoint, so are $(a bitand b)$ and $(c bitand (a xor b))$,
and hence we can replace that top-level XOR with a field addition to compute $(a bitand b) + (c bitand (a xor b))$,
needing fewer intermediate columns.
Similarly, `ch` can be written as $(e bitand f) + ((2^32 - 1 - e) bitand g)$.
#render_constraint_table(sha256roundchip, config, groups: "value")

Then we constrain the addition for the new state.
Since `out_e` is the range-checked sum of three range-checked words, we
can constrain `carry_e` to be 0, 1 or 2 with a degree 3 constraint instead of a lookup.
#render_constraint_table(sha256roundchip, config, groups: "addition")

Finally, we chain the rounds together through the interactions.
#render_constraint_table(sha256roundchip, config, groups: "output")

== Padding

#render_chip_padding_table(sha256roundchip, config)

= #rotxor chip

Since all uses of the chip can be reordered to have `r2 < 16`, we can leave out the high bit of `r2` from the chip.

== Columns

#let nr_variables = total_nr_variables(rotxorchip)
#let nr_columns = total_nr_instantiated_columns(rotxorchip, config)
The #rotxor chip leverages #nr_variables variables, spanning #nr_columns columns:

#render_chip_column_table(rotxorchip, config)

== Assumptions

Range checking for all elements is inherited from the bitwise lookups.
We can safely assume that no `ri_low` will be zero, and avoid extra work due to right rotation needing `16 - shift` as arguments to the `HWSL` interactions.
#render_chip_assumptions(rotxorchip, config)

== Constraints

We first compute all rotations (or shifts) of `a`.
#render_constraint_table(rotxorchip, config, groups: "shift")

Then the bitwise XOR of the results.
#render_constraint_table(rotxorchip, config, groups: "xor")

And finally contribute to the lookup argument.
#render_constraint_table(rotxorchip, config, groups: "output")

== Padding

#render_chip_padding_table(rotxorchip, config)

= Notes/optimizations
- This could instead be designed following the #link("https://github.com/riscv/riscv-crypto")[RISC-V Crypto Scalar extension `Zknh`],
  for wider compatibility, but this design is likely to be more efficient.
- The message schedule could be exposed as its own ECALL instead, but the direct integration leads to better efficiency.
- Some of these chips could be made narrower, at the cost of introducing some extra lookups and extra tables to compute and store intermediate results.
