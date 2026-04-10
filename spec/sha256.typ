#import "/book.typ": book-page, aside, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_variable_table,
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

The following chips constitute an accelerator for the SHA256 compression function; 
other aspects of SHA256 hashing (such as repeated compression invocation, 
input padding and state initialization) fall outside the scope of this accelerator.

The base #sha256 chip provides the `ECALL` interface, interacts with memory and then delegates to the #sha256msgsched and #sha256round chips
to perform the message schedule and the compression rounds, respectively.
The `SHA256_M` interaction signature is used to represent the output of the message schedule.
The `SHA256_K` interaction signature is used to represent the `k` constants.
It could either be instantiated with a (short) precomputed table, or through hardcoded LogUp contributions in this chip.
For this exposition, we choose the former option, and present a table further below.
Additionally, we introduce a #rotxor chip to perform the common action of computing the XOR of three rotations (or shifts) of a word.

Most of the structure and variable naming follows the pseudocode of the wikipedia page#footnote(link("https://web.archive.org/web/20260320010021/https://en.wikipedia.org/wiki/SHA-2#Pseudocode")).

= #sha256 chip

== Columns
#let nr_variables = total_nr_variables(sha256chip)
#let nr_columns = total_nr_instantiated_columns(sha256chip, config)

The #sha256 chip leverages #nr_variables variables, spanning #nr_columns columns:
#render_chip_variable_table(sha256chip, config)

== Constraints

The first responsibility of the chip is to read the current state and message chunk from memory,
passed as arguments through pointers.
Since the memory ranges could overlap, we read the chunk first (in @sha256:c:read_chunk, at timestamp `timestamp`), before reading and writing the state (in @sha256:c:read_state, at timestamp `timestamp + 1`).
The addresses containing the state and the current chunk are passed in as arguments `A0 = x10` and `A1 = x11`, respectively.
Note that following the SHA256 spec, this state and the chunks are read and written as big-endian.
#render_constraint_table(sha256chip, config, groups: "memory")

Then we prepare the message schedule, by emitting the input chunk with multiplicities
corresponding to the number of times it will be read during a compression evaluation.
The #sha256msgsched chip itself is implicitly invoked by itself and #sha256round, setting the `amount`
column appropriately for the number of times the `w` value is required.
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
#render_chip_variable_table(sha256msgschedchip, config)

== Assumptions

#render_chip_assumptions(sha256msgschedchip, config)

== Constraints

First, we gather the dependencies from earlier in the message schedule.

#render_constraint_table(sha256msgschedchip, config, groups: "lookback")

Then, we calculate the result.
It suffices to check that the carry of adding four range-checked words
into a range-checked word is not too big, following the logic from @add.
In this case, using the `IS_BYTE` constraint allows us to add multiple words together
at the same time, without needing to store and range-check intermediate results.
#render_constraint_table(sha256msgschedchip, config, groups: "calc")

Finally, we contribute to the LogUp.
#render_constraint_table(sha256msgschedchip, config, groups: "output")

= #sha256round chip

== Columns

#let nr_variables = total_nr_variables(sha256roundchip)
#let nr_columns = total_nr_instantiated_columns(sha256roundchip, config)

The #sha256round chip leverages #nr_variables variables, spanning #nr_columns columns:
#render_chip_variable_table(sha256roundchip, config)

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

Then we constrain the addition for the new state, constraining additions with the same `IS_BYTE` trick as before.
#render_constraint_table(sha256roundchip, config, groups: "addition")

Finally, we chain the rounds together through the interactions.
#render_constraint_table(sha256roundchip, config, groups: "output")

== Padding

#render_chip_padding_table(sha256roundchip, config)

= #rotxor chip


This chip takes as input `a`, `r0`, `r1`, `r2` (4-bit values) and a bit `last_rot` to compute
$
  cases(
    (a >>> (16 + r_0)) xor (a >>> (16 + r_0 - r_1)) xor (a >>> r_2) quad "if" #`last_rot`,
    (a >>> (16 + r_0)) xor (a >>> (16 + r_0 - r_1)) xor (a >> r_2) quad "if" #`!last_rot`
  ),
$
where we let $>>>$ denote right rotation and $>>$ logical shift right.
We choose this representation so that all shift amounts required fit into 4 bits,
making the usage of `HWSL` more straightforward and avoid extra columns to represent more bits.

== Columns

#let nr_variables = total_nr_variables(rotxorchip)
#let nr_columns = total_nr_instantiated_columns(rotxorchip, config)
The #rotxor chip leverages #nr_variables variables, spanning #nr_columns columns:

#render_chip_variable_table(rotxorchip, config)

== Assumptions

Range checking for all elements is inherited from the bitwise lookups.
We can safely assume that no `r_i` will be zero, and avoid extra work due to right rotation needing `16 - shift` as arguments to the `HWSL` interactions.
#render_chip_assumptions(rotxorchip, config)

== Constraints

We first compute all rotations (or shifts) of `a`.
`a1` is computed as a left rotation of `a0`, in order to not need
additional columns to represent the full right-rotation amounts.
#render_constraint_table(rotxorchip, config, groups: "shift")

Then the bitwise XOR of the results.
#render_constraint_table(rotxorchip, config, groups: "xor")

And finally contribute to the lookup argument.
#render_constraint_table(rotxorchip, config, groups: "output")

== Padding

#render_chip_padding_table(rotxorchip, config)

= Constant lookup

#let sha256_kchip = load_chip("src/sha256consts.toml", config)
#let sha256_k = raw(sha256_kchip.name)

As mentioned, we provide the round constants through a short precomputed lookup table: #sha256_k.

#render_chip_variable_table(sha256_kchip, config)
#render_constraint_table(sha256_kchip, config)

= Notes/optimizations
- This could instead be designed following the #link("https://github.com/riscv/riscv-crypto")[RISC-V Crypto Scalar extension `Zknh`],
  for wider compatibility, but this design is likely to be more efficient.
  It is still possible, if desired, to expose #rotxor (or a selection of parameter instantiations thereof)
  as implementation for these primitives.
- The message schedule could be exposed as its own ECALL instead, but the direct integration leads to better efficiency.
- Some of these chips could be made narrower, at the cost of introducing some extra lookups and extra tables to compute and store intermediate results.
