#import "/meta.typ": aside
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

#let config = load_config()
#let chip = load_chip("src/copy.toml", config)
#let copy = raw(chip.name)

The #copy chip copies a range of bytes from one location to another.
This single chip serves the standard `C` memory functions `memcpy`, `memmove` and 
(a variation to#footnote[See @reading-parameters]) `memset`, as well as the `write` syscall.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #copy chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Assumptions
#render_chip_assumptions(chip, config)

= Constraints
The behaviour of this chip is slightly different, depending on the exact 
function it is expected to perform.
The only exception here are `memcpy` and `memmove`: the definition of `memmove`
subsumes that of `memcpy`, meaning that we can perform both operation in the same way. 
As such, this chip need only distinguish three states.
This is achieved by means of the `is_write` and `is_set` flags,
which are to be set according to the below table.
Importantly, the `is_write` and `is_set` flags are not to be set simultaneously,
as enforced by @copy:c:one_hot.

#figure(
  table(
    stroke: none,
    columns: (auto, auto, auto),
    align: (right, center, center),
    table.header(
      [], [`is_write`], [`is_set`],
    ),
    table.hline(),
    table.vline(x: 1),
    [`memmove`/`memcpy`], [0], [0],
    [`memset`], [0], [1],
    [`write`], [1], [0],
    [-], [1], [1],
  )
)

#render_constraint_table(chip, config, groups: "flags")

== Receiving `ECALL`

Each of the three chip states corresponds to a syscall number: $-30$ for `memmove`/`memcpy`, $-31$ for `memset`, and $64$ to `write`.#footnote([RISC-V GNU-toolchain, `unistd.h`; version 2026-01-23, #link("https://github.com/riscv-collab/riscv-gnu-toolchain/blob/2026.01.23/linux-headers/include/asm-generic/unistd.h#L174")[[src]]])
The chip is to accept `ECALL`s with all three of these numbers.
Rather than having a separate interactions for each, this chip leverages the 
prover-hinted `is_write` and `is_set` flags to extract the correct syscall number from the bus:

#render_constraint_table(chip, config, groups: "ecall")

Note that this step is performed exactly when the prover-hinted `first` flag is set,
ensuring this flag is only set on the first row of a copy sequence.

== Reading parameters <reading-parameters>

Each of the four operations accepts three input parameters, and produces one return value.
The `mem*` operations have the following interfaces:
```cpp
void* memcpy(  void* dest, const void* src, std::size_t count );
void* memmove( void* dest, const void* src, std::size_t count );		
void* memset(  void* dest,          int ch, std::size_t count );
```
To deal with only a single interface, this accelerator only provides support for repeating a selection of eight bytes over a desired length `count`, rather than supporting `memset` proper. 
The implementation of `memset(dest, ch, count)` can then consist of writing `ch`
for the first $min(8, #`count`)$ bytes, and invoking this ECALL if needed.

Having settled on the interface for the `mem*` operations, we turn our attention the `write` syscall, which has the following interface#footnote([Linux man-page on `write`; man7.org, version 6.16, 2025-10-29. #link("https://man7.org/linux/man-pages/man2/write.2.html")[[src]]]):
```c
ssize_t write(size_t count; int fd, const void buf[count], size_t count);
```
To invoke the `write` syscall, one should call `ECALL` with the appropriate syscall number after writing the file descriptor `fd` to `A0` (= `x10`), filling `A1` (= `x11`) with the address of `buf`'s first byte, and making `A2` (= `x12`) contain `count`.
Once the syscall returns, `A0` stores the number of bytes that were written to the `fd` during the syscall.

Given that this syscall already exists, its interface is fixed and should not be modified: the other operations have to be mapped onto it.
To this end, we require the parameters of the function to be stored in order in `A0`-`A2` before requesting the `ECALL`; this chip ensures the return value is stored in `A0` before handing control back to the `CPU`.

With all four operations now supplying a `src` and `count` in the same parameter position, reading them from `x11` and `x12` is easily achieved by means of @copy:c:read_src and @copy:c:read_count.
The destination address is found in `x10` for the `mem*` operations; for `write` it is not provided. 
To ensure `write` does not overwrite previously written values, register `x254` (initialized to 0),
is used to keep track of the number of bytes previously written.
The value in this register can thus act as the starting point for the `write`-sequence, 
as long as it is appropriately updated with each `write` syscall.
Hence, @copy:c:read_dst reads `dst` from either `x10` or `x254` depending on `is_write`,
and overwrites this with the `dst_res`, which is $#`dst` + #`count`$ for `write`-calls (@copy:c:dst_res_is_write)
and just `dst` otherwise (@copy:c:dst_res_non_write).

What remains is reading `fd` and returning `count` when $#`is_write`=1$,
which is enforced by @copy:c:read_fd.
Since we only support writing to `stdout` (which corresponds to $#`fd` = 1$ #footnote([The Open Group Standard for Information Technology --- Portable Operating System Interface (POSIX) Base Specifications, `unistd.h`; The Open Group, issue 8, #link("https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/unistd.h.html")[[src]]])),
this constraint enforces that a $1$ is read from `x10`.
Since this chip will always perfectly execute a `write`, this constraint moreover writes `count` as the return value to this register in the same interaction.

#render_constraint_table(chip, config, groups: "read_input")

Note: @copy:c:range_dst_res is included to satisfy assumptions made 
by the `ADD` template.


== Chaining

Rather than performing the full copy in a single row, this chip splits each operation
in "chunks", with each row in the table representing one such chunk.
Each row copies `step` bytes (which is either $1$ or $8$) from the source to
the destination.
Note that by this model, a copy operation on $x$ bytes can be represented using 
$floor.l x / 8 floor.r + x mod 8$ rows.
#footnote[
  The prover is free to select the `step` of each row; an $x$-byte copy could
  thus be spread over as many as $x$ rows.
  This freedom allows the prover to align the `MEMW` requests, such that they 
  can be handled by the `MEMW_A` chip, rather than the less efficient `MEMW` chip.
]

As long as $#`count` > #`step`$, the copy is not done yet, and an extra row
must be introduced.
As such, each row computes an updated `count`, as well as the updated `src` and
`dst` addresses for the bytes that remain to be copied, and forwards this
to the next row using the `COPY_NEXT` interaction @copy:c:send_next_chunk.
The next row then receives this information through @copy:c:receive_next_chunk,
instead of reading this information from the registers as is done on `first`-rows (@reading-parameters).

#render_constraint_table(chip, config, groups: "forward")

Note: constraints @copy:c:range_src_incr and @copy:c:range_dst_incr are included to satisfy assumptions made by the `ADD` template; @copy:c:range_count_decr satisfies the assumption made by the `SUB` template (which is inlined through @copy:c:count_borrows).

== Terminating the recursion

Observe from @copy:c:send_next_chunk that raising the prover-hinted flag `end`
stops the recursive behaviour.
Without constraining this flag, the prover has two possibilities to cheat: set the 
flag early, or set the flag late.
To see how this can be prevented, we first observe that there are two valid states 
that should end a sequence: $#`count` = #`step`$ and $(#`count`, #`first`) = (0,1)$.
#footnote[This second case is required to allow a zero-length copy.]
Note that by asserting @copy:c:borrow_implies_first and @copy:c:borrow_implies_count_is_zero, we can use `count_borrows[1]` as proxy to indicate the second state.
The statement $#`end` arrow.double.l.r #`count`=#`step` or (#`count` = 0 and #`first`)$ is then decomposed into
- $#`end` arrow.double.r #`count`=#`step` or (#`count` = 0 and #`first`)$
- $#`end` arrow.double.l #`count` = #`step`$
- $#`end` arrow.double.l #`count` = 0 and #`first`$
where the first implication is captured by constraints
@copy:c:end_implies_count_is_step_or_zc_lower and
@copy:c:end_implies_count_is_step_or_zc_upper, and
the third by @copy:c:borrow_implies_end.
Note that the second implication is already captured indirectly by @copy:c:borrow_implies_first: not setting the flag in this case will lead to a new row with $#`first` = 0$ being introduced (by @copy:c:send_next_chunk) which is guaranteed to have $#`count_borrows[1]` = 1$, thus violating this constraint.

#render_constraint_table(chip, config, groups: "restrict_end")

== Performing the move
Constraints @copy:c:read_value and @copy:c:write_value respectively read and write
`value` from/to memory.
Whether eight or just one byte is copied, is determined by the prover-hinted `single` flag.
When set, the top seven bytes in `value` are fixed at zero (@copy:c:single_lanes).

For most copy operations, the value is read at $#`timestamp`$ and written at $#`timestamp` + 1$.
The exception to this rule, are `memset` operations, performing the write before the read instead.
While this behaviour may seem counterintuitive, this inverse timing technique
ensures the same eight bytes are repeated across the requested memory domain.

One more requirement for a proper replication, is that $#`dst` = #`src` + 8$,
which is enforced by @copy:c:set_gap.

#render_constraint_table(chip, config, groups: "copy")


== Bits
Lastly, `Bit` variables must be bits, and both `first` and `end` must imply $#`μ` = 1$ to keep the multiplicities $-(#`μ` - #`first`)$ and $#`μ` - #`end`$ binary.
Note that $#`μ` - #`count_borrows[1]`$ is always binary, since $#`count_borrows[1]` => #`μ` = 1$ indirectly holds
via @copy:c:borrow_implies_first.
Additionally, we should make sure that `is_set` and `is_write` do not occur on padding rows, and that `first_write` is correctly defined.
#render_constraint_table(chip, config, groups: ("bits", "functionality"))

= Padding
To pad this chip, use the below data.
#render_chip_padding_table(chip, config)

= The Accelerated Memory Operations standard
This chip is part of the effort to support the Ethereum Foundation's Accelerated
Memory Operations standard.
This standard fixes what an accelerated `memcpy`, `memmove` and `memset` must provide.
#footnote([Accelerated Memory Operations; eth-act/zkevm-standards, commit `a97934c`, 2026-08-11. #link("https://github.com/eth-act/zkevm-standards/blob/a97934cae02693d3b69f07a7cd6e3ed6fb5e8053/standards/accelerated-memory-operations/README.md")[[src]]])
As requested by the standard, this chip accepts arbitrary operand alignment.

Note that this chip is only an accelerator; in practice, a library with functions
has to be made to resolve the differences in calling ABI between the
`memset`/`memcpy`/`memmove` functions and the syscalls this chip accelerates.
The `write` interface of this chip corresponds exactly to the syscall ABI,
making it compatible with any normal usage thereof (e.g. by a libc function).

Moreover, the standard's fourth operation, `memcmp`, is not covered by this chip, as it does not involve copying.
Its two remaining requirements fall outside this chapter: that the accelerated symbol behave identically to the C library function, which the guest stub is responsible for, and that it be a strong definition in an unconditionally linked object, which is a matter of linking.

= Notes/optimizations
- One could refactor the chip to move sixteen or thirty-two bytes per row instead of eight.
  It is suspected that this could be achieved using a 41c/18i and 57c/22i configuration, respectively.
- The chip could be modified to accept the default `memset` interface.
