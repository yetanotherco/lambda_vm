#import "/book.typ": book-page, aside
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

#show: book-page("ecall.typ")
= About `ECALL`
When `ECALL` is executed, it is assumed that:
- register `A7` contains the system call number
  #footnote(link("https://libriscv.no/docs/concepts/syscalls/#the-risc-v-system-call-abi")[The RISC-V system call ABI; libriscv.no. Accessed Feb 4, 2026.]),
- the arguments are located in registers `A0`-`A6`, and
- the return value is written to `A0`,
where `A0`-`A7` are symbolic names for the registers `x10`-`x17`
#footnote(link("https://en.wikipedia.org/wiki/RISC-V#Register_sets")[RISC-V - Register sets; en.wikipedia.org. Accessed on Feb 4, 2026.]).


#let config = load_config()
#let chip = load_chip("src/halt.toml", config)
#let halt = raw(chip.name)
= #halt chip

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The #halt chip leverages #nr_variables variable, spanning #nr_columns columns:
#render_chip_column_table(chip, config)

== Assumptions
It is assumed the input is range checked:
#render_chip_assumptions(chip, config)

== Constraints
The #halt chip:
+ makes sure register `x10` (containing the exit code) equals $0$ (@halt:c:read_zero_exit_code),
+ writes $0$ to all other registers (@halt:c:zeroize_registers_lo/@halt:c:zeroize_registers_hi), and
+ sets `pc` equal to $1$ (@halt:c:pc).
Note that the writes performed by all these interactions are accompanied by the timestamp $2^64-1$; the maximum timestamp.
This prevents any other operation involving memory from being executed hereafter.
#render_constraint_table(chip, config, groups: "all")

#aside("Note on register clean up",
[
  Observe that --- in its current state --- this solution puts the burden of verifying the register cleanup on the verifier inside of the lookup argument.
  Alternatively, one could add 31 lookups to the "memory" table to remove the _known_ final tokens for the registers there.
])

=== Lookup
The HALT chip contributes the following interaction to the lookup-argument:
#render_constraint_table(chip, config, groups: "lookup")

*Note*: #link("https://github.com/riscv-collab/riscv-gnu-toolchain/blob/master/linux-headers/include/asm-generic/unistd.h#L258")[$93$ is the system call number corresponding to `sys_exit`.]

== Padding
This chip should only contain a single row.
Given that $2^0 = 1$, this chip does not need to be padded.
As such, no padding is defined.


#let config = load_config()
#let chip = load_chip("src/commit.toml", config)
#let commit = raw(chip.name)
= #commit chip

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The #commit chip leverages #nr_variables variables, spanning #nr_columns columns:
#render_chip_column_table(chip, config)

== Constraints
In this VM, committing is considered equivalent to writing a value to `stdout`.
Hence, this chip responds to `ECALL`s with system call number 64.
#footnote([$64$ is the system call number corresponding to `sys_write`. #link("https://github.com/riscv-collab/riscv-gnu-toolchain/blob/master/linux-headers/include/asm-generic/unistd.h#L174")[[src]]])
Since we do not know how many bytes are to be committed, this chip employs a recursive design:
each iteration commits one byte, and recursively "call" itself to commit the remaining bytes.
As such, only the call from the CPU to this chip (i.e., the `first` in the recursion tree) should accept the `ECALL`; later recursive calls should not.
This is why @commit:c:receive_ecall has multiplicity $-#`first`$.
#render_constraint_table(chip, config, groups: "incoming")
*Note*: the prover is free to specify the value of `first` as they see fit; @commit:c:range_first only makes sure it must be a `Bit`.
Also, `first` being set must imply that that this is not a padding row (@commit:c:first_implies_mu).

The `write` operation --- writing to a file descriptor --- has the following signature:
#footnote([Linux man-page on `write`; man7.org, #link("https://man7.org/linux/man-pages/man2/write.2.html")[[src]]. Accessed Feb 4, 2026.])
#[
#show raw.where(block: true): it => block(it, fill: luma(230), inset: 1em, width: 100%, radius: 5pt)
```c
ssize_t write(size_t count; int fd, const void buf[count], size_t count);
```
]
That is to say,
- `A0` contains the file descriptor,
- `A1` contains the address of `buf`'s first byte, 
- `A2` contains `count`, and
- the written count should be written to `A0`.

Since we only support writing to `stdout` (which corresponds to $#`fd` = 1$
#footnote([The Open Group Base Specifications, `unistd.h`; The Open Group, issue 7, #link("https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/unistd.h.html")[[src]]. Accessed Feb 4, 2026.]))
we assert that `x10` contains $1$ in @commit:c:read_fd_write_count.
Note that this constraint _also_ writes `count` to `A0`; 
in this VM it is impossible for multiple for a commit to be interrupted or fail.
Furthermore, @commit:c:read_address reads `address` from `x11` and @commit:c:read_count reads `count` from `x12`.
Again, these memory interactions only take place when this is the `first` call in the recursion tree.
#render_constraint_table(chip, config, groups: "read_input")

Next, we read the `value` located at buffer address `address` and commit to it:
#render_constraint_table(chip, config, groups: "commit")

In parallel, we compute $#`address_incr` = #`address` + 1$ (@commit:c:address_incr) as address of the next byte to commit, and $#`count_decr` = #`count` - 1$ (@commit:c:count_decr) as the number of bytes that still has to be committed.
@commit:c:range_address_incr and @commit:c:range_count_decr are included to satisfy @add:a:sum respectively @add:a:rhs.
#render_constraint_table(chip, config, groups: "incr_decr")

When `count_decr` (the number of bytes still to be committed) hits $0$, we should stop recursing.
To this end, `last` is set this is the case (@commit:c:last).
To prevent undesired lookups from occurring, `last` should only be set when we're not padding (@commit:c:last_implies_mu).
Also, we must make sure `mu` is a bit (@commit:c:range_mu).
#render_constraint_table(chip, config, groups: "last")

Lastly, when this was not the `last` byte to commit in this recursion sequence, we recursively _Commit the Next Byte_ (`CNB`), specifying the timestamp, address to continue reading and the number of bytes that should still be committed (@commit:c:send_commit_next_byte).
Since that certainly won't be the `first` call in the sequence, we read `address_incr` and `count_decr` from the previous recursion level into `address` and `count` and continue executing the commit.
#render_constraint_table(chip, config, groups: "lookups")

== Padding
To pad this chip, use the below data.
This corresponds to committing a $0$ from address $0$.
#render_chip_padding_table(chip, config)
