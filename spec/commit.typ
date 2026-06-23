#import "/book.typ": book-page, aside
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

#show: book-page("commit.typ")

#let config = load_config()
#let chip = load_chip("src/commit.toml", config)
#let commit = raw(chip.name)

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #commit chip leverages #nr_variables variables, spanning #nr_columns columns and leverages #nr_interactions interactions:
#render_chip_variable_table(chip, config)

= Constraints
In this VM, committing is considered equivalent to writing a value to `stdout`.
Hence, this chip responds to `ECALL`s with system call number 64.
#footnote([RISC-V GNU-toolchain, `unistd.h`; version 2026-01-23, #link("https://github.com/riscv-collab/riscv-gnu-toolchain/blob/2026.01.23/linux-headers/include/asm-generic/unistd.h#L174")[[src]]])
Since we do not know how many bytes are to be committed, this chip employs a recursive design:
each iteration commits one byte, and recursively "calls" itself to commit the remaining bytes.
As such, only the call from the CPU to this chip (i.e., the `first` in the recursion tree) should accept the `ECALL`; later recursive calls should not.
This is why @commit:c:receive_ecall has multiplicity $-#`first`$.
#render_constraint_table(chip, config, groups: "incoming")

The `write` operation --- writing to a file descriptor --- has the following signature:
#footnote([Linux man-page on `write`; man7.org, version 6.16, 2025-10-29. #link("https://man7.org/linux/man-pages/man2/write.2.html")[[src]]])

```c
ssize_t write(size_t count; int fd, const void buf[count], size_t count);
```

That is to say,
- `A0` contains the file descriptor,
- `A1` contains the address of `buf`'s first byte, 
- `A2` contains `count`, and
- the written count should be written to `A0`.

@commit:c:read_address reads `address` from `x11` (=`A1`) and @commit:c:read_count reads `count` from `x12` (=`A2`).
Since we only support writing to `stdout` (which corresponds to $#`fd` = 1$
#footnote([The Open Group Standard for Information Technology --- Portable Operating System Interface (POSIX) Base Specifications, `unistd.h`; The Open Group, issue 8, #link("https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/unistd.h.html")[[src]]]))
we assert that `x10` contains $1$ in @commit:c:read_fd_write_count.
Note that this constraint _also_ writes `count` to `A0`; 
in this VM it is impossible for a commit to be interrupted or fail.
Lastly, the `index` is read from `x254`#footnote([In this VM, register 254 is reserved for containing the commitment index.]); in the same operation, $#`index` + #`count`$ is written back to this location by @commit:c:read_index.
This, too, leverages the fact that a commit will not be interrupted or fail to update the `index` for the next commit sequence.
Again, each of these memory interactions only take place when this is the `first` call in the recursion tree.

#render_constraint_table(chip, config, groups: "read_input")

*Note*: the observant reader will notice that @commit:c:read_index casts `count` to a `BaseField`, potentiallly losing information.
This is indeed correct.
However, since it is practically impossible to commit more than $2^64-2^32$ bytes in a single VM execution, it was decided to permit this.

Next, we read the `value` located at buffer address `address` and commit to it under the given `index`.
This is only performed when we have not yet reached the `end` of the commit sequence.
#render_constraint_table(chip, config, groups: "commit")

In parallel, we compute $#`address_incr` = #`address` + 1$ (@commit:c:address_incr) as address of the next byte to commit, and $#`count_decr` = #`count` - 1$ (@commit:c:count_decr) as the number of bytes that still has to be committed after committing this byte.
@commit:c:range_address_incr and @commit:c:range_count_decr are included to satisfy @add:a:sum respectively @sub:a:diff.
#render_constraint_table(chip, config, groups: "incr_decr")

When `count` hits $0$, we should stop performing further recursive calls.
We use the `end` bit to indicate these circumstances.

#render_constraint_table(chip, config, groups: "end")

*Note*: 
+ Rather than setting $#`end` = 1$ when $#`count` = 0$, we do so when $#`count_decr` = -1$.
  This technique allows `count` to be stored in a `DWordWL` rather than a `DWordHL`, saving two columns.
+ $forall i in [0, 3]: 65535 - #`count_decr`_i >= 0$ as a result of @commit:c:range_count_decr.
 Hence, 
  $
  sum_(i=0)^3 65535 - #`count_decr`_i = 0 arrow.l.r.double.long forall i in [0, 3]: #`count_decr`_i = 65535
  $

When this was not the `end` byte to commit in this recursion sequence, we recursively _Commit the Next Byte_ (`CNB`), specifying the timestamp, address to continue reading and the number of bytes that should still be committed (@commit:c:send_commit_next_byte).
Since that certainly won't be the `first` call in the sequence, we read `address_incr` and `count_decr` from the previous recursion level into `address` and `count` and continue executing the commit.
#render_constraint_table(chip, config, groups: "lookups")

Lastly, we must make sure `first`, `end` and `μ` are bits (@commit:c:range_first, @commit:c:range_end, @commit:c:range_mu), and that when either $#`first` = 1$ or $#`end` = 1$ imply that $#`μ` = 1$ (@commit:c:first_or_end_implies_mu).
These are required to ensure the multiplicities $-(#`μ` - #`first`)$ and $#`μ` - #`end`$ are binary.
#render_constraint_table(chip, config, groups: "bits")

= Padding
To pad this chip, use the below data.
#render_chip_padding_table(chip, config)

= Notes/optimizations
- The current version only supports writing to `stdout`.
  This chip could potentially be extended to support writing to arbitrary `fd`s
- One might be able to replace @commit:c:end by `end => count = 0`.
  While loosening the constraint (`count = 0 => end` is no longer enforced), this should not cause any problems:
  if the prover does not set `end` when `count=0`, they simply cannot complete the proof.
  First of all, one would have to recursively work through all $2^64$ values of `count`, something that is practically infeasible.
  Moreover, if this is done with a sequence that originally has $#`count` > 0$, one will inevitably have to read a memory address twice at the same timestamp, which is impossible to prove.
  In addition to dropping the `ZERO` lookup, this optimization might also permit moving `count_decr` from a `DWordHL` to a `DWordWL`, saving two columns.
- Given that it is practically infeasible to commit more than $#`p`-1 = 2^64-2^32$ bytes in a program, it might suffice to store `count_decr` in a `BaseField`.
  Note that this would probably involve having an extra (virtual) column storing `count` in `BaseField` form as well.
  Moreover, one might need to add a lookup to `LT` to ensure $#`count` <= #`p`-1$ when being read from memory at the beginning of each commitment sequence.
