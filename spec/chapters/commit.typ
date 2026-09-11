#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  compute_nr_interactions,
  render_constraint_table,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/commit.toml", config)
#let commit = raw(chip.name)

The #commit chip handles the `write` system call: it accepts the call, checks the file descriptor, advances the commitment index and hands the bytes themselves to `MEMMOVE` (@memmove).
It is one row per system call; the loop over the buffer lives in the other chip.

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
This, too, leverages the fact that a commit will not be interrupted or fail to update the `index` for the next commitment sequence.
#render_constraint_table(chip, config, groups: "read_input")

*Note*: the observant reader will notice that @commit:c:read_index casts `count` to a `BaseField`, potentially losing information.
This is indeed correct.
However, since it is practically impossible to commit more than $2^64-2^32$ bytes in a single VM execution, it was decided to permit this.

The bytes themselves are copied by `MEMMOVE`, from `address` in RAM to the commitment domain starting at `index`.
@commit:c:defer_to_memmove is the whole of that hand-off: this chip states where the buffer is, where it lands and how long it is, and the other chip walks it eight bytes at a time and emits the commitment tuples.
It is the only sender on that bus, which is what lets `MEMMOVE` decode the commitment functionality --- and with it the destination domain --- from the mere fact that it received the tuple.
#render_constraint_table(chip, config, groups: "defer")

Note that this chip therefore does not itself constrain the committed values, nor even see them, and that `index` is the only place where the two chips have to agree on more than the buffer: the verifier reconstructs the commitment side of the memory argument out of the committed output, so it is `index` that has to line up with the position of these bytes in that output.
@commit:c:read_index is what makes it so, by advancing `x254` by exactly `count`.

Lastly, we must make sure `μ` is a bit.
#render_constraint_table(chip, config, groups: "bits")

= Padding
To pad this chip, use the below data.
#render_chip_padding_table(chip, config)

Since every constraint in this chip is conditioned on `μ`, a padding row is all-zero.

= Notes/optimizations
- The current version only supports writing to `stdout`.
  This chip could potentially be extended to support writing to arbitrary `fd`s.
- Nothing here bounds `count`, and neither does `MEMMOVE` on this path, so one `write` appends rows to that chip in proportion to its length.
  That is a bound on prover cost only --- the commitment bus balances against the committed output, which the verifier knows in full --- but a `LT` lookup here, paired with chunking in the guest stub, would make the cost of a single system call bounded like every other one.
