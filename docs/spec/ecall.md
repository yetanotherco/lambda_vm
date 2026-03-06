# ECALL Chips

ECALLs provide system-level functionalities to the guest program.

When `ECALL` is executed, it is assumed that: - register `A7` contains the system call number

- the arguments are located in registers `A0`-`A6`, and - the return value is written to `A0`, where `A0`-`A7` are symbolic names for the registers `x10`-`x17`

=  chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which to halt the program |

The  chip leverages  variable, spanning  columns:

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `HALT-A1.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

It is assumed the input is range checked:

## Constraints

The  chip: + makes sure register `x10` (containing the exit code) equals `0` ([halt:c:read_zero_exit_code]), + writes `0` to all other registers ([halt:c:zeroize_registers_lo]/[halt:c:zeroize_registers_hi]), and + sets `pc` equal to `1` ([halt:c:pc]). Note that the writes performed by all these interactions are accompanied by the timestamp `2^64-1`; the maximum timestamp. This prevents any other operation involving memory from being executed hereafter.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `HALT-C1.i` | i ∈ [1, 9] | `MEMW[1, (2 * i)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C2` |  | `MEMW[0::BaseField[8]; 1, (2 * 10)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C3.i` | i ∈ [11, 31] | `MEMW[1, (2 * i)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C4` |  | `MEMW[1, (2 * 255)::DWordWL, ['arr', 1, 0, 0, 0, 0, 0, 0, 0], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |

[ Observe that --- in its current state --- this solution puts the burden of verifying the register cleanup on the verifier inside of the lookup argument. Alternatively, one could add 31 lookups to the "memory" table to remove the _known_ final tokens for the registers there. ])

### Lookup

In this VM, halting is considered equivalent to executing a `sys_exit`. Hence, this chip responds to `ECALL`s with system call number 93.

The HALT chip therefore contributes the following interaction to the lookup-argument:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `HALT-C5` | `ECALL[timestamp, 93::DWordWL]` | -1 |

## Padding

This chip should only contain a single row. Given that `2^0 = 1`, this chip does not need to be padded. As such, no padding is defined.

=  chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which to commit |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `index` | `BaseField` | Index of value being committed. |
| `address` | `DWordWL` | Address of first byte to commit. |
| `address_incr` | `DWordHL` | $`address` + 1$ |
| `count` | `DWordWL` | number of bytes to commit |
| `count_decr` | `DWordHL` | $`count` - 1$ |
| `first` | `Bit` | Whether this is the first commitment in this sequence. |
| `end` | `Bit` | Whether this is the end of the commitment sequence. |
| `value` | `Byte` | Byte stored at `address`. |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

The  chip leverages  variables, spanning  columns:

## Constraints

In this VM, committing is considered equivalent to writing a value to `stdout`. Hence, this chip responds to `ECALL`s with system call number 64.

Since we do not know how many bytes are to be committed, this chip employs a recursive design: each iteration commits one byte, and recursively "calls" itself to commit the remaining bytes. As such, only the call from the CPU to this chip (i.e., the `first` in the recursion tree) should accept the `ECALL`; later recursive calls should not. This is why [commit:c:receive_ecall] has multiplicity `-`first``.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `COMMIT-C1` | `ECALL[timestamp, 64::DWordWL]` | -first |

The `write` operation --- writing to a file descriptor --- has the following signature:

```c ssize_t write(size_t count; int fd, const void buf[count], size_t count); ```

That is to say, - `A0` contains the file descriptor, - `A1` contains the address of `buf`'s first byte, - `A2` contains `count`, and - the written count should be written to `A0`.

[commit:c:read_address] reads `address` from `x11` (=`A1`) and [commit:c:read_count] reads `count` from `x12` (=`A2`). Since we only support writing to `stdout` (which corresponds to ``fd` = 1`

we assert that `x10` contains `1` in [commit:c:read_fd_write_count]. Note that this constraint _also_ writes `count` to `A0`; in this VM it is impossible for a commit to be interrupted or fail. Lastly, the `index` is read from `x254`; in the same operation, ``index` + `count`` is written back to this location by [commit:c:read_index]. This, too, leverages the fact that a commit will not be interrupted or fail to update the `index` for the next commit sequence. Again, each of these memory interactions only take place when this is the `first` call in the recursion tree.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `COMMIT-C2` | `MEMW[['arr', ['idx', 'address', 0], ['idx', 'address', 1], 0, 0, 0, 0, 0, 0]; 1, (2 * 11)::DWordWL, ['arr', ['idx', 'address', 0], ['idx', 'address', 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | first |
| `COMMIT-C3` | `MEMW[['arr', ['idx', 'count', 0], ['idx', 'count', 1], 0, 0, 0, 0, 0, 0]; 1, (2 * 12)::DWordWL, ['arr', ['idx', 'count', 0], ['idx', 'count', 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | first |
| `COMMIT-C4` | `MEMW[['arr', 1, 0, 0, 0, 0, 0, 0, 0]; 1, (2 * 10)::DWordWL, ['arr', ['idx', 'count', 0], ['idx', 'count', 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | first |
| `COMMIT-C5` | `MEMW[['arr', 'index', 0, 0, 0, 0, 0, 0, 0]; 1, (2 * 254)::DWordWL, ['arr', ['+', 'index', ['cast', 'count', 'BaseField']], 0, 0, 0, 0, 0, 0, 0], timestamp, 0, 0, 0]` | first |

*Note*: the observant reader will notice that [commit:c:read_index] casts `count` to a `BaseField`, potentiallly losing information. This is indeed correct. However, since it is practically impossible to commit more than `2^64-2^32` bytes in a single VM execution, it was decided to permit this.

Next, we read the `value` located at buffer address `address` and commit to it under the given `index`. This is only performed when we have not yet reached the `end` of the commit sequence.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `COMMIT-C6` | `MEMW[['arr', 'value', 0, 0, 0, 0, 0, 0, 0]; 0, address, ['arr', 'value', 0, 0, 0, 0, 0, 0, 0], timestamp, 0, 0, 0]` | μ - end |
| `COMMIT-C7` | `COMMIT[index, value]` | μ - end |

In parallel, we compute ``address_incr` = `address` + 1` ([commit:c:address_incr]) as address of the next byte to commit, and ``count_decr` = `count` - 1` ([commit:c:count_decr]) as the number of bytes that still has to be committed after committing this byte. [commit:c:range_address_incr] and [commit:c:range_count_decr] are included to satisfy [add:a:sum] respectively [add:a:rhs].

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `COMMIT-C8` |  | `ADD<address_incr::DWordWL; address, 1::DWordWL>` |  |
| `COMMIT-C9.i` | i ∈ [0, 3] | `IS_HALF[address_incr[i]]` | μ |
| `COMMIT-C10` |  | `SUB<count_decr::DWordWL; count, 1::DWordWL>` |  |
| `COMMIT-C11.i` | i ∈ [0, 3] | `IS_HALF[count_decr[i]]` | μ |

When `count` hits `0`, we should stop performing further recursive calls. We use the `end` bit to indicate these circumstances.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `COMMIT-C12` | `ZERO[end; (65535 - count_decr[0]) + (65535 - count_decr[1]) + (65535 - count_decr[2]) + (65535 - count_decr[3])]` | μ |

*Note*: + Rather than setting ``end` = 1` when ``count` = 0`, we do so when ``count_decr` = -1`. This technique allows `count` to be stored in a `DWordWL` rather than a `DWordHL`, saving two columns. + `forall i in [0, 3]: 65535 - `count_decr`_i >= 0` as a result of [commit:c:range_count_decr]. Hence, $ sum_(i=0)^3 65535 - `count_decr`_i = 0 arrow.l.r.double.long forall i in [0, 3]: `count_decr`_i = 65535 $

When this was not the `end` byte to commit in this recursion sequence, we recursively _Commit the Next Byte_ (`CNB`), specifying the timestamp, address to continue reading and the number of bytes that should still be committed ([commit:c:send_commit_next_byte]). Since that certainly won't be the `first` call in the sequence, we read `address_incr` and `count_decr` from the previous recursion level into `address` and `count` and continue executing the commit.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `COMMIT-C13` | `CNB[timestamp, index + 1, address_incr::DWordWL, count_decr::DWordWL]` | μ - end |
| `COMMIT-C14` | `CNB[timestamp, index, address, count]` | -(μ - first) |

Lastly, we must make sure `first`, `end` and `μ` are bits ([commit:c:range_first], [commit:c:range_end], [commit:c:range_mu]), and that when either ``first` = 1` or ``end` = 1` imply that ``μ` = 1` ([commit:c:first_or_end_implies_mu]). These are required to ensure the multiplicities `-(`μ` - `first`)` and ``μ` - `end`` are binary.

| Tag | Description |
|-----|-------------|
| `COMMIT-C15` | `IS_BIT<first>` |
| `COMMIT-C16` | `IS_BIT<end>` |
| `COMMIT-C17` | `IS_BIT<μ>` |
| `COMMIT-C18` | `first` + `end` => `μ` = 1 |
| | _polynomial:_ `(first + end) * (1 - μ) = 0` |

## Padding

To pad this chip, use the below data.

## Notes/optimizations

- The current version only supports writing to `stdout`. This chip could potentially be extended to support writing to arbitrary `fd`s - One might be able to replace [commit:c:end] by `end => count = 0`. While loosening the constraint (`count = 0 => end` is no longer enforced), this should not cause any problems: if the prover does not set `end` when `count=0`, they simply cannot complete the proof. First of all, one would have to recursively work through all `2^64` values of `count`, something that is practically infeasible. Moreover, if this is done with a sequence that originally has ``count` > 0`, one will inevitably have to read a memory address twice at the same timestamp, which is impossible to prove. In addition to dropping the `ZERO` lookup, this optimization might also permit moving `count_decr` from a `DWordHL` to a `DWordWL`, saving two columns. - Given that it is practically infeasible to commit more than ``p`-1 = 2^64-2^32` bytes in a program, it might suffice to store `count_decr` in a `BaseField`. Note that this would probably involve having an extra (virtual) column storing `count` in `BaseField` form as well. Moreover, one might need to add a lookup to `LT` to ensure ``count` <= `p`-1` when being read from memory at the beginning of each commitment sequence.