# ECALL Chips

=  chip

## Columns

The  chip leverages  variable, spanning  columns:

## Assumptions

It is assumed the input is range checked:

## Constraints

The  chip: + makes sure register `x10` (containing the exit code) equals `0` ([halt:c:read_zero_exit_code]), + writes `0` to all other registers ([halt:c:zeroize_registers_lo]/[halt:c:zeroize_registers_hi]), and + sets `pc` equal to `1` ([halt:c:pc]). Note that the writes performed by all these interactions are accompanied by the timestamp `2^64-1`; the maximum timestamp. This prevents any other operation involving memory from being executed hereafter.

[ Observe that --- in its current state --- this solution puts the burden of verifying the register cleanup on the verifier inside of the lookup argument. Alternatively, one could add 31 lookups to the "memory" table to remove the _known_ final tokens for the registers there. ])

### Lookup

The HALT chip contributes the following interaction to the lookup-argument:

*Note*: [`93` is the system call number corresponding to `sys_exit`.]

## Padding

This chip should only contain a single row. Given that `2^0 = 1`, this chip does not need to be padded. As such, no padding is defined.