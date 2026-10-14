# HALT Chip

## Variables

The  chip leverages  variable, spanning  columns and leverages  interactions:

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which to halt the program |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | The `next_pc` value the CPU wrote during the instruction HALT was invoked |

## Assumptions

It is assumed the input is range checked:

| Tag | Range | Description |
|-----|-------|-------------|
| `HALT-A1.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

## Constraints

The  chip: + makes sure register `x10` (containing the exit code) equals `0` ([halt:c:read_zero_exit_code]), + writes `0` to all other registers ([halt:c:zeroize_registers_lo]/[halt:c:zeroize_registers_hi]), and + sets `pc` equal to `1` ([halt:c:consume_pc], [halt:c:emit_pc]). Note that the writes performed by all these interactions --- except for the `pc` --- are accompanied by the timestamp `2^64-1`; the maximum timestamp. This prevents any other operation involving memory from being executed hereafter. The `pc` is consumed and re-emitted at the same timestamp to enable padding rows for the CPU. This means that the verifier will have to know the final timestamp at which a CPU padding `pc` was written to be able to balance the final LogUp.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `HALT-C1.i` | i ∈ [1, 9] | `MEMW[1, (2 * i)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C2` |  | `MEMW[0::BaseField[8]; 1, (2 * 10)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C3.i` | i ∈ [11, 31] | `MEMW[1, (2 * i)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C4.i` | i ∈ [0, 1] | `memory[1, [2 * 255 + i, 0], [timestamp[0] + 1, timestamp[1]], pc[i]]` | 1 |
| `HALT-C5.i` | i ∈ [0, 1] | `memory[1, [2 * 255 + i, 0], [timestamp[0] + 1, timestamp[1]], [1, 0][i]]` | -1 |

[ Observe that --- in its current state --- this solution puts the burden of verifying the register cleanup on the verifier inside of the lookup argument. Alternatively, one could add 31 lookups to the "memory" table to remove the _known_ final tokens for the registers there. ])

### Lookup

In this VM, halting is considered equivalent to executing a `sys_exit`. Hence, this chip responds to `ECALL`s with system call number 93.

The HALT chip therefore contributes the following interaction to the lookup-argument:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `HALT-C6` | `ECALL[timestamp, 93::DWordWL]` | -1 |

## Padding

This chip should only contain a single row. Given that `2^0 = 1`, this chip does not need to be padded. As such, no padding is defined.