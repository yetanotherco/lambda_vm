# HALT Chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which to halt the program |

## Assumptions

| Ref | Range | Description |
|-----|-------|-------------|
| `A1` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

## Constraints

### all

| Ref | Kind | Range | Description | Multiplicity |
|-----|------|-------|-------------|--------------|
| `halt:c:zeroize_registers_lo` | interaction | i ∈ [1, 9] | `MEMW[1, 2 * i, 0, 2^64 - 1, 1, 0, 0]` | 1 |
| `halt:c:read_zero_exit_code` | interaction |  | `MEMW[1, 2 * 10, 0, 2^64 - 1, 1, 0, 0]` | 1 |
| `halt:c:zeroize_registers_hi` | interaction | i ∈ [11, 31] | `MEMW[1, 2 * i, 0, 2^64 - 1, 1, 0, 0]` | 1 |
| `halt:c:pc` | interaction |  | `MEMW[1, 2 * 255, 1, 2^64 - 1, 1, 0, 0]` | 1 |

### lookup

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `halt:c:lookup` | interaction | `ECALL[timestamp, 93]` | -1 |
