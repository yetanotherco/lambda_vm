# MEMW Chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `is_register` | `Bit` | Whether the address represents a register index |
| `base_address` | `DWordWL` | The base address to read/write from/to, gets offset by $[0, 7]$, depending on how big the access is |
| `value` | `BaseField[8]` | The values to store in memory. For regular memory, these should be (up to) 8 range-checked `Byte`s; registers are stored as two range-checked `Word`s |
| `timestamp` | `DWordWL` | The timestamp at which this memory access is said to occur |
| `write2` | `Bit` | Whether to write exactly 2 values |
| `write4` | `Bit` | Whether to write exactly 4 values |
| `write8` | `Bit` | Whether to write exactly 8 values |

### Output

| Name | Type | Description |
|------|------|-------------|
| `old` | `BaseField[8]` | The old value written at `base_address`. See `value` for information about representation. Only the elements corresponding to the `writeN` bits are guaranteed |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `address_add` | `DWordHL[7]` | `address_add[i] = base_address + i + 1` |
| `old_timestamp` | `DWordWL[8]` | The timestamp at which the address was last accessed |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `w2` | `Bit` | writing at least 2 bytes |
| `w4` | `Bit` | writing at least 4 bytes |
| `μ_sum` | `Bit` |  |

**Definition of `w2`:**
```
w2 := write2 + write4 + write8
```

**Definition of `w4`:**
```
w4 := write4 + write8
```

**Definition of `μ_sum`:**
```
μ_sum := μ_read + μ_write
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ_read` | `Bit` | Whether we are performing a read (and hence return `out`) |
| `μ_write` | `Bit` | Whether we are performing a write (and hence not return `out`) |

## Assumptions

| Ref | Range | Description |
|-----|-------|-------------|
| `A1` | i ∈ [0, 1] | `IS_WORD[base_address[i]]` |
| `A2` |  | `IS_BIT<write2>` |
| `A3` |  | `IS_BIT<write4>` |
| `A4` |  | `IS_BIT<write8>` |
| `A5` |  | `IS_BIT<write2 + write4 + write8>` |
| `A6` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

## Constraints

### consistency

| Ref | Kind | Range | Description | Multiplicity |
|-----|------|-------|-------------|--------------|
| `1` | template |  | `IS_BIT<μ_sum>` |  |
| `2` | arith |  | `w2` => `μ_sum` |  |
| | | _polynomial:_ `w2 * (1 - μ_sum) = 0` | |
| `3` | template |  | `ADD<address_add[0]::DWordWL; base_address, 1>` | w2 |
| `4` | template | i ∈ [1, 2] | `ADD<address_add[i]::DWordWL; base_address, i + 1>` | w4 |
| `5` | template | i ∈ [3, 6] | `ADD<address_add[i]::DWordWL; base_address, i + 1>` | write8 |
| `6` | interaction | i ∈ [0, 6], j ∈ [0, 3] | `IS_HALFWORD[address_add[i][j]]` |  |
| `7` | interaction |  | `LT[1; old_timestamp[0], timestamp]` | μ_sum |
| `8` | interaction |  | `LT[1; old_timestamp[1], timestamp]` | w2 |
| `9` | interaction | i ∈ [2, 3] | `LT[1; old_timestamp[i], timestamp]` | w4 |
| `10` | interaction | i ∈ [4, 7] | `LT[1; old_timestamp[i], timestamp]` | write8 |

### overflow

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `R1` | interaction | `LT[1; base_address, address_add[0]::DWordWL]` | write2 |
| `R2` | interaction | `LT[1; base_address, address_add[2]::DWordWL]` | write4 |
| `R3` | interaction | `LT[1; base_address, address_add[6]::DWordWL]` | write8 |

### memory

| Ref | Kind | Range | Description | Multiplicity |
|-----|------|-------|-------------|--------------|
| `M1` | interaction |  | `memory[is_register, base_address, old_timestamp[0], old[0]]` | μ_sum |
| `M2` | interaction |  | `memory[is_register, base_address, timestamp, value[0]]` | -μ_sum |
| `M3` | interaction |  | `memory[is_register, address_add[0], old_timestamp[1], old[1]]` | w2 |
| `M4` | interaction |  | `memory[is_register, address_add[0], timestamp, value[1]]` | -w2 |
| `M5` | interaction | i ∈ [2, 3] | `memory[is_register, address_add[i - 1], old_timestamp[i], old[i]]` | w4 |
| `M6` | interaction | i ∈ [2, 3] | `memory[is_register, address_add[i - 1], timestamp, value[i]]` | -w4 |
| `M7` | interaction | i ∈ [4, 7] | `memory[is_register, address_add[i - 1], old_timestamp[i], old[i]]` | write8 |
| `M8` | interaction | i ∈ [4, 7] | `memory[is_register, address_add[i - 1], timestamp, value[i]]` | -write8 |

### output

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `O1` | interaction | `MEMW[old; is_register, base_address, value, timestamp, write2, write4, write8]` | μ_read |
| `O2` | interaction | `MEMW[is_register, base_address, value, timestamp, write2, write4, write8]` | μ_write |
