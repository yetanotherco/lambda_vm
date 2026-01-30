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

The `MEMW` chip is comprised of  variables that are expressed using  columns:

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `MEMW-A1.i` | i ∈ [0, 1] | `IS_WORD[base_address[i]]` |
| `MEMW-A2` |  | `IS_BIT<write2>` |
| `MEMW-A3` |  | `IS_BIT<write4>` |
| `MEMW-A4` |  | `IS_BIT<write8>` |
| `MEMW-A5` |  | `IS_BIT<write2 + write4 + write8>` |
| `MEMW-A6.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

Our assumptions do not explicitly cover any range checks for the `is_register` and `value` columns, as these are not necessary for the correctness of this chip in isolation. These properties are necessary for the consistency of the system as a whole, and therefore we document it here, keeping the type information as a reading help.

## Constraints

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW-C1` |  | `IS_BIT<μ_sum>` |  |
| `MEMW-C2` |  | `w2` => `μ_sum` |  |
| | | _polynomial:_ `w2 * (1 - μ_sum) = 0` | |
| `MEMW-C3` |  | `ADD<address_add[0]::DWordWL; base_address, 1>` | w2 |
| `MEMW-C4.i` | i ∈ [1, 2] | `ADD<address_add[i]::DWordWL; base_address, i + 1>` | w4 |
| `MEMW-C5.i` | i ∈ [3, 6] | `ADD<address_add[i]::DWordWL; base_address, i + 1>` | write8 |
| `MEMW-C6.i` | i ∈ [0, 6], j ∈ [0, 3] | `IS_HALFWORD[address_add[i][j]]` |  |
| `MEMW-C7` |  | `LT[1; old_timestamp[0], timestamp, 0]` | μ_sum |
| `MEMW-C8` |  | `LT[1; old_timestamp[1], timestamp, 0]` | w2 |
| `MEMW-C9.i` | i ∈ [2, 3] | `LT[1; old_timestamp[i], timestamp, 0]` | w4 |
| `MEMW-C10.i` | i ∈ [4, 7] | `LT[1; old_timestamp[i], timestamp, 0]` | write8 |

As long as `timestamp` is properly range-checked, the presence of `old_timestamp` in the memory argument automatically ensures appropriate range checking (as long as no external entities provide negative multiplicities without range checking the timestamp). This ensures the assumptions for `LT` are satisfied.

We additionally check that the address does not overflow for more significant bytes of the access.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW-CR11` | `LT[1; base_address, address_add[0]::DWordWL, 0]` | write2 |
| `MEMW-CR12` | `LT[1; base_address, address_add[2]::DWordWL, 0]` | write4 |
| `MEMW-CR13` | `LT[1; base_address, address_add[6]::DWordWL, 0]` | write8 |

The chip adds the following tuples to the lookup argument, to effectuate that part of the memory argument.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW-CM14` |  | `memory[is_register, base_address, old_timestamp[0], old[0]]` | μ_sum |
| `MEMW-CM15` |  | `memory[is_register, base_address, timestamp, value[0]]` | -μ_sum |
| `MEMW-CM16` |  | `memory[is_register, address_add[0], old_timestamp[1], old[1]]` | w2 |
| `MEMW-CM17` |  | `memory[is_register, address_add[0], timestamp, value[1]]` | -w2 |
| `MEMW-CM18.i` | i ∈ [2, 3] | `memory[is_register, address_add[i - 1], old_timestamp[i], old[i]]` | w4 |
| `MEMW-CM19.i` | i ∈ [2, 3] | `memory[is_register, address_add[i - 1], timestamp, value[i]]` | -w4 |
| `MEMW-CM20.i` | i ∈ [4, 7] | `memory[is_register, address_add[i - 1], old_timestamp[i], old[i]]` | write8 |
| `MEMW-CM21.i` | i ∈ [4, 7] | `memory[is_register, address_add[i - 1], timestamp, value[i]]` | -write8 |

This chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW-CO22` | `MEMW[old; is_register, base_address, value, timestamp, write2, write4, write8]` | μ_read |
| `MEMW-CO23` | `MEMW[is_register, base_address, value, timestamp, write2, write4, write8]` | μ_write |

## Future optimization ideas

- Fast path for aligned memory access where all bytes have the same old timestamp - MEMB chip that deals does a one-byte write to remove old_timestamp from here (uncertain tradeoffs) - Compute `base_address[1] + 1` once and have high words of `address_add` as Words - Improve overflow trapping somehow so we don't need `LT` (could tie into previous one by checking carry bit of the +1) - Adding `μ_sum`/`w2`/`w4`/`write8` multiplicities to the `IS_HALFWORD` lookups may make some GKR things faster if there are known zeroes.