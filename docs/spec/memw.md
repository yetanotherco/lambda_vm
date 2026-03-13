# MEMW Chip

The  chip is used to read and write memory locations (both RAM and registers) in chunks of 1, 2, 4 or 8 values. It introduces the old value and last-accessed timestamps of memory addresses internally, in order to satisfy the design of the memory argument ([memory]).

= Columns

The `MEMW` chip is comprised of  variables that are expressed using  columns:

= Assumptions

Our assumptions do not explicitly cover any range checks for the `is_register` and `value` columns, as these are not necessary for the correctness of this chip in isolation. These properties are necessary for the consistency of the system as a whole, and therefore we document it here, keeping the type information as a reading help.

= Constraints

We can compute the addresses for the later bytes based on a single bit each, indicating whether adding `i` to `base_address` overflows the lower limb. We can safely assume that additions for which this bit is not correctly set will have either an overflow on the upper or lower word, and hence not match any existing memory tokens, which are only initialized for correctly formatted and range-checked doublewords (see [memory]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW-C1` |  | `IS_BIT<μ_sum>` |  |
| `MEMW-C2` |  | `w2` => `μ_sum` |  |
| | | _polynomial:_ `w2 * (1 - μ_sum) = 0` | |
| `MEMW-C3.i` | i ∈ [0, 6] | `IS_BIT<add_limb_overflow[i]>` |  |
| `MEMW-C4` |  | `LT[1; old_timestamp[0], timestamp, 0]` | μ_sum |
| `MEMW-C5` |  | `LT[1; old_timestamp[1], timestamp, 0]` | w2 |
| `MEMW-C6.i` | i ∈ [2, 3] | `LT[1; old_timestamp[i], timestamp, 0]` | w4 |
| `MEMW-C7.i` | i ∈ [4, 7] | `LT[1; old_timestamp[i], timestamp, 0]` | write8 |

As long as `timestamp` is properly range-checked, the presence of `old_timestamp` in the memory argument automatically ensures appropriate range checking (as long as no external entities provide negative multiplicities without range checking the timestamp). This ensures the assumptions for `LT` are satisfied.

There is no need to check that the address does not overflow, as our address calculations are not performed modulo `2^64` here, and any overflow will result in an address without matching initialization.

The chip adds the following tuples to the lookup argument, to effectuate that part of the memory argument.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW-CM8` |  | `memory[is_register, base_address, old_timestamp[0], old[0]]` | μ_sum |
| `MEMW-CM9` |  | `memory[is_register, base_address, timestamp, value[0]]` | -μ_sum |
| `MEMW-CM10` |  | `memory[is_register, address_add[0], old_timestamp[1], old[1]]` | w2 |
| `MEMW-CM11` |  | `memory[is_register, address_add[0], timestamp, value[1]]` | -w2 |
| `MEMW-CM12.i` | i ∈ [2, 3] | `memory[is_register, address_add[i - 1], old_timestamp[i], old[i]]` | w4 |
| `MEMW-CM13.i` | i ∈ [2, 3] | `memory[is_register, address_add[i - 1], timestamp, value[i]]` | -w4 |
| `MEMW-CM14.i` | i ∈ [4, 7] | `memory[is_register, address_add[i - 1], old_timestamp[i], old[i]]` | write8 |
| `MEMW-CM15.i` | i ∈ [4, 7] | `memory[is_register, address_add[i - 1], timestamp, value[i]]` | -write8 |

This chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW-CO16` | `MEMW[old; is_register, base_address, value, timestamp, write2, write4, write8]` | μ_read |
| `MEMW-CO17` | `MEMW[is_register, base_address, value, timestamp, write2, write4, write8]` | μ_write |

= Read-size aligned fast path

When a memory access happens at an address with proper alignment (that is, enough trailing zeros) for its access size, and all accessed elements were last accessed at the same timestamp, we can instead use the  chip to save on total column count. The saving comes from only requiring a single old timestamp to be stored, as well as being able to guarantee that all values of `add_limb_overflow` would be zero. A minor extra cost is introduced in the form of a check that the alignment is indeed correct, and the corresponding decomposition of the `base_address`.

Further logic remains essentially the same, so we briefly present the relevant tables for this chip.

The  chip only needs  variables, expressed through  columns.

= Future optimization ideas

- `MEMB` chip that does a one-byte write to remove old_timestamp from here (uncertain tradeoffs) - Additional fast path for registers? (Always guaranteed same timestamp, alignment could be an assumption, always only two values) - Adding `μ_sum`/`w2`/`w4`/`write8` multiplicities to the `IS_HALF` lookups may make some GKR things faster if there are known zeroes.

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
| `add_limb_overflow` | `Bit[7]` | Whether adding `i` to `base_address[0]` as a field element exceeds $2^32$ |
| `old_timestamp` | `DWordWL[8]` | The timestamp at which the address was last accessed |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `w2` | `Bit` | writing at least 2 bytes |
| `w4` | `Bit` | writing at least 4 bytes |
| `address_add` | `DWordWL[7]` | `address_add[i] = base_address + i + 1` |
| `μ_sum` | `Bit` |  |

**Definition of `w2`:**
```
w2 := write2 + write4 + write8
```

**Definition of `w4`:**
```
w4 := write4 + write8
```

**Definition of `address_add`:**
```
address_add := ['arr', ['+', ['idx', 'base_address', 0], 'i', 1, ['*', ['-', ['^', 2, 32]], ['idx', 'add_limb_overflow', 'i']]], ['+', ['idx', 'base_address', 1], ['idx', 'add_limb_overflow', 'i']]]
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

| Tag | Range | Description |
|-----|-------|-------------|
| `MEMW-A1.i` | i ∈ [0, 1] | `IS_WORD[base_address[i]]` |
| `MEMW-A2` |  | `IS_BIT<write2>` |
| `MEMW-A3` |  | `IS_BIT<write4>` |
| `MEMW-A4` |  | `IS_BIT<write8>` |
| `MEMW-A5` |  | `IS_BIT<write2 + write4 + write8>` |
| `MEMW-A6.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |