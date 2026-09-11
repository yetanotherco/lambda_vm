# MEMW Chip

The  chip is used to read and write memory locations (both RAM and registers) in chunks of 1, 2, 4 or 8 values. It introduces the old value and last-accessed timestamps of memory addresses internally, in order to satisfy the design of the memory argument ([memory]).

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `is_register` | `Bit` | Whether the address represents a register index |
| `base_address` | `DWordWL` | The base address to read from/write to. Gets offset by $[0, 7]$ depending on the size of the access |
| `value` | `BaseField[8]` | The values to store in memory. For RAM, these should be (up to) 8 range-checked `Byte`s; registers are stored as two range-checked `Word`s |
| `timestamp` | `DWordWL` | The timestamp at which this memory access occurs |
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
| `carry` | `Bit[7]` | Whether `base_address[0] + i + 1` $>= 2^32$ |
| `old_timestamp` | `DWordWL[8]` | The timestamp at which address `base_address + i` was last accessed |

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
address_add := [base_address[0] + i + 1 - 2^32 * carry[i], base_address[1] + carry[i]]
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

Some of the assumptions can be checked with only arithmetic constraints, so we provide these below.

| Tag | Description |
|-----|-------------|
| `MEMW-C1` | `IS_BIT<write2>` |
| `MEMW-C2` | `IS_BIT<write4>` |
| `MEMW-C3` | `IS_BIT<write8>` |
| `MEMW-C4` | `IS_BIT<write2 + write4 + write8>` |

Our assumptions do not explicitly cover any range checks for the `is_register` and `value` columns, as these are not necessary for the correctness of this chip in isolation. Still, these properties are necessary for the consistency of the system as a whole, and therefore we document it here, keeping the type information as a reading help.

## Constraints

Depending on the values of `write2`, `write4` and `write8`, the addresses following `base_address` need to be constructed. Rather than computing these in full (which would require the later addresses to be instantiated), it suffices to know the `carry`: the bit indicating whether ``base_address`_0 + t >= 2^32`, i.e., whether adding `t in [1, 7]` to `base_address` requires a carry from the lower to the upper limb. Note that it is safe for the prover to chose these bits: additions for which this bit is not correctly set will yield an address where either the lower or upper limb is out of bounds. As such, the constructed address will not match any existing memory tokens, which are only initialized for correctly formatted and range-checked doublewords (see [memory]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW-C5` |  | `IS_BIT<μ_read>` |  |
| `MEMW-C6` |  | `IS_BIT<μ_write>` |  |
| `MEMW-C7` |  | `IS_BIT<μ_sum>` |  |
| `MEMW-C8` |  | `w2` => `μ_sum` |  |
| | | _polynomial:_ `w2 * (1 - μ_sum) = 0` | |
| `MEMW-C9.i` | i ∈ [0, 6] | `IS_BIT<carry[i]>` |  |
| `MEMW-C10` |  | `ALU[[1, 0]; old_timestamp[0], timestamp, ⧼LT⧽]` | μ_sum |
| `MEMW-C11` |  | `ALU[[1, 0]; old_timestamp[1], timestamp, ⧼LT⧽]` | w2 |
| `MEMW-C12.i` | i ∈ [2, 3] | `ALU[[1, 0]; old_timestamp[i], timestamp, ⧼LT⧽]` | w4 |
| `MEMW-C13.i` | i ∈ [4, 7] | `ALU[[1, 0]; old_timestamp[i], timestamp, ⧼LT⧽]` | write8 |

As long as `timestamp` is properly range-checked, the presence of `old_timestamp` in the memory argument automatically ensures it is appropriately range checked (this assumes no external entities provide negative multiplicities without range checking the timestamp). This ensures the assumptions for `LT` are satisfied.

There is no need to check that the additions do not overflow, as our address calculations are not performed modulo `2^64` here, and any overflow will result in an address without matching initialization.

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

This chip contributes the following to the lookup argument:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW-CO22` | `MEMW[old; is_register, base_address, value, timestamp, write2, write4, write8]` | -μ_read |
| `MEMW-CO23` | `MEMW[is_register, base_address, value, timestamp, write2, write4, write8]` | -μ_write |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `is_register` | `0` |
| `base_address` | `0` |
| `value` | `0` |
| `timestamp` | `0` |
| `write2` | `0` |
| `write4` | `0` |
| `write8` | `0` |
| `old` | `0` |
| `carry` | `0` |
| `old_timestamp` | `0` |
| `μ_read` | `0` |
| `μ_write` | `0` |

## Read-size aligned fast path

When a memory access happens at an address with proper alignment for its access size (i.e., adding the access size to `base_address`'s lowest limb does not overflow), and all accessed elements were last accessed at the same timestamp, we can instead use the  chip to save on total column count. The saving comes from only requiring a single old timestamp to be stored, as well as being able to guarantee that all values of `add_limb_overflow` would be zero. A minor extra cost is introduced in the form of a check that the alignment is indeed correct, and the corresponding decomposition of the `base_address`.

Further logic remains essentially the same, so we briefly present the relevant tables for this chip.

The  chip only needs  variables, expressed through  columns; it leverages  interactions.

### Input

| Name | Type | Description |
|------|------|-------------|
| `is_register` | `Bit` | Whether the address represents a register index |
| `base_address` | `DWordWHH` | The base address to read from/write to. Gets offset by $[0, 7]$ depending on the size of the access |
| `value` | `BaseField[8]` | The values to store in memory. For regular memory, these should be (up to) 8 range-checked `Byte`s; registers are stored as two range-checked `Word`s |
| `timestamp` | `DWordWL` | The timestamp at which this memory access is said to occur |
| `write2` | `Bit` | Whether to write exactly 2 values |
| `write4` | `Bit` | Whether to write exactly 4 values |
| `write8` | `Bit` | Whether to write exactly 8 values |

### Output

| Name | Type | Description |
|------|------|-------------|
| `old` | `BaseField[8]` | The old value written at `base_address + i`. See `value` for information about representation. Only the elements corresponding to the `writeN` bits are guaranteed |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `old_timestamp` | `DWordWL` | The timestamp at which the address was last accessed |

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

| Tag | Range | Description |
|-----|-------|-------------|
| `MEMW_A-A1.i` | i ∈ [0, 1] | `IS_HALF[base_address[i]]` |
| `MEMW_A-A2` |  | `IS_WORD[base_address[2]]` |
| `MEMW_A-A3` |  | `IS_BIT<write2>` |
| `MEMW_A-A4` |  | `IS_BIT<write4>` |
| `MEMW_A-A5` |  | `IS_BIT<write8>` |
| `MEMW_A-A6` |  | `IS_BIT<write2 + write4 + write8>` |
| `MEMW_A-A7.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

Some of the assumptions can be checked with only arithmetic constraints, so we provide these below.

| Tag | Description |
|-----|-------------|
| `MEMW_A-C1` | `IS_BIT<write2>` |
| `MEMW_A-C2` | `IS_BIT<write4>` |
| `MEMW_A-C3` | `IS_BIT<write8>` |
| `MEMW_A-C4` | `IS_BIT<write2 + write4 + write8>` |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW_A-C9` | `IS_HALF[base_address[0] + write2 + 3 * write4 + 7 * write8]` | μ_sum |
| `MEMW_A-C10` | `IS_BIT<μ_read>` |  |
| `MEMW_A-C11` | `IS_BIT<μ_write>` |  |
| `MEMW_A-C12` | `IS_BIT<μ_sum>` |  |
| `MEMW_A-C13` | `w2` => `μ_sum` |  |
| | _polynomial:_ `w2 * (1 - μ_sum) = 0` | |
| `MEMW_A-C14` | `ALU[[1, 0]; old_timestamp, timestamp, ⧼LT⧽]` | μ_sum |

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW_A-CM15` |  | `memory[is_register, base_address::DWordWL, old_timestamp, old[0]]` | μ_sum |
| `MEMW_A-CM16` |  | `memory[is_register, base_address::DWordWL, timestamp, value[0]]` | -μ_sum |
| `MEMW_A-CM17` |  | `memory[is_register, base_address::DWordWL + 1::DWordWL, old_timestamp, old[1]]` | w2 |
| `MEMW_A-CM18` |  | `memory[is_register, base_address::DWordWL + 1::DWordWL, timestamp, value[1]]` | -w2 |
| `MEMW_A-CM19.i` | i ∈ [2, 3] | `memory[is_register, base_address::DWordWL + i::DWordWL, old_timestamp, old[i]]` | w4 |
| `MEMW_A-CM20.i` | i ∈ [2, 3] | `memory[is_register, base_address::DWordWL + i::DWordWL, timestamp, value[i]]` | -w4 |
| `MEMW_A-CM21.i` | i ∈ [4, 7] | `memory[is_register, base_address::DWordWL + i::DWordWL, old_timestamp, old[i]]` | write8 |
| `MEMW_A-CM22.i` | i ∈ [4, 7] | `memory[is_register, base_address::DWordWL + i::DWordWL, timestamp, value[i]]` | -write8 |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW_A-CO23` | `MEMW[old; is_register, base_address::DWordWL, value, timestamp, write2, write4, write8]` | -μ_read |
| `MEMW_A-CO24` | `MEMW[is_register, base_address::DWordWL, value, timestamp, write2, write4, write8]` | -μ_write |

### Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `is_register` | `0` |
| `base_address` | `0` |
| `value` | `0` |
| `timestamp` | `0` |
| `write2` | `0` |
| `write4` | `0` |
| `write8` | `0` |
| `old` | `0` |
| `old_timestamp` | `0` |
| `μ_read` | `0` |
| `μ_write` | `0` |

## Register fast-path

The  chip provides a fast-path for accessing registers. This fast-path leverages that registers + can be addressed using a `Byte`, rather than a full `DWord`, + are constantly accessed, i.e., ``timestamp` - `old_timestamp`` is small, and + have a fixed access pattern to achieve a footprint that is significantly smaller than both  and .

Note: as a result of hard optimization, this chip can only be used for register accesses for which + ``timestamp` - `old_timestamp` in [1, 2^16]`, and + ``timestamp[0]` > `old_timestamp[0]`` If either of these rules does not apply to your access, you should fall back to using `MEMW_A`.

Note moreover that this chip does not guard against misaligned register access faults: to access register with a given `address`, one must provide `2 dot `address`` in the lookup.

### Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interactions:

### Input

| Name | Type | Description |
|------|------|-------------|
| `address` | `Byte` | address of the register being accessed |
| `timestamp` | `DWordWL` | timestamp at which the access takes place |
| `val` | `DWordWL` | value being written to this register |

### Output

| Name | Type | Description |
|------|------|-------------|
| `old` | `DWordWL` | value of this register at `old_timestamp`. |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `old_timestamp_lo` | `Word` | the lower limb of `old_timestamp` |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `old_timestamp` | `DWordWL` | timestamp at which this register was last accessed |
| `μ_sum` | `Bit` |  |

**Definition of `old_timestamp`:**
```
old_timestamp := [old_timestamp_lo, timestamp[1]]::DWordWL
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

### Assumptions

The following range checks are assumed to be performed/enforced outside of this chip:

| Tag | Range | Description |
|-----|-------|-------------|
| `MEMW_R-A1.i` | i ∈ [0, 1] | `IS_WORD[val[i]]` |
| `MEMW_R-A2.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

### Constraints

Since most registers are frequently accessed, the difference between `timestamp` and `old_timestamp` is small most of the times. Rather than storing their (nearly) identical upper limbs twice, it is instead assumed that ``old_timestamp[1]` = `timestamp[1]``;  can be used for accesses where this is not the case.

Verifying that ``timestamp` > `old_timestamp`` now simplifies to verifying that ``timestamp[0]` - `old_timestamp[0]` > 0`. For most accesses, this value will be small enough to fit in a `Half`. This chip thus enforces this by means of the following constraint:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW_R-C1` | `IS_HALF[timestamp[0] - old_timestamp[0] - 1]` | μ_sum |

With ``old_timestamp`<`timestamp`` asserted, `old` is read from the register ([regw:c:read_old]) and `val` is written back ([regw:c:write_val]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW_R-C2.i` | i ∈ [0, 1] | `memory[1, [(2 * address + i)::Word, 0], old_timestamp, old[i]]` | μ_sum |
| `MEMW_R-C3.i` | i ∈ [0, 1] | `memory[1, [(2 * address + i)::Word, 0], timestamp, val[i]]` | -μ_sum |

This chip can either just write (``μ_write` = 1`), or both read and write (``μ_read` = 1`) in the same cycle. It must be asserted that at most one of these two options is selected:

| Tag | Description |
|-----|-------------|
| `MEMW_R-C4` | `IS_BIT<μ_read>` |
| `MEMW_R-C5` | `IS_BIT<μ_write>` |
| `MEMW_R-C6` | `IS_BIT<μ_sum>` |

Lastly, this chip contributes the following interactions to the logup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW_R-C7` | `MEMW[[old[0], old[1], 0, 0, 0, 0, 0, 0]; 1, [(2 * address)::Word, 0], [val[0], val[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | -μ_read |
| `MEMW_R-C8` | `MEMW[1, [(2 * address)::Word, 0], [val[0], val[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | -μ_write |

### Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `address` | `0` |
| `timestamp` | `0` |
| `val` | `0` |
| `old` | `0` |
| `old_timestamp_lo` | `0` |
| `μ_read` | `0` |
| `μ_write` | `0` |

## Notes/optimizations

The following ideas may prove to be optimizations for the // chip: - `MEMB` chip that does a one-byte write to remove old_timestamp from here (uncertain tradeoffs) - Adding `μ_sum`/`w2`/`w4`/`write8` multiplicities to the `IS_HALF` lookups may make some GKR things faster if there are known zeroes. - For the register fast-path, one may upgrade the `IS_HALF` check to an `IS_B20` check for extended range at the cost of looking through a larger table.