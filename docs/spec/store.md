# STORE Chip

The  chip provides functionality to store a value to memory. It decomposes a `DWord` into bytes and delegates low-level memory handling to the `MEMW` chip ([memw]).

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `base_address` | `DWordWL` | The base address to write to, gets offset by $[0, 7]$, depending on how big the access is |
| `timestamp` | `DWordWL` | The timestamp at which this memory access is said to occur |
| `write2` | `Bit` | Whether to write exactly 2 bytes |
| `write4` | `Bit` | Whether to write exactly 4 bytes |
| `write8` | `Bit` | Whether to write exactly 8 bytes |
| `value` | `DWordBL` | The value to store |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `write1` | `Bit` | Whether to write exactly 1 byte |

**Definition of `write1`:**
```
write1 := μ - write2 - write4 - write8
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `STORE-A1.i` | i ∈ [0, 1] | `IS_WORD[base_address[i]]` |
| `STORE-A2.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

## Constraints

The chip delegates the actual memory interaction to the `MEMW` chip, and ensures the values are proper bytes.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `STORE-C1` |  | `IS_BIT<μ>` |  |
| `STORE-C2` |  | `IS_BIT<write2>` |  |
| `STORE-C3` |  | `IS_BIT<write4>` |  |
| `STORE-C4` |  | `IS_BIT<write8>` |  |
| `STORE-C5` |  | `IS_BIT<write2 + write4 + write8>` |  |
| `STORE-C6` |  | `write2` + `write4` + `write8` => `μ` = 1 |  |
| | | _polynomial:_ `(write2 + write4 + write8) * (1 - μ) = 0` | |
| `STORE-C7.i` | i ∈ [0, 7] | μ ⇒ `IS_BYTE<value[i]>` |  |
| `STORE-C8` |  | `MEMW[0, base_address, value, timestamp, write2, write4, write8]` | μ |

The chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `STORE-C9` | `MEMOP[0::DWordWL; timestamp, base_address, value::DWordWL, 1 + 4 * write2 + 8 * write4 + 16 * write8]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `base_address` | `0` |
| `timestamp` | `0` |
| `write2` | `0` |
| `write4` | `0` |
| `write8` | `0` |
| `value` | `0` |
| `μ` | `0` |