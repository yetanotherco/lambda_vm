# LOAD Chip

The  chip provides functionality to read values from memory and sign-extend them where appropriate. It delegates low-level memory handling to the `MEMW` chip ([memw]).

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `base_address` | `DWordWL` | The base address to read from, gets offset by $[0, 7]$, depending on how big the access is |
| `timestamp` | `DWordWL` | The timestamp at which this memory access is said to occur |
| `read2` | `Bit` | Whether to read exactly 2 bytes |
| `read4` | `Bit` | Whether to read exactly 4 bytes |
| `read8` | `Bit` | Whether to read exactly 8 bytes |
| `signed` | `Bit` | Whether to sign-extend (1) or zero-extend (0) |

### Output

| Name | Type | Description |
|------|------|-------------|
| `res` | `DWordBL` | The result of reading (up to) 8 bytes from `base_address`, extended corresponding to `signed`. |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `sign_bit` | `Bit` | The sign bit extracted from the bytes retrieved from memory |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `read1` | `Bit` | Whether to read exactly 1 byte |

**Definition of `read1`:**
```
read1 := μ - read2 - read4 - read8
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `LOAD-A1.i` | i ∈ [0, 1] | `IS_WORD[base_address[i]]` |
| `LOAD-A2.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

## Constraints

The chip delegates the actual memory interaction to the `MEMW` chip, and ensures correctness of the requested sign/zero extension. The output `res` is correctly range-checked as long as the memory contents are.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `LOAD-C1` |  | `IS_BIT<signed>` |  |
| `LOAD-C2` |  | `IS_BIT<read2>` |  |
| `LOAD-C3` |  | `IS_BIT<read4>` |  |
| `LOAD-C4` |  | `IS_BIT<read8>` |  |
| `LOAD-C5` |  | `IS_BIT<read2 + read4 + read8>` |  |
| `LOAD-C6` |  | `read2` + `read4` + `read8` => `μ` |  |
| | | _polynomial:_ `(read2 + read4 + read8) * (1 - μ) = 0` | |
| `LOAD-C7` |  | `MEMW[res; 0, base_address, res::BaseField[8], timestamp, read2, read4, read8]` | μ |
| `LOAD-C8` |  | `MSB8[sign_bit; res[0]]` | read1 |
| `LOAD-C9` |  | `MSB8[sign_bit; res[1]]` | read2 |
| `LOAD-C10` |  | `MSB8[sign_bit; res[3]]` | read4 |
| `LOAD-C11.i` | i ∈ [4, 7] | !`read8` => `res`_i = `signed` dot `sign_bit` dot 255 |  |
| | | _polynomial:_ `(1 - read8) * (res[i] - signed * sign_bit * 255) = 0` | |
| `LOAD-C12.i` | i ∈ [2, 3] | !(`read4` + `read8`) => `res`_i = `signed` dot `sign_bit` dot 255 |  |
| | | _polynomial:_ `(1 - read4 - read8) * (res[i] - signed * sign_bit * 255) = 0` | |
| `LOAD-C13` |  | !(`read2` + `read4` + `read8`) => `res`_1 = `signed` dot `sign_bit` dot 255 |  |
| | | _polynomial:_ `(1 - read2 - read4 - read8) * (res[1] - signed * sign_bit * 255) = 0` | |

The chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LOAD-C14` | `MEMOP[res::DWordWL; timestamp, base_address, 0::DWordWL, 2 * signed + 4 * read2 + 8 * read4 + 16 * read8]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `base_address` | `0` |
| `timestamp` | `0` |
| `read2` | `0` |
| `read4` | `0` |
| `read8` | `0` |
| `signed` | `0` |
| `res` | `0` |
| `sign_bit` | `0` |
| `μ` | `0` |