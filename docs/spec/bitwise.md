# BITWISE Chips

The  chips deal with precomputed lookup tables for bitwise boolean operations and convenience functionalities over small domains.

## Variables

The  chip is comprised of  variables that are expressed using  columns. Of these, the _input_ and _output_ variables ( in total) are precomputed.

### Input

| Name | Type | Description |
|------|------|-------------|
| `X` | `Byte` |  |
| `Y` | `Byte` |  |
| `Z` | `B4` |  |

### Output

| Name | Type | Description |
|------|------|-------------|
| `AND` | `Byte` | the binary AND of `X` and `Y` |
| `OR` | `Byte` | the binary OR of `X` and `Y` |
| `XOR` | `Byte` | the binary XOR of `X` and `Y` |
| `MSB8` | `Bit` | the most significant bit of `X` |
| `MSB16` | `Bit` | the most significant bit of `Y` |
| `ZERO` | `Bit` | whether $`X` = 0$, $`Y` = 0$ and $`Z` = 0$. |
| `SLL` | `Half` | `X\|\|Y` logically left-shifted by `Z`: $((`X` + 256`Y`) `<<` `Z`) mod 2^16$ |
| `SLLC` | `Half` | `X\|\|Y` logically right-shifted by `Z`: $(`X` + 256`Y`) `>>` (16 - `Z`)$ |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ_AND` | `BaseField` |  |
| `μ_OR` | `BaseField` |  |
| `μ_XOR` | `BaseField` |  |
| `μ_MSB8` | `BaseField` |  |
| `μ_MSB16` | `BaseField` |  |
| `μ_ZERO` | `BaseField` |  |
| `μ_IS_BYTE` | `BaseField` |  |
| `μ_ARE_BYTES` | `BaseField` |  |
| `μ_IS_HALF` | `BaseField` |  |
| `μ_IS_B20` | `BaseField` |  |
| `μ_HWSL` | `BaseField` |  |

*Note*: This table contains one row for every possible value of `(X, Y, Z)`. As such, it has length `2^8 dot 2^8 dot 2^4 = 2^(20)`.

We use the ALU operation descriptors from [decode] to identify the operations in the `BYTE_ALU` interaction. Since each of the three columns is only `2^16` rows long, they can be combined in a single `2^20` column (with room to spare).

## Lookup

This chip adds the following interactions to the lookup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `BITWISE-C1` | `BYTE_ALU[AND; ⧼AND⧽, X, Y]` | -μ_AND |
| `BITWISE-C2` | `BYTE_ALU[OR; ⧼OR⧽, X, Y]` | -μ_OR |
| `BITWISE-C3` | `BYTE_ALU[XOR; ⧼XOR⧽, X, Y]` | -μ_XOR |
| `BITWISE-C4` | `MSB8[MSB8; X]` | -μ_MSB8 |
| `BITWISE-C5` | `MSB16[MSB16; X + 256 * Y]` | -μ_MSB16 |
| `BITWISE-C6` | `ZERO[ZERO; X + 256 * Y + 65536 * Z]` | -μ_ZERO |
| `BITWISE-C7` | `ARE_BYTES[X, Y]` | -μ_ARE_BYTES |
| `BITWISE-C8` | `IS_HALF[X + 256 * Y]` | -μ_IS_HALF |
| `BITWISE-C9` | `IS_B20[X + 256 * Y + 65536 * Z]` | -μ_IS_B20 |
| `BITWISE-C10` | `HWSL[[SLL, SLLC]; X + 256 * Y, Z]` | -μ_HWSL |

## Notes/Optimizations

The following ideas may prove to be optimizations for the  chip: + Drop `MSB8` column, and instead define the `MSB8` lookup as `MSB8<X> := MSB16[256X]`. Note: currently, `MSB8` also implicity range checks the input `X` (the lookup fails if `X` is not a `Byte`). This optimization should only be executed when all chips leveraging `MSB8` do _not_ need this implicit range check. + Place the 16-bit (`AND`, `OR`, `XOR`, `MSB16`, etc.) and 20-bit (`HWSL`, `IS_B20`, `ZERO`) lookups in separate tables.