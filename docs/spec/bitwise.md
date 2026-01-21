# BITWISE Chip

## Columns

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
| `ZERO` | `Bit` | whether $`X` = 0 and `Y` = 0$ |
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
| `μ_IS_HALF` | `BaseField` |  |
| `μ_IS_B20` | `BaseField` |  |
| `μ_HWSL` | `BaseField` |  |
| `μ_HWSLC` | `BaseField` |  |

## Constraints

### contributions

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `1` | interaction | `AND_BYTE[AND; X, Y]` | -μ_AND |
| `2` | interaction | `OR_BYTE[OR; X, Y]` | -μ_OR |
| `3` | interaction | `XOR_BYTE[XOR; X, Y]` | -μ_XOR |
| `4` | interaction | `MSB8[MSB8; X]` | -μ_MSB8 |
| `5` | interaction | `MSB16[MSB16; X + 256 * Y]` | -μ_MSB16 |
| `6` | interaction | `ZERO[ZERO; X + 256 * Y]` | -μ_ZERO |
| `7` | interaction | `IS_BYTE[X]` | -μ_IS_BYTE |
| `8` | interaction | `IS_HALF[X + 256 * Y]` | -μ_IS_HALF |
| `9` | interaction | `IS_B20[X + 256 * Y + 65536 * Z]` | -μ_IS_B20 |
| `10` | interaction | `HWSL[SLL; X + 256 * Y, Z]` | -μ_HWSL |
| `11` | interaction | `HWSLC[SLLC; X + 256 * Y, Z]` | -μ_HWSLC |
