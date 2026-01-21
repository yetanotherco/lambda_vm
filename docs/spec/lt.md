# LT Chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `DWordHHW` | The left operand |
| `rhs` | `DWordHHW` | The right operand |
| `signed` | `Bit` | whether to interpret `lhs` and `rhs` as signed integers (1) or not (0) |

### Output

| Name | Type | Description |
|------|------|-------------|
| `lt` | `Bit` | Whether $`lhs` < `rhs`$, taking `signed` into account |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `lhs_sub_rhs` | `DWordHL` | $`lhs` - `rhs`$ |
| `lhs_msb` | `Bit` | The most significant bit of `lhs` |
| `rhs_msb` | `Bit` | The most significant bit of `rhs` |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `carry` | `Bit[2]` | The carry for adding `lhs_sub_rhs` back to `rhs` |
| `unsigned_lt` | `Bit` | Whether $`lhs` < `rhs`$, as unsigned integers |

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * (rhs[0] + (lhs_sub_rhs::DWordWL)[0] - lhs[0])
carry (when iter=1) := 2^-32 * ((rhs::DWordWL)[1] + (lhs_sub_rhs::DWordWL)[1] + carry[0] - (lhs::DWordWL)[1])
```

**Definition of `unsigned_lt`:**
```
unsigned_lt := carry[1]
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

| Ref | Range | Description |
|-----|-------|-------------|
| `lt:a:range_lhs` | i ∈ [1, 2] | `IS_HALFWORD[lhs[i]]` and `IS_WORD[lhs[0]]` |
| `lt:a:range_rhs` | i ∈ [1, 2] | `IS_HALFWORD[rhs[i]]` and `IS_WORD[rhs[0]]` |
| `lt:a:range_signed` |  | `IS_BIT<signed>` |

## Constraints

### defs
_Enforce that variables have been correctly computed_

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `lt:c:lhs_msb` | interaction | `MSB16[lhs_msb; lhs[2]]` | μ |
| `lt:c:rhs_msb` | interaction | `MSB16[rhs_msb; rhs[2]]` | μ |
| `lt:c:lt` | arith | `lt` = `signed` dot (A (1 - B) + A C + (1 - B) C) + (1 - `signed`) dot `unsigned_lt` |  |
| | | _polynomial:_ `lt - signed * (lhs_msb * (1 - rhs_msb) + lhs_msb * carry[1] + (1 - rhs_msb) * carry[1]) - (1 - signed) * unsigned_lt = 0` | |
| | | _note:_ Where $A = #`lhs_msb`$, $B = #`rhs_msb`$ and $C = #`carry[1]`$ | |

### sub
_Constrain the subtraction_

| Ref | Kind | Range | Description | Multiplicity |
|-----|------|-------|-------------|--------------|
| `1` | template | i ∈ [0, 1] | `IS_BIT<carry[i]>` |  |
| `lt:c:lhs_sub_rhs_range` | interaction | i ∈ [0, 3] | `IS_HALFWORD[lhs_sub_rhs[i]]` | μ |

### output
_Each row contributes the following to the LogUp sum_

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `1` | interaction | `LT[lt; lhs, rhs, signed]` | -μ |
