# MUL Chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `DWordHL` | the left hand operator. |
| `lhs_signed` | `Bit` | whether to interpret `lhs` as a signed integer (1) or not (0). |
| `rhs` | `DWordHL` | the right hand operator. |
| `rhs_signed` | `Bit` | whether to interpret `rhs` as a signed integer (1) or not (0). |

### Output

| Name | Type | Description |
|------|------|-------------|
| `res` | `QuadWL` | the (extended) multiplication result |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `lhs_is_negative` | `Bit` | whether `lhs` is negative (1) or not (0) |
| `rhs_is_negative` | `Bit` | whether `rhs` is negative (1) or not (0) |
| `raw_product` | `B51[4]` | raw multiplication output |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `lhs_ext` | `Half[8]` | sign-extended value of `lhs` |
| `rhs_ext` | `Half[8]` | sign-extended value of `rhs` |
| `carry` | `B20[4]` | carry values |
| `μ_sum` | `BaseField` | sum of multiplicies |

**Definition of `lhs_ext`:**
```
lhs_ext := lhs[i]
lhs_ext := 65535 * lhs_is_negative
```

**Definition of `rhs_ext`:**
```
rhs_ext := rhs[i]
rhs_ext := 65535 * rhs_is_negative
```

**Definition of `carry`:**
```
carry := 2^-32 * (raw_product[0] - res[0])
carry := 2^-32 * (raw_product[i] + carry[i - 1] - res[i])
```

**Definition of `μ_sum`:**
```
μ_sum := μ_lo + μ_hi
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ_lo` | `BaseField` |  |
| `μ_hi` | `BaseField` |  |

## Assumptions

| Ref | Range | Description |
|-----|-------|-------------|
| `A1` |  | `IS_HALF[lhs[i]]` |
| `A2` |  | `IS_HALF[rhs[i]]` |
| `mul:a:res` |  | `IS_WORD[res[i]]` |

## Constraints

### def

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `mul:c:lhs_is_negative` | template | `SIGN<lhs_is_negative; lhs[3], lhs_signed>` |  |
| `mul:c:rhs_is_negative` | template | `SIGN<rhs_is_negative; rhs[3], rhs_signed>` |  |
| `mul:c:carry` | interaction | `IS_B20[carry[i]]` | μ_sum |

### prod

| Ref | Kind | Description |
|-----|------|-------------|
| `mul:c:raw_product` | arith | `raw_product[i]` = sum_(`k`=0)^1 2^(16k) sum_(`j`=0)^(2i+k) `lhs_ext[j]` dot `rhs_ext[2i+k-j]` |
| | | _polynomial:_ `Σ_k = 0^1 2^(16 * k) * Σ_j = 0^2 * i + k lhs_ext[j] * rhs_ext[2 * i + k - j] - raw_product[i] = 0` |

### lookup

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `mul:c:lookup_lo` | interaction | `MUL[res[0:4]; lhs, lhs_signed, rhs, rhs_signed, 0]` | -μ_lo |
| `mul:c:lookup_hi` | interaction | `MUL[res[4:8]; lhs, lhs_signed, rhs, rhs_signed, 1]` | -μ_hi |
