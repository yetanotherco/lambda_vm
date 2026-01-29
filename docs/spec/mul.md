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
| `lo` | `DWordHL` | the lower limbs of the (extended) multiplication result |
| `hi` | `DWordHL` | the upper limbs of the (extended) multiplication result |

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
| `res` | `QuadWL` | concatenation of `lo` and `hi`. |
| `carry` | `B20[4]` | carry values |
| `μ_sum` | `BaseField` | sum of multiplicies |

**Definition of `lhs_ext`:**
```
lhs_ext (when iter=[0, 3]) := lhs[i]
lhs_ext (when iter=[4, 7]) := 65535 * lhs_is_negative
```

**Definition of `rhs_ext`:**
```
rhs_ext (when iter=[0, 3]) := rhs[i]
rhs_ext (when iter=[4, 7]) := 65535 * rhs_is_negative
```

**Definition of `res`:**
```
res (when iter=[0, 1]) := (lo::DWordWL)[i]
res (when iter=[2, 3]) := (hi::DWordWL)[i - 2]
```

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * (raw_product[0] - res[0])
carry (when iter=[1, 3]) := 2^-32 * (raw_product[i] + carry[i - 1] - res[i])
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
| `A1` | i ∈ [0, 3] | `IS_HALF[lhs[i]]` |
| `A2` | i ∈ [0, 3] | `IS_HALF[rhs[i]]` |

## Constraints

### def

| Ref | Kind | Range | Description | Multiplicity |
|-----|------|-------|-------------|--------------|
| `mul:c:lhs_is_negative` | template |  | `SIGN<lhs_is_negative; lhs[3], lhs_signed>` |  |
| `mul:c:rhs_is_negative` | template |  | `SIGN<rhs_is_negative; rhs[3], rhs_signed>` |  |
| `mul:c:range_lo` | interaction | i ∈ [0, 3] | `IS_HALF[lo[i]]` | μ_sum |
| `mul:c:range_hi` | interaction | i ∈ [0, 3] | `IS_HALF[hi[i]]` | μ_sum |
| `mul:c:carry` | interaction | i ∈ [0, 3] | `IS_B20[carry[i]]` | μ_sum |

### prod

| Ref | Kind | Range | Description |
|-----|------|-------|-------------|
| `mul:c:raw_product` | arith | i ∈ [0, 3] | `raw_product[i]` = sum_(`k`=0)^1 2^(16k) sum_(`j`=0)^(2i+k) `lhs_ext[j]` dot `rhs_ext[2i+k-j]` |
| | | _polynomial:_ `Σ_k = 0^1 2^(16 * k) * Σ_j = 0^2 * i + k lhs_ext[j] * rhs_ext[2 * i + k - j] - raw_product[i] = 0` |

### lookup

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `mul:c:lookup_lo` | interaction | `MUL[lo::DWordWL; lhs, lhs_signed, rhs, rhs_signed, 0]` | -μ_lo |
| `mul:c:lookup_hi` | interaction | `MUL[hi::DWordWL; lhs, lhs_signed, rhs, rhs_signed, 1]` | -μ_hi |
