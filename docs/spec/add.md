# ADD Chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `DWordWL` | left-hand operator |
| `rhs` | `DWordWL` | right-hand operator |

### Output

| Name | Type | Description |
|------|------|-------------|
| `sum` | `DWordWL` | $`lhs` + `rhs`$ |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `carry` | `Bit[2]` | Carry values used to constrain the addition |

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * (lhs[0] + rhs[0] - sum[0])
carry (when iter=1) := 2^-32 * (lhs[1] + rhs[1] + carry[0] - sum[1])
```

### Condition

| Name | Type | Description |
|------|------|-------------|
| `cond` | `BaseField` | Whether the relation should be enforced ($eq.not 0$) or not ($0$). |

## Assumptions

| Ref | Range | Description |
|-----|-------|-------------|
| `add:a:lhs` | i ∈ [0, 1] | `IS_WORD[lhs[i]]` |
| `add:a:rhs` | i ∈ [0, 1] | `IS_WORD[rhs[i]]` |
| `add:a:sum` | i ∈ [0, 1] | `IS_WORD[sum[i]]` |

## Constraints

### all

| Ref | Kind | Range | Description |
|-----|------|-------|-------------|
| `add:c:carry` | template | i ∈ [0, 1] | cond ⇒ `IS_BIT<carry[i]>` |
