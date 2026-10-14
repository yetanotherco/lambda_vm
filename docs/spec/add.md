# ADD/SUB Template

For ease of notation, we moreover introduce the  constraint template $

$ in both conditional and unconditional versions. It constrains that ``diff` equiv `lhs` - `rhs` (mod 2^64)` when the expression `cond` is non-zero.

## Variables

This template introduces  interaction(s).

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

| Tag | Range | Description |
|-----|-------|-------------|
| `ADD-A1.i` | i ∈ [0, 1] | `IS_WORD[lhs[i]]` |
| `ADD-A2.i` | i ∈ [0, 1] | `IS_WORD[rhs[i]]` |
| `ADD-A3.i` | i ∈ [0, 1] | `IS_WORD[sum[i]]` |

## Constraints

This template introduces the following constraints

| Tag | Range | Description |
|-----|-------|-------------|
| `ADD-C1.i` | i ∈ [0, 1] | cond ⇒ `IS_BIT<carry[i]>` |