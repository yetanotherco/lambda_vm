# ADD/SUB Template

box( inset: (left: 4pt, right: 4pt), outset: (top: 4pt, bottom: 4pt), radius: 2pt, fill: luma(230), raw(code)) }

## Notation

The  constraint template has the following interface:

where `cond` is any value described by an expression _of degree at most `1`_.

### 

For ease of notation, we moreover introduce the  constraint template. Its interface

maps onto the  template as

It constrains that ``diff` = `lhs` - `rhs` mod 2^64` when the expression `cond` is non-zero. As with ,  can be used to denote the _unconditional_ application of the template.

## Variables

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `ADD-A1.i` | i ∈ [0, 1] | `IS_WORD[lhs[i]]` |
| `ADD-A2.i` | i ∈ [0, 1] | `IS_WORD[rhs[i]]` |
| `ADD-A3.i` | i ∈ [0, 1] | `IS_WORD[sum[i]]` |

## Constraints

This template introduces the following constraints

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

### all

| Tag | Range | Description |
|-----|-------|-------------|
| `ADD-C1.i` | i ∈ [0, 1] | cond ⇒ `IS_BIT<carry[i]>` |