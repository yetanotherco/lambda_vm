# NEG Template

It requires `cond` to be a bit.

## Variables

This template introduces  interaction(s).

### Input

| Name | Type | Description |
|------|------|-------------|
| `x` | `DWordHL` | value to compute negation of |

### Output

| Name | Type | Description |
|------|------|-------------|
| `neg` | `DWordWL` | negation of `x` if $`cond` != 0$; unconstrained otherwise. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `carry` | `Bit[2]` | carries of the addition $`neg` + `x`$. |

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * ((x::DWordWL)[0] + neg[0])
carry (when iter=1) := 2^-32 * ((x::DWordWL)[1] + neg[1] + carry[0])
```

### Condition

| Name | Type | Description |
|------|------|-------------|
| `cond` | `Bit` | condition on whether to negate x |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `NEG-A1.i` | i ∈ [0, 3] | `IS_HALF[x[i]]` |
| `NEG-A2` |  | `IS_BIT<cond>` |

## Constraints

We constrain this equality using two constraints:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `NEG-C1` | `ZERO[1 - carry[0]; x[0] + x[1]]` | cond |
| `NEG-C2` | `ZERO[1 - carry[1]; x[0] + x[1] + x[2] + x[3]]` | cond |

### Correctness argument

The constraints force the `carry` values to be fixed. Writing `carry`'s definition, we then find that $

## cases(

2^32 - (`x as DWordWL`)_0 & "if" (`x as DWordWL`)_0 != 0, 0 & "if" (`x as DWordWL`)_0 = 0 ),\

2^32 - (`x as DWordWL`)_1 - 1 & "if" `x` != 0, 0 & "if" `x` = 0 $ Clearly, ``neg` = 0` when ``x` = 0` (and `cond` is set). For non-zero `x`, we distinguish two cases. When `(`x as DWordWL`)_0 = 0`, $

&= 2^32 dot `neg`_1 + `neg`_0\ &= 2^32 dot (2^32 - (`x as DWordWL`)_1) + 0\ &= 2^32 dot (2^32 - (`x as DWordWL`)_1) + (`x as DWordWL`)_0\ &= 2^64 - (2^32 dot (`x as DWordWL`)_1 + (`x as DWordWL`)_0)\ &= 2^64 - `x`\ &equiv -x mod 2^64, $ while when `(`x as DWordWL`)_0 != 0`, $

&= 2^32 dot `neg`_1 + `neg`_0\ &= 2^32 dot (2^32 - (`x as DWordWL`)_1 - 1) + (2^32 - (`x as DWordWL`)_0)  \ &= 2^64 - 2^32 dot (`x as DWordWL`)_1 - 2^32 + 2^32 - (`x as DWordWL`)_0  \ &= 2^64 - ((`x as DWordWL`)_0 + 2^32 dot (`x as DWordWL`)_1) \ &= 2^64 - `x`\ &equiv -x mod 2^64 $ when `cond` is set. When `cond` is not set, the two lookups are not executed, allowing `neg` to take any value in either case.

It is worth noting that this construction does _not_ require the limbs of `neg` to be range checked, thus allowing it be represented by the unrangecheckable `DWordWL` rather than a `DWordHL`. The input value `x` is still assumed to be range-checked, however. ]