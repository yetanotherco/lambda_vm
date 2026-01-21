# IS_BIT Chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `X` | `BaseField` | Value for which to assert that it lies in the range ${0, 1}$. |

### Condition

| Name | Type | Description |
|------|------|-------------|
| `cond` | `BaseField` | Whether the constraint should be applied ($eq.not 0$) or not ($0$). |

## Constraints

### all

| Ref | Kind | Description |
|-----|------|-------------|
| `isbit:c:isbit` | arith | `cond` => `X` (1-`X`) = 0 |
| | | _polynomial:_ `cond * X * (1 - X) = 0` |
