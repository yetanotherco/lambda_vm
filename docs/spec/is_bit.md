# IS_BIT Template

Barring exceptional cases, this template is used to assert that a variable of type `Bit` assumes a valid value under some condition.

## Variables

The  template operates on  variables:

### Input

| Name | Type | Description |
|------|------|-------------|
| `X` | `BaseField` | Value for which to assert that it lies in the range ${0, 1}$. |

### Condition

| Name | Type | Description |
|------|------|-------------|
| `cond` | `BaseField` | Whether the constraint should be applied ($eq.not 0$) or not ($0$). |

## Constraints

It takes only one constraint to enforce that `X` must be either `0` or `1` whenever ``cond` eq.not 0`:

| Tag | Description |
|-----|-------------|
| `IS_BIT-C1` | `cond` => `X` (1-`X`) = 0 |
| | _polynomial:_ `cond * X * (1 - X) = 0` |

*Note*: - In case of _unconditional_ template application, `cond` can be dropped from the constraint, simplifying it to ``X` (1- `X`) = 0`. - As described earlier, the `cond` variable must be describable by a degree-1 (i.e., linear) expression. This is to make sure that [isbit:c:isbit]'s expression has degree at most 3.

### Correctness argument

If `cond` is `0`, [isbit:c:isbit] is trivially satisfied: `X` can assume any value and the polynomial constraint will evaluate to `0` regardless. When ``cond` eq.not 0`, it follows that the statement can only be proven when ``X` (1-`X`) equiv 0 mod p`, with `p` the modulus of the field. Because `BaseField` is a prime field, this equality is only satisfied if either ``X` equiv 0 mod p` or `1-`X` equiv 0 mod p`. Hence, it is proven that when ``cond` eq.not 0`, [isbit:c:isbit] is only satisfied if ``X` in {0, 1}`.