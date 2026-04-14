# SIGN Template

It constrains that `sign` is set to `1` when both `X`'s most significant bit and `signed` are `1`, and `0` otherwise.

## Variables

The  template introduces  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `X` | `Half` | Value for which to extract its sign. |
| `signed` | `Bit` | Whether `X` represents a signed value (1) or not (0) |

### Output

| Name | Type | Description |
|------|------|-------------|
| `sign` | `Bit` | Sign of `X` |

## Assumptions

The  template operates on the following assumptions:

| Tag | Range | Description |
|-----|-------|-------------|
| `SIGN-A1` |  | `IS_BIT<signed>` |

If `sign` is set to `1`, `X` will be range-checked to be a halfword, and hence proving may fail if this is not ensured.

## Constraints

It takes only two constraints to compute the `sign` of `X`, given whether `X` represents a `signed` value or not. When ``signed` = 1`, the sign of `X` is equal to its most significant bit. This value is extracted in [sign:c:sign_if_signed]. If `X` is unsigned (i.e., ``signed` = 0`), its sign is always `0`. This is constrained by [sign:c:sign_if_unsigned].

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SIGN-C1` | `MSB16[sign; X]` | signed |
| `SIGN-C2` | not`signed` => `sign` = 0 |  |
| | _polynomial:_ `(1 - signed) * sign = 0` | |