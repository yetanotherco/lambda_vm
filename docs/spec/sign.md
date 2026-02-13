# SIGN Template

box( inset: (left: 4pt, right: 4pt), outset: (top: 4pt, bottom: 4pt), radius: 2pt, fill: luma(230), raw(code)) }

## Interface

The  constraint template has the following interface:

It constrains that `sign` is set to `1` when both `X`'s most significant bit and `signed` are `1`, and `0` otherwise.

## Variables

The  template operates on three variables:

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `SIGN-A1` |  | `IS_HALF[X]` |
| `SIGN-A2` |  | `IS_BIT<signed>` |

The  template operates on the following assumptions:

## Constraints

It takes only two constraints to compute the `sign` of `X`, given whether `X` represents a `signed` value or not. When ``signed` = 1`, the sign of `X` is equal to its most significant bit. This value is extracted in [sign:c:sign_if_signed]. If `X` is unsigned (i.e., ``signed` = 0`), its sign is always `0`. This is constrained by [sign:c:sign_if_unsigned].

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `X` | `Half` | Value for which to extract its sign. |
| `signed` | `Bit` | Whether `X` represents a signed value (1) or not (0) |

### Output

| Name | Type | Description |
|------|------|-------------|
| `sign` | `Bit` | Sign of `X` |

### all

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SIGN-C1` | `MSB16[sign; X]` | signed |
| `SIGN-C2` | not`signed` => `sign` = 0 |  |
| | _polynomial:_ `(1 - signed) * sign = 0` | |