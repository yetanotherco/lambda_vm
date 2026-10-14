# EQ Chip

The  chip is an ALU chip that compares two values and outputs a bit indicating whether they are equal or not. It optionally inverts the result if the `invert` flag is set.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `a` | `DWordWL` | The first input |
| `b` | `DWordWL` | The second input |
| `invert` | `Bit` | Whether to invert the result |

### Output

| Name | Type | Description |
|------|------|-------------|
| `res` | `Bit` | The result |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `diff` | `DWordHL` | The difference `a - b` |
| `eq` | `Bit` | The bit indicating `a == b` |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `EQ-A1.i` | i ∈ [0, 1] | `IS_WORD[a[i]]` |
| `EQ-A2.i` | i ∈ [0, 1] | `IS_WORD[b[i]]` |

## Constraints

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `EQ-C1.i` | i ∈ [0, 3] | `IS_HALF[diff[i]]` | μ |
| `EQ-C2` |  | `IS_BIT<invert>` |  |
| `EQ-C3` |  | `SUB<diff::DWordWL; a, b>` |  |
| `EQ-C4` |  | `ZERO[eq; diff[0] + diff[1] + diff[2] + diff[3]]` | μ |
| `EQ-C5` |  | `res` = `eq` xor `invert` |  |
| | | _polynomial:_ `res + 2 * eq * invert - eq - invert = 0` | |
| `EQ-C6` |  | `ALU[[res, 0]; a, b, ⧼EQ⧽ + 64 * invert]` | -μ |

## Padding

The chip can be padded with the following values:

| Column | Padding value |
|--------|---------------|
| `a` | `0` |
| `b` | `0` |
| `invert` | `0` |
| `res` | `0` |
| `diff` | `0` |
| `eq` | `0` |
| `μ` | `0` |