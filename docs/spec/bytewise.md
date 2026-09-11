# BYTEWISE Chip

The  chip is an ALU chip that decomposes the input `DWordWL` values into bytes and performs a `BITWISE` operation pairwise (AND, OR, XOR). The `BITWISE` lookup inherently performs a range check, so no further constraints are necessary.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `a` | `DWordBL` | The first input |
| `b` | `DWordBL` | The second input |
| `op` | `Byte` | The operation to perform |

### Output

| Name | Type | Description |
|------|------|-------------|
| `res` | `DWordBL` | The result |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` |  |

## Constraints

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `BYTEWISE-C1.i` | i ∈ [0, 7] | `BYTE_ALU[res[i]; op, a[i], b[i]]` | μ |
| `BYTEWISE-C2` |  | `ALU[res::DWordWL; a::DWordWL, b::DWordWL, op]` | -μ |

## Padding

The chip can be padded with the following values:

| Column | Padding value |
|--------|---------------|
| `a` | `0` |
| `b` | `0` |
| `op` | `0` |
| `res` | `0` |
| `μ` | `0` |