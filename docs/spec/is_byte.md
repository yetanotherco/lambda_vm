# IS_BYTE Template

When a chip leverages this template twice or more, implementors are encouraged to merge pairs of  interactions with identical conditions into `ARE_BYTES` interactions; the  template is included for convenience of notation, and to complete the specification of chips that use an odd number of  range checks.

## Variables

The  template leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `X` | `BaseField` | Value for which to assert that it lies in the range $[0, 255]$. |

### Condition

| Name | Type | Description |
|------|------|-------------|
| `cond` | `BaseField` |  |

## Constraints

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `IS_BYTE-C1` | `ARE_BYTES[0, X]` | cond |