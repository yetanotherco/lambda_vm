# BRANCH Chip

The  chip computes the target address of a branching instruction.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | The current pc, used as base address when `!JALR` |
| `offset` | `DWordWL` | The offset from the base address to jump to |
| `register` | `DWordWL` | The base address to use when `JALR` |
| `JALR` | `Bit` | Selects between `pc` and `register` as base address, needed for the `JALR` instruction |

### Output

| Name | Type | Description |
|------|------|-------------|
| `next_pc_high` | `Half[3]` | The upper part of the next pc |
| `next_pc_low` | `Byte[2]` | The lower part of the next pc |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `unmasked_low_byte` | `Byte` | The low byte of the next pc, before masking the LSB. Used to constraint the raw addition. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `next_pc_unmasked` | `DWordWL` | The combination of `next_pc_high`, `next_pc_low[1]` and `unmasked_low_byte` to constrain the addition. This is the computed value for the next pc, before masking off the LSB as required by the ISA. |
| `next_pc` | `DWordWL` | The computed next pc, after masking off the LSB as required by the ISA. |

**Definition of `next_pc_unmasked`:**
```
next_pc_unmasked (when iter=0) := 2^16 * next_pc_high[0] + 2^8 * next_pc_low[1] + unmasked_low_byte
next_pc_unmasked (when iter=1) := 2^16 * next_pc_high[2] + next_pc_high[1]
```

**Definition of `next_pc`:**
```
next_pc (when iter=0) := 2^16 * next_pc_high[0] + 2^8 * next_pc_low[1] + next_pc_low[0]
next_pc (when iter=1) := 2^16 * next_pc_high[2] + next_pc_high[1]
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `BRANCH-A1.i` | i ∈ [0, 1] | `pc` is range checked, `IS_WORD[pc[i]]` |
| `BRANCH-A2` |  | `offset` is range checked, `IS_WORD[offset]` |
| `BRANCH-A3.i` | i ∈ [0, 1] | `register` is range checked, `IS_WORD[register[i]]` |
| `BRANCH-A4` |  | `IS_BIT<JALR>` |

Some of the assumptions can be checked with only arithmetic constraints, so we provide these below.

| Tag | Description |
|-----|-------------|
| `BRANCH-C1` | `IS_BIT<JALR>` |

## Constraints

We constrain `next_pc` to be ``base_address` + `offset``, where `base_address` equals `pc` when ``JALR` = 0` and `register` otherwise.

The range checks on `unmasked_low_byte` and `next_pc_low[0]` are performed implicitly by the `AND_BYTE` lookup.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `BRANCH-C2` |  | 1 - JALR ⇒ `ADD<next_pc_unmasked; pc, offset::DWordWL>` |  |
| `BRANCH-C3` |  | JALR ⇒ `ADD<next_pc_unmasked; register, offset::DWordWL>` |  |
| `BRANCH-C4` |  | μ ⇒ `IS_BYTE<next_pc_low[1]>` |  |
| `BRANCH-C5` |  | `BYTE_ALU[next_pc_low[0]; ⧼AND⧽, unmasked_low_byte, 254]` | μ |
| `BRANCH-C6.i` | i ∈ [0, 2] | `IS_HALF[next_pc_high[i]]` | μ |

This chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `BRANCH-C7` | `BRANCH[next_pc; pc, offset, register, JALR]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `pc` | `0` |
| `offset` | `0` |
| `register` | `0` |
| `JALR` | `0` |
| `next_pc_high` | `[0, 0, 0]` |
| `next_pc_low` | `0` |
| `unmasked_low_byte` | `0` |
| `μ` | `0` |