# BRANCH Chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | The current pc, used as base address when `!JALR` |
| `offset` | `Word` | The offset from the base address to jump to |
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
next_pc_unmasked (when iter=0) := 2^16 * next_pc_high[0] + 2^8 * next_pc_low[1] + unmasked_low_byte[0]
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

| Ref | Range | Description |
|-----|-------|-------------|
| `A1` | i ∈ [0, 1] | `pc` is range checked, `IS_WORD[pc[i]]` |
| `A2` |  | `offset` is range checked, `IS_WORD[offset]` |
| `A3` | i ∈ [0, 1] | `register` is range checked, `IS_WORD[register[i]]` |
| `A4` |  | `IS_BIT<JALR>` |

## Constraints

### all

| Ref | Kind | Range | Description | Multiplicity |
|-----|------|-------|-------------|--------------|
| `1` | template |  | 1 - JALR ⇒ `ADD<next_pc_unmasked; pc, offset::DWordWL>` |  |
| `2` | template |  | JALR ⇒ `ADD<next_pc_unmasked; register, offset::DWordWL>` |  |
| `3` | interaction |  | `IS_BYTE[next_pc_low[1]]` | μ |
| `4` | interaction |  | `AND_BYTE[next_pc_low[0]; unmasked_low_byte[0], 254]` | μ |
| `5` | interaction | i ∈ [0, 2] | `IS_HALFWORD[next_pc_high[i]]` | μ |

### output
_Each row contributes the following to the LogUp sum_

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `1` | interaction | `BRANCH[next_pc; pc, offset, register, JALR]` | -μ |
