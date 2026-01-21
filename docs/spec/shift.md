# SHIFT Chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `in` | `DWordHL` | The value being shifted |
| `shift` | `Byte` | Number of bits to shift `in` by. |
| `direction` | `Bit` | Whether to shift left (0) or right (1). |
| `signed` | `Bit` | Whether to interpret `in` as a signed integer. |
| `word_instr` | `Bit` | Whether this is a Word-instruction (1) or not (0). |

### Output

| Name | Type | Description |
|------|------|-------------|
| `out` | `DWordWL` | $`in <</>>/>>>` (`shift` mod 32 dot (2 - `word_instr`))$ |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `is_negative` | `Bit` | Whether `in` is negative |
| `bit_shift` | `Byte` | Value by which to shift `in` to obtain `X` and `Y` |
| `zbs` | `Bit` | Whether `bit_shift` is zero (1) or not (0). |
| `X` | `Half[5]` | scratch variable. |
| `Y` | `Half[4]` | scratch variable. |
| `limb_shift` | `Bit[4]` | One-hot vector indicating whether $floor.l `shift` / 16 floor.r equiv i mod s$, where $s = 2$ when $`word_instr` = 1$ and $4$ otherwise. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `extension` | `Half` | sign extension of `in`. |
| `left` | `Bit` | Whether to perform a left-shift. |
| `right` | `Bit` | Whether to perform a right-shift. |
| `intra_limb_left` | `DWordHL` | `in << (shift % 16)` if `left` |
| `intra_limb_right` | `DWordHL` | `in >>> (shift % 16)` if `right` and `signed`;\ `in >> (shift % 16)` if `right` and `!signed` |
| `shifted` | `DWordHL` | $`in <</>>/>>>` (`shift` mod 32 dot (2 - `word_instr`))$ |

**Definition of `extension`:**
```
extension := 65535 * is_negative
```

**Definition of `left`:**
```
left := μ - direction
```

**Definition of `right`:**
```
right := direction
```

**Definition of `intra_limb_left`:**
```
intra_limb_left (when iter=0) := X[0]
intra_limb_left (when iter=[1, 3]) := X[i] + Y[i - 1]
```

**Definition of `intra_limb_right`:**
```
intra_limb_right := Y[i] + X[i + 1]
```

**Definition of `shifted`:**
```
shifted := left * Σ_j = 0^i limb_shift[j] * intra_limb_left[i - j] + right * (Σ_j = 0^3 - i limb_shift[j] * intra_limb_right[i + j] + extension * Σ_j = 3 - i^3 limb_shift[j])
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

| Ref | Range | Description |
|-----|-------|-------------|
| `shift:a:range_in` | i ∈ [0, 3] | `IS_HALFWORD[in[i]]` |
| `shift:a:range_shift` |  | `IS_BYTE[shift]` |
| `shift:a:direction` |  | `IS_BIT<direction>` |
| `shift:a:signed` |  | `IS_BIT<signed>` |
| `shift:a:word_instr` |  | `IS_BIT<word_instr>` |

## Constraints

### left_flag

| Ref | Kind | Description |
|-----|------|-------------|
| `shift:c:direction_implies_mu` | arith | `direction` => `μ` = 1 |
| | | _polynomial:_ `direction * (1 - μ) = 0` |
| | | _note:_ enforces `left` is `Bit`. |

### is_negative

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `shift:c:is_negative_if_signed` | interaction | `MSB16[is_negative; in[3]]` | signed |

### bit_shift

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `shift:c:bit_shift_if_left` | interaction | `AND_BYTE[bit_shift; shift, 15]` | left |
| `shift:c:bit_shift_if_right` | interaction | `AND_BYTE[bit_shift; 2^8 - shift, 15]` | right |
| `shift:c:zbs` | template | `IsZero<zbs; bit_shift>` | μ |

### intra_limb_shift

| Ref | Kind | Range | Description | Multiplicity |
|-----|------|-------|-------------|--------------|
| `shift:c:hwsl_if_not_zero` | interaction | i ∈ [0, 3] | `HWSL[X[i]; in[i], bit_shift]` | 1 - zbs |
| `shift:c:zbs_implies_X` | arith | i ∈ [0, 3] | `zbs` => `X[i]` = `in[i]` dot `left` |  |
| | | _polynomial:_ `zbs * (X[i] - in[i] * left) = 0` | |
| `shift:c:hwsl_x4_if_not_zero` | interaction |  | `HWSL[X[4]; extension, bit_shift]` | 1 - zbs |
| `shift:c:zbs_implies_X_4` | arith |  | `zbs` => `X[4]` = 0 |  |
| | | _polynomial:_ `zbs * X[4] = 0` | |
| `shift:c:hwslc_if_not_zero` | interaction | i ∈ [0, 3] | `HWSLC[Y[i]; in[i], bit_shift]` | 1 - zbs |
| `shift:c:zbs_implies_Y` | arith | i ∈ [0, 3] | `zbs` => `Y[i]` = `in[i]` dot `right` |  |
| | | _polynomial:_ `zbs * (Y[i] - in[i] * right) = 0` | |

### limb_shifting

| Ref | Kind | Range | Description | Multiplicity |
|-----|------|-------|-------------|--------------|
| `shift:c:limb_shift_is_bit` | template | i ∈ [0, 3] | `IS_BIT<limb_shift[i]>` |  |
| `shift:c:limb_shift_lookup` | interaction |  | `AND_BYTE[(1 - limb_shift[0]) + 15 * limb_shift[1] + 31 * limb_shift[2] + 47 * limb_shift[3]; shift, 48 - 32 * word_instr]` | μ |
| `shift:c:out_eq_shifted` | arith | i ∈ [0, 1] | `out[:2]` = `shifted[:4]` |  |
| | | _polynomial:_ `out[i] - (shifted::DWordWL)[i] = 0` | |

### lookups

| Ref | Kind | Description | Multiplicity |
|-----|------|-------------|--------------|
| `shift:c:lookup` | interaction | `SHIFT[out; in, shift, direction, signed, word_instr]` | -μ |
