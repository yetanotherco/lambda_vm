# SHIFT Chip

The  chip is designed to constrain that $

$ $

$ Here, `<<` and `>>` denote the _logical_ left and right shift operations, while `>>>` denotes the _arithmetic_ right shift operation.

## Variables

The `SHIFT` chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `in` | `DWordHL` | The value being shifted |
| `shift` | `DWordWHBB` | Number of bits to shift `in` by. |
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
| `limb_shift_raw` | `Bit[3]` | One-hot vector indicating whether $floor.l `shift` / 16 floor.r equiv i mod s$, where $s = 2$ when $`word_instr` = 1$ and $4$ otherwise. These columns store the first 3 values, and the 4th is derived from the one-hot property. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `limb_shift` | `Bit[4]` |  |
| `extension` | `Half` | sign extension of `in`. |
| `left` | `Bit` | Whether to perform a left-shift. |
| `right` | `Bit` | Whether to perform a right-shift. |
| `intra_limb_left` | `DWordHL` | `in << (shift % 16)` if `left` |
| `intra_limb_right` | `DWordHL` | `in >>> (shift % 16)` if `right` and `signed`;\ `in >> (shift % 16)` if `right` and `!signed` |
| `shifted` | `DWordHL` | $`in <</>>/>>>` (`shift` mod 32 dot (2 - `word_instr`))$ |

**Definition of `limb_shift`:**
```
limb_shift (when iter=[0, 2]) := limb_shift_raw[i]
limb_shift (when iter=3) := 1 - Σ_j = 0^2 limb_shift_raw[j]
```

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
shifted := left * Σ_j = 0^i limb_shift[j] * intra_limb_left[i - j] + right * (Σ_j = 0^3 - i limb_shift[j] * intra_limb_right[i + j] + extension * Σ_j = 4 - i^3 limb_shift[j])
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Explanation

This chip has a rather complex design as a result of designing it to fit in as few columns possible. We briefly discuss the intricacies of the design, attempting to illustrate its correctness.

The chip's design revolves around a two-phase shifting process: 1. shift `in` by `x := `shift` mod 16` bits, 2. shift that result by `(`shift`-x) mod 64` (or `mod 32` if ` `word_instr` = 1`). The intermediate value representing the state between the two phases is stored in the scratch variables `X` and `Y`. The definition of `shifted` describes how one can combine the `X`, `Y` and `extension` variables to construct the output value as described using `Half`-limbs. The output variable `out` is equivalent to `shifted`, but expressed using `Word`-limbs.

In the following, we cover how these two phases were designed to complement one another. Here, we start with discussing the _logical_ left/right shift operations only; the modifications required to compute the _arithmetic_ right shift will be discussed at the end.

### First phase

We zoom in on the first step. Here, we make use of the lookup operation `HWSL` (short for "HalfWord Shift Left"): ` `HWSL[x: Half, y: B4]` := [(`x` `<<` `y`) mod 2^16, `x` `>>` (16 - `y`)]. ` One can use this to compute `out: Half[4] := in << y` as: $

$ as long as ``y` < 16`. Observing that ``HWSL[x,` 16-`y]`_0 = (`x` `<<` (16-`y`)) mod 2^16`, and ``HWSL[x,` 16-`y]`_1 = `x` `>>` `y`` for ``y` in [1, 15]`, one can also use it to compute `out := in >> y` as $

$ as long as `0 < `y` < 16`.

Observe now that the values being looked up are (almost) independent from the direction of the shift: only the shift-amount varies slightly. When we now define $

(16-`shift`) mod 16 & "when shifting right" ), $ it only takes some rearranging and combining of the values ``X[`i`] := HWSL[in[`i`], bit_shift]`_0` and ``Y[`i`] := HWSL[in[`i`], bit_shift]`_1` to form the limbs of ``in <</>> shift` mod 16`. In the remaining case that ``right` = 1` and ``shift` = 0 mod 16`, the limbs of ``in <</>> shift` mod 16` simply match those of `in`.

### Second phase

Since we're operating on 16-bit limbs, all the limbs in ``in <</>> shift`` must also occur somewhere in ``in <</>> shift` mod 16`. The number of full-limbs we still need to shift is determined by the fifth and sixth least significant bit of `shift`. With `limb_shift` containing a unary decoding of the integer represented by these two bits, we find that the intermediate value needs to be shifted over by `i` limbs (to the `left` or `right`) when ``limb_shift[`i`]` = 1`. These things combined yield `shifted`'s definition.

Of course, when ``word_instr` = 1` and, thus, only ``shift` mod 32` should be considered, the bit-mask for the lookup constraining `limb_shift` is adjusted appropriately (see [shift:c:limb_shift_lookup]).

### Arithmetic right shift

Lastly, we discuss the case of performing the _arithmetic_ right shift. Here, `extension` is constrained to contain a repetition of `in`'s most significant bit. Copies of this variable are used for any full limbs shifted in when ``right` = `signed` = 1`. Moreover, `X[4]` contains a copy of `extension` shifted over by the right number of bits, to allow the construction of ``in >>> shift` mod 16` as the appropriate intermediate.

## Constraints

First, we range check our inputs appropriately.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C1.i` | i ∈ [0, 3] | `IS_HALF[in[i]]` | μ |
| `SHIFT-C2` |  | `IS_HALF[shift[2]]` | μ |
| `SHIFT-C3.i` | i ∈ [0, 1] | `IS_BYTE<shift[i]>` |  |
| `SHIFT-C4` |  | `IS_BIT<direction>` |  |
| `SHIFT-C5` |  | `IS_BIT<signed>` |  |
| `SHIFT-C6` |  | `IS_BIT<word_instr>` |  |

Then, we constrain `bit_shift` based on whether we are left or right-shifting. [shift:c:zbs] makes sure `zbs` is set to `1` if and only if `bit_shift = 0`. This flag is used to indicate the special case that ``right` = 1` and ``shift` = 0 mod 16`.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C7` | `BYTE_ALU[bit_shift; ⧼AND⧽, shift[0], 15]` | left |
| `SHIFT-C8` | `BYTE_ALU[bit_shift; ⧼AND⧽, 2^8 - 16 * zbs - shift[0], 15]` | right |
| `SHIFT-C9` | `ZERO[zbs; bit_shift]` | μ |

Next, we shift the limbs of `in` left and right by the appropriate amount, storing the results in `X` and `Y` respectively. When `zbs = 1`, the output cannot be used to compose ``in >>/>>> shift` mod 16`. To resolve this, we override `Y[i] := in[i]` and `X[i] := 0` in this case.

The case of `left`-shifting and ``bit_shift` = 0` will be used for padding rows. To prevent unnecessary lookups in padding rows, we override ``X[i]` := `in[i]`` and ``Y[i]` := 0` here.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C10.i` | i ∈ [0, 3] | `HWSL[[X[i], Y[i]]; in[i], bit_shift]` | 1 - zbs |
| `SHIFT-C11.i` | i ∈ [0, 3] | `zbs` => `X[i]` = `in[i]` dot `left` |  |
| | | _polynomial:_ `zbs * (X[i] - in[i] * left) = 0` | |
| `SHIFT-C12.i` | i ∈ [0, 3] | `zbs` => `Y[i]` = `in[i]` dot `right` |  |
| | | _polynomial:_ `zbs * (Y[i] - in[i] * right) = 0` | |
| `SHIFT-C13` |  | `HWSL[[X[4], extension - X[4]]; extension, bit_shift]` | 1 - zbs |
| `SHIFT-C14` |  | `zbs` => `X[4]` = 0 |  |
| | | _polynomial:_ `zbs * X[4] = 0` | |

### Full-limb shifting

Next, we constrain that `limb_shift` is a proper unary encoding of the fifth (and sixth if ``word_instr` = 0`) bit of `shift`. For this to be the case, three requirements must be satisfied: + *unary(0)*: ``limb_shift[`i`]` in {0, 1}` for `i in [0, 3]`, + *unary(1)*: ``limb_shift[`i`]` = 1` for exactly one `i`, and + *proper encoding*: ``limb_shift[`i`]` = 1 <=> 1/16 (`shift &` (48-32 dot `word_instr`)) = i` The first requirement is enforced by constraint [shift:c:limb_shift_is_bit]. To construct a constraint for the second and third requirement, observe that $ 1/16 dot (`shift &` (48-32 dot `word_instr`)) in cases( {0, 1, 2, 3} &"if" `word_instr` = 0, {0, 1} &"if" `word_instr` = 1 $ Observe moreover that, assuming *unary(0)*, the expression $ 1/16 dot (1 + sum_(i=0)^3 (16i-1) dot `limb_shift[`i`]`) $ can evaluate to `i` if and only if ``limb_shift[`i`]` = 1`, while the others are `0`. This means that the relation $ 1 + sum_(i=0)^3 (16i-1) dot `limb_shift[`i`]` = `shift &` (48-32 dot `word_instr`) $ enforces both *unary(1)* and *proper encoding*. This is the exact relation [shift:c:limb_shift_lookup] enforces.

Hereafter, one must only check that `out` is the proper cast of `shifted` into a `DWordWL`.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C15.i` | i ∈ [0, 3] | `IS_BIT<limb_shift[i]>` |  |
| `SHIFT-C16` |  | `BYTE_ALU[(1 - limb_shift[0]) + 15 * limb_shift[1] + 31 * limb_shift[2] + 47 * limb_shift[3]; ⧼AND⧽, shift[0], 48 - 32 * word_instr]` | μ |
| `SHIFT-C17.i` | i ∈ [0, 1] | `out[:2]` = `shifted[:4]` |  |
| | | _polynomial:_ `out[i] - (shifted::DWordWL)[i] = 0` | |

### Miscellaneous

| Tag | Description |
|-----|-------------|
| `SHIFT-C18` | `direction` => `μ` = 1 |
| | _polynomial:_ `direction * (1 - μ) = 0` |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C19` | `MSB16[is_negative; in[3]]` | signed |

*Note*: `is_negative` is not used when `signed = 0`. As such, there is no problem with it being unconstrained in this case.

### Lookups

This chip adds the following interaction to the lookup.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C20` | `ALU[out; in::DWordWL, shift::DWordWL, ⧼SHIFT⧽ + word_instr + 32 * signed + 64 * direction]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `in` | `0` |
| `shift` | `0` |
| `direction` | `0` |
| `signed` | `0` |
| `word_instr` | `0` |
| `out` | `0` |
| `is_negative` | `0` |
| `bit_shift` | `0` |
| `zbs` | `1` |
| `X` | `[0, 0, 0, 0, 0]` |
| `Y` | `[0, 0, 0, 0]` |
| `limb_shift_raw` | `[0, 0, 0]` |
| `μ` | `0` |