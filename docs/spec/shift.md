# SHIFT Chip

## Interface

The  chip has the following interface:

``` // param in: the value being shifted // param shift: the number of bits to shift `in` by // param direction: whether to shift left (0) or right (1) // param signed: whether to interpret `in` as a signed (1) or unsigned (0) integer // param word_instr: whether to execute the SLL/SR* (0) or SLLW/SR*W (1) instruction // out shifted: the resulting value SHIFT[shifted: DWord; in: DWord, shift: Byte, direction: Bit, signed: Bit, word_instr: Bit] ``` In other words, the  chip is designed to constrain that $

$ $

$ Here, `<<` and `>>` denote the _logical_ left and right shift operations, while `>>>` denotes the _arithmetic_ right shift operation.

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

The `SHIFT` chip is comprised of  variables that are expressed using  columns:

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `SHIFT-A1.i` | i ∈ [0, 3] | `IS_HALFWORD[in[i]]` |
| `SHIFT-A2` |  | `IS_BYTE[shift]` |
| `SHIFT-A3` |  | `IS_BIT<direction>` |
| `SHIFT-A4` |  | `IS_BIT<signed>` |
| `SHIFT-A5` |  | `IS_BIT<word_instr>` |

## Explanation

This chip has a rather complex design as a result of designing it to fit in as few columns possible. We briefly discuss the intricacies of the design, attempting to illustrate its correctness.

The chip's design revolves around a two-phase shifting process: 1. shift `in` by `x := `shift` mod 16` bits, 2. shift that result by `(`shift`-x) mod 64` (or `mod 32` if ` `word_instr` = 1`). The intermediate value representing the state between the two phases is stored in the scratch variables `X` and `Y`. The definition of `shifted` describes how one can combine the `X`, `Y` and `extension` variables to construct the output value as described using `Half`-limbs. The output variable `out` is equivalent to `shifted`, but expressed using `Word`-limbs.

In the following, we cover how these two phases were designed to complement one another. Here, we start with discussing the _logical_ left/right shift operations only; the modifications required to compute the _arithmetic_ right shift will be discussed at the end.

### First phase

We zoom in on the first step. Here, we make use of the two lookup operations - ``HWSL[x: Half, y: B4]` := (`x` `<<` `y`) mod 2^16` (short for "HalfWord Shift Left"), and - ``HWSLC[x: Half, y: B4]` := `x` `>>` (16-`y`)` (short for "HalfWord Shift Left's Carry") Note here that one can use these two lookups to compute `out: Half[4] := in << y` as: $

$ as long as ``y` < 16`. Observing that ``HWSL[x,` 16-`y]` = (`x` `<<` (16-`y`)) mod 2^16`, and ``HWSLC[x,` 16-`y]` = `x` `>>` `y`` for ``y` in [1, 15]`, one can also use these lookups to compute `out := in >> y` as $

$ as long as `0 < `y` < 16`.

Observe now that the values being looked up are (almost) independent from the direction of the shift: only the shift-amount varies slightly. When we now define $

(16-`shift`) mod 16 & "when shifting right" ), $ it only takes some rearranging and combining of the values ``X[`i`] := HWSL[in[`i`], bit_shift]`` and ``Y[`i`] := HWSLC[in[`i`], bit_shift]`` to form the limbs of ``in <</>> shift` mod 16`. In the remaining case that ``right` = 1` and ``shift` = 0 mod 16`, the limbs of ``in <</>> shift` mod 16` simply match those of `in`.

### Second phase

Since we're operating on 16-bit limbs, all the limbs in ``in <</>> shift`` must also occur somewhere in ``in <</>> shift` mod 16`. The number of full-limbs we still need to shift is determined by the fifth and sixth least significant bit of `shift`. With `limb_shift` containing a unary decoding of the integer represented by these two bits, we find that the intermediate value needs to be shifted over by `i` limbs (to the `left` or `right`) when ``limb_shift[`i`]` = 1`. These things combined yield `shifted`'s definition.

Of course, when ``word_instr` = 1` and, thus, only ``shift` mod 32` should be considered, the bit-mask for the lookup constraining `limb_shift` is adjusted appropriately (see [shift:c:limb_shift_lookup]).

### Arithmetic right shift

Lastly, we discuss the case of performing the _arithmetic_ right shift. Here, `extension` is constrained to contain a repetition of `in`'s most significant bit. Copies of this variable are used for any full limbs shifted in when ``right` = `signed` = 1`. Moreover, `X[4]` contains a copy of `extension` shifted over by the right number of bits, to allow the construction of ``in >>> shift` mod 16` as the appropriate intermediate.

## Constraints

First, we constrain `bit_shift` based on whether we are left or right-shifting. [shift:c:zbs] makes sure `zbs` is set to `1` if and only if `bit_shift = 0`. This flag is used to indicate the special case that ``right` = 1` and ``shift` = 0 mod 16`.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C3` | `AND_BYTE[bit_shift; shift, 15]` | left |
| `SHIFT-C4` | `AND_BYTE[bit_shift; 2^8 - shift, 15]` | right |
| `SHIFT-C5` | μ ⇒ `IsZero<zbs; bit_shift>` |  |

Next, we shift the limbs of `in` left and right by the appropriate amount, storing the results in `X` and `Y` respectively. When `zbs = 1`, the output cannot be used to compose ``in >>/>>> shift` mod 16`. To resolve this, we override `Y[i] := in[i]` and `X[i] := 0` in this case.

The case of `left`-shifting and ``bit_shift` = 0` will be used for padding rows. To prevent unnecessary lookups in padding rows, we override ``X[i]` := `in[i]`` and ``Y[i]` := 0` here.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C6.i` | i ∈ [0, 3] | `HWSL[X[i]; in[i], bit_shift]` | 1 - zbs |
| `SHIFT-C7.i` | i ∈ [0, 3] | `zbs` => `X[i]` = `in[i]` dot `left` |  |
| | | _polynomial:_ `zbs * (X[i] - in[i] * left) = 0` | |
| `SHIFT-C8` |  | `HWSL[X[4]; extension, bit_shift]` | 1 - zbs |
| `SHIFT-C9` |  | `zbs` => `X[4]` = 0 |  |
| | | _polynomial:_ `zbs * X[4] = 0` | |
| `SHIFT-C10.i` | i ∈ [0, 3] | `HWSLC[Y[i]; in[i], bit_shift]` | 1 - zbs |
| `SHIFT-C11.i` | i ∈ [0, 3] | `zbs` => `Y[i]` = `in[i]` dot `right` |  |
| | | _polynomial:_ `zbs * (Y[i] - in[i] * right) = 0` | |

### Full-limb shifting

Next, we constrain that `limb_shift` is a proper unary encoding of the fifth (and sixth if ``word_instr` = 0`) bit of `shift`. For this to be the case, three requirements must be satisfied: + *unary(0)*: ``limb_shift[`i`]` in {0, 1}` for `i in [0, 3]`, + *unary(1)*: ``limb_shift[`i`]` = 1` for exactly one `i`, and + *proper encoding*: ``limb_shift[`i`]` = 1 <=> 1/16 (`shift &` (48-32 dot `word_instr`)) = i` The first requirement is enforced by constraint [shift:c:limb_shift_is_bit]. To construct a constraint for the second and third requirement, observe that $ 1/16 dot (`shift &` (48-32 dot `word_instr`)) in cases( {0, 1, 2, 3} &"if" `word_instr` = 0, {0, 1} &"if" `word_instr` = 1 $ Observe moreover that, assuming *unary(0)*, the expression $ 1/16 dot (1 + sum_(i=0)^3 (16i-1) dot `limb_shift[`i`]`) $ can evaluate to `i` if and only if ``limb_shift[`i`]` = 1`, while the others are `0`. This means that the relation $ 1 + sum_(i=0)^3 (16i-1) dot `limb_shift[`i`]` = `shift &` (48-32 dot `word_instr`) $ enforces both *unary(1)* and *proper encoding*. This is the exact relation [shift:c:limb_shift_lookup] enforces.

Hereafter, one must only check that `out` is the proper cast of `shifted` into a `DWordWL`.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C12.i` | i ∈ [0, 3] | `IS_BIT<limb_shift[i]>` |  |
| `SHIFT-C13` |  | `AND_BYTE[(1 - limb_shift[0]) + 15 * limb_shift[1] + 31 * limb_shift[2] + 47 * limb_shift[3]; shift, 48 - 32 * word_instr]` | μ |
| `SHIFT-C14.i` | i ∈ [0, 1] | `out[:2]` = `shifted[:4]` |  |
| | | _polynomial:_ `out[i] - (shifted::DWordWL)[i] = 0` | |

### Miscellaneous

*Note*: `is_negative` is not used when `signed = 0`. As such, there is no problem with it being unconstrained in this case.

### Lookups

This chip adds the following interaction to the lookup.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C15` | `SHIFT[out; in, shift, direction, signed, word_instr]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

### left_flag

| Tag | Description |
|-----|-------------|
| `SHIFT-C1` | `direction` => `μ` = 1 |
| | _polynomial:_ `direction * (1 - μ) = 0` |

### is_negative

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C2` | `MSB16[is_negative; in[3]]` | signed |