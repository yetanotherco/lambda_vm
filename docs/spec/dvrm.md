# DVRM Chip

The  chip provides division and remainder functionality, both signed and unsigned.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `n` | `DWordHL` | The numerator |
| `d` | `DWordHL` | The denominator |
| `signed` | `Bit` | Whether to interpret the input as signed (1) or unsigned (0) integers. |

### Output

| Name | Type | Description |
|------|------|-------------|
| `q` | `DWordHL` | The quotient; $`n` / `d`$ rounded towards zero. |
| `r` | `DWordHL` | The remainder; $`n` - `q` `d`$. |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `div_by_zero` | `Bit` | Whether $`d`=0$. |
| `overflow` | `Bit` | Whether $`n` = -2^63$ and $`d`=-1$. |
| `abs_r` | `DWordWL` | Absolute value of `r`. |
| `abs_d` | `DWordWL` | Absolute value of `d`. |
| `n_sub_r` | `DWordHL` | $`n`-`r`$. |
| `sign_n_sub_r` | `Bit` | Sign of `n_sub_r`. |
| `sign_n` | `Bit` | Sign of `n`. |
| `sign_d` | `Bit` | Sign of `d`. |
| `sign_q` | `Bit` | Sign of `q`. |
| `sign_r` | `Bit` | Sign of `r`. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `extended_n` | `QuadHL` | sign-extended value of `n`. |
| `extended_r` | `QuadHL` | sign-extended value of `r`. |
| `extension_n_sub_r` | `DWordHL` | sign-extension limbs of `n_sub_r`. |
| `extended_n_sub_r` | `QuadHL` | sign-extended value of `n_sub_r`. |
| `carry` | `Bit[4]` | carries for adding `extended_n_sub_r` to `extended_r`, forming `extended_n`. |
| `μ_sum` | `BaseField` | sum of multiplicities |

**Definition of `extended_n`:**
```
extended_n (when iter=[0, 3]) := n[i]
extended_n (when iter=[4, 7]) := 65535 * sign_n
```

**Definition of `extended_r`:**
```
extended_r (when iter=[0, 3]) := r[i]
extended_r (when iter=[4, 7]) := 65535 * sign_r
```

**Definition of `extension_n_sub_r`:**
```
extension_n_sub_r := 65535 * sign_n_sub_r
```

**Definition of `extended_n_sub_r`:**
```
extended_n_sub_r (when iter=[0, 3]) := n_sub_r[i]
extended_n_sub_r (when iter=[4, 7]) := extension_n_sub_r[i - 4]
```

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * ((extended_n_sub_r::QuadWL)[i] + (extended_r::QuadWL)[i] - (extended_n::QuadWL)[i])
carry (when iter=[1, 3]) := 2^-32 * ((extended_n_sub_r::QuadWL)[i] + (extended_r::QuadWL)[i] + carry[i - 1] - (extended_n::QuadWL)[i])
```

**Definition of `μ_sum`:**
```
μ_sum := μ_q + μ_r
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ_q` | `BaseField` |  |
| `μ_r` | `BaseField` |  |

## Constraints

First, we range-check all inputs.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C1.i` | i ∈ [0, 3] | `IS_HALF[n[i]]` | μ_sum |
| `DVRM-C2.i` | i ∈ [0, 3] | `IS_HALF[d[i]]` | μ_sum |
| `DVRM-C3` |  | `IS_BIT<signed>` |  |

From the ISA, we gather five requirements for the `DIV[U][W]` and `REM[U][W]` instructions:

enum.item([ _For both signed and unsigned division, except in the case of_ overflow, _it holds that ``n` = `q` `d` + `r``._ ]), enum.item([ _`DIV` and `DIVU` perform [...] signed and unsigned integer division [...] rounding towards zero._ ]), enum.item([ _For `REM`, the sign of a nonzero [remainder] equals the sign of the [numerator]._ ]), enum.item([ In case of _division-by-zero_, ``r` = `n`` and ``q` = 2^64-1` (unsigned) or ``q` = -1` (signed). ]), enum.item([ In case of _overflow_, ``q` = `n`` and ``r` = 0` ]), where _overflow_ occurs when ``n` = -2^(63)` and ``d` = -1` (and, hence, ``signed` = 1`), and _division-by-zero_ indicates that ``d` = 0`. In the following, we list the constraints associated with the  chip, and explain how these together enforce all five of these requirements.

### R3: Sign remainder equals sign numerator

We start with R3, which is straightforwardly asserted by constraint [dvrm:c:sign_r_equals_sign_n].

| Tag | Description |
|-----|-------------|
| `DVRM-C4` | `r` eq.not 0 => `sign_r` = `sign_n` |
| | _polynomial:_ `Σ_i = 0^3 r[i] * (sign_r - sign_n) = 0` |

### R2: rounding towards zero

R2 states that "_[in] signed and unsigned integer division [the quotient is] round[ed] towards zero._" In other words, + the sign of ``n`-`qd`` must match that of `n` (unless ``qd` = `n``), and + `|`n`-`qd`|  < |`d`|` (unless ``d` = 0`).

Leveraging R1 , we can rewrite these as + the sign of ``r`` must match that of `n` (unless ``r` = 0`), and + `|`r`|  < |`d`|` (unless ``d` = 0`).

Focusing on the first statement, we observe that this trivially holds when ``signed` = 0`, while R3 deals with the case that ``signed` = 1`. The second statement is enforced by [dvrm:c:abs_r_lt_abs_d]. [dvrm:c:abs_r_if_negative] and [dvrm:c:abs_r_if_nonnegative] (resp. [dvrm:c:abs_d_if_negative] and [dvrm:c:abs_d_if_nonnegative]) are included to ensure that `abs_r` (resp. `abs_d`) is the absolute values of `r` (resp. `d`).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C5` |  | `ALU[[1 - div_by_zero, 0]; abs_r, abs_d, ⧼LT⧽]` | μ_sum |
| `DVRM-C6` |  | sign_r ⇒ `NEG<abs_r; r>` |  |
| `DVRM-C7.i` | i ∈ [0, 1] | not`sign_r` => `abs_r` = `r` |  |
| | | _polynomial:_ `(1 - sign_r) * (abs_r[i] - (r::DWordWL)[i]) = 0` | |
| `DVRM-C8` |  | sign_d ⇒ `NEG<abs_d; d>` |  |
| `DVRM-C9.i` | i ∈ [0, 1] | not`sign_d` => `abs_d` = `d` |  |
| | | _polynomial:_ `(1 - sign_d) * (abs_d[i] - (d::DWordWL)[i]) = 0` | |

### R5: overflow

The ISA requires that ``q` = `n`` and ``r` = 0` in the event of overflow (i.e., when ``n` = -2^63` and ``d` = -1`). We note that the second half of this requirement is already satisfied by R2: since ``d` = -1 != 0`, R2 requires that `|`r`| < |`d`| = 1`, to which ``r` = 0` is the only satisfying value.

We moreover find that R1 can be leveraged to enforce the correct value of `q`. While ``n` = `qd` + `r`` (R1) does _not_ hold in the case of overflow, the relation ``n` = |`q`|`d` + `r`` _does_. We moreover note that the 64-bit _signed_ two's complement representation of `-2^63` is identical to the 64-bit _unsigned_ representation of `|-2^63| = 2^63`. As such, by interpreting `q` as an unsigned integer when ``overflow` = 1`, it follows that R1 will enforce ``q` = `0x80...00``.

In summary, in case of overflow R2 enforces that ``r` = 0`. Moreover it suffices to interpret `q` as unsigned integer ([dvrm:c:sign_q]); R1 will ensure it contains the correct value.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `DVRM-C10` | `sign_q` = `signed` dot (1- `overflow`) |  |
| | _polynomial:_ `signed * (1 - overflow) - sign_q = 0` | |
| `DVRM-C11` | `ZERO[overflow; n[0] + n[1] + n[2] + (n[3] - 2^15 * sign_n) + (1 - sign_n) + (65535 - d[0]) + (65535 - d[1]) + (65535 - d[2]) + (65535 - d[3])]` | μ_sum |

We highlight [dvrm:c:overflow]. Recall that the `overflow` flag should be set if and only if (i) ``signed` = 1`, (ii) ``n` = `0x80...00``, and (iii) ``d` = `0xFF...FF``. These requirements are equivalent to the state where: $ forall i in [0, 3]:&& 65535 - `d`_i &= 0,\ forall i in [0, 2]:&& `n`_i &= 0,\ && `n`_3 - 2^15 dot `sign_n` &= 0,\ && 1 - `sign_n` &= 0,\ $ where ``signed` = 1` follows from the last equality. The requirement is phrased in this way, because the left-hand sides of the above expressions are `>= 0` by construction. Given that the sum of these expressions does not exceed `2^19` (and thus never wraps in the field), we can now say that the `overflow` bit should be set to `1` if and only if their sum evaluates to `0`. The `ZERO` lookup guarantees this to be the case.

### R1: $#`n` = #`qd` + #`r`$

Rewriting R1, we find the constraint `not`overflow` => `n` - `r` = `qd``.

Since `n`, `d`, `q` and `r` are all 64-bit integers, we must assert this equality `mod 2^128`, rather than `mod 2^64`. To this end, we introduce `extended_n_sub_r` and leverage the `MUL` chip to verify that it is equal to ``qd` mod 2^128` using constraints [dvrm:c:mul_lower] and [dvrm:c:mul_upper]; [dvrm:c:q_range] is included to uphold assumption [mul:c:rhs].

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C12` |  | `ALU[n_sub_r::DWordWL; d::DWordWL, q::DWordWL, ⧼MUL⧽ + 32 * signed + 64 * sign_q]` | μ_sum |
| `DVRM-C13` |  | `ALU[extension_n_sub_r::DWordWL; d::DWordWL, q::DWordWL, ⧼MUL⧽ + 32 * signed + 64 * sign_q + 128]` | μ_sum |
| `DVRM-C14.i` | i ∈ [0, 3] | `IS_HALF[q[i]]` | μ_sum |

It now remains to enforce that `extended_n_sub_r` is the _signed_ 128-bit representation of ``n`-`r``. Here, we introduce `extended_n` and `extended_r`. By their definition, these variables contain the signed 128-bit representations of `n` and `r`. The `carry` variable has been defined such that it mimics those in the `ADD` chip, except that here we add two `QuadHL`s rather than two `DWordHL`, thus needing four carry bits instead of two. With this in place, [dvrm:c:n_sub_r] (mimicking [add:c:carry]) ensures `extended_n_sub_r` must contain the correct value.

Lastly, observe that ``n` - `r` in (-2^64, 2^64)`, _regardless_ of the value of `signed`. Moreover, note that the upper halves of the 128-bit representations of all values in this range are either `0xFFFFFFFF` (negative) or `0x00000000` (non-negative). This means that we do not need to store all 128 bits of `extended_n_sub_r`. Rather, we need only store the lower 64-bits, and a separate bit (`sign_n_sub_r`) indicating whether the top limbs are all-ones or all-zeroes. The prover is free to select the value for `sign_n_sub_r`; only one of the two will fit the proof.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C15.i` | i ∈ [0, 3] | `IS_BIT<carry[i]>` |  |
| `DVRM-C16.i` | i ∈ [0, 3] | `IS_HALF[r[i]]` | μ_sum |
| `DVRM-C17.i` | i ∈ [0, 3] | `IS_HALF[n_sub_r[i]]` | μ_sum |
| `DVRM-C18` |  | `IS_BIT<sign_n_sub_r>` |  |

### R4: division-by-zero

R4 requires that ``q` = 2^64-1` (unsigned) or `-1` (signed) and ``r` = n` when ``d` = 0`. Recalling R1, we see that ``n` = `q` `d` + `r` = `r`` when ``d` = 0`, already enforces the latter. Next, we note that, in two's complement, the _unsigned_ value `2^64-1` and _signed_ value `-1` are both represented by the bit string `0xFFFFFFFF`. Hence, only [dvrm:c:q_if_div_by_zero] is required to completely constrain R4; [dvrm:c:div_by_zero] just ensures the `div_by_zero` flag is set when ``d` = 0`.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C19.i` | i ∈ [0, 3] | `div_by_zero` => `q[i]` = 65535 |  |
| | | _polynomial:_ `div_by_zero * (q[i] - 65535) = 0` | |
| `DVRM-C20` |  | `ZERO[div_by_zero; d[0] + d[1] + d[2] + d[3]]` | μ_sum |

### Other

The following constraints are included to enforce the values of `sign_n`, `sign_r` and `sign_d` are correct.

| Tag | Description |
|-----|-------------|
| `DVRM-C21` | `SIGN<sign_n; n[3], signed>` |
| `DVRM-C22` | `SIGN<sign_r; r[3], signed>` |
| `DVRM-C23` | `SIGN<sign_d; d[3], signed>` |

### Output

Lastly, this chip contributes the following to the lookup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `DVRM-C24` | `ALU[q::DWordWL; n::DWordWL, d::DWordWL, ⧼DIVREM⧽ + 32 * signed]` | -μ_q |
| `DVRM-C25` | `ALU[r::DWordWL; n::DWordWL, d::DWordWL, ⧼DIVREM⧽ + 32 * signed + 128]` | -μ_r |

## Padding

To pad the  table, we use the following data, representing the unsigned division `frac(0, 0, style: "horizontal")`:

| Column | Padding value |
|--------|---------------|
| `n` | `0` |
| `d` | `0` |
| `signed` | `0` |
| `q` | `0` |
| `r` | `0` |
| `div_by_zero` | `1` |
| `overflow` | `0` |
| `abs_r` | `0` |
| `abs_d` | `0` |
| `n_sub_r` | `0` |
| `sign_n_sub_r` | `0` |
| `sign_n` | `0` |
| `sign_d` | `0` |
| `sign_q` | `0` |
| `sign_r` | `0` |
| `μ_q` | `0` |
| `μ_r` | `0` |