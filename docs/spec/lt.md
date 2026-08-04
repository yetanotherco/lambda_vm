# LT Chip

The  chip constrains an indicator bit for the less-than relation, signed or unsigned. If the `invert` flag is set, it inverts the result.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `DWordHHW` | The left operand |
| `rhs` | `DWordHHW` | The right operand |
| `signed` | `Bit` | whether to interpret `lhs` and `rhs` as signed integers (1) or not (0) |
| `invert` | `Bit` | Whether to invert the result |

### Output

| Name | Type | Description |
|------|------|-------------|
| `res` | `Bit` | The result |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `lhs_sub_rhs` | `DWordHL` | $`lhs` - `rhs`$ |
| `lhs_msb` | `Bit` | The most significant bit of `lhs` |
| `rhs_msb` | `Bit` | The most significant bit of `rhs` |
| `lt` | `Bit` | Whether $`lhs` < `rhs`$, taking `signed` into account |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `carry` | `Bit[2]` | The carry for adding `lhs_sub_rhs` back to `rhs` |
| `unsigned_lt` | `Bit` | Whether $`lhs` < `rhs`$, as unsigned integers |

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * (rhs[0] + (lhs_sub_rhs::DWordWL)[0] - lhs[0])
carry (when iter=1) := 2^-32 * ((rhs::DWordWL)[1] + (lhs_sub_rhs::DWordWL)[1] + carry[0] - (lhs::DWordWL)[1])
```

**Definition of `unsigned_lt`:**
```
unsigned_lt := carry[1]
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

We assume the inputs `lhs`, `rhs` and `signed` are partially range checked.

| Tag | Range | Description |
|-----|-------|-------------|
| `LT-A1` |  | `IS_WORD[lhs[0]]` |
| `LT-A2` |  | `IS_WORD[rhs[0]]` |

## Constraints

We first constrain that all inputs are range checked and all variables correspond to their definition. For the defining constraint of `lt`, [lt:c:lt], observe that it is a choice between two options, depending on the input flag `signed`. In the case of unsigned comparison, we simply need `unsigned_lt`, indicating that a wraparound (carry bit) modulo `2^64` is needed to go from `rhs` to `lhs` via addition. For the case of signed comparison, we first need some case analysis.

We split `a < b` into four disjoint cases, conditioned on the sign of `a` and `b`. Recall that the sign of a number in two's complement can be read off from the MSB, being `1` for a negative number and `0` for a positive one. For this analysis, we denote the MSB of `a` as `A` and the MSB of `b` as `B`. The four disjoint cases then become:

+ `dash(A) and B and (a < b)` + `A and dash(B) and (a < b)` + `A and B and (a < b)` + `dash(A) and dash(B) and (a < b)`

The first case is evidently false, while the second case simplifies to `A and dash(B)`. For the third and fourth case, observe that when `A = B`, the `<` relation is preserved by the modular correspondence between `[-2^(31), 2^(31))` and `[0, 2^(64))`. Importantly, this modular correspondence is merely a reinterpretation of the bits or values of `a` and `b`, due to the representation in two's complement. Hence, we can introduce the value `C = `unsigned_lt``, that accurately represents the relation `a < b` when `A = B`.

Combining our three remaining cases, we obtain the boolean formula `A dash(B) or A B C or dash(A) dash(B) C`. Since the cases are disjoint, this can be computed with the binary-valued polynomial `P(A, B, C) = A (1 - B) + A B C + (1 - A) (1 - B) C`.

The polynomial `P` can be simplified to a total degree of two. We claim that the polynomial `Q(A, B, C) = A (1 - B) + A C + (1 - B) C` is, for the purposes of this chip, equivalent to `P`. An exhaustive check shows that `P(A, B, C) != Q(A, B, C)` only for the triple `(A, B, C) = (1, 0, 1)`. This is, however, impossible due to the correctness of `ADD`. In more detail, if we let `s` be the (range-checked) difference `a - b` (so the equivalent of the `lhs_sub_rhs` column), and `x'` denote the most significant word of a variable `x`, we need `c dot 2^32 + a' = b' + s' + `carry[0]``, by the definition of `carry`. However, the left hand side of this is at least `3 dot 2^31`, as `(A, C) = (1, 1)`, and the right hand side is at most `(2^31 - 1) + (2^32 - 1) + 1 = 3 dot 2^31 - 1`. Therefore, we can use `Q` to constrain `lt` when `signed = 1`.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LT-C1` | `IS_HALF[lhs[1]]` | μ |
| `LT-C2` | `IS_HALF[rhs[1]]` | μ |
| `LT-C3` | `IS_BIT<signed>` |  |
| `LT-C4` | `IS_BIT<invert>` |  |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LT-C5` | `MSB16[lhs_msb; lhs[2]]` | μ |
| `LT-C6` | `MSB16[rhs_msb; rhs[2]]` | μ |
| `LT-C7` | `lt` = `signed` dot (A (1 - B) + A C + (1 - B) C) + (1 - `signed`) dot `unsigned_lt` |  |
| | _polynomial:_ `lt - signed * (lhs_msb * (1 - rhs_msb) + lhs_msb * carry[1] + (1 - rhs_msb) * carry[1]) - (1 - signed) * unsigned_lt = 0` | |
| `LT-C8` | `res` = `lt` xor `invert` |  |
| | _polynomial:_ `res + 2 * lt * invert - lt - invert = 0` | |

And then we constrain the subtraction, taking care of the remaining range checking not yet covered by the assumptions or the `MSB16` lookup.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `LT-C9.i` | i ∈ [0, 1] | `IS_BIT<carry[i]>` |  |
| `LT-C10.i` | i ∈ [0, 3] | `IS_HALF[lhs_sub_rhs[i]]` | μ |

The chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LT-C11` | `ALU[[res, 0]; lhs::DWordWL, rhs::DWordWL, ⧼LT⧽ + 32 * signed + 64 * invert]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `lhs` | `0` |
| `rhs` | `0` |
| `signed` | `0` |
| `invert` | `0` |
| `res` | `0` |
| `lhs_sub_rhs` | `0` |
| `lhs_msb` | `0` |
| `rhs_msb` | `0` |
| `lt` | `0` |
| `μ` | `0` |

## Potential optimizations

- Split the chip into a signed and an unsigned chip, making the unsigned version cheaper.