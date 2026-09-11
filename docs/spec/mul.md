# MUL Chip

The  chip constrains multiplication, both signed and unsigned, as well as providing access to the low and high halfs of the multiplication result.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `DWordHL` | the left hand operator. |
| `lhs_signed` | `Bit` | whether to interpret `lhs` as a signed integer (1) or not (0). |
| `rhs` | `DWordHL` | the right hand operator. |
| `rhs_signed` | `Bit` | whether to interpret `rhs` as a signed integer (1) or not (0). |

### Output

| Name | Type | Description |
|------|------|-------------|
| `lo` | `DWordHL` | the lower limbs of the (extended) multiplication result |
| `hi` | `DWordHL` | the upper limbs of the (extended) multiplication result |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `lhs_is_negative` | `Bit` | whether `lhs` is negative (1) or not (0) |
| `rhs_is_negative` | `Bit` | whether `rhs` is negative (1) or not (0) |
| `raw_product` | `B51[4]` | raw multiplication output |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `lhs_ext` | `Half[8]` | sign-extended value of `lhs` |
| `rhs_ext` | `Half[8]` | sign-extended value of `rhs` |
| `res` | `QuadWL` | concatenation of `lo` and `hi`. |
| `carry` | `B20[4]` | carry values |
| `μ_sum` | `BaseField` | sum of multiplicies |

**Definition of `lhs_ext`:**
```
lhs_ext (when iter=[0, 3]) := lhs[i]
lhs_ext (when iter=[4, 7]) := 65535 * lhs_is_negative
```

**Definition of `rhs_ext`:**
```
rhs_ext (when iter=[0, 3]) := rhs[i]
rhs_ext (when iter=[4, 7]) := 65535 * rhs_is_negative
```

**Definition of `res`:**
```
res (when iter=[0, 1]) := (lo::DWordWL)[i]
res (when iter=[2, 3]) := (hi::DWordWL)[i - 2]
```

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * (raw_product[0] - res[0])
carry (when iter=[1, 3]) := 2^-32 * (raw_product[i] + carry[i - 1] - res[i])
```

**Definition of `μ_sum`:**
```
μ_sum := μ_lo + μ_hi
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ_lo` | `BaseField` |  |
| `μ_hi` | `BaseField` |  |

`mat(delim: , top; bottom)` }

## Constraints

### Overview

When `lhs` and `rhs` are _unsigned_ integers, computing their product `mod 2^128` comes down to evaluating $ (sum_(j=0)^3 2^(16j) dot `lhs`_j) dot (sum_(i=0)^3 2^(16i) dot `rhs`_i) mod 2^128. $ If `lhs` and `rhs` are signed instead, the computation remains nearly identical: based on their signs, one must either zero or one-extend `lhs` and `rhs` --- forming `lhs_ext` and `rhs_ext` respectively --- and compute their product `mod 2^128`: $ (sum_(j=0)^7 2^(16j) dot `lhs_ext`_j) dot (sum_(i=0)^7 2^(16i) dot `rhs_ext`_i) mod 2^128. $ where `lhs_ext` and `rhs_ext` are treated as _unsigned_ integers. Note that by setting the extension limbs of `lhs` and/or `rhs` to `0` when the integer is (i) unsigned or (ii) signed and non-negative, this second formula still applies. For the purposes of constraining the multiplication operation, we rewrite this formula as

$ &(sum_(j=0)^7 2^(16j) dot `lhs_ext`_j) dot (sum_(i=0)^7 2^(16i) dot `rhs_ext`_i) mod 2^128 \ &equiv sum_(j=0)^7 sum_(i=0)^7 2^(16(i+j)) dot `lhs_ext`_j dot `rhs_ext`_i mod 2^128 \ &stackrel(triangle, equiv) sum_(j=0)^7 sum_(i=0)^(7-j) 2^(16(i+j)) dot `lhs_ext`_j dot `rhs_ext`_i mod 2^128 \ &stackrel(square, equiv) sum_(j=0)^7 sum_(i=j)^(7) 2^(16i) dot `lhs_ext`_j dot `rhs_ext`_(i-j) mod 2^128 \ &stackrel(penta, equiv) sum_(i=0)^7 sum_(j=0)^(i) 2^(16i) dot `lhs_ext`_j dot `rhs_ext`_(i-j) mod 2^128 \ &equiv sum_(i=0)^3 sum_(k=0)^1 sum_(j=0)^(2i+k) 2^(16(2i+k)) dot `lhs_ext`_j dot `rhs_ext`_(2i+k-j) mod 2^128 \ &equiv sum_(i=0)^3 2^(32i) dot sum_(k=0)^1 2^(16k) dot sum_(j=0)^(2i+k) `lhs_ext`_j dot `rhs_ext`_(2i+k-j) mod 2^128 $ where at step - `triangle` we can ignore `i > 7-j`, since that makes `2^(16(i+j)) equiv 0 mod 2^128`, - `square` we rewrite the second summation such that `i` iterates from `j` to 7, rather than `0` to `7-j`, and - `penta` we swap the sums.

We let `raw_product` capture the second summation in this last formula (see [mul:c:raw_product]). By construction, ``raw_product`_i < 2^51` for all `i in [0, 3]`, far exceeding the 32-bits that fit in a single `Word`-limb. What remains then is to reduce each limb of `raw_product` `mod 2^32`, carrying the overflow of each limb to the next, constructing the output `res` in doing so.

This reduce-and-carry operation is constrained by [mul:c:range_lo]/[mul:c:range_hi] and [mul:c:carry], combined with `carry`'s definition. [mul:c:carry] and `carry`'s definition enforce that $ forall i in [0, 3]: `raw_product`_i + `carry`_(i-1) - `res`_i in { k dot 2^32 | k in [0, 2^20) } $ with ``carry`_(-1) = 0` for simplicity. In other words: ``res`_i equiv `raw_product`_i + `carry`_(i-1) (mod 2^32)`. With [mul:c:range_lo]/[mul:c:range_hi] forcing ``res`_i < 2^32`, ``res`_i` can only assume one value: ``raw_product`_i + `carry`_(i-1) mod 2^32`.

*Note*: one may have observed that [mul:c:carry] requires ``carry`_i in [0, 2^20)`, while no limb of a valid carry value would ever exceed `2^19`. This is indeed the case. However, there is some slack in how tight one has to constrain the `carry` values. In fact, in this situation it suffices to assert that ``carry`_i < frac(p, 2^32, style: "skewed") approx 2^31`, where `p` denotes the field's modulus. Given that other chips also use 20-bit lookups, using `IS_B20` makes for a simpler design.

### Definitions

We constrain `lhs_is_negative` and `rhs_is_negative` according to their definition; `lo`, `hi` and `carry` are appropriately range checked.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MUL-C1` |  | `IS_BIT<lhs_signed>` |  |
| `MUL-C2` |  | `IS_BIT<rhs_signed>` |  |
| `MUL-C3.i` | i ∈ [0, 3] | `IS_HALF[lhs[i]]` | μ_sum |
| `MUL-C4.i` | i ∈ [0, 3] | `IS_HALF[rhs[i]]` | μ_sum |
| `MUL-C5` |  | `SIGN<lhs_is_negative; lhs[3], lhs_signed>` |  |
| `MUL-C6` |  | `SIGN<rhs_is_negative; rhs[3], rhs_signed>` |  |
| `MUL-C7.i` | i ∈ [0, 3] | `IS_HALF[lo[i]]` | μ_sum |
| `MUL-C8.i` | i ∈ [0, 3] | `IS_HALF[hi[i]]` | μ_sum |
| `MUL-C9.i` | i ∈ [0, 3] | `IS_B20[carry[i]]` | μ_sum |

### Product

[mul:c:raw_product] defines `raw_product` in terms of the (sign extended) input values `lhs` and `rhs`.

| Tag | Range | Description |
|-----|-------|-------------|
| `MUL-C10.i` | i ∈ [0, 3] | `raw_product[i]` = sum_(`k`=0)^1 2^(16k) sum_(`j`=0)^(2i+k) `lhs_ext[j]` dot `rhs_ext[2i+k-j]` |
| | | _polynomial:_ `Σ_k = 0^1 2^(16 * k) * Σ_j = 0^2 * i + k lhs_ext[j] * rhs_ext[2 * i + k - j] - raw_product[i] = 0` |

### Lookup

The  chip contributes the following to the lookup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MUL-C11` | `ALU[lo::DWordWL; lhs::DWordWL, rhs::DWordWL, ⧼MUL⧽ + 32 * lhs_signed + 64 * rhs_signed]` | -μ_lo |
| `MUL-C12` | `ALU[hi::DWordWL; lhs::DWordWL, rhs::DWordWL, ⧼MUL⧽ + 32 * lhs_signed + 64 * rhs_signed + 128]` | -μ_hi |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `lhs` | `0` |
| `lhs_signed` | `0` |
| `rhs` | `0` |
| `rhs_signed` | `0` |
| `lo` | `0` |
| `hi` | `0` |
| `lhs_is_negative` | `0` |
| `rhs_is_negative` | `0` |
| `raw_product` | `0` |
| `μ_lo` | `0` |
| `μ_hi` | `0` |

## Notes/optimizations

- `lo` and `hi` are stored in `DWordHL`s (rather than `DWordWL`s) because of their values being range checked. Since it is not required that both `μ_lo` and `μ_hi` are non-zero at the same time, one cannot safely assume their range to be checked elsewhere. - As an optimization, one might be able to use a `DWordWL` and `DWordHL` to store `lo` and `hi`, where one would decide which to store in which based on the multiplicities `μ_lo` and `μ_hi`; the value sent into the lookup could then be assumed range-checked by the other side of the relation. This optimization was not included at this moment because of its negative impact on the readability and verifiability of the chip.