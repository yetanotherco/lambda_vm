# ECSM Accelerator

## Theory behind Elliptic Curves

An elliptic curve `E(a, b, p)` in _short Weierstrass_ form has parameters `a,b in FF_p` for some prime `p` with `4a^3+27b^2 eq.not 0`, and coordinates `(x, y) in FF_p^2` satisfying the equation $ y^2=x^3+a x+b. $

Additionally, there is the _point at infinity_, `⁠`, which has no native short-Weierstrass representation. It acts as the identity element (zero) in the group: given non-zero curve point `P`, it holds that $

$

The negation of curve point `P = (x_P, y_P)` is constructed as `-P := (x_P, -y_P)`. Naturally, `P + (-P) = `.

The addition of points `P, Q` distinguishes three cases. For `x_P eq.not x_Q`, one uses $ (x_R, y_R) := (lambda^2 - x_P - x_Q, lambda (x_P - x_R) - y_P) $ with `lambda = frac((y_Q - y_P), (x_Q - x_P), style: "horizontal")`. When `x_P = x_Q` and `y_P eq.not - y_Q`, one instead uses `lambda = frac(3x_P^2, 2y_P, style: "horizontal")`. The remaing case that `(x_P, y_P) = (x_Q, -y_Q)` corresponds with `Q = -P`; the addition results in ``.

An addition operation gives rise to an algorithm for scalar multiplication. Given curve point `P` and scalar `k`, the multiple `k times P` can trivially be computed as `P + P + ... + P`. This accelerator instead leverages the _double-and-add_ ) technique, which utilizes only `O(log(k))` additions for the full multiplication.

The purpose of this accelerator is to speed up the scalar multiplication `k times G` for scalar `k in [1, N)` and point `G in E(0, b, p) without {}` with `p in [2^248, 2^256)`. In particular, the accelerator supports the curve ``secp256k1` = E(0, 7, 2^256-2^32 - 977)`. This accelerator leverages _double-and-add_, executing the multiplication in `O(log(k))` doublings and `O(w_H (k)) = O(log(k))` additions, where `w_H (dot)` denotes the hamming-weight of a bitstring.

## Overview

The accelerator comprises three chips: - *`ECSM` (Elliptic Curve Scalar Multiply)*; this chip is responsible for loading inputs `x_G` and `k` from memory, reconstructing `y_G`, dispatching a double-and-add sequence request to the `ECDAS` chip, and writing the result point `x_R` back to memory. - *`ECDAS` (Elliptic Curve Double/Add Sequence)* is responsible for the consecutive doubling/adding the provided point to itself, ultimately arriving at `k times G`. - *`EC_SCALAR`* serves `k` bit-by-bit to the `ECDAS` chip to inform the flow of the double-and-add sequence.

## ECSM <ecsm-sm>

The  (Elliptic Curve Scalar Multiply) chip is generic over the constants - `b`, the second curve coefficient, - `p`, the prime field modulus, and - `N`, the order of the curve group. To support scalar multiplication over different curves, one chip instance should be created for each curve.

The chip is triggered by executing `ECALL`, with the ECALL-number is set to `-3`. The chip expects - `x10` to contain the address where `x_R := (k times G)_x` is to be stored, - `x11` to contain the address at which the least significant byte of `x_G` is to be found, - `x12` to contain the address at which the least significant byte of `k` is to be found, where it is assumed that `x_G, x_R` and `k` are provided as little-endian.

### Columns

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which this operation is executed |
| `addr_xG` | `DWordWL` | address at which `x`-coordinate of start point `G` is stored |
| `addr_k` | `DWordWL` | address at which scalar `k` is stored |
| `addr_xR` | `DWordWL` | address to which the `x`-coordinate of result point `R` is to be written |

### Output

| Name | Type | Description |
|------|------|-------------|
| `xR` | `U256BL` | $(`k` times `G`)_x$ |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `yR` | `U256BL` | $(`k` times `G`)_y$ |
| `k` | `U256BL` | `k` |
| `len_k` | `Byte` | Position of `k`'s most significant 1-bit |
| `xG` | `U256BL` | $x_G$ |
| `yG` | `U256BL` | $y_G$ |
| `x2` | `U256BL` | $x_G^2$ |
| `q0` | `U256BL` | quotient for computing `x2` |
| `c0` | `BaseField[64]` | carries for computing `x2` |
| `q1` | `Byte[33]` | quotient for computing `yG` |
| `c1` | `BaseField[64]` | carries for computing `yG` |
| `k_sub_N` | `U256HL` | $`k`- `N` mod 2^256$ |
| `xR_sub_p` | `U256HL` | $x_R - `p` mod 2^256$ |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `c2` | `Bit[8]` | carries for computing $`N` + `k_sub_N`$ |
| `c3` | `Bit[8]` | carries for computing $`P` + `xR_sub_p`$ |
| `XG` | `Byte[64]` | zero-extension of `xG` |
| `YG` | `Byte[64]` | zero-extension of `yG` |
| `X2` | `Byte[64]` | zero-extension of `x2` |
| `Q0` | `Byte[64]` | zero-extension of `q` |
| `Q1` | `Byte[64]` | zero-extension of `q1` |
| `B` | `Byte[64]` | zero-extension of `b` |
| `P` | `Byte[64]` | zero-extension of `p` |

**Definition of `c2`:**
```
c2 (when iter=['i', 0]) := 2^-32 * ((N::U256WL)[i] + (k_sub_N::U256WL)[i] - (k::U256WL)[i])
c2 (when iter=['i', 1, 7]) := 2^-32 * ((N::U256WL)[i] + (k_sub_N::U256WL)[i] + c2[i - 1] - (k::U256WL)[i])
```

**Definition of `c3`:**
```
c3 (when iter=['i', 0]) := 2^-32 * ((p::U256WL)[i] + (xR_sub_p::U256WL)[i] - (xR::U256WL)[i])
c3 (when iter=['i', 1, 7]) := 2^-32 * ((p::U256WL)[i] + (xR_sub_p::U256WL)[i] + c3[i - 1] - (xR::U256WL)[i])
```

**Definition of `XG`:**
```
XG (when iter=['i', 0, 31]) := xG[i]
XG (when iter=['i', 32, 63]) := 0
```

**Definition of `YG`:**
```
YG (when iter=['i', 0, 31]) := yG[i]
YG (when iter=['i', 32, 63]) := 0
```

**Definition of `X2`:**
```
X2 (when iter=['i', 0, 31]) := x2[i]
X2 (when iter=['i', 32, 63]) := 0
```

**Definition of `Q0`:**
```
Q0 (when iter=['i', 0, 31]) := q0[i]
Q0 (when iter=['i', 32, 63]) := 0
```

**Definition of `Q1`:**
```
Q1 (when iter=['i', 0, 32]) := q1[i]
Q1 (when iter=['i', 33, 63]) := 0
```

**Definition of `B`:**
```
B (when iter=['i', 0, 31]) := b[i]
B (when iter=['i', 32, 63]) := 0
```

**Definition of `P`:**
```
P (when iter=['i', 0, 31]) := p[i]
P (when iter=['i', 32, 63]) := 0
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

### Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `ECSM-A1` |  | $(#`addr_xG` mod 2^32) + 24 < 2^32$ |
| `ECSM-A2` |  | $(#`addr_k` mod 2^32) + 31 < 2^32$ |
| `ECSM-A3` |  | $(#`addr_xR` mod 2^32) + 24 < 2^32$ |

### Constraints

#### Interactions

This chip is triggered by an `ECALL` with the opcode indicating this chip:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `ECSM-C1` | `IS_BIT<μ>` |  |
| `ECSM-C2` | `ECALL[timestamp, [2^32 - 3, 2^32 - 1]]` | -μ |

#### Read `xG`

Once triggered, it loads register `x11` to see where `x_G` is stored in memory ([ec:c:read_addr_xG]) and subsequently load `x_G` in ([ec:c:read_xG]). Assumption [ec:a:addr_xG_alignment] ensures no overflows happen when incrementing the address in [ec:c:read_xG]. Note: `xG` is assumed to be range checked, since they're read from memory.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C3` |  | `MEMW[[addr_xG[0], addr_xG[1], 0, 0, 0, 0, 0, 0]; 1, (2 * 11)::DWordWL, [addr_xG[0], addr_xG[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | μ |
| `ECSM-C4.i` | i ∈ [0, 3] | `MEMW[[xG[8 * i + 0], xG[8 * i + 1], xG[8 * i + 2], xG[8 * i + 3], xG[8 * i + 4], xG[8 * i + 5], xG[8 * i + 6], xG[8 * i + 7]]; 0, addr_xG + (8 * i)::DWordWL, [xG[8 * i + 0], xG[8 * i + 1], xG[8 * i + 2], xG[8 * i + 3], xG[8 * i + 4], xG[8 * i + 5], xG[8 * i + 6], xG[8 * i + 7]], timestamp, 0, 0, 1]` | μ |

#### Constrain `Gy`

With `x_G` read and range checked, we direct our attention to `y_G`. Rather than reading it from memory, the prover provides it as a witness and proves it to be correct. In particular, the chip enforces the relations $ x_G^2 - `x2` - q_0 dot p &= 0,\ y_G^2 - x_G dot `x2` - b + (p - q_1)p &= 0\ $ where non-negative `q_0` and `q_1` are prover-provided witnesses. Note that these are equivalent to $

y_G^2 &equiv x_G dot `x2` + b  mod p\ $ which combine to `y_G^2 equiv x_G^3 + b mod p`. Rewriting the two relations, we get $ q_0 &= (x_G^2 - `x2`) dot p^(-1),\ q_1 &= (y_G^2 - x_G dot `x2`-b) dot p^(-1) + p. $ Using the fact that `x_G, y_G, `x2` in [0, p)`, we find that `q_0 in [0, p)` and `q_1 in [0, 2p)`. We therefore restrict the choice of quotients to `q_0 in [0, 2^256)` and `q_1 in [0, 2^257)`.

Below, we enforce the first of the two sub-relations. We emphasize here that [ec:c:c0_63_is_zero] is required to ensure the sum evaluates to `0`, rather than just `0 mod 2^256`. The constraints [ec:c:c0_0] and [ec:c:c0_i], as well as the magic number `8160` in [ec:c:range_c0] are discussed in [ecsm]-limb_carry.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C5.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<x2[i]>` |  |
| `ECSM-C6.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<q0[i]>` |  |
| `ECSM-C7` |  | 2^8 dot `c0`_0 = `XG`_0 dot `XG`_0 - `X2`_0 - `Q0`_0 dot `P`_0 |  |
| | | _polynomial:_ `XG[0] * XG[0] - X2[0] - Q0[0] * P[0] - 2^8 * c0[0] = 0` | |
| `ECSM-C8.i` | i ∈ [1, 63] | 2^8 dot `c0`_i = `c0`_(i-1) - `X2`_i + sum_(j=0)^i `XG`_j dot `XG`_(i-j) - `Q0`_j dot `P`_(i-j) |  |
| | | _polynomial:_ `(c0[i - 1] - 2^8 * c0[i] - X2[i]) + Σ_j = 0^i (XG[j] * XG[i - j] - Q0[j] * P[i - j]) = 0` | |
| `ECSM-C9` |  | `c0`_63 = 0 |  |
| | | _polynomial:_ `c0[63] = 0` | |
| `ECSM-C10.i` | i ∈ [0, 62] | `IS_HALF[c0[i] + 8160]` | μ |

Next, we restrict the witness pair `(y_G, `q1`)`. Note there that [ec:c:c1_0] and [ec:c:c1_i] multiply `B` by `μ` to simplify the padding; there are no other side-effects to this since ``μ` = 1` on non-padding rows ([ec:c:mu_isbit]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C11.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<yG[i]>` |  |
| `ECSM-C12.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<q1[i]>` |  |
| `ECSM-C13` |  | `IS_BIT<q1[32]>` |  |
| `ECSM-C14` |  | 2^8 dot `c1`_0 = `YG`_0 dot `YG`_0 - `X2`_0 dot `XG`_0 - `μ` dot `B`_0 + `P`_0 dot `P`_0 - `Q1`_0 dot `P`_0 |  |
| | | _polynomial:_ `YG[0] * YG[0] + P[0] * P[0] - X2[0] * XG[0] - μ * B[0] - Q1[0] * P[0] - 2^8 * c1[0] = 0` | |
| `ECSM-C15.i` | i ∈ [1, 63] | 2^8 dot `c1`_i = `c1`_(i-1) - `μ` dot `B`_i + sum_(j=0)^i (`YG`_j dot `YG`_(i-j) + `P`_j dot `P`_(i-j) - `X2`_j dot `XG`_(i-j) - `Q1`_j dot `P`_(i-j)) |  |
| | | _polynomial:_ `(c1[i - 1] - 2^8 * c1[i] - μ * B[i]) + Σ_j = 0^i (YG[j] * YG[i - j] + P[j] * P[i - j] - X2[j] * XG[i - j] - Q1[j] * P[i - j]) = 0` | |
| `ECSM-C16` |  | `c1`_63 = 0 |  |
| | | _polynomial:_ `c1[63] = 0` | |
| `ECSM-C17.i` | i ∈ [0, 62] | `IS_HALF[c1[i] + 16319]` | μ |

#### Read and verify `k`

After reading `addr_k` from `x12` ([ec:c:read_addr_k]), we read `k` from this address ([ec:c:load_k]). Similar to `addr_xG`, assumption [ec:a:addr_k_alignment] ensures the address offsets in [ec:c:load_k] do not overflow the lower limb. To prevent the point at infinity from showing up during the scalar multiplication, we require that ``k` < `N``. This is achieved by requiring that the addition ``N` + (`k` - `N`)` overflows `mod 2^256` ([ec:c:k_lt_N]). Additionally, [ec:c:k_gt_0] ensures that ``k` > 0`, preventing a case where ``k` times `G` = `.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C18` |  | `MEMW[[addr_k[0], addr_k[1], 0, 0, 0, 0, 0, 0]; 1, (2 * 12)::DWordWL, [addr_k[0], addr_k[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | μ |
| `ECSM-C19.i` | i ∈ [0, 3] | `MEMW[[k[8 * i + 0], k[8 * i + 1], k[8 * i + 2], k[8 * i + 3], k[8 * i + 4], k[8 * i + 5], k[8 * i + 6], k[8 * i + 7]]; 0, addr_k + (8 * i)::DWordWL, [k[8 * i + 0], k[8 * i + 1], k[8 * i + 2], k[8 * i + 3], k[8 * i + 4], k[8 * i + 5], k[8 * i + 6], k[8 * i + 7]], timestamp, 0, 0, 1]` | μ |
| `ECSM-C20.i` | i ∈ [0, 15] | `IS_HALF[k_sub_N[i]]` | μ |
| `ECSM-C21.i` | i ∈ [0, 6] | μ ⇒ `IS_BIT<c2[i]>` |  |
| `ECSM-C22` |  | `μ` => `c2`_7 = 1 |  |
| | | _polynomial:_ `μ * (1 - c2[7]) = 0` | |
| `ECSM-C23` |  | `ZERO[k[0] + k[1] + k[2] + k[3] + k[4] + k[5] + k[6] + k[7] + k[8] + k[9] + k[10] + k[11] + k[12] + k[13] + k[14] + k[15] + k[16] + k[17] + k[18] + k[19] + k[20] + k[21] + k[22] + k[23] + k[24] + k[25] + k[26] + k[27] + k[28] + k[29] + k[30] + k[31]]` | μ |

#### Subroutine

With point `G` and scalar `k` fully constructed, we delegate bit-by-bit serving of the scalar `k` to the `EC_SCALAR` chip. Here, we capture the index of the most significant 1-bit of `k` in `len_k`. Note: if the prover decides to capture a lesser significant bit here, the LogUp will not balance, as the skipped bits will never taken off the bus. Next, we interact with the `ECDAS` chip, providing `G` both as the accumulator, and increment ([ec:c:start_double_add]); we specifically instruct the chip to start with a _double_-operation. After completing its double-and-add sequence, the result is captured in `R` ([ec:c:receive_double_add]).

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `ECSM-C24` | `SERVE_K[timestamp, addr_k::DWordWL, 31]` | μ |
| `ECSM-C25` | `BIT[timestamp, len_k]` | -μ |
| `ECSM-C26` | `ECDAS[timestamp, xG, yG, xG, yG, len_k - 1, 0]` | μ |
| `ECSM-C27` | `ECDAS[timestamp, xR, yR, xG, yG, -1, 0]` | -μ |

#### Range check `xR`

Before storing `x_R`, it is verified that `x_R in [0, p)`. To this end, witness ``xR_sub_p` := `xR` - p mod 2^256` is added to `p`; if the addition sums to `xR` and overflows `mod 2^256`, it must hold that ``xR` < p`. The addition is constrained by requiring that `c3` are bits ([ec:c:range_c3]); an overflow occurs if and only if ``c3[7]` = 1` ([ec:c:xR_addition_overflows]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C28.i` | i ∈ [0, 15] | `IS_HALF[xR_sub_p[i]]` | μ |
| `ECSM-C29.i` | i ∈ [0, 6] | μ ⇒ `IS_BIT<c3[i]>` |  |
| `ECSM-C30` |  | `μ` => `c3`_7 = 1 |  |
| | | _polynomial:_ `μ * (1 - c3[7]) = 0` | |

#### Write `xR`

We read `addr_xR` from register `x10` ([ec:c:load_addrR]), and subsequently write `xR` to this address ([ec:c:write_xR]). Note that the `timestamp` on both memory accesses is offset to allow `addr_xR` to equal `addr_xG` and thus for `x_R` to overwrite `x_G` in memory. Similar to `addr_xG` and `addr_k`, it is assumed that the addition of the small offsets will not overflow the lower limb of `addr_xR` ([ec:a:addr_xR_alignment]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C31` |  | `MEMW[[addr_xR[0], addr_xR[1], 0, 0, 0, 0, 0, 0]; 1, (2 * 10)::DWordWL, [addr_xR[0], addr_xR[1], 0, 0, 0, 0, 0, 0], timestamp + 1::DWordWL, 1, 0, 0]` | μ |
| `ECSM-C32.i` | i ∈ [0, 3] | `MEMW[0, addr_xR + (8 * i)::DWordWL, [xR[8 * i + 0], xR[8 * i + 1], xR[8 * i + 2], xR[8 * i + 3], xR[8 * i + 4], xR[8 * i + 5], xR[8 * i + 6], xR[8 * i + 7]], timestamp + 2::DWordWL, 0, 0, 1]` | μ |

### Padding

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `addr_xG` | `0` |
| `addr_k` | `0` |
| `addr_xR` | `0` |
| `xR` | `0` |
| `yR` | `0` |
| `k` | `0` |
| `len_k` | `0` |
| `xG` | `0` |
| `yG` | `0` |
| `x2` | `0` |
| `q0` | `0` |
| `c0` | `0` |
| `q1` | `p` |
| `c1` | `0` |
| `k_sub_N` | `0` |
| `xR_sub_p` | `0` |
| `μ` | `0` |

## ECDAS chip <ecdas>

The  chip (_Elliptic Curve Double-and-Add Sequence_) is responsible for accelerating the addition of two curve points, or the doubling of a single curve point. More specifically, given curve points `A` (accumulator) and `G` (generator), and selector bit `op`, it performs the mapping $ (A, G) mapsto cases( (A + A, &G) &text("if") `op` = 0, (A + G, &G) &text("if") `op` = 1 $

Recall that the addition of two curve points `A, B` is treated differently based on three cases:

enum.item[`x_A eq.not x_B`], enum.item[`x_A eq x_B` and `y_A eq.not -y_B`, or], enum.item[`x_A eq x_B` and `y_A eq -y_B`] where _double_ may encounter the last two cases, while _add_ may encounter all three. Cases 2 and 3 may, for specific inputs, evaluate to ``: a point that has no native short-Weierstrass representation. Therefore, the  and  chips were designed to avoid this case. To see how, note that  + is the sole chip that can "activate" the  chip by issuing an `ECDAS` lookup, + enforces that `G` and the initial `A` do not equal ``, and + ensures `k in [1, N)`, where `N` denotes the order of the curve. This combined yields that neither doubling `A` or adding `A + G` can produce ``:

*Double.* For `2A` to equal ``, the curve must have _even_ order. Since the order of the `secp256k1` curve is _odd_, such a point does not exist.

*Add.* If `A + G = `, then `A = -G =  - G = r N G - G` for some `r >= 0`. Because  initializes `A = G eq.not `, it must hold that `r >= 1`. Furthermore, the restriction that `k <= N-1` ensures `r <= 1`. Hence, `A = (N-1)G`. Since `N-1` is the maximal value of `k`, the previous round producing `A = (N-1)G` was the last round of this scalar multiplication. This means that now `round` is negative, which will fail constraint [ecdas:c:range_round].

### Columns

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | Timestamp at which the operation is executed. |
| `xG` | `U256BL` | $x_`G`$ |
| `yG` | `U256BL` | $y_`G`$ |
| `xA` | `U256BL` | $x_`A`$ |
| `yA` | `U256BL` | $y_`A`$ |
| `round` | `Byte` | scaling round |
| `op` | `Bit` | whether to double (0) or add (1) |

### Output

| Name | Type | Description |
|------|------|-------------|
| `xR` | `U256BL` | $x_`R`$ |
| `yR` | `U256BL` | $y_`R`$ |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `λ` | `U256BL` | `λ` |
| `q0` | `Byte[33]` | quotient used to constrain `λ` |
| `c0` | `BaseField[64]` | carries used to constrain `λ` |
| `q1` | `Byte[33]` | quotient used to constrain `xR` |
| `c1` | `BaseField[64]` | carries used to constrain `xR` |
| `q2` | `Byte[33]` | quotient used to constrain `yR` |
| `c2` | `BaseField[64]` | carries used to constrain `yR` |
| `next_op` | `Bit` | `op`-flag for the next iteration |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `XG` | `U512BL` | zero-extension of `xG` |
| `YG` | `U512BL` | zero-extension of `yG` |
| `XA` | `U512BL` | zero-extension of `xA` |
| `YA` | `U512BL` | zero-extension of `yA` |
| `XR` | `U512BL` | zero-extension of `xR` |
| `YR` | `U512BL` | zero-extension of `yR` |
| `Λ` | `U512BL` | zero-extension of `λ` |
| `Q0` | `U512BL` | zero-extension of `q0` |
| `Q1` | `U512BL` | zero-extension of `q1` |
| `Q2` | `U512BL` | zero-extension of `q2` |
| `P` | `U512BL` | zero-extension of `p` |
| `R` | `U512BL` | zero-extension of `r` |

**Definition of `XG`:**
```
XG (when iter=['i', 0, 31]) := xG[i]
XG (when iter=['i', 32, 63]) := 0
```

**Definition of `YG`:**
```
YG (when iter=['i', 0, 31]) := yG[i]
YG (when iter=['i', 32, 63]) := 0
```

**Definition of `XA`:**
```
XA (when iter=['i', 0, 31]) := xA[i]
XA (when iter=['i', 32, 63]) := 0
```

**Definition of `YA`:**
```
YA (when iter=['i', 0, 31]) := yA[i]
YA (when iter=['i', 32, 63]) := 0
```

**Definition of `XR`:**
```
XR (when iter=['i', 0, 31]) := xR[i]
XR (when iter=['i', 32, 63]) := 0
```

**Definition of `YR`:**
```
YR (when iter=['i', 0, 31]) := yR[i]
YR (when iter=['i', 32, 63]) := 0
```

**Definition of `Λ`:**
```
Λ (when iter=['i', 0, 31]) := λ[i]
Λ (when iter=['i', 32, 63]) := 0
```

**Definition of `Q0`:**
```
Q0 (when iter=['i', 0, 32]) := q0[i]
Q0 (when iter=['i', 33, 63]) := 0
```

**Definition of `Q1`:**
```
Q1 (when iter=['i', 0, 32]) := q1[i]
Q1 (when iter=['i', 33, 63]) := 0
```

**Definition of `Q2`:**
```
Q2 (when iter=['i', 0, 32]) := q2[i]
Q2 (when iter=['i', 33, 63]) := 0
```

**Definition of `P`:**
```
P (when iter=['i', 0, 31]) := p[i]
P (when iter=['i', 32, 63]) := 0
```

**Definition of `R`:**
```
R (when iter=['i', 0, 32]) := r[i]
R (when iter=['i', 33, 63]) := 0
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

### Constraints

First, the chips receives the input for this double/add step:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `ECDAS-C1` | `ECDAS[timestamp, xA, yA, xG, yG, round, op]` | -μ |
| `ECDAS-C2` | `IS_BIT<μ>` |  |
| `ECDAS-C3` | μ ⇒ `IS_BYTE<round>` |  |

#### Operation switching

The `op`-flag determines whether `R := 2A` (0) or `R:= A+G` (1). This chip introduces a set of constraints that properly constrains `R` depending on this flag. To illustrate how this is achieved, we split addition up in three relations: $ lambda &equiv (y_G - y_A)/(x_G - x_A) &&mod p,\ x_R &equiv lambda^2 - x_A - x_G &&mod p,\ y_R &equiv lambda (x_A - x_R) - y_A &&mod p.\ $ Introducing the non-negative witnesses `q'_0, q'_1` and `q_2`, we can convert these relations into $ lambda (x_G - x_A) - y_G + y_A + (`r` - q'_0) p &= 0,\ lambda^2 - x_A - x_G - x_R + (`r` - q'_1) p &= 0,\ lambda (x_A - x_R) - y_A - y_R + (`r` - q_2) p &= 0,\ $ for some `r in NN` to be fixed later.

Special attention should be paid to the first relation: if `x_A = x_G`, `lambda` can be chosen freely. By design, this situation cannot occur.

Observe that this would require either `A = G` or `A = -G`. With the latter situation previously ruled out, only the first remains. For `A = (r N + 1) G` for some `r in NN` and `N` the order of the curve, all cases with `r>0` can be ruled out since  verifies that the scalar `k < N`. The remaining case `A=G` is the intial state pushed onto the LogUp by  ([ec:c:start_double_add]), with `op`-flag set to `0` (_double_), not `add`. Hence, this situation cannot occur. ]

We rewrite the relations to find $ q'_0 &= `r` + p^(-1) dot (lambda (x_G - x_A) - y_G + y_A),\ q'_1 &= `r` + p^(-1) dot (lambda^2 - x_A - x_G - x_R),\ q_2  &= `r` + p^(-1) dot (lambda (x_A - x_R) - y_A - y_R)\ $ from which we can conclude that `q'_0, q_2 in (`r`-p, `r`+p)` and `q'_1 in (`r`, `r` + p)`. When doubling, only the formulae for `lambda` and `x_R` are different: $ lambda &equiv (3x_A^2)/(2y_A) &&mod p,\ x_R &equiv lambda^2 - 2x_A &&mod p.\ $ Introducing non-negative witnesses `q''_0` and `q''_1`, we convert these into $ 2lambda y_A - 3x_A^2 + (`r` - q''_0) p &= 0,\ lambda^2 - 2x_A - x_G - x_R + (`r` - q''_1) p &= 0.\ $

Special attention should be paid to the first relation: if `y_A = 0`, `lambda` can again be chosen freely. As previously established, `y_A != 0` for all points on the `secp256k1` curve. Hence, this situation will not occur. ] Reordering yields $ q''_0 &= `r` + p^(-1) dot (2lambda y_A - 3x_A^2 ),\ q''_1 &= `r` + p^(-1) dot (lambda^2 - 2x_A - x_G - x_R ).\ $ where `q''_0 in (`r`-3p, `r` + 2p)`, and `q''_1 = (`r`, `r` + p)`. We can now leverage the `op`-flag to merge the relations for `lambda` and `x_R` into $

lambda^2 - x_A - x_G - x_R + (1-`op`) (x_G - x_A) + (`r` - q_1) p &= 0\ $ which yields $ q_0 &= `r` + p^(-1) dot (`op` dot ((x_G - x_A)lambda - y_G + y_A) + (1-`op`) (2lambda y_A - 3x_A^2)),\ q_1 &= `r` + p^(-1) dot ((lambda^2 - x_A - x_G - x_R + (1-`op`) (x_G - x_A)).\ $ with `q_0 in (r-3p, r+2p)` and `q_1 in (r, r+p)`. By setting `r := 3p`, we ensure `q_0 in (0, 5p), q_1 in (3p, 4p)` and `q_2 in (2p, 4p)` are non-negative for all inputs.

#### Constraining $lambda$

We start by establishing the relation $

$

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECDAS-C4.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<λ[i]>` |  |
| `ECDAS-C5.i` | i ∈ [0, 32] | μ ⇒ `IS_BYTE<q0[i]>` |  |
| `ECDAS-C6` |  | 2^8 dot `c0`_0 = `op` dot (`Λ`_0 dot (`XG`_0 - `XA`_0) + `YA`_0 - `YG`_0) + (1 - `op`) dot (2 dot `Λ`_0 dot `YA`_0 - 3 dot `XA`_0 dot `XA`_0) + `R`_0 dot `P`_0 - `Q0`_0 dot `P`_0 = 0 |  |
| | | _polynomial:_ `2^8 * c0[0] + Q0[0] * P[0] - R[0] * P[0] - op * (Λ[0] * (XG[0] - XA[0]) + YA[0] - YG[0]) - (1 - op) * (2 * Λ[0] * YA[0] - 3 * XA[0] * XA[0]) = 0` | |
| `ECDAS-C7.i` | i ∈ [1, 63] | 2^8 dot `c0`_i = `c0`_(i-1) + `op` dot (`YA`_i - `YG`_i) + sum_(j=0)^i `op` dot `Λ`_j dot (`XG`_(i-j) - `XA`_(i-j)) + (1 - `op`) dot (2 dot `Λ`_j dot `YA`_(i-j) - 3 dot `XA`_j dot `XA`_(i-j)) + `R`_j dot `P`_(i-j) - `Q0`_j dot `P`_(i-j) |  |
| | | _polynomial:_ `2^8 * c0[i] - c0[i - 1] - op * (YA[i] - YG[i]) - Σ_j = 0^i (op * Λ[j] * (XG[i - j] - XA[i - j]) + (1 - op) * (2 * Λ[j] * YA[i - j] - 3 * XA[j] * XA[i - j]) + (R[j] * P[i - j] - Q0[j] * P[i - j])) = 0` | |
| `ECDAS-C8` |  | `c0`_63 = 0 |  |
| | | _polynomial:_ `c0[63] = 0` | |
| `ECDAS-C9.i` | i ∈ [0, 62] | `IS_HALF[c0[i] + 32636]` | μ |

#### Constraining $x_R$

Secondly, we establish $ lambda^2 - x_A - x_G - x_R - (1-`op`) (x_A - x_G) + (`r` - q_1) p &= 0 $

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECDAS-C10.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<xR[i]>` |  |
| `ECDAS-C11.i` | i ∈ [0, 32] | μ ⇒ `IS_BYTE<q1[i]>` |  |
| `ECDAS-C12` |  | 2^8 dot `c1`_0 = `Λ`_0 dot `Λ`_0 - `XA`_0 - `XG`_0 - `XR`_0 - (1-`op`) (`XA`_0 - `XG`_0) + `R`_0 dot `P`_0 - `Q1`_0 dot `P`_0 |  |
| | | _polynomial:_ `Λ[0] * Λ[0] + R[0] * P[0] - Q1[0] * P[0] - XA[0] - XG[0] - XR[0] - (1 - op) * (XA[0] - XG[0]) - 2^8 * c1[0] = 0` | |
| `ECDAS-C13.i` | i ∈ [1, 63] | 2^8 dot `c1`_i = `c1`_(i-1) - `XA`_i - `XG`_i - `XR`_i - (1- `op`) (`XA`_i - `XG`_i) + sum_(j=0)^i `Λ`_j dot `Λ`_(i-j) + `R`_j dot `P`_(i-j) - `Q1`_j dot `P`_(i-j) |  |
| | | _polynomial:_ `c1[i - 1] - 2^8 * c1[i] - XA[i] - XG[i] - XR[i] - (1 - op) * (XA[i] - XG[i]) - Σ_j = 0^i (Q1[j] * P[i - j] - R[j] * P[i - j] - Λ[j] * Λ[i - j]) = 0` | |
| `ECDAS-C14` |  | `c1`_63 = 0 |  |
| | | _polynomial:_ `c1[63] = 0` | |
| `ECDAS-C15.i` | i ∈ [0, 62] | `IS_HALF[c1[i] + 8161]` | μ |

#### Constraining $y_R$

Third, $ lambda (x_A - x_R) - y_A - y_R + (`r` - q_2) p &= 0 $ is constrained:

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECDAS-C16.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<yR[i]>` |  |
| `ECDAS-C17.i` | i ∈ [0, 32] | μ ⇒ `IS_BYTE<q2[i]>` |  |
| `ECDAS-C18` |  | 2^8 dot `c2`_0 = `Λ`_0 dot (`XA`_0 - `XR`_0) - `YA`_0 - `YR`_0 + `R`_0 dot `P`_0 - `Q2`_0 dot `P`_0 |  |
| | | _polynomial:_ `Λ[0] * (XA[0] - XR[0]) + R[0] * P[0] - Q2[0] * P[0] - YA[0] - YR[0] - 2^8 * c2[0] = 0` | |
| `ECDAS-C19.i` | i ∈ [1, 63] | 2^8 dot `c2`_i = `c2`_(i-1) - `YA`_i - `YR`_i + sum_(j=0)^i `Λ`_j dot (`XA`_(i-j) - `XR`_(i-j)) + `R`_j dot `P`_(i-j) - `Q2`_j dot `P`_(i-j) |  |
| | | _polynomial:_ `c2[i - 1] - 2^8 * c2[i] - YA[i] - YR[i] - Σ_j = 0^i (Q2[j] * P[i - j] - R[j] * P[i - j] - Λ[j] * (XA[i - j] - XR[i - j])) = 0` | |
| `ECDAS-C20` |  | `c2`_63 = 0 |  |
| | | _polynomial:_ `c2[63] = 0` | |
| `ECDAS-C21.i` | i ∈ [0, 62] | `IS_HALF[c2[i] + 16320]` | μ |

Lastly, the updated accumulator is sent out for the next step to be processed ([ecdas:c:send]). To determine whether the next step should be an addition or doubling, the `next_op` bit is provided as witness by the prover. Setting this bit to 1 can only be done in active rows ([ecdas:c:next_op_implies_mu]), when the current ``op` = 0` (double), and does require the scalar bit in this position to be set ([ecdas:c:receive_next_op]).

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `ECDAS-C22` | `IS_BIT<next_op>` |  |
| `ECDAS-C23` | `BIT[timestamp, round]` | -next_op |
| `ECDAS-C24` | `op` = 1 => `next_op` = 0 |  |
| | _polynomial:_ `op * next_op = 0` | |
| `ECDAS-C25` | `next_op` = 1 => `μ` = 1 |  |
| | _polynomial:_ `next_op * (1 - μ) = 0` | |
| `ECDAS-C26` | `ECDAS[timestamp, xR, yR, xG, yG, round - 1 - next_op, next_op]` | μ |

### Padding

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `xG` | `0` |
| `yG` | `0` |
| `xA` | `0` |
| `yA` | `0` |
| `round` | `0` |
| `op` | `0` |
| `xR` | `0` |
| `yR` | `0` |
| `λ` | `0` |
| `q0` | `r` |
| `c0` | `0` |
| `q1` | `r` |
| `c1` | `0` |
| `q2` | `r` |
| `c2` | `0` |
| `next_op` | `0` |
| `μ` | `0` |

## EC-Scalar

### Columns

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which to serve the constant |
| `ptr` | `DWordWL` | pointer to the first byte of the scalar |
| `offset` | `Byte` | index of limb |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `limb_bits` | `Bit[8]` | bit-decomposition of the limb being read |
| `last_limb` | `Bit` | whether this is the last limb to read |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `limb` | `Byte` | limb being read |

**Definition of `limb`:**
```
limb := Σ_i = 0^7 2^i * limb_bits[i]
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

### Assumptions

This chip makes an assumption:

| Tag | Range | Description |
|-----|-------|-------------|
| `EC_SCALAR-A1` |  | $#`ptr` + #`offset`$ does not overflow the bottom limb |

### Constraints

The chip starts by extracting the input information from the bus when its multiplicity is set.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `EC_SCALAR-C1` | `SERVE_K[timestamp, ptr, offset]` | -μ |
| `EC_SCALAR-C2` | `IS_BIT<μ>` |  |

Next, it reads `limb` from address ``ptr` + `offset``. Note that the read-timestamp is offset by `1` to prevent a collision with read of `k` performed by . Since `limb` is reconstructed from `limb_bits`, it is ensured those are in fact bits.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `EC_SCALAR-C3` |  | `MEMW[[limb, 0, 0, 0, 0, 0, 0, 0]; 0, ptr + [offset::Word, 0], [limb, 0, 0, 0, 0, 0, 0, 0], timestamp + 1::DWordWL, 0, 0, 0]` | μ |
| `EC_SCALAR-C4.i` | i ∈ [0, 7] | `IS_BIT<limb_bits[i]>` |  |

For each `limb_bit` that is set, an `BIT`-interaction is sent on the bus, to inform the double-and-add sequence on the  chip. To prevent interactions from occurring in padding rows, an active limb bit requires a non-zero multiplicity.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `EC_SCALAR-C5.i` | i ∈ [0, 7] | `BIT[timestamp, 8 * offset + i]` | limb_bits[i] |
| `EC_SCALAR-C6.i` | i ∈ [0, 7] | `limb_bits`_i = 1 => `μ` = 1 |  |
| | | _polynomial:_ `limb_bits[i] * (1 - μ) = 0` | |

Unless this was the `last_limb` (i.e., ``offset` = 0`), we recurse on serving the previous limb.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `EC_SCALAR-C7` | `IS_BIT<last_limb>` |  |
| `EC_SCALAR-C8` | `last_limb` = 1 => `μ` = 1 |  |
| | _polynomial:_ `last_limb * (1 - μ) = 0` | |
| `EC_SCALAR-C9` | `last_limb` = 1 => `offset` = 0 |  |
| | _polynomial:_ `last_limb * offset = 0` | |
| `EC_SCALAR-C10` | `SERVE_K[timestamp, ptr, offset - 1]` | μ - last_limb |

`last_limb` is a witness provided by the prover, which, technically, could be kept at `0` when ``offset` = 0`. However, that would require an additional `2^64` table entries to balance out the LogUp bus. Since this is assumed infeasible, the prover is constrained to set `last_limb` appropriately.

### Padding

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `ptr` | `0` |
| `offset` | `0` |
| `limb_bits` | `[0, 0, 0, 0, 0, 0, 0, 0]` |
| `last_limb` | `0` |
| `μ` | `0` |

## Notes / optimizations

- To utilize the  /  chips for different curves, consider introducing a lookup table for the curve-constants `a`, `b`, `p`, `r` and `N`, and look them up when a scalar multiplication selects them. The selection procedure could be done through the `ECALL` number; the  chip would accept multiple numbers, setting an internal "curve-selector" field accordingly. - Transitioning from `U256BL`s to `U256HL`s would roughly halve the number of columns in both the  and  chips. This would likely require increasing the sizes of the carries from 16 to 24 bits. Since the carries need to be range checked, one would have to investigate whether - it would be possible to perform a 24-bit range-check lookup, - one could set up a 24-bit range-check table. This could be as narrow as two columns. - have some hybrid version, where there is a native lookup table for x-bits, and a dynamic table for outliers (high carries are not encountered frequently).

## Discussing the carries <ecsm-limb_carry>

To constrain `x2` and `y_G` in , and `lambda`, `x_R` and `y_R` in , we use (variations of) the same technique: - multiplications are performed limb-by-limb, - a set of carry-limbs is used to exchange the underflow/overflow from one limb to another, and - the carry limbs are range constrained to ensure only one output value is possible.

We now explore this carry-technique and provide some proofs.

### Lemma 1

Let `V in NN` and `A,M in [0, V)`. For `i >= 1`, we define $ r_i &:= A (V-1) + M sum_(j=1)^i (V-1)^2 = i M(V-1)^2 + A(V-1),\ v_i &:= r_i + c_(i-1) mod V,\ c_i &:= V^(-1) (r_i + c_(i-1) - v_i),\ c_0 &:= 0 $ It holds that $ c_i = i M(V-1) + A - M - delta_(M<A) $ where kronecker delta `delta_x` equals `1` if `x` is true, and `0` otherwise.

For `i = 1`, we find that $ r_1 &= M(V - 1)^2 + A(V-1) \ &= M(V^2-2V) + (A-delta) V + delta V + M - A \ v_1 &equiv delta V + M - A mod V\ c_1 &= V^(-1) (M(V^2 - 2V) + (A-delta) V)\ &= M(V-2) + A-delta $ Suppose the statement to hold for arbitrary `i >= 1`. We find that $ d_(i+1) &= (i+1)M(V-1)^2 + A(V-1)\ v_(i+1) &equiv (i+1)M(V^2 - 2V) + (i+1)M + A V - A + i M(V-2) + (i-1)M + A-delta &&mod V\ &equiv (i+1)M(V^2 - 2V) + (A + i M - delta)V + delta (V-1) &&mod V\ &equiv delta (V-1) &&mod V\ c_(i+1) &= V^(-1) dot ((i+1)M(V^2 - 2V) + V(A + i M - delta))\ &= (i+1)M(V - 2) + A + i M - delta $ `qed`

### Corollary 1

Let `L` be a number of limbs, `b` be the number of bits per limb, `M in [0, L)` the number of multiplications in the formula, and `A in [0, L)` the number of additions. The maximum value of the carry is $ L M (2^b-1) + A - M - delta_(M < A) $

Applying the corollary to the relations $ x_G^2 - `x2` - q_0 dot p &= 0,\ y_G^2 - x_G dot `x2` - b + (p - q_1)p &= 0,\

lambda^2 - x_A - x_G - x_R + (1-`op`) (x_G - x_A) + (`r` - q_1) p &= 0,\ lambda (x_A - x_R) - y_A - y_R + (`r` - q_2) p &= 0.\ $ We find that the carries for sixteen 8-bit limbs are in the range $ (1): [-8160, 8159]\ (2): [-16319, 16318]\ (3): [-32636, 24477]\ (4): [-8161, 16318]\ (5): [-16320, 16318]\ $