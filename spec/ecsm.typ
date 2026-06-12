#import "/book.typ": book-page, aside, et
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_assumptions,
  render_chip_padding_table,
  compute_nr_interactions
)

#let config = load_config()

#show: book-page("ecsm.typ")

#let ecsm_chip = load_chip("src/ecsm.toml", config)
#let ecsm = raw(ecsm_chip.name)

= Theory behind Elliptic Curves

#let inf = math.cal("O")
An elliptic curve $E(a, b, p)$ in _short Weierstrass_ form has parameters $a,b in FF_p$ for some prime $p$ with $4a^3+27b^2 eq.not 0$, and coordinates $(x, y) in FF_p^2$ satisfying the equation 
$
  y^2=x^3+a x+b.
$

#strong("Point at infinity.")
Additionally, there is the _point at infinity_, $⁠#inf$, which has no native short-Weierstrass representation.
It acts as the identity element (zero) in the group:
given non-zero curve point $P$, it holds that
$
  #inf + #inf &= #inf,\
  #inf + P &= P.\
$

#strong("Point negation.")
The negation of curve point $P = (x_P, y_P)$ is constructed as $-P := (x_P, -y_P)$.
Naturally, $P + (-P) = #inf$.

#strong("Point addition.")
The addition of points $P, Q$ distinguishes three cases.
For $x_P eq.not x_Q$, one uses
$ 
(x_R, y_R) := (lambda^2 - x_P - x_Q, lambda (x_P - x_R) - y_P)
$
with $lambda = frac((y_Q - y_P), (x_Q - x_P), style: "horizontal")$.
When $x_P = x_Q$ and $y_P eq.not - y_Q$, one instead uses $lambda = frac(3x_P^2, 2y_P, style: "horizontal")$.
The remaing case that $(x_P, y_P) = (x_Q, -y_Q)$ corresponds with $Q = -P$; the addition results in $#inf$.

#strong("Scalar multiplication.")
An addition operation gives rise to an algorithm for scalar multiplication.
Given curve point $P$ and scalar $k$, the multiple $k times P$ can trivially be computed as $P + P + ... + P$.
This accelerator instead leverages the _double-and-add_ #footnote(link("https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Double-and-add")) technique, which utilizes only $O(log(k))$ additions for the full multiplication.

#strong("This accelerator.")
The purpose of this accelerator is to speed up the scalar multiplication $k times G$ for scalar $k in [1, N)$ and point $G in E(0, b, p) without {#inf}$ with $p in [2^248, 2^256)$.
In particular, the accelerator supports the curve $#`secp256k1` = E(0, 7, 2^256-2^32 - 977)$.
This accelerator leverages _double-and-add_, executing the multiplication in $O(log(k))$ doublings and $O(w_H (k)) = O(log(k))$ additions, where $w_H (dot)$ denotes the hamming-weight of a bitstring.

= Overview
The accelerator comprises three chips:
- *`ECSM` (Elliptic Curve Scalar Multiply)*; this chip is responsible for loading inputs $x_G$ and $k$ from memory,
  reconstructing $y_G$,
  dispatching a double-and-add sequence request to the `ECDAS` chip, and writing the result point $x_R$ back to memory.
- *`ECDAS` (Elliptic Curve Double/Add Sequence)* is responsible for the consecutive doubling/adding the provided point to itself, ultimately arriving at $k times G$.
- *`EC_SCALAR`* serves $k$ bit-by-bit to the `ECDAS` chip to inform the flow of the double-and-add sequence.

= ECSM <ecsm-sm>

The #ecsm (Elliptic Curve Scalar Multiply) chip is generic over the constants
- $b$, the second curve coefficient,
- $p$, the prime field modulus, and
- $N$, the order of the curve group.
To support scalar multiplication over different curves, one chip instance should be created for each curve.

The chip is triggered by executing `ECALL`, with the ECALL-number is set to $-3$.
The chip expects 
- `x10` to contain the address where $x_R := (k times G)_x$ is to be stored, 
- `x11` to contain the address at which the least significant byte of $x_G$ is to be found,
- `x12` to contain the address at which the least significant byte of $k$ is to be found,
where it is assumed that $x_G, x_R$ and $k$ are provided as little-endian.

== Columns
#let nr_variables = total_nr_variables(ecsm_chip)
#let nr_columns = total_nr_instantiated_columns(ecsm_chip, config)
#let nr_interactions = compute_nr_interactions(ecsm_chip)

The #ecsm chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(ecsm_chip, config)

== Assumptions
#render_chip_assumptions(ecsm_chip, config)

== Constraints

=== Interactions
This chip is triggered by an `ECALL` with the opcode indicating this chip:
#render_constraint_table(ecsm_chip, config, groups: "ecall")

=== Read `xG`
Once triggered, it loads register `x11` to see where $x_G$ is stored in memory (@ec:c:read_addr_xG) and subsequently load $x_G$ in (@ec:c:read_xG).
Assumption @ec:a:addr_xG_alignment ensures no overflows happen when incrementing the address in @ec:c:read_xG.
Note: `xG` is assumed to be range checked, since they're read from memory.
#render_constraint_table(ecsm_chip, config, groups: "read_xG")

=== Constrain `Gy`
With $x_G$ read and range checked, we direct our attention to $y_G$.
Rather than reading it from memory, the prover provides it as a witness and proves it to be correct.
In particular, the chip enforces the relations 
$
  x_G^2 - #`x2` - q_0 dot p &= 0,\
  y_G^2 - x_G dot #`x2` - b + (p - q_1)p &= 0\
$
where non-negative $q_0$ and $q_1$ are prover-provided witnesses.
Note that these are equivalent to
$
  #`x2` &equiv x_G^2 mod p,\
  y_G^2 &equiv x_G dot #`x2` + b  mod p\
$
which combine to $y_G^2 equiv x_G^3 + b mod p$.
Rewriting the two relations, we get
$
  q_0 &= (x_G^2 - #`x2`) dot p^(-1),\
  q_1 &= (y_G^2 - x_G dot #`x2`-b) dot p^(-1) + p.
$
Using the fact that $x_G, y_G, #`x2` in [0, p)$, we find that $q_0 in [0, p)$ and $q_1 in [0, 2p)$.
We therefore restrict the choice of quotients to $q_0 in [0, 2^256)$ and $q_1 in [0, 2^257)$.

Below, we enforce the first of the two sub-relations.
We emphasize here that @ec:c:c0_63_is_zero is required to ensure the sum evaluates to $0$, rather than just $0 mod 2^256$.
The constraints @ec:c:c0_0 and @ec:c:c0_i, as well as the magic number $8160$ in @ec:c:range_c0 are discussed in @ecsm-limb_carry.
#render_constraint_table(ecsm_chip, config, groups: "xG2")

Next, we restrict the witness pair $(y_G, #`q1`)$.
Note there that @ec:c:c1_0 and @ec:c:c1_i multiply `B` by `μ` to simplify the padding; there are no other side-effects to this since $#`μ` = 1$ on non-padding rows (@ec:c:mu_isbit).

#render_constraint_table(ecsm_chip, config, groups: "yG")

=== Read and verify `k`
After reading `addr_k` from `x12` (@ec:c:read_addr_k), we read `k` from this address (@ec:c:load_k).
Similar to `addr_xG`, assumption @ec:a:addr_k_alignment ensures the address offsets in @ec:c:load_k do not overflow the lower limb.
To prevent the point at infinity from showing up during the scalar multiplication, we require that $#`k` < #`N`$.
This is achieved by requiring that the addition $#`N` + (#`k` - #`N`)$ overflows $mod 2^256$ (@ec:c:k_lt_N).
Additionally, @ec:c:k_gt_0 ensures that $#`k` > 0$, preventing a case where $#`k` times #`G` = #inf$.
#render_constraint_table(ecsm_chip, config, groups: "verify_k")

=== Subroutine
With point $G$ and scalar $k$ fully constructed, we delegate bit-by-bit serving of the scalar `k` to the `EC_SCALAR` chip.
Here, we capture the index of the most significant 1-bit of `k` in `idx_k`.
Note: if the prover decides to capture a lesser significant bit here, the LogUp will not balance, as the skipped bits will never taken off the bus.
Next, we interact with the `ECDAS` chip, providing `G` both as the accumulator, and increment (@ec:c:start_double_add); we specifically instruct the chip to start with a _double_-operation.
After completing its double-and-add sequence, the result is captured in `R` (@ec:c:receive_double_add).
#render_constraint_table(ecsm_chip, config, groups: "delegate")

=== Range check `xR`
Before storing $x_R$, it is verified that $x_R in [0, p)$.
To this end, witness $#`xR_sub_p` := #`xR` - p mod 2^256$ is added to `p`; if the addition sums to `xR` and overflows $mod 2^256$, it must hold that $#`xR` < p$.
The addition is constrained by requiring that `c3` are bits (@ec:c:range_c3); an overflow occurs if and only if $#`c3[7]` = 1$ (@ec:c:xR_addition_overflows).

#render_constraint_table(ecsm_chip, config, groups: "range_xR")

=== Write `xR`
We read `addr_xR` from register `x10` (@ec:c:load_addrR), and subsequently write `xR` to this address (@ec:c:write_xR).
Note that the `timestamp` on both memory accesses is offset to allow `addr_xR` to equal `addr_xG` and thus for $x_R$ to overwrite $x_G$ in memory.
Similar to `addr_xG` and `addr_k`, it is assumed that the addition of the small offsets will not overflow the lower limb of `addr_xR` (@ec:a:addr_xR_alignment).
#render_constraint_table(ecsm_chip, config, groups: "write_xR")

== Padding
#render_chip_padding_table(ecsm_chip, config)


= ECDAS chip <ecdas>
#let ecdas_chip = load_chip("src/ecdas.toml", config)
#let ecdas = raw(ecdas_chip.name)

The #ecdas chip (_Elliptic Curve Double-and-Add Sequence_) is responsible for accelerating the addition of two curve points, or the doubling of a single curve point. 
More specifically, given curve points $A$ (accumulator) and $G$ (generator), and selector bit `op`, it performs the mapping
$
  (A, G) mapsto cases(
    (A + A, &G) &text("if") #`op` = 0,
    (A + G, &G) &text("if") #`op` = 1
  )
$

Recall that the addition of two curve points $A, B$ is treated differently based on three cases:
#enum(indent: 1em, numbering: n => strong(raw("Case "+str(n)+".")),
  enum.item[$x_A eq.not x_B$],
  enum.item[$x_A eq x_B$ and $y_A eq.not -y_B$, or],
  enum.item[$x_A eq x_B$ and $y_A eq -y_B$]
)
where _double_ may encounter the last two cases, while _add_ may encounter all three.
Cases 2 and 3 may, for specific inputs, evaluate to $#inf$:
a point that has no native short-Weierstrass representation.
Therefore, the #ecsm and #ecdas chips were designed to avoid this case.
To see how, note that #ecsm
+ is the sole chip that can "activate" the #ecdas chip by issuing an `ECDAS` lookup,
+ enforces that $G$ and the initial $A$ do not equal $#inf$, and
+ ensures $k in [1, N)$, where $N$ denotes the order of the curve.
This combined yields that neither doubling $A$ or adding $A + G$ can produce $#inf$:

*Double.*
For $2A$ to equal $#inf$, the curve must have _even_ order.
Since the order of the `secp256k1` curve is _odd_, such a point does not exist.

*Add.*
If $A + G = #inf$, then $A = -G = #inf - G = r N G - G$ for some $r >= 0$.
Because #ecsm initializes $A = G eq.not #inf$, it must hold that $r >= 1$.
Furthermore, the restriction that $k <= N-1$ ensures $r <= 1$.
Hence, $A = (N-1)G$.
Since $N-1$ is the maximal value of $k$, the previous round producing $A = (N-1)G$ was the last round of this scalar multiplication.
This means that now `round` is negative, which will fail constraint @ecdas:c:range_round.


== Columns
#let nr_variables = total_nr_variables(ecdas_chip)
#let nr_columns = total_nr_instantiated_columns(ecdas_chip, config)
#let nr_interactions = compute_nr_interactions(ecdas_chip)

The #ecdas chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(ecdas_chip, config)

== Constraints
First, the chips receives the input for this double/add step:
#render_constraint_table(ecdas_chip, config, groups: "receive")

=== Operation switching
The `op`-flag determines whether $R := 2A$ (0) or $R:= A+G$ (1).
This chip introduces a set of constraints that properly constrains $R$ depending on this flag.
To illustrate how this is achieved, we split addition up in three relations:
$
  lambda &equiv (y_G - y_A)/(x_G - x_A) &&mod p,\
  x_R &equiv lambda^2 - x_A - x_G &&mod p,\
  y_R &equiv lambda (x_A - x_R) - y_A &&mod p.\
$
Introducing the non-negative witnesses $q'_0, q'_1$ and $q_2$, we can convert these relations into
$
  lambda (x_G - x_A) - y_G + y_A + (#`r` - q'_0) p &= 0,\
  lambda^2 - x_A - x_G - x_R + (#`r` - q'_1) p &= 0,\
  lambda (x_A - x_R) - y_A - y_R + (#`r` - q_2) p &= 0,\
$
for some $r in NN$ to be fixed later.

#aside("The case of " + $x_A = x_G$ + ".")[
  Special attention should be paid to the first relation: if $x_A = x_G$, $lambda$ can be chosen freely.
  By design, this situation cannot occur.

  Observe that this would require either $A = G$ or $A = -G$.
  With the latter situation previously ruled out, only the first remains.
  For $A = (r N + 1) G$ for some $r in NN$ and $N$ the order of the curve, all cases with $r>0$ can be ruled out since #ecsm verifies that the scalar $k < N$.
  The remaining case $A=G$ is the intial state pushed onto the LogUp by #ecsm (@ec:c:start_double_add), with `op`-flag set to $0$ (_double_), not `add`.
  Hence, this situation cannot occur.  
]

We rewrite the relations to find
$
  q'_0 &= #`r` + p^(-1) dot (lambda (x_G - x_A) - y_G + y_A),\
  q'_1 &= #`r` + p^(-1) dot (lambda^2 - x_A - x_G - x_R),\
  q_2  &= #`r` + p^(-1) dot (lambda (x_A - x_R) - y_A - y_R)\
$
from which we can conclude that $q'_0, q_2 in (#`r`-p, #`r`+p)$ and $q'_1 in (#`r`, #`r` + p)$.
When doubling, only the formulae for $lambda$ and $x_R$ are different:
$
  lambda &equiv (3x_A^2)/(2y_A) &&mod p,\
  x_R &equiv lambda^2 - 2x_A &&mod p.\
$
Introducing non-negative witnesses $q''_0$ and $q''_1$, we convert these into
$
  2lambda y_A - 3x_A^2 + (#`r` - q''_0) p &= 0,\
  lambda^2 - 2x_A - x_G - x_R + (#`r` - q''_1) p &= 0.\
$
#aside("The case of " + $y_A = 0$ + ".")[
  Special attention should be paid to the first relation: if $y_A = 0$, $lambda$ can again be chosen freely.
  As previously established, $y_A != 0$ for all points on the `secp256k1` curve.
  Hence, this situation will not occur.
]
Reordering yields
$
  q''_0 &= #`r` + p^(-1) dot (2lambda y_A - 3x_A^2 ),\
  q''_1 &= #`r` + p^(-1) dot (lambda^2 - 2x_A - x_G - x_R ).\
$
where $q''_0 in (#`r`-3p, #`r` + 2p)$, and $q''_1 = (#`r`, #`r` + p)$.
We can now leverage the `op`-flag to merge the relations for $lambda$ and $x_R$ into
$
  #`op` dot ((x_G - x_A)lambda - y_G + y_A) + (1-#`op`) (2lambda y_A - 3x_A^2) + (#`r` - q_0) p &= 0,\
  lambda^2 - x_A - x_G - x_R + (1-#`op`) (x_G - x_A) + (#`r` - q_1) p &= 0\
$
which yields
$
  q_0 &= #`r` + p^(-1) dot (#`op` dot ((x_G - x_A)lambda - y_G + y_A) + (1-#`op`) (2lambda y_A - 3x_A^2)),\
  q_1 &= #`r` + p^(-1) dot ((lambda^2 - x_A - x_G - x_R + (1-#`op`) (x_G - x_A)).\
$
with $q_0 in (r-3p, r+2p)$ and $q_1 in (r, r+p)$.
By setting $r := 3p$, we ensure $q_0 in (0, 5p), q_1 in (3p, 4p)$ and $q_2 in (2p, 4p)$ are non-negative for all inputs.

=== Constraining $lambda$
We start by establishing the relation
$
  #`op` dot (lambda (x_G - x_A) - y_G + y_A) + (1-#`op`) (2lambda y_A - 3x_A^2) + (#`r` - q_0) p &= 0.\
$
#render_constraint_table(ecdas_chip, config, groups: "lambda")


=== Constraining $x_R$
Secondly, we establish
$
  lambda^2 - x_A - x_G - x_R - (1-#`op`) (x_A - x_G) + (#`r` - q_1) p &= 0
$

#render_constraint_table(ecdas_chip, config, groups: "xR")

=== Constraining $y_R$
Third,
$
  lambda (x_A - x_R) - y_A - y_R + (#`r` - q_2) p &= 0
$
is constrained:

#render_constraint_table(ecdas_chip, config, groups: "yR")

Lastly, the updated accumulator is sent out for the next step to be processed (@ecdas:c:send).
To determine whether the next step should be an addition or doubling, the `next_op` bit is provided as witness by the prover.
Setting this bit to 1 can only be done in active rows (@ecdas:c:next_op_implies_mu), when the current $#`op` = 0$ (double), and does require the scalar bit in this position to be set (@ecdas:c:receive_next_op).
#render_constraint_table(ecdas_chip, config, groups: "send")

== Padding
#render_chip_padding_table(ecdas_chip, config)


= EC-Scalar
#let ecscalar_chip = load_chip("src/ec_scalar.toml", config)
#let ecscalar = raw(ecscalar_chip.name)

== Columns
#let nr_variables = total_nr_variables(ecscalar_chip)
#let nr_columns = total_nr_instantiated_columns(ecscalar_chip, config)
#let nr_interactions = compute_nr_interactions(ecscalar_chip)

The #ecscalar chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(ecscalar_chip, config)

== Assumptions
This chip makes an assumption:
#render_chip_assumptions(ecscalar_chip, config)

== Constraints
The chip starts by extracting the input information from the bus when its multiplicity is set.
#render_constraint_table(ecscalar_chip, config, groups: "recv")

Next, it reads `limb` from address $#`ptr` + #`offset`$.
Note that the read-timestamp is offset by $1$ to prevent a collision with read of $k$ performed by #ecsm.
Since `limb` is reconstructed from `limb_bits`, it is ensured those are in fact bits.
#render_constraint_table(ecscalar_chip, config, groups: "read")

For each `limb_bit` that is set, an `BIT`-interaction is sent on the bus, to inform the double-and-add sequence on the #ecdas chip. 
To prevent interactions from occurring in padding rows, an active limb bit requires a non-zero multiplicity.
#render_constraint_table(ecscalar_chip, config, groups: "serve")

Unless this was the `last_limb` (i.e., $#`offset` = 0$), we recurse on serving the previous limb.

#render_constraint_table(ecscalar_chip, config, groups: "recurse")
`last_limb` is a witness provided by the prover, which, technically, could be kept at $0$ when $#`offset` = 0$.
However, that would require an additional $2^64$ table entries to balance out the LogUp bus.
Since this is assumed infeasible, the prover is constrained to set `last_limb` appropriately.

== Padding
#render_chip_padding_table(ecscalar_chip, config)

= Notes / optimizations
- To utilize the #ecsm / #ecdas chips for different curves, consider introducing a lookup table for the
  curve-constants $a$, $b$, $p$, $r$ and $N$, and look them up when a scalar multiplication selects them.
  The selection procedure could be done through the `ECALL` number; the #ecsm chip would accept multiple numbers, setting an internal "curve-selector" field accordingly.
- Transitioning from `U256BL`s to `U256HL`s would roughly halve the number of columns in both the #ecsm and #ecdas chips.
  This would likely require increasing the sizes of the carries from 16 to 24 bits.
  Since the carries need to be range checked, one would have to investigate whether
    - it would be possible to perform a 24-bit range-check lookup,
    - one could set up a 24-bit range-check table. This could be as narrow as two columns.
    - have some hybrid version, where there is a native lookup table for x-bits, and a dynamic table for outliers (high carries are not encountered frequently).

= Discussing the carries <ecsm-limb_carry>
To constrain `x2` and $y_G$ in #ecsm, and $lambda$, $x_R$ and $y_R$ in #ecdas, we use (variations of) the same technique:
- multiplications are performed limb-by-limb, 
- a set of carry-limbs is used to exchange the underflow/overflow from one limb to another, and
- the carry limbs are range constrained to ensure only one output value is possible.

We now explore this carry-technique and provide some proofs.

== Lemma 1
Let $V in NN$ and $A,M in [0, V)$.
For $i >= 1$, we define
$
r_i &:= A (V-1) + M sum_(j=1)^i (V-1)^2 = i M(V-1)^2 + A(V-1),\
v_i &:= r_i + c_(i-1) mod V,\
c_i &:= V^(-1) (r_i + c_(i-1) - v_i),\
c_0 &:= 0
$
It holds that
$
c_i = i M(V-1) + A - M - delta_(M<A)
$
where kronecker delta $delta_x$ equals $1$ if $x$ is true, and $0$ otherwise.

#emph("Proof:")
For $i = 1$, we find that 
$
r_1 
&= M(V - 1)^2 + A(V-1) \
&= M(V^2-2V) + (A-delta) V + delta V + M - A \
v_1 
&equiv delta V + M - A mod V\
c_1 
&= V^(-1) (M(V^2 - 2V) + (A-delta) V)\
&= M(V-2) + A-delta
$
Suppose the statement to hold for arbitrary $i >= 1$.
We find that 
$
d_(i+1)
&= (i+1)M(V-1)^2 + A(V-1)\
v_(i+1)
&equiv (i+1)M(V^2 - 2V) + (i+1)M + A V - A + i M(V-2) + (i-1)M + A-delta &&mod V\
&equiv (i+1)M(V^2 - 2V) + (A + i M - delta)V + delta (V-1) &&mod V\
&equiv delta (V-1) &&mod V\
c_(i+1)
&= V^(-1) dot ((i+1)M(V^2 - 2V) + V(A + i M - delta))\
&= (i+1)M(V - 2) + A + i M - delta
$ 
$qed$

== Corollary 1
Let $L$ be a number of limbs, $b$ be the number of bits per limb, $M in [0, L)$ the number of multiplications in the formula, and $A in [0, L)$ the number of additions.
The maximum value of the carry is
$
  L M (2^b-1) + A - M - delta_(M < A)
$

Applying the corollary to the relations
$
  x_G^2 - #`x2` - q_0 dot p &= 0,\
  y_G^2 - x_G dot #`x2` - b + (p - q_1)p &= 0,\

  #`op` dot ((x_G - x_A)lambda - y_G + y_A) + (1-#`op`) (2lambda y_A - 3x_A^2) + (#`r` - q_0) p &= 0,\
  lambda^2 - x_A - x_G - x_R + (1-#`op`) (x_G - x_A) + (#`r` - q_1) p &= 0,\
  lambda (x_A - x_R) - y_A - y_R + (#`r` - q_2) p &= 0.\
$
We find that the carries for sixteen 8-bit limbs are in the range
$
  (1): [-8160, 8159]\
  (2): [-16319, 16318]\
  (3): [-32636, 24477]\
  (4): [-8161, 16318]\
  (5): [-16320, 16318]\
$