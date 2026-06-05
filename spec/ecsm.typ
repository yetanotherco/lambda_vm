#import "/book.typ": book-page, et
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
An elliptic curve $E(a, b, p)$ in _short Weierstrass_ form has parameters $a,b in FF_p$ for some prime $p$ with $a^3+27b^2 eq.not 0$, and coordinates $(x, y) in FF_p^2$ satisfying the equation 
$
  y^2=x^3+a x+b.
$

#strong("Point at infinity.")
Additionally, there is the _point at infinity_, $⁠#inf$, which has no native short-Weierstrass representation.
It acts as the identity element (zero) in the group:
given non-zero curve point $P$, it holds that
$
  #inf + #inf &= #inf\
  #inf + P &= P\
$

#strong("Point negation.")
The negation of curve point $P = (x_P, y_P)$ is constructed as $-P := (x_P, -y_P)$.
Naturally, $P + (-P) = #inf$.

#strong("Point addition.")
The addition of points $P, Q$ distinguishes two cases.
For $x_P eq.not x_Q$, one uses
$ 
(x_R, y_R) := (lambda^2 - x_P - x_Q, lambda (x_P - x_R) - y_P)
$
with $lambda = frac((y_Q - y_P), (x_Q - x_P), style: "horizontal")$.
When $x_P = x_Q$ and $y_P eq.not - y_Q$, one instead uses $lambda = frac(3x_P^2, 2y_P, style: "horizontal")$.
Note, the remaing case that $(x_P, y_P) = (x_Q, -y_Q)$ corresponds with $Q = -P$; the addition results in $#inf$.

#strong("Scalar multiplication.")
An addition operation gives rise to an algorithm for scalar multiplication.
Given curve point $A$ and scalar $k$, the multiple $k times A$ can trivially be computed as $A + A + ... + A$.
This accelerator instead leverages the _double-and-add_ #footnote(link("https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Double-and-add")) technique, which utilizes only $O(log(k))$ additions for the full multiplication.

#strong("This accelerator.")
The purpose of this accelerator is to speed up the scalar multiplication $k times G$ for scalar $k in [1, N)$ and point $G in E(0, b, p) without {#inf}$ with $p in [2^248, 2^256)$.
In particular, the accelerator supports the curve $#`secp256k1` = E(0, 7, 2^256-2^32 - 977)$.
This accelerator leverages _double-and-add_, executing the multiplication in $D_k := ceil.l log_2(k) ceil.r$ doublings and $A_k := w_H (k) - 1$ additions, where $w_H (dot)$ denotes the hamming-weight of a bitstring.

= Overview
The accelerator comprises three chips:
- *`ECSM` (Elliptic Curve Scalar Multiply)*; this chip is responsible for loading inputs $x_G$ and $k$ from memory,
  reconstructing $y_G$,
  dispatching a double-and-add sequence request to the `ECDAS` chip, and writing the result point $x_R$ back to memory.
- *`ECDAS` (Elliptic Curve Double/Add Sequence)* is responsible for the consecutive doubling/adding the provided point to itself, ultimately arriving at $k times G$.
- *`LOAD_K`* serves $k$ bit-by-bit to the `ECDAS` chip to inform the flow of the double-and-add sequence.

= ECSM <ecsm-sm>

The #ecsm (Elliptic Curve Scalar Multiply) chip a generic over the constants
- $a$, the first curve coefficient,
- $b$, the second curve coefficient,
- $p$, the prime field modulus, and
- $N$, the order of the curve group.
To support scalar multiplication over different curves, one chip instance should be created for each curve.

The chip is triggered by executing `ECALL`, with the ECALL-number is set to $-3$.
The chip expects 
- `x10` to contain the address where $x_R := (k times G)_x$ is to be stored, 
- `x11` to contain the address at which the first byte of $x_G$ is to be found,
- `x12` to contain the address at which the first byte of $k$ is to be found.

== Columns
#let nr_variables = total_nr_variables(ecsm_chip)
#let nr_columns = total_nr_instantiated_columns(ecsm_chip, config)
#let nr_interactions = compute_nr_interactions(ecsm_chip)

The #ecsm chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(ecsm_chip, config)

== Constraints

=== Interactions
This chip is triggered by an `ECALL` with the opcode indicating this chip:
#render_constraint_table(ecsm_chip, config, groups: "ecall")

=== Read `xG`
Once triggered, it loads register `x11` to see where $x_G$ is stored in memory (@ec:c:read_addr_xG) and subsequently load $x_G$ in (@ec:c:read_xG).
Note here that @ec:c:verify_addr_xG_alignment enforces the requirement that $#`addr_xG[0]` in [0, 2^16 - 24)$; this is to ensure no overflows happen when incrementing the address in @ec:c:read_xG.
Note: `xG` is assumed to be range checked, since they're read from memory.
#render_constraint_table(ecsm_chip, config, groups: "read_xG")

=== Constrain `Gy`
With $x_G$ read and range checked, we direct our attention to $y_G$.
Rather than reading it from memory, the prover provides it as a witness and proves it to be correct.
In particular, the chip enforces the relations 
$
  x_G^2 - #`x2` - #`q0` dot p &= 0,\
  y_G^2 - x_G dot #`x2` - b + (p - #`q1`)p &= 0\
$
where `q0` and `q1` are prover-provided witnesses.
Note that these are equivalent to
$
  #`x2` &equiv x_G^2 mod p,\
  y_G^2 &equiv x_G dot #`x2` + b  mod p\
$
which combine to $y_G^2 equiv x_G^3 + b mod p$.
Rewriting the two statements in terms of the `q`s, we get
$
  #`q0` &= (x_G^2 - #`x2`) dot p^(-1),\
  #`q1` &= (y_G^2 - x_G dot #`x2`-b) dot p^(-1) + p.
$
Using the fact that $x_G, y_G, #`x2` in [0, p)$, we find that
$
  (x_G^2 - #`x2`) dot p^(-1) &in [0, p-2),\
  (y_G^2 - x_G dot #`x2`-b) dot p^(-1) + p &in [0, 2p-2).
$
Hence, we must restrict the choice of quotients to $#`q0` in [0, 2^256)$ and $#`q1` in [0, 2^257)$.
Below, we enforce the first of the two sub-relations.
We emphasize here that @ec:c:c0_63_is_zero is required to ensure the sum evaluates to $0$, rather than $0 mod 2^256$.
The constraints @ec:c:c0_0 and @ec:c:c0_i, as well as the magic number $8160$ in @ec:c:range_c0 are discussed in @ecsm-limb_carry.
#render_constraint_table(ecsm_chip, config, groups: "xG2")

Next, we restrict the witness pair $(y_G, #`q1`)$:

#render_constraint_table(ecsm_chip, config, groups: "yG")

=== Read and verify `k`
After reading `addr_k` from `x12` (@ec:c:read_addr_k), we read `k` from this address (@ec:c:load_k).
Similar to `addr_xG`, we assume that $#`addr_k[0]` in [0, 2^16 - 24)$ (@ec:c:verify_addr_k_alignment).
To prevent the point at infinity from showing up during the scalar multiplication, we require that $#`k` < #`N`$.
This is achieved by requiring that the addition $#`N` + (#`k` - #`N`)$ overflows $mod 2^256$ (@ec:c:k_lt_N).
Additionally, @ec:c:k_gt_0 ensures that $#`k` > 0$, preventing one case where $#`k` times #`G` = #inf$.
#render_constraint_table(ecsm_chip, config, groups: "verify_k")

=== Subroutine
With point $G$ and scalar $k$ fully constructed, we delegate bit-by-bit serving of the scalar `k` to the `EC-SCALAR` chip.
Here, we capture the index of the most significant 1-bit of `k` in `len_k`; if the index of a different bit is captured, the logup will not balance, as the skipped bits will not be consumed by the `ECDAS` chip.
Next, we interact with the `ECDAS` chip, providing `G` both as the accumulator, and increment (@ec:c:start_double_add).
After completing its double-and-add sequence, the result is captured in `R` (@ec:c:receive_double_add).
#render_constraint_table(ecsm_chip, config, groups: "delegate")

=== Range check `xR`
Before storing $x_R$, it is verified that $x_R in [0, p)$.
To this end, witness $#`xR_sub_P` := #`xR` - p mod 2^256$ is added to `p`; if the addition sums to `xR` and overflows $mod 2^256$, it must hold that $#`xR` < p$.
The addition is constrained by requiring that `c3` are bits (@ec:c:range_c3); an overflow occurs if and only if $#`c3[7]` = 1$ (@ec:c:xR_addition_overflows).

#render_constraint_table(ecsm_chip, config, groups: "range_xR")

=== Write `xR`
We read `addr_xR` from register `x10` (@ec:c:load_addrR), and subsequently write `xR` to this address (@ec:c:write_xR).
Similar to `addr_xG` and `addr_k`, we require that $#`addr_xR[0]` in [0, 2^16 - 24)$ (@ec:c:verify_addrR_alignment).
#render_constraint_table(ecsm_chip, config, groups: "write_xR")

= ECDAS chip <ecdas>
#let ecdas_chip = load_chip("src/ecdas.toml", config)
#let ecdas = raw(ecdas_chip.name)

The #ecdas chip (Elliptic Curve Double/Add Sequence) is responsible for accelerating the addition of two curve points, or the doubling of a single curve point. 
More specifically, given curve points $A$ (accumulator) and $G$ (generator), and selector bit `op`, it performs the mapping
$
  (A, G) mapsto cases(
    (A + A, &G) &text("if") #`op` = 0,
    (A + G, &G) &text("if") #`op` = 1
  )
$

== Doubling and adding
To add two curve points $A, B in E(0, b, p)$, we must consider three situations.
When $x_A eq.not x_B$, we construct the sum $R := A + B$ as
$
 lambda &:= frac((y_B - y_A), (x_B - x_A), style: "horizontal"), #h(3em)
  x_R &:= lambda^2 - x_A - x_B, #h(3em)
  y_R &:= lambda (x_A - x_R) - y_A.
$
Second, when $x_A = x_B$ and $y_A eq.not -y_B$, we compute $R := A + B = 2A$ as
$
  lambda &:= frac(3x_A^2, 2y_A, style: "horizontal"), #h(3em)
  x_R &:= lambda^2 - 2x_A, #h(3em)
  y_R &:= lambda (x_A - x_R) - y_A.
$
Lastly, when $x_A = x_B$ and $y_A eq -y_B$, $R$ becomes the 'point at infinity'; a point that has no native representation on the curve. 
It is, however, ensured by the #ecsm chip that this case cannot occur.
As such, we do need to consider it.


== Columns
#let nr_variables = total_nr_variables(ecdas_chip)
#let nr_columns = total_nr_instantiated_columns(ecdas_chip, config)
#let nr_interactions = compute_nr_interactions(ecdas_chip)

The #ecdas chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(ecdas_chip, config)

== Constraints
First, the chips receives the input for this double/add step:
#render_constraint_table(ecdas_chip, config, groups: "receive")

The `op`-flag determines whether $R := A + G$ (0) or $R:= 2A$ (1).
As previously discussed, this flag influences the computations of $lambda$ and $x_R$.
Rather than computing both potential values and selecting the correct one based on the `op` flag, we merge the relations that have to be checked and "weave" the `op`-flag in this way:
In particular, we let the prover provide witnesses $lambda$, $x_R$ and $y_R$ and we will prove that
$
#`op` dot (lambda (x_G - x_A) - (y_G - y_A)) + (1 - #`op`) dot (2 lambda y_A - 3x_A^2)  &equiv 0 mod p\
lambda^2 - x_A - (1- #`op` ) dot x_A - #`op` dot x_G  - x_R &equiv 0 mod p\
lambda (x_A - x_R) - y_A - y_R &equiv 0 mod p
$

To start, we let the prover provide witness $#`q0` in [-2^255, 2^255)$ and have them prove that
$
  #`op` dot (lambda dot x_G - lambda dot x_A + y_A - y_G) + (1 - #`op`) dot (2 lambda dot y_A - 3x_A dot x_A) + #`q0` dot p = 0
$

#render_constraint_table(ecdas_chip, config, groups: "lambda")

With $lambda$ constrained, we continue with $x_R$.
Here, we let the prover provide witness $#`q1` in [-2^255, 2^255)$ and have them prove that
$
  lambda^2 - x_R - x_A - x_G - #`op` dot (x_A - x_G) - #`q1` dot p = 0
$

#render_constraint_table(ecdas_chip, config, groups: "xR")

Next, we constrain $y_R$.
Rewriting the earlier equality, we find that 
$
  lambda dot x_A - lambda dot x_R - y_A - y_R + #`q2` dot p = 0
$
for some prover-provided witness $#`q2` in [-2^255, 2^255)$.

#render_constraint_table(ecdas_chip, config, groups: "yR")

Lastly, the updated accumulator is sent out for the next step to be processed (@ecdas:c:send).
To determine whether the next step should be an addition or doubling, the `next_op` bit is provided as witness by the prover.
Setting this bit to 1 can only be done in active rows (@ecdas:c:next_op_implies_mu) and does require the scalar bit in this position to be set (@ecdas:c:receive_next_op).
#render_constraint_table(ecdas_chip, config, groups: "send")


= EC-Scalar
#let ecscalar_chip = load_chip("src/ec_scalar.toml", config)
#let ecscalar = raw(ecscalar_chip.name)

== Columns
#let nr_variables = total_nr_variables(ecscalar_chip)
#let nr_columns = total_nr_instantiated_columns(ecscalar_chip, config)
#let nr_interactions = compute_nr_interactions(ecscalar_chip)

The #ecdas chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(ecscalar_chip, config)

== Constraints

#render_constraint_table(ecscalar_chip, config)

= Notes / optimizations
- To merge the #ecsm / #ecdas chips for different curves, consider introducing a lookup table for the curve-constants $a$, $b$, $p$ and $N$, and include them for each scalar multiplication when they're selected.
The selection procedure could be done through the `ECALL` number; the #ecsm chip would accept multiple numbers, setting an internal "curve-selector" field accordingly.

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
  x_G dot x_G - (#`q0` dot p + #`x2`) &= 0\
  y_G dot y_G - (#`x2` dot x_G + #`q1` dot p) &= 0\
  
   lambda dot x_G + #`q0` dot p + y_A - (lambda dot x_A + y_G) &= 0\
   2 lambda dot y_A + #`q0` dot p - 3x_A dot x_A  &= 0\
  lambda^2 - (#`q1` dot p + x_R + 2x_A) &= 0\
  lambda^2 - (#`q1` dot p + x_R + x_A + x_G) &= 0\
  lambda dot x_A + #`q2` dot p - (lambda dot x_R + y_A + y_R)  &= 0\
$
We find that the carries for sixteen 8-bit limbs are in the range
$
  (1): [-8160, 8159]\
  (2): [-16318, 8159]\
  (3): [-24477, 24477]\
  (4): [-8161, 8159]\
  (5): [-8160, 16318]\
$