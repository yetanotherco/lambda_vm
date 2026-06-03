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

Let $E(a, b, FF)$ with $a^3+27b^2 eq.not 0$ and $FF$ a finite field describe a _short Weierstrass_ elliptic curve, with all points $(x, y) in FF^2$ on the curve satisfying the equation
$
y^2 = x^3 + a x + b.
$
Combined with definitions for addition and negation, these points form a group; let $N$ to denote its order.

The purpose of this accelerator is to speed up the scalar multiplication $k times G$ for scalar $k in [1, N)$ and point $G in E(0, a, FF_p)$ with prime $p < 2^256$.
Note that `secp256k1` corresponds to such a curve with $b=7$ and $p = 2^256 - 2^32 - 977$.
This accelerator leverages the so-called "double-and-add" method: the bit-decomposition of `k` is used to guide a sequence of doubling and adding $G$ to itself.
The accelerator executes $D_k := ceil.l log_2(k) ceil.r$ doublings and $A_k := w_H (k) - 1$ additions, where $w_H (dot)$ denotes the hamming-weight of a bitstring.

= Overview
The accelerator comprises three components:
- *`ECSM` (Elliptic Curve Scalar Multiply)*; this chip is responsible for loading inputs $x$ and $k$ from memory, 
  dispatching a double-and-add sequence request to the `ECDAS` chip, and writing the result point $x$ back to memory.
- *`ECDAS` (Elliptic Curve Double/Add Sequence)* is responsible for the consecutive doubling/adding the provided point to itself, ultimately arriving at $k times G$.
- *`LOAD_K`* serves $k$ bit-by-bit to the `ECDAS` chip to inform the flow of the double-and-add sequence.

= ECSM <ecsm-sm>

The #ecsm (Elliptic Curve Scalar Multiply) chip is parametrized by the constants
- $b$, the second curve coefficient,
- $p$, the prime field modulus, and
- $N$, the order of the curve group.

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
Once triggered, we look up where $x_#`G`$ is stored in memory (@ec:c:read_addr_xG) and subsequently load it in (@ec:c:read_xG).
Note here that @ec:c:verify_addr_xG_alignment enforces the requirement that $#`addr_xG[0]` + 24 < 2^16$; this is to ensure no overflows happen when incrementing the address in @ec:c:read_xG.
#render_constraint_table(ecsm_chip, config, groups: "read_xG")

=== Range check `xG`
Before proceeding, it is verified that $x_G in [0, p)$
#et[why was this again?]
To this end, witness `xG_sub_P` is added to `P`; if addition sums to `xG` and overflows $mod 2^256$, it must hold that $#`xG` < #`P`$.
The valid addition is constrained by requiring that `c2` are bits (@ec:c:range_c2); an overflow occurs if and only if $#`c2[7]` = 1$ (@ec:c:xG_addition_overflows).

#render_constraint_table(ecsm_chip, config, groups: "range_xG")

=== Constrain `Gy`
With $x_G$ read and range checked, we direct our attention to $y_G$.
Rather than reading it from memory, it is provided as witness by the prover and constrained to be correct.
In particular, we enforce that the relation $y^2 equiv x^3 + b mod p$ holds.

This relation is enforced in two steps.
First, it is established that witness `x2` satisfies $#`x2` equiv x_#`G`^2 mod p$.
Note that, for this to be the case, there must exist some quotient $#`q`_0 >= 0$ such that
$
  x_#`G`^2 - #`q`_0 dot #`p` - #`x2` = 0.
$

In the following, @ec:c:range_q0 ensures $#`q`_0 >= 0$ while @ec:c:range_x2 ensures `x2` has valid limbs.
@ec:c:c0_0 and @ec:c:c0_i ensure that the carries linking the limbs together are small valid.

#et[discuss limb decomposition & proving the carry]

#render_constraint_table(ecsm_chip, config, groups: "xG2")

With $#`x2` := x_#`G`^2 mod p$ in place, we can now rewrite the original relation as
$
  y_#`G`^2 = #`x2` dot x_#`G` + b mod p
$
Leveraging the same trick, we constrain witness pair $(y_#`G`, #`q`_1)$ such that
$
  y_#`G`^2 - #`x2` dot x_#`G` + #`q`_1 dot #`p` - #`b` = 0.
$
Note that this time around it may be that $#`q`_1 < 0$.

#render_constraint_table(ecsm_chip, config, groups: "yG")

=== Read and verify `k`
After reading `addr_k` from `x12` (@ec:c:read_addr_k), we read `k` from this address (@ec:c:load_k).
Similar to `addr_xG`, we assume that $#`addr_k` + 24 < 2^16$ (@ec:c:verify_addr_k_alignment).
To prevent the point at infinity from showing up during the scalar multiplication, we require that $#`k` < #`N`$.
This is achieved by requiring that the addition $#`N` + (#`k` - #`N`)$ overflows $mod 2^256$ (@ec:c:k_lt_N).
Additionally, @ec:c:k_gt_0 ensures that $#`k` > 0$, implying that $#`k` times #`G`$ will not be the point at infinity.
#render_constraint_table(ecsm_chip, config, groups: "verify_k")

=== Subroutine
With point $G$ and scalar $k$ fully constructed, we delegate bit-by-bit serving of the scalar `k` to the #et[todo] chip.
Here, we capture the index of the most significant 1-bit of `k` in `bitlen_k`; if the index of a lesser significant bit is captured, the logup will not balance, as the `ECDAS` chip will only consume those lower than the `bitlen_k` it is presented.
Next, we interact with the `ECDAS` chip, providing `G` both as the accumulator, and increment (@ec:c:start_double_add).
After completing its double-and-add sequence, the result is captured in `R` (@ec:c:receive_double_add).
#render_constraint_table(ecsm_chip, config, groups: "delegate")

=== Write `xR`
We read `addr_xR` from register `x10` (@ec:c:load_addrR), and subsequently write `xR` to this address (@ec:c:write_xR).
Similar to `addr_xG` and `addr_k`, we require that $#`addr_xR` + 24 < 2^16$ (@ec:c:verify_addrR_alignment).
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


// #render_constraint_table(ecdas_chip, config)

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
