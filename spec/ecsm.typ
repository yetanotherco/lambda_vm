#import "/book.typ": book-page, aside, attention
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

#show math.equation.where(block: false): box

#let ecsm_chip = load_chip("src/ecsm.toml", config)
#let ecsm = raw(ecsm_chip.name)

= Elliptic Curve Background

#let inf = math.cal("O")
An elliptic curve $E(a, b, p)$ in _short Weierstrass_ form has parameters $a,b in FF_p$ for some prime $p$ with $4a^3+27b^2 eq.not 0$, and points $(x, y) in FF_p^2$ satisfying the equation 
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

= Overview
This accelerator provides a compact way to prove the product $k times G$ for scalar $k in [1, N)$ and point $G in E(a, b, p) without {#inf}$ with $p in [3, 2^256)$ that induce curves of odd order.
In particular, the accelerator supports the curves `secp256k1` and `secp256r1`.

The accelerator serves two ECALL variants, selected by the `is_affine` column:
/ $x$-only ($#`is_affine` = 0$): the guest supplies $x_G$ (32 bytes) and receives $x_R := (k times G)_x$ (32 bytes). The matching $y_G$ is never read from memory; the prover witnesses it and the chip merely proves it to be _a_ root of the curve equation.
/ affine ($#`is_affine` = 1$): the guest supplies the full point $x_G ‖ y_G$ (64 bytes) and receives the full point $x_R ‖ y_R$ (64 bytes).

A single chip instance serves both variants: `is_affine` selects the ECALL-number the chip answers to, and gates the two memory accesses the affine variant adds (@ec:c:read_yG, @ec:c:write_yR) together with the address derivations (@ec:c:extrapolate_addr_yG, @ec:c:extrapolate_addr_yR) and address range checks (@ec:c:range_addr_yG, @ec:c:range_addr_yR) they need.
Every other constraint is shared between the two.
Of the constraints this variant adds, the ones _not_ gated on `is_affine` are the three of the $y_R < p$ check (@ec:c:range_yR_sub_p, @ec:c:range_c5, @ec:c:yR_addition_overflows), which are gated on `μ` and so apply on every active row — obliging the $x$-only path to witness a canonical $y_R$ as well — and @ec:c:is_affine_isbit, which carries no condition at all.
Returning $y_R$ spares the guest a second scalar multiplication: without it, recovering $y(k times G)$ means either a second query $x((k+1) times G)$ plus the chord-addition law, or a modular square root of $x_R^3 + a x_R + b$, which leaves the sign undetermined.

#attention("Variable space.")[
    This accelerator is _variable-space_ in the value of $k$; different values of $k$ may result in different table sizes.
    As such, *this accelerator should only be used for input sets with public $k$.*
]

The accelerator comprises two chips:
- *`ECSM` (Elliptic Curve Scalar Multiply)*.
    This chip is responsible for
    - loading $k$ from memory and verifying that it is contained in $[1, N)$,
    - loading input $x_G$, verifying $x_G < p$, and either reconstructing $y_G$ ($x$-only) or loading it from memory (affine),
    - verifying $(k times G)_x < p$ and $(k times G)_y < p$, and
    - writing $(k times G)_x$ to memory, together with $(k times G)_y$ on the affine variant.
    It interacts with the `ECDAS` chip, sending $k$ and $G$ as input, and receiving $k times G$ as result.
- *`ECDAS` (Elliptic Curve Double/Add Sequence)*.
    This chip computes $k times G$ by recursively interacting with itself.
    At each step, the chip either adds the point to the accumulator ($A <- A + G$) or doubles the accumulator ($A <- 2A$), where the sequence of doubles and adds is dependent on $k$.
    The process is repeated until $A = k times G$.
    This technique is called _double-and-add_ #footnote(link("https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Double-and-add")) and manages to compute the multiplication in $O(log(k))$ doubling-steps and $O(w_H (k)) = O(log(k))$ addition-steps, where $w_H (dot)$ denotes the hamming-weight of a bitstring.

= ECSM <ecsm-sm>

The #ecsm (Elliptic Curve Scalar Multiply) chip is generic over the constants
- $p in NN$, the prime field modulus,
- $a < p$, the first curve coefficient,
- $b < p$, the second curve coefficient, and
- $N in NN$, the order of the curve group.
To support scalar multiplication over different curves, one chip instance should be created for each curve, where each instance is given a unique `id`.

To prevent cross-instance communication between the `ecsm` and `ecdas` chips, each curve instance must be assigned a unique `id` that acts as a domain separator.
Here follows the present `id` mapping:
#align(center)[#table(
  columns: (auto, auto),
  table.header(`id`, "curve"),
  "0",  `secp256k1`,
  "1",  `secp256r1`,
)]
Supporting other curves only requires assigning them a unique `id`.#footnote([Note that adding a curve does require `id`'s type to be updated as well, since its current type (`Bit`) is now saturated. Since each curve now claims _two_ ECALL-numbers (see below), it also consumes the reserved range twice as fast: $#`id` = 4$ would collide with `FEXT_LOAD` at $-20$.])

#attention("Only " + `secp256k1` + " is instantiated.")[
  The constraints below are written generically in $a$, $b$, $p$ and $N$, but only $#`id` = 0$ has ever been instantiated, and that curve has $a = 0$.
  The $y_G$ relation carries a single $p^2$ offset (@ec:c:c1_0, @ec:c:c1_i), which is enough to keep $q_1$ non-negative only while $a dot x_G$ is small.
  For a curve with large $a$ --- `secp256r1` has $a = p - 3$ --- the offset is insufficient, and the more so on the affine variant, where @ec:c:read_yG pins $y_G$ and so removes the prover's freedom to pick whichever root gives a representable quotient.
  Instantiating $#`id` = 1$ therefore requires widening the offset _and_ `q1`'s top limb; the ECALL-numbers $-13$ and $-14$ are reserved, not usable.
]

The chip is triggered by executing `ECALL`, with the ECALL-number set to $-11 - 2 dot #`id` - #`is_affine`$:
#align(center)[#table(
  columns: (auto, auto, auto),
  table.header("ECALL number", "curve", "variant"),
  "-11", `secp256k1`, [$x$-only],
  "-12", `secp256k1`, "affine",
  "-13", `secp256r1`, [$x$-only],
  "-14", `secp256r1`, "affine",
)]
Since `id` is a per-instance constant, the ECALL-number is _linear_ in `is_affine`: the receiver (@ec:c:receive_ecall) reconstructs it as $(-11 - 2#`id`) - #`is_affine`$.
The `CPU` chip sends the guest's actual `A7` on the same bus, so a row that claims the wrong variant leaves the `ECALL` LogUp unbalanced.
This is what pins `is_affine`, and thereby the two memory accesses it gates, to the ECALL the guest really executed.

The chip expects
- `x10` to contain the address where $x_R := (k times G)_x$ is to be stored,
- `x11` to contain the address at which the least significant byte of $x_G$ is to be found,
- `x12` to contain the address at which the least significant byte of $k$ is to be found,
where it is assumed that $x_G$ and $k$ are provided as little-endian integers; $x_R$ is written to memory in little-endian form.
On the affine variant, the two point buffers are 64 bytes wide rather than 32: $y_G$ is read from 32 bytes above the address held in `x11`, and $y_R$ is written 32 bytes above the address held in `x10`, both again little-endian.
No additional registers are consumed.

Widening the buffers widens the caller's obligations, and neither is enforced by this chip.
The buffers must not overlap the scalar, since $x_G ‖ y_G$ is read at `timestamp` and $k$ at $#`timestamp` + 1$, and the memory argument cannot serve one address twice in one cycle.
An operand's address must also stay clear of a $2^32$ boundary --- by 64 bytes for the two point buffers and 32 for $k$ --- because the implementation adds each per-access offset to the low half of the address alone and cannot carry into the high half.
The constraints below abstract over that: they derive every address with a full 64-bit `ADD` (@ec:c:extrapolate_addr_yG, @ec:c:extrapolate_addr_yR), which carries correctly and so admits addresses the `ECALL` itself rejects.

== Columns
#let nr_variables = total_nr_variables(ecsm_chip)
#let nr_columns = total_nr_instantiated_columns(ecsm_chip, config)
#let nr_interactions = compute_nr_interactions(ecsm_chip)

The #ecsm chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(ecsm_chip, config)

== Constraints

=== Interactions
This chip is triggered by an `ECALL` with the opcode indicating this chip and the requested variant.
Constraint @ec:c:is_affine_implies_mu forces $#`is_affine` = 0$ on padding rows, so the buses it gates cannot fire there.
#render_constraint_table(ecsm_chip, config, groups: "ecall")

=== Read `xG`
Once triggered, it loads register `x11` to see where $x_G$ is stored in memory (@ec:c:read_addr_xG) and subsequently loads $x_G$ into `xG` (@ec:c:read_xG).
#render_constraint_table(ecsm_chip, config, groups: "read_xG")

=== Read `yG`
On the affine variant, the input point comes with its $y$-coordinate.
The four addresses at which it is stored are derived from `addr_xG[0]` rather than from a fourth register (@ec:c:extrapolate_addr_yG), since the guest passes $x_G ‖ y_G$ as one contiguous 64-byte buffer.
The read itself (@ec:c:read_yG) carries multiplicity `is_affine`, so it is inert on $x$-only rows — where the guest has no $y_G$ in memory to read — and on padding rows.
It shares its `timestamp` with the $x_G$-read (@ec:c:read_xG); the two cover disjoint addresses, which is exactly the condition under which @memory:aside:granularity permits a shared timestamp.
#render_constraint_table(ecsm_chip, config, groups: "read_yG")

=== Range check `xG`
Before continuing, it is verified that $x_G in [0, p)$.
To this end, witness $#`xG_sub_p` := #`xG` - p mod 2^256$ is added to `p`; if the addition sums to `xG` and overflows $mod 2^256$, it must hold that $#`xG` < p$.
The addition is constrained by requiring that `c2` are bits (@ec:c:range_c2); an overflow occurs if and only if $#`c2[7]` = 1$ (@ec:c:xG_addition_overflows).

#render_constraint_table(ecsm_chip, config, groups: "range_xG")

=== Constrain `yG`
With $x_G$ read and range checked, we direct our attention to $y_G$.
On the $x$-only variant it is never read from memory; the prover provides it as a witness and proves it to be correct.
On the affine variant the same witness is additionally pinned to the caller's buffer by @ec:c:read_yG, but the relations below are enforced in both cases.
In particular, the chip enforces the relations
$
  x_G^2 - #`x2` - q_0 dot p &= 0,\
  y_G^2 - x_G dot #`x2` - a dot x_G - b + (2p - q_1)p &= 0\
$
where non-negative $q_0$ and $q_1$ are prover-provided witnesses.
Note that these are equivalent to
$
  #`x2` &equiv x_G^2 mod p,\
  y_G^2 &equiv x_G dot #`x2` + a dot x_G + b  mod p\
$
which combine to $y_G^2 equiv x_G^3 + a x_G + b mod p$.
Rewriting the two relations, we get
$
  q_0 &= (x_G^2 - #`x2`)/p,\
  q_1 &= (y_G^2 - x_G dot #`x2` - a dot x_G - b)/p + 2p.
$
Using the fact that $x_G, y_G, #`x2`,a in [0, p)$, we find that $q_0 in [0, p)$ and $q_1 in [0, 3p)$.
We must therefore support quotients $q_0 in [0, 2^256)$ and $q_1 in [0, 2^258)$.

#aside("Two options for " + $y_G$)[
  In most cases, $y_G^2$ has _two_ roots $mod p$.
  This means that enforcing the above relation does not fully constrain the prover: it can choose which of the two to provide as hint.
  On the $x$-only variant this is not a problem: the chip only outputs the $x$-coordinate of $k times G$, and $x(k times G) = x(k times (-G))$, so both choices yield the same output.
]

#attention("The affine variant must pin the sign of " + $y_G$)[
  As soon as $y_R$ is published, the freedom described above becomes exploitable.
  A prover that answers with $-y_G$ computes $k times (-G)$: a correct multiple of a _different_ point.
  The curve equation cannot tell the two apart, and neither can the guest, which delegated the multiplication precisely because it cannot perform it.
  Constraint @ec:c:read_yG is what closes this: it pins the `yG` witness to the bytes the caller placed at $#`addr_xG` + 32$.
  This is also why the read fires with multiplicity `is_affine` rather than `μ` — the $x$-only path has nothing to pin it to, and does not need it.
]

Below, we enforce the first of the two sub-relations.
We emphasize here that @ec:c:c0_63_is_zero is required to ensure the sum evaluates to $0$, rather than just $0 mod 2^256$.
#render_constraint_table(ecsm_chip, config, groups: "xG2")

Next, we restrict the witness pair $(y_G, #`q1`)$.
Note there that @ec:c:c1_0 and @ec:c:c1_i multiply `B` and `P` by `μ` to simplify the padding; there are no other side-effects to this since $#`μ` = 1$ on non-padding rows (@ec:c:mu_isbit).

#render_constraint_table(ecsm_chip, config, groups: "yG")

=== Read and verify `k`
After reading `addr_k` from `x12` (@ec:c:read_addr_k), we read `k` from this address (@ec:c:load_k).
To prevent the point at infinity from showing up during the scalar multiplication, we require that $#`k` < #`N`$.
This is achieved by requiring that the addition $#`N` + (#`k` - #`N`)$ overflows $mod 2^256$ (@ec:c:k_lt_N).
Additionally, @ec:c:k_gt_0 ensures that $#`k` > 0$, preventing a case where $#`k` times #`G` = #inf$.
#render_constraint_table(ecsm_chip, config, groups: "verify_k")

=== Subroutine
With point $G$ and scalar $k$ fully constructed, we serve scalar `k` bit-by-bit to the `ECDAS` chip.
On this chip, we do capture the index of the most significant 1-bit of `k` in `idx_k`, to instruct the `ECDAS` chip where to start.
Note: if the prover decides to capture a lesser significant bit here, the LogUp will not balance, as the skipped bits will never taken off the bus.
Next, we interact with the `ECDAS` chip, providing `G` both as the accumulator and generator, and increment (@ec:c:start_double_add); we specifically instruct the chip to start with a _double_-operation.
After completing its double-and-add sequence, the result is captured in `(xR,yR)` (@ec:c:receive_double_add).
#render_constraint_table(ecsm_chip, config, groups: "delegate")

=== Range check `xR`
Before storing $x_R$, it is verified that $x_R in [0, p)$.
To this end, witness $#`xR_sub_p` := #`xR` - p mod 2^256$ is added to `p`; if the addition sums to `xR` and overflows $mod 2^256$, it must hold that $#`xR` < p$.
The addition is constrained by requiring that `c4` are bits (@ec:c:range_c4); an overflow occurs if and only if $#`c4[7]` = 1$ (@ec:c:xR_addition_overflows).

#render_constraint_table(ecsm_chip, config, groups: "range_xR")

=== Range check `yR`
The same treatment is given to $y_R$: witness $#`yR_sub_p` := #`yR` - p mod 2^256$ is added to `p`, and the addition is required to overflow (@ec:c:yR_addition_overflows), which holds if and only if $#`yR` < p$.

Unlike the $y_G$-read, this check fires on _every_ active row rather than only on affine ones.
`yR` is witnessed in both variants anyway, so gating it would save no columns; it would drop the 16 `IS_HALF` lookups of @ec:c:range_yR_sub_p and the 7 `IS_BIT` terms of @ec:c:range_c5 on $x$-only rows, which we judge not worth a second selector.

#aside("Why " + $y_R$ + " needs a canonicality check at all")[
  The relations that produce $y_R$ absorb a multiple of $p$ into their quotient columns, and $y_R$ is bounded only below $2^256$.
  A prover could therefore publish $y_R + p$ whenever $y_R < 2^256 - p$ and still satisfy every other constraint.
  For `secp256k1` that band has width $2^256 - p approx 2^32$, and it is populated: the curve has points with very small $y$.
  Constraint @ec:c:yR_addition_overflows is what rules the non-canonical representative out, given that `c5` are bits (@ec:c:range_c5).
  $x_R$ was already covered by @ec:c:xR_addition_overflows; publishing $y_R$ is what makes _its_ representation observable too.

  The $2^256$ bound the argument rests on is worth locating precisely, because it is not local to this group.
  On a row that delegates, it comes from the `ECDAS` chip, which range-checks its own $y_R$ bytes before sending them.
  But when $#`k` = 1$ the delegation collapses: @ec:c:start_double_add and @ec:c:receive_double_add carry identical tuples and cancel on the bus, so no `ECDAS` row exists at all, and the bound then comes from @ec:c:range_yG via $y_R = y_G$.
  Either way $y_R$ is byte-bounded, but this chip nowhere range-checks `yR` itself.
]

#render_constraint_table(ecsm_chip, config, groups: "range_yR")

=== Write `xR`
We read `addr_xR` from register `x10` (@ec:c:load_addr_xR), and subsequently write `xR` to this address (@ec:c:write_xR).
Note that the `timestamp` on both memory accesses is offset to allow `addr_xR` to equal `addr_xG` and thus for $x_R$ to overwrite $x_G$ in memory.
#render_constraint_table(ecsm_chip, config, groups: "write_xR")

=== Write `yR`
On the affine variant, $y_R$ is written directly after $x_R$, at addresses derived from `addr_xR[0]` (@ec:c:extrapolate_addr_yR); as on the input side, the output buffer is one contiguous 64-byte region and no extra register is read.
The write carries multiplicity `is_affine` (@ec:c:write_yR) and uses $#`timestamp` + 3$, the fourth and last of the cycle's sub-timestamps (@memory:aside:granularity): $x_G$ and $y_G$ occupy `timestamp`, $k$ occupies $#`timestamp` + 1$ and $x_R$ occupies $#`timestamp` + 2$.
The two halves of the output cover disjoint addresses, so they could legally share a sub-timestamp; $#`timestamp` + 3$ is simply the slot left over, and taking it leaves this chip with no further headroom in the cycle.
#render_constraint_table(ecsm_chip, config, groups: "write_yR")

== Carry offsets
We close by deriving the values of `offsets`.
To this end, we decompose the formulae
$
  #`xG`^2 - #`x2` - q_0 dot p &= 0,\
  y_G^2 - x_G dot #`x2` - a dot x_G - b + (2p - q_1)p &= 0
$
in terms of the positive and negative components to find
$
  #`xG`^2 - (#`x2` + q_0 dot p) &= 0, text("and")\
  (y_G^2 + 2p^2) - (x_G dot #`x2` + a dot x_G + b + q_1 dot p) &= 0.
$
Applying @limbs:cor:carry-upper-bound with $(L, n) = (2^8, 32)$, we find that
$
  #`c0`_i &in [-8160, 8159],\
  #`c1`_i &in [-24478, 24477].\
$
When we select $#`offsets` = (8160, 24478)$, we arrive at
$
  #`c0`_i + #`offsets[1]` &in [0, &16319] subset.eq [2^16],\
  #`c1`_i + #`offsets[2]` &in [0, &48955] subset.eq [2^16].\
$

== Padding
#render_chip_padding_table(ecsm_chip, config)

#pagebreak(weak: true)

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
Cases 2 and 3 may, for specific inputs, evaluate to $#inf$:
a point that has no native short-Weierstrass representation.
Therefore, the #ecdas chip is designed to avoid these cases:

*Double.*
For $2A$ to equal $#inf$, the curve must have _even_ order; on curves with _odd_ order (@ecdas:a:curve_odd_order), such a point does not exist.

*Add.*
If $A + G = #inf$, then $A = -G = #inf - G = r N G - G$ for some $r >= 0$.
Since $A = G eq.not #inf$ (@ecdas:a:A_is_valid, @ecdas:a:G_is_valid), it must hold that $r >= 1$.
Furthermore, the assumption that $k <= N-1$ (@ecdas:a:k_lt_order) ensures $r <= 1$.
Hence, $A = (N-1)G$.
Since $N-1$ is the maximal value of $k$, the previous round producing $A = (N-1)G$ was the last round of this scalar multiplication.
This means that now `round` is negative, which will fail constraint @ecdas:c:range_round.


== Columns
#let nr_variables = total_nr_variables(ecdas_chip)
#let nr_columns = total_nr_instantiated_columns(ecdas_chip, config)
#let nr_interactions = compute_nr_interactions(ecdas_chip)

The #ecdas chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(ecdas_chip, config)

== Assumptions
#render_chip_assumptions(ecdas_chip, config)

== Constraints
First, the chips receives the input for this double/add step:
#render_constraint_table(ecdas_chip, config, groups: "receive")

=== Operation switching
The `op`-flag determines whether $R := 2A$ (0) or $R:= A+G$ (1).
This chip introduces a set of three constraints that correctly constrains $R$ depending on this flag:
$
  #`op` dot ((x_G - x_A)lambda - y_G + y_A) + (1-#`op`) (2lambda y_A - 3x_A^2 - a) + (#`r` - q_0) p &= 0,\
  lambda^2 - x_A - x_G - x_R + (1-#`op`) (x_G - x_A) + (#`r` - q_1) p &= 0,\
  lambda (x_A - x_R) - y_A - y_R + (#`r` - q_2) p &= 0,
$
To see how, note that these relations reorder to
$
  2lambda y_A - 3x_A^2 - a + (#`r` - q_0) p = 0 &<==>& lambda &equiv (3x_A^2 + a)/(2y_A) mod p,\
  lambda^2 - 2x_A - x_R + (#`r` - q_1) p = 0 &<==>& x_R &equiv lambda^2 - 2x_A mod p,\
  lambda (x_A - x_R) - y_A - y_R + (#`r` - q_2) p = 0 &<==>& y_R &equiv lambda(x_A - x_R) - y_A mod p.
$
when $#`op`=0$.
If instead $#`op`=1$, they reorder to
$
  (x_G - x_A)lambda - y_G + y_A + (#`r` - q_0) p = 0 &<==>& lambda &equiv (y_G - y_A)/(x_G - x_A) mod p,\
  lambda^2 - x_A - x_G - x_R + (#`r` - q_1) p = 0 &<==>& x_R &equiv lambda^2 - x_A - x_G mod p,\
  lambda (x_A - x_R) - y_A - y_R + (#`r` - q_2) p = 0 &<==>& y_R &equiv lambda(x_A - x_R) - y_A mod p.
$
By selecting $r = 3p$, we ensure $q_0 in (0, 5p)$, $q_1 in (3p-3, 4p)$ and $q_2 in (2p, 4p)$ are non-zero, irrespective of the value of `op`.

The observant reader my notice that $lambda$ is underconstrained when $(#`op`, y_A) = (1, 0)$ and $(#`op`, x_A) = (0, x_G)$.
The first case is ruled out because this accelerator restricts itself to odd-order curves; such curves do not have a point with $y = 0$.
For the second to occur, it must be that $A - G = inf$; the case that $A + G = inf$ was previously ruled out.
This requires that $A = (r N + 1) G$ for some $r in NN$ and $N$ the order of the curve.
Note that all cases with $r>0$ can be ruled out since #ecsm verifies that the scalar $k < N$.
The final case $A=G$ is the intial state pushed onto the LogUp by #ecsm (@ec:c:start_double_add), with `op`-flag set to $0$ (_double_), not `add`.
Hence, this situation cannot occur either. 


=== Constraining $lambda$
We start by establishing the relation
$
  #`op` dot (lambda (x_G - x_A) - y_G + y_A) + (1-#`op`) (2lambda y_A - 3x_A^2 - a) + (#`r` - q_0) p &= 0.\
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

== Carry offsets
We derive the values of `offsets`.
We start with the three formulae
$
  #`op` dot (lambda (x_G - x_A) - y_G + y_A) + (1-#`op`) (2lambda y_A - 3x_A^2 - a) + (#`r` - q_0) p &= 0,\
  lambda^2 - x_A - x_G - x_R - (1-#`op`) (x_A - x_G) + (#`r` - q_1) p &= 0,\
  lambda (x_A - x_R) - y_A - y_R + (#`r` - q_2) p &= 0
$
which we rewrite in terms of the positive and negative components to find
$
  (#`op` dot (lambda x_G + y_A) + (1-#`op`) dot 2lambda y_A + #`r` dot p) - (#`op` dot (lambda x_A + y_G) + (1-#`op`) (3x_A^2 + a) + q_0 dot p) &= 0,\
  (lambda^2 + #`r` dot p) - (x_A + x_G + x_R + (1-#`op`) (x_A - x_G) + q_1 dot p) &= 0,\
  (lambda x_A + #`r` dot p) - (lambda x_R + y_A + y_R + q_2 dot p) &= 0.
$
Leveraging @limbs:cor:carry-upper-bound with $(L, n) = (2^8, 33)$ and maximizing for the value of $#`op` in {0, 1}$, we gather
$
  #`c0`_i &in [-33657, 25242],\
  #`c1`_i &in [-8416, 16828], text("and")\
  #`c2`_i &in [-16830, 16828].\
$
Selecting $#`offsets` = (33657,8416,16830)$, we arrive at
$
  #`c0`_i + #`offsets[0]` &in [0, 58899] subset.eq [2^16],\
  #`c1`_i + #`offsets[1]` &in [0, 25244] subset.eq [2^16],text("and")\
  #`c2`_i + #`offsets[2]` &in [0, 33658] subset.eq [2^16].
$

== Padding
#render_chip_padding_table(ecdas_chip, config)

= Notes / optimizations
- To utilize the #ecsm / #ecdas chips for different curves, consider introducing a lookup table for the
  curve-constants $a$, $b$, $p$, $r$ and $N$, and look them up when a scalar multiplication selects them.
  The selection procedure could be done through the `ECALL` number, in the same way `is_affine` already selects the variant: the #ecsm chip accepts several numbers and sets an internal selector column accordingly, pinned by the `ECALL` bus (@ec:c:receive_ecall).
- Transitioning from `U256BL`s to `U256HL`s would roughly halve the number of columns in both the #ecsm and #ecdas chips.
  This would likely require increasing the sizes of the carries from 16 to 24 bits.
  Since the carries need to be range checked, one would have to investigate whether
    - it would be possible to perform a 24-bit range-check lookup,
    - one could set up a 24-bit range-check table. This could be as narrow as two columns.
    - have some hybrid version, where there is a native lookup table for x-bits, and a dynamic table for outliers (high carries are not encountered frequently).
- `addr_xG[0]`, `addr_k[0]` and `addr_xR[0]` could be `DWordWL`s rather than `HL`s.
  We use `HL`s as conventient notation.
  This modification saves 6 columns.
- the design of these chip is generic, and makes no assumptions on the parameters $a$, $b$, $p$ and $N$.
  It might be possible to arrive at more compact design by making some assumptions on these values.
- Constraints @ec:c:c1_0 and @ec:c:c1_i can be simplified to degree-2 constraints by padding `q1` to `p` rather than `0`. 
  Note: this requires a non-trivial modification to the padding verification tooling.
