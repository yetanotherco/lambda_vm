#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
)

#let config = load_config()
#let chip = load_chip("src/lt.toml", config)

#show: book-page.with(title: "LT chip")

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `LT` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

== Assumptions
We assume the inputs `lhs`, `rhs` and `signed` are appropriately range checked.
#render_chip_assumptions(chip, config)

== Constraints
We first constrain that all variables correspond to their definition.
For the defining constraint of `lt`, @lt:c:lt, observe that it is a choice
between two options, depending on the input flag `signed`.
In the case of unsigned comparison, we simply need `unsigned_lt`, indicating
that a wraparound (carry bit) modulo $2^64$ is needed to go from `rhs` to `lhs` via addition.
For the case of signed comparison, we first need some case analysis.
We can conclude that $a < b$ exactly when any of the following disjoint events happens.

- $(a < 0) and (b >= 0)$
- $(a < 0) and (b < 0) and (a < b)$
- $(a >= 0) and (b >= 0) and (a < b)$

We represent can the comparisons of inputs to zero as the MSB or sign bits,
which we shall denote as $A$ for `lhs` and $B$ for `rhs`.
From this, we obtain the boolean formula $A dash(B) or A B C or dash(A) dash(B) C$,
where we let $C$ also denote the indicator of $(a - b < 0) and (A == B)$.
Since our cases were disjoint, this can be computed as the binary-valued polynomial
$P(A, B, C) = A (1 - B) + A B C + (1 - A) (1 - B) C$.

Observe that after modular reduction to the range $[0, 2^64)$,
when $A == B$, the ordering $a < b$ is preserved, so $C$ can be expressed
---just as in the unsigned case--- by the overflow of the addition.

The polynomial $P$ can be simplified to a total degree of two.
We claim that the polynomial $Q(A, B, C) = A dot (1 - B) + A dot C + (1 - B) dot C$
is, for the purposes of this chip, equivalent to $P$.
Through exhaustive checking, one can verify that the only binary input
for which $P(A, B, C) != Q(A, B, C)$, is the triple $(A, B, C) = (1, 0, 1)$.
This, however, corresponds to the case where $a > b$ in the reduction to $[0, 2^64)$,
*and* an overflow is said to occur to go $b$ to $a$ by addition,
which is a contradiction with the correctness of the addition.
In more detail, if we let $s$ be the (range-checked) difference $a - b$
(so the equivalent of the #`lhs_sub_rhs` column),
and $x'$ be the most significant word of $x$,
we need $c dot 2^32 + a' = b' + s' + #`carry[0]`$, by the definition of `carry`.
However, the left hand side of this, is at least $3 dot 2^31$, as $(A, C) = (1, 1)$,
and the right hand side is at most $(2^31 - 1) + (2^32 - 1) + 1 = 3 dot 2^31 - 1$.
Therefore, we can use $Q$ to constrain `lt` when `signed = 1`.

#render_constraint_table(chip, config, groups: "defs")

And then we constrain the subtraction.

#render_constraint_table(chip, config, groups: "sub")

The chip contributes the following to the lookup argument.

#render_constraint_table(chip, config, groups: "output")
