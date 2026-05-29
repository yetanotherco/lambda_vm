#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_variable_table,
  render_chip_padding_table,
  render_constraint_table,
  total_nr_instantiated_columns,
  total_nr_variables,
  compute_nr_interactions,
)

#let config = load_config()
#let chip = load_chip("src/lt.toml", config)

#show: book-page(chip.name)
#let lt = raw(chip.name)

The #lt chip constrains an indicator bit for the less-than relation, signed or unsigned.
If the `invert` flag is set, it inverts the result.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #lt chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Assumptions
We assume the inputs `lhs`, `rhs` and `signed` are partially range checked.
#render_chip_assumptions(chip, config)

= Constraints
We first constrain that all inputs are range checked and all variables correspond to their definition.
For the defining constraint of `lt`, @lt:c:lt, observe that it is a choice
between two options, depending on the input flag `signed`.
In the case of unsigned comparison, we simply need `unsigned_lt`, indicating
that a wraparound (carry bit) modulo $2^64$ is needed to go from `rhs` to `lhs` via addition.
For the case of signed comparison, we first need some case analysis.

We split $a < b$ into four disjoint cases, conditioned on the sign of $a$ and $b$.
Recall that the sign of a number in two's complement can be read off from the MSB,
being $1$ for a negative number and $0$ for a positive one.
For this analysis, we denote the MSB of $a$ as $A$ and the MSB of $b$ as $B$.
The four disjoint cases then become:

+ $dash(A) and B and (a < b)$
+ $A and dash(B) and (a < b)$
+ $A and B and (a < b)$
+ $dash(A) and dash(B) and (a < b)$

The first case is evidently false, while the second case simplifies to $A and dash(B)$.
For the third and fourth case, observe that when $A = B$, the $<$ relation is preserved
by the modular correspondence between $[-2^(31), 2^(31))$ and $[0, 2^(64))$.
Importantly, this modular correspondence is merely a reinterpretation of the
bits or values of $a$ and $b$, due to the representation in two's complement.
Hence, we can introduce the value $C = #`unsigned_lt`$, that accurately represents
the relation $a < b$ when $A = B$.

Combining our three remaining cases, we obtain the boolean formula $A dash(B) or A B C or dash(A) dash(B) C$.
Since the cases are disjoint, this can be computed with the binary-valued polynomial
$P(A, B, C) = A (1 - B) + A B C + (1 - A) (1 - B) C$.

The polynomial $P$ can be simplified to a total degree of two.
We claim that the polynomial $Q(A, B, C) = A (1 - B) + A C + (1 - B) C$
is, for the purposes of this chip, equivalent to $P$.
An exhaustive check shows that $P(A, B, C) != Q(A, B, C)$ only for the triple $(A, B, C) = (1, 0, 1)$.
This is, however, impossible due to the correctness of `ADD`.
In more detail, if we let $s$ be the (range-checked) difference $a - b$
(so the equivalent of the #`lhs_sub_rhs` column),
and $x'$ denote the most significant word of a variable $x$,
we need $c dot 2^32 + a' = b' + s' + #`carry[0]`$, by the definition of `carry`.
However, the left hand side of this is at least $3 dot 2^31$, as $(A, C) = (1, 1)$,
and the right hand side is at most $(2^31 - 1) + (2^32 - 1) + 1 = 3 dot 2^31 - 1$.
Therefore, we can use $Q$ to constrain `lt` when `signed = 1`.

#render_constraint_table(chip, config, groups: ("range", "defs"))

And then we constrain the subtraction,
taking care of the remaining range checking not yet covered by the assumptions or the `MSB16` lookup.

#render_constraint_table(chip, config, groups: "sub")

The chip contributes the following to the lookup argument.

#render_constraint_table(chip, config, groups: "output")

= Padding

The table can be padded to the next power of two with the following value assignments:

#render_chip_padding_table(chip, config)

= Potential optimizations

- Split the chip into a signed and an unsigned chip, making the unsigned version cheaper.
