#import "/book.typ": book-page, et
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_column_table, render_chip_assumptions, render_constraint_table

#let config = load_config()
#let chip = load_chip("src/neg.toml", config)
#show: book-page(chip.name)

#let neg = raw(chip.name)

#let highlighted_code(code) = {
  box(
    inset: (left: 4pt, right: 4pt), 
    outset: (top: 4pt, bottom: 4pt), 
    radius: 2pt,
    fill: luma(230), 
    raw(code))
}

#neg is a constraint template that is used to assert that $#`neg` = -#`x`$, under the condition that `cond` is non-zero.

== Notation
The #neg constraint template has the following interface:
#block(radius: 5pt, width: 100%, inset: 1.5em, fill: luma(230), raw("cond => NEG<neg; x>"))
where `cond` is a bit value (i.e., lies in ${0, 1}$)  described by an expression _of degree at most $1$_.
#highlighted_code("NEG<neg; x>") can be used to denote the _unconditional_ application of the #neg template to `x` and `neg` (which is equivalent to $#`cond` = 1$).

== Variables
#render_chip_column_table(chip, config)

== Assumptions
#render_chip_assumptions(chip, config)

== Constraints
For `neg` to equal $-#`x`$, both values must add to $0 mod 2^64$.
Zooming in on the addition, we find that the carry values of adding the limbs must both equal to $1$, except when $#`x` = 0$, in which case they should both be $0$.

To this end, we require $#`carry`_0$ is $1$, unless $#`x` = 0$ (@neg:c:carry) and
that both limbs of `carry` are identical (@neg:c:identical_carry ensuring).
Of course, both constraints are conditional on $#`cond` = 1$.
Lastly, note that this @neg:c:carry implicitly enforces that $#`carry`_0 in {0, 1}$, while @neg:c:identical_carry ensures $#`carry`_1$ must also be binary.

#render_constraint_table(chip, config)

== Note
It is worth noting that this construction does _not_ require the limbs of `neg` to be range checked, 
thus allowing it be represented by the unrangecheckable `DWordWL` rather than a `DWordHL`.
The input value `x` is still assumed to be range-checked, however.
