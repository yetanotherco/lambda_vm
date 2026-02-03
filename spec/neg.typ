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
We constrain this equality using two constraints:
#render_constraint_table(chip, config)
The constraints force the `carry` values to be fixed.
Writing `carry`'s definition, we then find that
$
  #`neg`_0 &= 2^32 dot #`carry`_0 - (#`x as DWordWL`)_0
 = cases(
  2^32 - (#`x as DWordWL`)_0 & "if" (#`x as DWordWL`)_0 != 0,
  0 & "if" (#`x as DWordWL`)_0 = 0
 ),\
  #`neg`_1 &= 2^32 dot #`carry`_1 - (#`x as DWordWL`)_1 - #`carry`_0 = cases(
  2^32 - (#`x as DWordWL`)_1 - 1 & "if" #`x` != 0,
  0 & "if" #`x` = 0
 )
$
Clearly, $#`neg` = 0$ when $#`x` = 0$ (and `cond` is set); for non-zero `x`, it holds that
$
  #`neg` 
  &= #`neg`_0 + 2^32 dot #`neg`_1 \
  &= (2^32 - (#`x as DWordWL`)_0) + 2^32 dot (2^32 - (#`x as DWordWL`)_1 - 1) \
  &= 2^32 - (#`x as DWordWL`)_0 + 2^64 - 2^32 dot (#`x as DWordWL`)_1 - 2^32\
  &= 2^64 - ((#`x as DWordWL`)_0 + 2^32 dot (#`x as DWordWL`)_1) \
  &= 2^64 - #`x`\
  &equiv -x mod 2^64
$
when `cond` is set.
When `cond` is not set, the two lookups are not executed, allowing `neg` to take any value.

== Note
It is worth noting that this construction does _not_ require the limbs of `neg` to be range checked, 
thus allowing it be represented by the unrangecheckable `DWordWL` rather than a `DWordHL`.
The input value `x` is still assumed to be range-checked, however.
