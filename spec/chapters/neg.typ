#import "/meta.typ": aside, et
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, render_chip_assumptions, render_constraint_table, compute_nr_interactions,

#let config = load_config()
#let chip = load_chip("src/neg.toml", config)

#let nr_interactions = compute_nr_interactions(chip)

#let neg = raw(chip.name)

#neg is a constraint template that is used to assert that $#`neg` = -#`x`$, under the condition that `cond` is non-zero.
It requires `cond` to be a bit.

= Variables
This template introduces #nr_interactions interaction(s).
#render_chip_variable_table(chip, config)

= Assumptions
#render_chip_assumptions(chip, config)

= Constraints
We constrain this equality using two constraints:
#render_constraint_table(chip, config)

== Correctness argument
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
Clearly, $#`neg` = 0$ when $#`x` = 0$ (and `cond` is set).
For non-zero `x`, we distinguish two cases.
When $(#`x as DWordWL`)_0 = 0$,
$
  #`neg` 
  &= 2^32 dot #`neg`_1 + #`neg`_0\
  &= 2^32 dot (2^32 - (#`x as DWordWL`)_1) + 0\
  &= 2^32 dot (2^32 - (#`x as DWordWL`)_1) + (#`x as DWordWL`)_0\
  &= 2^64 - (2^32 dot (#`x as DWordWL`)_1 + (#`x as DWordWL`)_0)\
  &= 2^64 - #`x`\
  &equiv -x mod 2^64,
$
while when $(#`x as DWordWL`)_0 != 0$,
$
  #`neg` 
  &= 2^32 dot #`neg`_1 + #`neg`_0\
  &= 2^32 dot (2^32 - (#`x as DWordWL`)_1 - 1) + (2^32 - (#`x as DWordWL`)_0)  \
  &= 2^64 - 2^32 dot (#`x as DWordWL`)_1 - 2^32 + 2^32 - (#`x as DWordWL`)_0  \
  &= 2^64 - ((#`x as DWordWL`)_0 + 2^32 dot (#`x as DWordWL`)_1) \
  &= 2^64 - #`x`\
  &equiv -x mod 2^64
$
when `cond` is set.
When `cond` is not set, the two lookups are not executed, allowing `neg` to take any value in either case.

#aside("Missing range check?")[
  It is worth noting that this construction does _not_ require the limbs of `neg` to be range checked, 
  thus allowing it be represented by the unrangecheckable `DWordWL` rather than a `DWordHL`.
  The input value `x` is still assumed to be range-checked, however.
]
