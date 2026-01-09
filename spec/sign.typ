#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_column_table,   render_chip_assumptions, render_constraint_table


#let config = load_config()
#let chip = load_chip("src/sign.toml", config)
#show: book-page(chip.name)

#let sign = raw(chip.name)

#let highlighted_code(code) = {
  box(
    inset: (left: 4pt, right: 4pt), 
    outset: (top: 4pt, bottom: 4pt), 
    radius: 2pt,
    fill: luma(230), 
    raw(code))
}

#sign is a constraint template that is used to extract a `Half`word's sign.

== Interface
The #sign constraint template has the following interface:
#block(radius: 5pt, width: 100%, inset: 1.5em, fill: luma(230), raw("SIGN<sign; X, signed>"))
It constrains that `sign` is set to `1` when both `X`'s most significant bit and `signed` are $1$, and $0$ otherwise.

== Variables
The #sign template operates on three variables:
#render_chip_column_table(chip, config)

== Assumptions
The #sign template operates on the following assumptions:
#render_chip_assumptions(chip, config)

== Constraints
It takes only two constraints to compute the `sign` of `X`, given whether `X` represents a `signed` value or not. 
When $#`signed` = 1$, the sign of `X` is equal to its most significant bit. 
This value is extracted in @sign:c:sign_if_signed.
If `X` is unsigned (i.e., $#`signed` = 0$), its sign is always $0$.
This is constrained by @sign:c:sign_if_unsigned.
#render_constraint_table(chip, config)
