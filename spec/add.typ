#import "/book.typ": book-page, et
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_column_table, render_chip_assumptions, render_constraint_table

#show: book-page("add.typ")

#let config = load_config()
#let chip = load_chip("src/add.toml", config)

#let add = raw(chip.name)

#let highlighted_code(code) = {
  box(
    inset: (left: 4pt, right: 4pt), 
    outset: (top: 4pt, bottom: 4pt), 
    radius: 2pt,
    fill: luma(230), 
    raw(code))
}

#add is a constraint template that is used to assert that $#`sum` = #`lhs` + #`rhs` mod 2^64$, under the condition that `cond` is non-zero.

== Notation
The #add constraint template has the following interface:
#block(radius: 5pt, width: 100%, inset: 1.5em, fill: luma(230), raw("cond => ADD<sum; lhs, rhs>"))
where `cond` is any value described by an expression _of degree at most $1$_.
#highlighted_code("ADD<sum; lhs, rhs>") can be used to denote the _unconditional_ application of the #add template to `lhs`, `rhs`, and `sum`.

#let sub = raw("SUB")
=== #sub
For ease of notation, we moreover introduce the #sub constraint template.
Its interface
#block(radius: 5pt, width: 100%, inset: 1.5em, fill: luma(230), raw("cond => SUB<diff; lhs, rhs>"))
maps onto the #add template as 
#block(radius: 5pt, width: 100%, inset: 1.5em, fill: luma(230), raw("cond => ADD<lhs; rhs, diff>"))
It constrains that $#`diff` = #`lhs` - #`rhs` mod 2^64$ when the expression `cond` is non-zero.
As with #add, #highlighted_code("SUB<diff; lhs, rhs>") can be used to denote the _unconditional_ application of the template.

== Variables
#render_chip_column_table(chip, config)

== Assumptions
#render_chip_assumptions(chip, config)

== Constraints
This template introduces the following constraints
#render_constraint_table(chip, config)
