#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_column_table, render_constraint_table

#show: book-page("is_bit.typ")

#let config = load_config()
#let chip = load_chip("src/is_bit.toml", config)

#let is_bit = raw(chip.name)

#let highlighted_code(code) = {
  box(
    inset: (left: 4pt, right: 4pt), 
    outset: (top: 4pt, bottom: 4pt), 
    radius: 2pt,
    fill: luma(230), 
    raw(code))
}

#is_bit is a constraint template that is used to assert that a variable lies in the range ${0, 1}$ if some second variable is non-zero.
Barring exceptional cases, this template is used to assert that a variable of type `Bit` assumes a valid value under some condition.

== Interface
The #is_bit constraint template has the following interface:
#block(radius: 5pt, width: 100%, inset: 1.5em, fill: luma(230), raw("cond => IS_BIT<X>"))
where `cond` is any value described by an expression _of degree at most $1$_.
Note that #highlighted_code("IS_BIT<X>") can be used to denote the _unconditional_ application of the #is_bit template to `X`.

== Variables
The #is_bit template operates on two variables: `cond` and `X`:
#render_chip_column_table(chip, config)

== Constraints
It takes only one constraint to enforce that `X` must be either $0$ or $1$ whenever $#`cond` eq.not 0$:
#render_constraint_table(chip, config)
*Note*: 
- In case of _unconditional_ template application, `cond` can be dropped from the constraint, simplifying it to $#`X` (1- #`X`) = 0$.
- As described earlier, the `cond` variable must be describable by a degree-1 (i.e., linear) expression.
  This is to make sure that @isbit:c:isbit's expression has degree at most 3.

== Proof of correctness
If `cond` is $0$, @isbit:c:isbit is trivially satisfied: `X` can assume any value and the polynomial constraint will evaluate to $0$ regardless. 
When $#`cond` eq.not 0$, it follows that the statement can only be proven when $#`X` (1-#`X`) equiv 0 mod p$, with $p$ the modulus of the field.
Because `BaseField` is a prime field, this equality is only satisfied if either $#`X` equiv 0 mod p$ or $1-#`X` equiv 0 mod p$.
Hence, it is proven that when $#`cond` eq.not 0$, @isbit:c:isbit is only satisfied if $#`X` in {0, 1}$. #align(right, $qed$)
