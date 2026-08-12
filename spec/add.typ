#import "/book.typ": book-page, et
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, render_chip_assumptions, render_constraint_table, set_nr_interactions, compute_nr_interactions,

#let config = load_config()
#let chip = load_chip("src/add.toml", config)
#let subchip = load_chip("src/sub.toml", config)
#let nwchip = load_chip("src/add_nw.toml", config)

#show: book-page(chip.name)

#set_nr_interactions(chip, name: "SUB")
#let nr_interactions = compute_nr_interactions(chip)
#let nw_interactions = compute_nr_interactions(nwchip)

#let add = raw(chip.name)
#let sub = raw(subchip.name)
#let addnw = raw(nwchip.name)

= #add
#add is a constraint template that is used to assert that $#`sum` equiv #`lhs` + #`rhs` (mod 2^64)$, under the condition that `cond` is non-zero.

== Variables
This template introduces #nr_interactions interaction(s).
#render_chip_variable_table(chip, config)

== Assumptions
#render_chip_assumptions(chip, config)

== Constraints
This template introduces the following constraints
#render_constraint_table(chip, config)

Note that the correctness of these constraints follows from @limbs:lm:limb-decomposition-constraint-correctness, when applied to $(S, L, C, alpha, mu) = (2^64, 2^32, 2, 2, 0)$:
- the definition of `carry` matches that of @limbs:eq:def_ci and @limbs:eq:c_-1_is_zero, 
- @limbs:eq:range_ci is enforced by @add:c:carry, and
- @limbs:eq:range_wi follows from @add:a:sum.

= #sub

For ease of notation, we moreover introduce the #sub constraint template
$
#`SUB<diff; lhs, rhs>` := #`ADD<lhs; diff, rhs>`,
$
in both conditional and unconditional versions.
It constrains that $#`diff` equiv #`lhs` - #`rhs` (mod 2^64)$ when the expression `cond` is non-zero.

== Variables
This template introduces #nr_interactions interaction(s).
#render_chip_variable_table(subchip, config)

== Assumptions
#render_chip_assumptions(subchip, config)

== Constraints
This template introduces the following constraints
#render_constraint_table(subchip, config)

= #addnw

#add asserts an equality modulo $2^64$; #addnw is the variant that rules out the wraparound.
It constrains that $#`sum` = #`lhs` + #`rhs`$ _over the integers_ when the expression `cond` is non-zero, and is intended for chips whose operands are addresses, where a wraparound would silently move an access to an unrelated region of memory.

The two limbs are treated asymmetrically, and deliberately so.
The carry out of the _least_ significant limb is constrained on every row, so the low limb of `sum` always means what it says.
The carry out of the _most_ significant limb is pinned only where `cond` is non-zero, which leaves `sum`'s high limb free on the rows where a chip does not consume the result --- typically padding rows, and the terminal row of a recursive sequence.
Constraining it there would buy nothing and would force those rows to carry a well-formed successor they never use.

== Variables
This template introduces #nw_interactions interaction(s).
#render_chip_variable_table(nwchip, config)

== Assumptions
#render_chip_assumptions(nwchip, config)

== Constraints
This template introduces the following constraints
#render_constraint_table(nwchip, config)

Note that `carry` is defined exactly as it is in #add, so @addnw:c:no_wraparound is precisely the statement that the addition of the most significant limbs does not carry out;
combined with @addnw:a:sum, that is equivalent to $#`lhs` + #`rhs` < 2^64$.
