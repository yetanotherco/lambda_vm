#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_assumptions,
)

#let config = load_config()
#let chip = load_chip("src/mul.toml", config)

#show: book-page.with(title: "MUL chip")

#let mul = raw(chip.name)

= Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `MUL` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

#let stackrel(top, bottom) = {
 $mat(delim: #none, top; bottom)$
}

= Assumptions
#render_chip_assumptions(chip, config)

= Constraints

== Overview
When `lhs` and `rhs` are _unsigned_ integers, computing their product $mod 2^128$ comes down to evaluating
$
(sum_(j=0)^3 2^(16j) dot #`lhs`_j) dot (sum_(i=0)^3 2^(16i) dot #`rhs`_i) mod 2^128.
$
If `lhs` and `rhs` are signed instead, the computation remains nearly identical: 
one must sign extend `lhs` and `rhs` to twice their size --- forming `lhs_ext` and `rhs_ext` respectively --- and compute
$
(sum_(j=0)^7 2^(16j) dot #`lhs_ext`_j) dot (sum_(i=0)^7 2^(16i) dot #`rhs_ext`_i) mod 2^128.
$
where the limbs of `lhs_ext` and `rhs_ext` are treated as _unsigned_ integers.
Note that by setting the extension limbs of `lhs` and/or `rhs` to $0$ when the integer is unsigned or signed and positive, the second formula still applies.
Observe that we can rewrite this formula as
$
  &(sum_(j=0)^7 2^(16j) dot #`lhs_ext`_j) dot (sum_(i=0)^7 2^(16i) dot #`rhs_ext`_i) mod 2^128 \
  &equiv sum_(j=0)^7 sum_(i=0)^7 2^(16(i+j)) dot #`lhs_ext`_j dot #`rhs_ext`_i mod 2^128 \
  &stackrel(triangle, equiv) sum_(j=0)^7 sum_(i=0)^(7-j) 2^(16(i+j)) dot #`lhs_ext`_j dot #`rhs_ext`_i mod 2^128 \
  &stackrel(square, equiv) sum_(j=0)^7 sum_(i=j)^(7) 2^(16i) dot #`lhs_ext`_j dot #`rhs_ext`_(i-j) mod 2^128 \
  &stackrel(penta, equiv) sum_(i=0)^7 sum_(j=0)^(i) 2^(16i) dot #`lhs_ext`_j dot #`rhs_ext`_(i-j) mod 2^128 \
  &equiv sum_(i=0)^7 2^(16i) dot sum_(j=0)^(i) #`lhs_ext`_j dot #`rhs_ext`_(i-j) mod 2^128 \
$
where at step
- $triangle$ we can ignore $j > 7-i$, since that makes $2^(16(i+j)) equiv 0 mod 2^128$,
- $square$ we rewrite the summation that $i$ iterates from $j$ to 7, rather than $0$ to $7-j$, and
- $penta$ we swap the sums.
Note that `limb_product` is defined as the second summation in this last formula.
We can rewrite this as
$
  &sum_(i=0)^7 2^(16i) dot #`limb_product`_i mod 2^128 \
  &equiv sum_(i=0)^3 sum_(k=0)^1 2^(16(2i+k)) dot #`limb_product`_(2i+k) mod 2^128 \
  &equiv sum_(i=0)^3 2^(32i) dot sum_(k=0)^1 2^(16k) dot #`limb_product`_(2i+k) mod 2^128 \
$
where we now capture the second summation in the variable `raw_product` (see @mul:c:raw_product).

At this point, the limbs in `raw_product` may require up to 51 bits to be represented.
The last step is then to carry the overflow of each limb to the next, ensuring `res` represents the same value as `raw_product`, but with limbs in the range $[0, 2^32)$.
This is simply constrained by `carry`'s definition and @mul:c:carry.
From these two, we gather that
$
  #`raw_product`_0 - #`res`_0 in { i dot 2^32 | i in [0, 2^19) }
$
With @mul:a:res in place, $#`res`_0$ can only assume one value: the unique multiple of $2^32$ that is smaller than or equal to $#`raw_product`_0$.
In other words, $#`res`_0$ is constrained to equal $#`raw_product`_0 mod 2^32$.
The correctness of $#`res`_i$ for $i in [1, 3]$ follows analogously.

== Definitions
We constrain `lhs_is_negative` and `rhs_is_negative` according to their definition; `carry` is appropriately range checked.
#render_constraint_table(chip, config, groups: "def")

== Product
@mul:c:raw_product defines `raw_product` in terms of the input values `lhs` and `rhs`.
#render_constraint_table(chip, config, groups: "prod")

== Lookup
The #mul chip contributes the following to the lookup:
#render_constraint_table(chip, config, groups: "lookup")