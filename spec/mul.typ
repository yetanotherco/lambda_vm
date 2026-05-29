#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  compute_nr_interactions,
  render_constraint_table,
  render_chip_assumptions,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/mul.toml", config)

#show: book-page(chip.name)

#let mul = raw(chip.name)

The #mul chip constrains multiplication, both signed and unsigned,
as well as providing access to the low and high halfs of the multiplication result.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #mul chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

#let stackrel(top, bottom) = {
 $mat(delim: #none, top; bottom)$
}


= Constraints
== Overview
When `lhs` and `rhs` are _unsigned_ integers, computing their product $mod 2^128$ comes down to evaluating
$
(sum_(j=0)^3 2^(16j) dot #`lhs`_j) dot (sum_(i=0)^3 2^(16i) dot #`rhs`_i) mod 2^128.
$
If `lhs` and `rhs` are signed instead, the computation remains nearly identical: 
based on their signs, one must either zero or one-extend `lhs` and `rhs` --- forming `lhs_ext` and `rhs_ext` respectively --- and compute their product $mod 2^128$:
$
(sum_(j=0)^7 2^(16j) dot #`lhs_ext`_j) dot (sum_(i=0)^7 2^(16i) dot #`rhs_ext`_i) mod 2^128.
$
where `lhs_ext` and `rhs_ext` are treated as _unsigned_ integers.
Note that by setting the extension limbs of `lhs` and/or `rhs` to $0$ when the integer is (i) unsigned or (ii) signed and non-negative, this second formula still applies.
For the purposes of constraining the multiplication operation, we rewrite this formula as
#show math.equation: set block(breakable: true)
$
  &(sum_(j=0)^7 2^(16j) dot #`lhs_ext`_j) dot (sum_(i=0)^7 2^(16i) dot #`rhs_ext`_i) mod 2^128 \
  &equiv sum_(j=0)^7 sum_(i=0)^7 2^(16(i+j)) dot #`lhs_ext`_j dot #`rhs_ext`_i mod 2^128 \
  &stackrel(triangle, equiv) sum_(j=0)^7 sum_(i=0)^(7-j) 2^(16(i+j)) dot #`lhs_ext`_j dot #`rhs_ext`_i mod 2^128 \
  &stackrel(square, equiv) sum_(j=0)^7 sum_(i=j)^(7) 2^(16i) dot #`lhs_ext`_j dot #`rhs_ext`_(i-j) mod 2^128 \
  &stackrel(penta, equiv) sum_(i=0)^7 sum_(j=0)^(i) 2^(16i) dot #`lhs_ext`_j dot #`rhs_ext`_(i-j) mod 2^128 \
  &equiv sum_(i=0)^3 sum_(k=0)^1 sum_(j=0)^(2i+k) 2^(16(2i+k)) dot #`lhs_ext`_j dot #`rhs_ext`_(2i+k-j) mod 2^128 \
  &equiv sum_(i=0)^3 2^(32i) dot sum_(k=0)^1 2^(16k) dot sum_(j=0)^(2i+k) #`lhs_ext`_j dot #`rhs_ext`_(2i+k-j) mod 2^128
$
where at step
- $triangle$ we can ignore $i > 7-j$, since that makes $2^(16(i+j)) equiv 0 mod 2^128$,
- $square$ we rewrite the second summation such that $i$ iterates from $j$ to 7, rather than $0$ to $7-j$, and
- $penta$ we swap the sums.

We let `raw_product` capture the second summation in this last formula (see @mul:c:raw_product).
By construction, $#`raw_product`_i < 2^51$ for all $i in [0, 3]$, far exceeding the 32-bits that fit in a single `Word`-limb.
What remains then is to reduce each limb of `raw_product` $mod 2^32$, carrying the overflow of each limb to the next, constructing the output `res` in doing so.

This reduce-and-carry operation is constrained by @mul:c:range_lo/@mul:c:range_hi and @mul:c:carry, combined with `carry`'s definition.
@mul:c:carry and `carry`'s definition enforce that
$
  forall i in [0, 3]: #`raw_product`_i + #`carry`_(i-1) - #`res`_i in { k dot 2^32 | k in [0, 2^20) }
$
with $#`carry`_(-1) = 0$ for simplicity.
In other words: $#`res`_i equiv #`raw_product`_i + #`carry`_(i-1) (mod 2^32)$.
With @mul:c:range_lo/@mul:c:range_hi forcing $#`res`_i < 2^32$, $#`res`_i$ can only assume one value: $#`raw_product`_i + #`carry`_(i-1) mod 2^32$.

*Note*: one may have observed that @mul:c:carry requires $#`carry`_i in [0, 2^20)$, while no limb of a valid carry value would ever exceed $2^19$.
This is indeed the case.
However, there is some slack in how tight one has to constrain the `carry` values.
In fact, in this situation it suffices to assert that $#`carry`_i < frac(p, 2^32, style: "skewed") approx 2^31$, where $p$ denotes the field's modulus.
Given that other chips also use 20-bit lookups, using `IS_B20` makes for a simpler design.

== Definitions
We constrain `lhs_is_negative` and `rhs_is_negative` according to their definition; `lo`, `hi` and `carry` are appropriately range checked.
#render_constraint_table(chip, config, groups: "def")

== Product
@mul:c:raw_product defines `raw_product` in terms of the (sign extended) input values `lhs` and `rhs`.
#render_constraint_table(chip, config, groups: "prod")

== Lookup
The #mul chip contributes the following to the lookup:
#render_constraint_table(chip, config, groups: "lookup")

= Padding

The table can be padded to the next power of two with the following value assignments:

#render_chip_padding_table(chip, config)

= Notes/optimizations
- `lo` and `hi` are stored in `DWordHL`s (rather than `DWordWL`s) because of their values being range checked.
  Since it is not required that both `μ_lo` and `μ_hi` are non-zero at the same time, one cannot safely assume their range to be checked elsewhere.
- As an optimization, one might be able to use a `DWordWL` and `DWordHL` to store `lo` and `hi`, 
  where one would decide which to store in which based on the multiplicities `μ_lo` and `μ_hi`;
  the value sent into the lookup could then be assumed range-checked by the other side of the relation.
  This optimization was not included at this moment because of its negative impact on the readability and verifiability of the chip.
