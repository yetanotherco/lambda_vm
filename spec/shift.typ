#import "/book.typ": book-page, et
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
#let chip = load_chip("src/shift.toml", config)

#let shift = raw(chip.name)

#show: book-page(chip.name)

The #shift chip is designed to constrain that 
$ 
#`shifted` := cases(
  #`in` #`<<` #`s` " if" #`direction` = 0,
  #`in` #`>>` #`s` " if" #`direction` = 1 and #`signed` = 0,
  #`in` #`>>>` #`s` "if" #`direction` = 1 and #`signed` = 1,
) 
$
where
$
#`s` := cases(
  #`shift` mod 32 "if" #`word_instr` = 1,
  #`shift` mod 64 "if" #`word_instr` = 0,
) 
$
Here, `<<` and `>>` denote the _logical_ left and right shift operations, while `>>>` denotes the _arithmetic_ right shift operation.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The `SHIFT` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Explanation
This chip has a rather complex design as a result of designing it to fit in as few columns possible.
We briefly discuss the intricacies of the design, attempting to illustrate its correctness.

The chip's design revolves around a two-phase shifting process:
1. shift `in` by $x := #`shift` mod 16$ bits, 
2. shift that result by $(#`shift`-x) mod 64$ (or $mod 32$ if $ #`word_instr` = 1$).
The intermediate value representing the state between the two phases is stored in the scratch variables `X` and `Y`.
The definition of `shifted` describes how one can combine the `X`, `Y` and `extension` variables to construct the output value as described using `Half`-limbs.
The output variable `out` is equivalent to `shifted`, but expressed using `Word`-limbs.

In the following, we cover how these two phases were designed to complement one another.
Here, we start with discussing the _logical_ left/right shift operations only; the modifications required to compute the _arithmetic_ right shift will be discussed at the end.

== First phase
We zoom in on the first step.
Here, we make use of the lookup operation `HWSL` (short for "HalfWord Shift Left"):
$ #`HWSL[x: Half, y: B4]` := [(#`x` #`<<` #`y`) mod 2^16, #`x` #`>>` (16 - #`y`)]. $
One can use this to compute `out: Half[4] := in << y` as:
$
  #`out[`i#`]` = cases(
    #`HWSL[in[`0#`], y]`_0 &"if" i = 0,
    #`HWSL[in[`i#`], y]`_0 | #`HWSL[in[`i-1#`], y]`_1 &"if" i in [1, 3]   
  )
$
as long as $#`y` < 16$.
Observing that 
$#`HWSL[x,` 16-#`y]`_0 = (#`x` #`<<` (16-#`y`)) mod 2^16$, and
$#`HWSL[x,` 16-#`y]`_1 = #`x` #`>>` #`y`$ for $#`y` in [1, 15]$,
one can also use it to compute `out := in >> y` as
$
  #`out[`i#`]` = cases(
    #`HWSL[in[`i#`],` 16-#`y]`_1 | #`HWSL[in[`i+1#`], y]`_0 &"if" i in [0, 2],
    #`HWSL[in[`3#`],` 16-#`y]`_1 &"if" i = 3
  )
$
as long as $0 < #`y` < 16$.

Observe now that the values being looked up are (almost) independent from the direction of the shift: only the shift-amount varies slightly.
When we now define
$
  #`bit_shift` := cases(
    #`shift` mod 16 & "when shifting left",
    (16-#`shift`) mod 16 & "when shifting right"
  ),  
$
it only takes some rearranging and combining of the values $#`X[`i#`] := HWSL[in[`i#`], bit_shift]`_0$ and $#`Y[`i#`] := HWSL[in[`i#`], bit_shift]`_1$ to form the limbs of $#`in <</>> shift` mod 16$.
In the remaining case that $#`right` = 1$ and $#`shift` = 0 mod 16$, the limbs of $#`in <</>> shift` mod 16$ simply match those of `in`.

== Second phase
Since we're operating on 16-bit limbs, all the limbs in $#`in <</>> shift`$ must also occur somewhere in $#`in <</>> shift` mod 16$.
The number of full-limbs we still need to shift is determined by the fifth and sixth least significant bit of `shift`.
With `limb_shift` containing a unary decoding of the integer represented by these two bits, we find that the intermediate value needs to be shifted over by $i$ limbs (to the `left` or `right`) when $#`limb_shift[`i#`]` = 1$.
These things combined yield `shifted`'s definition.

Of course, when $#`word_instr` = 1$ and, thus, only $#`shift` mod 32$ should be considered, the bit-mask for the lookup constraining `limb_shift` is adjusted appropriately (see @shift:c:limb_shift_lookup).

== Arithmetic right shift
Lastly, we discuss the case of performing the _arithmetic_ right shift.
Here, `extension` is constrained to contain a repetition of `in`'s most significant bit.
Copies of this variable are used for any full limbs shifted in when $#`right` = #`signed` = 1$.
Moreover, `X[4]` contains a copy of `extension` shifted over by the right number of bits, to allow the construction of $#`in >>> shift` mod 16$ as the appropriate intermediate.

= Constraints
First, we range check our inputs appropriately.
#render_constraint_table(chip, config, groups: "input")

Then, we constrain `bit_shift` based on whether we are left or right-shifting.
@shift:c:zbs makes sure `zbs` is set to `1` if and only if `bit_shift = 0`. 
This flag is used to indicate the special case that $#`right` = 1$ and $#`shift` = 0 mod 16$.
#render_constraint_table(chip, config, groups: "bit_shift")

Next, we shift the limbs of `in` left and right by the appropriate amount, storing the results in `X` and `Y` respectively.
When `zbs = 1`, the output cannot be used to compose $#`in >>/>>> shift` mod 16$.
To resolve this, we override `Y[i] := in[i]` and `X[i] := 0` in this case.

The case of `left`-shifting and $#`bit_shift` = 0$ will be used for padding rows.
To prevent unnecessary lookups in padding rows, we override $#`X[i]` := #`in[i]`$ and $#`Y[i]` := 0$ here.
#render_constraint_table(chip, config, groups: "intra_limb_shift")

== Full-limb shifting
Next, we constrain that `limb_shift` is a proper unary encoding of the fifth (and sixth if $#`word_instr` = 0$) bit of `shift`.
For this to be the case, three requirements must be satisfied:
+ *unary(0)*: $#`limb_shift[`i#`]` in {0, 1}$ for $i in [0, 3]$,
+ *unary(1)*: $#`limb_shift[`i#`]` = 1$ for exactly one $i$, and
+ *proper encoding*: $#`limb_shift[`i#`]` = 1 <=> 1/16 (#`shift &` (48-32 dot #`word_instr`)) = i$
The first requirement is enforced by constraint @shift:c:limb_shift_is_bit.
To construct a constraint for the second and third requirement, observe that
$
1/16 dot (#`shift &` (48-32 dot #`word_instr`)) in cases(
  {0, 1, 2, 3} &"if" #`word_instr` = 0,
  {0, 1} &"if" #`word_instr` = 1
)
$
Observe moreover that, assuming *unary(0)*, the expression
$
  1/16 dot (1 + sum_(i=0)^3 (16i-1) dot #`limb_shift[`i#`]`)
$
can evaluate to $i$ if and only if $#`limb_shift[`i#`]` = 1$, while the others are $0$.
This means that the relation
$
  1 + sum_(i=0)^3 (16i-1) dot #`limb_shift[`i#`]` = #`shift &` (48-32 dot #`word_instr`)
$
enforces both *unary(1)* and *proper encoding*.
This is the exact relation @shift:c:limb_shift_lookup enforces.


Hereafter, one must only check that `out` is the proper cast of `shifted` into a `DWordWL`.
#render_constraint_table(chip, config, groups: "limb_shifting")

== Miscellaneous 
#render_constraint_table(chip, config, groups: ("left_flag", "is_negative"))
*Note*: `is_negative` is not used when `signed = 0`.
As such, there is no problem with it being unconstrained in this case.

== Lookups
This chip adds the following interaction to the lookup.
#render_constraint_table(chip, config, groups: "lookups")

= Padding

The table can be padded to the next power of two with the following value assignments:

#render_chip_padding_table(chip, config)
