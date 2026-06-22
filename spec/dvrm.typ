#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  compute_nr_interactions,
  render_constraint_table,
  render_chip_padding_table,
  render_chip_assumptions
)

#let config = load_config()
#let chip = load_chip("src/dvrm.toml", config)
#let dvrm = raw(chip.name)

The #dvrm chip provides division and remainder functionality, both signed and unsigned.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #dvrm chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)


= Constraints

First, we range-check all inputs.
#render_constraint_table(chip, config, groups: "range")

From the ISA, we gather five requirements for the `DIV[U][W]` and `REM[U][W]` instructions:
#enum(numbering: "R1.",
  enum.item([
    _For both signed and unsigned division, except in the case of_ overflow, _it holds that $#`n` = #`q` #`d` + #`r`$._
  ]),
  enum.item([
    _`DIV` and `DIVU` perform [...] signed and unsigned integer division [...] rounding towards zero._
  ]),
  enum.item([
    _For `REM`, the sign of a nonzero [remainder] equals the sign of the [numerator]._
  ]),
  enum.item([
    In case of _division-by-zero_, $#`r` = #`n`$ and $#`q` = 2^64-1$ (unsigned) or $#`q` = -1$ (signed).
  ]),
  enum.item([
    In case of _overflow_, $#`q` = #`n`$ and $#`r` = 0$
  ]),
)
where _overflow_ occurs when $#`n` = -2^(63)$ and $#`d` = -1$ (and, hence, $#`signed` = 1$), and _division-by-zero_ indicates that $#`d` = 0$.
In the following, we list the constraints associated with the #dvrm chip, and explain how these together enforce all five of these requirements.

== R3: Sign remainder equals sign numerator
We start with R3, which is straightforwardly asserted by constraint @dvrm:c:sign_r_equals_sign_n.
#render_constraint_table(chip, config, groups:("sign_equality", ))

== R2: rounding towards zero
R2 states that "_[in] signed and unsigned integer division [the quotient is] round[ed] towards zero._"
In other words,
+ the sign of $#`n`-#`qd`$ must match that of `n` (unless $#`qd` = #`n`$), and 
+ $|#`n`-#`qd`|  < |#`d`|$ (unless $#`d` = 0$).

Leveraging R1 #footnote([Note: we need not worry about the _overflow_ case in applying this relation, since R5 requires specific values for `q` and `r` in this case.]), we can rewrite these as
+ the sign of $#`r`$ must match that of `n` (unless $#`r` = 0$), and 
+ $|#`r`|  < |#`d`|$ (unless $#`d` = 0$).

Focusing on the first statement, we observe that this trivially holds when $#`signed` = 0$, 
while R3 deals with the case that $#`signed` = 1$.
The second statement is enforced by @dvrm:c:abs_r_lt_abs_d.
@dvrm:c:abs_r_if_negative and @dvrm:c:abs_r_if_nonnegative (resp. @dvrm:c:abs_d_if_negative and @dvrm:c:abs_d_if_nonnegative) are included to ensure that `abs_r` (resp. `abs_d`) is the absolute values of `r` (resp. `d`).

#render_constraint_table(chip, config, groups:("abs_diff", ))

== R5: overflow
The ISA requires that $#`q` = #`n`$ and $#`r` = 0$ in the event of overflow (i.e., when $#`n` = -2^63$ and $#`d` = -1$).
We note that the second half of this requirement is already satisfied by R2: since $#`d` = -1 != 0$, R2 requires that $|#`r`| < |#`d`| = 1$, to which $#`r` = 0$ is the only satisfying value.

We moreover find that R1 can be leveraged to enforce the correct value of `q`.
While $#`n` = #`qd` + #`r`$ (R1) does _not_ hold in the case of overflow, the relation $#`n` = |#`q`|#`d` + #`r`$ _does_.
We moreover note that the 64-bit _signed_ two's complement representation of $-2^63$ is identical to the 64-bit _unsigned_ representation of $|-2^63| = 2^63$.
As such, by interpreting `q` as an unsigned integer when $#`overflow` = 1$, it follows that R1 will enforce $#`q` = #`0x80...00`$.

In summary, in case of overflow R2 enforces that $#`r` = 0$.
Moreover it suffices to interpret `q` as unsigned integer (@dvrm:c:sign_q); R1 will ensure it contains the correct value.

#render_constraint_table(chip, config, groups:"overflow")

We highlight @dvrm:c:overflow.
Recall that the `overflow` flag should be set if and only if (i) $#`signed` = 1$, (ii) $#`n` = #`0x80...00`$, and (iii) $#`d` = #`0xFF...FF`$.
These requirements are equivalent to the state where:
$
  forall i in [0, 3]:&& 65535 - #`d`_i &= 0,\
  forall i in [0, 2]:&& #`n`_i &= 0,\
  && #`n`_3 - 2^15 dot #`sign_n` &= 0,\
  && 1 - #`sign_n` &= 0,\
$
where $#`signed` = 1$ follows from the last equality.
The requirement is phrased in this way, because the left-hand sides of the above expressions are $>= 0$ by construction.
Given that the sum of these expressions does not exceed $2^19$ (and thus never wraps in the field), we can now say that the `overflow` bit should be set to $1$ if and only if their sum evaluates to $0$.
The `ZERO` lookup guarantees this to be the case.

== R1: $#`n` = #`qd` + #`r`$
Rewriting R1, we find the constraint $not#`overflow` => #`n` - #`r` = #`qd`$.
#footnote([Recall that @dvrm:c:sign_q allows to assert this equality even when `overflow`.])
Since `n`, `d`, `q` and `r` are all 64-bit integers, we must assert this equality $mod 2^128$, rather than $mod 2^64$.
To this end, we introduce `extended_n_sub_r` and leverage the `MUL` chip to verify that it is equal to $#`qd` mod 2^128$ using constraints @dvrm:c:mul_lower and @dvrm:c:mul_upper;
@dvrm:c:q_range is included to uphold assumption @mul:c:rhs.

#render_constraint_table(chip, config, groups:("equality", ))

It now remains to enforce that `extended_n_sub_r` is the _signed_ 128-bit representation of $#`n`-#`r`$.
Here, we introduce `extended_n` and `extended_r`.
By their definition, these variables contain the signed 128-bit representations of `n` and `r`.
The `carry` variable has been defined such that it mimics those in the `ADD` chip,
except that here we add two `QuadHL`s rather than two `DWordHL`, thus needing four carry bits instead of two.
With this in place, @dvrm:c:n_sub_r (mimicking @add:c:carry) ensures `extended_n_sub_r` must contain the correct value.

Lastly, observe that $#`n` - #`r` in (-2^64, 2^64)$, _regardless_ of the value of `signed`.
Moreover, note that the upper halves of the 128-bit representations of all values in this range are either `0xFFFFFFFF` (negative) or `0x00000000` (non-negative).
This means that we do not need to store all 128 bits of `extended_n_sub_r`.
Rather, we need only store the lower 64-bits, and a separate bit (`sign_n_sub_r`) indicating whether the top limbs are all-ones or all-zeroes.
The prover is free to select the value for `sign_n_sub_r`; only one of the two will fit the proof.

#render_constraint_table(chip, config, groups:("n_sub_r", ))

== R4: division-by-zero
R4 requires that $#`q` = 2^64-1$ (unsigned) or $-1$ (signed) and $#`r` = n$ when $#`d` = 0$.
Recalling R1, we see that $#`n` = #`q` #`d` + #`r` = #`r`$ when $#`d` = 0$, already enforces the latter.
Next, we note that, in two's complement, the _unsigned_ value $2^64-1$ and _signed_ value $-1$ are both represented by the bit string `0xFFFFFFFF`.
Hence, only @dvrm:c:q_if_div_by_zero is required to completely constrain R4; @dvrm:c:div_by_zero just ensures the `div_by_zero` flag is set when $#`d` = 0$.

#render_constraint_table(chip, config, groups:("div_by_zero", ))

== Other
The following constraints are included to enforce the values of `sign_n`, `sign_r` and `sign_d` are correct.
#render_constraint_table(chip, config, groups:("defs", ))

== Output
Lastly, this chip contributes the following to the lookup:
#render_constraint_table(chip, config, groups:("output", ))

= Padding
To pad the #dvrm table, we use the following data, representing the unsigned division $frac(0, 0, style: "horizontal")$:
#render_chip_padding_table(chip, config)
