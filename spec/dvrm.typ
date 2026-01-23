#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_assumptions
)

#let config = load_config()
#let chip = load_chip("src/dvrm.toml", config)
#let dvrm = raw(chip.name)

#show: book-page.with(title: "DVRM chip")

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The `DVRM` chip is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

== Assumptions
#render_chip_assumptions(chip, config)

== Constraints
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

=== R3: Sign remainder equals sign numerator
We start with R3.
Note that the signs of `r` and `n` are captured by their most significant bits whenever $#`signed` = 1$.
As such, R3 is asserted by the constraint @dvrm:c:sign_r_equals_sign_n.

#render_constraint_table(chip, config, groups:("sign_equality", ))

=== R2: rounding towards zero
R2 states that "_[in] signed and unsigned integer division [the quotient is] round[ed] towards zero._"
In other words,
+ the sign of $#`n`-#`qd`$ must match that of `n` (unless $#`qd` = #`n`$), and 
+ $|#`n`-#`qd`|  < |#`d`|$ (unless $#`d` = 0$).

Leveraging R1 #footnote([Note: we need not worry about the _overflow_ case in applying this relation, since R5 requires specific values for `q` and `r` in this case.]), we can rewrite these as
+ the sign of $#`r`$ must match that of `n` (unless $#`r` = 0$), and 
+ $|#`r`|  < |#`d`|$ (unless $#`d` = 0$).

Focusing on the first statement, we observe that this trivially holds when $#`signed` = 0$, 
while R3 deals with the case that $#`signed` = 0$.
The second statement is enforced by @dvrm:c:abs_r_lt_abs_d.
@dvrm:c:abs_r_if_negative and @dvrm:c:abs_r_if_nonnegative, respectively @dvrm:c:abs_d_if_negative and @dvrm:c:abs_d_if_nonnegative are included to ensure that `abs_r` and `abs_d` are the absolute values of `r` respectively `d`.
@dvrm:c:abs_r_range_check and @dvrm:c:abs_d_range_check are required to uphold assumption @add:a:lhs required by the `SUB` chip.

#render_constraint_table(chip, config, groups:("abs_diff", ))

=== R5: overflow
The ISA requires that $#`q` = #`n`$ and $#`r` = 0$ in the event of overflow.
We note that, while $#`n` = #`qd` + #`r`$ (R1) does _not_ hold in the case of overflow, the relation $#`n` = |#`q`|#`d` + #`r`$ _does_.
We moreover note that the _signed_ two's complement representation of `q` is identical to the _unsigned_ representation of $|#`q`|$.
As such, by interpreting `q` as an unsigned integer when $#`overflow` = 1$, it follows that R1 will enforce $#`r` = 0$.

In summary, it suffices to enforce that $#`overflow` => #`q` = #`n`$ (@dvrm:c:q_if_overflow) and to interpret `q` as unsigned in the multiplication when $#`overflow` = 1$ (@dvrm:c:sign_q).

#render_constraint_table(chip, config, groups:"overflow")

We briefly highlight @dvrm:c:overflow.
Recall that the `overflow` flag should be set if and only if (i) $#`signed` = 1$, (ii) $#`n` = #`0x80...00`$, and (iii) $#`d` = #`0xFF...FF`$.
While the `IsEqual` template can handle the first and third equality directly, the second equality was rewritten as the set of equalities
$
  #`n[`i#`]` &= 0 "for" i in [0, 2],\
  #`n[`3#`]` - 2^15 dot #`msb_n` &= 0,\
  #`msb_n` &= 1
$
where we note that $#`n[3]` - 2^15 dot #`msb_n` = 0 <=> #`n[3]` equiv 0 mod 2^15$. 
Hence, the last two equalities are satisfied if and only if $#`n[3]` = 2^15$.

=== R1: $#`n` = #`qd` + #`r`$
Rewriting R1, we find the constraint $not#`overflow` => #`n` - #`r` = #`qd`$.
Since `n`, `d`, `q` and `r` are all 64-bit (unsigned) integers, we must assert this equality $mod 2^128$.

While one can construct the 128-bit extension of $#`n` - #`r`$ through subtraction-after-extension, extension-after-subtraction permits a more compact (yet complex) solution.
In particular, determining the sign of the result is slightly more complex.
To this end, we introduce @dvrm:sign_table.
This table lists the sign of $#`n` - #`r`$, given `signed`, $#`msb`\(#`r`)$, and $#`msb`\(#`n`)$.

#figure(table(
  columns: (auto, auto, auto, auto),
  stroke: none,
  table.header([`signed`], [`msb_n`], [`msb_r`], [`sign`]),
  table.hline(stroke: 1pt),
  table.vline(x:3),
  [0],[0],[0],[$#`msb`\(#`n`-#`r`)$],
  [0],[0],[1],[`?`],
  [0],[1],[0],[0],
  [0],[1],[1],[$#`msb`\(#`n`-#`r`)$],
  [1],[0],[0],[$#`msb`\(#`n`-#`r`)$],
  [1],[0],[1],[#sym.crossmark],
  [1],[1],[0],[1],
  [1],[1],[1],[$#`msb`\(#`n`-#`r`)$],
  ),
  caption: [Sign of `n-r`, given `signed`, `msb_r` and `msb_n`]
) <dvrm:sign_table>

First, note that the case labelled '#sym.crossmark' cannot occur due to @dvrm:c:sign_r_equals_sign_n.
Second, the case labelled '`?`' should not occur, since it implies $#`r` > #`n`$.
To this end, we introduce @dvrm:c:unsigned_implies_msb_r_lt_msb_n to prevent it from occurring.
Next, observe that 
$
  #`sign` = cases(
    #`msb`\(#`n`-#`r`\) & "when" 1 + #`msb_r` - #`msb_n` = 1,
    #`signed` & "when" 1 + #`msb_r` - #`msb_n` = 0,
  )
$
Moreover note that, ignoring the two excluded cases, $1 + #`msb_r` - #`msb_n` in {0, 1}$.
`sign_n_sub_r` is assigned its appropriate value based on these cases in @dvrm:c:sign_n_sub_r_eq_msb and @dvrm:c:sign_n_sub_r_eq_signed.
@dvrm:c:n_sub_r now computes $#`n` - #`r`$, and is combined with `sign_n_sub_r` to form `extended_n_sub_r` (see its #link(<dvrm:v:extended_n_sub_r>)[definition]).
Lastly, @dvrm:c:n_sub_r_range is included to uphold assumption @add:a:lhs required by the `SUB` chip.

#render_constraint_table(chip, config, groups:("n_sub_r", ))

With `n_sub_r` constructed, the relation $#`n` - #`r` = #`qd`$ is asserted by @dvrm:c:mul_lower and @dvrm:c:mul_upper.

#render_constraint_table(chip, config, groups:("equality", ))

=== R4: division-by-zero
R4 requires that $#`q` = 2^64-1$ (unsigned) or $-1$ (signed) and $#`r` = n$ when $#`d` = 0$.
Recalling R1, we see that $#`n` = #`q` #`d` + #`r` = #`r`$ when $#`d` = 0$, already enforcing the latter.
Next, we note that, in two's complement, the _unsigned_ value $2^64-1$ and _signed_ value $-1$ are both represented by the bit string `0xFFFFFFFF`.
Hence, only @dvrm:c:q_if_div_by_zero is required to completely constrain R5; @dvrm:c:div_by_zero just ensures the `div_by_zero` flag is set when $#`d` = 0$.

#render_constraint_table(chip, config, groups:("div_by_zero", ))

=== Other
The following constraints are included to enforce the values of `msb_n`, `msb_r`, `sign_r` and `sign_d` are correct.
#render_constraint_table(chip, config, groups:("defs", ))

=== Output
Lastly, this chip contributes the following to the lookup:
#render_constraint_table(chip, config, groups:("output", ))