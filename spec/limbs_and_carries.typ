#import "/book.typ": book-page
#import "@preview/ctheorems:1.1.3": *
#import "@preview/equate:0.3.2": equate


// Theorem/lemma formatting
#show: thmrules.with(qed-symbol: $square$)
#let lemma = thmbox("lemma", "Lemma", fill: rgb("#eee"),base_level: 0)
#let proof = thmproof("proof", "Proof")

// Equation formatting
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")
#show math.equation: box

#show: book-page("limbs_and_carries.typ")

In this section, we discuss, in order, 
+ the multiplication and addition of limb-decomposed integers (involving carries), 
+ prove an upper bound on the size of the carries in terms of the number of
  multiplications and additions, and
+ prove correctness of a set of constraints that can be used to constrain quadratic relations.

= Limb decomposition
Let $[X]$ denote the set ${0, ..., X-1} subset.eq NN$.
Let $S := L^n in NN$ be an upper bound on the integers we want to represent, 
where $L in NN without {0, 1}$ is the number of values a limb can represent 
and $n in N$ denotes the number of limbs.

Observe that for all $x in [S]$, there exists a unique tuple of integers 
$(x_0, x_1, x_2, x_(n-1)) in [L]^(n)$ such that
$
  x 
  = sum_(i=0)^(n-1) x_i dot L^i.
$ #<eq:decomposition>
To simplify future notation, we let $x_i = 0$ for all $i >= n$.

Let us define the family of multi-linear functions $f_(alpha, mu) (x, y, z) := mu dot x dot y + alpha dot z$, 
over variables $x, y, z in [S]$ and parameters $alpha, mu in NN$.
Working towards the limb-decomposition of $f_(alpha, mu)$, we rewrite
$
  f_(alpha, mu) (x, y, z) 
  &= mu dot x dot y + alpha dot z\
  &= mu dot (sum_(i=0)^(n-1) x_i dot L^i) dot (sum_(j=0)^(n-1) y_j dot L^j) + alpha dot sum_(k=0)^(n-1) z_k dot L^k\
  &= (sum_(i=0)^(n-1) sum_(j=0)^(n-1) mu dot x_i dot y_j dot L^(i + j)) + sum_(k=0)^(n-1) alpha dot z_k dot L^k\
  &= sum_(i=0)^(n-1) (alpha dot z_i dot L^i + sum_(j=0)^(n-1) mu dot x_i dot y_j dot L^(i + j))\
  &= sum_(r=0)^(2(n-1)) alpha dot z_r dot L^r + sum_(j=0)^r mu dot x_(r-j) dot y_j dot L^r\
  &= sum_(r=0)^(2(n-1)) (alpha dot z_r + mu dot sum_(j=0)^r x_(r-j) dot y_j) dot L^r\
  &= sum_(r=0)^(2(n-1)) overline(w)_r dot L^r
$
where
$
  overline(w)_i := alpha dot z_i + mu dot sum_(j=0)^i x_(i-j) dot y_j.
$
While this decomposition closely resembles that of a limb-decomposition (@eq:decomposition), 
there is the problem that $overline(w)_i$ will exceed $L$ for sufficiently large $alpha, mu, x, y, z$, and $i$.
We can introduce helper sequence $c_i$ to transform $overline(w)_i$ into a proper decomposition $w_i$ as:
$
  w_i &:= overline(w)_i + c_(i-1) mod L &text("for") i >= 0,\
  c_i &:= (overline(w)_i + c_(i-1) - w_i ) dot L^(-1) &text("for") i >= 0,\
$
with $c_i in NN$ and $c_(-1) := 0$.
Note that these $c_i$ effectively move the "overflow" from one limb to the next limb up; 
they're commonly referred to as the _carry_ values.

#lemma[
    For all $g >= 2n$, 
    $(w_0, w_1, ..., w_(g-1)) in [L]^(g)$ 
    is the unique $g$-limb decomposition of $f_(alpha, mu)$
    if and only if $c_(g-1) = 0$.
]<lm:wi_decomp_f>
#proof[
    Reordering the definition of $c_i$, we find the equality $w_i = overline(w)_i + c_(i-1) - c_i dot L$.
    Leveraging this, we see that
    $  
    sum_(r=0)^(g-1) w_r dot L^r
    &= sum_(r=0)^(g-1) (overline(w)_r + c_(r-1) - c_r dot L) dot L^r\
    &= (sum_(r=0)^(g-1) overline(w)_r dot L^r) + (sum_(s=0)^(g-1) c_(s-1) dot L^s) - (sum_(t=0)^(g-1) c_t dot L^(t+1))\
    &= (sum_(r=0)^(g-1) overline(w)_r dot L^r) + (sum_(s=-1)^(g-2) c_(s) dot L^(s+1)) - (sum_(t=0)^(g-1) c_t dot L^(t+1))\
    &= (sum_(r=0)^(g-1) overline(w)_r dot L^r) + c_(-1) - c_(g-1) dot L^(g-1+1)\
    &= (sum_(r=0)^(g-1) overline(w)_r dot L^r) - c_(g-1) dot L^(g),\
    &= (sum_(r=0)^(2(n-1)) overline(w)_r dot L^r) - c_(g-1) dot L^(g),\
    &= f_(alpha, mu)(x, y, z) + c_(g-1) dot L^(g),\
    $
    where the second-to-last step follows from the observation that $overline(w)_j = 0$ for $j >= 2n-1$.
    This limb decomposition is proper if and only if $c_(g-1) = 0$.
]

= Upper bounding the carry
To find the conditions under which $c_(g-1) = 0$, we prove an upperbound for $c_i$.

#lemma("Carry upper bound [part 1]")[
  For $alpha, mu in [L]$, it holds that
  $
  c_i <= mu (i+1) (L-1) + alpha - mu - delta_(mu < alpha)
  $
  where kronecker delta $delta_x$ equals $1$ if $x$ holds, and $0$ otherwise.
]<lm:carry-upperbound-pt1>

#proof[
  Since $w_i in [L]$,
  $c_i 
  :=(overline(w)_i + c_(i-1) - w_i ) dot L^(-1) 
  = floor.l (overline(w)_i + c_(i-1)) dot L^(-1) floor.r.$
  Hence, $c_i$ is maximized when both $overline(w)_i$ and $c_(i-1)$ are, and thus, 
  by induction, when $overline(w)_j$ is maximized for all $j in [0, i]$.
  Given that $x_i, y_i, z_i <= L-1$, it follows that
  $
  &overline(w)_0
  &=& alpha dot z_0 + mu dot sum_(j=0)^0 x_(0-j) dot y_j\
  &&<=& alpha (L-1) + mu (L - 1)^2 \
  &&=& (alpha - delta) L + delta L - alpha + mu (L^2 - 2L) + mu\
  &&=& (alpha - delta) L + mu (L^2 - 2L) + delta L + mu - alpha,\
  text("and hence") &c_0 
  &<=& floor.l ((alpha - delta) L + mu (L - 2)L + delta L + mu - alpha) dot L^(-1) floor.r\
  &&=& alpha - delta + mu (L - 2)  \
  &&=& mu dot 1 dot (L - 1) + alpha - mu - delta \
  $
  where $delta := delta_(mu < alpha)$.
  Assuming the statement holds for up to some $i >= 0$, we find that
  $
  overline(w)_(i+1)
  &= alpha dot z_i + mu dot sum_(j=0)^(i+1) x_(i+1-j) dot y_j\
  &<= alpha (L-1) +  mu (i+2) (L-1)^2,\
  c_(i+1) 
  &<= floor.l (alpha (L-1) + mu (i+2) (L-1)^2 + mu (i+1) (L-1) + alpha - mu - delta) dot L^(-1) floor.r\
  &= floor.l ((alpha - delta) L + mu (i+2) (L - 2)L + mu (i+1) L + delta (L-1)) dot L^(-1) floor.r\
  &= alpha - delta + mu (i+2) (L - 2) + mu (i+1)\
  &= mu (i+2) (L - 1) + alpha - mu - delta.
  $
]

Inspecting this upper bound, we find that it is _tight_ for all $i<n$; for $x = y = z = S-1$, $c_i$ achieves the upper bound.
Since $x_i = y_i = z_i = 0$ for $i >= n$, the upper bound for $overline(w)_i$ is no longer tight.
We therefore introduce a second lemma:

#lemma("Carry upper bound [part 2]")[
  For $alpha in [L], mu in [L/2]$, it holds that
  $
    c_(n+k) <= mu (n - k - 1)(L-2) + mu (n-k) - delta_(alpha < 2mu + delta)
  $
  for $k in [0, n)$.
]<lm:carry-upperbound-pt2>

#proof[
  Starting with $k=0$, we find
  $
  overline(w)_(n)
  &= alpha dot z_n + mu dot sum_(j=0)^(n) x_(n-j) dot y_j\
  &= mu dot sum_(j=1)^(n-1) x_(n-j) dot y_j\
  &<= mu (n-1) (L-1)^2
  $
  since $x_i, y_i, z_i = 0$ for $i >= n$.
  Applying this upper bound to $c_n$, we obtain
  $
  c_(n)
  &= floor.l (overline(w)_n + c_(n-1)) dot L^(-1) floor.r\
  &<= floor.l (mu (n-1) (L-1)^2 + mu n (L-1) + alpha - mu - delta) dot L^(-1) floor.r\
  &= floor.l (mu (n-1) (L-2)L + mu (n-1) + mu n (L-1) + alpha - mu - delta) dot L^(-1) floor.r\
  &= floor.l (mu (n-1) (L-2)L + (mu n - delta') L + delta'L + alpha - 2mu - delta) dot L^(-1) floor.r\
  &= mu (n-1) (L-2) + mu n - delta'\
  &= mu (n-0-1) (L-2) + mu (n-0) - delta'\
  $
  where $delta' := delta_(alpha < 2mu + delta)$.
  When the bound holds for some $i=n+k-1$ with $k in [1, n)$, it follows that
  $
  overline(w)_(n+k)
  &= alpha dot z_(n+k) + mu dot sum_(j=0)^(n+k) x_(n+k-j) dot y_j\
  &= mu dot sum_(j=k+1)^(n-1) x_(n+k-j) dot y_j\
  &<= mu (n-k-1) (L-1)^2\
  c_(n+k)
  &<= floor.l (mu (n-k-1) (L-1)^2 + mu (n-k)(L-2) + mu (n - k + 1) - delta') dot L^(-1) floor.r\
  &= floor.l (mu (n-k-1) (L-2)L + mu (n-k)L - delta') dot L^(-1) floor.r\
  &= floor.l (mu (n-k-1) (L-2)L + mu (n-k)L - delta'L + delta'(L-1)) dot L^(-1) floor.r\
  &= mu (n-k-1) (L-2) + mu (n-k)-delta'.
  $
  The claimed upper bound now follows for all $k in [0, n)$ by induction. 
]
Again, we note that this upper bound is tight; 
$c_(n+k)$ achieves the bound for all $k in [0, n)$ when $x = y = z = S-1$.
This includes $k = n-1$, which yields $c_(n+(n-1)) = c_(2n-1) <= mu - delta'$.
Moreover, $c_(2n) <= floor.l (mu - delta') dot L^(-1) floor.r = 0$ since $mu in [L]$ and thus $c_(2n + i) = 0$ for all $i >= 0$.
Applying this to @lm:wi_decomp_f, we conclude that $(w_0, w_1, ..., w_(2n-1)) in [L]^(2n)$ is the unique limb decomposition
of $f_(alpha, mu) (x, y, z)$ when $mu = 0$ and $alpha < L$, or $mu = 1$ and $alpha <= 2$.

Combining both bounds, we find that
$
&max(&max_(i in [n]) #h(1em) mu (i+1) (L-1) + alpha - mu - delta,\ 
  &max_(k in [n]) #h(1em) mu (2n-k-1)(L-2) + mu (2n-k) - delta')\
&= max(& mu n (L-1) + alpha - mu - delta, mu (2n-1)(L-2) + 2 mu n - delta')\
&= mu n (L-1) + alpha - mu - delta
$
acts as an upper bound on all carry elements.


= Proof of Correctness
Lastly, we prove that there exists a correct method of constraining the relation between $overline(w)_i$, $w_i$ and $c_i$ inside this VM:

#lemma("Constraint correctness")[
    Let $c_i, w_i in FF_p$ with $p$ prime. 
    The constraints
    $
        c_i &= (overline(w)_i + c_(i-1) - w_i) dot L^(-1), #<eq:def_ci>\
        c_i &in [C], #<eq:range_ci>\
        c_(-1) &= 0, #<eq:c_-1_is_zero>\
        w_i &in [L] #<eq:range_wi>
    $
    together enforce $w_i = overline(w)_i + c_(i-1) mod L$ as long as $C in [mu n L + alpha, frac(p,L, style:"horizontal"))$.
]<lm:limb-decomposition-constraint-correctness>

#proof[
    Combining @eq:def_ci and @eq:range_ci, we find that
    $
    &&c_i &in [C]\
    &<=>& overline(w)_i + c_(i-1) - w_i &in {0, L, ..., (C-1)L},\
    &<=>& w_i &in {overline(w)_i + c_(i-1), overline(w)_i + c_(i-1) - L, ..., overline(w)_i + c_(i-1) - (C-1)L}.
    $
    Let us use $X_(i)$ to refer to this last set.
    Now --- under the assumption that $c_(i-1)$ is correct --- 
    observe that $w_i := overline(w)_i + c_(i-1) mod L in X_i$, since
    $
        overline(w)_i + c_(i-1)
        &<= (mu n (L-1)^2 + alpha (L-1)) + (mu (n-1) (L-1) + alpha - mu - delta)\
        &<= mu n L^2 + alpha L\
        &<= C L,\
    $
    where the utilized upper bound 
    $overline(w)_i <= mu n(L-1)^2 + alpha (L-1)$ 
    can be extracted from the proofs of @lm:carry-upperbound-pt1 and @lm:carry-upperbound-pt2.
    Moreover, observe that $|X_i inter [L]| <= 1$ since $C < frac(p,L, style:"horizontal")$.
    Constraint @eq:range_wi therefore enforces that $w_i = overline(w)_i + c_(i-1) mod L$ if $c_(i-1)$ (and therefore $w_(i-1)$) is correct, 
    and as a result, $c_i = floor.l frac(overline(w)_i + c_(i-1), L, style: "horizontal") floor.r$ is correct.
    The proof now follows by induction, with @eq:c_-1_is_zero enforcing the base case.
]
