#import "/book.typ": book-page
#import "@preview/ctheorems:1.1.3": *
#import "@preview/equate:0.3.2": equate


// Theorem/lemma formatting
#show: thmrules.with(qed-symbol: $square$)
#let lemma = thmbox("lemma", "Lemma", fill: rgb("#eee"),base_level: 0)
#let corollary = thmbox("lemma", "Corollary", fill: rgb("#eee"), base_level: 0)
#let proof = thmproof("proof", "Proof")

// Equation formatting
#show: equate.with(breakable: true, sub-numbering: true, number-mode: "label")
#set math.equation(numbering: "(1.1)")
#show math.equation.where(block: false): box

#show: book-page("limbs_and_carries.typ")

In this section, we discuss, in order, 
+ the multiplication and addition of limb-decomposed integers (involving carries), 
+ prove an upper bound on the size of the carries in terms of the number of
  multiplications and additions, and
+ prove correctness of a set of constraints that can be used to constrain quadratic relations.

= Limb decomposition
Let $[X]$ denote the set ${0, ..., X-1} subset.eq NN$, and $[X]^n$ with $n in NN$ the $n$-dimensional self-product of this set.
Let $S := L^n in NN$ be an upper bound on the integers we want to represent, 
where $L in NN$ with $L >= 4$ is the number of values a limb can represent 
and $n in NN$ denotes the number of limbs.

Observe that for all $x in [S]$, there exists a unique "limb decomposition"
$(x_0, x_1, x_2, ...,  x_(n-1)) in [L]^(n)$ such that
$
  x 
  = sum_(i=0)^(n-1) x_i dot L^i.
$ #<limbs:eq:decomposition>
To simplify future notation, we define $x_i := 0$ for all $i >= n$.

#let vec(x) = math.bold(math.upright(x))

Next, we define the family of functions 
$
f_(mu, alpha) (vec(x), vec(y), vec(z)) := sum_(m in [mu]) (vec(x)^((m)) dot vec(y)^((m))) + sum_(a in [alpha]) vec(z)^((a))
$
over variables $vec(x), vec(y) in [S]^mu$ and $vec(z) in [S]^alpha$, with parameters $mu, alpha in NN$.
Working towards the limb decomposition of $w = f_(mu, alpha)(vec(x), vec(y), vec(z))$, we rewrite

$
  f_(mu, alpha) (vec(x), vec(y), vec(z)) 
  &= sum_(m in [mu]) (sum_(i=0)^(n-1) vec(x)^((m))_i dot L^i) (sum_(j=0)^(n-1) vec(y)^((m))_j dot L^j) + sum_(a in [alpha]) (sum_(k=0)^(n-1) vec(z)^((a))_k dot L^k)\
  &= sum_(m in [mu]) (sum_(i=0)^(n-1) sum_(j=0)^(n-1) vec(x)^((m))_i dot vec(y)^((m))_j dot L^(i+j)) + sum_(a in [alpha]) (sum_(k=0)^(n-1) vec(z)^((a))_k dot L^k)\
  &= sum_(i=0)^(n-1) sum_(j=0)^(n-1) (sum_(m in [mu]) vec(x)^((m))_i dot vec(y)^((m))_j dot L^(i+j)) + sum_(k=0)^(n-1) (sum_(a in [alpha]) vec(z)^((a))_k dot L^k)\
  &= sum_(i=0)^(n-1) (sum_(j=0)^(n-1) (sum_(m in [mu]) vec(x)^((m))_i dot vec(y)^((m))_j dot L^(i+j)) + sum_(a in [alpha]) vec(z)^((a))_i dot L^i )\
  &= sum_(r=0)^(2(n-1)) (sum_(j=0)^(r) (sum_(m in [mu]) vec(x)^((m))_(r-j) dot vec(y)^((m))_j dot L^r) + sum_(a in [alpha]) vec(z)^((a))_r dot L^r )\
  &= sum_(r=0)^(2(n-1)) (sum_(j=0)^(r) (sum_(m in [mu]) vec(x)^((m))_(r-j) dot vec(y)^((m))_j) + sum_(a in [alpha]) vec(z)^((a))_r) dot L^r\
  &= sum_(r=0)^(2(n-1)) overline(w)_r dot L^r, #<limbs:eq:f-semi-decomposition>
$
where
$
  overline(w)_i := sum_(m in [mu]) (sum_(j=0)^(i) vec(x)^((m))_(i-j) dot vec(y)^((m))_j) + sum_(a in [alpha]) vec(z)^((a))_i.
$

While @limbs:eq:f-semi-decomposition closely resembles that of a limb-decomposition (@limbs:eq:decomposition), 
there is the problem that $overline(w)_i$ will generally not be bounded by $L$.
We therefore introduce a helper sequence, $c_i$, to transform $overline(w)_i$ into a proper decomposition $w_i$ as:
$
  w_i &:= overline(w)_i + c_(i-1) mod L &text("for") i >= 0,\
  c_i &:= (overline(w)_i + c_(i-1) - w_i )/L &text("for") i >= 0,\
$
with $c_i in NN$ and $c_(-1) := 0$.
Note that these $c_i$ effectively move the "overflow" from one limb to the next limb up; 
they're commonly referred to as the _carry_ values.

#lemma[
    For all $g >= 2n-1$, 
    $(w_0, w_1, ..., w_(g-1)) in [L]^(g)$ 
    is the unique $g$-limb decomposition of $w = f_(mu, alpha)(vec(x), vec(y), vec(z))$
    if and only if $c_(g-1) = 0$.
]<limbs:lm:wi_decomp_f>
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
    &= f_(mu, alpha)(vec(x), vec(y), vec(z)) - c_(g-1) dot L^(g),\
    $
    where the second-to-last step follows from the observation that $overline(w)_j = 0$ for $j > 2(n-1)$.
    We conclude that $w_i$ is a proper $g$-limb decomposition of $w$ if and only if $c_(g-1) = 0$.
]

= Upper bounding the carry
To bound for which $g$ we can guarantee that $c_(g-1) = 0$, we prove two upper bounds for $c_i$.

#lemma("Carry upper bound [part 1]")[
  For $alpha, mu in [L]$, it holds that
  $
  c_i <= mu (i+1) (L-1) + alpha - mu - delta_(mu < alpha)
  $
  where kronecker delta $delta_x$ equals $1$ if $x$ holds, and $0$ otherwise.
]<limbs:lm:carry-upperbound-pt1>

#proof[
  Since $w_i in [L]$,
  $c_i 
  := frac((overline(w)_i + c_(i-1) - w_i ), L, style: "horizontal")
  = floor.l frac((overline(w)_i + c_(i-1)), L, style: "horizontal") floor.r.$
  Hence, $c_i$ is maximized when both $overline(w)_i$ and $c_(i-1)$ are, and thus, 
  by induction, when $overline(w)_j$ is maximized for all $j <= i$.
  Given that for all $m in [mu], a in [alpha], i in [n]: vec(x)^((m))_i, vec(y)^((m))_i, vec(z)^((a))_i <= L-1$, it follows that
  $

  &overline(w)_0
  &=& sum_(m in [mu]) (vec(x)^((m))_0 dot vec(y)^((m))_0) + sum_(a in [alpha]) vec(z)^((a))_0\
  &&<=& mu (L - 1)^2 + alpha (L-1)\
  &&=& mu (L - 2)L + mu + (alpha - delta) L + delta L - alpha\
  &&=& mu (L - 2)L + (alpha - delta) L + delta L + mu - alpha,\
  text("and hence") &c_0 
  &<=& lr(floor.l (mu (L - 2)L + (alpha - delta) L + delta L + mu - alpha)/ L floor.r)\
  &&=& mu (L - 2) + alpha - delta \
  &&=& mu dot 1 dot (L - 1) + alpha - mu - delta \
  $
  where $delta := delta_(mu < alpha)$.
  Assuming the statement holds for up to some $i >= 0$, we find that
  $
  overline(w)_(i+1)
  &= sum_(m in [mu]) (sum_(j=0)^(i+1) vec(x)^((m))_(i+1-j) dot vec(y)^((m))_j) + sum_(a in [alpha]) vec(z)^((a))_(i+1) \
  &<= mu (i+2)(L-1)^2 + alpha (L-1),\
  c_(i+1) 
  &<= lr(floor.l (mu (i+2) (L-1)^2 + alpha (L-1) + mu (i+1) (L-1) + alpha - mu - delta)/L floor.r)\
  &= lr(floor.l (mu (i+2) (L - 2)L + mu (i+1) L + (alpha - delta) L + delta (L-1))/ L floor.r)\
  &= mu (i+2) (L - 2) + mu (i+1) + alpha - delta\
  &= mu (i+2) (L - 1) + alpha - mu - delta.
  $
]

Inspecting this upper bound, we find that it is _tight_ for all $i<n$; for $vec(x)^((m)) = vec(y)^((m)) = vec(z)^((a)) = S-1$, $c_i$ achieves the upper bound.
To achieve a tight upper bound for $i >= n$, we introduce a second lemma:

#lemma("Carry upper bound [part 2]")[
  For $alpha in [L], mu in [L/2]$, it holds that
  $
    c_(n+k) <= mu (n - k - 1)(L-2) + mu (n-k) - delta_(alpha < 2mu + delta)
  $
  for $k in [n]$.
]<limbs:lm:carry-upperbound-pt2>

#proof[
  Starting with $k=0$, we find
  $
  overline(w)_n 
  &= sum_(m in [mu]) (sum_(j=0)^(n) vec(x)^((m))_(n-j) dot vec(y)^((m))_j) + sum_(a in [alpha]) vec(z)^((a))_n\
  &<= mu (n-1) (L-1)^2
  $
  since $vec(x)^((m))_i$, $vec(y)^((m))_i$, and $vec(z)^((a))_i$ are $0$ for $i >= n$.
  Applying this upper bound to $c_n$, we obtain
  $
  c_(n)
  &= lr(floor.l (overline(w)_n + c_(n-1))/ L floor.r)\
  &<= lr(floor.l (mu (n-1) (L-1)^2 + mu n (L-1) + alpha - mu - delta)/ L floor.r)\
  &= lr(floor.l (mu (n-1) (L-2)L + mu (n-1) + mu n (L-1) + alpha - mu - delta) / L floor.r)\
  &= lr(floor.l (mu (n-1) (L-2)L + (mu n - delta') L + delta'L + alpha - 2mu - delta) / L floor.r)\
  &= mu (n-1) (L-2) + mu n - delta'\
  &= mu (n-0-1) (L-2) + mu (n-0) - delta'\
  $
  where $delta' := delta_(alpha < 2mu + delta)$.
  When the bound holds for some $i=n+k-1$ with $k in [1, n)$, it follows that
  $
  overline(w)_(n+k)
  &= sum_(m in [mu]) (sum_(j=0)^(n+k) vec(x)^((m))_(n+k-j) dot vec(y)^((m))_j) + sum_(a in [alpha]) vec(z)^((a))_(n+k)\
  &<= mu (n-k-1) (L-1)^2\
  c_(n+k)
  &<= lr(floor.l (mu (n-k-1) (L-1)^2 + mu (n-k)(L-2) + mu (n - k + 1) - delta') / L floor.r)\
  &= lr(floor.l (mu (n-k-1) (L-2)L + mu (n-k)L - delta') / L floor.r)\
  &= lr(floor.l (mu (n-k-1) (L-2)L + mu (n-k)L - delta'L + delta'(L-1)) / L floor.r)\
  &= mu (n-k-1) (L-2) + mu (n-k)-delta'.
  $
  The claimed upper bound now follows for all $k in [0, n)$ by induction. 
]
Note that this upper bound is tight: 
$c_(n+k)$ achieves the bound for all $k in [n]$ when $vec(x)^((m)) = vec(y)^((m)) = vec(z)^((a)) = S-1$ for all $m in [mu]$ and $a in [alpha]$.
For $k = n-1$, this yields $c_(n+(n-1)) = c_(2n-1) <= mu - delta'$, which evaluates to zero when $mu = 0$ and $alpha < L$, or $mu = 1$ and $alpha <= 2$.
We can therefore conclude that $(w_0, w_1, ..., w_(2n-1)) in [L]^(2n)$ is a valid $2n$-limb decomposition of $f_(mu, alpha) (vec(x), vec(y), vec(z))$ in these cases.
For larger values of $mu in [2, L/2]$, we note that $c_(2n) <= floor.l frac((mu - delta'), L, style: "horizontal") floor.r = 0$ and thus $c_(2n + i) = 0$ for all $i >= 0$.
Hence, attaching extra limb $w_(2n) := c_(2n)$ yields a $(2n+1)$-limb decomposition for these cases.

Combining both upper bounds, we now find that

#corollary[
Given $alpha in [L]$ and $mu in [L/2]$, then for all $i <= 2n$:
$
c_i <= &max(&max_(i in [n]) #h(1em) mu (i+1) (L-1) + alpha - mu - delta,\ 
  &max_(k in [n]) #h(1em) mu (n-k-1)(L-2) + mu (n-k) - delta')\
&= max(& mu n (L-1) + alpha - mu - delta, mu (n-1)(L-2) + mu n - delta')\
&= mu n (L-1) + alpha - mu - delta
$
]<limbs:cor:carry-upper-bound>

= Proof of Correctness
Lastly, we prove that there exists a correct method of constraining the relation between $overline(w)_i$, $w_i$ and $c_i$ inside this VM:

#lemma("Constraint correctness")[
    Let $c_i, w_i in FF_p$ with $p$ prime. 
    The constraints
    $
        c_i &= (overline(w)_i + c_(i-1) - w_i) dot L^(-1), #<limbs:eq:def_ci>\
        c_i &in [C], #<limbs:eq:range_ci>\
        c_(-1) &= 0, #<limbs:eq:c_-1_is_zero>\
        w_i &in [L] #<limbs:eq:range_wi>
    $
    together enforce $w_i = overline(w)_i + c_(i-1) mod L$ as long as $C in [mu n L + alpha, frac(p,L, style:"horizontal"))$.
]<limbs:lm:limb-decomposition-constraint-correctness>

#proof[
    Combining @limbs:eq:def_ci and @limbs:eq:range_ci, we find that
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
    can be extracted from the proofs of @limbs:lm:carry-upperbound-pt1 and @limbs:lm:carry-upperbound-pt2, while the upper bound for $c_(i-1)$ follows from @limbs:cor:carry-upper-bound.
    Moreover, observe that $|X_i inter [L]| <= 1$ since $0 <= C L < p$.
    Constraint @limbs:eq:range_wi therefore enforces that $w_i = overline(w)_i + c_(i-1) mod L$ if $c_(i-1)$ (and therefore $w_(i-1)$) is correct, 
    and as a result, $c_i = floor.l frac((overline(w)_i + c_(i-1)), L, style: "horizontal") floor.r$ is correct.
    The proof now follows by induction, with @limbs:eq:c_-1_is_zero enforcing the base case.
]
