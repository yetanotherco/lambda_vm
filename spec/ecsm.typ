#import "/book.typ": book-page, aside, todo

#show: book-page("ecsm.typ")

#let ecsm = raw("ECSM")
#let ecdas = raw("ECDAS")
#let lincomb = raw("LINCOMB2")

The elliptic-curve accelerator computes scalar multiples on the secp256k1
curve. Its purpose is Ethereum's `ecrecover`: given a signature, recover the
public key that produced it, which for a real block is the single most
expensive operation the guest performs.

Two chips exist today. #ecsm handles one `ECALL` and owns everything about the
_inputs and outputs_ of a scalar multiplication: range checks, curve
membership, and the scalar's bit decomposition. #ecdas proves one step of the
double-and-add chain and holds no opinion about what the chain means; the two
are joined by a bus that telescopes the chain from #ecsm's seed to #ecsm's
drain. A third mode, #lincomb, is _specified but not yet implemented_ and is
described in the #lincomb section below; it replaces four #ecsm calls per
`ecrecover` with
one, and it is the reason this chapter has a security-assumptions section
("Security assumptions").

#todo[
  The machine-rendered variable and constraint tables that every other chip
  chapter carries (`render_chip_variable_table`, `render_constraint_table`) are
  missing here: there is no `src/ecsm.toml` or `src/ecdas.toml` yet, so the
  column and constraint listings below are prose. Several doc comments in the
  prover (`prover/src/tables/ecsm.rs`, `ecdas.rs`, `crypto/ecsm/src/*.rs`)
  already point at `spec/src/ecsm.toml` and `ecsm.typ`; those references were
  dead until this chapter, and the `.toml` half is still outstanding.
]

= The curve

secp256k1 is $y^2 = x^3 + b$ over $FF_p$ with
#footnote([Standards for Efficient Cryptography 2 (SEC 2), version 2.0, section 2.4.1, Certicom Research.])

$ p = 2^256 - 2^32 - 977, quad b = 7, $

of prime order $N$, with generator $G$. Three properties of these constants are
load-bearing and are used without further comment below:

+ $p equiv 3 mod 4$, so square roots are $a^((p+1)\/4)$ and a coordinate lift
  needs no Tonelli--Shanks;
+ $N$ is prime, so every nonzero residue mod $N$ is invertible;
+ $N$ is odd and $-7$ is not a cube mod $p$, so the curve has no point of order
  two: there is no on-curve point with $y equiv 0$. Every doubling in the chain
  therefore has a well-defined tangent slope.

All values crossing the `ECALL` boundary are 32-byte *little-endian* integers.
Inside the chips every 256-bit quantity is carried as 32 range-checked `Byte`
variables, least significant first.

= #ecsm chip

#ecsm answers system call number $-11$ and contributes one row of 667 columns
per call. Like every accelerator it must place a receiver on the `Ecall` bus,
or that bus does not balance.

== Interface

/ `A0` #h(0.6em) `= x10`: address to which the result $x_R$ is written
/ `A1` #h(0.6em) `= x11`: address of the input $x$-coordinate $x_G$
/ `A2` #h(0.6em) `= x12`: address of the scalar $k$

The accelerator is *$x$-only*: it takes an $x$-coordinate, lifts it to the
point with *even* $y$, and returns only the $x$-coordinate of the product. This
is sound because $x(k dot P) = x(k dot (-P))$, so the choice of lift cannot
change the answer. Parity is therefore not this chip's business, and the guest
that needs a signed point recovers the sign itself.

The executor rejects the call outright --- it does not produce a row --- when
$k = 0$, $k >= N$, $x_G >= p$, or $x_G^3 + b$ is a non-residue. Execution fails
before anything is written, so invalid inputs are *unexecutable* rather than
wrong: no trace containing them exists. The chip nevertheless enforces its own
range checks rather than inheriting them from the executor, because soundness
must not depend on the prover having run the executor honestly.

== What the row proves

Writing $mu in {0, 1}$ for the row-live flag, one #ecsm row witnesses $y_G$ and
establishes:

+ *Curve membership.* $y_G^2 equiv x_G^3 + b mod p$, via two byte-limb
  convolution relations with quotients $q_0, q_1$: first $x_2 = x_G^2 - q_0 p$,
  then $y_G^2 + mu p^2 - x_G x_2 - mu b - q_1 p = 0$.
+ *Canonical input.* $x_G < p$, as a $2^256$-overflow witness in 16 halfwords.
+ *Scalar range.* $0 < k < N$, likewise.
+ *Canonical output.* $x_R < p$. This one is genuinely load-bearing rather than
  hygiene: without it a result $v < 2^32 + 977$ also admits the non-canonical
  representation $v + p$, and the guest would receive a different 32-byte
  string for the same field element.
+ *Bit decomposition.* $k$ is spread over 256 boolean columns, served to #ecdas
  on the `Bit` bus.

== Limb arithmetic

Every relation above is a polynomial identity between 256-bit values, proved
one byte-limb at a time. For a relation whose signed limb sequence is $S_i$,
the chip witnesses a carry sequence $c_i$ satisfying

$ 2^8 dot c_i = c_(i-1) + S_i, quad c_(-1) = 0, quad c_63 = 0. $

Telescoping gives $sum_i 2^(8i) S_i = 0$ over the integers, which is the
intended identity. Two details make it work over a 64-bit base field:

- *Carry offsets.* The $c_i$ are signed; each is range-checked as a `Half`
  after adding a per-relation constant, so the shifted value lands in
  $[0, 2^16)$. The constants differ per relation because the limb sums do.
- *The closing check $c_63 = 0$.* This is what forbids an unconstrained
  overflow at the top limb, and it is not redundant: dropping it admits a
  decoded value that is nonzero mod $p$.

Intermediate quantities are only byte-bounded, i.e. $< 2^256 approx 5.4 p$, not
reduced. That is deliberate --- the relations are mod-$p$ identities and the
quotient absorbs the slack --- and it is why the canonicalisation checks sit at
the boundaries rather than on every row.

= #ecdas chip

#ecdas contributes one row of 521 columns per double-or-add step. A row receives an
accumulator $A$, a fixed addend $G$, a round index and an operation flag from
the `Ecdas` bus; proves $R = 2A$ (when $op = 0$) or $R = A + G$ (when
$op = 1$); and sends $R$ onward with the next round's bookkeeping. Three
convolution relations of the shape described above do the work:

$ &"slope:" && op dot (lambda (x_G - x_A) + y_A - y_G) + (1 - op) dot (2 lambda y_A - 3 x_A^2) equiv 0 \
  &x_R":" && lambda^2 - x_A - x_G - x_R - (1 - op)(x_A - x_G) equiv 0 \
  &y_R":" && lambda (x_A - x_R) - y_A - y_R equiv 0 $

all mod $p$. The single $lambda$ column carries the tangent slope on a doubling
and the chord slope on an addition; the $op$ selector picks the branch.

== The chain

Soundness of a scalar multiplication is not a per-row property; it lives in two
bus arguments.

*Telescoping.* #ecsm sends a seed tuple and receives a drain tuple keyed by the
call's timestamp; each #ecdas row receives a state and sends its successor.
Exact multiset balance then forces the live rows to form disjoint paths from
seeds to drains. The round index never increases and cannot be $-1$ (it is
byte-checked), so no path can cycle, and distinct calls have distinct
timestamps, so no two calls can share a path. For $k = 1$ the seed is drained
immediately and the chip echoes its input.

*Bit counting.* #ecsm receives one `Bit` tuple per set bit of $k$; the senders
are the one #ecsm send at $"len"_k$ plus one send per #ecdas row that is about
to perform an addition. Balance forces $"len"_k$ to name a set bit, forces an
addition to occur at exactly the set bits below it, and forbids additions at
zero bits. Together with the seed's constants this pins the path's $("round", op)$
sequence to the reference MSB-first double-and-add schedule of $k$.

== The incomplete-addition edge

The chord formula is only valid for $A != plus.minus G$. If $A = G$ the slope
relation degenerates to $0 = 0$ and $lambda$ is *unconstrained*, which would let
a prover choose the result; if $A = -G$ the relation becomes $-2 y_G equiv 0$,
which no on-curve point satisfies, so that case merely rejects.

For the single-scalar chain the dangerous case is *unreachable*, and
unconditionally so. At an addition the accumulator is $c dot P$ for $c$ the
binary prefix of $k$ consumed so far, and reaching $A = plus.minus G$ needs
$c equiv plus.minus 1 mod N$. After the first doubling $c >= 2$, and $c <= k < N$
throughout, so $c equiv 1$ is impossible; $c equiv -1$ needs $c = N - 1 >= 2^255$,
which forces $k >= 2N - 2 > N$ and contradicts the range check on $k$. The
argument consumes the $k < N$ check and the prefix structure, and nothing else.

This paragraph is the one that does *not* survive the joint chain of
the #lincomb chip below. That is the subject of "Security assumptions".

== Verification status

The constraint systems of #ecsm and #ecdas have been machine-checked with an
SMT solver: the limb/carry lifting, the mod-$p$ step lemmas and their side
conditions, the chain argument, and an end-to-end pin against an independent
reference are recorded as lemmas L1--L8 in `thoughts/ec-recover-opt/gate/`, with
the transcription anchored by evaluating the model on real prover witnesses.
Two checks were confirmed load-bearing by exhibiting the forgery that appears
when they are removed ($c_63 = 0$ and the $x_R < p$ canonicalisation); four
others are individually redundant but retained. The results rest on stated
contracts for the range-check tables, the `LogUp` multiset argument, `ECALL`
binding and timestamp uniqueness, plus the primality of $p$ and $N$.

= #lincomb chip (specified, not implemented)

`ecrecover` evaluates $Q = u_1 G + u_2 R$. With an $x$-only accelerator the
guest must issue *four* scalar multiplications --- $x(u_1 G)$, $x((u_1 + 1)G)$,
$x(u_2 R)$, $x((u_2 + 1)R)$ --- because the $+1$ queries are the only way to
recover a $y$ from an $x$-only oracle. #lincomb computes the linear combination
directly and returns both coordinates, replacing roughly 1,530 #ecdas rows per
`ecrecover` with roughly 450.

== Interface

The call takes system call number $-12$, the slot directly below #ecsm's, so
that the elliptic-curve accelerators stay contiguous.

/ `A0` #h(0.6em) `= x10`: address to which $x_Q parallel y_Q$ (64 bytes) is written; on return, a status word
/ `A1` #h(0.6em) `= x11`: address of $x_(P_1) parallel y_(P_1)$
/ `A2` #h(0.6em) `= x12`: address of $x_(P_2) parallel y_(P_2)$
/ `A3` #h(0.6em) `= x13`: address of $u_1 parallel u_2$

Unlike #ecsm, this call *does not trap* on degenerate input. The status word is
returned *in a register* rather than as a byte of the result buffer, so the
guest can branch before it reads memory at all and the result region stays a
clean aligned 64 bytes. Status $0$ means success; each rejection class --- a
zero or out-of-range scalar, an off-curve or non-canonical point,
$P_1 = plus.minus P_2$, $Q$ at infinity, and $P_1 != G$ --- has its own nonzero
value, so a debugging or benchmarking path can tell them apart while the guest
only tests against zero. On a nonzero status nothing is written to the result
buffer and no row is emitted: there is no witness to prove.

A nonzero status is always *sound*, whoever produced it: the fallback is
ordinary guest code and is proven by the CPU tables, so a lying status can only
waste cycles, never change a result. The reason to prefer this over trapping is
availability: a trap would let one crafted transaction make an entire block
unprovable.

== Schedule

The chain is a joint (Shamir--Straus) double-and-add over both scalars at once,
MSB-first, with one doubling per round regardless of the digits:

#raw(
"  precompute :  P12 = P1 + P2                      (a standalone chord add)
  for round = len-1 .. 0:
      double  :  acc = 2*acc
      if the joint digit is nonzero:
          add :  acc = acc + addend,  addend in {P1, P2, P12}
  correction :  Q = acc + (-2^len * T0)",
  block: true,
)

with $"len" = max(op("msb")(u_1), op("msb")(u_2)) + 1$, and the accumulator
seeded at the blinding point $T_0$ below rather than at infinity.

Two rows break the otherwise uniform telescoping and are the places a chip
implementation is most likely to go quietly wrong: the *precompute* row sits
off the accumulator line entirely (its left operand is $P_1$, not the previous
row's result), and the *correction* row consumes the last accumulator against a
constant addend. Every other row's left operand is its predecessor's result.

On a doubling row the addend cancels out of all three relations --- the slope's
$op = 0$ branch mentions neither coordinate, $x_R$'s $-x_G$ term cancels against
$(1 - op)(x_G - x_A)$, and $y_R$ uses neither --- so doubling rows carry the
addend $(0, 0)$ and consume nothing from the addend bus.

== The blinding point $T_0$

$T_0$ is a nothing-up-my-sleeve point: it must be *verifiably not chosen*, since
its whole purpose is that nobody knows its discrete logarithm. It is derived by
SHA-256 try-and-increment from a fixed tag:

#raw(
"tag = \"lambdavm/ecsm/lincomb2/T0/v1\"          (28 ASCII bytes)
for counter = 0, 1, 2, ...:
    x = int_be( SHA-256( tag || counter_be32 ) )
    if x < p and x^3 + 7 is a square mod p:
        y = the EVEN square root of x^3 + 7 mod p
        return (x, y)",
  block: true,
)

Counter $0$ yields no valid $x$; counter $1$ succeeds:

$ &x(T_0) = "0xaf319aa90f91a86b297de85edb330a665efba79aa98893db1b49070cb1ae7864" \
  &y(T_0) = "0x1481a038143c0732071db0bcf3b05b8ca2e624fa217d82193f3c254a606277a0" $

The even root is taken for determinism; either root would serve, since only
ignorance of the discrete logarithm matters.

The correction row subtracts $2^"len" dot T_0$, supplied from a preprocessed
table that stores the *negation* $-2^i dot T_0$ directly --- the correction row
adds its addend, so storing the negation removes a per-row negation from the
chip and the table is constant either way. The table is keyed by
$"len" - 1 in [0, 255]$ and has exactly 256 real rows with no padding, which
makes the bound $"len" <= 256$ *structural*: there is no row for a larger index,
so no consumer has to enforce it. Its contents are a deterministic function of
$T_0$ and need no separate trust.

The blind buys one unambiguous simplification: $"len"$ no longer has to be
pinned to the exact MSB. Any $"len"$ at least as large as the true one yields
the same $Q$, because the extra leading doublings only double $T_0$ and the
keyed correction absorbs them. It was also intended to buy soundness for the
incomplete-addition edge; see "Security assumptions" for why it does not.

== Canonicalisation obligations

#lincomb handles a point whose *sign* matters, which #ecsm never did: an
$x$-only chip is free to lift to whichever $y$ it likes, because
$x(k P) = x(k(-P))$, and that symmetry is exactly what a linear combination
gives up. It is worth being precise about what this does and does not require,
because the two ends of the call are not symmetric.

*The inputs are already bound.* The guest decompresses $R$ from the signature's
$(r, v)$ in software and writes both coordinates to memory; that is proven CPU
execution, and `MEMW` binds what the chip reads to what the guest wrote. The
guest is therefore the parity authority, and the prover has no freedom to
substitute $-P_2$ for $P_2$: doing so would require the guest's own proven
execution to have written different bytes.

In particular a $y_(P_2) < p$ check does *not* separate a point from its
negation, and should not be described as doing so. Negation is
$(x, y) |-> (x, p - y)$, and both $y$ and $p - y$ are already below $p$; the
non-canonical encoding $y + p$ (which fits in 32 bytes only for the vanishingly
rare $y < 2^32 + 977$) is congruent to $y$, i.e. the *same* point, and changes
no result. The witness carries a $y_(P_2) < p$ column, and it is worth keeping
so that the chip's soundness argument stands on its own constraints rather than
on the correctness of guest code --- the same reason #ecsm re-checks ranges the
executor has already enforced --- but it is defence in depth, not a forgery
closed.

*The output is not bound by anything else,* and here the checks are genuinely
load-bearing:

/ $x_Q < p$ #h(0.3em) and #h(0.3em) $y_Q < p$: the chip *writes* these bytes, so
  the prover chooses them, and `MEMW` then faithfully binds memory to whatever
  was written. The relations pin $Q$ only modulo $p$, so a coordinate $v$ below
  $2^32 + 977$ also admits the encoding $v + p$: the same field element, a
  different 32-byte string. The guest hashes those bytes to form an address, so
  the two encodings recover *different addresses*. This is the same finding as
  $x_R < p$ for #ecsm, where removing the check was shown to produce a concrete
  forgery.

= Security assumptions

== What is unconditional

Everything the #ecsm chip proves, and the per-step content of #ecdas, follows from the
constraint system alone, given the range-check contracts and the primality of
$p$ and $N$. For the single-scalar chain that includes the incomplete-addition
edge, which is closed outright by the $k < N$ check.

== The assumption #lincomb introduces

The argument that closes the incomplete-addition edge for #ecsm does not
transfer to the joint chain, and no analogue of it exists. In the joint chain
both scalars, both points, and the message are attacker-supplied: the values
$(z, v, r, s)$ of an `ecrecover` are free bytes, $z$ is not forced through any
hash, and a prover may pick $rho$ and set $r = x(rho G)$, thereby *knowing*
$log_G R$. It can then solve for a prefix of $u_1$ that drives the accumulator
onto its addend at a chosen step, retry cheaply over $rho$ and $u_2$, and land a
row whose $lambda$ is unconstrained --- and therefore a forged $Q$, a forged
recovered address, and an apparently valid transaction from a sender who never
signed it.

Blinding the accumulator with $T_0$ is intended to answer this. Every
intermediate accumulator becomes $2^j T_0 + (c_1 P_1 + c_2 P_2)$, so a collision
appears to require a known linear relation involving $log(T_0)$ --- a discrete
logarithm nobody has. Stated as a named assumption:

#aside("Assumption T0-DL (blinding-point discrete log) --- AWAITING SIGN-OFF")[
  No efficient prover can produce a known linear relation on $log(T_0)$.

  Formally: no probabilistic polynomial-time algorithm, given the public
  description of secp256k1 and the point $T_0$ derived above, outputs integers
  $(alpha, beta)$ with $alpha equiv.not 0 mod N$ and $alpha T_0 = beta G$.

  Because $N$ is prime, $alpha$ is invertible and this is *equivalent* to
  computing $log_G (T_0)$ outright. The assumption is therefore exactly the
  discrete-logarithm problem on secp256k1, instantiated at one fixed,
  verifiably-unchosen point --- strictly within what ECDSA and `ecrecover`
  already assume about the very chain being proved. It introduces no new
  hardness class.

  Its cost is nonetheless real and should not be glossed: it converts one lemma
  of the elliptic-curve soundness argument from *unconditional* to
  *computational*. #ecsm today rests on no cryptographic assumption at all.
]

== Open: the reduction to T0-DL is incomplete

#todo[
  *This section blocks sign-off.* The assumption above is *necessary but not
  sufficient*: the reduction from the incomplete-addition edge to T0-DL does not
  hold as stated, and there is a constructive counterexample. Do not read the
  blinded design as closing the edge.
]

The reduction assumes the prover cannot relate $P_1$ and $P_2$ to $T_0$. It can.
For `ecrecover`, $P_2 = R = "lift"_x (r)$ and $r$ is a signature component the
submitter chooses freely --- the same freedom the attack above already uses. So
the prover may set $P_2 = mu T_0$ for a $mu$ it picks, and then the $T_0$
coefficient of the collision equation cancels against $P_2$'s. Writing the
accumulator entering the addition at round $r r$ as
$"acc" = alpha T_0 + beta_1 G + beta_2 P_2$ with $alpha, beta_1, beta_2$ public
functions of the schedule and the scalar bits, and taking $u_1 < 2^(r r)$ so
that $beta_1 = e_1 = 0$, the collision $"acc" = "addend" = P_2$ reduces to a
single scalar equation

$ alpha + mu (beta_2 - 1) equiv 0 mod N quad ==> quad mu = -alpha \/ (beta_2 - 1), $

one modular inversion. The construction costs one scalar multiplication and no
search at all --- cheaper than the attack on the *unblinded* chain that the
blind was introduced to prevent. Concrete instances, together with the
`ecrecover` inputs $(z, v, r, s)$ that reach them and a check that the guest's
own decomposition reproduces them, are in
`thoughts/ec-recover-opt/lincomb2/FINDING-nums-blinding.log`.

Consequently #lincomb needs an explicit non-degeneracy obligation on addition
rows --- a witnessed inverse $d^(-1) (x_B - x_A) equiv 1 mod p$, or a
detect-and-branch row variant --- which closes the edge *unconditionally* and
without appeal to T0-DL. Which of the two, and what it costs, is not settled
here.

The blind remains worth keeping for the $"len"$ simplification noted above,
but that is a convenience, not a soundness property, and does
not by itself require T0-DL.
