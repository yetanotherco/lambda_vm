#import "/book.typ": book-page, aside, todo

#show: book-page("ecsm.typ")

#let ecsm = raw("ECSM")
#let ecdas = raw("ECDAS")
#let ecsm2 = raw("ECSM2")
#let ecdas2 = raw("ECDAS2")
#let ect0 = raw("EC_T0")
#let lincomb = raw("LINCOMB2")

The elliptic-curve accelerator computes scalar multiples on the secp256k1
curve. Its purpose is Ethereum's `ecrecover`: given a signature, recover the
public key that produced it, which for a real block is the single most
expensive operation the guest performs.

*Two implementations ship side by side.* The _single-scalar path_ proves one
$k dot P$ per call: #ecsm owns the inputs and outputs of a scalar
multiplication and #ecdas proves one step of its double-and-add chain. The
_joint-chain path_ proves a whole two-term linear combination
$Q = u_1 P_1 + u_2 P_2$ in one call: #ecsm2 owns the call and #ecdas2 proves one
step of a joint Shamir--Straus chain. Both pairs are joined by the same
`Ecdas` bus, which telescopes a chain from a core chip's seed to its drain.

`ecrecover` uses the joint-chain path. The single-scalar path remains live and
is the general-purpose $x$-only accelerator; it is also the chip family the
joint chain was derived from, and every argument in this chapter that is not
explicitly marked as new is shared by both.

#todo[
  The machine-rendered variable and constraint tables that every other chip
  chapter carries (`render_chip_variable_table`, `render_constraint_table`) are
  missing here: there is no `src/ecsm.toml`, `ecdas.toml`, `ecsm2.toml` or
  `ecdas2.toml` yet, so the column and constraint listings below are prose.
  Several doc comments in the prover (`prover/src/tables/ecsm.rs`, `ecdas.rs`,
  `crypto/ecsm/src/*.rs`) already point at `spec/src/ecsm.toml` and `ecsm.typ`;
  those references were dead until this chapter, and the `.toml` half is still
  outstanding.
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

= Byte range checks are paired

All four chips obtain their byte bounds from the `AreBytes` lookup, and an
`AreBytes` send carries *two* elements, both of which the `BITWISE` receiver
matches against the precomputed table's `X` and `Y` columns. That table
enumerates the full byte-pair space, so a send $[a, b]$ matches a row if and
only if $a$ *and* $b$ are bytes.

Every byte range check originally shipped as its own send shaped $[b, 0]$,
which is the $y = 0$ special case of that same contract and wastes the second
slot. Adjacent bytes now share one send:

/ #ecdas: the 32-byte prefixes of `LAMBDA`, `XR`, `YR`, `Q0`, `Q1`, `Q2` pair
  internally as $(2i, 2i+1)$, giving 96 sends; of the four remaining odd bytes,
  two pairs are formed across blocks --- `(ROUND, Q0[32])` and
  `(Q1[32], Q2[32])`. 196 sends become 98, and interactions per row fall from
  388 to *290*.
/ #ecsm: `X2`, `Q0`, `YG` and `Q1`'s 32-byte prefix pair the same way, giving
  64 sends; the odd 33rd quotient byte rides alone as `[q1[32], 0]`. 129 sends
  become 65, and interactions per row fall from 579 to *515*.

#ecdas2 and #ecsm2 use the identical layout from the start.

*This is a bus repacking and nothing else.* No witness value, column count
(667 for #ecsm, 521 for #ecdas), constraint count (413 and 200) or maximum
degree changes; only `bus_interactions()` and the trace builder's mirrored
multiplicity collectors move, and they move together.

The saving is in the `LogUp` auxiliary trace, which costs one cubic-extension
column per two interactions --- $1.5$ committed base cells per interaction. For
#ecdas that is 194 auxiliary extension columns down to 145, i.e. 147 fewer
committed base cells against a row of 1,103: *$-13.3%$ of the table's committed
footprint*, on the table that carries the volume.

Soundness is preserved verbatim, with no lemma re-run. Every lemma of the
machine-checked gate consumes the byte contract *per column*, never per send,
and the set of columns covered is identical before and after --- each
previously-checked byte appears in exactly one paired send. Pairing bytes from
different logical operands is equally sound: an `AreBytes` row asserts no
relation between its two elements, since every combination exists in the table,
so pairing unrelated bytes adds no coupling. The argument is recorded in
`thoughts/ec-recover-opt/gate/pairing-equivalence.md`.

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

The same machinery, with the same carry offsets, is reused unchanged by
#ecdas, #ecsm2 and #ecdas2; it is described once here.

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

This paragraph is the one that does *not* survive the joint chain, whose
scalars, points and message are all attacker-supplied. #ecdas2 closes the same
edge a different way --- with a witnessed inverse, and equally unconditionally.

== Verification status

The constraint systems of #ecsm and #ecdas have been machine-checked with an
SMT solver: the limb/carry lifting, the mod-$p$ step lemmas and their side
conditions, the chain argument, and an end-to-end pin against an independent
reference are recorded as lemmas L1--L8 in `thoughts/ec-recover-opt/gate/`, with
the transcription anchored by evaluating the model on real prover witnesses
(872k checks).
Two checks were confirmed load-bearing by exhibiting the forgery that appears
when they are removed ($c_63 = 0$ and the $x_R < p$ canonicalisation); four
others are individually redundant but retained. The results rest on stated
contracts for the range-check tables, the `LogUp` multiset argument, `ECALL`
binding and timestamp uniqueness, plus the primality of $p$ and $N$.

= The joint-chain chips

#ecsm2 and #ecdas2 evaluate $Q = u_1 P_1 + u_2 P_2$ over one joint chain and
return *both* coordinates. They are what `ecrecover` calls.

== Why the joint chain exists

`ecrecover` evaluates $Q = u_1 G + u_2 R$. With an $x$-only accelerator the
guest cannot get that in one piece: the old path issued *four* scalar
multiplications --- $x(u_1 G)$, $x((u_1 + 1)G)$, $x(u_2 R)$, $x((u_2 + 1)R)$
--- and then reconstructed the two missing $y$ coordinates in software, because
the $+1$ queries are the only way to recover a $y$ from an $x$-only oracle. One
joint chain replaces all four chains and the reconstruction.

== Interface

The call takes system call number $-12$, the slot directly below #ecsm's, so
that the elliptic-curve accelerators stay contiguous.

/ `A0` #h(0.6em) `= x10`: address to which $x_Q parallel y_Q$ (64 bytes) is written; on return, a status word
/ `A1` #h(0.6em) `= x11`: address of $x_(P_1) parallel y_(P_1)$
/ `A2` #h(0.6em) `= x12`: address of $x_(P_2) parallel y_(P_2)$
/ `A3` #h(0.6em) `= x13`: address of $u_1 parallel u_2$

*$P_1$ is pinned to the generator.* The chip has no membership sub-witness for
an arbitrary first point, so instead of proving $P_1$ on-curve it asserts that
memory at `A1` holds exactly $G$, by giving the eight doubleword reads
*constant* values. That costs zero columns and zero constraints and makes
$P_1$'s curve membership a compile-time fact. The executor agrees by
construction: any other bytes at `A1` return status $7$, which is the software
fallback rather than an unprovable block. Generalising later means adding a
membership block and its witness, not reworking the ABI.

The status word is returned *in a register* rather than as a byte of the result
buffer, so the guest can branch before it reads memory at all and the result
region stays a clean aligned 64 bytes. Status $0$ means success; each rejection
class --- a zero scalar, an out-of-range scalar, an off-curve point, a
non-canonical point, $P_1 = plus.minus P_2$, $Q$ at infinity, and
$P_1 != G$ --- has its own nonzero value, so a debugging or benchmarking path
can tell them apart while the guest only tests against zero.

== The status contract

Unlike #ecsm, this call *does not trap* on degenerate input, and that choice is
about availability: a trap would let one crafted transaction make an entire
block unprovable. A nonzero status makes the guest run
`ProjectivePoint::lincomb` in software, which is ordinary proven CPU execution,
so a status that lies in that direction can only waste cycles.

The converse direction has to be *enforced*, or a prover simply declines to
prove anything, writes status $0$, and the guest reads a fabricated $Q$ out of
memory. #ecsm2 therefore carries two flags rather than one:

/ `MU`: a real lincomb2 ecall happened at this timestamp. Gates the `Ecall`
  receive and the combined `x10` read+write that binds the status.
/ `OK`: the status is $0$, i.e. the chain is proven. Gates *everything else* ---
  every operand read, the result write, every range check, every relation, and
  every chain, addend and digit bus.

both `IS_BIT`, with the three constraints

$ &"OK" dot (1 - "MU") = 0 \
  &"OK" dot "STATUS" = 0 \
  &"MU" dot ("STATUS" dot "S"_"INV" - (1 - "OK")) = 0. $

The first says a proven chain implies a real ecall; the second says claiming
the chain forces the status to zero; the third says a real ecall that does not
claim the chain must carry an *invertible*, i.e. nonzero, status. The witnessed
inverse is what keeps the per-variant error codes distinguishable --- a boolean
status would work too, and would lose them.

An error row is therefore not the same thing as a padding row. It sets
$"OK" = 0$ with every math column zero, so all convolution and carry relations
close at zero carries by exactly the argument padding rows already use, but it
keeps $"MU" = 1$ and carries the real result address and status that the `x10`
access binds. The split is not stylistic. The CPU sends on `Ecall` for every
ecall, so an unmatched syscall unbalances that bus and the all-zero padding
trick is unavailable; and the $P_1 != G$ status must stay provable, which it
would not be if the `A1` reads --- which assert that memory there holds $G$ ---
were gated by `MU` instead of `OK`.

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

Two rows break the otherwise uniform telescoping, and they are the places the
implementation had to work hardest: the *precompute* row sits off the
accumulator line entirely (its left operand is $P_1$, not the previous row's
result), and the *correction* row consumes the last accumulator against a
constant addend. Every other row's left operand is its predecessor's result.

Neither break can be distinguished by the round index --- both special rows are
emitted at $"round" = 0$, and the main loop's last iteration also produces
genuine round-$0$ rows. The chain is instead split into three separately keyed
*phases*, carried as $"PHASE" = "PH"_1 + 2 "PH"_2$ with $"PH"_1 dot "PH"_2 = 0$
inside the `Ecdas` tuple:

/ phase 0 --- precompute: exactly one row, seeded with $a = P_1 = G$ and addend
  $P_2$, drained into #ecsm2's $P_(12)$ columns.
/ phase 1 --- main chain: seeded with $a = T_0$ at round $"len" - 1$; $"len"$
  doublings and their adds.
/ phase 2 --- correction: exactly one row, seeded with the last accumulator and
  addend $-2^"len" dot T_0$, drained into #ecsm2's $Q$ columns.

#ecsm2 pins every segment at *both* ends at multiplicity `OK`, so a row can
execute in a phase only if #ecsm2 published that phase, and exactly one phase-0
and one phase-2 row can exist per proven call.

The phase-1 to phase-2 hand-off deliberately goes *through* #ecsm2 --- the
phase-1 drain is received into accumulator columns and re-sent as the phase-2
seed --- rather than along the chain. A direct hand-off is not expressible: a
row's outgoing tuple pins its successor's $op$ to its own $"NB"$ flag, and the
last main-chain row has $"NB" = 0$ while the correction row is an addition.

*Round bookkeeping.* A doubling and its optional add share a round, so the
successor round is $"round" - 1 + "NB"$ and the successor $op$ is $"NB"$ ---
exactly the mechanism the single-scalar chain uses, under the joint name "an
add follows me at this round". On a doubling $"NB"$ is the OR of that round's
two digits; the defining constraint is $op$-gated, because an add row carries
the same digits (it needs them to select its addend) but always has $"NB" = 0$.

*Doubling rows carry no addend.* On $op = 0$ no relation reads the addend: the
slope's $op = 0$ branch mentions neither coordinate, $x_R$'s $-x_G$ term
cancels exactly against $(1 - op)(x_G - x_A)$, $y_R$ uses neither, and the
non-degeneracy relation below sits entirely inside its own gate. So doubling
rows carry the addend $(0, 0)$ and stay silent on the addend bus --- but only
because $op = S_1 + S_2 + S_3 + S_"CORR"$ forces every selector to zero there.
Without that one degree-1 constraint the cancellation is still real and the
*gating* is forgeable: a prover would set a selector on a doubling and mint a
spurious addend receive.

== Shape and cost

#ecsm2 contributes one row of 1,155 columns and 817 interactions per ecall;
#ecdas2 one row of 658 columns, 288 constraints and 388 interactions per joint
step. At $1.5$ committed base cells per interaction that is about 2,380 cells
for the #ecsm2 row and *1,240* cells per #ecdas2 row.

The chain is $449.1$ rows per `ecrecover` on average. *Capacity must be
budgeted at 514*, which is $1 + 256 + 256 + 1$: the worst case over the valid
input domain is $(u_1, u_2) = (2^255, 2^255 - 1)$, both in $[1, N)$ with
*complementary* bit patterns, so every one of the 256 rounds carries a nonzero
joint digit and therefore an addition. It is complementarity that maximises,
not popcount --- $(N-1, N-1)$ shares every addition and reaches only 449. A
submitter can construct the worst case deliberately and cheaply, so no bound
may be read off a random sample; the mean governs the cost model and nothing
else.

Against the live post-pairing baseline of $1.467"M"$ committed base cells per
`ecrecover` (four chains of 382 #ecdas rows plus four #ecsm rows), the joint
chain costs $0.559"M"$ at the mean and $0.640"M"$ at the 514-row worst case:

#figure(table(
  columns: 4,
  align: (left, right, right, right),
  table.header([], [rows], [cells per `ecrecover`], [vs. baseline]),
  [baseline (4 $times$ #ecsm)], [4 $times$ 382], [1.467M], [---],
  [joint chain, mean], [449.1], [0.559M], [$-61.9%$],
  [joint chain, worst case], [514], [0.640M], [$-56.4%$],
))

The design document's headline of $-74.3%$ should *not* be quoted: it is
denominated against the pre-pairing baseline of $1.69"M"$, so it re-banks the
paired-range-check win described above on top of this one. The
non-degeneracy relation, which is what makes the chain unconditionally sound,
also costs roughly 129 columns and 96 interactions per #ecdas2 row --- so part
of the gap between the two figures was spent buying soundness, and that is
worth stating alongside the number.

The secondary win is guest cycles: $approx 78.5"k"$ fewer per `ecrecover`,
measured on two ethrex blocks that agree to $0.4%$. The design document
predicted 100--150k, so the measurement is below the low end of the estimate.

== The blinding point $T_0$

$T_0$ is a nothing-up-my-sleeve point: it must be *verifiably not chosen*. It
is derived by SHA-256 try-and-increment from a fixed tag:

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

The even root is taken for determinism; either root would serve.

*What the blind buys is one simplification, and nothing else.* Because the
accumulator starts at $T_0$ and the correction is keyed by $"len"$, any
$"len"$ at least as large as the true one yields the same $Q$: the extra
leading doublings only double $T_0$, and the keyed correction absorbs them. So
$"len"$ never has to be pinned to the exact MSB, and the sub-lemma that would
have done so is dropped outright. The blind was *also* intended to close the
incomplete-addition edge; it does not, and "Security assumptions" below says
why. It is a convenience, not a defence.

== The #ect0 table

The correction row subtracts $2^"len" dot T_0$, supplied from a preprocessed
table of 66 columns and exactly 256 rows, every one of them real. Row $j$ holds
the key $"len" - 1 = j$ and the point for $"len" = j + 1$.

The table stores the *negation* $-2^"len" dot T_0$ directly. That is not a
preference: it is what the witness generator emits as the correction row's
addend, so the lookup wires straight into #ecdas2's addend columns with no
in-circuit modular negation. Only $y$ differs from the positive blind, since
$x(-P) = x(P)$, which makes mixing the two conventions a silent sign flip that
still type-checks.

Keying by $"len" - 1$ rather than $"len"$ makes the bound $"len" <= 256$
*structural*. The consumer sends a plain $"len"$ and the table's receive key
re-adds the $1$, so the published key range is exactly $[1, 256]$ with one row
per value and nothing else in the table --- a send outside it matches no row,
`LogUp` cannot balance, and the proof is rejected. No consumer-side check is
needed, and none should be added. An earlier design keyed by $"len"$ directly
over 257 rows padded to 512, and *did* need one: a send at $"len" > 256$ would
have resolved to a zeroed padding row holding the off-curve point $(0, 0)$,
which the correction row would then have added.

The contents are a deterministic doubling chain off the pinned $T_0$, so the
table is constant in every sense that matters and needs no separate trust.

== Buses

Beyond `Ecall`, `Memw`, `AreBytes`, `IsHalfword` and `Zero`, the joint chain
uses:

/ `Addend` (29): $["ts"_"lo", "ts"_"hi", "sel", x(32), y(32)]$. #ecsm2 publishes
  the three point addends at witnessed counts and the correction constant at
  multiplicity `OK`; #ecdas2 receives once per addend-consuming row at
  multiplicity $S_1 + S_2 + S_3 + S_"CORR"$, with
  $"sel" = S_1 + 2 S_2 + 3 S_3 + 4 S_"CORR" in {1, 2, 3, 4}$. The multiplicity
  is linear in four terms rather than the three-term form, because the
  correction row is the fourth. $"sel"$ is *never* $0$: a bus element that is
  zero on a row contributes nothing to the fingerprint, so a $"sel" = 0$ addend
  would alias a shorter tuple.
/ `EcT0` (32): the #ect0 lookup described above.
/ `JointBit` (33): $["ts"_"lo", "ts"_"hi", "round", "stream"]$ with
  $"stream" in {1, 2}$, one stream per scalar. #ecdas2 sends at multiplicity
  $D_1$ or $D_2$; #ecsm2 receives at $2 dot "bit"$.

The factor of $2$ on the `JointBit` receive is load-bearing, not bookkeeping. A
set digit is carried by *both* the round's doubling and its add, and both send,
so a $2 times$ receive is what forces the add to exist at all --- with only the
doubling available the total can never reach two. At $1 times$ there is a
concrete counterexample: at a round where both digits are set, a prover splits
them across the two rows and the add then selects $P_2$ where the schedule
calls for $P_(12)$.

The `Ecdas` bus (28) is *shared* with the single-scalar chain. What separates
them is the first tuple element, a constant $0$ for the old chain and $1$ for
the joint one, so the difference of the two fingerprint polynomials has a
nonzero coefficient at a fixed power of the challenge. Differing arity would
*not* have sufficed: the fingerprint's positional weights advance
unconditionally and trailing zeros never re-align a tuple, but zero-padding a
tuple to a common width is a designed-in feature, so arity alone never
separates two chips on one bus.

== The non-degeneracy relation

#ecdas2 carries a *fourth* convolution relation beside the three of #ecdas:

$ D_"INV" dot (x_B - x_A) equiv 1 mod p. $

*This is the check that makes the joint chain sound, and it rests on no
computational assumption.* When $x_B = x_A$ the slope relation degenerates:
with $y_B = y_A$ it reads $0 = 0$ for *every* $lambda$, and $x_R$ and $y_R$
then produce a point of the prover's choosing which the rest of the chain
accepts --- the addend balance, the digit counting and the phase pinning are
all still satisfied. The row proves nothing. A witnessed inverse of
$x_B - x_A$ exists exactly when $x_B equiv.not x_A mod p$, so imposing it costs
no completeness and closes the case outright.

It is gated by $S_1 + S_2 + S_3 + S_"CORR"$ --- the same sum that receives the
addend --- so it covers every row that consumes one, including the precompute
and correction rows (both chord adds), and never covers doublings. Gating by
that sum rather than by $op$ is deliberate even though a constraint makes the
two equal: it ties the check to the very expression that counts the addend
receive, so it cannot drift away from the rows that consume one.

A gated-off row is not a hole. With the gate at zero only the shifted-quotient
term survives, and the limb-lifting argument turns that into
$p dot (mu dot R - q_3) = 0$ --- so the quotient is *pinned* to $3p$ on a live
doubling and to $0$ on a padding row, not left free. Gating at all is a cost
choice rather than a correctness one: $x = 0$ is not on secp256k1, so
$x_B - x_A = -x_A$ would be invertible on doublings too, but there is no addend
there and the cells would be wasted.

One more degeneracy is closed by its own constraint: the phase-0 row must add
$P_2$. Without that, a prover points the precompute at $P_1$, making the chord
$P_1 + P_1$, whose slope relation degenerates the same way and admits an
arbitrary $P_(12)$.

== Padding rows must be inert

"The columns are zero as generated" is *not* an argument --- a malicious prover
fills padding rows freely. Every interaction has to be inert by *constraint*,
and the question to ask of each one is not "is it gated?" but "which column
supplies its multiplicity, and what forces that column to zero?".

Most interactions in both chips take their multiplicity from `MU` or `OK` and
are inert for free. Two families do not: #ecdas2's digit sends count the raw
digit columns, and its addend receive counts the raw selector sum. The family

$ (1 - "MU") dot {D_1, D_2, S_1, S_2, S_3, S_"CORR"} = 0 $

is what closes them, and both were live forgeries before it was added:

+ A row with $"MU" = 0$, $"PH"_1 = 1$, $"NB" = 1$, $D_1 = 1$ at any round $r$
  satisfies every other constraint and emits a real `JointBit` digit. *Two* such
  rows supply the $2 times$ receive that an honest round pays with its doubling
  and its add, so the prover can drop the round-$r$ addition entirely --- with
  both digits zero on the real doubling, nothing demands one. The chain then
  computes $(u_1 - 2^r) P_1 + u_2 P_2$, and back-solving the signature for a
  chosen target needs one modular inversion and no discrete logarithm. The
  result is an *arbitrary chosen recovered public key*.
+ A row with $"MU" = 0$, $op = 1$, $S_2 = 1$ keeps $op = sum S$ satisfied and
  mints a spurious addend receive.

$"NB"$ needs no companion: with every selector zero, $op = sum S$ forces
$op = 0$, and the round-bookkeeping constraint then reads $"NB" = D_1 or D_2 = 0$.

#ecsm2 carries the same discipline for its own two ungated families --- the
scalar bit columns, which are the `JointBit` receive multiplicities, and the
addend publish counts --- with $(sum "bit") dot (1 - "OK") = 0$ and
$N_j dot (1 - "OK") = 0$.

== Canonicalisation obligations

The joint chain handles a point whose *sign* matters, which #ecsm never did:
an $x$-only chip is free to lift to whichever $y$ it likes, because
$x(k P) = x(k(-P))$, and that symmetry is exactly what a linear combination
gives up. The two ends of the call are not symmetric, and it is worth being
precise about which is which.

*The inputs are already bound.* The guest decompresses $R$ from the signature's
$(r, v)$ in software and writes both coordinates to memory; that is proven CPU
execution, and `MEMW` binds what the chip reads to what the guest wrote. The
guest is therefore the parity authority, and the prover has no freedom to
substitute $-P_2$ for $P_2$: doing so would require the guest's own proven
execution to have written different bytes.

In particular a $y_(P_2) < p$ check does *not* separate a point from its
negation, and must not be described as doing so. Negation is
$(x, y) |-> (x, p - y)$, and both $y$ and $p - y$ are already below $p$; the
non-canonical encoding $y + p$ (which fits in 32 bytes only for the vanishingly
rare $y < 2^32 + 977$) is congruent to $y$, i.e. the *same* point, and changes
no result. The witness carries a $y_(P_2) < p$ column, and it is worth keeping
so that the chip's soundness argument stands on its own constraints rather than
on the correctness of guest code --- the same reason #ecsm re-checks ranges the
executor has already enforced --- but it is *defence in depth, not a forgery
closed*. Removing it is the one negative control in the suite that comes back
with no forgery, and that is the expected result.

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

== Verification status

The joint-chain chips have their own machine-checked board, recorded in
`thoughts/ec-recover-opt/gate/RESULTS-lincomb2.md`. *Every lemma is discharged
and no lemma is open.*

The three relation arms #ecdas2 shares with #ecdas were shown textually
identical per arm, modulo the rename of the fixed generator to the per-row
addend. Every lemma quantified over those three relations --- the limb lifting,
the value lemmas, the step lemmas and the no-$y equiv 0$ side condition, L1
through L5a --- therefore transfers verbatim rather than being re-proved. What
was genuinely redone:

- *The width audit*, because the addend now varies per row and one of its
  values ($P_(12)$) is an interior chip output that is byte-bounded but never
  proven $< p$. The existing carry windows still bound it with about $2^39$ of
  headroom. The argument never depended on the addend being *canonical*, only
  on its limbs being *bytes* --- which is inherited through the keyed addend
  tuple, and which would silently break if that tuple were ever repacked to
  carry more than one byte per element.
- *The step lemma's side condition*, now discharged unconditionally by the
  non-degeneracy relation rather than by any assumption.
- *The counting argument*, over two interleaved digit streams, three phases and
  a per-row addend selection.
- *The exact-MSB sub-lemma*, dropped: the blind makes any
  $"len" in ["msb" + 1, 256]$ yield the same $Q$, and $"len" <= 256$ is
  structural.

The battery of negative controls produced *seven constructive forgeries* and
zero live holes. Two of the seven were live holes in the chips when the gate
began --- the padding-row digit sends and the missing non-degeneracy relation
--- which is the strongest available evidence that the gate can see real bugs.
Two controls came back redundant and are recorded as such rather than papered
over.

Before any result was trusted, the transcribed model was evaluated on real
prover witnesses: 265 cases, *5,960 #ecdas2 rows and about 3.3 million
individual checks*, every constraint value zero, every carry inside its window,
every quotient inside 33 bytes, and the prover's own non-degeneracy columns
equal to an independent group-law derivation.

The standing residual risk is the same one the keccak chapter names: the model
is transcribed by hand from the chips, mitigated but not eliminated by that
anchor. The durable fix is to generate the SMT problem from the constraint IR.

= Security assumptions

== What is unconditional

*Everything.* Both chip families follow from their constraint systems alone,
given the range-check contracts, the `LogUp` multiset argument, `ECALL` binding
and timestamp uniqueness, and the primality of $p$ and $N$.

For the single-scalar chain the incomplete-addition edge is closed outright by
the $k < N$ check. For the joint chain it is closed outright by the witnessed
inverse $D_"INV" dot (x_B - x_A) equiv 1 mod p$. *The joint-chain path rests on
no cryptographic assumption that the single-scalar path did not.*

== The assumption that was proposed, and why it is not used

The joint chain was originally designed to close the incomplete-addition edge
by blinding: seed the accumulator at $T_0$, so that every intermediate
accumulator is $2^j T_0 + (c_1 P_1 + c_2 P_2)$ and a collision appears to
require a known linear relation on $log(T_0)$ --- a discrete logarithm nobody
has. That would have named an assumption:

#aside("Assumption T0-DL (blinding-point discrete log) --- NOT USED, DO NOT SIGN")[
  No efficient prover can produce a known linear relation on $log(T_0)$: no
  probabilistic polynomial-time algorithm, given the public description of
  secp256k1 and the point $T_0$, outputs integers $(alpha, beta)$ with
  $alpha equiv.not 0 mod N$ and $alpha T_0 = beta G$.

  Because $N$ is prime this is *equivalent* to computing $log_G (T_0)$, i.e. it
  is exactly the discrete-logarithm problem instantiated at one fixed,
  verifiably-unchosen point. As an assumption it is unobjectionable and
  introduces no new hardness class.

  *The problem is not the assumption. It is that the reduction to it does not
  close* --- so assuming it would buy nothing.
]

The reduction assumes the prover cannot relate $P_1$ and $P_2$ to $T_0$. It
can. For `ecrecover`, $P_2 = R = "lift"_x (r)$ and $r$ is a signature component
the submitter chooses freely, so the prover may set $P_2 = mu T_0$ for a $mu$
it picks, and the $T_0$ coefficient of the collision equation cancels against
$P_2$'s. Writing the accumulator entering the addition at round $j$ as
$"acc" = alpha T_0 + beta_1 G + beta_2 P_2$, with $alpha, beta_1, beta_2$
public functions of the schedule and the scalar bits, and taking
$u_1 < 2^j$ so that $beta_1 = 0$, the collision
$"acc" = "addend" = P_2$ reduces to a single scalar equation

$ alpha + mu (beta_2 - 1) equiv 0 mod N quad ==> quad mu = -alpha \/ (beta_2 - 1), $

one modular inversion. With $P_1 = G$ and $u_1 = 1$ it takes the concrete form
$mu (c_2 - 1) equiv -2^j mod N$. The construction costs one scalar
multiplication and no search at all --- *cheaper* than the corresponding attack
on the unblinded chain, which the blind was introduced to prevent. It was
verified 5 out of 5 over a range of schedule lengths, each instance packaged as
a well-formed $(z, v, r, s)$, corroborated by the Python reference, the Rust
witness and an independent Jacobian implementation.
#footnote([`thoughts/ec-recover-opt/oracle/nums_blinding_probe.py`; writeup in `thoughts/ec-recover-opt/lincomb2/FINDING-nums-blinding.log`.])

The witness generator rejects all of these with a nonzero status, so the
honest path never reaches them; the forgery lives entirely on the
malicious-prover side, where a row never passes through the witness generator
at all. *Only a constraint can catch it*, which is what the non-degeneracy
relation is.

== What the blinding is retained for

The blind survives for a reason that has nothing to do with soundness: it lets
any $"len"$ at least as large as the true MSB yield the correct $Q$, which
drops the exact-MSB sub-lemma from the counting argument and lets $"len" <= 256$
be structural in the #ect0 table. That is a convenience. It should not be
described as a defence, and no part of the soundness argument may appeal to it.
