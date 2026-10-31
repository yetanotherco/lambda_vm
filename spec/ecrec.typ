#import "/book.typ": book-page, todo

#show: book-page("ecrec.typ")

#let ecrec = raw("ECREC")
#let ecdas = raw("ECDAS")

The ECDSA recovery accelerator returns the secp256k1 public key of a signature
in one call: given $(r, s)$, the message hash $z$ and the parity of $R$'s $y$, the
key $Q = r^(-1) (s R - z G) = u_1 G + u_2 R$ with $u_1 = -z r^(-1)$ and
$u_2 = s r^(-1) mod N$. It reuses the elliptic-curve machinery of the `ECSM`
call: the two scalar multiplications are two double-and-add chains of #ecdas
rows, and the final addition is one more #ecdas row. The #ecrec chip proves the
scalar arithmetic modulo $N$, the decompression of $R$, and binds the three
#ecdas conversations to the call.

#todo[
  No `src/ecrec.toml` yet: the listings below are prose, and the code
  (`prover/src/tables/ecrec.rs`) is the reference.
]

= The call

ECALL number $-12$ (`u64::MAX - 11`).

/ `x10`: address of the output, 72 bytes: $Q_x$ and $Q_y$ (32 bytes each,
  big-endian), then a status doubleword (little-endian).
/ `x11`: address of the signature $r || s$, 64 bytes, each big-endian.
/ `x12`: address of the hash $z$, 32 bytes, big-endian.
/ `x13`: the parity of $R$'s $y$ (0 or 1), as a value.

Status 0: the output holds the key. Status 1: the call *bailed*; the key bytes
are zero and the caller recovers the key another way. A bail proves nothing and
needs nothing proven: the caller's own path decides. The executor bails exactly
when $r$ or $s$ is not in $[1, N)$, when $r$ is not the $x$ of a curve point,
when $u_1 = 0$ ($z equiv 0 mod N$), or when $x(u_1 G) = x(u_2 R)$ (the chord is
undefined; this covers $Q$ at infinity). It stops with an error when `x13` is
above 1, when the signature and hash regions overlap, or when an address's low
32-bit limb plus its operand's last offset (71, 63, 31) reaches $2^32$.

= #ecrec

One row per call, all-zero padding rows ($mu = 0$). $"OK"$ is a bit with
$"OK" <= mu$; $mu - "OK"$ is the status byte written. Every relation below is
multiplied by $"OK"$, and every range check is sent with multiplicity $"OK"$.

== Memory

$r$, $s$ (eight doubleword reads) and the `x11` register at $T$; `x12`, `x13`
(whose value must be $("PAR", 0)$) and $z$ (four reads) at $T + 1$; `x10` and
the nine output doubleword writes at $T + 2$. A big-endian memory byte $m$ of a
value is its little-endian limb $31 - m$, a column permutation.

== Relations, under OK

Byte-limb convolutions with 64 carries and a closing carry of zero, as `ECSM`'s:

- $r w - 1 - q_w N = 0$: $w = r^(-1)$;
- $z w + u_1 - q_z N = 0$: $u_1 = -z w$ ($z$ as read, unreduced);
- $s w - u_2 - q_u N = 0$;
- $x_2 = r^2 mod p$ and $y_R^2 + p^2 - r x_2 - 7 - q_1 p = 0$: $R = (r, y_R)$
  is on the curve;
- $y_R[0] = 2 H + "PAR"$ with $H$ a byte: the requested root.

$u_1$ and $u_2$ are 256 bit columns each (all zero unless $"OK"$). Bounds by
the overflow-carry chains: $r, s, u_1, u_2 < N$ and
$y_R, x_A, x_B, Q_x, Q_y < p$; $r, s, u_1, u_2 eq.not 0$ by `Zero` lookups.
$x_A eq.not x_B$: with $d_c$ the eight 32-bit word differences of
$x_B - x_A$ and witnesses $"DINV"_c$, $sum_c d_c "DINV"_c = "OK"$. On a bail,
$(mu - "OK") dot Q_i = 0$ for every key byte.

== The three conversations

- First chain $A = u_1 G$ at key $T$: seed $[0, T, G, G, ell_1 - 1, 0]$,
  drain $[0, T, A, G, -1, 0]$, `Bit` receives $[T, i]$ with multiplicity
  $u_(1,i)$, the top bit $[T, ell_1]$ self-sent.
- Second chain $B = u_2 R$ at key $T + 1$, the same shape on $R = (r, y_R)$.
- Final addition at key $T + 2$: send $[0, T + 2, A, B, 0, 1]$, receive
  $[0, T + 2, Q, B, -1, 0]$: one #ecdas addition row.

= Soundness

Under $"OK"$, $R$ is the curve point with $x = r$ and the requested parity;
$u_1, u_2$ are the reduced scalars of $(r, s, z)$; each chain is the
double-and-add of its scalar on its point, non-degenerate by `ECSM`'s argument
($0 < u < N$, prime order); the final addition is non-degenerate by
$x_A eq.not x_B$ on canonical $x_A, x_B$; the key is canonical. The keys
$T, T + 1, T + 2$ cannot meet another call's: every ECALL timestamp is a
multiple of 4, because the CPU's preprocessed timestamp column holds $4 (i + 1)$
and each CPU table adds its base $4 dot S_k$, four times the rows of the tables
before it, which the verifier computes from the heights (the offsets 1 and 2 are
checked against the stride 4 when the prover is built), every #ecdas row
inherits its key from a seed, and only `ECSM`, #ecdas and #ecrec use the `Ecdas`
and `Bit` buses. Without the offset of the second chain, the two chains' `Bit`
messages would be one multiset and a prover could swap their scalars.
