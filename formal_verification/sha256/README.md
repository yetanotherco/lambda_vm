# Formal verification of the SHA-256 chips — z3/QF-BV gate

Machine-checks that `prover/src/tables/sha256_round.rs` computes one SHA-256
round and that `prover/src/tables/sha256_schedule.rs` computes the message
schedule recurrence, for **every** constraint-satisfying assignment of their
trace columns, *given* the contracts of the helper chips they call.

Follows the method in `../keccak/README.md`, which is the canonical template.
Read that first; this file records only what is specific to SHA-256.

```
python3 run_gate.py        # the whole board, ~4 s
```

## What is proven

Assert `chip_output ≠ reference(input)` and ask z3 for a counterexample. UNSAT
means no such assignment exists.

| | claim |
|---|---|
| lemma 1 | `sigma()` over the 32 bit columns is FIPS 180-4's rotate-xor, all four kinds |
| lemma 2 | the five `BYTE_ALU` results, summed as the AIR sums them, are `Ch` and `Maj` |
| lemma 3 | the two addition constraints force `out_a`/`out_e` to the low 32 bits, uniquely |
| lemma 4 | the `temp1`/`temp2` grouping is the spec's `T1`/`T2` |
| lemma 5 | the schedule row computes `w[i] = w[i-16] + σ0(w[i-15]) + w[i-7] + σ1(w[i-2])`, and a zero-multiplicity row cannot claim reads |
| lemma 6 | the core's feed-forward: `out_i` is uniquely `(h_i + last_i) mod 2^32` |
| lemma 7 | `emit_add_pair`'s range bounds **suffice mod p**, for both halves of a pointer |
| lemma 8 | `be(c)` is the big-endian reading of four memory bytes |

## Why this gate is five lemmas and keccak's is one query

**Keccak's round is purely bitwise** — XOR, AND, NOT, rotations, no carries — so
its whole transition bit-blasts into one tractable QF-BV query. **A SHA-256
round adds five 32-bit values.** A monolithic query over the bit columns, the
sigma polynomials and those carry chains did not terminate here in ten minutes;
the first attempt is deleted rather than left in the directory looking runnable.

The split is not a shortcut: it is the same assume-guarantee move the template
already makes for helper chips, applied inward. Lemma 1 discharges the
rotations, lemma 2 the bitwise functions, and lemmas 3–5 then treat those as
free 32-bit values and reason about pure arithmetic.

## Lemma 7 is not a bitvector query, on purpose

The core's pointer arithmetic goes through `emit_add_pair`
(`prover/src/constraints/templates.rs:334-374`), which recovers each carry as

```
carry = (lhs_limb + rhs_limb - sum_limb) * INV_SHIFT_32
```

where `INV_SHIFT_32` is the inverse of `2^32` **in the Goldilocks field**. QF-BV
cannot represent that: mod `2^n` the factor `2^32` is a zero divisor and has no
inverse, so any bitvector encoding of this construction is unsound or vacuous.

`lemma_core.py` models it over the integers with explicit congruences mod `p`,
and proves the thing QF-BV structurally cannot: that the range bounds are
**sufficient** — that no assignment with limbs in `0..2^32` and carries in
`{0,1}` can satisfy the field equation while the arithmetic is wrong. That is
the companion check `../keccak/README.md` lists as the template's first
follow-up, here done for this chip.

The same reasoning is why `feed_forward` is a congruence and not an integer
identity: written as `== 0` over the integers, the limb ranges force the carry
to a bit on their own, and the `drop_carry_bit` control sat green while testing
nothing. A constraint has to be modeled in the ring it lives in.

## Contracts assumed

| Contract | Guarantee modeled |
|---|---|
| `ByteAlu(op, a, b, c)` | `a,b,c` are bytes and `c = a op b`, `op ∈ {AND, XOR}`. The `255 − x` complement form `byte_bits` emits is checked to be bitwise NOT in lemma 2 rather than assumed. |
| `AreBytes(a, b)` | both in `0..256`. Carried structurally by an 8-bit sort where it is not load-bearing, and by an explicit bound where it is. |
| `IsHalfword(h)` | `h ∈ 0..65536`. This is what makes the `(out, carry)` split unique. |
| `ShaK(t, k)` | `k = K[t]`, the preprocessed round-constant table. Pinned to the concrete `K[t]`. |
| `ShaM(ts, i, w)` | `w` is the schedule word at index `i`, discharged by SHA256MSGSCHED (lemma 5). |
| `ShaRound(ts, i, state)` | the eight state words arrive range-checked from the previous row's `IsHalfword`s. **This is the one width not pinned inside the chip being verified.** |

There is no ROTXOR contract, because this design has no ROTXOR chip: a rotation
is an index permutation of the bit columns, so lemma 1 verifies it directly
instead of assuming a table.

## Width audit

Every lifted width and the constraint that backs it:

| value | width | backed by |
|---|---|---|
| `A[i]`, `E_[i]`, `B15[i]`, `B2[i]` | 1 bit | `check_bits` (`sha256_round.rs:216-218`, `sha256_schedule.rs:158-160`) |
| `B,C,FF,G[j]` and the BYTE_ALU outputs | 8 bits | the BYTE_ALU contract: the table has only byte rows |
| `OUT_A/E` halves, schedule `OUT` halves | 16 bits | `IsHalfword` (`sha256_round.rs:163-166`, `sha256_schedule.rs:98-100`) |
| `CARRY_A/E`, schedule `CARRY` | 8 bits | `AreBytes` (`sha256_round.rs:168`, `sha256_schedule.rs:101-106`) |
| `d`, `h` | 32 bits | **not pinned locally** — the ShaRound bus carries them from the previous row's range-checked halves |

**The bit columns are modeled as 8-bit vectors with an explicit `ULE(v, 1)`,
never as a 1-bit sort.** That is deliberate and it is the single most important
modeling decision here. A constraint carried by a variable's sort is outside the
falsifiable set: with a 1-bit sort, deleting `check_bits` from the Rust would
leave this gate printing VERIFIED while the bit columns went free over the
field — and `Σ 2^i · bit_i` with unconstrained columns reaches any field
element, so σ and Σ would be forgeable. With the explicit bound,
`bug="drop_bit_bounds"` flips to SAT in lemma 1.

Non-overflow side conditions: the round sums seven 32-bit values (`< 2^35`), the
schedule four (`< 2^34`); the models compute in 40–48 bits, so the bitvector
arithmetic cannot wrap where the field arithmetic would not.

**The carry bound is not needed for uniqueness**, and lemma 3 shows that
directly: `drop_carry_bounds` stays UNSAT. It is needed so the field arithmetic
cannot wrap the modulus, which QF-BV cannot see — the same scope gap #950
documents for the keccak gate, and it needs an integer-mod-`p` model to close.

## Two controls that had to be relocated

Both are recorded because a control that cannot flip is worse than no control —
it sits green and tests nothing.

- **`drop_h_from_temp1`** was in lemma 3, where both sides of the query are
  built from the same `temp1`, so dropping a term moved them together and the
  control stayed UNSAT. It lives in lemma 4, where it flips.
- **`sigma_swap`** was vacuous in lemma 5 while σ0/σ1 entered as free values —
  addition commutes, so swapping two frees is a relabeling. Lemma 5 now derives
  them from the bit columns, and the control flips.

Likewise `drop_bit_bounds` is deliberately **absent** from lemma 5, where both
sides build σ from the same columns; its necessity is proven in lemma 1.

## A bug this gate caught, in the oracle

The first run of lemma 1 came back SAT. The circuit was right: its Σ0 agreed
with the validated concrete mirror and with FIPS 180-4; the reference was wrong,
because z3py's `>>` on a bitvector is an **arithmetic** shift and sign-extended
once the top bit was set. `LShR` fixed it. This is exactly what
`model_dataflow.py` + `test_dataflow.py` exist to catch, and it is why "the
reference must be independent *and* anchored" is not a formality.

## Citations

`check_citations.py` re-checks every `sha256_*.rs:NN` citation against the
files, and verifies the cited range still contains the identifier the comment
names — not merely that the file is long enough. On its first run it found 12
stale citations written twenty minutes earlier, because adding one method to a
chip shifted every line below it. PR #950 found the same class of rot across
the whole keccak gate. Run it before trusting anything here.

## Not covered

- The **helper chips themselves** — BYTE_ALU, BITWISE, the preprocessed `ShaK`
  table. Each is a separately verifiable chip; this gate assumes their contracts.
- **Cross-row properties**: that the round chain is 64 links, that the schedule
  covers 16..64, that the timestamps bind a call together. QF-BV here models one
  row; the bus topology is what enforces those.
- Whether the range bounds **suffice mod `p`** — outside QF-BV entirely.

## Files

| file | role |
|---|---|
| `sha256_ref.py` | the independent reference: K and H generated from their FIPS definitions, the round, the schedule, the compression function |
| `test_ref.py` | anchors it to `hashlib` and cross-checks K/H against the repo |
| `model_dataflow.py` | the concrete mirror of the circuit's equations, with the column-role map |
| `test_dataflow.py` | mirror vs reference forward, plus falsifiability of seven injected bugs |
| `lemma_sigma.py` | lemma 1 |
| `lemma_chmaj.py` | lemma 2 |
| `lemma_add.py` | lemma 3 |
| `lemma_compose.py` | lemma 4 |
| `lemma_schedule.py` | lemma 5 |
| `lemma_core.py` | lemmas 6-8, over the integers mod `p` |
| `check_citations.py` | citation freshness |
| `run_gate.py` | the board |
