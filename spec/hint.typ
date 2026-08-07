#import "/book.typ": book-page, aside, attention
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  compute_nr_interactions,
  render_constraint_table,
  render_chip_padding_table,
)

#show: book-page("hint.typ")

#let config = load_config()
#let chip = load_chip("src/hint.toml", config)
#let hint = raw(chip.name)

= Overview

The #hint chip serves an `ECALL` that hands the guest a value the host computed for it.
It is not an accelerator in the sense of @keccak or @ecsm: those chips _constrain_ the function they compute, so the guest may use their output directly.
This one constrains nothing about the value it delivers.
Its purpose is to move a computation that is expensive in the guest --- and cheap to _check_ in the guest --- out of the circuit entirely, leaving only the check behind.
The motivating example is modular inversion: a software inversion in the guest costs thousands of constrained instructions, while verifying a candidate inverse costs one multiplication.

The value is chosen by the prover.
The chip therefore constrains only _where_ it lands and _that it is 32 bytes_:

+ it receives the `ECALL`, so the interaction the `CPU` emits is taken off the bus (@hint:c:receive_ecall);
+ it binds the three operands to the registers the `CPU` held at this timestamp, and range checks them to the set the VM accepts (@hint:c:read_selector through @hint:c:range_addr_out);
+ it sends the four `MEMW` writes that place the value in memory (@hint:c:write_out), since the `ECALL` writes guest memory directly, bypassing the `STORE` path; and
+ it range checks the 32 written cells as bytes (@hint:c:range_out).

#attention("The delivered value is unconstrained.")[
  No constraint in this chip relates `out` to the value stored at `addr_in`, and none can be added without paying for the computation the chip exists to avoid.
  A guest program that _uses_ a hinted value without verifying it is unsound, and the machine will not catch this: the trace of an honest execution and the trace of an execution fed a fabricated hint are both valid.
  The obligation this places on the guest is stated in @hint:s:guest, and it is a property of the _program_, not of the VM.
]

= Interface

The chip is triggered by executing `ECALL` with `ECALL` number $-31$.
The `CPU` puts this on the `ECALL` bus as the pair $[2^32-31, 2^32-1]$; the chip takes it off again in @hint:c:receive_ecall.
It expects
- `x10` (`A0`) to contain the `selector`, naming the operation to perform,
- `x11` (`A1`) to contain the address of the 32-byte input, and
- `x12` (`A2`) to contain the address of the 32-byte output buffer.
Both buffers are big-endian.
#footnote[
  This is the opposite of @ecsm, whose operands are little-endian.
  There, the byte order is dictated by the chip that consumes the limbs; here, both buffers are only ever read and written by guest software, so the order is the one the guest's own serialization uses.
]
Nothing is returned in `A0`.

The following `selector` values are assigned, all three over `secp256k1`:
#align(center)[#table(
  columns: (auto, auto, auto),
  table.header(`selector`, "operation", "check available to the guest"),
  "0",  [base-field inverse $x^(-1) mod p$], $x dot #`out` = 1$,
  "1",  [scalar-field inverse $x^(-1) mod n$], $x dot #`out` = 1$,
  "2",  [base-field square root $sqrt(x) mod p$], $#`out`^2 = x$,
)]
The set is deliberately small: a `selector` earns its place by naming an operation whose _result_ is cheaper to verify than to compute.
Adding one requires raising the bound in @hint:c:range_selector in step, so that the AIR keeps accepting exactly the set the VM accepts.

== Host behaviour

Two malformed calls are rejected up front, and the VM halts with an error rather than executing them:

/ Unknown `selector`: any value outside the table above. It is important that this is an error and not a zero result --- see @hint:s:guest.
/ Operand out of range: either operand address for which $(#`addr` mod 2^32) + 31 >= 2^32$. Such a buffer straddles the boundary between the two address limbs; @hint:s:limb explains why the chip cannot represent it.

Everything else is executed.
In particular, an input that is _numerically_ unusable --- a non-canonical field element, a zero to be inverted, a quadratic non-residue --- is not an error: the host writes 32 zero bytes and execution continues.
This is a deliberate choice, and it is the only sound one.
A hint is prover-chosen, so "the host reports failure" and "the host lies about failure" are the same event as far as the guest is concerned; if the guest branched on it, the prover would control the branch.
Zeros are simply a value that will not pass the guest's check, and the guest treats them like any other value that does not pass.

= Guest obligation <hint:s:guest>

A hinted value may only ever _save work_.
It must never change the guest's answer.
That rules out two distinct failure modes, and closing only the first leaves a hole.

*Accepting a wrong value.*
Every hint must be verified in-guest with ordinary constrained instructions before use: $x dot #`out` = 1$ for the inverses, $#`out`^2 = x^3 + 7$ together with the parity selection for the root.
This is the obvious obligation, and the cheap one --- it is the whole point of the ecall.

*Steering a rejection.*
A failed check must _not_ be read as "the input was invalid".
It must trigger a software recomputation, whose result is authoritative.
Without this, a prover that feeds garbage turns a verification failure into a rejection: for `ECRECOVER`, a valid signature that fails to recover, an empty return value, and a different state root --- with both the honest and the attacked execution provable.
The guest cannot distinguish the two cases, so it must not try: it recomputes, and the software result decides.

#aside("Why the fallback is not dead code.")[
  For the inverses it is: the caller establishes $x eq.not 0$ before asking, so the inverse exists and a failed check can only mean the host lied.
  For the square root it is not: a genuine non-residue has no root, and the software path must still return "no such point".
  This is exactly why the fallback has to be a _recomputation_ and not a rejection --- the two cases are indistinguishable from inside the guest, and only one of them is allowed to produce a negative answer.
]

Both obligations sit outside this specification, in the program.
This chapter states them because the chip is only usable in their presence.

= Columns

#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #hint chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s).
One row serves one hint call.
#render_chip_variable_table(chip, config)

= Constraints

== Receiving the `ECALL`

#render_constraint_table(chip, config, groups: "ecall")

@hint:c:mu_isbit is load-bearing, not a restatement of something the bus already gives.
The `ECALL` tuple carries `timestamp`, which is a free column of this chip: `LogUp` therefore pins only the _sum_ of `μ` over the rows sharing a tuple with the `CPU`'s send.
A witness may spread that sum --- a $+1$ row together with a $+1$/$-1$ pair, each row keeping its own `addr_out` --- and use the extra rows to write wherever it likes.
The `MEMW` interactions do not catch it: `MEMW` only ever receives the legal $+1$, while the $-1$ cancels an honest `STORE` on the sender side.
Constraining `μ` to a bit makes the argument local to this chip.

== Binding and checking the operands

The three operand registers are read at `timestamp`, which pins `selector`, `addr_in` and `addr_out` to the values the `CPU` held when it issued the `ECALL`.
For `addr_out` this is the difference between a chip that writes to an address the program chose and an arbitrary-memory-write gadget: the write bases in @hint:c:write_out are derived from `addr_out`, so were it not bound, the witness --- not the program --- would choose the destination.
No in-guest check can repair that, since the adversary simply targets a different buffer than the one the guest verifies.

#render_constraint_table(chip, config, groups: "operands")

@hint:c:range_selector holds the AIR to the set of operations the VM implements.
It is the same bound the host applies, and the two must move together: a `selector` the AIR accepts but the host rejects lets a prover prove an execution that the VM halts on.

=== The address bound <hint:s:limb>

@hint:c:range_addr_in and @hint:c:range_addr_out both assert $#`addr`_0 < 2^32 - 31$, the low limb bounded so that the whole 32-byte buffer stays inside it.
The reason is @hint:c:write_out: it offsets the _low_ limb of `addr_out` by $8i$ and passes the high limb through unchanged.
A carry out of the low limb has no representation there, so an address within $31$ of the limb boundary describes a buffer the chip cannot address.
Compare @ecsm, which materializes its four write addresses as columns of their own and derives them with `ADD`; that is the more expensive way to buy the same property, and it costs the columns and the range checks that come with them.

Both operands need the check, for different reasons.
`addr_in` is not on the memory bus at all (@hint:s:noread), so nothing else bounds it.
`addr_out` _is_ on the bus, but the bus bounds it only to $2^32-25$: the largest write base is $#`addr_out`_0 + 24$, and `MEMW`'s carry handling resolves the bytes beyond it correctly.
That leaves the seven values $2^32-31, ..., 2^32-25$, which the memory argument accepts and the host rejects --- again a provable execution that the VM halts on.

The bound has a second use.
It guarantees that the four write bases $#`addr_out`_0 + 8i$ do not wrap, hence that the four 8-byte ranges are pairwise disjoint.
Together with the three registers being distinct, that is what allows all seven memory interactions of this chip to carry the same `timestamp`: no address is touched twice at one timestamp.

== Writing the output

#render_constraint_table(chip, config, groups: "write_out")

The `ECALL` writes guest memory directly rather than through the `STORE` path, so these writes appear in no `CPU` operation.
Without @hint:c:write_out the initial-to-final chain for those 32 cells is unexplained and the memory argument does not balance (@memory).

@hint:c:range_out is required because `MEMW` range checks nothing it receives, and `out` is a free column.
Every chip that puts _fresh_ values into memory carries its own check for this reason --- `STORE` does, and in @keccak and @ecsm the written bytes are pinned by the chip's own arithmetic instead.
Here there is no arithmetic to lean on.
Without it, the witness can put field elements outside $[0, 256)$ into memory while keeping the linear combination on the bus consistent, and break the byte decomposition that `LOAD` and the ALU rely on.

=== The input read is not modelled <hint:s:noread>

The `ECALL` reads 32 bytes at `addr_in`, and this chip emits no interaction for it.
This is sound: a read leaves memory unchanged, so the memory argument has nothing to balance, and the guest placed the input there with ordinary `STORE`s that are already accounted for.
It is also unobservable, precisely because the value is unconstrained --- there is no constraint that would relate what was read to what was written, so modelling the read would bind nothing.

= Padding

Padding rows set $#`μ` = 0$, which disables every interaction of this chip.
#render_chip_padding_table(chip, config)

= Notes / optimizations

- The chip could constrain the value after all, for the two inverses, by reading the input into columns of its own and asserting $x dot #`out` = 1$ over the appropriate modulus.
  That would make the ecall self-sufficient and remove the guest obligation of @hint:s:guest, at the cost of the input columns, the read that binds them, and the modular-multiplication constraints --- which is very nearly the multiplication the guest performs anyway.
  The trade is between paying once in the AIR for every hint call, or paying in the guest only where a hint is actually used.
- `selector` and both addresses are `DWordWL`s, but only their low limbs carry information the chip uses: the high limbs are constrained by the register binding alone.
  @hint:c:range_selector compares the full 64-bit `selector`, whereas the address checks compare the low limb against a literal zero high limb, matching the host.
- Should the hint set grow beyond values that fit a single 32-byte input and output, the interface would need a length or a shape parameter; today both buffers are fixed at 32 bytes and the four `MEMW` writes are unrolled accordingly.
