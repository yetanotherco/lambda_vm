#import "/book.typ": book-page, aside, attention
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

#show: book-page("hint.typ")

#let config = load_config()
#let chip = load_chip("src/hint.toml", config)
#let hint = raw(chip.name)

= Overview

The #hint chip serves an `ECALL` that hands the guest a value the prover computed for it.
It is not an accelerator in the sense of @keccak or @ecsm: those chips _constrain_ the function they compute, so the guest may use their output directly.
This one constrains nothing about the value it delivers.
Its purpose is to move a computation that is expensive in the guest --- and cheap to _check_ in the guest --- out of the constraint system entirely, leaving only the check behind.
The motivating example is modular inversion: a software inversion in the guest costs thousands of constrained instructions, while verifying a candidate inverse costs one multiplication.

The value is chosen by the prover.
The chip therefore constrains only the _frame_ of the call:

+ it receives the `ECALL`, so the interaction the `CPU` emits is taken off the bus (@hint:c:receive_ecall);
+ it binds the three operands to the registers the `CPU` held at this timestamp, and range checks them against the set the VM accepts (@hint:c:read_selector through @hint:c:range_addr_out);
+ it sends the four `MEMW` writes that place the value in memory (@hint:c:write_out), since the `ECALL` writes guest memory directly, bypassing the `STORE` chip (@store); and
+ it range checks the 32 written cells as bytes (@hint:c:range_out).

#attention("The delivered value is unconstrained.")[
  No constraint in this chip relates `out` to the value stored at `addr_in`, and none can be added without paying for the computation the chip exists to avoid.
  A guest program that _uses_ a hinted value without verifying it is unsound, and the machine will not catch this: the trace of an honest execution and the trace of an execution fed a fabricated hint are both valid.
  This places an obligation on the calling program (@hint:a:guest_verifies), discussed in @hint:s:guest.
]

= Interface

The chip is triggered by executing `ECALL` with `ECALL` number $-31$.
The `CPU` carries that number as a two's complement `DWordWL`, so the `ECALL` bus sees the limb pair $[2^32-31, 2^32-1]$; the chip takes it off again in @hint:c:receive_ecall.
It expects
- `x10` (`A0`) to contain the `selector`, naming the operation to perform,
- `x11` (`A1`) to contain the address of the 32-byte input, and
- `x12` (`A2`) to contain the address of the 32-byte output buffer.
Both buffers hold big-endian integers.
#footnote[
  This is the opposite of @ecsm, whose operands are little-endian.
  There, the byte order is dictated by the chip that consumes the limbs; here, no chip ever interprets the bytes --- they are only copied --- so the order is free to match the serialization the guest already uses.
]
Unlike the general `ECALL` convention (@ecall), nothing is returned in `A0`.

We assign the following `selector` values, all three over `secp256k1`, writing $x$ for the 32-byte value at `addr_in`:
#align(center)[#table(
  columns: (auto, auto, auto),
  table.header(`selector`, "operation", "check available to the guest"),
  "0",  [base-field inverse $x^(-1) mod p$], $x dot #`out` = 1$,
  "1",  [scalar-field inverse $x^(-1) mod n$], $x dot #`out` = 1$,
  "2",  [base-field square root $sqrt(x) mod p$], $#`out`^2 = x$,
)]
The set is deliberately small: a `selector` earns its place by naming an operation whose _result_ is cheaper to verify than to compute.
Adding one requires raising the bound in @hint:c:range_selector to match, so that this chip's constraints admit exactly the set the VM accepts.

== VM behaviour

Two kinds of malformed call are rejected up front: the VM halts with an error rather than executing the call (@hint:a:selector_rejected, @hint:a:addr_rejected).

/ Unknown `selector`: any value outside the table above. This is a program bug rather than an adversarial condition, so it should fail loudly; returning zeros would silently degrade the guest to the software path instead.
/ Operand out of range: an operand address satisfying $(#`addr` mod 2^32) + 31 >= 2^32$. Such a buffer straddles the boundary between the two address limbs; @hint:s:limb explains what goes wrong.

Everything else is executed.
In particular, an input that is _numerically_ unusable --- a non-canonical field element, a zero to be inverted, a quadratic non-residue --- is not an error: the VM writes 32 zero bytes and execution continues.
The VM must not instead signal the failure to the guest.
A hint is prover-chosen, so "the prover reports failure" and "the prover lies about failure" are the same event as far as the guest is concerned; a guest that branched on the report would hand the prover control of the branch.
Zeros are simply one value that will not pass the guest's check
#footnote[
  Zeros parse as a canonical field element, so they are rejected by the algebraic check rather than by the decoding step.
  For the square root that check is $#`out`^2 = x$, which $#`out` = 0$ fails whenever $x eq.not 0$; the one caller asks for $sqrt(x_P^3+7)$, and $x^3+7$ is never zero on `secp256k1` because the curve has odd order and therefore no affine point with $y = 0$ --- a root $x_0$ of $x^3+7$ would give exactly such a point $(x_0, 0)$.
]
--- nothing distinguishes them from any other value that fails, which is exactly what is wanted.

= Guest obligation <hint:s:guest>

The obligation stated here (@hint:a:guest_verifies) falls on the calling _program_, and no constraint of this chip enforces it: that is the price of leaving the value unconstrained.

A hinted value may only ever _save work_.
It must never change the guest's answer.
That rules out two distinct failure modes, and closing only the first leaves a hole.

*Accepting a wrong value.*
The guest must reject a non-canonical encoding of `out` --- @hint:c:range_out constrains it only to be 32 bytes, so it may exceed $p$ or $n$ --- and must then verify the value with ordinary constrained instructions, using the check tabulated above for its `selector`, in terms of the value it placed at `addr_in`.
For a point recovery that value is $x_P^3 + 7$, so the guest asks for $sqrt(x_P^3+7)$ and checks $#`out`^2 = x_P^3+7$; it must additionally select the root of the requested parity, using the canonical encoding it has just validated.
That selection is a fix-up rather than a test: both roots are legitimate answers, so the guest corrects the sign itself instead of rejecting.

*Steering a rejection.*
A failed check must _not_ be read as "the input was invalid".
It must trigger a software recomputation, whose result is authoritative.
Without this, a prover that feeds garbage turns a verification failure into a rejection: for signature recovery, recovery of a valid signature fails, the returned value is empty, and the resulting state root differs --- with both the honest and the attacked execution provable.
The guest cannot distinguish the two cases, so it must not try: it recomputes, and the software result decides.

#aside("When the fallback runs.")[
  For the two inverses the fallback exists only for the dishonest case, so it never runs against an honest prover: the caller validates $x$ as canonical before asking, and the inverse always exists because the callers exclude $x = 0$.
  #footnote[Directly, for the scalar inverse. For the base-field inverse the argument is a product of three factors the caller guards, two doubled $y$-coordinates and the constant $2$; the $y$-coordinates are non-zero by the same odd-order argument as above.]
  It is nonetheless mandatory: precisely what runs when the prover lies.
  For the square root the fallback also runs on honest inputs: a genuine non-residue has no root, and the software path must still return "no such point".
  That is why the fallback must be a _recomputation_ and not a rejection --- the two cases are indistinguishable from inside the guest, and only one of them is allowed to produce a negative answer.
]

= Columns

#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #hint chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s).
One row serves one hint call.
#render_chip_variable_table(chip, config)

= Assumptions
#render_chip_assumptions(chip, config)

= Constraints

== Receiving the `ECALL`

#render_constraint_table(chip, config, groups: "ecall")

@hint:c:mu_isbit is load-bearing, not a restatement of something the bus already gives.
The LogUp argument pins only the _net_ multiplicity of each distinct tuple, and every interaction of this chip is scaled by `μ`.
Without the bit constraint, take two rows agreeing in every column except `out`, one with $#`μ` = 1$ and one with $#`μ` = -1$.
Every interaction whose tuple does not mention `out` cancels between them: the `ECALL` receive, the three register reads, and the three operand range checks.
The pair is therefore invisible to the `CPU` --- it does not need an `ECALL` to have been issued at all --- and the cancellation destroys the only thing that tied `timestamp` and `addr_out` to any value the program chose.
What survives is @hint:c:write_out, whose tuple carries `out`, and @hint:c:range_out.
The $+1$ row's four writes are received by `MEMW` and land in memory.
The $-1$ row's four cannot be: `MEMW` receives at $-#`μ_read`$ and $-#`μ_write`$, both bits, so a receiver never contributes $+1$.
They must instead cancel four $+1$s from another _sender_, which fixes what to aim at --- a sender that emits four 8-byte writes at one timestamp.
`ECSM`'s output write (@ecsm) is exactly that: four write-only `MEMW` sends in the `RAM` domain with $#`write8` = 1$, at bases $8$ apart.
Setting the $-1$ row's `timestamp`, `addr_out` and `out` to match them cancels all four, so the accelerator's result never reaches memory, and the $+1$ row's four writes put 32 witness-chosen bytes there instead.
#footnote[
  A single-write sender such as @store can also be used, but only in a weaker form: the pair must then agree on the three doublewords it does not intend to replace, so that three of the four writes cancel within the pair and one is left to match the store.
]
@hint:c:range_out does not catch the difference: `ARE_BYTES` is received at a free multiplicity, which absorbs the $-1$.
Constraining `μ` to a bit makes the argument local to this chip.

== Binding and checking the operands

We read the three operand registers at `timestamp`, which pins `selector`, `addr_in` and `addr_out` to the values the `CPU` held when it issued the `ECALL`.
For `addr_out` this is the difference between a chip that writes to an address the program chose and an arbitrary-memory-write gadget: the write bases in @hint:c:write_out are derived from `addr_out`, so were it not bound, the witness --- not the program --- would choose the destination.
No in-guest check can repair that, since the adversary simply targets a buffer other than the one the guest verifies.

#render_constraint_table(chip, config, groups: "operands")

@hint:c:range_selector holds this chip's constraints to the set of operations the VM implements.
The VM applies the same bound (@hint:a:selector_rejected), and the two must move together: a `selector` the constraints admit but the VM rejects lets a prover prove an execution that halts.

=== The address bound <hint:s:limb>

@hint:c:range_addr_in and @hint:c:range_addr_out both assert $#`addr`_0 < 2^32 - 31$, i.e. the low limb is bounded so that the whole 32-byte buffer stays within the limb's range.
The reason is @hint:c:write_out: it offsets the _low_ limb of `addr_out` by $8i$ and passes the high limb through unchanged, so a carry has no representation in the four write _bases_.
The bytes _past_ a base are not a problem --- `MEMW` derives them from its own address columns, whose carry does reach the high limb (@memw).
What each base must satisfy is `MEMW`'s own assumption that a base address is a `Word` in each limb, and that first fails at $#`addr_out`_0 = 2^32-24$, where the last base $#`addr_out`_0 + 24$ would be $2^32$.

The bound the chip asserts is stricter than that, and deliberately so: it is the VM's bound (@hint:a:addr_rejected), and the two must agree.
The memory argument alone would admit up to $2^32-25$, leaving the seven values $2^32-31, ..., 2^32-25$ that it accepts and the VM rejects --- again a provable execution that halts.
`addr_in` needs its own check for a simpler reason: no memory access at `addr_in` is modelled (@hint:s:noread), so nothing on the memory side bounds it at all.

All three range checks need @lt:a:range_lhs for the limb they compare, and the register reads above are what supply it: registers live in the memory argument with `Word` limbs, so a value read out of a register is range checked as long as every write into one is (@cpu:memory).
That is an invariant of the system rather than something `MEMW` enforces --- its assumptions deliberately cover no range check for `value` (@memw) --- so the bound rests on the register writers, not on the operands' presence in the memory argument as such.

Compare @ecsm, which materializes its four write addresses as columns of their own and derives them with `ADD`.
In its constraints that is the strictly stronger property, since a base may then cross the limb boundary; the VM applies the same 31-byte rejection to `ECSM`'s operands, so the extra capability is unused today.
The bound here is the cheaper trade, but it is a real limitation: whether a 32-byte buffer lands within 31 bytes of a $2^32$ boundary is the allocator's choice, not the programmer's.

Because no base may leave the low limb, the four 8-byte ranges cannot wrap and are pairwise disjoint.
The three register reads land in a different memory domain (@memory), and their two-address spans are disjoint within it.
That is what allows all seven memory interactions of this chip to carry the same `timestamp`: no address is touched twice at one timestamp.

== Writing the output

#render_constraint_table(chip, config, groups: "write_out")

The `ECALL` writes guest memory directly rather than through the `STORE` chip (@store), so these writes appear in no `CPU` operation.
Without @hint:c:write_out the initial-to-final chain for those 32 cells is unexplained and the memory argument does not balance (@memory).

@hint:c:range_out is required because `MEMW` range checks nothing it receives, and `out` is 32 free columns.
Every chip that puts _fresh_ values into the `RAM` domain pins those cells to bytes somewhere.
#footnote[The field-storage domains are a separate matter: @fext writes whole field elements into a cell and range checks them against the field modulus instead.]
`STORE` (@store) and the `PAGE` chip (@streaming:chip:page) check them locally; `KECCAK` (@keccak) and `ECSM` (@ecsm) do not, and instead receive the cells over a bus that pins them on the far side --- `output_state` is a `BYTE_ALU` output of the keccak-round chip, and `xR` is range checked by the `ECDAS` chip.
We should be precise about the second: `ECSM`'s $#`xR` < p$ check is _not_ what pins `xR`'s bytes, since it carries at 32-bit word granularity, and a word does not determine how its four cells split.
`out` arrives over no bus at all, so there is no far side, and the check has to be local.
Without it, the witness can put field elements outside $[0, 256)$ into memory while keeping the linear combination on the bus consistent, and break the byte decomposition that `LOAD` (@load) and the ALU rely on.

=== The input read is not modelled <hint:s:noread>

The `ECALL` reads 32 bytes at `addr_in`, and this chip emits no interaction for it.
This is sound for one reason only: the value is unconstrained.
No constraint relates what was read to what was written, so an interaction binding the read to memory would bind nothing that any other constraint depends on.
It is _not_ sound merely because a read leaves memory contents unchanged.
Omitting a read does leave the memory argument balanced --- the token survives at its old timestamp until finalization.
But that is exactly as true of `LOAD` (@load) and of `ECSM`'s read of $x_G$ (@ecsm), whose reads are also value-preserving and which nonetheless pay for the interaction: there the read is what _binds_ the operand to memory.
Here there is no operand to bind.

= Padding

#render_chip_padding_table(chip, config)

= Notes / optimizations

- The chip could constrain the value after all, for the two inverses, by reading the input into columns of its own and asserting $x dot #`out` = 1$ over the appropriate modulus.
  That would make the `ECALL` self-sufficient and discharge the guest obligation (@hint:a:guest_verifies), at the cost of the input columns, the read that binds them, and the modular-multiplication constraints --- which is very nearly the multiplication the guest performs anyway.
  The trade is between paying once in the constraints for every hint call and paying in the guest only where a hint is actually used.
  Note that such a constraint must not read `out` as a `U256BL` in the usual limb order: this chip stores it most-significant-byte-first, so that @hint:c:write_out lays the buffer out big-endian in memory.
- `addr_in`'s high limb is the only part of the three operands that no constraint uses; it is bound by the register read alone.
  `addr_out`'s high limb is passed through to the write bases, and `selector`'s is pinned to $0$ by @hint:c:range_selector, which compares the full 64-bit value.
  The address checks instead pass the low limb with a literal zero high limb, so only the low 32 bits are compared --- matching the VM, which ignores the high limb.
- Should the hint set grow beyond values that fit a single 32-byte input and output, the interface would need a length or a shape parameter; today both buffers are fixed at 32 bytes and the four `MEMW` writes are unrolled accordingly.
