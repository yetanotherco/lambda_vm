#import "/meta.typ": aside
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, total_nr_variables, total_nr_instantiated_columns, compute_nr_interactions, render_constraint_table, render_chip_padding_table
#import "/expr.typ": expr_to_math

#let next(x) = expr_to_math(("next", x))

This chapter describes, in line with the split between a binary and a field VM from @recursion,
the ISA and an arithmetization of a dedicated field VM.
The ISA is centered around a single, versatile instruction, that can handle both
extension field arithmetic and program flow.

= ISA<field-VM:sec:isa>

The field VM is a machine that has access to a read-only memory `MEM`, modeled as a flat array
that can be indexed by base field elements.
This `MEM` can be implemented as a committed table containing the memory as well as the multiplicities for
the number of times each cell was accessed.
As additional memory, the VM has a set of $N + 2$ mutable registers that are not part of the `MEM` array.

The central instruction of the ISA is a constraint for a fused multiply-add over the extension field:
`FMA d == a * b + c`.
Here all of `d`, `a`, `b` and `c` are arguments following the addressing scheme described below.

This constraint-based view generally goes well with a read-only memory.
The memory system gives us that guarantee that whenever we access `MEM` at the same index,
we get the same value back, and the constraint allows us to enforce that these
values in memory are consistent with the structure we want it to have.#footnote[
  In the most central application of the VM, we check that the memory consists of a correct proof
  and any auxiliary data needed for this verification.
]
For the mutable registers, however, this approach is insufficient, as the instruction does not have
a way to actually mutate a register.
We deal with this through a system we call _register hinting_ --- which can be further distinguished
into _input hinting_ and _output hinting_ --- described further below.


== Arguments and addressing

The execution of a guest program can be seen as a succession of _states_ of the machine.
A state is then the tuple of all values the registers have at a given point in time during the execution.#footnote[
  If we consider states across different executions, the contents of `MEM` should also be considered part of the state.
]
Each instruction acts upon _current state_ to produce the _future state_.#footnote[
  And since the program counter is part of the current state,
  each possible state has at most one associated instruction.
]
Or rather --- since instructions are constraints --- each instruction constrains
a correct transition from the current state to the future state.
We write $next("x")$ for the register `x` in the future state, both in prose and later in the constraints.

The $N + 2$ registers making up a state of the VM are:
a `ZERO` bit-register, the base field `PC` register and $N$ general purpose extension field registers.
The number of registers was chosen as a tradeoff between the versatility of having more mutable state,
and the extra cost in committed columns and decoding logic that grows with $N$.
We index the registers from $0$ to $N + 1$ in the order above, so `ZERO` gets index $0$,
`PC` gets index $1$ and then follow $N$ general purpose registers with indices $2...N+1$.

The `ZERO` register indicates whether the previous instruction had a zero result,
i.e. $next("ZERO") <=> #`d` = 0$.
The `PC` register stores the program counter: the address of the current instruction,
and --- except when branches are taken, through register hinting --- is incremented by $1$
for every consecutive state.
As the name implies, the general purpose registers are available for arbitrary usage.

Each argument to the `FMA` constraint has either of the two following forms:
- $#`imm`_0 dot #`reg` + #`imm`_1$
- $#`MEM[`#`imm`_0 dot #`reg` + #`imm`_1#`]`$
where each immediate is a base field element, encoded in the instruction for a specific argument.
The `d` argument to the instruction obtains its register value from the future state.

== Register hinting

Each general-purpose register in the current state can be marked as _hinted_ by the acting instruction.
This means that from the current state onwards,#footnote[Until it is hinted again.]
the register can take a value that is independent from the previous value,
except as constrained by the instruction.
Additionally, the _output_ can be marked as hinted, meaning that the register used in the `d` argument
will change from the future state onwards, and as such in the `d` argument too.
This applies to the _register_ of the `d` argument, regardless of the additional immediates and `MEM[]`
access that may happen in the instruction.
We distinguish between these two types by naming them respectively _input hinting_ and _output hinting_.

The clearest use of output hinting is to enable the `FMA` instruction to perform computation.
If, for instance, $#`d` = 1 dot #`reg` + 0$, then we can interpret the instruction
as computing $#`a` dot #`b` + #`c`$ and assigning the result to $next("reg")$.
Performing the hint only in the future state ensures that the original value of `reg` remains
available throughout the computation.
Additionally, output hinting allows for `PC` to be hinted,#footnote[
  Note that we disallow this in input hinting, as it would allow for instructions
  that can effectively hijack program execution.
] enabling causal jumps and control flow in the program.

In contrast, input hinting does not look like any traditional model of execution,
instead allowing to update one or more values in the state, as long as the resulting state still satisfies
the `FMA` constraint.
This can, e.g., be used to compute field inverses and square roots, which have a degree-2 constraint
on the result.
There may even be situations where hinting multiple values can be chosen simultaneously, such as a decomposition
$a = b + c$ in a divide-and-conquer algorithm.
Even hinting registers that are not used in the current instruction may provide useful in limited situations.
Though we approach it differently in @field-VM:sec:calling, one can imagine a calling convention
where the frame pointer is updated directly during the jump instruction, without being further involved
in the computation of the next `PC`.

Any register that is not output hinted in the current instruction nor input hinted in the future instruction
will have the same value in the future state as in the current state,
with the appropriate exceptions in behaviour for the `ZERO` and `PC` registers.
More example uses of register hinting can be found below in our suggested pseudoinstructions.

#aside("Hint collisions")[
One may observe that an output hint for state `i` and an input hint on state `i + 1` can affect
the same register in a single state.
While this is true in theory, it is not a problem in practice, as two successive states are,
in almost all cases, operated on by two consecutive --- in the program text --- instructions.
As such, hinting collisions can be easily identified, and most actual programs
should have no reason to have hinting collisions.
The most likely practical collision scenario would be that instruction `i` does not
output-hint, but instruction `i + 1` input-hints the output register of state `i`.
This would lead to confusing behaviour on instruction `i`, as it may not be operating on the output
value a programmer would assume it to be.
As input hints are likely to occur only seldom, we advise extra care for the surrounding
instructions of any input-hinting instruction.

The only case in which two consecutive states are not operated on by two consecutive instructions
is when a jump occurs, which necessarily implies that `PC` was output-hinted in the earlier instruction.
`PC` can, however, not be input-hinted, so no collision is possible there.
]

== Instruction notation

A potential way to write down an FMA instruction would be the following:
```
FMA [1 * X + 2] == [3 * Y + 4] * (5 * Z) + [W + 6], hint out + Z
```

- `[]` indicate memory access
- `()` indicate grouping to separate the arguments
- `X, Y, Z, W` are placeholder register names
- `hint` notation indicates which registers are hinted (default unhinted), `hint out` means hinting `d` as above

We note that this may be insufficient for the execution/prover side of the program,
as this provides no information on _which_ value exactly should be hinted,
but leave this as an implementation detail to be decided upon based on practical experience.

We label the instruction with an `FMA` mnemonic --- even though that is the only possible "real" instruction ---
to allow program listings to include other mnemonics to indicate pseudoinstructions that
map more specialized semantics onto the FMA functionality.
Next, we suggest some potential pseudoinstructions along with their translation.
This list is meant as an example, rather than an exhaustive enumeration;
implementers and practitioners are encouraged to discover and use their own,
as experience may point out further useful abstractions.

#table(columns: (auto, 2fr, 1fr),
       stroke: 0pt,
       inset: (right: .5em),
       table.header[*Pseudoinstr.*][*Translation*][*Comment*], table.hline(stroke: 1.5pt))[
  `ADD d, a, b`][`FMA d == (0 * X + 1) * a + b, hint out`][Addition][
  `MUL d, a, b`][`FMA d == a * b + (0 * X), hint out`][Multiplication][
  `INV d, a`][`FMA d == (a + 1) * d - 1, hint <register of d>`][Extension field inversion. Note: `d` and `a` cannot use the same register here, and `d` should not be input-hinted in the next instruction.][
  `J a`][`FMA PC == a, hint out`][Jump. Can be to a register, memory content, or absolute address, depending on the addressing mode of `a`, even relative to PC][
  `JZA imm`][`FMA PC == (ZERO)*(-1*PC+(imm-1))+(1*PC+1), hint out`][Jump if ZERO, absolute target address][
  `JZR a`][`FMA PC == (ZERO) * (a - 1) + (PC + 1), hint out`][Jump if ZERO, PC-relative target address][
  `JNZA imm`][`FMA PC == ZERO * (PC - imm) + (ZERO + imm), hint out`][Jump if not ZERO, absolute target address][
  `JNZR a`][`FMA PC == (a - 1) * (-1 * ZERO + 1) + (PC + 1), hint out`][Jump if not ZERO, PC-relative target address]

Eventually, usage may inform a set of common pseudoinstructions,
along with informing potential optimizations that remove unused capabilities
(e.g. reducing the number of immediates involved).

== Calling convention<field-VM:sec:calling>

Since the VM makes use of read-only memory, traditional usage of a program stack does not work.
We assume that each function invocation (unless other optimizations apply) will have an associated _frame_,
pointed to by a _frame pointer_ `fp`.
We assume here that `fp` is one of the general purpose registers.
Observe that we let `fp` point into the middle of the frame, such that the information relevant to the callee
starts at offset 0.
In this frame, the following data is stored:

/ `MEM[fp - k]...MEM[fp - 1]`: `k` saved registers from the calling function
/ `MEM[fp + 0]`: The stored parent frame pointer
/ `MEM[fp + 1]`: The return address
/ `MEM[fp + 2]...MEM[fp + l]`: Additional information required by the function

Then, to facilitate function calls, we describe a possible implementation of the `CALL` and `RET` pseudoinstructions, that, respectively, perform a new function call and return back to the caller.

```
CALL target:
  FMA [fp] == fp, hint out       // Hint a new frame address, store the old fp
  FMA [fp - i] == STORED_REG_i   // Store on the caller side of the frame
  FMA [fp + 1] == (PC + 2)       // Store the return address to the frame
  FMA PC == target, hint out     // Hint the PC to jump
                                 // Returning jumps here
  FMA fp == [fp], hint out       // Hint the old frame pointer to restore it

RET:
  FMA PC == [fp + 1], hint out   // Hint the PC to jump to the stored return address
```

As a halting state, we choose to let the VM loop to itself at `PC = 0`, hinting all inputs.
That means the decoding will always contain `FMA PC == PC, hint out, hint 2, ..., hint (N + 1)` at that address.
For technical reasons, in @field-VM:sec:boundary, execution of the VM starts at `PC = 1`, with `FMA 0 = 0` and no hinting.

= Arithmetization

#let config = load_config()
#let chip = load_chip("/src/field_vm.toml", config)
#let fieldvm = raw(chip.name)

#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #fieldvm is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):

== Variables
#render_chip_variable_table(chip, config)

== Constraints

First, we compute all values $#`imm`_0 dot #`reg` + #`imm`_1$,
where we need to multiplex out of `registers`, based on `argument_registers[i]`.
We do this by constructing the Lagrange basis polynomials $f_(i)(x)$ such that $f_(i)(j) = 1$
for $i = j in [0, N + 1]$ and $f_(i)(j) = 0$ for $i != j in [0, N + 1]$.#footnote[
  Note that we allow ourselves to multiplex $N + 2$ registers here,
  combining the $N$ general purpose registers, the PC and the `ZERO` register.
]
Since the degree of these $f_(i)(x)$ can grow too large to express in a single polynomial constraint,
we perform a _"degree split"_:
$ f_(i)(x) &= f_(i, 0)(x) + x^(d - 1) (f_(i, 1)(x) + x^(d - 2) (f_(i, 2) + x^(d - 2) (f_(i, 3) + ...)))\
           &= 1 dot f_(i,0)(x) + x^(d - 1) f_(i,1)(x) + ... + x^(d - 1 + (t - 1) dot (d - 2)) f(i, t)(x), $
for a maximal constraint degree $d$.
Here, $deg(f_(i, 0)) <= d - 2$ and $deg(f_(i, k)) <= d - 3$.
We denote by $t + 1$ the number of non-zero $f_(i, k)$ for fixed $i$.
This allows us to first compute the values of
$#`argument_registers[i]`^(d - 1)$, $#`argument_registers[i]`^(2d - 3)$ and so on
to `arg_register_powers` with constraints of degree $<= d$,
and then compute
$
  #`args_premem[i]` &= #`argument_scalars[i]` dot sum_(j = 0)^(N + 1) #`registers[j]` dot f_(j)(#`argument_registers[i]`)\
                    &+ #`argument_offsets[i]`.
$
The coefficients for all $f_(i, k)$ are pre-computed once, based on the choices of $N$, $d$ and $t$,
and used through the `MUX` constant columns.
In this way, $f_(i, 0)$ can have degree at most $d - 2$, as it gets multiplied with $#`imm`_0$ and the register value,
and the other $f_(i, k)$ can have degree at most $d - 3$, as they also get multiplied with the appropriate power of $x$.
This leads to a total degree of $deg(f_(i)) = d - 1 + t dot (d - 2) - 1$ for a maximal number of registers $N + 2 <= deg(f_(i)) + 1$.
Hence, for a fixed choice of $d$ and $t$, this scheme can support up to $N <= (t + 1) dot (d - 2) - 1$ general purpose registers.
Currently, the parametrization is set to be $(d, N, t) = (5, 5, 1)$.

While @field-VM:c:first-mux, and the other constraints using this multiplexing technique, look like they have
a total degree of $d + 1$, this is purely a syntactical matter.
Due to our choices to set `arg_register_powers[0] = 1` and `MUX[j][k][d - 2] = 0` for $k != 0$ (by construction of the $f_(j,k)$ polynomials), we stay at a total degree $d$.
Also observe that the handling for `argument_registers[0]` is separated as @field-VM:c:out-mux,
as this represents the output argument, which should take its values from the next row in the table.

#render_constraint_table(chip, config, groups: "mux")

Once we have these values, we can then perform an optional indexing into memory, and copy over the values otherwise.
The cast of the `ExtField` value into `BaseField` is mostly technical here, as a means to make the signature look reasonable.
Verification should fail if the value does not fit.
This failure is automatically satisfied by keeping the `ExtField` value as-is, since the `BaseField` would get reinterpreted as `ExtField`
in the LogUp, and the memory table should only provide `BaseField` addresses.

#render_constraint_table(chip, config, groups: "memory")

Now everything is in place to check the core operation of the VM: the FMA constraint.

#render_constraint_table(chip, config, groups: "fma")

We must ensure the consistency between consecutive rows of the table, and allow for hinting.
We again make use of the multiplexing machinery from before.
The constraints we want to enforce on a register index $r$ are as follows:
- $!next("hint_input")_r and !#`hint_output` => next("registers")_r = #`registers`_r$,\ `r` could not have been hinted,
  since it was not input-hinted in the next row, and there was no output hint, so the next `r` should remain the same.
- $!next("hint_input")_r and f_(r)(#`argument_registers`_0) = 0 => next("registers")_r = #`registers`_r$,\
  `r` was not input-hinted in the next row, and it was not the output register, so it once again stays the same.

Together, these constraints are logically equivalent to $!next("hint_input")_r and not (#`hint_output` and f_(r)(#`argument_registers`_0) = 1) => next("registers")_r = #`registers`_r$, but expressed in a way that polynomial constraints can more easily handle.

Naturally, the `PC` and `ZERO` registers are exceptions since we need $next("pc") = #`pc` + 1$ if it is not (output-)hinted,
and $next("ZERO")$ purely depends on $#`args`_0$ and not on `ZERO`.

#render_constraint_table(chip, config, groups: "transition")

Finally, to decode the instruction at the current PC, we would like to compress the information coming from
the decoding table to reduce its number of columns.
Doing so would require the elements being combined into one column to be range checked on this side
of the interaction, ideally without needing any extra interactions or committed columns.
For `Bit` variables, this is no problem with the `IS_BIT` template from @isbit.
For `argument_registers` however, which should be in the range $[0, N + 1]$,
a first attempt to range-check with a constraint of degree $<= d$ fails.
The standard way to construct the range-check polynomial $g$ would be to choose
$ g(x) = (x - 0) dot (x - 1) dot ... dot (x - (N + 1)), $
which has degree $N + 2$.
Our polynomial approach to multiplexing already provides a way to evaluate a polynomial of degree $<= N + 1$,
which falls short of one coefficient to evaluate $g$.
However, recall that $deg(f_(i,t)) <= d - 2$, and unlike in multiplexing,
$g$ needs no further multiplications to be used in an arithmetic constraint.
So we can simply add one extra coefficient to the last split polynomial to achieve our goal.#footnote[
  We can in theory choose any of the split polynomials to increase, but we need to ensure
  that we can still use the same `arg_register_powers` as before to recombine the results,
  so as to avoid the need for extra columns.
]
In the constraints, we write `RANGE` for the coefficients of $g$, in a similar structure to `MUX[r]`.
We assume $N <= 254$, such that each register index takes up at most 8 bits in the compressed column.

To compress base field columns, we can batch 3 base field columns as coefficients of an extension field element.
In order to do so, we write the constant column `X` as the extension field element, such that $(1, #`X`, #`X`^2)$
is the canonical basis of the extension field over the base field.

#render_constraint_table(chip, config, groups: "decode")

== Boundary constraints<field-VM:sec:boundary>

Besides enforcing the FMA constraints and the correct transitions between states, we also need to ensure that execution
starts at the correct instruction and ends with a halting instruction.
This means that the verifier must check that the first row of the table corresponds to a state at `PC = 1` and all other variables set to $0$;
as well as that the last row of the table corresponds to the halt/padding state.
This is also why the halt state has all inputs hinted, so that all registers can be set to zero and be known.

== Padding

The halting self-loop also functions as a padding state.

#render_chip_padding_table(chip, config)

= Notes and potential optimizations

- Depending on observed use, in the future, we can restrict this design in some potential ways, to make proving it faster, without sacrificing too much utility:
 - We can restrict the amount or targets of hinting allowed
 - We can reduce the places in which immediates are valid
 - We can reduce for which arguments a memory access can be specified
 - Do we need input hinting per register, or can we reduce things to input hinting for (some of) the used registers only
- Since memory accesses can probably be presumed to have `BaseField` indices, we may be able to reduce area/hashing somewhat by working with the overlap of `args_premem` and `args`
