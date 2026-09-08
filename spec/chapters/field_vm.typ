#import "/meta.typ": rj, aside
#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, total_nr_variables, total_nr_instantiated_columns, compute_nr_interactions, render_constraint_table

This chapter describes, in line with the split between a binary and a field VM from @recursion,
the ISA and an arithmetization of a dedicated field VM.
The ISA is centered around a single, versatile instruction, that can handle both
extension field arithmetic and program flow.

= ISA

The central instruction of the ISA is a constraint for a fused multiply-add over the extension field:
`FMA o == a * b + c`.
Here all of `o`, `a`, `b` and `c` are arguments following the addressing scheme described below.
The field VM uses only read-only memory, which is implemented as a committed table,
together with the multiplicities with which each element is accessed.

== Arguments and addressing

The VM has a state consisting of $N$ general purpose extension field registers,
a base-field `PC` register, and a bit-register `ZERO`.
The number of registers was chosen as a tradeoff between the versatility of having more mutable state,
and the extra cost in committed columns and decoding logic that grows with $N$.
#rj[Register index for `ZERO` = $0$ and `PC` = $1$ and gp register `Ri` = $i + 2$]

Each argument to the FMA constraint has either of the two following forms:
- `imm_0 * reg + imm_1`
- `MEM[imm_0 * reg + imm_1]`
where each immediate is a base field element, encoded in the instruction for a specific argument,
and `MEM[...]` reads memory.
The `o` argument to the instruction obtains its register values from the _future_ state.
That is, the state from which the next instruction will get its input register values.
The `ZERO` register of the future state contains a bit (as a base field element) that indicates whether or not
the `o` value was zero for the current instruction.
The `PC` register contains the program counter (as a base field element) and indicates which instruction is to be executed.
Every other register can hold an arbitrary extension field element.

== Register hints

Each register (other than `ZERO` and `PC`) in the current state can be marked as _hinted_ in an instruction.
This means that its value from the current instruction onward can get a new value
that is independent from the previous value, except as constrained by the instruction.
Additionally, the _output_ can be marked as hinted, meaning that the register used in the `o` argument
whether wrapped in a `MEM[]` lookup or not, will change in the future state, and as such in the `o` argument too.
Output hinting is commonly used to assign the result of a computation: `FMA X == X * X, hint out` would
compute `X * X` and re-assign it to `X` in the future state.
The output hint, in contradiction with the input hints, does allow `PC` to be hinted, so as to enable
causal jumps and control flow in the program.
Any register that is not hinted will have the same value in the future state as in the current state,
with the appropriate exceptions in behaviour for the `ZERO` and `PC` registers.

#aside("Hint collisions")[
One may observe that an output hint for state `i` and an input hint on state `i + 1` can affect
an identical register in an identical state.
While this is true in theory, it is not a problem in practice, as two successive states are,
in almost all cases, operated on by two consecutive ---in the program text--- instructions,
and as such, hinting collisions can be easily identified, and most actual programs
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
- `hint` notation indicates which registers are hinted (default unhinted), `hint out` means hinting `o` as above

We note that this may be insufficient for the execution/prover side of the program,
as this provides no information on _which_ value exactly should be hinted,
but leave this as an implementation detail to be decided upon based on practical experience.

We label the instruction with an `FMA` mnemonic, even though that is the only possible "real" instruction,
in order to allow program listings to include other mnemonics to indicate pseudoinstructions that
map more specialized semantics onto the FMA functionality.
Next, we suggest some potential pseudoinstructions along with their translation.
This list is meant as an example, rather than an exhaustive enumeration;
implementers and practitioners are encouraged to discover and use their own,
as experience may point out further useful abstractions.

/ `ADD o, a, b`: Addition: `FMA o == (0 * X + 1) * a + b, hint out`
/ `MUL o, a, b`: Multiplication: `FMA o == a * b + (0 * X), hint out`
/ `INV o, a`: Extension field inversion. Note: `o` and `a` cannot use the same register here: `FMA o == (a + 1) * o - 1, hint <register of o>`
/ `J a`: Jump. Can be to a register, memory content, or absolute address, depending on the addressing mode of `a`, even relative to PC: `FMA PC == a, hint out`
/ `JZA imm`: Jump if ZERO, absolute target address: `FMA PC == (ZERO)*(-1*PC+(imm-1))+(1*PC+1), hint out`
/ `JZR a`: Jump if ZERO, PC-relative target address: `FMA PC == (ZERO) * (a - 1) + (PC + 1), hint out`
/ `JNZA imm`: Jump if not ZERO, absolute target address: `FMA PC == ZERO * (PC - imm) + (ZERO + imm), hint out`
/ `JNZR a`: Jump if not ZERO, PC-relative target address: `FMA PC == (a - 1) * (-1 * ZERO + 1) + (PC + 1), hint out`

Eventually, we hope that a set of common pseudoinstructions can be extracted from actual usage,
and inform potential optimizations that remove unused capabilities (e.g. reducing the number of immediates involved).

== Calling convention

Since the VM makes use of read-only memory, traditional usage of a program stack does not work.
We assume that each function invocation (unless other optimizations apply) will have an associated _frame_,
pointed to by a _frame pointer_ `fp`, one of the general purpose registers.
In this frame, the following data is stored:

/ `MEM[fp - k]...MEM[fp - 1]`: `k` saved registers from the calling function
/ `MEM[fp + 0]`: The stored parent frame pointer
/ `MEM[fp + 1]`: The return address
/ `MEM[fp + 2]...MEM[fp + l]`: Additional information required by the function

Then, to facilitate function calls, we describe a possible implementation of the `CALL` and `RET` pseudoinstructions, that, respectively, perform a new function call and return back to the caller.

```
CALL target:
  FMA [fp] == fp, hint out
  FMA [fp - i] == STORED_REG_i
  FMA [fp + 1] == (PC + 2)
  FMA PC == target, hint out
  FMA fp == [fp], hint out

RET:
  FMA PC == [fp + 1], hint out
```

#rj[Figure out halting, can be a simple self-loop, just have to figure out how to have the verifier assert this]

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

We first decode the instruction at the current PC.

#render_constraint_table(chip, config, groups: "decode")

Then, we compute all values $#`imm`_0 dot #`reg` + #`imm`_1$,
where we need to multiplex out of `registers`, based on `argument_registers[i]`.
We do this by constructing the Lagrange basis polynomials $f_(i)(x)$ such that $f_(i)(j) = 1$
for $i = j in [0, N + 1]$ and $f_(i)(j) = 0$ for $i != j in [0, N + 1]$.#footnote[
  Note that we allow ourselves to multiplex $N + 2$ registers here,
  combining the $N$ general purpose registers, the PC and the `ZERO` register.
]
Since the degree of these $f_(i)(x)$ can grow too large to express in a single polynomial constraint,
we perform a _"degree split"_:
$ f_(i)(x) = f_(i, 0)(x) + x^(d - 1) (f_(i, 1)(x) + x^(d - 2) (f_(i, 2) + x^(d - 2) (f_(i, 3) + ...))), $
for a maximal constraint degree $d$.
Here, $deg f_(i, 0) <= d - 2$ and $deg f_(i, k) <= d - 3$.
We denote by $t + 1$ the number of non-zero $f_(i, k)$ for fixed $i$.
This allows us to first compute the values of
$#`argument_registers[i]`^(d - 1)$, $#`argument_registers[i]`^(2d - 3)$ and so on
to `arg_register_powers` with constraints of degree $<= d$,
and then compute $#`args_premem[i]` = #`argument_scalars[i]` dot sum_(j = 0)^(N + 1) #`registers[j]` dot f_(j)(#`argument_registers[i]`) + #`argument_offsets[i]`$.
The coefficients for all $f_(i, k)$ are pre-computed once, based on the choices of `N`, `d` and `t`,
and used through the `MUX` constant columns.
#rj[Analysis of relation between $d$, $N$, $t$; mention current choice of $(d, N, t) = (5, 5, 1)$]
Observe that the handling for `argument_registers[0]` is separate, as this represents the output argument,
which should take its values from the next row in the table.

#render_constraint_table(chip, config, groups: "mux")

Once we have these values, we can then perform an optional indexing into memory, and copy over the values otherwise.
The case of the `ExtField` value into `BaseField` is mostly technical here, as a means to make the signature look reasonable.
Verification should fail if the value does not fit.
This failure is automatically satisfied by keeping the `ExtField` value as-is, since the `BaseField` would get reinterpreted as `ExtField`
in the LogUp, and the memory table should only provide `BaseField` addresses.

#render_constraint_table(chip, config, groups: "memory")

Now everything is in place to check the core operation of the VM: the FMA constraint.

#render_constraint_table(chip, config, groups: "fma")

Finally, we must ensure the consistency between consecutive rows of the table,
and allow for hinting.
We again make use of the multiplexing machinery from before.
The constraints we want to enforce on a register index $r$ are as follows:
- $!#`hint_input`'_r and !#`hint_output` => #`registers`'_r = #`registers`_r$, `r` could not have been hinted,
  since it was not input-hinted in the next row, and there was no output hint, so the next `r` should remain the same.
- $!#`hint_input`'_r and f_r(#`argument_registers`_0) = 0 => #`registers`'_r = #`registers_r`$
  `r` was not input-hinted in the next row, and it was not the output register, so it once again stays the same.

This is equivalent to the logical statement $!#`hint_input`'_r and not (#`hint_output` and f_r(#`argument_registers`_0) = 1) => #`registers`'_r = #`registers`_r$, but expressed in a way that polynomial constraints can more easily handle.
Naturally, the PC gets an exception since if it is not (output-)hinted, we need $#`pc`' = #`pc` + 1$,
and the $#`ZERO`'$ register purely depends on $#`args`_0$ and not on `ZERO`.

#render_constraint_table(chip, config, groups: "transition")

== Padding

#rj[...]

= Notes and potential optimizations

- Depending on observed use, in the future, we can restrict this design in some potential ways, to make proving it faster, without sacrificing too much utility:
 - We can restrict the amount or targets of hinting allowed
 - We can reduce the places in which immediates are valid
 - We can reduce for which arguments a memory access can be specified
 - Do we need input hinting per register, or can we reduce things to input hinting for (some of) the used registers only
- The `FIELD_VM_DECODE` table can be further compressed, including potentially fitting multiple base field elements into single extension field elements
- Since memory accesses can probably be presumed to have `BaseField` indices, we may be able to reduce area/hashing somewhat by working with the overlap of `args_premem` and `args`
