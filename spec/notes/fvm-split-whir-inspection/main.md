# Inspecting the Field VM and Split Proving Implementation

This note is based on commit `44f0afacabce96e97f2c16bbfa440c444039bd7a`.

## Initial Questions

- How is the Field VM implemented in relation to the LFM, or of Lambda VM?

## The Various Machines

### The Lambda Field Machine

The [LFM](/prover/src/lfm/mod.rs) is a machine with a limited instruction set
designed to execute straight-line verification of proofs.
The workflow it serves is the following:

1. *Building.* A Rust function initialises an `LfmBuilder` and calls it to
   step-by-step build an `LfmProgramSource`, emitted by a final call to
   `LfmBuilder.finish()`.
   This source contains a list of LFM instructions together with memory access
   information.
1. *Compiling.* The `compile()` function takes such an `LfmProgramSource` and
   emits an `LfmProgram`: a set of pre-processed columns containing the
   table-encoded instructions for the straight-line verification.
   This resulting `LfmProgram` can be checked with `validate()` to ensure it
   doesn't violate the requirements for LFM execution.
1. *Committing.* The preprocessed columns are committed as a fixed reference to
   the program's identity.
1. *Executing.* `execute()` takes an `LfmProgram` together with some "arenas"
   and an `LfmHasher` and generates an `LfmExecution`: per-chip row records
   encoding the execution of the verification program on the proof (encoded
   within the "arenas") together with publicly emitted words and the final state
   of the write-once memory.
1. *Proving.* `lfm_prove()` and `lfm_prove_whir()` then generates a proof, with
   a STARK or WHIR respectively, that the `LfmExecution` of an `LfmProgram` on
   some given "arenas" satisfies the AIR constraints of the LFM's ISA and that
   the communication buses balances.

### The Field Machine

## Experiments as Implemented

The [Field VM module](/prover/src/field_vm/mod.rs) contains some toy tests in
[lfm_compare.rs](/prover/src/field_vm/lfm_compare.rs).

### Horner Evaluation

[horner_field_vm_vs_lfm()](/prover/src/field_vm/lfm_compare.rs#L219) reports on
the performance of both VMs for Horner polynomial evaluation.
The FVM Program is constructed in the Rust code using the Field VM's native
`Asm` builder; it's written in two different forms: one with a loop, the other
unrolled.

Initially, running this Horner test appears to show that the FVM suffers in
comparison with the LFM; the following table reports on trace cell count:

|  len | machine      | program |    rows |      cells |
| ---: | ------------ | ------: | ------: | ---------: |
| 2^10 | fvm-loop     |       9 |   6,160 |    284,880 |
|      | fvm-unrolled |   1,028 |   4,112 |     95,864 |
|      | lfm          |   2,049 |  68,636 |    237,980 |
| 2^14 | fvm-loop     |       9 |  98,320 |  4,554,960 |
|      | fvm-unrolled |  16,388 |  65,552 |  1,524,344 |
|      | lfm          |  32,769 | 114,716 |    837,020 |
| 2^16 | fvm-loop     |       9 | 393,232 | 18,219,216 |
|      | fvm-unrolled |  65,540 | 262,160 |  6,095,480 |
|      | lfm          | 131,073 | 262,172 |  2,753,948 |

However, it seems that the comparison is flawed in several aspects:

1. The machines' cell counts are inaccurate.
   The `census()` function assigns a unit cost per row for each column in the
   machine's tables, including interaction columns which are typed over the
   extension field.
   This undercounts the interaction cells by a factor of 3, which favours the
   LFM since it likely has a larger share of interactions than the FVM.
   Furthermore, the `census()` function also gives the same weight to
   preprocessed columns as it does to main witness columns, whereas the
   commitment work done for preprocessed columns can potentially be amortized if
   the same program is proven repeatedly for different inputs.
   For a better comparison, the test should separately report preprocessed,
   witness (including interactions at 3 base-field elements each), and
   fixed-table cells.
1. The FVM's cell count is inflated because the machine uses 5 GP registers by
   default when it only requires 2 to run the program.

#### LFM Guest

In [`report_lfm()`](/prover/src/field_vm/lfm_compare.rs#L131) a new `LfmBuilder`
is used to construct a guest program that executes Horner evaluation of length
`len`.

- An `arena` is declared with length `len + 1`.

- The evaluation point `x` is hinted into this `arena` at index `0`, and
  interpreted as an extension field element.

  [`hint_word()`](/prover/src/lfm/builder.rs#L679) allocates a new word of
  memory and pushes a `Hint` instruction which takes the hinted word from the
  `arena` at the given index and stores it at the allocated address; it returns
  a handle to the cell at the allocated memory address.

- The polynomial's coefficients are hinted into the rest of the `arena`.

  1. A range of `u32` integers from `1` to `len` (inclusive) is declared.
  1. With `map()`, each `u32` is interpreted as an index passed to
     `hint_word()` for the `arena`.
     This pushes `len` `Hint` instructions to
     the program.
  1. Each returned `Cell` is interpreted as an extension field element and all
     are collected as `coeffs: Vec<Ext>`.

- The LFM EDSL's [`horner_ext()`](/prover/src/lfm/edsl.rs#L1048) is called with
  evaluation target the `Ext` value at `x` and with coefficients the `Ext`
  values

  For each coefficient after the highest one in `coeffs`, a `MulAdd` extension
  ALU instruction is triggered increasing an accumulated sum.
  This instruction reads each of the three operands, allocates a new memory cell
  for the result, and pushes an `ExtAlu` instruction to the program.
  In the Rust code driving the Horner loop, the result's address overwrites the
  previous accumulator value's address, ready to be consumed as an operand for
  the next `MulAdd`.

  The final accumulator value's address is returned as the polynomial's evaluated
  value.

- The resulting cell's address is exposed on the public bus with
  [`LfmBuilder::public()`](/prover/src/lfm/builder.rs#L695)

At this point, the `LfmProgramSource` for Horner evaluation is obtained by
calling [`LfmBuilder::finish()`](/prover/src/lfm/builder.rs#L702).

This source is then compiled with
[`compile()`](/prover/src/lfm/compiler.rs#L327) to produce an `LfmProgram`.
This compilation step backfills reading multiplicities for each address,
checking for invariant satisfaction, and then emits per-chip preprocessed column
groups that become the program's identity.
