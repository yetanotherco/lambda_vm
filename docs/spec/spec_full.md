# Lambda VM Specification

# LogUp Argument

The _LogUp_ proof system conducts a permutation check based on summing partial derivatives. This check ensures that whatever tuple is sent to be "looked-up" by a _source table_ is indeed received in the expected _destination table_.

## Notation

### VM Notation

#### Preliminary notation

- `NN`: the set of non-negative natural integers. - `BaseField`: the base finite field used by the arithmetisation. - `ExtensionField`: a finite extension of `BaseField` of cryptographic size. - `[n]` for `n in NN`: the set of integers `{0, dots, n - 1}`. - `X[i]` for tuple `X`: the `i`-th element of `X`, starting at `0`.

#### Arithmetisation notation

- `numTables in NN`: number of tables `Table_i` in the arithmetisation of the VM. - `TableSet`: set of all tables `Table_i` in the arithmetisation of the VM. - `numColumns_i in NN`: number of _columns_ in table `Table_i` (not the number of variables). - `numRows_i in NN`: number of _rows_ in table `Table_i`.

### Interaction Notation

The `j`-th _interaction_ `Interaction_j` of table `Table_i` is defined by the following tuple:

columns: (auto, auto), inset: 6pt, align: horizon, stroke: none, table.header([*Symbol*], [*Description*]), table.hline(stroke: 1pt), table.vline(stroke: 1pt, x: 1), [`id_(i,j) in FF`], [the _type identifier_ of the interaction, usually the identifier of the chip that is constraining the relation expected to hold within the looked-up tuple.], [`numElements_(i,j) in NN`], [the _length_ of the tuple of elements being looked-up.], [ $weightFunction_(i,j) : FF^(numColumns_i) & arrow FF^(numElements_(i,j) + 1) \ R & mapsto arrow(t)_(i,j) || mu_(i,j)$ ], [the _weight function_ that maps a row `R` of table `Table_i` to the looked-up tuple `arrow(t)_(i,j)` and its multiplicity `mu_(i,j) in BaseField`.],

## Vanilla LogUp

### Protocol Description

+ Prover commits to all traces.

+ Verifier samples a random _(global) LogUp challenge_ `logupChallenge in ExtensionField` and a random _fingerprint coefficient_ `fingerprintCoeff in ExtensionField` and sends them to the Prover.

+ Prover commits to (i) interaction contribution, (ii) table running sum columns, and (iii) each table's contribution:

+ For each table `Table_i`, populate the interaction contribution columns and compute the _table (LogUp) contribution_:

+ For each interaction `Interaction_j` of table `Table_i`, initialize an empty _interaction contribution column_ of length `numRows_i`.

+ Initialise a _table running sum column_ `S_i in ExtensionField^(numRows_i)` with the first value `S_i [0]` populated according to the constraint choice.

+ *Constrain* the first row if required by selected constraint choice.

+ For each `j`-th row `R_j in BaseField^(numColumns_i)` of `Table_i`, for `j in [numRows_i - 1]`: + For each `k`-th interaction `Interaction_k` of table `Table_i`: + Compute the _interaction contribution numerator_ ` n_(j,k) = mu_(i,k) = w_(i,k)(R_j)[numElements_(i,k)] ` + If `n eq.not 0`, compute the _interaction contribution denominator_ ` d_(j,k) = logupChallenge + fingerprintCoeff dot id_(i,k) + sum_(l = 0)^(numElements_(i,k) - 1) fingerprintCoeff^(l + 2) dot weightFunction_(i,k) (R_j)[l]. ` + Save the _interaction contribution_ as `n_(j,k)/d_(j,k) in ExtensionField` in the corresponding interaction contribution column for this interaction. + *Constrain* the interaction contribution column according to the definitions of `n` and~`d`.

+ Compute the _row contribution_ as the sum `s_(j) = sum_k n_(j,k) / d_(j,k)` and compute the next row's table running sum value `S_i [j+1] = S_i [j] + s_(j)`.

+ *Constrain* the transition of the running sum column as indicated by the constraint choice.

+ *Constrain* the last row if required by selected constraint choice.

+ Batch-commit to every table's interaction contribution columns and running sum columns with the column commitment scheme and commit to the table's overall contribution `S_i [N_i - 1]` by sending it in the clear to the verifier.

+ Verifier checks that the sum of every table's overall contribution is equal to zero: `sum_i S_i [N_i - 1] = 0_ExtensionField`, and delegates the checks of the constraints to the STARK.

### Running Sum Constraint Choices <constraint_choices>

#### Choice 1: transitions looking back

tl,dr: implicit `0_ExtensionField` initial value, explicit final value.

+ (*Boundary, first row*) Constrain first row of running sum column to equal the sum of the first row of every interaction contribution column. (This is analogous an implicit `-1`-th row initialised at `0_ExtensionField`.) + (*Transition, looking back, applied to rows `1, dots, numRows_i - 1`*) For each row _other than the first_, constrain the _current_ running sum value to equal the sum of every current interaction contribution column added to the _previous_ running sum value. + (*Boundary, last row*) Constrain last row of running sum column to equal the claimed table contribution.

Total constraints: 2 boundary + 1 transition over `numRows_i - 1` rows.

#### Choice 2: transitions looking forward

tl,dr: explicit `0_ExtensionField` initial value, implicit final value.

+ (*Boundary, first row*) Constrain first row of running sum column to equal `0_ExtensionField`. + (*Transition, looking forward, applied to rows `0, dots, numRows_i - 2`*) For each row _other than the last_, constrain the _next_ running sum value to equal the sum of every current interaction contribution column added to the _current_ running sum value. + (*Boundary, last row*) Constrain last row of running sum column added to sum of last row of every interaction column to equal the claimed table contribution. (That is, the claimed table contribution is implicit in the last row of the table, but not written to last value of running sum column.)

Total constraints: 2 boundary + 1 transition over `numRows_i - 1` rows.

#### Choice 3: circular transitions looking back/forward

+ For each row, constrain the _current/next_ (wrapping to first on last if "next") running sum value to equal the sum of every current interaction contribution value added to the _previous/current_ (wrapping to last on first if "previous") running sum value added to claimed table contribution divided by `numRows_i`.

Total constraints: 1 _circular_ transition over `numRows_i` rows.

This single circular constraint checks that each row's contribution `s_(i,j)` is added to the running sum column, either in the current row's cell or in the next row's. In order to avoid boundary constraints, the look-back or peek-forward into the running sum column wraps around the beginning or end of the table.

This alone implies that difference between first and last row's values will be the table's overall real contribution `sum_j s_(i,j)`, which will be incompatible with the circularity of the constraint. Since boundary constraints are avoided, the way to check that `sum_j s_(i,j)` equals the claimed contribution `L_i` is to remove a fraction of `L_i` at each row in such a way that `L_i` is removed completely after summing all `numRows_i` rows; i.e., the constraint subtracts the public term `L_i / numRows_i` from the running sum at every row.

If the expected equality `sum_j s_(i,j) = L_i` holds, then the circularity of the constraint will also hold. ]

---

# Memory Argument

As part of fully proving the correct execution of a RISC-V program, the VM must ensure that memory reads and writes are consistent. That is, every byte read from some address corresponds to the byte that was last written to that address --- or the initial value if nothing has been written yet. We consider "memory" in a broad sense here: both RAM and the general purpose registers can be seen as instantiations of memory and are therefore handled simultaneously.

While RAM is byte addressed, we do choose to store registers as a `DWordWL` over two word addresses. ]

On a high level, we ensure memory consistency by an interacting system of reads and writes to a lookup argument, combined with an initialization and finalization scheme. The initialization and finalization schemes together ensure both that (1) the necessary preconditions for the lookup system are satisfied, and (2) the program is executed with the correct initial memory and register contents as specified by the ELF binary and the ISA.

## Memory types

A commonly made distinction of memory types is that of _read-only_ and _read-write_ memory, with the more restrictive read-only variant often allowing for more efficient solutions (be that regarding prover time, verifier time or proof size) via table lookup proofs. Naturally, the VM’s main memory and registers should be handled by a read-write system as the guest program/environment can issue instructions that write to memory. While there are some subsystems that can be modelled as read-only memory ---e.g., the program memory and instruction decoding--- we opt to integrate these into the proof system via chip interactions (relying on techniques derived from table lookup arguments). As such, we only concern ourselves with read-write memory, moving forward.

## Memory operations

Every memory operation has some conceptual attributes that are relevant to mention or discuss:

- The type of operation (read or write) - The memory address --- this is an address in the broad sense: main memory and registers have their own dedicated part of the unified address space. - The value being read from or written to the memory address - When the value was read or written, see the below paragraph

Since we will have to ensure that memory accesses are temporally consistent within the execution of the VM, we additionally consider a _timestamp_ for  every memory access, that should be strictly increasing. As such, it should never be possible for the system to generate accesses to the same address at identical timestamps. Multiple memory accesses can (and indeed will, consider e.g. register reads) occur in a single execution cycle of the VM, so we cannot use the cycle counter directly as timestamp for register accesses. We can, however, statically bound the maximal number of memory accesses made during a single execution by a granularity constant `k` and derive timestamps from the cycle counter. The `i`th possible memory access in cycle `c` will obtain as timestamp the value `k dot c + i`. For simplicity, we will always reserve a timestamp for every possible memory access, and leave the timestamp unused if an instruction does not use it.

For reasons of completeness (since temporal integrity as discussed below is a security necessity), we cannot deal with multiple accesses to the same address at identical timestamps. However, if multiple accesses are guaranteed to be independent (that is, to different addresses), they can still share a timestamp --- consider, e.g., the case of reading a word as 4 bytes with the `LW` load instruction. This property is already taken into account where possible in the design of the system. For instance, in the CPU chip, we can ensure that there are at most 3 memory accesses not guaranteed to be independent, so a timestamp granularity of 4 timestamps per cycle is enough. ]

## Permutation argument

We can conceptually organise the state of the memory as a collection of "tokens" that represent tuples `(serif("timestamp"), serif("address"), serif("value"))`, meaning the current value written to `serif("address")` is `serif("value")`, last written to memory at `serif("timestamp")`. Having exactly one value associated with any address will be ensured (see further down in this document) by the interaction of memory initialization, memory finalization, and the effects of memory operations.

Each memory operation will then do two things:

- Consume the current token in the memory - Emit a new token to replace it

Naturally, for a read operation, the _values_ embedded in the consumed and emitted tokens must be identical. From the need to consume a token even on the first memory access, we can see the necessity for a memory initialization procedure ---in addition to having to make sure the initial memory content lines up with what the binary dictates.

So long as we can properly constrain temporal integrity (that is, no memory operation can consume future tokens), this "balancing" act of tokens can be integrated (with sufficient domain separation) into the existing LogUp argument ([logup]): consuming a token corresponds to a "receive" and emitting a new token is a "send".

## Temporal integrity

To ensure temporal integrity, every memory operation needs to be constrained for the newly emitted token to have a strictly greater timestamp than the consumed token. This raises the question of how to represent timestamps and cleanly perform this check, as over a finite field the “less than” relation is ill-defined (though it is common and natural to consider it as the less than relation over the natural lift of the field into the integers). We choose to represent timestamps as machine words, using the existing `LT` chip ([lt]) functionality for comparisons. The full implementation of the timestamp system can be seen in the `timestamp` column of the `CPU` ([cpu]) and `MEMW` chips ([memw]). The `CPU` merely passes in the current timestamp, while `MEMW` can recall the previously written timestamp and constrain the correct sequencing.

- Clean definition of “less-than”, using the already existing `LT` functionality in the ALU - Harder to perform increments, needing extra constraints beyond field arithmetic - But this can be alleviated by providing a precomputed column that has a fixed increment per CPU row ][ - Comparison is more annoying, but can work by: - Decomposition into a machine word and chip interaction with the LT chip - Bit decomposition and comparison constraints - Range-checking the difference to be sufficiently small w.r.t. the field characteristic. - Increments and basic arithmetic operations are cheap ] ]

## Initialization and Finalization

Because the LogUp argument handling token consumption and emission needs to be fully balanced --- every token emitted should be consumed, and vice versa --- we need to have a system to emit the initial tokens and consume the final tokens. This needs to ensure that every address has at most a single initializing emission, and at most one finalizing consumption. Having at most one initialization will, through the correctness of the lookup argument, immediately lead to having at most one correct finalization, and vice versa.

The initialization will need to correspond to a fixed initial register state for the VM, as well as the memory loaded from the program binary, zero-initialization of memory elsewhere, and private input provided by the prover. The contribution of initialization with static data from the ELF executable and the initial register state to the sum can be handled directly by the verifier, ensuring correctness corresponding to the ELF binary being proven. This leaves only zero-initialization and prover input as prover-side concerns for initialization, alongside the finalization of the entire used memory.

For our chosen scheme (which we refer to as "paged initialization/finalization"), the available memory range is split into equally (power-of-two) sized "pages". Each address can then be represented as `address = page_base_address + page_offset`, with `page_base_address` being "page-aligned", and `page_offset` belonging to a limited range (the page size). As such, initialization or finalization of a page is represented by a table with columns `page`, `offset`, `value`, and ---for finalization--- `timestamp`. The `page` column is a preprocessed, constant value (which can be entirely virtualized/inlined into the constraints for this table), and the `offset` column is a preprocessed column containing its row index. Depending on the type of initialization, `value` can be a prover-committed column (input data), or a precomputed, constant column containing `0` (free memory space). This table then feeds into the LogUp system in the normal way, emitting the initial tokens for all addresses in a page, without consuming any tokens. Since the `offset` column is always the same, it can be reused across all paged initialization and finalization tables.

Concretely, each page gets an associated `PAGE` table, consisting of N variables over N columns. For each such table, the `page` variable is instantiated as the constant base address of the page. The `offset` column is preprocessed, which helps the verifier ensure that each page has a single fixed size, but the verifier should still check that no pages overlap and all `page` values are page-aligned.

### Page initialization

> **Note:** check whether we need `fini` to be range-checked

We present here a set of constraints on the `PAGE` table that

+ enforces the initial and final values of each address are bytes + adds the initial and final interaction to the LogUp argument

For zero-initialized pages, `init` can be a constant `0`, and hence doesn't need a column, nor a range check.

### Input

| Name | Type | Description |
|------|------|-------------|
| `offset` | `RowIndex` | The offset from the page base address. |
| `init` | `Byte` | The initial value of this address. Can be replaced by a constant zero for zero-initialization |
| `fini` | `Byte` | The final value this address took |
| `timestamp` | `DWordWL` | The timestamp at which this address was last accessed |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `address` | `DWordWL` | Adding `offset` to the page base address `page`. `page` is a constant with respect to a single instance of this table. |

**Definition of `address`:**
```
address := page + offset * 1::DWordWL
```

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `PAGE-C1` | `IS_BYTE[init]` | 1 |
| `PAGE-C2` | `IS_BYTE[fini]` | 1 |
| `PAGE-C3` | `memory[0, address, 0::DWordWL, init]` | -1 |
| `PAGE-C4` | `memory[0, address, timestamp, fini]` | 1 |

We identify a few alternatives that would achieve the desired initialization/finalization functionalities, and consider their respective trade-offs.

_"Free-zero" initialization_

Zero-initialization could be achieved by allowing the `MEMW` chip to output a zero without consuming a token from the lookup argument. This would in turn be made secure by finalization consuming at most one token per address: if an address is initialized more than once, the proof cannot be finalized. - This requires fewer pages (and hence tables) for zero-initialization. - But it comes at a cost of added complexity in the `MEMW `chip, and likely some extra columns to handle this. Keeping track of initialized addresses, and potentially having to initialize only some of the bytes in a word-read may make bookkeeping challenging. - This is an alternative form of sparse initialization (see below), so it is incompatible with paged finalization. Paged finalization can be made into a compatible sparse form by adding a bit-checked multiplicity column.

_Sparse initialization/finalization_

One or more STARK tables (depending on the amount of memory used) consisting of `(address, value)` columns are introduced, where for zero-initialization, `value` can be constant zero. Transition constraints ensure that `address` is strictly increasing, enforcing the "at most once" property; `value` is range-checked to consist of bytes. Similar to paged finalization, an additional `timestamp` column is added, containing the final timestamp each address was accessed. This table is then further used to contribute to the LogUp sum as with any other interactions. - The transition constraints can be chosen to only apply on finalization, as at-most-once finalization is enough to ensure consistency. - Sparse initialization is incompatible with paged finalization, see also the remark under free-zero initialization above. - This would require transition constraints, which currently are not needed elsewhere in the VM design - Additionally, for memory use exceeding the capacity of a single initialization/finalization table, some form of transition constraint between tables is needed - Alternatively, transition constraints could potentially be avoided by more integration into the LogUp system, but this could turn out more costly in practice - This is compatible with the above "free zero" initialization - Since a prover-committed address column is needed (rather than a precomputed one), the number of required columns increases. - As an optimization, the address column could potentially be used simultaneously for initialization and finalization - Sparse initialization/finalization reduces the cost for sparse memory access patterns, where only a few addresses would be accessed per page. Most programs and compilers should however favor a memory locality that makes paged initialization/finalization comparable. ]

### Register initialization/finalization

The initial and final state of registers can be entirely known by the verifier, since the relevant initialization values are either zero, or embedded in the ELF, and the final values can be set to a known value by the `HALT` ecall ([ecall]). As additionally, the number of registers is small, the verifier can directly add the required balancing terms to the LogUp sum.

## Notes and considerations

- Register reads and writes may interact within a single cycle, so a correct and fixed ordering needs to be ensured - Correctness of initialization and completeness of finalization need to be ensured

## Future topics of interest

- Optimize memory systems after determining factual bottlenecks (e.g. taking inspiration from Twist and Shout, or other recent research) - Double check whether IS_BYTE constraints are needed for fini

---

# Variables

While this VM operates on 64-bit words, the proving system's base field has fewer than `2^64` elements available and thus cannot represent all words natively. To this end, we introduce the concept of "variables" as an abstraction layer on top of the VM's field elements. The following table lists all variable types used in this VM.

columns: (auto, 1fr, auto), inset: 7pt, align: (top+left, top+left, top+center, ), table.header([*Name*], [*Description*], [*\*]), ..for type in config.variables.types { ([], [], [.subtypes.len()]) },

---

# Signatures

// Render a signature

let (lb, rb) = if sig.kind == "interaction" { (`[`, `]`) } else if sig.kind == "template" { (`<`, `>`) }

let cond = sig.at("cond", default: none) let cond_str = if cond != none { raw(cond) + ` => ` } else {``}

let input_str = sig.input.map(elt => { if type(elt) == array { raw(elt.at(0)) + `[` + raw(str(elt.at(1))) + `]` } else { raw(elt) } }).join(`, `)

let output = sig.at("output", default: none) let output_str = if output != none { if type(output) == array { raw(output.at(0)) + `[` + raw(str(output.at(1))) + `]` } else { raw(output) } + `; ` } else {``}

return [] }

// Compute the bus size of an interaction

let vars = sig.input + if "output" in sig { (sig.output, )} else {()}

return vars.map(v => { let (label, factor) = if type(v) == array { (v.at(0), v.at(1)) } else { (v, 1) } config.variables.types.filter(type => type.label == label).first().subtypes.len() * factor }) .sum() }

The following lists signatures of the .len() interactions in this VM.

columns: (1fr, auto), inset: 7pt, align: (top+left, center), stroke: none, table.header([*Signature*], [*Bus size*]), table.hline(stroke: 1pt), table.vline(stroke: 1pt, x: 1), ..for sig in interactions { ([], []) }, ))

Below, we list the signatures of the .len() templates in this VM.

columns: 1fr, inset: 7pt, align: (top+left, center), stroke: none, table.header([*Signature*]), table.hline(stroke: 1pt), ..for sig in templates { ([], ) }, ))

---

# IS_BIT Template

Barring exceptional cases, this template is used to assert that a variable of type `Bit` assumes a valid value under some condition.

## Variables

The  template operates on  variables:

### Input

| Name | Type | Description |
|------|------|-------------|
| `X` | `BaseField` | Value for which to assert that it lies in the range ${0, 1}$. |

### Condition

| Name | Type | Description |
|------|------|-------------|
| `cond` | `BaseField` | Whether the constraint should be applied ($eq.not 0$) or not ($0$). |

## Constraints

It takes only one constraint to enforce that `X` must be either `0` or `1` whenever ``cond` eq.not 0`:

| Tag | Description |
|-----|-------------|
| `IS_BIT-C1` | `cond` => `X` (1-`X`) = 0 |
| | _polynomial:_ `cond * X * (1 - X) = 0` |

*Note*: - In case of _unconditional_ template application, `cond` can be dropped from the constraint, simplifying it to ``X` (1- `X`) = 0`. - As described earlier, the `cond` variable must be describable by a degree-1 (i.e., linear) expression. This is to make sure that [isbit:c:isbit]'s expression has degree at most 3.

### Correctness argument

If `cond` is `0`, [isbit:c:isbit] is trivially satisfied: `X` can assume any value and the polynomial constraint will evaluate to `0` regardless. When ``cond` eq.not 0`, it follows that the statement can only be proven when ``X` (1-`X`) equiv 0 mod p`, with `p` the modulus of the field. Because `BaseField` is a prime field, this equality is only satisfied if either ``X` equiv 0 mod p` or `1-`X` equiv 0 mod p`. Hence, it is proven that when ``cond` eq.not 0`, [isbit:c:isbit] is only satisfied if ``X` in {0, 1}`.

---

# SIGN Template

It constrains that `sign` is set to `1` when both `X`'s most significant bit and `signed` are `1`, and `0` otherwise.

## Variables

The  template introduces  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `X` | `Half` | Value for which to extract its sign. |
| `signed` | `Bit` | Whether `X` represents a signed value (1) or not (0) |

### Output

| Name | Type | Description |
|------|------|-------------|
| `sign` | `Bit` | Sign of `X` |

## Assumptions

The  template operates on the following assumptions:

| Tag | Range | Description |
|-----|-------|-------------|
| `SIGN-A1` |  | `IS_BIT<signed>` |

If `sign` is set to `1`, `X` will be range-checked to be a halfword, and hence proving may fail if this is not ensured.

## Constraints

It takes only two constraints to compute the `sign` of `X`, given whether `X` represents a `signed` value or not. When ``signed` = 1`, the sign of `X` is equal to its most significant bit. This value is extracted in [sign:c:sign_if_signed]. If `X` is unsigned (i.e., ``signed` = 0`), its sign is always `0`. This is constrained by [sign:c:sign_if_unsigned].

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SIGN-C1` | `MSB16[sign; X]` | signed |
| `SIGN-C2` | not`signed` => `sign` = 0 |  |
| | _polynomial:_ `(1 - signed) * sign = 0` | |

---

# ADD/SUB Template

For ease of notation, we moreover introduce the  constraint template $

$ in both conditional and unconditional versions. It constrains that ``diff` equiv `lhs` - `rhs` (mod 2^64)` when the expression `cond` is non-zero.

## Variables

This template introduces  interaction(s).

### Input

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `DWordWL` | left-hand operator |
| `rhs` | `DWordWL` | right-hand operator |

### Output

| Name | Type | Description |
|------|------|-------------|
| `sum` | `DWordWL` | $`lhs` + `rhs`$ |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `carry` | `Bit[2]` | Carry values used to constrain the addition |

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * (lhs[0] + rhs[0] - sum[0])
carry (when iter=1) := 2^-32 * (lhs[1] + rhs[1] + carry[0] - sum[1])
```

### Condition

| Name | Type | Description |
|------|------|-------------|
| `cond` | `BaseField` | Whether the relation should be enforced ($eq.not 0$) or not ($0$). |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `ADD-A1.i` | i ∈ [0, 1] | `IS_WORD[lhs[i]]` |
| `ADD-A2.i` | i ∈ [0, 1] | `IS_WORD[rhs[i]]` |
| `ADD-A3.i` | i ∈ [0, 1] | `IS_WORD[sum[i]]` |

## Constraints

This template introduces the following constraints

| Tag | Range | Description |
|-----|-------|-------------|
| `ADD-C1.i` | i ∈ [0, 1] | cond ⇒ `IS_BIT<carry[i]>` |

---

# NEG Template

It requires `cond` to be a bit.

## Variables

This template introduces  interaction(s).

### Input

| Name | Type | Description |
|------|------|-------------|
| `x` | `DWordHL` | value to compute negation of |

### Output

| Name | Type | Description |
|------|------|-------------|
| `neg` | `DWordWL` | negation of `x` if $`cond` != 0$; unconstrained otherwise. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `carry` | `Bit[2]` | carries of the addition $`neg` + `x`$. |

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * ((x::DWordWL)[0] + neg[0])
carry (when iter=1) := 2^-32 * ((x::DWordWL)[1] + neg[1] + carry[0])
```

### Condition

| Name | Type | Description |
|------|------|-------------|
| `cond` | `Bit` | condition on whether to negate x |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `NEG-A1.i` | i ∈ [0, 3] | `IS_HALF[x[i]]` |
| `NEG-A2` |  | `IS_BIT<cond>` |

## Constraints

We constrain this equality using two constraints:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `NEG-C1` | `ZERO[1 - carry[0]; x[0] + x[1]]` | cond |
| `NEG-C2` | `ZERO[1 - carry[1]; x[0] + x[1] + x[2] + x[3]]` | cond |

### Correctness argument

The constraints force the `carry` values to be fixed. Writing `carry`'s definition, we then find that $

## cases(

2^32 - (`x as DWordWL`)_0 & "if" (`x as DWordWL`)_0 != 0, 0 & "if" (`x as DWordWL`)_0 = 0 ),\

2^32 - (`x as DWordWL`)_1 - 1 & "if" `x` != 0, 0 & "if" `x` = 0 $ Clearly, ``neg` = 0` when ``x` = 0` (and `cond` is set). For non-zero `x`, we distinguish two cases. When `(`x as DWordWL`)_0 = 0`, $

&= 2^32 dot `neg`_1 + `neg`_0\ &= 2^32 dot (2^32 - (`x as DWordWL`)_1) + 0\ &= 2^32 dot (2^32 - (`x as DWordWL`)_1) + (`x as DWordWL`)_0\ &= 2^64 - (2^32 dot (`x as DWordWL`)_1 + (`x as DWordWL`)_0)\ &= 2^64 - `x`\ &equiv -x mod 2^64, $ while when `(`x as DWordWL`)_0 != 0`, $

&= 2^32 dot `neg`_1 + `neg`_0\ &= 2^32 dot (2^32 - (`x as DWordWL`)_1 - 1) + (2^32 - (`x as DWordWL`)_0)  \ &= 2^64 - 2^32 dot (`x as DWordWL`)_1 - 2^32 + 2^32 - (`x as DWordWL`)_0  \ &= 2^64 - ((`x as DWordWL`)_0 + 2^32 dot (`x as DWordWL`)_1) \ &= 2^64 - `x`\ &equiv -x mod 2^64 $ when `cond` is set. When `cond` is not set, the two lookups are not executed, allowing `neg` to take any value in either case.

It is worth noting that this construction does _not_ require the limbs of `neg` to be range checked, thus allowing it be represented by the unrangecheckable `DWordWL` rather than a `DWordHL`. The input value `x` is still assumed to be range-checked, however. ]

---

# MEMW Chip

The  chip is used to read and write memory locations (both RAM and registers) in chunks of 1, 2, 4 or 8 values. It introduces the old value and last-accessed timestamps of memory addresses internally, in order to satisfy the design of the memory argument ([memory]).

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `is_register` | `Bit` | Whether the address represents a register index |
| `base_address` | `DWordWL` | The base address to read from/write to. Gets offset by $[0, 7]$ depending on the size of the access |
| `value` | `BaseField[8]` | The values to store in memory. For RAM, these should be (up to) 8 range-checked `Byte`s; registers are stored as two range-checked `Word`s |
| `timestamp` | `DWordWL` | The timestamp at which this memory access occurs |
| `write2` | `Bit` | Whether to write exactly 2 values |
| `write4` | `Bit` | Whether to write exactly 4 values |
| `write8` | `Bit` | Whether to write exactly 8 values |

### Output

| Name | Type | Description |
|------|------|-------------|
| `old` | `BaseField[8]` | The old value written at `base_address`. See `value` for information about representation. Only the elements corresponding to the `writeN` bits are guaranteed |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `carry` | `Bit[7]` | Whether `base_address[0] + i + 1` $>= 2^32$ |
| `old_timestamp` | `DWordWL[8]` | The timestamp at which address `base_address + i` was last accessed |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `w2` | `Bit` | writing at least 2 bytes |
| `w4` | `Bit` | writing at least 4 bytes |
| `address_add` | `DWordWL[7]` | `address_add[i] = base_address + i + 1` |
| `μ_sum` | `Bit` |  |

**Definition of `w2`:**
```
w2 := write2 + write4 + write8
```

**Definition of `w4`:**
```
w4 := write4 + write8
```

**Definition of `address_add`:**
```
address_add := ['arr', ['-', ['+', ['idx', 'base_address', 0], 'i', 1], ['*', ['^', 2, 32], ['idx', 'carry', 'i']]], ['+', ['idx', 'base_address', 1], ['idx', 'carry', 'i']]]
```

**Definition of `μ_sum`:**
```
μ_sum := μ_read + μ_write
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ_read` | `Bit` | Whether we are performing a read (and hence return `out`) |
| `μ_write` | `Bit` | Whether we are performing a write (and hence not return `out`) |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `MEMW-A1.i` | i ∈ [0, 1] | `IS_WORD[base_address[i]]` |
| `MEMW-A2` |  | `IS_BIT<write2>` |
| `MEMW-A3` |  | `IS_BIT<write4>` |
| `MEMW-A4` |  | `IS_BIT<write8>` |
| `MEMW-A5` |  | `IS_BIT<write2 + write4 + write8>` |
| `MEMW-A6.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

Our assumptions do not explicitly cover any range checks for the `is_register` and `value` columns, as these are not necessary for the correctness of this chip in isolation. Still, these properties are necessary for the consistency of the system as a whole, and therefore we document it here, keeping the type information as a reading help.

## Constraints

Depending on the values of `write2`, `write4` and `write8`, the addresses following `base_address` need to be constructed. Rather than computing these in full (which would require the later addresses to be instantiated), it suffices to know the `carry`: the bit indicating whether ``base_address`_0 + t >= 2^32`, i.e., whether adding `t in [1, 7]` to `base_address` requires a carry from the lower to the upper limb. Note that it is safe for the prover to chose these bits: additions for which this bit is not correctly set will yield an address where either the lower or upper limb is out of bounds. As such, the constructed address will not match any existing memory tokens, which are only initialized for correctly formatted and range-checked doublewords (see [memory]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW-C1` |  | `IS_BIT<μ_read>` |  |
| `MEMW-C2` |  | `IS_BIT<μ_write>` |  |
| `MEMW-C3` |  | `IS_BIT<μ_sum>` |  |
| `MEMW-C4` |  | `w2` => `μ_sum` |  |
| | | _polynomial:_ `w2 * (1 - μ_sum) = 0` | |
| `MEMW-C5.i` | i ∈ [0, 6] | `IS_BIT<carry[i]>` |  |
| `MEMW-C6` |  | `LT[1; old_timestamp[0], timestamp, 0]` | μ_sum |
| `MEMW-C7` |  | `LT[1; old_timestamp[1], timestamp, 0]` | w2 |
| `MEMW-C8.i` | i ∈ [2, 3] | `LT[1; old_timestamp[i], timestamp, 0]` | w4 |
| `MEMW-C9.i` | i ∈ [4, 7] | `LT[1; old_timestamp[i], timestamp, 0]` | write8 |

As long as `timestamp` is properly range-checked, the presence of `old_timestamp` in the memory argument automatically ensures it is appropriately range checked (this assumes no external entities provide negative multiplicities without range checking the timestamp). This ensures the assumptions for `LT` are satisfied.

There is no need to check that the additions do not overflow, as our address calculations are not performed modulo `2^64` here, and any overflow will result in an address without matching initialization.

The chip adds the following tuples to the lookup argument, to effectuate that part of the memory argument.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW-CM10` |  | `memory[is_register, base_address, old_timestamp[0], old[0]]` | μ_sum |
| `MEMW-CM11` |  | `memory[is_register, base_address, timestamp, value[0]]` | -μ_sum |
| `MEMW-CM12` |  | `memory[is_register, address_add[0], old_timestamp[1], old[1]]` | w2 |
| `MEMW-CM13` |  | `memory[is_register, address_add[0], timestamp, value[1]]` | -w2 |
| `MEMW-CM14.i` | i ∈ [2, 3] | `memory[is_register, address_add[i - 1], old_timestamp[i], old[i]]` | w4 |
| `MEMW-CM15.i` | i ∈ [2, 3] | `memory[is_register, address_add[i - 1], timestamp, value[i]]` | -w4 |
| `MEMW-CM16.i` | i ∈ [4, 7] | `memory[is_register, address_add[i - 1], old_timestamp[i], old[i]]` | write8 |
| `MEMW-CM17.i` | i ∈ [4, 7] | `memory[is_register, address_add[i - 1], timestamp, value[i]]` | -write8 |

This chip contributes the following to the lookup argument:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW-CO18` | `MEMW[old; is_register, base_address, value, timestamp, write2, write4, write8]` | -μ_read |
| `MEMW-CO19` | `MEMW[is_register, base_address, value, timestamp, write2, write4, write8]` | -μ_write |

## Padding

The table can be padded to the next power of two with the following value assignments:

## Read-size aligned fast path

When a memory access happens at an address with proper alignment for its access size (i.e., adding the access size to `base_address`'s lowest limb does not overflow), and all accessed elements were last accessed at the same timestamp, we can instead use the  chip to save on total column count. The saving comes from only requiring a single old timestamp to be stored, as well as being able to guarantee that all values of `add_limb_overflow` would be zero. A minor extra cost is introduced in the form of a check that the alignment is indeed correct, and the corresponding decomposition of the `base_address`.

Further logic remains essentially the same, so we briefly present the relevant tables for this chip.

The  chip only needs  variables, expressed through  columns; it leverages  interactions.

### Input

| Name | Type | Description |
|------|------|-------------|
| `is_register` | `Bit` | Whether the address represents a register index |
| `base_address` | `DWordWHH` | The base address to read from/write to. Gets offset by $[0, 7]$ depending on the size of the access |
| `value` | `BaseField[8]` | The values to store in memory. For regular memory, these should be (up to) 8 range-checked `Byte`s; registers are stored as two range-checked `Word`s |
| `timestamp` | `DWordWL` | The timestamp at which this memory access is said to occur |
| `write2` | `Bit` | Whether to write exactly 2 values |
| `write4` | `Bit` | Whether to write exactly 4 values |
| `write8` | `Bit` | Whether to write exactly 8 values |

### Output

| Name | Type | Description |
|------|------|-------------|
| `old` | `BaseField[8]` | The old value written at `base_address + i`. See `value` for information about representation. Only the elements corresponding to the `writeN` bits are guaranteed |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `old_timestamp` | `DWordWL` | The timestamp at which the address was last accessed |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `w2` | `Bit` | writing at least 2 bytes |
| `w4` | `Bit` | writing at least 4 bytes |
| `μ_sum` | `Bit` |  |

**Definition of `w2`:**
```
w2 := write2 + write4 + write8
```

**Definition of `w4`:**
```
w4 := write4 + write8
```

**Definition of `μ_sum`:**
```
μ_sum := μ_read + μ_write
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ_read` | `Bit` | Whether we are performing a read (and hence return `out`) |
| `μ_write` | `Bit` | Whether we are performing a write (and hence not return `out`) |

| Tag | Range | Description |
|-----|-------|-------------|
| `MEMW_A-A1.i` | i ∈ [0, 1] | `IS_HALF[base_address[i]]` |
| `MEMW_A-A2` |  | `IS_WORD[base_address[2]]` |
| `MEMW_A-A3` |  | `IS_BIT<write2>` |
| `MEMW_A-A4` |  | `IS_BIT<write4>` |
| `MEMW_A-A5` |  | `IS_BIT<write8>` |
| `MEMW_A-A6` |  | `IS_BIT<write2 + write4 + write8>` |
| `MEMW_A-A7.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW_A-C1` | `IS_HALF[base_address[0] + write2 + 3 * write4 + 7 * write8]` | μ_sum |
| `MEMW_A-C2` | `IS_BIT<μ_read>` |  |
| `MEMW_A-C3` | `IS_BIT<μ_write>` |  |
| `MEMW_A-C4` | `IS_BIT<μ_sum>` |  |
| `MEMW_A-C5` | `w2` => `μ_sum` |  |
| | _polynomial:_ `w2 * (1 - μ_sum) = 0` | |
| `MEMW_A-C6` | `LT[1; old_timestamp, timestamp, 0]` | μ_sum |

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW_A-CM7` |  | `memory[is_register, base_address::DWordWL, old_timestamp, old[0]]` | μ_sum |
| `MEMW_A-CM8` |  | `memory[is_register, base_address::DWordWL, timestamp, value[0]]` | -μ_sum |
| `MEMW_A-CM9` |  | `memory[is_register, base_address::DWordWL + 1::DWordWL, old_timestamp, old[1]]` | w2 |
| `MEMW_A-CM10` |  | `memory[is_register, base_address::DWordWL + 1::DWordWL, timestamp, value[1]]` | -w2 |
| `MEMW_A-CM11.i` | i ∈ [2, 3] | `memory[is_register, base_address::DWordWL + i::DWordWL, old_timestamp, old[i]]` | w4 |
| `MEMW_A-CM12.i` | i ∈ [2, 3] | `memory[is_register, base_address::DWordWL + i::DWordWL, timestamp, value[i]]` | -w4 |
| `MEMW_A-CM13.i` | i ∈ [4, 7] | `memory[is_register, base_address::DWordWL + i::DWordWL, old_timestamp, old[i]]` | write8 |
| `MEMW_A-CM14.i` | i ∈ [4, 7] | `memory[is_register, base_address::DWordWL + i::DWordWL, timestamp, value[i]]` | -write8 |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW_A-CO15` | `MEMW[old; is_register, base_address::DWordWL, value, timestamp, write2, write4, write8]` | -μ_read |
| `MEMW_A-CO16` | `MEMW[is_register, base_address::DWordWL, value, timestamp, write2, write4, write8]` | -μ_write |

### Padding

The table can be padded to the next power of two with the following value assignments:

## Register fast-path

The  chip provides a fast-path for accessing registers. This fast-path leverages that registers + can be addressed using a `Byte`, rather than a full `DWord`, + are constantly accessed, i.e., ``timestamp` - `old_timestamp`` is small, and + have a fixed access pattern to achieve a footprint that is significantly smaller than both  and .

Note: as a result of hard optimization, this chip can only be used for register accesses for which + ``timestamp` - `old_timestamp` in [1, 2^16]`, and + ``timestamp[0]` > `old_timestamp[0]`` If either of these rules does not apply to your access, you should fall back to using `MEMW_A`.

Note moreover that this chip does not guard against misaligned register access faults: to access register with a given `address`, one must provide `2 dot `address`` in the lookup.

### Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interactions:

### Input

| Name | Type | Description |
|------|------|-------------|
| `address` | `Byte` | address of the register being accessed |
| `timestamp` | `DWordWL` | timestamp at which the access takes place |
| `val` | `DWordWL` | value being written to this register |

### Output

| Name | Type | Description |
|------|------|-------------|
| `old` | `DWordWL` | value of this register at `old_timestamp`. |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `old_timestamp_lo` | `Word` | the lower limb of `old_timestamp` |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `old_timestamp` | `DWordWL` | timestamp at which this register was last accessed |
| `μ_sum` | `Bit` |  |

**Definition of `old_timestamp`:**
```
old_timestamp := ['arr', 'old_timestamp_lo', ['idx', 'timestamp', 1]]::DWordWL
```

**Definition of `μ_sum`:**
```
μ_sum := μ_read + μ_write
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ_read` | `Bit` | Whether we are performing a read (and hence return `out`) |
| `μ_write` | `Bit` | Whether we are performing a write (and hence not return `out`) |

### Assumptions

The following range checks are assumed to be performed/enforced outside of this chip:

| Tag | Range | Description |
|-----|-------|-------------|
| `MEMW_R-A1.i` | i ∈ [0, 1] | `IS_WORD[val[i]]` |
| `MEMW_R-A2.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

### Constraints

Since most registers are frequently accessed, the difference between `timestamp` and `old_timestamp` is small most of the times. Rather than storing their (nearly) identical upper limbs twice, it is instead assumed that ``old_timestamp[1]` = `timestamp[1]``;  can be used for accesses where this is not the case.

Verifying that ``timestamp` > `old_timestamp`` now simplifies to verifying that ``timestamp[0]` - `old_timestamp[0]` > 0`. For most accesses, this value will be small enough to fit in a `Half`. This chip thus enforces this by means of the following constraint:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW_R-C1` | `IS_HALF[timestamp[0] - old_timestamp[0] - 1]` | μ_sum |

With ``old_timestamp`<`timestamp`` asserted, `old` is read from the register ([regw:c:read_old]) and `val` is written back ([regw:c:write_val]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW_R-C2.i` | i ∈ [0, 1] | `memory[1, ['arr', ['cast', ['+', ['*', 2, 'address'], 'i'], 'Word'], 0], old_timestamp, old[i]]` | μ_sum |
| `MEMW_R-C3.i` | i ∈ [0, 1] | `memory[1, ['arr', ['cast', ['+', ['*', 2, 'address'], 'i'], 'Word'], 0], timestamp, val[i]]` | -μ_sum |

This chip can either just write (``μ_write` = 1`), or both read and write (``μ_read` = 1`) in the same cycle. It must be asserted that at most one of these two options is selected:

| Tag | Description |
|-----|-------------|
| `MEMW_R-C4` | `IS_BIT<μ_read>` |
| `MEMW_R-C5` | `IS_BIT<μ_write>` |
| `MEMW_R-C6` | `IS_BIT<μ_sum>` |

Lastly, this chip contributes the following interactions to the logup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW_R-C7` | `MEMW[['arr', ['idx', 'old', 0], ['idx', 'old', 1], 0, 0, 0, 0, 0, 0]; 1, ['arr', ['cast', ['*', 2, 'address'], 'Word'], 0], ['arr', ['idx', 'val', 0], ['idx', 'val', 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | -μ_read |
| `MEMW_R-C8` | `MEMW[1, ['arr', ['cast', ['*', 2, 'address'], 'Word'], 0], ['arr', ['idx', 'val', 0], ['idx', 'val', 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | -μ_write |

### Padding

The table can be padded to the next power of two with the following value assignments:

## Notes/optimizations

The following ideas may prove to be optimizations for the // chip: - `MEMB` chip that does a one-byte write to remove old_timestamp from here (uncertain tradeoffs) - Adding `μ_sum`/`w2`/`w4`/`write8` multiplicities to the `IS_HALF` lookups may make some GKR things faster if there are known zeroes. - For the register fast-path, one may upgrade the `IS_HALF` check to an `IS_B20` check for extended range at the cost of looking through a larger table.

---

# DECODE Table

All `RV64IMC` instruction are to be decoded to a format that can be interpreted by the VM. This section outlines the decoding table being used in the VM. For reasons of efficiency, data in this table is significantly compressed. Since reasoning about this compressed form is needlessly complex, the `decode (uncompressed)` section presents the same table in uncompressed form, and explains how to decode `RV64IM` assembly instructions to it. Instructions on how to compress the uncompressed table to form the compressed decode table, can be derived from the `packed_decode` variable provided below.

## Variables

The  table is comprised of  variables that are expressed using  columns:

### Output

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | value of the program counter this instruction is associated with. |
| `packed_decode` | `BaseField` | Ordered concatenation of several small variables. The `decode (uncompressed)` section explains the purpose of each variable.\ A list of each variable and the bit(-range) in which it is located:\ [0] `read_register1`, \ [1] `read_register2`, \ [2] `write_register`, \ [3] `memory_2bytes`, \ [4] `memory_4bytes`, \ [5] `memory_8bytes`, \ [6] `c_type`, \ [7] `signed`, \ [8] `mp_selector`, \ [9] `muldiv_selector`, \ [10] `word_instr`, \ [11] `ADD`, \ [12] `SUB`, \ [13] `SLT`, \ [14] `AND`, \ [15] `OR`, \ [16] `XOR`, \ [17] `SHIFT`, \ [18] `JALR`, \ [19] `BEQ`, \ [20] `BLT`, \ [21] `LOAD`, \ [22] `STORE`, \ [23] `MUL`, \ [24] `DIVREM`, \ [25] `ECALL`, \ [26] `EBREAK`; \ [27:35] `rs1`, \ [35:43] `rs2`, \ [43:51] `rd`, \ the remaining bits are set to zero.  |
| `imm` | `DWordWL` | the *fully extended (!)* 64-bit version of the immediate. |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` | The multiplicity with which this instruction is looked up in the `CPU` table. |

## Padding

The  table must be padded to a length that is a power of two. Empty rows with the following content can be added to achieve this:

Note that this row sets the `EBREAK` flag. Given that `CPU` asserts that `EBREAK = 0` (see [cpu:c:ebreak_traps]), using this "padding-instruction" would immediately make the CPU table unprovable. Note moreover that the `pc` is set to `7`. This value is the _smallest odd number_ (i.e., not reachable during regular execution) that is more than _`4`_ (i.e., the max `pc`-increment) greater than _`1`_ (i.e., the `pc`-value used in the [additional instruction] referred to by `CPU`-padding lines).

## Decoding

For the purposes of explaining decoding, we decompress 's `packed_decode` variable into its constituent variables. Note that the below table is _not_ used in practice: it is solely used for the purposes of this explanation.

### Output

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | value of the program counter this instruction is associated with. |
| `rs1` | `Byte` | index of source register 1. |
| `rs2` | `Byte` | index of source register 2. |
| `rd` | `Byte` | index of destination register. |
| `read_register1` | `Bit` | whether to load the contents of address `rs1` (1) or `0` (0) into `rv1`. |
| `read_register2` | `Bit` | whether to load the contents of address `rs2` (1) or `0` (0) into `rv2`. |
| `write_register` | `Bit` | whether the result should be written to `rd` ($=0$ for memory write and when $`rd` = `x0`$. |
| `mem_2B` | `Bit` | whether the memory access (read or write) touches exactly $2$ bytes. |
| `mem_4B` | `Bit` | whether the memory access (read or write) touches exactly $4$ bytes. |
| `mem_8B` | `Bit` | whether the memory access (read or write) touches exactly $8$ bytes. |
| `c_type` | `Bit` | Whether the instruction is of type `C`, i.e., whether it is $2$ bytes long instead of $4$. |
| `imm` | `DWordWL` | the *fully extended (!)* 64-bit version of the immediate. |
| `signed` | `Bit` | selector used to indicate signed or unsigned input interpretation. |
| `mp_selector` | `Bit` | Multi-purpose selector used by the CPU to to configure several ALU operations in different ways.            See the `CPU` chip for more details. |
| `muldiv_selector` | `Bit` | selects which output of `MUL` (lo/hi) or `DVRM` (quo/rem) is wanted. |
| `word_instr` | `Bit` | Whether the instruction is a `*W` instruction, requiring the inputs and outputs to be (sign) extended. |
| `ADD` | `Bit` | ALU selector flag |
| `SUB` | `Bit` | ALU selector flag |
| `SLT` | `Bit` | ALU selector flag |
| `AND` | `Bit` | ALU selector flag |
| `OR` | `Bit` | ALU selector flag |
| `XOR` | `Bit` | ALU selector flag |
| `SHIFT` | `Bit` | ALU selector flag |
| `JALR` | `Bit` | ALU selector flag |
| `BEQ` | `Bit` | ALU selector flag |
| `BLT` | `Bit` | ALU selector flag |
| `LOAD` | `Bit` | ALU selector flag |
| `STORE` | `Bit` | ALU selector flag |
| `MUL` | `Bit` | ALU selector flag |
| `DIVREM` | `Bit` | ALU selector flag |
| `ECALL` | `Bit` | ALU selector flag |
| `EBREAK` | `Bit` | ALU selector flag |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` | The multiplicity with which this instruction is looked up in the `CPU` table. |

We will illustrate how each instruction should be expressed in this (uncompressed) decoding table. The columns of the accompanying table represent the following: - *`operation`*: the assembly operation being encoded. - *`op-flag`*: which of the "`ALU` selector flags" operation flags to set. Each operation sets exactly one. - *`w_instr`*, *`signed`*: whether to set the `word_instr` and `signed` flags, respectively. - *other*: the other flags that should be set or variables that should be given specific values.

For the purpose of brevity and readability, the table uses the following rules-of-thumb: + `rd`, `rs1`, `rs2`, and `imm` are mapped to the values provided by the instruction; when a value is not specified by an instruction it defaults to `0`. + `read_register1`, `read_register2` and `write_register` are set to `1` when respectively ``rs1` != 0`, ``rs2` != 0`, or  ``rd` != 0`. + Any flag that is not listed is set to `0`, with the exception of the `c_type` flag. *The `c_type` flag is set independently of the below table*, as explained next.

Further clarification is provided in the notes following the table.

### C-type instructions

The `RV64C` extension for compressed instructions specifies that \~50% of all instructions can be represented using a 16-bit instruction (rather than 32-bits), saving \~25% in code size. This execution of assembly code is _not_ agnostic to an instruction's compression state; after executing a compressed instruction, the `pc` should be incremented by `2` rather than `4`. To indicate an instruction is provided in compressed form, the `c_type` flag is introduced. *This flag should be set to `1` whenever the decoded instruction is provided in compressed form and `0` otherwise.*

/// Add a reference to one or more notes following this table.

super("[" + refs.pos().map(r => ref(r)).join(",") + "]") }

show figure: set block(breakable: true)

figure(table( columns: (auto, auto, auto, auto, 1fr, auto), stroke: 0pt, inset: (right: .5em), align: (left, right, center, center, left, right), fill: (_, y) => // Overlay a low-opacity fill color to distinguish the different rows better if calc.odd(y) and y <= lines.len() { color.rgb(0, 0, 100, 20) } else { color.rgb(255, 255, 255, 20) }, table.header([*Operation*], [*op-flag*], [*`w_instr`*], [*`signed`*], [*other*], []), table.hline(stroke: 1.5pt), table.vline(x: 1, start: 1, end: lines.len() + 1, stroke: .5pt), ..lines.flatten(), table.hline(stroke: 1.5pt), table.footer([*Operation*], [*op-flag*], [*`w_instr`*], [*`signed`*], [*other*]), )) }

// OP-IMM ([`ADDI[W]   rd, rs1, imm`], [`ADD`], [`[W]`], [], [], []), ([`SLTI[U]   rd, rs1, imm`], [`SLT`], [], [.not`[U]`], [], []), ([`ANDI      rd, rs1, imm`], [`AND`], [], [], [], []), ([`ORI       rd, rs1, imm`], [`OR`],   [], [], [], []), ([`XORI      rd, rs1, imm`], [`XOR`], [], [], [], []), ([`SLLI[W]   rd, rs1, imm`], [`SHIFT`], [`[W]`], [], [], []), ([`SRLI[W]   rd, rs1, imm`], [`SHIFT`], [`[W]`], [], [`mp_selector`], []), ([`SRAI[W]   rd, rs1, imm`], [`SHIFT`], [`[W]`], [1], [`mp_selector`], []), // OP ([`ADD[W]    rd, rs1, rs2`], [`ADD`], [`[W]`], [], [], []), ([`SUB[W]    rd, rs1, rs2`], [`SUB`], [`[W]`], [], [], []), ([`SLT[U]    rd, rs1, rs2`], [`SLT`], [], [.not`[U]`], [], []), ([`AND       rd, rs1, rs2`], [`AND`], [], [], [], []), ([`OR        rd, rs1, rs2`], [`OR`], [], [], [], []), ([`XOR       rd, rs1, rs2`], [`XOR`], [], [], [], []), ([`SLL[W]    rd, rs1, rs2`], [`SHIFT`], [`[W]`], [], [], []), ([`SRL[W]    rd, rs1, rs2`], [`SHIFT`], [`[W]`], [], [`mp_selector`], []), ([`SRA[W]    rd, rs1, rs2`], [`SHIFT`], [`[W]`], [1], [`mp_selector`], []), // OP - M ([`MUL[W]    rd, rs1, rs2`], [`MUL`], [`[W]`], [1], [`mp_selector`], []), ([`MULH      rd, rs1, rs2`], [`MUL`], [], [1], [`mp_selector`, `muldiv_selector`], []), ([`MULHU     rd, rs1, rs2`], [`MUL`], [], [], [`muldiv_selector`], []), ([`MULHSU    rd, rs1, rs2`], [`MUL`], [], [1], [`muldiv_selector`], []), ([`DIV[U][W] rd, rs1, rs2`], [`DIVREM`], [`[W]`], [.not`[U]`], [], []), ([`REM[U][W] rd, rs1, rs2`], [`DIVREM`], [`[W]`], [.not`[U]`], [`muldiv_selector`], []), // LUI/AUIPC ([`LUI       rd, imm`], [`ADD`], [], [], [], []), ([`AUIPC     rd, imm`], [`ADD`], [], [], [`rs1 := x255`], []), ([`JAL       rd, imm`], [`JALR`], [], [], [`rs1 := x255`], []), // Branching ([`JALR      rd, rs1, imm`], [`JALR`], [], [], [], []), ([`BEQ      rs1, rs2, imm`], [`BEQ`], [], [], [], []), ([`BNE      rs1, rs2, imm`], [`BEQ`], [], [], [`mp_selector`], []), ([`BLT[U]   rs1, rs2, imm`], [`BLT`], [], [.not`[U]`], [], []), ([`BGE[U]   rs1, rs2, imm`], [`BLT`], [], [.not`[U]`], [`mp_selector`], []), // LOAD ([`LD        rd, rs1, imm`], [`LOAD`], [], [], [`mem_8B`], []), ([`LW[U]     rd, rs1, imm`], [`LOAD`], [], [.not`[U]`], [`mem_4B`], []), ([`LH[U]     rd, rs1, imm`], [`LOAD`], [], [.not`[U]`], [`mem_2B`], []), ([`LB[U]     rd, rs1, imm`], [`LOAD`], [], [.not`[U]`], [], []), // STORE ([`SD       rs1, rs2, imm`], [`STORE`], [], [], [`mem_8B`], []), ([`SW       rs1, rs2, imm`], [`STORE`], [], [], [`mem_4B`], []), ([`SH       rs1, rs2, imm`], [`STORE`], [], [], [`mem_2B`], []), ([`SB       rs1, rs2, imm`], [`STORE`], [], [], [], []), // ECALL/EBREAK ([`ECALL`], [`ECALL`], [], [], [``rs1` := `x17``], []), ([`EBREAK`], [`EBREAK`], [], [], [], []), // FENCE ([`FENCE`], [`ADD`], [], [], [], []),

// Construct a note that can be referenced through `lbl`

show figure: (it) => align(left, []) [ ] }

### Notes

We note the following about the above decoding table:

enum.item( referenceable_note( "note_word_instr", [`word_instr`: `[W]` indicates that ``word_instr` = 1` for the `W`-variant of the operation, and `0` for the non-`W`-variant.] ), enum.item( referenceable_note( "note_signed", [`signed`: .not`[U]` indicates that ``signed` = 1` for the *non-`U`*-variant of the operation, and `0` for the `U`-variant.] ), enum.item( referenceable_note( "note-lui", [`LUI`: this operation loads the 20-bit `imm` in the upper bits of `rd`. Observe that this can be represented using `ADDI rd, x0, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[12:32]` of `imm` and extending it to 64 bits.*] ), enum.item( referenceable_note( "note-auipc", [`AUIPC`: this operation adds the 20-bit immediate to the upper bits of `pc` and stores the result in `rd`. Given that the `pc` is stored in `x255`, this operation can be represented using `ADDI rd, x255, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[12:32]` of `imm` and extending it to 64 bits.*] ), enum.item( referenceable_note( "note-jal", [`JAL`: this operation stores ``pc` + 4` in `rd` and adds two times the sign-extended 20-bit immediate to the `pc`. Note that this can be represented using `JALR rd, x255, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[1:21]` of `imm` and extending it to 64 bits; the least significant bit should always be 0.*] ), enum.item( referenceable_note( "note-ecall", [`ECALL`: "On RISC-V a system call has its own instruction: `ECALL`. [...] A7 [= register `x17`] contains the system call number." [[source]] ] ), enum.item( referenceable_note( "note-fence", [`FENCE`: currently, the VM interprets this operation as `ADDI x0 x0 0`; a no-op.]

### One more instruction <cpu-padding-decode-row>

In addition to decoding all instructions provided in the ELF and adding a corresponding entry to the  table, one must include an entry that has ``pc` = 1` and every other variable set to `0`. Note that this will never conflict with any entry in the ELF, since it has an odd `pc` value.

This entry is used to pad the `CPU` table. More details on this matter are provided in the `CPU` chip.

---

# CPU Chip

The  chip coordinates memory accesses and dispatches to other chips for arithmetic and logical operations. It bases its decisions on the entry of the `DECODE` table ([decode]) corresponding the the current program counter (PC).

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `Timestamp` | A preprocessed timestamp to coordinate the memory argument. Since we have at most 3 non-disjoint memory accesses (`(rs1, rs2, rd)`, `(rs1, pc, pc)`, `(LOAD)` or `(STORE)`) a maximum of 4 slots is enough. |
| `pc` | `DWordWL` | The program counter |
| `rs1` | `Byte` | Source register 1 index |
| `rs2` | `Byte` | Source register 2 index |
| `rd` | `Byte` | Destination register index |
| `read_register1` | `Bit` | Whether to read from `rs1` (1) or to place a 0 in `rv1` (0) |
| `read_register2` | `Bit` | Whether to read from `rs2` (1) or to place a 0 in `rv2` (0) |
| `write_register` | `Bit` | Whether to write back to the destination register |
| `memory_2bytes` | `Bit` | Whether the memory access (read or write) touches exactly 2 bytes |
| `memory_4bytes` | `Bit` | Whether the memory access (read or write) touches exactly 4 bytes |
| `memory_8bytes` | `Bit` | Whether the memory access (read or write) touches exactly 8 bytes |
| `c_type_instruction` | `Bit` | Whether the instruction is of C type, i.e., whether it is 2 bytes long instead of 4 |
| `imm` | `DWordWL` | The fully extended 64-bit version of the immediate |
| `signed` | `Bit` | Indicates whether we're dealing with a signed or unsigned instruction |
| `mp_selector` | `Bit` | Multi-purpose selector used by different ALU operations for different purposes. Currently, it is used     - by the `MUL` chip to select between `MUL`/`MULH` and `MULH[S]U`, and     - as flag for inverting the condition of conditional branches (see `branch_cond`)     - as direction (left or right) for `SHIFT` |
| `muldiv_selector` | `Bit` | Selects which output of `MUL` (lo/hi) or `DIV` (quo/rem) is wanted |
| `word_instr` | `Bit` | Whether the instruction is a \*W instruction, requiring the inputs and outputs to be (sign) extended |
| `ADD` | `Bit` | One-hot ALU selector flag |
| `SUB` | `Bit` | One-hot ALU selector flag |
| `SLT` | `Bit` | One-hot ALU selector flag |
| `AND` | `Bit` | One-hot ALU selector flag |
| `OR` | `Bit` | One-hot ALU selector flag |
| `XOR` | `Bit` | One-hot ALU selector flag |
| `SHIFT` | `Bit` | One-hot ALU selector flag |
| `JALR` | `Bit` | One-hot ALU selector flag |
| `BEQ` | `Bit` | One-hot ALU selector flag |
| `BLT` | `Bit` | One-hot ALU selector flag |
| `LOAD` | `Bit` | One-hot ALU selector flag |
| `STORE` | `Bit` | One-hot ALU selector flag |
| `MUL` | `Bit` | One-hot ALU selector flag |
| `DIVREM` | `Bit` | One-hot ALU selector flag |
| `ECALL` | `Bit` | One-hot ALU selector flag |
| `EBREAK` | `Bit` | One-hot ALU selector flag |

### Output

| Name | Type | Description |
|------|------|-------------|
| `next_pc` | `DWordWL` | The program counter for the next instruction |
| `rvd` | `DWordWL` | The value to (maybe) be written back to rvd |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `rv1` | `DWordWHH` | The value of register `rs1` |
| `rv2` | `DWordWHH` | The value of register `rs2` |
| `rv1_ext_bit` | `Bit` | The sign bit of `rv1` if seen as a 32-bit word, used for sign extension with `word_instr` |
| `arg1` | `DWordBL` | The extended version of `rv1`, depending on `word_instr` |
| `rv2_ext_bit` | `Bit` | The sign bit of `rv2` if seen as a 32-bit word, used for sign extension with `word_instr` |
| `arg2` | `DWordBL` | A multiplexed version of `rv2` and `imm`, to be used as second argument to ALU calls |
| `res_ext_bit` | `Bit` | The sign bit of `res`, if seen as a 32-bit word, used for sign extension with `word_instr` |
| `res` | `DWordBL` | The ALU result |
| `is_equal` | `Bit` | Whether `rv1` and `arg2` are equal |
| `branch_cond` | `Bit` | Whether a branch is taken, i.e., the branch condition |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `packed_decode` | `BaseField` | A packed representation of all bit flags and register indices obtained from the decoding |
| `pad` | `Bit` | When no flags are set, we must be in a padding row. |

**Definition of `packed_decode`:**
```
packed_decode := 2^0 * read_register1 + 2^1 * read_register2 + 2^2 * write_register + 2^3 * memory_2bytes + 2^4 * memory_4bytes + 2^5 * memory_8bytes + 2^6 * c_type_instruction + 2^7 * signed + 2^8 * mp_selector + 2^9 * muldiv_selector + 2^10 * word_instr + 2^11 * ADD + 2^12 * SUB + 2^13 * SLT + 2^14 * AND + 2^15 * OR + 2^16 * XOR + 2^17 * SHIFT + 2^18 * JALR + 2^19 * BEQ + 2^20 * BLT + 2^21 * LOAD + 2^22 * STORE + 2^23 * MUL + 2^24 * DIVREM + 2^25 * ECALL + 2^26 * EBREAK + 2^27 * rs1 + 2^35 * rs2 + 2^43 * rd
```

**Definition of `pad`:**
```
pad := 1 - ADD - SUB - SLT - AND - OR - XOR - SHIFT - JALR - BEQ - BLT - LOAD - STORE - MUL - DIVREM - ECALL - EBREAK
```

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `CPU-A1` |  | At most one ALU selector flag is 1 by the decoding, and every other flag is 0. |
| `CPU-A2` |  | When `STORE + LOAD + BEQ + BLT = 0`, either `rs2 = 0` or `imm = 0` should be enforced by the decoding. This is needed for `arg2`. |

## Constraints

First, we perform a decoding lookup for the current PC.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-C1` | `DECODE[pc, imm, packed_decode]` | 1 |

### Range checks

> **Note:** Make sure we argue for every column here

> **Note:** is `rvd` still sufficiently constrained? (can also be done through the memory argument like `pc`?)

We constrain all columns to have the appropriate ranges. The flags and register indices looked up from the decoding need to be checked, as they are communicated through the interaction in a packed form. In contrast, we know ahead of time that decoding will ensure proper range checks for `pc` and `imm`. Similarly, since `next_pc` will propagate through the memory argument and be looked up in the instruction decoding on the next cycle, it is forced to be in the correct range. For the auxiliary columns, we need to check the limbs of `arg1`, `arg2`, and `res`. The ranges of the other auxiliary columns are enforced through later constraints.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CR2` |  | `IS_BIT<read_register1>` |  |
| `CPU-CR3` |  | `IS_BIT<read_register2>` |  |
| `CPU-CR4` |  | `IS_BIT<write_register>` |  |
| `CPU-CR5` |  | `IS_BIT<memory_2bytes>` |  |
| `CPU-CR6` |  | `IS_BIT<memory_4bytes>` |  |
| `CPU-CR7` |  | `IS_BIT<memory_8bytes>` |  |
| `CPU-CR8` |  | `IS_BIT<c_type_instruction>` |  |
| `CPU-CR9` |  | `IS_BIT<signed>` |  |
| `CPU-CR10` |  | `IS_BIT<mp_selector>` |  |
| `CPU-CR11` |  | `IS_BIT<muldiv_selector>` |  |
| `CPU-CR12` |  | `IS_BIT<word_instr>` |  |
| `CPU-CR13` |  | `IS_BIT<ADD>` |  |
| `CPU-CR14` |  | `IS_BIT<SUB>` |  |
| `CPU-CR15` |  | `IS_BIT<SLT>` |  |
| `CPU-CR16` |  | `IS_BIT<AND>` |  |
| `CPU-CR17` |  | `IS_BIT<OR>` |  |
| `CPU-CR18` |  | `IS_BIT<XOR>` |  |
| `CPU-CR19` |  | `IS_BIT<SHIFT>` |  |
| `CPU-CR20` |  | `IS_BIT<JALR>` |  |
| `CPU-CR21` |  | `IS_BIT<BEQ>` |  |
| `CPU-CR22` |  | `IS_BIT<BLT>` |  |
| `CPU-CR23` |  | `IS_BIT<LOAD>` |  |
| `CPU-CR24` |  | `IS_BIT<STORE>` |  |
| `CPU-CR25` |  | `IS_BIT<MUL>` |  |
| `CPU-CR26` |  | `IS_BIT<DIVREM>` |  |
| `CPU-CR27` |  | `IS_BIT<ECALL>` |  |
| `CPU-CR28` |  | `IS_BIT<EBREAK>` |  |
| `CPU-CR29` |  | `IS_BYTE[rs1]` | 1 |
| `CPU-CR30` |  | `IS_BYTE[rs2]` | 1 |
| `CPU-CR31` |  | `IS_BYTE[rd]` | 1 |
| `CPU-CR32.i` | i ∈ [0, 7] | `IS_BYTE[arg1[i]]` | 1 |
| `CPU-CR33.i` | i ∈ [0, 7] | `IS_BYTE[arg2[i]]` | 1 |
| `CPU-CR34.i` | i ∈ [0, 7] | `IS_BYTE[res[i]]` | 1 |

### ALU

The ALU functionality is then obtained through judicious dispatching to the corresponding chips.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CA35` |  | ADD + LOAD ⇒ `ADD<res::DWordWL; arg1::DWordWL, arg2::DWordWL>` |  |
| `CPU-CA36` |  | STORE ⇒ `ADD<res::DWordWL; arg1::DWordWL, imm>` |  |
| `CPU-CA37` |  | SUB + BEQ ⇒ `SUB<res::DWordWL; arg1::DWordWL, arg2::DWordWL>` |  |
| `CPU-CA38` |  | `LT[res[0]; arg1::DWordWL, arg2::DWordWL, signed]` | SLT + BLT |
| `CPU-CA39.i` | i ∈ [1, 7] | `SLT` + `BLT` => `res[i]` = 0 |  |
| | | _polynomial:_ `(SLT + BLT) * res[i] = 0` | |
| `CPU-CA40.i` | i ∈ [0, 7] | `AND_BYTE[res[i]; arg1[i], arg2[i]]` | AND |
| `CPU-CA41.i` | i ∈ [0, 7] | `OR_BYTE[res[i]; arg1[i], arg2[i]]` | OR |
| `CPU-CA42.i` | i ∈ [0, 7] | `XOR_BYTE[res[i]; arg1[i], arg2[i]]` | XOR |
| `CPU-CA43` |  | `SHIFT[res::DWordWL; arg1::DWordHL, arg2[0], mp_selector, signed, word_instr]` | SHIFT |
| `CPU-CA44` |  | JALR ⇒ `ADD<res::DWordWL; pc, (2 * c_type_instruction + 4 * (1 - c_type_instruction)) * 1::DWordWL>` |  |
| `CPU-CA45` |  | `MUL[res::DWordWL; arg1::DWordHL, signed, arg2::DWordHL, mp_selector, muldiv_selector]` | MUL |
| `CPU-CA46` |  | `DVRM[res::DWordWL; arg1::DWordHL, arg2::DWordHL, signed, muldiv_selector]` | DIVREM |

### Memory

The interactions with the memory, both for register loading and storing, as for `LOAD` and `STORE` instructions are handled. Note that since registers need no byte-addressing, we store them in the memory argument with `Word` limbs. The timestamps are ensured to be disjoint for disjoint memory locations. One consequence of that is that `next_pc` is written at `timestamp + 1` to ensure the access is disjoint with the `pc` read into `rv1` as part of the `AUIPC` instruction.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CM47` |  | `MEMW[['arr', ['idx', ['cast', 'rv1', 'DWordWL'], 0], ['idx', ['cast', 'rv1', 'DWordWL'], 1], 0, 0, 0, 0, 0, 0]; 1, 2::DWordWL * rs1, ['arr', ['idx', ['cast', 'rv1', 'DWordWL'], 0], ['idx', ['cast', 'rv1', 'DWordWL'], 1], 0, 0, 0, 0, 0, 0], timestamp + 0::DWordWL, 1, 0, 0]` | read_register1 |
| `CPU-CM48.i` | i ∈ [0, 2] | `!read_register1` => `rv1[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register1) * rv1[i] = 0` | |
| `CPU-CM49` |  | `MEMW[['arr', ['idx', ['cast', 'rv2', 'DWordWL'], 0], ['idx', ['cast', 'rv2', 'DWordWL'], 1], 0, 0, 0, 0, 0, 0]; 1, 2::DWordWL * rs2, ['arr', ['idx', ['cast', 'rv2', 'DWordWL'], 0], ['idx', ['cast', 'rv2', 'DWordWL'], 1], 0, 0, 0, 0, 0, 0], timestamp + 1::DWordWL, 1, 0, 0]` | read_register2 |
| `CPU-CM50.i` | i ∈ [0, 2] | `!read_register2` => `rv2[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register2) * rv2[i] = 0` | |
| `CPU-CM51` |  | `MEMW[1, 2::DWordWL * rd, ['arr', ['idx', 'rvd', 0], ['idx', 'rvd', 1], 0, 0, 0, 0, 0, 0], timestamp + 2::DWordWL, 1, 0, 0]` | write_register |
| `CPU-CM52` |  | `LOAD[rvd; res::DWordWL, timestamp + 0::DWordWL, memory_2bytes, memory_4bytes, memory_8bytes, signed]` | LOAD |
| `CPU-CM53` |  | `MEMW[0, res::DWordWL, arg2::Byte[8], timestamp + 1::DWordWL, memory_2bytes, memory_4bytes, memory_8bytes]` | STORE |
| `CPU-CM54` |  | `MEMW[['arr', ['idx', 'pc', 0], ['idx', 'pc', 1], 0, 0, 0, 0, 0, 0]; 1, (2 * 255)::DWordWL, ['arr', ['idx', 'next_pc', 0], ['idx', 'next_pc', 1], 0, 0, 0, 0, 0, 0], timestamp + 1::DWordWL, 1, 0, 0]` | 1 - pad |

### System

The interactions with the wider system.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CS55` | `!EBREAK` |  |
| | _polynomial:_ `1 - EBREAK = 0` | |
| `CPU-CS56` | `ECALL[timestamp, rv1::DWordWL]` | ECALL |

### Input and output to the ALU

We constrain `arg1`, `arg2` and `rvd` to correspond to the wanted values, including the appropriate sign/zero extension, depending on `word_instr`.

| Tag | Description |
|-----|-------------|
| `CPU-CE57` | `SIGN<rv1_ext_bit; rv1[1], word_instr>` |
| `CPU-CE58` | `arg1[:4]` = `rv1[:2]` |
| | _polynomial:_ `(arg1::DWordWL)[0] - (rv1::DWordWL)[0] = 0` |
| `CPU-CE59` | `arg1[4:]` = `rv1[2]` dot (1 - `word_instr`) + (2^(32) - 1) dot `rv1_ext_bit` dot `signed` |
| | _polynomial:_ `(arg1::DWordWL)[1] - (1 - word_instr) * rv1[2] - signed * rv1_ext_bit * (2^32 - 1) = 0` |
| `CPU-CE60` | `SIGN<rv2_ext_bit; rv2[1], word_instr>` |
| `CPU-CE61` | `arg2[:4]` = (1 - `LOAD`) dot `rv2[:2]` + (1 - `BEQ` - `BLT` - `STORE`) dot `imm[0]` |
| | _polynomial:_ `(arg2::DWordWL)[0] - (1 - LOAD) * (rv2::DWordWL)[0] - (1 - BEQ - BLT - STORE) * imm[0] = 0` |
| `CPU-CE62` | `arg2[4:]` = (1 - `LOAD`) dot ((1 - `word_instr`) dot `rv2[2]` + `signed` dot `rv2_ext_bit` dot (2^(32) - 1)) + (1 - `BEQ` - `BLT` - `STORE`) dot `imm[1]` |
| | _polynomial:_ `(arg2::DWordWL)[1] - (1 - LOAD) * (1 - word_instr) * rv2[2] - (1 - LOAD) * signed * rv2_ext_bit * (2^32 - 1) - (1 - BEQ - BLT - STORE) * imm[1] = 0` |
| `CPU-CE63` | `SIGN<res_ext_bit; (res::DWordHL)[1], word_instr>` |
| `CPU-CE64` | `!LOAD` => `rvd[0]` = `res[:4]` |
| | _polynomial:_ `(1 - LOAD) * (rvd[0] - (res::DWordWL)[0]) = 0` |
| `CPU-CE65` | `!LOAD` => `rvd[1]` = (1 - `word_instr`) dot `res[4:]` + `res_ext_bit` dot (2^(32) - 1) |
| | _polynomial:_ `(1 - LOAD) * (rvd[1] - (1 - word_instr) * (res::DWordWL)[1] - res_ext_bit * (2^32 - 1)) = 0` |

### Other constraints

For [cpu:c:is_equal], note that [cpu:c:sub] sets `res` to be the difference between `arg1` and `arg2` whenever `BEQ` is `1`. Given that this difference is `0` when both are equal, [cpu:c:is_equal] ensures `is_equal` is set to `1` if and only if ``arg1` = `arg2`` and `BEQ` is set.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CO66` | `ZERO[is_equal; res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7]]` | BEQ |
| `CPU-CO67` | `branch_cond` = `JALR` or (`BLT` and (`res` xor `invert`)) or (`BEQ` and (`is_equal` xor `invert`)) |  |
| | _polynomial:_ `-branch_cond + JALR + res[0] * (1 - mp_selector) * BLT + (1 - res[0]) * mp_selector * BLT + is_equal * (1 - mp_selector) * BEQ + (1 - is_equal) * mp_selector * BEQ = 0` | |
| `CPU-CO68` | `BRANCH[next_pc; pc, imm, arg1::DWordWL, JALR]` | branch_cond |
| `CPU-CO69` | `ADD<next_pc; pc, (2 * c_type_instruction + 4 * (1 - c_type_instruction)) * 1::DWordWL>` |  |

> **Note:** Document the choice to not have a multiplicity column here for padding

## Padding

The CPU can be padded with the following values, which have a corresponding row in the DECODE table, at the _odd_ address 1, only reachable through a HALT ecall.

This approach minimizes the number of dependent lookups, increasing only multiplicities in the DECODE table and the IS_BYTE lookup.

---

# SHIFT Chip

The  chip is designed to constrain that $

$ $

$ Here, `<<` and `>>` denote the _logical_ left and right shift operations, while `>>>` denotes the _arithmetic_ right shift operation.

## Variables

The `SHIFT` chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `in` | `DWordHL` | The value being shifted |
| `shift` | `Byte` | Number of bits to shift `in` by. |
| `direction` | `Bit` | Whether to shift left (0) or right (1). |
| `signed` | `Bit` | Whether to interpret `in` as a signed integer. |
| `word_instr` | `Bit` | Whether this is a Word-instruction (1) or not (0). |

### Output

| Name | Type | Description |
|------|------|-------------|
| `out` | `DWordWL` | $`in <</>>/>>>` (`shift` mod 32 dot (2 - `word_instr`))$ |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `is_negative` | `Bit` | Whether `in` is negative |
| `bit_shift` | `Byte` | Value by which to shift `in` to obtain `X` and `Y` |
| `zbs` | `Bit` | Whether `bit_shift` is zero (1) or not (0). |
| `X` | `Half[5]` | scratch variable. |
| `Y` | `Half[4]` | scratch variable. |
| `limb_shift_raw` | `Bit[3]` | One-hot vector indicating whether $floor.l `shift` / 16 floor.r equiv i mod s$, where $s = 2$ when $`word_instr` = 1$ and $4$ otherwise. These columns store the first 3 values, and the 4th is derived from the one-hot property. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `limb_shift` | `Bit[4]` |  |
| `extension` | `Half` | sign extension of `in`. |
| `left` | `Bit` | Whether to perform a left-shift. |
| `right` | `Bit` | Whether to perform a right-shift. |
| `intra_limb_left` | `DWordHL` | `in << (shift % 16)` if `left` |
| `intra_limb_right` | `DWordHL` | `in >>> (shift % 16)` if `right` and `signed`;\ `in >> (shift % 16)` if `right` and `!signed` |
| `shifted` | `DWordHL` | $`in <</>>/>>>` (`shift` mod 32 dot (2 - `word_instr`))$ |

**Definition of `limb_shift`:**
```
limb_shift (when iter=[0, 2]) := limb_shift_raw[i]
limb_shift (when iter=3) := 1 - Σ_j = 0^2 limb_shift_raw[j]
```

**Definition of `extension`:**
```
extension := 65535 * is_negative
```

**Definition of `left`:**
```
left := μ - direction
```

**Definition of `right`:**
```
right := direction
```

**Definition of `intra_limb_left`:**
```
intra_limb_left (when iter=0) := X[0]
intra_limb_left (when iter=[1, 3]) := X[i] + Y[i - 1]
```

**Definition of `intra_limb_right`:**
```
intra_limb_right := Y[i] + X[i + 1]
```

**Definition of `shifted`:**
```
shifted := left * Σ_j = 0^i limb_shift[j] * intra_limb_left[i - j] + right * (Σ_j = 0^3 - i limb_shift[j] * intra_limb_right[i + j] + extension * Σ_j = 4 - i^3 limb_shift[j])
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `SHIFT-A1.i` | i ∈ [0, 3] | `IS_HALF[in[i]]` |
| `SHIFT-A2` |  | `IS_BYTE[shift]` |
| `SHIFT-A3` |  | `IS_BIT<direction>` |
| `SHIFT-A4` |  | `IS_BIT<signed>` |
| `SHIFT-A5` |  | `IS_BIT<word_instr>` |

## Explanation

This chip has a rather complex design as a result of designing it to fit in as few columns possible. We briefly discuss the intricacies of the design, attempting to illustrate its correctness.

The chip's design revolves around a two-phase shifting process: 1. shift `in` by `x := `shift` mod 16` bits, 2. shift that result by `(`shift`-x) mod 64` (or `mod 32` if ` `word_instr` = 1`). The intermediate value representing the state between the two phases is stored in the scratch variables `X` and `Y`. The definition of `shifted` describes how one can combine the `X`, `Y` and `extension` variables to construct the output value as described using `Half`-limbs. The output variable `out` is equivalent to `shifted`, but expressed using `Word`-limbs.

In the following, we cover how these two phases were designed to complement one another. Here, we start with discussing the _logical_ left/right shift operations only; the modifications required to compute the _arithmetic_ right shift will be discussed at the end.

### First phase

We zoom in on the first step. Here, we make use of the lookup operation `HWSL` (short for "HalfWord Shift Left"): ` `HWSL[x: Half, y: B4]` := [(`x` `<<` `y`) mod 2^16, `x` `>>` (16 - `y`)]. ` One can use this to compute `out: Half[4] := in << y` as: $

$ as long as ``y` < 16`. Observing that ``HWSL[x,` 16-`y]`_0 = (`x` `<<` (16-`y`)) mod 2^16`, and ``HWSL[x,` 16-`y]`_1 = `x` `>>` `y`` for ``y` in [1, 15]`, one can also use it to compute `out := in >> y` as $

$ as long as `0 < `y` < 16`.

Observe now that the values being looked up are (almost) independent from the direction of the shift: only the shift-amount varies slightly. When we now define $

(16-`shift`) mod 16 & "when shifting right" ), $ it only takes some rearranging and combining of the values ``X[`i`] := HWSL[in[`i`], bit_shift]`_0` and ``Y[`i`] := HWSL[in[`i`], bit_shift]`_1` to form the limbs of ``in <</>> shift` mod 16`. In the remaining case that ``right` = 1` and ``shift` = 0 mod 16`, the limbs of ``in <</>> shift` mod 16` simply match those of `in`.

### Second phase

Since we're operating on 16-bit limbs, all the limbs in ``in <</>> shift`` must also occur somewhere in ``in <</>> shift` mod 16`. The number of full-limbs we still need to shift is determined by the fifth and sixth least significant bit of `shift`. With `limb_shift` containing a unary decoding of the integer represented by these two bits, we find that the intermediate value needs to be shifted over by `i` limbs (to the `left` or `right`) when ``limb_shift[`i`]` = 1`. These things combined yield `shifted`'s definition.

Of course, when ``word_instr` = 1` and, thus, only ``shift` mod 32` should be considered, the bit-mask for the lookup constraining `limb_shift` is adjusted appropriately (see [shift:c:limb_shift_lookup]).

### Arithmetic right shift

Lastly, we discuss the case of performing the _arithmetic_ right shift. Here, `extension` is constrained to contain a repetition of `in`'s most significant bit. Copies of this variable are used for any full limbs shifted in when ``right` = `signed` = 1`. Moreover, `X[4]` contains a copy of `extension` shifted over by the right number of bits, to allow the construction of ``in >>> shift` mod 16` as the appropriate intermediate.

## Constraints

First, we constrain `bit_shift` based on whether we are left or right-shifting. [shift:c:zbs] makes sure `zbs` is set to `1` if and only if `bit_shift = 0`. This flag is used to indicate the special case that ``right` = 1` and ``shift` = 0 mod 16`.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C1` | `AND_BYTE[bit_shift; shift, 15]` | left |
| `SHIFT-C2` | `AND_BYTE[bit_shift; 2^8 - 16 * zbs - shift, 15]` | right |
| `SHIFT-C3` | `ZERO[zbs; bit_shift]` | μ |

Next, we shift the limbs of `in` left and right by the appropriate amount, storing the results in `X` and `Y` respectively. When `zbs = 1`, the output cannot be used to compose ``in >>/>>> shift` mod 16`. To resolve this, we override `Y[i] := in[i]` and `X[i] := 0` in this case.

The case of `left`-shifting and ``bit_shift` = 0` will be used for padding rows. To prevent unnecessary lookups in padding rows, we override ``X[i]` := `in[i]`` and ``Y[i]` := 0` here.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C4.i` | i ∈ [0, 3] | `HWSL[['arr', ['idx', 'X', 'i'], ['idx', 'Y', 'i']]; in[i], bit_shift]` | 1 - zbs |
| `SHIFT-C5.i` | i ∈ [0, 3] | `zbs` => `X[i]` = `in[i]` dot `left` |  |
| | | _polynomial:_ `zbs * (X[i] - in[i] * left) = 0` | |
| `SHIFT-C6.i` | i ∈ [0, 3] | `zbs` => `Y[i]` = `in[i]` dot `right` |  |
| | | _polynomial:_ `zbs * (Y[i] - in[i] * right) = 0` | |
| `SHIFT-C7` |  | `HWSL[['arr', ['idx', 'X', 4], ['-', 'extension', ['idx', 'X', 4]]]; extension, bit_shift]` | 1 - zbs |
| `SHIFT-C8` |  | `zbs` => `X[4]` = 0 |  |
| | | _polynomial:_ `zbs * X[4] = 0` | |

### Full-limb shifting

Next, we constrain that `limb_shift` is a proper unary encoding of the fifth (and sixth if ``word_instr` = 0`) bit of `shift`. For this to be the case, three requirements must be satisfied: + *unary(0)*: ``limb_shift[`i`]` in {0, 1}` for `i in [0, 3]`, + *unary(1)*: ``limb_shift[`i`]` = 1` for exactly one `i`, and + *proper encoding*: ``limb_shift[`i`]` = 1 <=> 1/16 (`shift &` (48-32 dot `word_instr`)) = i` The first requirement is enforced by constraint [shift:c:limb_shift_is_bit]. To construct a constraint for the second and third requirement, observe that $ 1/16 dot (`shift &` (48-32 dot `word_instr`)) in cases( {0, 1, 2, 3} &"if" `word_instr` = 0, {0, 1} &"if" `word_instr` = 1 $ Observe moreover that, assuming *unary(0)*, the expression $ 1/16 dot (1 + sum_(i=0)^3 (16i-1) dot `limb_shift[`i`]`) $ can evaluate to `i` if and only if ``limb_shift[`i`]` = 1`, while the others are `0`. This means that the relation $ 1 + sum_(i=0)^3 (16i-1) dot `limb_shift[`i`]` = `shift &` (48-32 dot `word_instr`) $ enforces both *unary(1)* and *proper encoding*. This is the exact relation [shift:c:limb_shift_lookup] enforces.

Hereafter, one must only check that `out` is the proper cast of `shifted` into a `DWordWL`.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C9.i` | i ∈ [0, 3] | `IS_BIT<limb_shift[i]>` |  |
| `SHIFT-C10` |  | `AND_BYTE[(1 - limb_shift[0]) + 15 * limb_shift[1] + 31 * limb_shift[2] + 47 * limb_shift[3]; shift, 48 - 32 * word_instr]` | μ |
| `SHIFT-C11.i` | i ∈ [0, 1] | `out[:2]` = `shifted[:4]` |  |
| | | _polynomial:_ `out[i] - (shifted::DWordWL)[i] = 0` | |

### Miscellaneous

| Tag | Description |
|-----|-------------|
| `SHIFT-C12` | `direction` => `μ` = 1 |
| | _polynomial:_ `direction * (1 - μ) = 0` |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C13` | `MSB16[is_negative; in[3]]` | signed |

*Note*: `is_negative` is not used when `signed = 0`. As such, there is no problem with it being unconstrained in this case.

### Lookups

This chip adds the following interaction to the lookup.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C14` | `SHIFT[out; in, shift, direction, signed, word_instr]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

---

# BRANCH Chip

The  chip computes the target address of a branching instruction.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | The current pc, used as base address when `!JALR` |
| `offset` | `DWordWL` | The offset from the base address to jump to |
| `register` | `DWordWL` | The base address to use when `JALR` |
| `JALR` | `Bit` | Selects between `pc` and `register` as base address, needed for the `JALR` instruction |

### Output

| Name | Type | Description |
|------|------|-------------|
| `next_pc_high` | `Half[3]` | The upper part of the next pc |
| `next_pc_low` | `Byte[2]` | The lower part of the next pc |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `unmasked_low_byte` | `Byte` | The low byte of the next pc, before masking the LSB. Used to constraint the raw addition. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `next_pc_unmasked` | `DWordWL` | The combination of `next_pc_high`, `next_pc_low[1]` and `unmasked_low_byte` to constrain the addition. This is the computed value for the next pc, before masking off the LSB as required by the ISA. |
| `next_pc` | `DWordWL` | The computed next pc, after masking off the LSB as required by the ISA. |

**Definition of `next_pc_unmasked`:**
```
next_pc_unmasked (when iter=0) := 2^16 * next_pc_high[0] + 2^8 * next_pc_low[1] + unmasked_low_byte
next_pc_unmasked (when iter=1) := 2^16 * next_pc_high[2] + next_pc_high[1]
```

**Definition of `next_pc`:**
```
next_pc (when iter=0) := 2^16 * next_pc_high[0] + 2^8 * next_pc_low[1] + next_pc_low[0]
next_pc (when iter=1) := 2^16 * next_pc_high[2] + next_pc_high[1]
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `BRANCH-A1.i` | i ∈ [0, 1] | `pc` is range checked, `IS_WORD[pc[i]]` |
| `BRANCH-A2` |  | `offset` is range checked, `IS_WORD[offset]` |
| `BRANCH-A3.i` | i ∈ [0, 1] | `register` is range checked, `IS_WORD[register[i]]` |
| `BRANCH-A4` |  | `IS_BIT<JALR>` |

## Constraints

We constrain `next_pc` to be ``base_address` + `offset``, where `base_address` equals `pc` when ``JALR` = 0` and `register` otherwise.

The range checks on `unmasked_low_byte` and `next_pc_low[0]` are performed implicitly by the `AND_BYTE` lookup.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `BRANCH-C1` |  | 1 - JALR ⇒ `ADD<next_pc_unmasked; pc, offset::DWordWL>` |  |
| `BRANCH-C2` |  | JALR ⇒ `ADD<next_pc_unmasked; register, offset::DWordWL>` |  |
| `BRANCH-C3` |  | `IS_BYTE[next_pc_low[1]]` | μ |
| `BRANCH-C4` |  | `AND_BYTE[next_pc_low[0]; unmasked_low_byte, 254]` | μ |
| `BRANCH-C5.i` | i ∈ [0, 2] | `IS_HALF[next_pc_high[i]]` | μ |

This chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `BRANCH-C6` | `BRANCH[next_pc; pc, offset, register, JALR]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

---

# LT Chip

The  chip constrains an indicator bit for the less-than relation, signed or unsigned.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `DWordHHW` | The left operand |
| `rhs` | `DWordHHW` | The right operand |
| `signed` | `Bit` | whether to interpret `lhs` and `rhs` as signed integers (1) or not (0) |

### Output

| Name | Type | Description |
|------|------|-------------|
| `lt` | `Bit` | Whether $`lhs` < `rhs`$, taking `signed` into account |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `lhs_sub_rhs` | `DWordHL` | $`lhs` - `rhs`$ |
| `lhs_msb` | `Bit` | The most significant bit of `lhs` |
| `rhs_msb` | `Bit` | The most significant bit of `rhs` |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `carry` | `Bit[2]` | The carry for adding `lhs_sub_rhs` back to `rhs` |
| `unsigned_lt` | `Bit` | Whether $`lhs` < `rhs`$, as unsigned integers |

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * (rhs[0] + (lhs_sub_rhs::DWordWL)[0] - lhs[0])
carry (when iter=1) := 2^-32 * ((rhs::DWordWL)[1] + (lhs_sub_rhs::DWordWL)[1] + carry[0] - (lhs::DWordWL)[1])
```

**Definition of `unsigned_lt`:**
```
unsigned_lt := carry[1]
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

We assume the inputs `lhs`, `rhs` and `signed` are partially range checked.

| Tag | Range | Description |
|-----|-------|-------------|
| `LT-A1` |  | `IS_WORD[lhs[0]]` |
| `LT-A2` |  | `IS_WORD[rhs[0]]` |
| `LT-A3` |  | `IS_BIT<signed>` |

## Constraints

We first constrain that all variables correspond to their definition. For the defining constraint of `lt`, [lt:c:lt], observe that it is a choice between two options, depending on the input flag `signed`. In the case of unsigned comparison, we simply need `unsigned_lt`, indicating that a wraparound (carry bit) modulo `2^64` is needed to go from `rhs` to `lhs` via addition. For the case of signed comparison, we first need some case analysis.

We split `a < b` into four disjoint cases, conditioned on the sign of `a` and `b`. Recall that the sign of a number in two's complement can be read off from the MSB, being `1` for a negative number and `0` for a positive one. For this analysis, we denote the MSB of `a` as `A` and the MSB of `b` as `B`. The four disjoint cases then become:

+ `dash(A) and B and (a < b)` + `A and dash(B) and (a < b)` + `A and B and (a < b)` + `dash(A) and dash(B) and (a < b)`

The first case is evidently false, while the second case simplifies to `A and dash(B)`. For the third and fourth case, observe that when `A = B`, the `<` relation is preserved by the modular correspondence between `[-2^(31), 2^(31))` and `[0, 2^(64))`. Importantly, this modular correspondence is merely a reinterpretation of the bits or values of `a` and `b`, due to the representation in two's complement. Hence, we can introduce the value `C = `unsigned_lt``, that accurately represents the relation `a < b` when `A = B`.

Combining our three remaining cases, we obtain the boolean formula `A dash(B) or A B C or dash(A) dash(B) C`. Since the cases are disjoint, this can be computed with the binary-valued polynomial `P(A, B, C) = A (1 - B) + A B C + (1 - A) (1 - B) C`.

The polynomial `P` can be simplified to a total degree of two. We claim that the polynomial `Q(A, B, C) = A (1 - B) + A C + (1 - B) C` is, for the purposes of this chip, equivalent to `P`. An exhaustive check shows that `P(A, B, C) != Q(A, B, C)` only for the triple `(A, B, C) = (1, 0, 1)`. This is, however, impossible due to the correctness of `ADD`. In more detail, if we let `s` be the (range-checked) difference `a - b` (so the equivalent of the `lhs_sub_rhs` column), and `x'` denote the most significant word of a variable `x`, we need `c dot 2^32 + a' = b' + s' + `carry[0]``, by the definition of `carry`. However, the left hand side of this is at least `3 dot 2^31`, as `(A, C) = (1, 1)`, and the right hand side is at most `(2^31 - 1) + (2^32 - 1) + 1 = 3 dot 2^31 - 1`. Therefore, we can use `Q` to constrain `lt` when `signed = 1`.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LT-C1` | `MSB16[lhs_msb; lhs[2]]` | μ |
| `LT-C2` | `MSB16[rhs_msb; rhs[2]]` | μ |
| `LT-C3` | `lt` = `signed` dot (A (1 - B) + A C + (1 - B) C) + (1 - `signed`) dot `unsigned_lt` |  |
| | _polynomial:_ `lt - signed * (lhs_msb * (1 - rhs_msb) + lhs_msb * carry[1] + (1 - rhs_msb) * carry[1]) - (1 - signed) * unsigned_lt = 0` | |
| `LT-C4` | `IS_HALF[lhs[1]]` | μ |
| `LT-C5` | `IS_HALF[rhs[1]]` | μ |

And then we constrain the subtraction, taking care of the remaining range checking not yet covered by the assumptions or the `MSB16` lookup.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `LT-C6.i` | i ∈ [0, 1] | `IS_BIT<carry[i]>` |  |
| `LT-C7.i` | i ∈ [0, 3] | `IS_HALF[lhs_sub_rhs[i]]` | μ |

The chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LT-C8` | `LT[lt; lhs::DWordWL, rhs::DWordWL, signed]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

---

# MUL Chip

The  chip constrains multiplication, both signed and unsigned, as well as providing access to the low and high halfs of the multiplication result.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `DWordHL` | the left hand operator. |
| `lhs_signed` | `Bit` | whether to interpret `lhs` as a signed integer (1) or not (0). |
| `rhs` | `DWordHL` | the right hand operator. |
| `rhs_signed` | `Bit` | whether to interpret `rhs` as a signed integer (1) or not (0). |

### Output

| Name | Type | Description |
|------|------|-------------|
| `lo` | `DWordHL` | the lower limbs of the (extended) multiplication result |
| `hi` | `DWordHL` | the upper limbs of the (extended) multiplication result |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `lhs_is_negative` | `Bit` | whether `lhs` is negative (1) or not (0) |
| `rhs_is_negative` | `Bit` | whether `rhs` is negative (1) or not (0) |
| `raw_product` | `B51[4]` | raw multiplication output |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `lhs_ext` | `Half[8]` | sign-extended value of `lhs` |
| `rhs_ext` | `Half[8]` | sign-extended value of `rhs` |
| `res` | `QuadWL` | concatenation of `lo` and `hi`. |
| `carry` | `B20[4]` | carry values |
| `μ_sum` | `BaseField` | sum of multiplicies |

**Definition of `lhs_ext`:**
```
lhs_ext (when iter=[0, 3]) := lhs[i]
lhs_ext (when iter=[4, 7]) := 65535 * lhs_is_negative
```

**Definition of `rhs_ext`:**
```
rhs_ext (when iter=[0, 3]) := rhs[i]
rhs_ext (when iter=[4, 7]) := 65535 * rhs_is_negative
```

**Definition of `res`:**
```
res (when iter=[0, 1]) := (lo::DWordWL)[i]
res (when iter=[2, 3]) := (hi::DWordWL)[i - 2]
```

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * (raw_product[0] - res[0])
carry (when iter=[1, 3]) := 2^-32 * (raw_product[i] + carry[i - 1] - res[i])
```

**Definition of `μ_sum`:**
```
μ_sum := μ_lo + μ_hi
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ_lo` | `BaseField` |  |
| `μ_hi` | `BaseField` |  |

`mat(delim: , top; bottom)` }

## Assumptions

The following range checks are assumed to be performed/enforced outside of this chip:

| Tag | Range | Description |
|-----|-------|-------------|
| `MUL-A1.i` | i ∈ [0, 3] | `IS_HALF[lhs[i]]` |
| `MUL-A2.i` | i ∈ [0, 3] | `IS_HALF[rhs[i]]` |

## Constraints

### Overview

When `lhs` and `rhs` are _unsigned_ integers, computing their product `mod 2^128` comes down to evaluating $ (sum_(j=0)^3 2^(16j) dot `lhs`_j) dot (sum_(i=0)^3 2^(16i) dot `rhs`_i) mod 2^128. $ If `lhs` and `rhs` are signed instead, the computation remains nearly identical: based on their signs, one must either zero or one-extend `lhs` and `rhs` --- forming `lhs_ext` and `rhs_ext` respectively --- and compute their product `mod 2^128`: $ (sum_(j=0)^7 2^(16j) dot `lhs_ext`_j) dot (sum_(i=0)^7 2^(16i) dot `rhs_ext`_i) mod 2^128. $ where `lhs_ext` and `rhs_ext` are treated as _unsigned_ integers. Note that by setting the extension limbs of `lhs` and/or `rhs` to `0` when the integer is (i) unsigned or (ii) signed and non-negative, this second formula still applies. For the purposes of constraining the multiplication operation, we rewrite this formula as

$ &(sum_(j=0)^7 2^(16j) dot `lhs_ext`_j) dot (sum_(i=0)^7 2^(16i) dot `rhs_ext`_i) mod 2^128 \ &equiv sum_(j=0)^7 sum_(i=0)^7 2^(16(i+j)) dot `lhs_ext`_j dot `rhs_ext`_i mod 2^128 \ &stackrel(triangle, equiv) sum_(j=0)^7 sum_(i=0)^(7-j) 2^(16(i+j)) dot `lhs_ext`_j dot `rhs_ext`_i mod 2^128 \ &stackrel(square, equiv) sum_(j=0)^7 sum_(i=j)^(7) 2^(16i) dot `lhs_ext`_j dot `rhs_ext`_(i-j) mod 2^128 \ &stackrel(penta, equiv) sum_(i=0)^7 sum_(j=0)^(i) 2^(16i) dot `lhs_ext`_j dot `rhs_ext`_(i-j) mod 2^128 \ &equiv sum_(i=0)^3 sum_(k=0)^1 sum_(j=0)^(2i+k) 2^(16(2i+k)) dot `lhs_ext`_j dot `rhs_ext`_(2i+k-j) mod 2^128 \ &equiv sum_(i=0)^3 2^(32i) dot sum_(k=0)^1 2^(16k) dot sum_(j=0)^(2i+k) `lhs_ext`_j dot `rhs_ext`_(2i+k-j) mod 2^128 $ where at step - `triangle` we can ignore `i > 7-j`, since that makes `2^(16(i+j)) equiv 0 mod 2^128`, - `square` we rewrite the second summation such that `i` iterates from `j` to 7, rather than `0` to `7-j`, and - `penta` we swap the sums.

We let `raw_product` capture the second summation in this last formula (see [mul:c:raw_product]). By construction, ``raw_product`_i < 2^51` for all `i in [0, 3]`, far exceeding the 32-bits that fit in a single `Word`-limb. What remains then is to reduce each limb of `raw_product` `mod 2^32`, carrying the overflow of each limb to the next, constructing the output `res` in doing so.

This reduce-and-carry operation is constrained by [mul:c:range_lo]/[mul:c:range_hi] and [mul:c:carry], combined with `carry`'s definition. [mul:c:carry] and `carry`'s definition enforce that $ forall i in [0, 3]: `raw_product`_i + `carry`_(i-1) - `res`_i in { k dot 2^32 | k in [0, 2^20) } $ with ``carry`_(-1) = 0` for simplicity. In other words: ``res`_i equiv `raw_product`_i + `carry`_(i-1) (mod 2^32)`. With [mul:c:range_lo]/[mul:c:range_hi] forcing ``res`_i < 2^32`, ``res`_i` can only assume one value: ``raw_product`_i + `carry`_(i-1) mod 2^32`.

*Note*: one may have observed that [mul:c:carry] requires ``carry`_i in [0, 2^20)`, while no limb of a valid carry value would ever exceed `2^19`. This is indeed the case. However, there is some slack in how tight one has to constrain the `carry` values. In fact, in this situation it suffices to assert that ``carry`_i < frac(p, 2^32, style: "skewed") approx 2^31`, where `p` denotes the field's modulus. Given that other chips also use 20-bit lookups, using `IS_B20` makes for a simpler design.

### Definitions

We constrain `lhs_is_negative` and `rhs_is_negative` according to their definition; `lo`, `hi` and `carry` are appropriately range checked.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MUL-C1` |  | `SIGN<lhs_is_negative; lhs[3], lhs_signed>` |  |
| `MUL-C2` |  | `SIGN<rhs_is_negative; rhs[3], rhs_signed>` |  |
| `MUL-C3.i` | i ∈ [0, 3] | `IS_HALF[lo[i]]` | μ_sum |
| `MUL-C4.i` | i ∈ [0, 3] | `IS_HALF[hi[i]]` | μ_sum |
| `MUL-C5.i` | i ∈ [0, 3] | `IS_B20[carry[i]]` | μ_sum |

### Product

[mul:c:raw_product] defines `raw_product` in terms of the (sign extended) input values `lhs` and `rhs`.

| Tag | Range | Description |
|-----|-------|-------------|
| `MUL-C6.i` | i ∈ [0, 3] | `raw_product[i]` = sum_(`k`=0)^1 2^(16k) sum_(`j`=0)^(2i+k) `lhs_ext[j]` dot `rhs_ext[2i+k-j]` |
| | | _polynomial:_ `Σ_k = 0^1 2^(16 * k) * Σ_j = 0^2 * i + k lhs_ext[j] * rhs_ext[2 * i + k - j] - raw_product[i] = 0` |

### Lookup

The  chip contributes the following to the lookup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MUL-C7` | `MUL[lo::DWordWL; lhs, lhs_signed, rhs, rhs_signed, 0]` | -μ_lo |
| `MUL-C8` | `MUL[hi::DWordWL; lhs, lhs_signed, rhs, rhs_signed, 1]` | -μ_hi |

## Padding

The table can be padded to the next power of two with the following value assignments:

## Notes/optimizations

- `lo` and `hi` are stored in `DWordHL`s (rather than `DWordWL`s) because of their values being range checked. Since it is not required that both `μ_lo` and `μ_hi` are non-zero at the same time, one cannot safely assume their range to be checked elsewhere. - As an optimization, one might be able to use a `DWordWL` and `DWordHL` to store `lo` and `hi`, where one would decide which to store in which based on the multiplicities `μ_lo` and `μ_hi`; the value sent into the lookup could then be assumed range-checked by the other side of the relation. This optimization was not included at this moment because of its negative impact on the readability and verifiability of the chip.

---

# DVRM Chip

The  chip provides division and remainder functionality, both signed and unsigned.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `n` | `DWordHL` | The numerator |
| `d` | `DWordHL` | The denominator |
| `signed` | `Bit` | Whether to interpret the input as signed (1) or unsigned (0) integers. |

### Output

| Name | Type | Description |
|------|------|-------------|
| `q` | `DWordHL` | The quotient; $`n` / `d`$ rounded towards zero. |
| `r` | `DWordHL` | The remainder; $`n` - `q` `d`$. |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `div_by_zero` | `Bit` | Whether $`d`=0$. |
| `overflow` | `Bit` | Whether $`n` = -2^63$ and $`d`=-1$. |
| `abs_r` | `DWordWL` | Absolute value of `r`. |
| `abs_d` | `DWordWL` | Absolute value of `d`. |
| `n_sub_r` | `DWordHL` | $`n`-`r`$. |
| `sign_n_sub_r` | `Bit` | Sign of `n_sub_r`. |
| `sign_n` | `Bit` | Sign of `n`. |
| `sign_d` | `Bit` | Sign of `d`. |
| `sign_q` | `Bit` | Sign of `q`. |
| `sign_r` | `Bit` | Sign of `r`. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `extended_n` | `QuadHL` | sign-extended value of `n`. |
| `extended_r` | `QuadHL` | sign-extended value of `r`. |
| `extension_n_sub_r` | `DWordHL` | sign-extension limbs of `n_sub_r`. |
| `extended_n_sub_r` | `QuadHL` | sign-extended value of `n_sub_r`. |
| `carry` | `Bit[4]` | carries for adding `extended_n_sub_r` to `extended_r`, forming `extended_n`. |
| `μ_sum` | `BaseField` | sum of multiplicities |

**Definition of `extended_n`:**
```
extended_n (when iter=[0, 3]) := n[i]
extended_n (when iter=[4, 7]) := 65535 * sign_n
```

**Definition of `extended_r`:**
```
extended_r (when iter=[0, 3]) := r[i]
extended_r (when iter=[4, 7]) := 65535 * sign_r
```

**Definition of `extension_n_sub_r`:**
```
extension_n_sub_r := 65535 * sign_n_sub_r
```

**Definition of `extended_n_sub_r`:**
```
extended_n_sub_r (when iter=[0, 3]) := n_sub_r[i]
extended_n_sub_r (when iter=[4, 7]) := extension_n_sub_r[i - 4]
```

**Definition of `carry`:**
```
carry (when iter=0) := 2^-32 * ((extended_n_sub_r::QuadWL)[i] + (extended_r::QuadWL)[i] - (extended_n::QuadWL)[i])
carry (when iter=[1, 3]) := 2^-32 * ((extended_n_sub_r::QuadWL)[i] + (extended_r::QuadWL)[i] + carry[i - 1] - (extended_n::QuadWL)[i])
```

**Definition of `μ_sum`:**
```
μ_sum := μ_q + μ_r
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ_q` | `BaseField` |  |
| `μ_r` | `BaseField` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `DVRM-A1.i` | i ∈ [0, 3] | `IS_HALF[n[i]]` |
| `DVRM-A2.i` | i ∈ [0, 3] | `IS_HALF[d[i]]` |
| `DVRM-A3` |  | `IS_BIT<signed>` |

## Constraints

From the ISA, we gather five requirements for the `DIV[U][W]` and `REM[U][W]` instructions:

enum.item([ _For both signed and unsigned division, except in the case of_ overflow, _it holds that ``n` = `q` `d` + `r``._ ]), enum.item([ _`DIV` and `DIVU` perform [...] signed and unsigned integer division [...] rounding towards zero._ ]), enum.item([ _For `REM`, the sign of a nonzero [remainder] equals the sign of the [numerator]._ ]), enum.item([ In case of _division-by-zero_, ``r` = `n`` and ``q` = 2^64-1` (unsigned) or ``q` = -1` (signed). ]), enum.item([ In case of _overflow_, ``q` = `n`` and ``r` = 0` ]), where _overflow_ occurs when ``n` = -2^(63)` and ``d` = -1` (and, hence, ``signed` = 1`), and _division-by-zero_ indicates that ``d` = 0`. In the following, we list the constraints associated with the  chip, and explain how these together enforce all five of these requirements.

### R3: Sign remainder equals sign numerator

We start with R3, which is straightforwardly asserted by constraint [dvrm:c:sign_r_equals_sign_n].

| Tag | Description |
|-----|-------------|
| `DVRM-C1` | `r` eq.not 0 => `sign_r` = `sign_n` |
| | _polynomial:_ `Σ_i = 0^3 r[i] * (sign_r - sign_n) = 0` |

### R2: rounding towards zero

R2 states that "_[in] signed and unsigned integer division [the quotient is] round[ed] towards zero._" In other words, + the sign of ``n`-`qd`` must match that of `n` (unless ``qd` = `n``), and + `|`n`-`qd`|  < |`d`|` (unless ``d` = 0`).

Leveraging R1 , we can rewrite these as + the sign of ``r`` must match that of `n` (unless ``r` = 0`), and + `|`r`|  < |`d`|` (unless ``d` = 0`).

Focusing on the first statement, we observe that this trivially holds when ``signed` = 0`, while R3 deals with the case that ``signed` = 1`. The second statement is enforced by [dvrm:c:abs_r_lt_abs_d]. [dvrm:c:abs_r_if_negative] and [dvrm:c:abs_r_if_nonnegative] (resp. [dvrm:c:abs_d_if_negative] and [dvrm:c:abs_d_if_nonnegative]) are included to ensure that `abs_r` (resp. `abs_d`) is the absolute values of `r` (resp. `d`).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C2` |  | `LT[1 - div_by_zero; abs_r, abs_d, 0]` | μ_sum |
| `DVRM-C3` |  | sign_r ⇒ `NEG<abs_r; r>` |  |
| `DVRM-C4.i` | i ∈ [0, 1] | not`sign_r` => `abs_r` = `r` |  |
| | | _polynomial:_ `(1 - sign_r) * (abs_r[i] - (r::DWordWL)[i]) = 0` | |
| `DVRM-C5` |  | sign_d ⇒ `NEG<abs_d; d>` |  |
| `DVRM-C6.i` | i ∈ [0, 1] | not`sign_d` => `abs_d` = `d` |  |
| | | _polynomial:_ `(1 - sign_d) * (abs_d[i] - (d::DWordWL)[i]) = 0` | |

### R5: overflow

The ISA requires that ``q` = `n`` and ``r` = 0` in the event of overflow (i.e., when ``n` = -2^63` and ``d` = -1`). We note that the second half of this requirement is already satisfied by R2: since ``d` = -1 != 0`, R2 requires that `|`r`| < |`d`| = 1`, to which ``r` = 0` is the only satisfying value.

We moreover find that R1 can be leveraged to enforce the correct value of `q`. While ``n` = `qd` + `r`` (R1) does _not_ hold in the case of overflow, the relation ``n` = |`q`|`d` + `r`` _does_. We moreover note that the 64-bit _signed_ two's complement representation of `-2^63` is identical to the 64-bit _unsigned_ representation of `|-2^63| = 2^63`. As such, by interpreting `q` as an unsigned integer when ``overflow` = 1`, it follows that R1 will enforce ``q` = `0x80...00``.

In summary, in case of overflow R2 enforces that ``r` = 0`. Moreover it suffices to interpret `q` as unsigned integer ([dvrm:c:sign_q]); R1 will ensure it contains the correct value.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `DVRM-C7` | `sign_q` = `signed` dot (1- `overflow`) |  |
| | _polynomial:_ `signed * (1 - overflow) - sign_q = 0` | |
| `DVRM-C8` | `ZERO[overflow; n[0] + n[1] + n[2] + (n[3] - 2^15 * sign_n) + (1 - sign_n) + (65535 - d[0]) + (65535 - d[1]) + (65535 - d[2]) + (65535 - d[3])]` | μ_sum |

We highlight [dvrm:c:overflow]. Recall that the `overflow` flag should be set if and only if (i) ``signed` = 1`, (ii) ``n` = `0x80...00``, and (iii) ``d` = `0xFF...FF``. These requirements are equivalent to the state where: $ forall i in [0, 3]:&& 65535 - `d`_i &= 0,\ forall i in [0, 2]:&& `n`_i &= 0,\ && `n`_3 - 2^15 dot `sign_n` &= 0,\ && 1 - `sign_n` &= 0,\ $ where ``signed` = 1` follows from the last equality. The requirement is phrased in this way, because the left-hand sides of the above expressions are `>= 0` by construction. Given that the sum of these expressions does not exceed `2^19` (and thus never wraps in the field), we can now say that the `overflow` bit should be set to `1` if and only if their sum evaluates to `0`. The `ZERO` lookup guarantees this to be the case.

### R1: $#`n` = #`qd` + #`r`$

Rewriting R1, we find the constraint `not`overflow` => `n` - `r` = `qd``.

Since `n`, `d`, `q` and `r` are all 64-bit integers, we must assert this equality `mod 2^128`, rather than `mod 2^64`. To this end, we introduce `extended_n_sub_r` and leverage the `MUL` chip to verify that it is equal to ``qd` mod 2^128` using constraints [dvrm:c:mul_lower] and [dvrm:c:mul_upper]; [dvrm:c:q_range] is included to uphold assumption [mul:a:rhs].

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C9` |  | `MUL[n_sub_r::DWordWL; d, signed, q, sign_q, 0]` | μ_sum |
| `DVRM-C10` |  | `MUL[extension_n_sub_r::DWordWL; d, signed, q, sign_q, 1]` | μ_sum |
| `DVRM-C11.i` | i ∈ [0, 3] | `IS_HALF[q[i]]` | μ_sum |

It now remains to enforce that `extended_n_sub_r` is the _signed_ 128-bit representation of ``n`-`r``. Here, we introduce `extended_n` and `extended_r`. By their definition, these variables contain the signed 128-bit representations of `n` and `r`. The `carry` variable has been defined such that it mimics those in the `ADD` chip, except that here we add two `QuadHL`s rather than two `DWordHL`, thus needing four carry bits instead of two. With this in place, [dvrm:c:n_sub_r] (mimicking [add:c:carry]) ensures `extended_n_sub_r` must contain the correct value.

Lastly, observe that ``n` - `r` in (-2^64, 2^64)`, _regardless_ of the value of `signed`. Moreover, note that the upper halves of the 128-bit representations of all values in this range are either `0xFFFFFFFF` (negative) or `0x00000000` (non-negative). This means that we do not need to store all 128 bits of `extended_n_sub_r`. Rather, we need only store the lower 64-bits, and a separate bit (`sign_n_sub_r`) indicating whether the top limbs are all-ones or all-zeroes. The prover is free to select the value for `sign_n_sub_r`; only one of the two will fit the proof.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C12.i` | i ∈ [0, 3] | `IS_BIT<carry[i]>` |  |
| `DVRM-C13.i` | i ∈ [0, 3] | `IS_HALF[r[i]]` | μ_sum |
| `DVRM-C14.i` | i ∈ [0, 3] | `IS_HALF[n_sub_r[i]]` | μ_sum |
| `DVRM-C15` |  | `IS_BIT<sign_n_sub_r>` |  |

### R4: division-by-zero

R4 requires that ``q` = 2^64-1` (unsigned) or `-1` (signed) and ``r` = n` when ``d` = 0`. Recalling R1, we see that ``n` = `q` `d` + `r` = `r`` when ``d` = 0`, already enforces the latter. Next, we note that, in two's complement, the _unsigned_ value `2^64-1` and _signed_ value `-1` are both represented by the bit string `0xFFFFFFFF`. Hence, only [dvrm:c:q_if_div_by_zero] is required to completely constrain R4; [dvrm:c:div_by_zero] just ensures the `div_by_zero` flag is set when ``d` = 0`.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C16.i` | i ∈ [0, 3] | `div_by_zero` => `q[i]` = 65535 |  |
| | | _polynomial:_ `div_by_zero * (q[i] - 65535) = 0` | |
| `DVRM-C17` |  | `ZERO[div_by_zero; d[0] + d[1] + d[2] + d[3]]` | μ_sum |

### Other

The following constraints are included to enforce the values of `sign_n`, `sign_r` and `sign_d` are correct.

| Tag | Description |
|-----|-------------|
| `DVRM-C18` | `SIGN<sign_n; n[3], signed>` |
| `DVRM-C19` | `SIGN<sign_r; r[3], signed>` |
| `DVRM-C20` | `SIGN<sign_d; d[3], signed>` |

### Output

Lastly, this chip contributes the following to the lookup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `DVRM-C21` | `DVRM[q::DWordWL; n, d, signed, 0]` | -μ_q |
| `DVRM-C22` | `DVRM[r::DWordWL; n, d, signed, 1]` | -μ_r |

## Padding

To pad the  table, we use the following data, representing the unsigned division `frac(0, 0, style: "horizontal")`:

---

# LOAD Chip

The  chip provides functionality to read values from memory and sign-extend them where appropriate. It delegates low-level memory handling to the `MEMW` chip ([memw]).

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `base_address` | `DWordWL` | The base address to read/write from/to, gets offset by $[0, 7]$, depending on how big the access is |
| `timestamp` | `DWordWL` | The timestamp at which this memory access is said to occur |
| `read2` | `Bit` | Whether to read exactly 2 bytes |
| `read4` | `Bit` | Whether to read exactly 4 bytes |
| `read8` | `Bit` | Whether to read exactly 8 bytes |
| `signed` | `Bit` | Whether to sign-extend (1) or zero-extend (0) |

### Output

| Name | Type | Description |
|------|------|-------------|
| `res` | `DWordBL` | The result of reading (up to) 8 bytes from `base_address`, extended corresponding to `signed`. |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `sign_bit` | `Bit` | The sign bit extracted from the bytes retrieved from memory |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `read1` | `Bit` | Whether to read exactly 1 byte |

**Definition of `read1`:**
```
read1 := μ - read2 - read4 - read8
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `LOAD-A1.i` | i ∈ [0, 1] | `IS_WORD[base_address[i]]` |
| `LOAD-A2` |  | `IS_BIT<signed>` |
| `LOAD-A3` |  | `IS_BIT<read2>` |
| `LOAD-A4` |  | `IS_BIT<read4>` |
| `LOAD-A5` |  | `IS_BIT<read8>` |
| `LOAD-A6` |  | `IS_BIT<read2 + read4 + read8>` |
| `LOAD-A7.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

## Constraints

The chip delegates the actual memory interaction to the `MEMW` chip, and ensures correctness of the requested sign/zero extension. The output `res` is correctly range-checked as long as the memory contents are.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `LOAD-C1` |  | `read2` + `read4` + `read8` => `μ` |  |
| | | _polynomial:_ `(read2 + read4 + read8) * (1 - μ) = 0` | |
| `LOAD-C2` |  | `MEMW[res; 0, base_address, res::BaseField[8], timestamp, read2, read4, read8]` | μ |
| `LOAD-C3` |  | `MSB8[sign_bit; res[0]]` | read1 |
| `LOAD-C4` |  | `MSB8[sign_bit; res[1]]` | read2 |
| `LOAD-C5` |  | `MSB8[sign_bit; res[3]]` | read4 |
| `LOAD-C6.i` | i ∈ [4, 7] | !`read8` => `res`_i = `signed` dot `sign_bit` dot 255 |  |
| | | _polynomial:_ `(1 - read8) * (res[i] - signed * sign_bit * 255) = 0` | |
| `LOAD-C7.i` | i ∈ [2, 3] | !(`read4` + `read8`) => `res`_i = `signed` dot `sign_bit` dot 255 |  |
| | | _polynomial:_ `(1 - read4 - read8) * (res[i] - signed * sign_bit * 255) = 0` | |
| `LOAD-C8` |  | !(`read2` + `read4` + `read8`) => `res`_1 = `signed` dot `sign_bit` dot 255 |  |
| | | _polynomial:_ `(1 - read2 - read4 - read8) * (res[1] - signed * sign_bit * 255) = 0` | |

The chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LOAD-C9` | `LOAD[res::DWordWL; base_address, timestamp, read2, read4, read8, signed]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

---

# BITWISE Chips

The  chips deal with precomputed lookup tables for bitwise boolean operations and convenience functionalities over small domains.

## Variables

The  chip is comprised of  variables that are expressed using  columns. Of these, the _input_ and _output_ variables ( in total) are precomputed.

### Input

| Name | Type | Description |
|------|------|-------------|
| `X` | `Byte` |  |
| `Y` | `Byte` |  |
| `Z` | `B4` |  |

### Output

| Name | Type | Description |
|------|------|-------------|
| `AND` | `Byte` | the binary AND of `X` and `Y` |
| `OR` | `Byte` | the binary OR of `X` and `Y` |
| `XOR` | `Byte` | the binary XOR of `X` and `Y` |
| `MSB8` | `Bit` | the most significant bit of `X` |
| `MSB16` | `Bit` | the most significant bit of `Y` |
| `ZERO` | `Bit` | whether $`X` = 0$, $`Y` = 0$ and $`Z` = 0$. |
| `SLL` | `Half` | `X\|\|Y` logically left-shifted by `Z`: $((`X` + 256`Y`) `<<` `Z`) mod 2^16$ |
| `SLLC` | `Half` | `X\|\|Y` logically right-shifted by `Z`: $(`X` + 256`Y`) `>>` (16 - `Z`)$ |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ_AND` | `BaseField` |  |
| `μ_OR` | `BaseField` |  |
| `μ_XOR` | `BaseField` |  |
| `μ_MSB8` | `BaseField` |  |
| `μ_MSB16` | `BaseField` |  |
| `μ_ZERO` | `BaseField` |  |
| `μ_IS_BYTE` | `BaseField` |  |
| `μ_IS_HALF` | `BaseField` |  |
| `μ_IS_B20` | `BaseField` |  |
| `μ_HWSL` | `BaseField` |  |

*Note*: This table contains one row for every possible value of `(X, Y, Z)`. As such, it has length `2^8 dot 2^8 dot 2^4 = 2^(20)`.

## Lookup

This chip adds the following interactions to the lookup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `BITWISE-C1` | `AND_BYTE[AND; X, Y]` | -μ_AND |
| `BITWISE-C2` | `OR_BYTE[OR; X, Y]` | -μ_OR |
| `BITWISE-C3` | `XOR_BYTE[XOR; X, Y]` | -μ_XOR |
| `BITWISE-C4` | `MSB8[MSB8; X]` | -μ_MSB8 |
| `BITWISE-C5` | `MSB16[MSB16; X + 256 * Y]` | -μ_MSB16 |
| `BITWISE-C6` | `ZERO[ZERO; X + 256 * Y + 65536 * Z]` | -μ_ZERO |
| `BITWISE-C7` | `IS_BYTE[X]` | -μ_IS_BYTE |
| `BITWISE-C8` | `IS_HALF[X + 256 * Y]` | -μ_IS_HALF |
| `BITWISE-C9` | `IS_B20[X + 256 * Y + 65536 * Z]` | -μ_IS_B20 |
| `BITWISE-C10` | `HWSL[['arr', 'SLL', 'SLLC']; X + 256 * Y, Z]` | -μ_HWSL |

## Notes/Optimizations

The following ideas may prove to be optimizations for the  chip: + Extend `IS_BYTE[X]` to `ARE_BYTES[X, Y]`, such that two bytes are range checked at once. When only a single check is required, one can still execute `IS_BYTE[X] := ARE_BYTES[X, 0]`. + Drop `MSB8` column, and instead define the `MSB8` lookup as `MSB8<X> := MSB16[256X]`. Note: currently, `MSB8` also implicity range checks the input `X` (the lookup fails if `X` is not a `Byte`). This optimization should only be executed when all chips leveraging `MSB8` do _not_ need this implicit range check. + Place the 16-bit (`AND`, `OR`, `XOR`, `MSB16`, etc.) and 20-bit (`HWSL`, `IS_B20`, `ZERO`) lookups in separate tables.

---

# About ECALL

ECALLs provide system-level functionalities to the guest program.

When `ECALL` is executed, it is assumed that: - register `A7` contains the system call number

- the arguments are located in registers `A0`-`A6`, and - the return value is written to `A0`, where `A0`-`A7` are symbolic names for the registers `x10`-`x17`

---

# HALT Chip

## Variables

The  chip leverages  variable, spanning  columns and leverages  interactions:

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which to halt the program |

## Assumptions

It is assumed the input is range checked:

| Tag | Range | Description |
|-----|-------|-------------|
| `HALT-A1.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

## Constraints

The  chip: + makes sure register `x10` (containing the exit code) equals `0` ([halt:c:read_zero_exit_code]), + writes `0` to all other registers ([halt:c:zeroize_registers_lo]/[halt:c:zeroize_registers_hi]), and + sets `pc` equal to `1` ([halt:c:pc]). Note that the writes performed by all these interactions are accompanied by the timestamp `2^64-1`; the maximum timestamp. This prevents any other operation involving memory from being executed hereafter.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `HALT-C1.i` | i ∈ [1, 9] | `MEMW[1, (2 * i)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C2` |  | `MEMW[0::BaseField[8]; 1, (2 * 10)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C3.i` | i ∈ [11, 31] | `MEMW[1, (2 * i)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C4` |  | `MEMW[1, (2 * 255)::DWordWL, ['arr', 1, 0, 0, 0, 0, 0, 0, 0], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |

[ Observe that --- in its current state --- this solution puts the burden of verifying the register cleanup on the verifier inside of the lookup argument. Alternatively, one could add 31 lookups to the "memory" table to remove the _known_ final tokens for the registers there. ])

### Lookup

In this VM, halting is considered equivalent to executing a `sys_exit`. Hence, this chip responds to `ECALL`s with system call number 93.

The HALT chip therefore contributes the following interaction to the lookup-argument:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `HALT-C5` | `ECALL[timestamp, 93::DWordWL]` | -1 |

## Padding

This chip should only contain a single row. Given that `2^0 = 1`, this chip does not need to be padded. As such, no padding is defined.

---

# COMMIT Chip

## Variables

The  chip leverages  variables, spanning  columns and leverages  interactions:

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which to commit |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `index` | `BaseField` | Index of value being committed. |
| `address` | `DWordWL` | Address of first byte to commit. |
| `address_incr` | `DWordHL` | $`address` + 1$ |
| `count` | `DWordWL` | number of bytes to commit |
| `count_decr` | `DWordHL` | $`count` - 1$ |
| `first` | `Bit` | Whether this is the first commitment in this sequence. |
| `end` | `Bit` | Whether this is the end of the commitment sequence. |
| `value` | `Byte` | Byte stored at `address`. |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Constraints

In this VM, committing is considered equivalent to writing a value to `stdout`. Hence, this chip responds to `ECALL`s with system call number 64.

Since we do not know how many bytes are to be committed, this chip employs a recursive design: each iteration commits one byte, and recursively "calls" itself to commit the remaining bytes. As such, only the call from the CPU to this chip (i.e., the `first` in the recursion tree) should accept the `ECALL`; later recursive calls should not. This is why [commit:c:receive_ecall] has multiplicity `-`first``.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `COMMIT-C1` | `ECALL[timestamp, 64::DWordWL]` | -first |

The `write` operation --- writing to a file descriptor --- has the following signature:

```c ssize_t write(size_t count; int fd, const void buf[count], size_t count); ```

That is to say, - `A0` contains the file descriptor, - `A1` contains the address of `buf`'s first byte, - `A2` contains `count`, and - the written count should be written to `A0`.

[commit:c:read_address] reads `address` from `x11` (=`A1`) and [commit:c:read_count] reads `count` from `x12` (=`A2`). Since we only support writing to `stdout` (which corresponds to ``fd` = 1`

we assert that `x10` contains `1` in [commit:c:read_fd_write_count]. Note that this constraint _also_ writes `count` to `A0`; in this VM it is impossible for a commit to be interrupted or fail. Lastly, the `index` is read from `x254`; in the same operation, ``index` + `count`` is written back to this location by [commit:c:read_index]. This, too, leverages the fact that a commit will not be interrupted or fail to update the `index` for the next commit sequence. Again, each of these memory interactions only take place when this is the `first` call in the recursion tree.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `COMMIT-C2` | `MEMW[['arr', ['idx', 'address', 0], ['idx', 'address', 1], 0, 0, 0, 0, 0, 0]; 1, (2 * 11)::DWordWL, ['arr', ['idx', 'address', 0], ['idx', 'address', 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | first |
| `COMMIT-C3` | `MEMW[['arr', ['idx', 'count', 0], ['idx', 'count', 1], 0, 0, 0, 0, 0, 0]; 1, (2 * 12)::DWordWL, ['arr', ['idx', 'count', 0], ['idx', 'count', 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | first |
| `COMMIT-C4` | `MEMW[['arr', 1, 0, 0, 0, 0, 0, 0, 0]; 1, (2 * 10)::DWordWL, ['arr', ['idx', 'count', 0], ['idx', 'count', 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | first |
| `COMMIT-C5` | `MEMW[['arr', 'index', 0, 0, 0, 0, 0, 0, 0]; 1, (2 * 254)::DWordWL, ['arr', ['+', 'index', ['cast', 'count', 'BaseField']], 0, 0, 0, 0, 0, 0, 0], timestamp, 0, 0, 0]` | first |

*Note*: the observant reader will notice that [commit:c:read_index] casts `count` to a `BaseField`, potentiallly losing information. This is indeed correct. However, since it is practically impossible to commit more than `2^64-2^32` bytes in a single VM execution, it was decided to permit this.

Next, we read the `value` located at buffer address `address` and commit to it under the given `index`. This is only performed when we have not yet reached the `end` of the commit sequence.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `COMMIT-C6` | `MEMW[['arr', 'value', 0, 0, 0, 0, 0, 0, 0]; 0, address, ['arr', 'value', 0, 0, 0, 0, 0, 0, 0], timestamp, 0, 0, 0]` | μ - end |
| `COMMIT-C7` | `COMMIT[index, value]` | μ - end |

In parallel, we compute ``address_incr` = `address` + 1` ([commit:c:address_incr]) as address of the next byte to commit, and ``count_decr` = `count` - 1` ([commit:c:count_decr]) as the number of bytes that still has to be committed after committing this byte. [commit:c:range_address_incr] and [commit:c:range_count_decr] are included to satisfy [add:a:sum] respectively [add:a:rhs].

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `COMMIT-C8` |  | `ADD<address_incr::DWordWL; address, 1::DWordWL>` |  |
| `COMMIT-C9.i` | i ∈ [0, 3] | `IS_HALF[address_incr[i]]` | μ |
| `COMMIT-C10` |  | `SUB<count_decr::DWordWL; count, 1::DWordWL>` |  |
| `COMMIT-C11.i` | i ∈ [0, 3] | `IS_HALF[count_decr[i]]` | μ |

When `count` hits `0`, we should stop performing further recursive calls. We use the `end` bit to indicate these circumstances.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `COMMIT-C12` | `ZERO[end; (65535 - count_decr[0]) + (65535 - count_decr[1]) + (65535 - count_decr[2]) + (65535 - count_decr[3])]` | μ |

*Note*: + Rather than setting ``end` = 1` when ``count` = 0`, we do so when ``count_decr` = -1`. This technique allows `count` to be stored in a `DWordWL` rather than a `DWordHL`, saving two columns. + `forall i in [0, 3]: 65535 - `count_decr`_i >= 0` as a result of [commit:c:range_count_decr]. Hence, $ sum_(i=0)^3 65535 - `count_decr`_i = 0 arrow.l.r.double.long forall i in [0, 3]: `count_decr`_i = 65535 $

When this was not the `end` byte to commit in this recursion sequence, we recursively _Commit the Next Byte_ (`CNB`), specifying the timestamp, address to continue reading and the number of bytes that should still be committed ([commit:c:send_commit_next_byte]). Since that certainly won't be the `first` call in the sequence, we read `address_incr` and `count_decr` from the previous recursion level into `address` and `count` and continue executing the commit.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `COMMIT-C13` | `CNB[timestamp, index + 1, address_incr::DWordWL, count_decr::DWordWL]` | μ - end |
| `COMMIT-C14` | `CNB[timestamp, index, address, count]` | -(μ - first) |

Lastly, we must make sure `first`, `end` and `μ` are bits ([commit:c:range_first], [commit:c:range_end], [commit:c:range_mu]), and that when either ``first` = 1` or ``end` = 1` imply that ``μ` = 1` ([commit:c:first_or_end_implies_mu]). These are required to ensure the multiplicities `-(`μ` - `first`)` and ``μ` - `end`` are binary.

| Tag | Description |
|-----|-------------|
| `COMMIT-C15` | `IS_BIT<first>` |
| `COMMIT-C16` | `IS_BIT<end>` |
| `COMMIT-C17` | `IS_BIT<μ>` |
| `COMMIT-C18` | `first` + `end` => `μ` = 1 |
| | _polynomial:_ `(first + end) * (1 - μ) = 0` |

## Padding

To pad this chip, use the below data.

## Notes/optimizations

- The current version only supports writing to `stdout`. This chip could potentially be extended to support writing to arbitrary `fd`s - One might be able to replace [commit:c:end] by `end => count = 0`. While loosening the constraint (`count = 0 => end` is no longer enforced), this should not cause any problems: if the prover does not set `end` when `count=0`, they simply cannot complete the proof. First of all, one would have to recursively work through all `2^64` values of `count`, something that is practically infeasible. Moreover, if this is done with a sequence that originally has ``count` > 0`, one will inevitably have to read a memory address twice at the same timestamp, which is impossible to prove. In addition to dropping the `ZERO` lookup, this optimization might also permit moving `count_decr` from a `DWordHL` to a `DWordWL`, saving two columns. - Given that it is practically infeasible to commit more than ``p`-1 = 2^64-2^32` bytes in a program, it might suffice to store `count_decr` in a `BaseField`. Note that this would probably involve having an extra (virtual) column storing `count` in `BaseField` form as well. Moreover, one might need to add a lookup to `LT` to ensure ``count` <= `p`-1` when being read from memory at the beginning of each commitment sequence.

---

# SHA256 Accelerator

The following chips constitute an accelerator for the SHA256 compression function; other aspects of SHA256 hashing (such as repeated compression invocation, input padding and state initialization) fall outside the scope of this accelerator.

The base  chip provides the `ECALL` interface, interacts with memory and then delegates to the  and  chips to perform the message schedule and the compression rounds, respectively. The `SHA256_M` interaction signature is used to represent the output of the message schedule. The `SHA256_K` interaction signature is used to represent the `k` constants. It could either be instantiated with a (short) precomputed table, or through hardcoded LogUp contributions in this chip. For this exposition, we choose the former option, and present a table further below. Additionally, we introduce a  chip to perform the common action of computing the XOR of three rotations (or shifts) of a word.

Most of the structure and variable naming follows the pseudocode of the wikipedia page).

## `SHA256` chip

### Columns

The  chip leverages  variables, spanning  columns:

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | Timestamp at which the ECALL is invoked. Used as unique identifier for this invocation. |
| `h` | `Byte[32]` | The state of the hash function. |
| `h_addr` | `DWordHL[4]` | The addresses of the doublewords of `h` |
| `m` | `Byte[64]` | The input chunk. |
| `m_addr` | `DWordHL[8]` | The addresses of the doublewords of `m` |

### Output

| Name | Type | Description |
|------|------|-------------|
| `out` | `Byte[32]` | The new state. |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `last_round_out` | `Word[8]` | The output from the last compression round |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

### Constraints

The first responsibility of the chip is to read the current state and message chunk from memory, passed as arguments through pointers. Since the memory ranges could overlap, we read the chunk first (in [sha256:c:read_chunk], at timestamp `timestamp`), before reading and writing the state (in [sha256:c:read_state], at timestamp `timestamp + 1`). The addresses containing the state and the current chunk are passed in as arguments `A0 = x10` and `A1 = x11`, respectively. Note that following the SHA256 spec, this state and the chunks are read and written as big-endian.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHA256-C1` |  | `MEMW[['arr', ['idx', ['cast', ['idx', 'm_addr', 0], 'DWordWL'], 0], ['idx', ['cast', ['idx', 'm_addr', 0], 'DWordWL'], 0], 0, 0, 0, 0, 0, 0]; 1, (2 * 11)::DWordWL, ['arr', ['idx', ['cast', ['idx', 'm_addr', 0], 'DWordWL'], 0], ['idx', ['cast', ['idx', 'm_addr', 0], 'DWordWL'], 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | μ |
| `SHA256-C2.i` | i ∈ [0, 7], j ∈ [0, 3] | `IS_HALF[m_addr[i][j]]` | μ |
| `SHA256-C3.i` | i ∈ [1, 7] | `ADD<m_addr[i]::DWordWL; m_addr[0]::DWordWL, (8 * i)::DWordWL>` |  |
| `SHA256-C4.i` | i ∈ [0, 7] | `MEMW[['arr', ['idx', 'm', ['+', ['*', 8, 'i'], 3]], ['idx', 'm', ['+', ['*', 8, 'i'], 2]], ['idx', 'm', ['+', ['*', 8, 'i'], 1]], ['idx', 'm', ['+', ['*', 8, 'i'], 0]], ['idx', 'm', ['+', ['*', 8, 'i'], 7]], ['idx', 'm', ['+', ['*', 8, 'i'], 6]], ['idx', 'm', ['+', ['*', 8, 'i'], 5]], ['idx', 'm', ['+', ['*', 8, 'i'], 4]]]; 0, m_addr[i]::DWordWL, ['arr', ['idx', 'm', ['+', ['*', 8, 'i'], 3]], ['idx', 'm', ['+', ['*', 8, 'i'], 2]], ['idx', 'm', ['+', ['*', 8, 'i'], 1]], ['idx', 'm', ['+', ['*', 8, 'i'], 0]], ['idx', 'm', ['+', ['*', 8, 'i'], 7]], ['idx', 'm', ['+', ['*', 8, 'i'], 6]], ['idx', 'm', ['+', ['*', 8, 'i'], 5]], ['idx', 'm', ['+', ['*', 8, 'i'], 4]]], timestamp, 0, 0, 1]` | μ |
| `SHA256-C5` |  | `MEMW[['arr', ['idx', ['cast', ['idx', 'h_addr', 0], 'DWordWL'], 0], ['idx', ['cast', ['idx', 'h_addr', 0], 'DWordWL'], 1], 0, 0, 0, 0, 0, 0]; 1, (2 * 10)::DWordWL, ['arr', ['idx', ['cast', ['idx', 'h_addr', 0], 'DWordWL'], 0], ['idx', ['cast', ['idx', 'h_addr', 0], 'DWordWL'], 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | μ |
| `SHA256-C6.i` | i ∈ [0, 3], j ∈ [0, 3] | `IS_HALF[h_addr[i][j]]` | μ |
| `SHA256-C7.i` | i ∈ [1, 3] | `ADD<h_addr[i]::DWordWL; h_addr[0]::DWordWL, 8 * i::DWordWL>` |  |
| `SHA256-C8.i` | i ∈ [0, 3] | `MEMW[['arr', ['idx', 'h', ['+', ['*', 8, 'i'], 3]], ['idx', 'h', ['+', ['*', 8, 'i'], 2]], ['idx', 'h', ['+', ['*', 8, 'i'], 1]], ['idx', 'h', ['+', ['*', 8, 'i'], 0]], ['idx', 'h', ['+', ['*', 8, 'i'], 7]], ['idx', 'h', ['+', ['*', 8, 'i'], 6]], ['idx', 'h', ['+', ['*', 8, 'i'], 5]], ['idx', 'h', ['+', ['*', 8, 'i'], 4]]]; 0, h_addr[i]::DWordWL, ['arr', ['idx', 'out', ['+', ['*', 8, 'i'], 3]], ['idx', 'out', ['+', ['*', 8, 'i'], 2]], ['idx', 'out', ['+', ['*', 8, 'i'], 1]], ['idx', 'out', ['+', ['*', 8, 'i'], 0]], ['idx', 'out', ['+', ['*', 8, 'i'], 7]], ['idx', 'out', ['+', ['*', 8, 'i'], 6]], ['idx', 'out', ['+', ['*', 8, 'i'], 5]], ['idx', 'out', ['+', ['*', 8, 'i'], 4]]], timestamp + 1::DWordWL, 0, 0, 1]` | μ |

Then we prepare the message schedule, by emitting the input chunk with multiplicities corresponding to the number of times it will be read during a compression evaluation. The  chip itself is implicitly invoked by itself and , setting the `amount` column appropriately for the number of times the `w` value is required.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHA256-C9.i` | i ∈ [0, 0] | `SHA256_M[2^0 * m[4 * i + 3] + 2^8 * m[4 * i + 2] + 2^16 * m[4 * i + 1] + 2^24 * m[4 * i + 0]; timestamp, i]` | -2 * μ |
| `SHA256-C10.i` | i ∈ [1, 8] | `SHA256_M[2^0 * m[4 * i + 3] + 2^8 * m[4 * i + 2] + 2^16 * m[4 * i + 1] + 2^24 * m[4 * i + 0]; timestamp, i]` | -3 * μ |
| `SHA256-C11.i` | i ∈ [9, 13] | `SHA256_M[2^0 * m[4 * i + 3] + 2^8 * m[4 * i + 2] + 2^16 * m[4 * i + 1] + 2^24 * m[4 * i + 0]; timestamp, i]` | -4 * μ |
| `SHA256-C12.i` | i ∈ [14, 15] | `SHA256_M[2^0 * m[4 * i + 3] + 2^8 * m[4 * i + 2] + 2^16 * m[4 * i + 1] + 2^24 * m[4 * i + 0]; timestamp, i]` | -5 * μ |

And finally, we provide the boundaries for the  chip and the final addition of the compression to the old state. Observe that we embed the addition into the upper 32 bits of a double word, in order to satisfy and use the `ADD` chip.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHA256-C13` |  | `SHA256ROUND[timestamp, ['arr', ['+', ['*', ['^', 2, 0], ['idx', 'h', 3]], ['*', ['^', 2, 8], ['idx', 'h', 2]], ['*', ['^', 2, 16], ['idx', 'h', 1]], ['*', ['^', 2, 24], ['idx', 'h', 0]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 7]], ['*', ['^', 2, 8], ['idx', 'h', 6]], ['*', ['^', 2, 16], ['idx', 'h', 5]], ['*', ['^', 2, 24], ['idx', 'h', 4]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 11]], ['*', ['^', 2, 8], ['idx', 'h', 10]], ['*', ['^', 2, 16], ['idx', 'h', 9]], ['*', ['^', 2, 24], ['idx', 'h', 8]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 15]], ['*', ['^', 2, 8], ['idx', 'h', 14]], ['*', ['^', 2, 16], ['idx', 'h', 13]], ['*', ['^', 2, 24], ['idx', 'h', 12]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 19]], ['*', ['^', 2, 8], ['idx', 'h', 18]], ['*', ['^', 2, 16], ['idx', 'h', 17]], ['*', ['^', 2, 24], ['idx', 'h', 16]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 23]], ['*', ['^', 2, 8], ['idx', 'h', 22]], ['*', ['^', 2, 16], ['idx', 'h', 21]], ['*', ['^', 2, 24], ['idx', 'h', 20]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 27]], ['*', ['^', 2, 8], ['idx', 'h', 26]], ['*', ['^', 2, 16], ['idx', 'h', 25]], ['*', ['^', 2, 24], ['idx', 'h', 24]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 31]], ['*', ['^', 2, 8], ['idx', 'h', 30]], ['*', ['^', 2, 16], ['idx', 'h', 29]], ['*', ['^', 2, 24], ['idx', 'h', 28]]]], 0]` | μ |
| `SHA256-C14` |  | `SHA256ROUND[timestamp, last_round_out, 64]` | -μ |
| `SHA256-C15.i` | i ∈ [0, 31] | `IS_BYTE[out[i]]` | μ |
| `SHA256-C16.i` | i ∈ [0, 7] | `ADD<['arr', 0, ['+', ['*', ['^', 2, 0], ['idx', 'out', ['+', ['*', 4, 'i'], 3]]], ['*', ['^', 2, 8], ['idx', 'out', ['+', ['*', 4, 'i'], 2]]], ['*', ['^', 2, 16], ['idx', 'out', ['+', ['*', 4, 'i'], 1]]], ['*', ['^', 2, 24], ['idx', 'out', ['+', ['*', 4, 'i'], 0]]]]]; ['arr', 0, ['idx', 'last_round_out', 'i']], ['arr', 0, ['+', ['*', ['^', 2, 0], ['idx', 'h', ['+', ['*', 4, 'i'], 3]]], ['*', ['^', 2, 8], ['idx', 'h', ['+', ['*', 4, 'i'], 2]]], ['*', ['^', 2, 16], ['idx', 'h', ['+', ['*', 4, 'i'], 1]]], ['*', ['^', 2, 24], ['idx', 'h', ['+', ['*', 4, 'i'], 0]]]]]>` |  |

In this VM, we assign syscall number -1 to the  accelerator. The chip therefore contributes the following interaction to the lookup-argument:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHA256-C17` | `IS_BIT<μ>` |  |
| `SHA256-C18` | `ECALL[timestamp, ['arr', ['-', ['^', 2, 32], 1], ['-', ['^', 2, 32], 1]]]` | -μ |

### Padding

## `SHA256`msgsched chip

### Columns

The  chip leverages  variables, spanning  columns:

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | The timestamp/identifier for this execution of the message schedule |
| `index` | `BaseField` | The index of the output word |
| `amount` | `BaseField` | The multiplicity with which to output the resulting word |

### Output

| Name | Type | Description |
|------|------|-------------|
| `out` | `WordHL` | The output, `w[index]` |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `back2` | `Word` | `w[index - 2]` |
| `back7` | `Word` | `w[index - 7]` |
| `back15` | `Word` | `w[index - 15]` |
| `back16` | `Word` | `w[index - 16]` |
| `s0` | `Word` | $`back15` >>> 7 xor `back15` >>> 18 xor `back15` >> 3$ |
| `s1` | `Word` | $`back2` >>> 17 xor `back2` >>> 19 xor `back2` >> 10$ |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `carry` | `Byte` | The carry of computing `out` |

**Definition of `carry`:**
```
carry := 2^-32 * (back16 + s0 + back7 + s1 - out::Word)
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

### Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `SHA256MSGSCHED-A1` |  | #`IS_WORD[SHA256_M[timestamp, i]]` for $0 <= i < #`index`$ |

### Constraints

First, we gather the dependencies from earlier in the message schedule.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHA256MSGSCHED-C1` | `IS_BYTE[index - 16]` | μ |
| `SHA256MSGSCHED-C2` | `SHA256_M[back2; timestamp, index - 2]` | μ |
| `SHA256MSGSCHED-C3` | `SHA256_M[back7; timestamp, index - 7]` | μ |
| `SHA256MSGSCHED-C4` | `SHA256_M[back15; timestamp, index - 15]` | μ |
| `SHA256MSGSCHED-C5` | `SHA256_M[back16; timestamp, index - 16]` | μ |

Then, we calculate the result. It suffices to check that the carry of adding four range-checked words into a range-checked word is not too big, following the logic from [add]. In this case, using the `IS_BYTE` constraint allows us to add multiple words together at the same time, without needing to store and range-check intermediate results.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHA256MSGSCHED-C6` |  | `ROTXOR[s0; back15, 2, 11, 3, 0]` | μ |
| `SHA256MSGSCHED-C7` |  | `ROTXOR[s1; back2, 3, 2, 10, 0]` | μ |
| `SHA256MSGSCHED-C8` |  | `IS_BYTE[carry]` | μ |
| `SHA256MSGSCHED-C9.i` | i ∈ [0, 1] | `IS_HALF[out[i]]` | μ |

Finally, we contribute to the LogUp.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHA256MSGSCHED-C10` | `IS_BIT<μ>` |  |
| `SHA256MSGSCHED-C11` | `μ` = 0 => `amount` = 0 |  |
| | _polynomial:_ `(1 - μ) * amount = 0` | |
| `SHA256MSGSCHED-C12` | `SHA256_M[out::Word; timestamp, index]` | -amount |

## `SHA256`round chip

### Columns

The  chip leverages  variables, spanning  columns:

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | The timestamp/identifier for this execution of the round function |
| `a` | `WordBL` | State element |
| `b` | `WordBL` | State element |
| `c` | `WordBL` | State element |
| `d` | `Word` | State element |
| `e` | `WordBL` | State element |
| `f` | `WordBL` | State element |
| `g` | `WordBL` | State element |
| `h` | `Word` | State element |
| `index` | `BaseField` | The round number/index |

### Output

| Name | Type | Description |
|------|------|-------------|
| `out_a` | `WordHL` | $`temp1` + `temp2`$ |
| `out_e` | `WordHL` | $`d` + `temp1`$ |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `a_and_b` | `WordBL` | $`a` class("binary", amp) `b`$. Part of `maj` |
| `a_xor_b` | `WordBL` | $`a` xor `b`$. Part of `maj` |
| `c_and_a_xor_b` | `WordBL` | $`c` class("binary", amp) (`a` xor `b`)$. Part of `maj` |
| `e_and_f` | `WordBL` | $`e` class("binary", amp) `f`$. Part of `ch` |
| `not_e_and_g` | `WordBL` | $(not `e`) class("binary", amp) `g`$. Part of `ch` |
| `kval` | `Word` | `k[index]` |
| `S0` | `Word` | Transformation of `a` |
| `S1` | `Word` | Transformation of `e` |
| `wval` | `Word` | `w[index]` |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `carry_a` | `Byte` | The carry from `out_a` |
| `carry_e` | `Byte` | The carry from `out_e` |
| `ch` | `Word` | ch value |
| `maj` | `Word` | maj value |
| `temp1` | `BaseField` | `temp1` value |
| `temp2` | `BaseField` | `temp2` value |

**Definition of `carry_a`:**
```
carry_a := 2^-32 * (temp1 + temp2 - out_a::Word)
```

**Definition of `carry_e`:**
```
carry_e := 2^-32 * (d + temp1 - out_e::Word)
```

**Definition of `ch`:**
```
ch := e_and_f::Word + not_e_and_g::Word
```

**Definition of `maj`:**
```
maj := a_and_b::Word + c_and_a_xor_b::Word
```

**Definition of `temp1`:**
```
temp1 := h + S1 + ch + kval + wval
```

**Definition of `temp2`:**
```
temp2 := S0 + maj
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

### Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `SHA256ROUND-A1` |  | All state values are valid words |

### Constraints

First, we compute the necessary intermediate values.

To compute `maj`, observe that ` (a bitand b) xor (a bitand c) xor (b bitand c) = (a bitand b) xor (c bitand (a xor b)), ` by distribution. Additionally, since for this form, `(a bitand b)` and `(a xor b)` are disjoint, so are `(a bitand b)` and `(c bitand (a xor b))`, and hence we can replace that top-level XOR with a field addition to compute `(a bitand b) + (c bitand (a xor b))`, needing fewer intermediate columns. Similarly, `ch` can be written as `(e bitand f) + ((2^32 - 1 - e) bitand g)`.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHA256ROUND-C1.i` | i ∈ [0, 3] | `AND_BYTE[a_and_b[i]; a[i], b[i]]` | μ |
| `SHA256ROUND-C2.i` | i ∈ [0, 3] | `XOR_BYTE[a_xor_b[i]; a[i], b[i]]` | μ |
| `SHA256ROUND-C3.i` | i ∈ [0, 3] | `AND_BYTE[c_and_a_xor_b[i]; c[i], a_xor_b[i]]` | μ |
| `SHA256ROUND-C4.i` | i ∈ [0, 3] | `AND_BYTE[e_and_f[i]; e[i], f[i]]` | μ |
| `SHA256ROUND-C5.i` | i ∈ [0, 3] | `AND_BYTE[not_e_and_g[i]; 255 - e[i], g[i]]` | μ |
| `SHA256ROUND-C6` |  | `SHA256_K[kval; index]` | μ |
| `SHA256ROUND-C7` |  | `SHA256_M[wval; timestamp, index]` | μ |
| `SHA256ROUND-C8` |  | `ROTXOR[S0; a::Word, 6, 9, 2, 1]` | μ |
| `SHA256ROUND-C9` |  | `ROTXOR[S1; e::Word, 9, 14, 6, 1]` | μ |

Then we constrain the addition for the new state, constraining additions with the same `IS_BYTE` trick as before.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHA256ROUND-C10.i` | i ∈ [0, 1] | `IS_HALF[out_a[i]]` | μ |
| `SHA256ROUND-C11` |  | `IS_BYTE[carry_a]` | μ |
| `SHA256ROUND-C12.i` | i ∈ [0, 1] | `IS_HALF[out_e[i]]` | μ |
| `SHA256ROUND-C13` |  | `IS_BYTE[carry_e]` | μ |

Finally, we chain the rounds together through the interactions.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHA256ROUND-C14` | `SHA256ROUND[timestamp, ['arr', ['cast', 'a', 'Word'], ['cast', 'b', 'Word'], ['cast', 'c', 'Word'], 'd', ['cast', 'e', 'Word'], ['cast', 'f', 'Word'], ['cast', 'g', 'Word'], 'h'], index]` | -μ |
| `SHA256ROUND-C15` | `SHA256ROUND[timestamp, ['arr', ['cast', 'out_a', 'Word'], ['cast', 'a', 'Word'], ['cast', 'b', 'Word'], ['cast', 'c', 'Word'], ['cast', 'out_e', 'Word'], ['cast', 'e', 'Word'], ['cast', 'f', 'Word'], ['cast', 'g', 'Word']], index + 1]` | μ |

### Padding

## `ROTXOR` chip

This chip takes as input `a`, `r0`, `r1`, `r2` (4-bit values) and a bit `last_rot` to compute $ cases( (a >>> (16 + r_0)) xor (a >>> (16 + r_0 - r_1)) xor (a >>> r_2) quad "if" `last_rot`, (a >>> (16 + r_0)) xor (a >>> (16 + r_0 - r_1)) xor (a >> r_2) quad "if" `!last_rot` ), $ where we let `>>>` denote right rotation and `>>` logical shift right. We choose this representation so that all shift amounts required fit into 4 bits, making the usage of `HWSL` more straightforward and avoid extra columns to represent more bits.

### Columns

The  chip leverages  variables, spanning  columns:

### Input

| Name | Type | Description |
|------|------|-------------|
| `a` | `WordHL` | The input value |
| `r0` | `Byte` | The first amount of rotation, low nibble |
| `r1` | `Byte` | The second amount of rotation, low nibble |
| `r2` | `Byte` | The third amount of rotation, low nibble |
| `last_rot` | `Bit` | Whether the rotation by `r2` is a rotation (1) or just a shift (0) |

### Output

| Name | Type | Description |
|------|------|-------------|
| `out` | `WordBL` | The output |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `a0_left` | `WordHL` | `a << (16 - r0)` |
| `a0_right` | `WordHL` | `a >> r0` |
| `a1_left` | `WordHL` | `a0 << r1` |
| `a1_right` | `WordHL` | `a0 >> (16 - r1)` |
| `a2_left` | `WordHL` | `a << (16 - r2)` |
| `a2_right` | `WordHL` | `a >> r2` |
| `a0` | `WordBL` | `a >>> (16 + r0)` |
| `a1` | `WordBL` | `a >>> (16 + r0 - r1)` (which is `a0 <<< r1`) |
| `a2` | `WordBL` | `a >>> r2` or `a >> r2` |
| `a01` | `WordBL` | $a_0 xor a_1$ |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` |  |

### Assumptions

Range checking for all elements is inherited from the bitwise lookups. We can safely assume that no `r_i` will be zero, and avoid extra work due to right rotation needing `16 - shift` as arguments to the `HWSL` interactions.

| Tag | Range | Description |
|-----|-------|-------------|
| `ROTXOR-A1` |  | $#`r0`, #`r1`, #`r2` in [1, 15]$ |

### Constraints

We first compute all rotations (or shifts) of `a`. `a1` is computed as a left rotation of `a0`, in order to not need additional columns to represent the full right-rotation amounts.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ROTXOR-C1.i` | i ∈ [0, 1] | `HWSL[['arr', ['idx', 'a0_left', 'i'], ['idx', 'a0_right', 'i']]; a[i], 16 - r0]` | μ |
| `ROTXOR-C2.i` | i ∈ [0, 1] | `HWSL[['arr', ['idx', 'a1_left', 'i'], ['idx', 'a1_right', 'i']]; (a0::WordHL)[i], r1]` | μ |
| `ROTXOR-C3.i` | i ∈ [0, 1] | `HWSL[['arr', ['idx', 'a2_left', 'i'], ['idx', 'a2_right', 'i']]; a[i], 16 - r2]` | μ |
| `ROTXOR-C4.i` | i ∈ [0, 1] | `a0[i]` = `a0_left[i]` + `a0_right[1 - i]` |  |
| | | _polynomial:_ `(a0::WordHL)[i] - a0_left[i] - a0_right[1 - i] = 0` | |
| `ROTXOR-C5.i` | i ∈ [0, 1] | `a1[i]` = `a1_left[i]` + `a1_right[1 - i]` |  |
| | | _polynomial:_ `(a1::WordHL)[i] - a1_left[i] - a1_right[1 - i] = 0` | |
| `ROTXOR-C6` |  | `a2[0]` = `a2_left[1]` + `a2_right[0]` |  |
| | | _polynomial:_ `(a2::WordHL)[0] - a2_left[1] - a2_right[0] = 0` | |
| `ROTXOR-C7` |  | `a2[1]` = `last_rot` dot `a2_left[0]` + `a2_right[1]` |  |
| | | _polynomial:_ `(a2::WordHL)[0] - last_rot * a2_left[0] - a2_right[1] = 0` | |

Then the bitwise XOR of the results.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ROTXOR-C8.i` | i ∈ [0, 3] | `XOR_BYTE[a01[i]; a0[i], a1[i]]` | μ |
| `ROTXOR-C9.i` | i ∈ [0, 3] | `XOR_BYTE[out[i]; a01[i], a2[i]]` | μ |

And finally contribute to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `ROTXOR-C10` | `ROTXOR[out::Word; a::Word, r0, r1, r2, last_rot]` | -μ |

### Padding

## Constant lookup

As mentioned, we provide the round constants through a short precomputed lookup table: .

### Input

| Name | Type | Description |
|------|------|-------------|
| `index` | `BaseField` |  |
| `K` | `Word` |  |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` |  |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHA256_K-C1` | `SHA256_K[K; index]` | -μ |

## Notes/optimizations

- This could instead be designed following the [RISC-V Crypto Scalar extension `Zknh`], for wider compatibility, but this design is likely to be more efficient. It is still possible, if desired, to expose  (or a selection of parameter instantiations thereof) as implementation for these primitives. - The message schedule could be exposed as its own ECALL instead, but the direct integration leads to better efficiency. - Some of these chips could be made narrower, at the cost of introducing some extra lookups and extra tables to compute and store intermediate results.

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | Timestamp at which the ECALL is invoked. Used as unique identifier for this invocation. |
| `h` | `Byte[32]` | The state of the hash function. |
| `h_addr` | `DWordHL[4]` | The addresses of the doublewords of `h` |
| `m` | `Byte[64]` | The input chunk. |
| `m_addr` | `DWordHL[8]` | The addresses of the doublewords of `m` |

### Output

| Name | Type | Description |
|------|------|-------------|
| `out` | `Byte[32]` | The new state. |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `last_round_out` | `Word[8]` | The output from the last compression round |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Constraints

### memory

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHA256-C1` |  | `MEMW[['arr', ['idx', ['cast', ['idx', 'm_addr', 0], 'DWordWL'], 0], ['idx', ['cast', ['idx', 'm_addr', 0], 'DWordWL'], 0], 0, 0, 0, 0, 0, 0]; 1, (2 * 11)::DWordWL, ['arr', ['idx', ['cast', ['idx', 'm_addr', 0], 'DWordWL'], 0], ['idx', ['cast', ['idx', 'm_addr', 0], 'DWordWL'], 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | μ |
| `SHA256-C2.i` | i ∈ [0, 7], j ∈ [0, 3] | `IS_HALF[m_addr[i][j]]` | μ |
| `SHA256-C3.i` | i ∈ [1, 7] | `ADD<m_addr[i]::DWordWL; m_addr[0]::DWordWL, (8 * i)::DWordWL>` |  |
| `SHA256-C4.i` | i ∈ [0, 7] | `MEMW[['arr', ['idx', 'm', ['+', ['*', 8, 'i'], 3]], ['idx', 'm', ['+', ['*', 8, 'i'], 2]], ['idx', 'm', ['+', ['*', 8, 'i'], 1]], ['idx', 'm', ['+', ['*', 8, 'i'], 0]], ['idx', 'm', ['+', ['*', 8, 'i'], 7]], ['idx', 'm', ['+', ['*', 8, 'i'], 6]], ['idx', 'm', ['+', ['*', 8, 'i'], 5]], ['idx', 'm', ['+', ['*', 8, 'i'], 4]]]; 0, m_addr[i]::DWordWL, ['arr', ['idx', 'm', ['+', ['*', 8, 'i'], 3]], ['idx', 'm', ['+', ['*', 8, 'i'], 2]], ['idx', 'm', ['+', ['*', 8, 'i'], 1]], ['idx', 'm', ['+', ['*', 8, 'i'], 0]], ['idx', 'm', ['+', ['*', 8, 'i'], 7]], ['idx', 'm', ['+', ['*', 8, 'i'], 6]], ['idx', 'm', ['+', ['*', 8, 'i'], 5]], ['idx', 'm', ['+', ['*', 8, 'i'], 4]]], timestamp, 0, 0, 1]` | μ |
| `SHA256-C5` |  | `MEMW[['arr', ['idx', ['cast', ['idx', 'h_addr', 0], 'DWordWL'], 0], ['idx', ['cast', ['idx', 'h_addr', 0], 'DWordWL'], 1], 0, 0, 0, 0, 0, 0]; 1, (2 * 10)::DWordWL, ['arr', ['idx', ['cast', ['idx', 'h_addr', 0], 'DWordWL'], 0], ['idx', ['cast', ['idx', 'h_addr', 0], 'DWordWL'], 1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | μ |
| `SHA256-C6.i` | i ∈ [0, 3], j ∈ [0, 3] | `IS_HALF[h_addr[i][j]]` | μ |
| `SHA256-C7.i` | i ∈ [1, 3] | `ADD<h_addr[i]::DWordWL; h_addr[0]::DWordWL, 8 * i::DWordWL>` |  |
| `SHA256-C8.i` | i ∈ [0, 3] | `MEMW[['arr', ['idx', 'h', ['+', ['*', 8, 'i'], 3]], ['idx', 'h', ['+', ['*', 8, 'i'], 2]], ['idx', 'h', ['+', ['*', 8, 'i'], 1]], ['idx', 'h', ['+', ['*', 8, 'i'], 0]], ['idx', 'h', ['+', ['*', 8, 'i'], 7]], ['idx', 'h', ['+', ['*', 8, 'i'], 6]], ['idx', 'h', ['+', ['*', 8, 'i'], 5]], ['idx', 'h', ['+', ['*', 8, 'i'], 4]]]; 0, h_addr[i]::DWordWL, ['arr', ['idx', 'out', ['+', ['*', 8, 'i'], 3]], ['idx', 'out', ['+', ['*', 8, 'i'], 2]], ['idx', 'out', ['+', ['*', 8, 'i'], 1]], ['idx', 'out', ['+', ['*', 8, 'i'], 0]], ['idx', 'out', ['+', ['*', 8, 'i'], 7]], ['idx', 'out', ['+', ['*', 8, 'i'], 6]], ['idx', 'out', ['+', ['*', 8, 'i'], 5]], ['idx', 'out', ['+', ['*', 8, 'i'], 4]]], timestamp + 1::DWordWL, 0, 0, 1]` | μ |

### sched

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHA256-C9.i` | i ∈ [0, 0] | `SHA256_M[2^0 * m[4 * i + 3] + 2^8 * m[4 * i + 2] + 2^16 * m[4 * i + 1] + 2^24 * m[4 * i + 0]; timestamp, i]` | -2 * μ |
| `SHA256-C10.i` | i ∈ [1, 8] | `SHA256_M[2^0 * m[4 * i + 3] + 2^8 * m[4 * i + 2] + 2^16 * m[4 * i + 1] + 2^24 * m[4 * i + 0]; timestamp, i]` | -3 * μ |
| `SHA256-C11.i` | i ∈ [9, 13] | `SHA256_M[2^0 * m[4 * i + 3] + 2^8 * m[4 * i + 2] + 2^16 * m[4 * i + 1] + 2^24 * m[4 * i + 0]; timestamp, i]` | -4 * μ |
| `SHA256-C12.i` | i ∈ [14, 15] | `SHA256_M[2^0 * m[4 * i + 3] + 2^8 * m[4 * i + 2] + 2^16 * m[4 * i + 1] + 2^24 * m[4 * i + 0]; timestamp, i]` | -5 * μ |

### compress

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHA256-C13` |  | `SHA256ROUND[timestamp, ['arr', ['+', ['*', ['^', 2, 0], ['idx', 'h', 3]], ['*', ['^', 2, 8], ['idx', 'h', 2]], ['*', ['^', 2, 16], ['idx', 'h', 1]], ['*', ['^', 2, 24], ['idx', 'h', 0]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 7]], ['*', ['^', 2, 8], ['idx', 'h', 6]], ['*', ['^', 2, 16], ['idx', 'h', 5]], ['*', ['^', 2, 24], ['idx', 'h', 4]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 11]], ['*', ['^', 2, 8], ['idx', 'h', 10]], ['*', ['^', 2, 16], ['idx', 'h', 9]], ['*', ['^', 2, 24], ['idx', 'h', 8]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 15]], ['*', ['^', 2, 8], ['idx', 'h', 14]], ['*', ['^', 2, 16], ['idx', 'h', 13]], ['*', ['^', 2, 24], ['idx', 'h', 12]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 19]], ['*', ['^', 2, 8], ['idx', 'h', 18]], ['*', ['^', 2, 16], ['idx', 'h', 17]], ['*', ['^', 2, 24], ['idx', 'h', 16]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 23]], ['*', ['^', 2, 8], ['idx', 'h', 22]], ['*', ['^', 2, 16], ['idx', 'h', 21]], ['*', ['^', 2, 24], ['idx', 'h', 20]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 27]], ['*', ['^', 2, 8], ['idx', 'h', 26]], ['*', ['^', 2, 16], ['idx', 'h', 25]], ['*', ['^', 2, 24], ['idx', 'h', 24]]], ['+', ['*', ['^', 2, 0], ['idx', 'h', 31]], ['*', ['^', 2, 8], ['idx', 'h', 30]], ['*', ['^', 2, 16], ['idx', 'h', 29]], ['*', ['^', 2, 24], ['idx', 'h', 28]]]], 0]` | μ |
| `SHA256-C14` |  | `SHA256ROUND[timestamp, last_round_out, 64]` | -μ |
| `SHA256-C15.i` | i ∈ [0, 31] | `IS_BYTE[out[i]]` | μ |
| `SHA256-C16.i` | i ∈ [0, 7] | `ADD<['arr', 0, ['+', ['*', ['^', 2, 0], ['idx', 'out', ['+', ['*', 4, 'i'], 3]]], ['*', ['^', 2, 8], ['idx', 'out', ['+', ['*', 4, 'i'], 2]]], ['*', ['^', 2, 16], ['idx', 'out', ['+', ['*', 4, 'i'], 1]]], ['*', ['^', 2, 24], ['idx', 'out', ['+', ['*', 4, 'i'], 0]]]]]; ['arr', 0, ['idx', 'last_round_out', 'i']], ['arr', 0, ['+', ['*', ['^', 2, 0], ['idx', 'h', ['+', ['*', 4, 'i'], 3]]], ['*', ['^', 2, 8], ['idx', 'h', ['+', ['*', 4, 'i'], 2]]], ['*', ['^', 2, 16], ['idx', 'h', ['+', ['*', 4, 'i'], 1]]], ['*', ['^', 2, 24], ['idx', 'h', ['+', ['*', 4, 'i'], 0]]]]]>` |  |

### lookup

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHA256-C17` | `IS_BIT<μ>` |  |
| `SHA256-C18` | `ECALL[timestamp, ['arr', ['-', ['^', 2, 32], 1], ['-', ['^', 2, 32], 1]]]` | -μ |