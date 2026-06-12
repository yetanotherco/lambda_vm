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

The initialization will need to correspond to a fixed initial register state for the VM, as well as the memory loaded from the program binary, zero-initialization of memory elsewhere, and private input provided by the prover. The contribution of initialization with static data from the ELF executable and the initial register state to the sum can be handled directly by the verifier, ensuring correctness corresponding to the ELF binary being proven. To enable the loading of the PC in [cpu]:memory, register initialization happens at timestamp 1. Register finalization is made possible for the verifier by having a known state from the HALT chip ([halt]). This leaves only zero-initialization and prover input as prover-side concerns for initialization, alongside the finalization of the entire used memory.

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
| `PAGE-C1` | `IS_BYTE<init>` |  |
| `PAGE-C2` | `IS_BYTE<fini>` |  |
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

The following lists signatures of the 29 interactions in this VM.

| Signature | Bus size |
|-----------|----------|
| `DECODE[DWordWL, DWordWL, BaseField]` | 5 |
| `CPU32[Byte; DWordWL, DWordWL]` | 5 |
| `ALU[DWordWL; DWordWL, DWordWL, Byte]` | 7 |
| `MEMOP[DWordWL; DWordWL, DWordWL, DWordWL, Byte]` | 9 |
| `BRANCH[DWordWL; DWordWL, DWordWL, DWordWL, Bit]` | 9 |
| `MEMW[BaseField[8]; Bit, DWordWL, BaseField[8], DWordWL, Bit, Bit, Bit]` | 24 |
| `MEMW[Bit, DWordWL, BaseField[8], DWordWL, Bit, Bit, Bit]` | 16 |
| `LOAD[DWordWL; DWordWL, DWordWL, Byte]` | 7 |
| `ECALL[DWordWL, DWordWL]` | 4 |
| `CNB[DWordWL, BaseField, DWordWL, DWordWL]` | 7 |
| `COMMIT[BaseField, Byte]` | 2 |
| `BYTE_ALU[Byte; Byte, Byte, Byte]` | 4 |
| `MSB8[Bit; Byte]` | 2 |
| `MSB16[Bit; Half]` | 2 |
| `ZERO[Bit; B20]` | 2 |
| `ARE_BYTES[Byte, Byte]` | 2 |
| `IS_HALF[Half]` | 1 |
| `IS_B20[B20]` | 1 |
| `HWSL[Half[2]; Half, B4]` | 4 |
| `memory[Bit, DWordWL, DWordWL, BaseField]` | 6 |
| `SHA256_K[Word; BaseField]` | 2 |
| `SHA256_M[Word; DWordWL, BaseField]` | 4 |
| `SHA256ROUND[DWordWL, Word[8], BaseField]` | 11 |
| `ROTXOR[Word; Word, Byte, Byte, Byte, Bit]` | 6 |
| `KECCAK[DWordWL, BaseField, Byte[8][5][5]]` | 203 |
| `KECCAK_RC[Byte[8]; BaseField]` | 9 |
| `ECDAS[DWordWL, U256BL, U256BL, U256BL, U256BL, Byte, Bit]` | 132 |
| `SERVE_K[DWordWL, DWordWL, Byte]` | 5 |
| `BIT[DWordWL, Byte]` | 3 |

Below, we list the signatures of the 6 templates in this VM.

| Signature |
|-----------|
| `BaseField => IS_BIT<BaseField>` |
| `BaseField => IS_BYTE<BaseField>` |
| `BaseField => ADD<DWordWL; DWordWL, DWordWL>` |
| `BaseField => SUB<DWordWL; DWordWL, DWordWL>` |
| `Bit => NEG<DWordWL; DWordHL>` |
| `SIGN<Bit; Half, Bit>` |

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

# IS_BYTE Template

When a chip leverages this template twice or more, implementors are encouraged to merge pairs of  interactions with identical conditions into `ARE_BYTES` interactions; the  template is included for convenience of notation, and to complete the specification of chips that use an odd number of  range checks.

## Variables

The  template leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `X` | `BaseField` | Value for which to assert that it lies in the range $[0, 255]$. |

### Condition

| Name | Type | Description |
|------|------|-------------|
| `cond` | `BaseField` |  |

## Constraints

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `IS_BYTE-C1` | `ARE_BYTES[0, X]` | cond |

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

# DECODE Table

All `RV64IMC` instruction are to be decoded to a format that can be interpreted by the VM. This section outlines the decoding table being used in the VM. For reasons of efficiency, data in this table is significantly compressed. Since reasoning about this compressed form is needlessly complex, the `decode (uncompressed)` section presents the same table in uncompressed form, and explains how to decode `RV64IM` assembly instructions to it. Instructions on how to compress the uncompressed table to form the compressed decode table, can be derived from the `packed_decode` variable provided below.

## Variables

The  table is comprised of  variables that are expressed using  columns:

### Output

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | value of the program counter this instruction is associated with. |
| `packed_decode` | `BaseField` | Ordered concatenation of several small variables. The `decode (uncompressed)` section explains the purpose of each variable.\ A list of each variable and the bit(-range) in which it is located:\ [0] `read_register1`, \ [1] `read_register2`, \ [2] `write_register`, \ [3] `word_instr`, \ [4] `ALU`, \ [5] `ADD`, \ [6] `SUB`, \ [7] `MEMORY`, \ [8] `BRANCH`, \ [9] `ECALL`, \ [10:17] `rs1`, \ [18:25] `rs2`, \ [26:33] `rd`, \ [34:41] `half_instruction_length`, \ [42:49] `alu_flags`, \ [50:57] `mem_flags`, \ the remaining bits are set to zero.  |
| `imm` | `DWordWL` | the *fully extended (!)* 64-bit version of the immediate. |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` | The multiplicity with which this instruction is looked up in the `CPU` table. |

## Padding

The  table must be padded to a length that is a power of two. Empty rows with the following content can be added to achieve this:

| Column | Padding value |
|--------|---------------|
| `pc` | `1` |
| `packed_decode` | `0` |
| `imm` | `0` |
| `μ` | `0` |

This is simultaneously the row that is used for padding rows in the CPU, if the multiplicity is nonzero, so we need to ensure that this table has at least one row of padding.

## Decoding<decode:decoding-overview>

For the purposes of explaining decoding, we decompress 's `packed_decode` variable into its constituent variables. Note that the below table is _not_ used in practice: it is solely used for the purposes of this explanation. The construction of the `alu_flags` and `mem_flags` columns is given here through virtual columns.

### Output

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | value of the program counter this instruction is associated with. |
| `rs1` | `Byte` | index of source register 1. |
| `rs2` | `Byte` | index of source register 2. |
| `rd` | `Byte` | index of destination register. |
| `read_register1` | `Bit` | whether to load the contents of address `rs1` (1) or `0` (0) into `rv1`. |
| `read_register2` | `Bit` | whether to load the contents of address `rs2` (1) or `0` (0) into `rv2`. |
| `write_register` | `Bit` | whether the result should be written to `rd` ($=0$ for memory write and when $`rd` = `x0`)$. |
| `imm` | `DWordWL` | the *fully extended (!)* 64-bit version of the immediate. |
| `word_instr` | `Bit` | Whether the instruction is a `*W` instruction, requiring the inputs and outputs to be (sign) extended. |
| `ALU` | `Bit` | Enable the ALU |
| `ADD` | `Bit` | ALU does an ADD |
| `SUB` | `Bit` | ALU does a SUB |
| `BRANCH` | `Bit` | The instruction is a branch |
| `MEMORY` | `Bit` | The instruction is a memory access |
| `ECALL` | `Bit` | Perform an ECALL |
| `half_instruction_length` | `Byte` | Half of how many bytes this instruction takes up in the program |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `alu_op` | `B4` | Operation selector value for the ALU |
| `signed` | `Bit` | selector used to indicate signed or unsigned input interpretation. |
| `signed2` | `Bit` | A second signed bit, useful for MUL instructions |
| `muldiv_selector` | `Bit` | selects which output of `MUL` (lo/hi) or `DVRM` (quo/rem) is wanted. |
| `invert` | `Bit` | Instructs the EQ or LT chip to invert its result, or inverts the direction of the SHIFT chip (right instead of left) |
| `memory_op` | `Bit` | Selects whether to LOAD (0) or STORE (1) |
| `mem_2B` | `Bit` | whether the memory access (read or write) touches exactly $2$ bytes. |
| `mem_4B` | `Bit` | whether the memory access (read or write) touches exactly $4$ bytes. |
| `mem_8B` | `Bit` | whether the memory access (read or write) touches exactly $8$ bytes. |
| `mem_signed` | `Bit` | Whether the memory operation is a signed one, this is distinct from `signed` to enable the `JALR` flag to alias `mem_flags` |
| `JALR` | `Bit` | The branch is a JAL(R) |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `alu_flags` | `Byte` | The combined ALU flags |
| `mem_flags` | `Byte` | The combined memory flags (or JALR when BRANCHing) |

**Definition of `alu_flags`:**
```
alu_flags := alu_op + 32 * signed + 64 * (signed2 + invert) + 128 * muldiv_selector
```

**Definition of `mem_flags`:**
```
mem_flags := JALR + memory_op + 2 * mem_signed + 4 * mem_2B + 8 * mem_4B + 16 * mem_8B
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` | The multiplicity with which this instruction is looked up in the `CPU` table. |

First, we provide a mapping from an an ALU operation "descriptor" to the numerical value as used for the `alu_op` column. This is the table used to find the value for the ) notation when performing `ALU` or `BYTE_ALU` interactions.

table(columns: (auto, auto), stroke: 0pt, inset: (right: .5em), align: (left, left), table.header[*Descriptor*][*value*], table.hline(stroke: 1.5pt))[ *AND*][0][ *OR*][1][ *XOR*][2][ *EQ*][3][ *LT*][4][ *SHIFT*][5][ *SHIFTW*][6][ *MUL*][7][ *DIVREM*][8]

We will illustrate how each instruction should be expressed in this (uncompressed) decoding table. The columns of the accompanying table represent the following: - *`operation`*: the assembly operation being encoded. - *`alu`*: Set to the descriptor of the ALU operation to be used for `alu_op`. If listed as `ADD` or `SUB`, the corresponding flag should be set, otherwise set `ALU = 1` when this column is not empty. - *`w_instr`*, *`signed`*: whether to set the `word_instr` and `signed` flags, respectively. - *other*: the other flags that should be set or variables that should be given specific values.

For the purpose of brevity and readability, the table uses the following rules-of-thumb: + `rd`, `rs1`, `rs2`, and `imm` are mapped to the values provided by the instruction; when a value is not specified by an instruction it defaults to `0`. + `read_register1`, `read_register2` and `write_register` are set to `1` when respectively ``rs1` != 0`, ``rs2` != 0`, or  ``rd` != 0`.

Further clarification is provided in the notes following the table.

/// Add a reference to one or more notes following this table.

super("[" + refs.pos().map(r => ref(r)).join(",") + "]") }

show figure: set block(breakable: true)

figure(table( columns: (auto, auto, auto, auto, 1fr, auto), stroke: 0pt, inset: (right: .5em), align: (left, right, center, center, left, right), fill: (_, y) => // Overlay a low-opacity fill color to distinguish the different rows better if calc.odd(y) and y <= lines.len() { color.rgb(0, 0, 100, 20) } else { color.rgb(255, 255, 255, 20) }, table.header([*Operation*], [*alu*], [*`w_instr`*], [*`signed`*], [*other*], []), table.hline(stroke: 1.5pt), table.vline(x: 1, start: 1, end: lines.len() + 1, stroke: .5pt), ..lines.flatten(), table.hline(stroke: 1.5pt), table.footer([*Operation*], [*alu*], [*`w_instr`*], [*`signed`*], [*other*]), )) }

// OP-IMM ([`ADDI[W]   rd, rs1, imm`], [`ADD`], [`[W]`], [], [], []), ([`SLTI[U]   rd, rs1, imm`], [`LT`], [], [.not`[U]`], [], []), ([`ANDI      rd, rs1, imm`], [`AND`], [], [], [], []), ([`ORI       rd, rs1, imm`], [`OR`],   [], [], [], []), ([`XORI      rd, rs1, imm`], [`XOR`], [], [], [], []), ([`SLLI[W]   rd, rs1, imm`], [`SHIFT[W]`], [`[W]`], [], [], []), ([`SRLI[W]   rd, rs1, imm`], [`SHIFT[W]`], [`[W]`], [], [`invert`], []), ([`SRAI[W]   rd, rs1, imm`], [`SHIFT[W]`], [`[W]`], [1], [`invert`], []), // OP ([`ADD[W]    rd, rs1, rs2`], [`ADD`], [`[W]`], [], [], []), ([`SUB[W]    rd, rs1, rs2`], [`SUB`], [`[W]`], [], [], []), ([`SLT[U]    rd, rs1, rs2`], [`LT`], [], [.not`[U]`], [], []), ([`AND       rd, rs1, rs2`], [`AND`], [], [], [], []), ([`OR        rd, rs1, rs2`], [`OR`], [], [], [], []), ([`XOR       rd, rs1, rs2`], [`XOR`], [], [], [], []), ([`SLL[W]    rd, rs1, rs2`], [`SHIFT[W]`], [`[W]`], [], [], []), ([`SRL[W]    rd, rs1, rs2`], [`SHIFT[W]`], [`[W]`], [], [`invert`], []), ([`SRA[W]    rd, rs1, rs2`], [`SHIFT[W]`], [`[W]`], [1], [`invert`], []), // OP - M ([`MUL[W]    rd, rs1, rs2`], [`MUL`], [`[W]`], [1], [`signed2`], []), ([`MULH      rd, rs1, rs2`], [`MUL`], [], [1], [`signed2`, `muldiv_selector`], []), ([`MULHU     rd, rs1, rs2`], [`MUL`], [], [], [`muldiv_selector`], []), ([`MULHSU    rd, rs1, rs2`], [`MUL`], [], [1], [`muldiv_selector`], []), ([`DIV[U][W] rd, rs1, rs2`], [`DIVREM`], [`[W]`], [.not`[U]`], [], []), ([`REM[U][W] rd, rs1, rs2`], [`DIVREM`], [`[W]`], [.not`[U]`], [`muldiv_selector`], []), // LUI/AUIPC ([`LUI       rd, imm`], [`ADD`], [], [], [], []), ([`AUIPC     rd, imm`], [`ADD`], [], [], [`rs1 := x255`], []), ([`JAL       rd, imm`], [], [], [], [`BRANCH`, `JALR`, `rs1 := x255`], []), // Branching ([`JALR      rd, rs1, imm`], [], [], [], [`BRANCH`, `JALR`], []), ([`BEQ      rs1, rs2, imm`], [`EQ`], [], [], [`BRANCH`], []), ([`BNE      rs1, rs2, imm`], [`EQ`], [], [], [`BRANCH`, `invert`], []), ([`BLT[U]   rs1, rs2, imm`], [`LT`], [], [.not`[U]`], [`BRANCH`], []), ([`BGE[U]   rs1, rs2, imm`], [`LT`], [], [.not`[U]`], [`BRANCH`, `invert`], []), // LOAD ([`LD        rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_8B`], []), ([`LW[U]     rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_signed := `.not`[U]`, `mem_4B`], []), ([`LH[U]     rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_signed := `.not`[U]`, `mem_2B`], []), ([`LB[U]     rd, rs1, imm`], [`ADD`], [], [], [`MEMORY`, `mem_signed := `.not`[U]`], []), // STORE ([`SD       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`, `mem_8B`], []), ([`SW       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`, `mem_4B`], []), ([`SH       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`, `mem_2B`], []), ([`SB       rs1, rs2, imm`], [`ADD`], [], [], [`MEMORY`, `memory_op`], []), // ECALL/EBREAK ([`ECALL`], [], [], [], [`ECALL`, ``rs1` := `x17``], []), // FENCE ([`FENCE`], [`ADD`], [], [], [], []),

Note that the above table has no entry for the `EBREAK` instruction. We treat `EBREAK` as an unprovable trap, and its absence from the table enables this by having no valid decoding available for when the instruction is encountered.

### C-type instructions

The `RV64C` extension for compressed instructions specifies that \~50% of all instructions can be represented using a 16-bit instruction (rather than 32-bits), saving \~25% in code size. This execution of assembly code is _not_ agnostic to an instruction's compression state; after executing a compressed instruction, the `pc` should be incremented by `2` rather than `4`. As such, we provide the `half_instruction_length` column that *must take on the value `1` for compressed instructions and `2` for regular instructions*. It is represented as half the number of bytes in the instruction to make misaligned instructions lengths unrepresentable. Additionally, having the variable opens the door for future optimizations involving "fused" instructions, where common sequences of instructions are merged into a single decoded version and need only a single CPU row to prove.

// Construct a note that can be referenced through `lbl`

show figure: (it) => align(left, []) [ ] }

### Notes

We note the following about the above decoding table:

enum.item( referenceable_note( "note_word_instr", [`word_instr`: `[W]` indicates that ``word_instr` = 1` for the `W`-variant of the operation, and `0` for the non-`W`-variant. Similarly, `SHIFT[W]` indicates the `SHIFTW` operation for the `W`-variant, and `SHIFT` otherwise.] ), enum.item( referenceable_note( "note_signed", [`signed`: .not`[U]` indicates that ``signed` = 1` for the *non-`U`*-variant of the operation, and `0` for the `U`-variant.] ), enum.item( referenceable_note( "note-lui", [`LUI`: this operation loads the 20-bit `imm` in the upper bits of `rd`. Observe that this can be represented using `ADDI rd, x0, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[12:32]` of `imm` and extending it to 64 bits.*] ), enum.item( referenceable_note( "note-auipc", [`AUIPC`: this operation adds the 20-bit immediate to the upper bits of `pc` and stores the result in `rd`. Given that the `pc` is stored in `x255`, this operation can be represented using `ADDI rd, x255, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[12:32]` of `imm` and extending it to 64 bits.*] ), enum.item( referenceable_note( "note-jal", [`JAL`: this operation stores ``pc` + `2 * half_instruction_length`` in `rd` and adds two times the sign-extended 20-bit immediate to the `pc`. Note that this can be represented using `JALR rd, x255, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[1:21]` of `imm` and extending it to 64 bits; the least significant bit should always be 0.*] ), enum.item( referenceable_note( "note-ecall", [`ECALL`: "On RISC-V a system call has its own instruction: `ECALL`. [...] A7 [= register `x17`] contains the system call number." [[source]] ] ), enum.item( referenceable_note( "note-fence", [`FENCE`: currently, the VM interprets this operation as `ADDI x0 x0 0`; a no-op.]

---

# CPU Chip

The  chip coordinates memory accesses and dispatches to other chips for arithmetic and logical operations. It bases its decisions on the entry of the `DECODE` table ([decode]) corresponding the current program counter (PC).

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `Timestamp` | A preprocessed timestamp to coordinate the memory argument. Since we have at most 3 non-disjoint memory accesses (`(rs1, rs2, rd)`, `(rs1, pc, pc)`, `MEMORY`) a maximum of 4 slots is enough. |
| `pc` | `DWordWL` | The program counter |
| `rs1` | `Byte` | Source register 1 index |
| `rs2` | `Byte` | Source register 2 index |
| `rd` | `Byte` | Destination register index |
| `read_register1` | `Bit` | Whether to read from `rs1` (1) or to place a 0 in `rv1` (0) |
| `read_register2` | `Bit` | Whether to read from `rs2` (1) or to place a 0 in `rv2` (0) |
| `write_register` | `Bit` | Whether to write back to the destination register |
| `imm` | `DWordWL` | The fully extended 64-bit version of the immediate |
| `half_instruction_length` | `Byte` | Half the number of bytes consumed by this instruction, commonly used to indicate whether the instruction is of C type, i.e., whether it is 2 bytes long (= 1) instead of 4 (= 2) |
| `word_instr` | `Bit` | Whether the instruction is a \*W instruction, requiring the inputs and outputs to be (sign) extended |
| `ALU` | `Bit` | Whether to use the ALU for this instruction |
| `alu_flags` | `Byte` | The ALU operation + flags (interpreting things as signed/unsigned, choosing the MUL/DVRM output, ...) to pass to the ALU |
| `ADD` | `Bit` | Addition fast-path bypassing the ALU |
| `SUB` | `Bit` | Subtraction fast-path bypassing the ALU |
| `MEMORY` | `Bit` | Whether this instruction touches memory (LOAD/STORE) |
| `mem_flags` | `Byte` | The flags to pass for MEMORY operations (LOAD vs STORE, number of bytes touched, signed) |
| `BRANCH` | `Bit` | Whether this instruction is a conditional branch (BLT, BEQ) |
| `ECALL` | `Bit` | Whether this instruction is an ECALL |

### Output

| Name | Type | Description |
|------|------|-------------|
| `next_pc` | `DWordWL` | The program counter for the next instruction |
| `rvd` | `DWordWL` | The value to (maybe) be written back to rvd |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `prev_pc_timestamp_borrow` | `Bit` | The borrow bit for computing the previous timestamp the PC was accessed |
| `pc_double_read` | `Bit` | Whether the PC is being read as a general purpose register (`rs1`) this cycle |
| `rv1` | `DWordWL` | The value of register `rs1` |
| `rv2` | `DWordWL` | The value of register `rs2` |
| `arg2` | `DWordWL` | A multiplexed version of `rv2` and `imm`, to be used as second argument to ALU calls |
| `res` | `DWordHL` | The ALU result |
| `branch_cond` | `Bit` | Whether a branch is taken: the branch condition evaluates to true, or we are doing an unconditional jump |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `JALR` | `Bit` | Read whether our BRANCH corresponds to a JAL(R) instruction from `mem_flags`, as `MEMORY` and `BRANCH` are mutually exclusive |
| `packed_decode` | `BaseField` | A packed representation of all bit flags and register indices obtained from the decoding |

**Definition of `JALR`:**
```
JALR := mem_flags
```

**Definition of `packed_decode`:**
```
packed_decode := 2^0 * read_register1 + 2^1 * read_register2 + 2^2 * write_register + 2^3 * word_instr + 2^4 * ALU + 2^5 * ADD + 2^6 * SUB + 2^7 * MEMORY + 2^8 * BRANCH + 2^9 * ECALL + 2^10 * rs1 + 2^18 * rs2 + 2^26 * rd + 2^34 * half_instruction_length + 2^42 * alu_flags + 2^50 * mem_flags
```

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `CPU-A1` |  | `MEMORY` and `BRANCH` are mutually exclusive |
| `CPU-A2` |  | When `MEMORY + BRANCH = 0`, either `read_register2 = 0` or `imm = 0` should be enforced by the decoding. This is needed for `arg2`. |
| `CPU-A3` |  | $#`!MEMORY` => #`IS_BIT<mem_flags>`$ |

Additionally, the following constraints can be used to provide defense-in-depth validation of the assumptions.

| Tag | Description |
|-----|-------------|
| `CPU-C1` | not (`MEMORY` and `BRANCH`) |
| | _polynomial:_ `MEMORY * BRANCH = 0` |
| `CPU-C2` | (1 - `MEMORY` - `BRANCH`) => (`read_register2` = 0 or `imm[i]` = 0) |
| | _polynomial:_ `(1 - MEMORY - BRANCH) * read_register2 * (imm[0] + imm[1]) = 0` |
| `CPU-C3` | 1 - MEMORY ⇒ `IS_BIT<mem_flags>` |

## Constraints

First, we perform a decoding lookup for the current PC. Instructions having the `word_instr` flag set are not decoded here, as they are delegated to the `CPU32` chip. In that case, we ensure that the current row of the CPU cannot have any other observable effects.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-C4` | `DECODE[pc, imm, packed_decode]` | 1 - word_instr |
| `CPU-C5` | `word_instr` => `MEMORY = 0` |  |
| | _polynomial:_ `word_instr * MEMORY = 0` | |
| `CPU-C6` | `word_instr` => `BRANCH = 0` |  |
| | _polynomial:_ `word_instr * BRANCH = 0` | |
| `CPU-C7` | `word_instr` => `ECALL = 0` |  |
| | _polynomial:_ `word_instr * ECALL = 0` | |
| `CPU-C8` | `word_instr` => `read_register1 = 0` |  |
| | _polynomial:_ `word_instr * read_register1 = 0` | |
| `CPU-C9` | `word_instr` => `read_register2 = 0` |  |
| | _polynomial:_ `word_instr * read_register2 = 0` | |
| `CPU-C10` | `word_instr` => `write_register = 0` |  |
| | _polynomial:_ `word_instr * write_register = 0` | |
| `CPU-C11` | `CPU32[half_instruction_length; timestamp, pc]` | word_instr |

### Range checks

We constrain all columns to have the appropriate ranges. All values in `packed_decode` need to be checked to ensure the packing is correct for the interaction. In contrast, we know ahead of time that decoding will ensure proper range checks for `pc` and `imm`. Similarly, since `next_pc` will propagate through the memory argument and be looked up in the instruction decoding on the next cycle, it is forced to be in the correct range; the final value for `next_pc` is similarly fixed by the memory finalization. For the auxiliary columns, we need to check the limbs of `res`, since `rv1` and `rv2` are enforced by the memory argument, and `rvd` is correct by the correctness of the dependent chips. The ranges of the other auxiliary columns are enforced through later constraints.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CR12` |  | `IS_BIT<read_register1>` |  |
| `CPU-CR13` |  | `IS_BIT<read_register2>` |  |
| `CPU-CR14` |  | `IS_BIT<write_register>` |  |
| `CPU-CR15` |  | `IS_BYTE<half_instruction_length>` |  |
| `CPU-CR16` |  | `IS_BIT<word_instr>` |  |
| `CPU-CR17` |  | `IS_BIT<ALU>` |  |
| `CPU-CR18` |  | `IS_BYTE<alu_flags>` |  |
| `CPU-CR19` |  | `IS_BIT<ADD>` |  |
| `CPU-CR20` |  | `IS_BIT<SUB>` |  |
| `CPU-CR21` |  | `IS_BIT<MEMORY>` |  |
| `CPU-CR22` |  | `IS_BYTE<mem_flags>` |  |
| `CPU-CR23` |  | `IS_BIT<BRANCH>` |  |
| `CPU-CR24` |  | `IS_BIT<ECALL>` |  |
| `CPU-CR25` |  | `IS_BYTE<rs1>` |  |
| `CPU-CR26` |  | `IS_BYTE<rs2>` |  |
| `CPU-CR27` |  | `IS_BYTE<rd>` |  |
| `CPU-CR28.i` | i ∈ [0, 3] | `IS_HALF[res[i]]` | 1 |

### ALU

The ALU functionality is then obtained through delegation to the `ALU` signature, backed by the various ALU chips, or by using the appropriate template. For the pure ALU path, `arg2` is computed as `rv2 + imm`, which relies on [cpu:a:arg2]-multiplex to be either `rv2` or `imm`, depending on the instruction. The other contributions for `arg2` are specific to the (mutually exclusive, [cpu:a:mem]-branch-mutex) `MEMORY` and `BRANCH` flags: - For the `MEMORY` path, we want the output of the ALU to be ``rv1` + `imm``, as that is the address at which the memory access occurs. - For the `BRANCH` path, we want the ALU output to reflect the branch condition (or just be inactive for JALR).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CA29.i` | i ∈ [0, 1] | `arg2` = `MEMORY` dot `imm` + `BRANCH` dot `rv2` + (1 - `MEMORY` - `BRANCH`) dot (`rv2` + `imm`) |  |
| | | _polynomial:_ `arg2[i] - MEMORY * imm[i] - BRANCH * rv2[i] - (1 - MEMORY - BRANCH) * (rv2 + imm)[i] = 0` | |
| `CPU-CA30` |  | ADD ⇒ `ADD<res::DWordWL; rv1, arg2>` |  |
| `CPU-CA31` |  | SUB ⇒ `SUB<res::DWordWL; rv1, arg2>` |  |
| `CPU-CA32` |  | `ALU[res::DWordWL; rv1, arg2, alu_flags]` | ALU |

### Memory<cpu:memory>

Note that since registers need no byte-addressing, we store them in the memory argument with `Word` limbs, simultaneously ensuring that register reads are properly range checked as long as all writes are. The `pc` register behaves very predictably with respect to its timestamps and when it is being read, so for performance reasons, we inline its memory interactions directly into the  chip.

Potentially overlapping memory accesses are ensured to have disjoint timestamps. One consequence of that is that `next_pc` is written at `timestamp + 1` to ensure the access is disjoint with the `pc` read into `rv1` as part of the `AUIPC` instruction (see [cpu:c:read_rv1] and [decode]:decoding-overview). Constraints regarding whether `pc_double_read` corresponds to an `AUIPC` instruction are not necessary, as regardless of its value, the old timestamp is guaranteed smaller than the new timestamp, and the integrity of the memory argument therefore ensures the correctness of this bit.

The memory interaction itself is handled by the `MEMORY` signature, which will read the `mem_flags` argument to perform either a `LOAD` or a `STORE`. We refer to the previous section's description of `arg2` for how the address is computed.

The value to (potentially) be written back to `rd` is stored in `rvd`, which can either come from the ALU --- in case of an ALU operation or a JALR branch --- or from the MEMORY interaction.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CM33` |  | `MEMW[[rv1[0], rv1[1], 0, 0, 0, 0, 0, 0]; 1, 2::DWordWL * rs1, [rv1[0], rv1[1], 0, 0, 0, 0, 0, 0], timestamp + 0::DWordWL, 1, 0, 0]` | read_register1 |
| `CPU-CM34.i` | i ∈ [0, 1] | `!read_register1` => `rv1[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register1) * rv1[i] = 0` | |
| `CPU-CM35` |  | `MEMW[[rv2[0], rv2[1], 0, 0, 0, 0, 0, 0]; 1, 2::DWordWL * rs2, [rv2[0], rv2[1], 0, 0, 0, 0, 0, 0], timestamp + 1::DWordWL, 1, 0, 0]` | read_register2 |
| `CPU-CM36.i` | i ∈ [0, 1] | `!read_register2` => `rv2[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register2) * rv2[i] = 0` | |
| `CPU-CM37` |  | `MEMW[1, 2::DWordWL * rd, [rvd[0], rvd[1], 0, 0, 0, 0, 0, 0], timestamp + 2::DWordWL, 1, 0, 0]` | write_register |
| `CPU-CM38` |  | `MEMOP[rvd; timestamp, res::DWordWL, rv2, mem_flags]` | MEMORY |
| `CPU-CM39.i` | i ∈ [0, 1] | `!MEMORY` and `!BRANCH` => `rvd` = `res` |  |
| | | _polynomial:_ `(1 - MEMORY - BRANCH) * (rvd[i] - (res::DWordWL)[i]) = 0` | |
| `CPU-CM40` |  | `IS_BIT<pc_double_read>` |  |
| `CPU-CM41` |  | `IS_BIT<prev_pc_timestamp_borrow>` |  |
| `CPU-CM42.i` | i ∈ [0, 1] | `memory[1, [2 * 255 + i, 0], [(timestamp[0] - 3 * (1 - pc_double_read)) + 2^32 * prev_pc_timestamp_borrow, timestamp[1] - prev_pc_timestamp_borrow], pc[i]]` | 1 |
| `CPU-CM43.i` | i ∈ [0, 1] | `memory[1, [2 * 255 + i, 0], timestamp + 1::DWordWL, next_pc[i]]` | -1 |

### Branching

A branch is expressed by having the `BRANCH` flag set to 1. Since `BRANCH` and `MEMORY` are mutually exclusive ([cpu:a:mem]-branch-mutex), we can repurpose the `mem_flags` field to indicate a JALR instruction. When JALR is not set, we have a conditional branch that is decided upon by the result of the ALU instructions, as set in the `res` variable. As such, we can set `branch_cond` appropriately as multiplicity flag for the `BRANCH` chip.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CB44` | `branch_cond` = `BRANCH` and (`JALR` or `res`) |  |
| | _polynomial:_ `branch_cond - BRANCH * JALR - BRANCH * (1 - JALR) * res[0] = 0` | |
| `CPU-CB45` | `BRANCH[next_pc; pc, imm, rv1, JALR]` | branch_cond |
| `CPU-CB46` | 1 - branch_cond ⇒ `ADD<next_pc; pc, [2 * half_instruction_length, 0]>` |  |
| `CPU-CB47` | BRANCH ⇒ `ADD<rvd; pc, [2 * half_instruction_length, 0]>` |  |

### System

The interactions with the wider system go through the `ECALL` interface. Since we treat `EBREAK` instructions as unprovable traps, we avoid emitting `DECODE` rows for these, and do not need any further handling in the CPU.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CS48` | `ECALL[timestamp, rv1]` | ECALL |

## Padding

The CPU can be padded with the following values, which have a corresponding row in the DECODE table, at the _odd_ address 1, only reachable through a HALT ecall.

| Column | Padding value |
|--------|---------------|
| `pc` | `1` |
| `rs1` | `0` |
| `rs2` | `0` |
| `rd` | `0` |
| `read_register1` | `0` |
| `read_register2` | `0` |
| `write_register` | `0` |
| `imm` | `0` |
| `half_instruction_length` | `2` |
| `word_instr` | `0` |
| `ALU` | `0` |
| `alu_flags` | `0` |
| `ADD` | `0` |
| `SUB` | `0` |
| `MEMORY` | `0` |
| `mem_flags` | `0` |
| `BRANCH` | `0` |
| `ECALL` | `0` |
| `next_pc` | `1` |
| `rvd` | `0` |
| `prev_pc_timestamp_borrow` | `0` |
| `pc_double_read` | `0` |
| `rv1` | `0` |
| `rv2` | `0` |
| `arg2` | `0` |
| `res` | `0` |
| `branch_cond` | `0` |

This approach minimizes the number of dependent lookups, increasing only multiplicities in the `DECODE` table and the `IS_BYTE` and `IS_HALF` lookups.

---

# CPU32 Chip

The  chip is used to delegate the 32-bit instructions of the RV64I instruction set from the main CPU table ([cpu]). All 32-bit instructions are ALU-only instructions, so the BRANCH, MEMORY and ECALL paths need no elaboration. The timestamp and PC have already been read by the CPU table at this point, and need no further checking; the PC for the next instruction will also already be handled by CPU.

The structure follows the regular ALU path, with some extra variables and constraints to contain the required sign extensions.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | The timestamp for the CPU row |
| `pc` | `DWordWL` | The PC at which the instruction occurs |

### Output

| Name | Type | Description |
|------|------|-------------|
| `half_instruction_length` | `Byte` | The length of this instruction |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `rs1` | `Byte` | Source register 1 |
| `read_register1` | `Bit` | Whether to read from `rs1` or not |
| `rv1` | `DWordWHH` | The value in register `rs1` |
| `rv1_sign` | `Bit` | The sign bit of the lower word of `rv1` |
| `arg1` | `DWordWL` | The sign-extended version of `rv1` |
| `rs2` | `Byte` | Source register 2 |
| `read_register2` | `Bit` | Whether to read from `rs2` |
| `rv2` | `DWordWHH` | The value in register `rs2` |
| `rv2_sign` | `Bit` | The sign bit of the lower word of `rv2` |
| `imm` | `DWordWL` | The fully sign-extended immediate to use |
| `arg2` | `DWordWL` | Either the sign-extended version of `rv2` or all of `imm` |
| `res` | `DWordHL` | The ALU result |
| `res_sign` | `Bit` | The sign bit of the lower word of `res` |
| `rd` | `Byte` | Destination register |
| `write_register` | `Bit` | Whether to write back to `rd` |
| `rvd` | `DWordWL` | The value to write back to `rd`, the sign-extended version of `res` |
| `ALU` | `Bit` | Whether the full ALU is active |
| `alu_flags` | `Byte` | The ALU operation + flags |
| `ADD` | `Bit` | Whether the full ALU is active |
| `SUB` | `Bit` | Whether the full ALU is active |
| `signed` | `Bit` | Whether the instruction is signed or not. Extracted from `alu_flags`, used to determine the extension for the inputs |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `packed_decode` | `BaseField` | The packed representation of all flags and information from the decode table |

**Definition of `packed_decode`:**
```
packed_decode := 2^0 * read_register1 + 2^1 * read_register2 + 2^2 * write_register + 2^3 * 1 + 2^4 * ALU + 2^5 * ADD + 2^6 * SUB + 2^10 * rs1 + 2^18 * rs2 + 2^26 * rd + 2^34 * half_instruction_length + 2^42 * alu_flags
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `CPU32-A1.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |
| `CPU32-A2.i` | i ∈ [0, 1] | `IS_WORD[pc[i]]` |
| `CPU32-A3` |  | `read_register2 = 0` or `imm = 0`, enforced by decoding. |

Some of the assumptions can be checked with only arithmetic constraints, so we provide these below.

| Tag | Description |
|-----|-------------|
| `CPU32-C1` | `read_register2` = 0 or `imm = 0` |
| | _polynomial:_ `read_register2 * (imm[0] + imm[1]) = 0` |

## Constraints

Most constraints correspond to those already present in the CPU, and we present them here first, including some updates to the range checking corresponding to the differing types. We also need to make sure that for padding rows (`mu = 0`), no side effects can occur.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU32-C2` | `DECODE[pc, imm, packed_decode]` | μ |

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU32-CR3` |  | `IS_BIT<μ>` |  |
| `CPU32-CR4` |  | `IS_BIT<read_register1>` |  |
| `CPU32-CR5` |  | `IS_BIT<read_register2>` |  |
| `CPU32-CR6` |  | `IS_BIT<write_register>` |  |
| `CPU32-CR7` |  | `IS_BYTE<half_instruction_length>` |  |
| `CPU32-CR8` |  | `IS_BIT<ALU>` |  |
| `CPU32-CR9` |  | `IS_BYTE<alu_flags>` |  |
| `CPU32-CR10` |  | `IS_BIT<ADD>` |  |
| `CPU32-CR11` |  | `IS_BIT<SUB>` |  |
| `CPU32-CR12` |  | `IS_BYTE<rs1>` |  |
| `CPU32-CR13` |  | `IS_BYTE<rs2>` |  |
| `CPU32-CR14` |  | `IS_BYTE<rd>` |  |
| `CPU32-CR15.i` | i ∈ [0, 1] | `IS_HALF[rv1[i]]` | μ |
| `CPU32-CR16.i` | i ∈ [0, 1] | `IS_HALF[rv2[i]]` | μ |
| `CPU32-CR17.i` | i ∈ [0, 3] | `IS_HALF[res[i]]` | μ |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU32-CA18` | ADD ⇒ `ADD<res::DWordWL; arg1, arg2>` |  |
| `CPU32-CA19` | SUB ⇒ `SUB<res::DWordWL; arg1, arg2>` |  |
| `CPU32-CA20` | `ALU[res::DWordWL; arg1, arg2, alu_flags]` | ALU |

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU32-CM21` |  | `MEMW[[(rv1::DWordWL)[0], rv1[2], 0, 0, 0, 0, 0, 0]; 1, 2::DWordWL * rs1, [(rv1::DWordWL)[0], rv1[2], 0, 0, 0, 0, 0, 0], timestamp + 0::DWordWL, 1, 0, 0]` | read_register1 |
| `CPU32-CM22.i` | i ∈ [0, 2] | `!read_register1` => `rv1[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register1) * rv1[i] = 0` | |
| `CPU32-CM23` |  | `MEMW[[(rv2::DWordWL)[0], rv2[2], 0, 0, 0, 0, 0, 0]; 1, 2::DWordWL * rs2, [(rv2::DWordWL)[0], rv2[2], 0, 0, 0, 0, 0, 0], timestamp + 1::DWordWL, 1, 0, 0]` | read_register2 |
| `CPU32-CM24.i` | i ∈ [0, 2] | `!read_register2` => `rv2[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register2) * rv2[i] = 0` | |
| `CPU32-CM25` |  | `MEMW[1, 2::DWordWL * rd, [rvd[0], rvd[1], 0, 0, 0, 0, 0, 0], timestamp + 2::DWordWL, 1, 0, 0]` | write_register |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU32-C26` | `!μ` => `read_register1 = 0` |  |
| | _polynomial:_ `(1 - μ) * read_register1 = 0` | |
| `CPU32-C27` | `!μ` => `read_register2 = 0` |  |
| | _polynomial:_ `(1 - μ) * read_register2 = 0` | |
| `CPU32-C28` | `!μ` => `write_register = 0` |  |
| | _polynomial:_ `(1 - μ) * write_register = 0` | |
| `CPU32-C29` | `CPU32[half_instruction_length; timestamp, pc]` | -μ |

Then, we have the constraints corresponding to the sign-extension and definition of `arg1`, `arg2` and `rd`. This includes a step where we extract the `signed` bit from the `alu_flags`, as this determines whether to sign extend the inputs or not.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU32-C30` | `signed` != 0 => `μ` = 1 |  |
| | _polynomial:_ `signed * (1 - μ) = 0` | |
| `CPU32-C31` | `BYTE_ALU[32 * signed; ⧼AND⧽, 32, alu_flags]` | μ |
| `CPU32-C32` | `SIGN<rv1_sign; rv1[1], signed>` |  |
| `CPU32-C33` | `arg1[0]` = `rv1[:2]` |  |
| | _polynomial:_ `arg1[0] - (rv1::DWordWL)[0] = 0` | |
| `CPU32-C34` | `arg1[1]` = (2^(32) - 1) dot `rv1_sign` |  |
| | _polynomial:_ `arg1[1] - (2^32 - 1) * rv1_sign = 0` | |
| `CPU32-C35` | `SIGN<rv2_sign; rv2[1], signed>` |  |
| `CPU32-C36` | `arg2[0]` = `rv2[:2]` + `imm[0]` |  |
| | _polynomial:_ `arg2[0] - (rv2::DWordWL)[0] - imm[0] = 0` | |
| `CPU32-C37` | `arg2[1]` = (2^(32) - 1) dot `rv2_sign` + `imm[1]` |  |
| | _polynomial:_ `arg2[1] - (2^32 - 1) * rv2_sign - imm[1] = 0` | |
| `CPU32-C38` | `SIGN<res_sign; res[1], μ>` |  |
| `CPU32-C39` | `rvd[0]` = `res[:2]` |  |
| | _polynomial:_ `rvd[0] - (res::DWordWL)[0] = 0` | |
| `CPU32-C40` | `rvd[1]` = (2^(32) - 1) dot `res_sign` |  |
| | _polynomial:_ `rvd[1] - (2^32 - 1) * res_sign = 0` | |

## Padding

The table can be padded with the following values:

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `pc` | `0` |
| `half_instruction_length` | `2` |
| `rs1` | `0` |
| `read_register1` | `0` |
| `rv1` | `0` |
| `rv1_sign` | `0` |
| `arg1` | `0` |
| `rs2` | `0` |
| `read_register2` | `0` |
| `rv2` | `0` |
| `rv2_sign` | `0` |
| `imm` | `0` |
| `arg2` | `0` |
| `res` | `0` |
| `res_sign` | `0` |
| `rd` | `0` |
| `write_register` | `0` |
| `rvd` | `0` |
| `ALU` | `0` |
| `alu_flags` | `0` |
| `ADD` | `0` |
| `SUB` | `0` |
| `signed` | `0` |
| `μ` | `0` |

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
| `shift` | `DWordWHBB` | Number of bits to shift `in` by. |
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

First, we range check our inputs appropriately.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C1.i` | i ∈ [0, 3] | `IS_HALF[in[i]]` | μ |
| `SHIFT-C2` |  | `IS_HALF[shift[2]]` | μ |
| `SHIFT-C3.i` | i ∈ [0, 1] | `IS_BYTE<shift[i]>` |  |
| `SHIFT-C4` |  | `IS_BIT<direction>` |  |
| `SHIFT-C5` |  | `IS_BIT<signed>` |  |
| `SHIFT-C6` |  | `IS_BIT<word_instr>` |  |

Then, we constrain `bit_shift` based on whether we are left or right-shifting. [shift:c:zbs] makes sure `zbs` is set to `1` if and only if `bit_shift = 0`. This flag is used to indicate the special case that ``right` = 1` and ``shift` = 0 mod 16`.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C7` | `BYTE_ALU[bit_shift; ⧼AND⧽, shift[0], 15]` | left |
| `SHIFT-C8` | `BYTE_ALU[bit_shift; ⧼AND⧽, 2^8 - 16 * zbs - shift[0], 15]` | right |
| `SHIFT-C9` | `ZERO[zbs; bit_shift]` | μ |

Next, we shift the limbs of `in` left and right by the appropriate amount, storing the results in `X` and `Y` respectively. When `zbs = 1`, the output cannot be used to compose ``in >>/>>> shift` mod 16`. To resolve this, we override `Y[i] := in[i]` and `X[i] := 0` in this case.

The case of `left`-shifting and ``bit_shift` = 0` will be used for padding rows. To prevent unnecessary lookups in padding rows, we override ``X[i]` := `in[i]`` and ``Y[i]` := 0` here.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C10.i` | i ∈ [0, 3] | `HWSL[[X[i], Y[i]]; in[i], bit_shift]` | 1 - zbs |
| `SHIFT-C11.i` | i ∈ [0, 3] | `zbs` => `X[i]` = `in[i]` dot `left` |  |
| | | _polynomial:_ `zbs * (X[i] - in[i] * left) = 0` | |
| `SHIFT-C12.i` | i ∈ [0, 3] | `zbs` => `Y[i]` = `in[i]` dot `right` |  |
| | | _polynomial:_ `zbs * (Y[i] - in[i] * right) = 0` | |
| `SHIFT-C13` |  | `HWSL[[X[4], extension - X[4]]; extension, bit_shift]` | 1 - zbs |
| `SHIFT-C14` |  | `zbs` => `X[4]` = 0 |  |
| | | _polynomial:_ `zbs * X[4] = 0` | |

### Full-limb shifting

Next, we constrain that `limb_shift` is a proper unary encoding of the fifth (and sixth if ``word_instr` = 0`) bit of `shift`. For this to be the case, three requirements must be satisfied: + *unary(0)*: ``limb_shift[`i`]` in {0, 1}` for `i in [0, 3]`, + *unary(1)*: ``limb_shift[`i`]` = 1` for exactly one `i`, and + *proper encoding*: ``limb_shift[`i`]` = 1 <=> 1/16 (`shift &` (48-32 dot `word_instr`)) = i` The first requirement is enforced by constraint [shift:c:limb_shift_is_bit]. To construct a constraint for the second and third requirement, observe that $ 1/16 dot (`shift &` (48-32 dot `word_instr`)) in cases( {0, 1, 2, 3} &"if" `word_instr` = 0, {0, 1} &"if" `word_instr` = 1 $ Observe moreover that, assuming *unary(0)*, the expression $ 1/16 dot (1 + sum_(i=0)^3 (16i-1) dot `limb_shift[`i`]`) $ can evaluate to `i` if and only if ``limb_shift[`i`]` = 1`, while the others are `0`. This means that the relation $ 1 + sum_(i=0)^3 (16i-1) dot `limb_shift[`i`]` = `shift &` (48-32 dot `word_instr`) $ enforces both *unary(1)* and *proper encoding*. This is the exact relation [shift:c:limb_shift_lookup] enforces.

Hereafter, one must only check that `out` is the proper cast of `shifted` into a `DWordWL`.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C15.i` | i ∈ [0, 3] | `IS_BIT<limb_shift[i]>` |  |
| `SHIFT-C16` |  | `BYTE_ALU[(1 - limb_shift[0]) + 15 * limb_shift[1] + 31 * limb_shift[2] + 47 * limb_shift[3]; ⧼AND⧽, shift[0], 48 - 32 * word_instr]` | μ |
| `SHIFT-C17.i` | i ∈ [0, 1] | `out[:2]` = `shifted[:4]` |  |
| | | _polynomial:_ `out[i] - (shifted::DWordWL)[i] = 0` | |

### Miscellaneous

| Tag | Description |
|-----|-------------|
| `SHIFT-C18` | `direction` => `μ` = 1 |
| | _polynomial:_ `direction * (1 - μ) = 0` |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C19` | `MSB16[is_negative; in[3]]` | signed |

*Note*: `is_negative` is not used when `signed = 0`. As such, there is no problem with it being unconstrained in this case.

### Lookups

This chip adds the following interaction to the lookup.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C20` | `ALU[out; in::DWordWL, shift::DWordWL, ⧼SHIFT⧽ + word_instr + 32 * signed + 64 * direction]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `in` | `0` |
| `shift` | `0` |
| `direction` | `0` |
| `signed` | `0` |
| `word_instr` | `0` |
| `out` | `0` |
| `is_negative` | `0` |
| `bit_shift` | `0` |
| `zbs` | `1` |
| `X` | `[0, 0, 0, 0, 0]` |
| `Y` | `[0, 0, 0, 0]` |
| `limb_shift_raw` | `[0, 0, 0]` |
| `μ` | `0` |

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

Some of the assumptions can be checked with only arithmetic constraints, so we provide these below.

| Tag | Description |
|-----|-------------|
| `BRANCH-C1` | `IS_BIT<JALR>` |

## Constraints

We constrain `next_pc` to be ``base_address` + `offset``, where `base_address` equals `pc` when ``JALR` = 0` and `register` otherwise.

The range checks on `unmasked_low_byte` and `next_pc_low[0]` are performed implicitly by the `AND_BYTE` lookup.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `BRANCH-C2` |  | 1 - JALR ⇒ `ADD<next_pc_unmasked; pc, offset::DWordWL>` |  |
| `BRANCH-C3` |  | JALR ⇒ `ADD<next_pc_unmasked; register, offset::DWordWL>` |  |
| `BRANCH-C4` |  | μ ⇒ `IS_BYTE<next_pc_low[1]>` |  |
| `BRANCH-C5` |  | `BYTE_ALU[next_pc_low[0]; ⧼AND⧽, unmasked_low_byte, 254]` | μ |
| `BRANCH-C6.i` | i ∈ [0, 2] | `IS_HALF[next_pc_high[i]]` | μ |

This chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `BRANCH-C7` | `BRANCH[next_pc; pc, offset, register, JALR]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `pc` | `0` |
| `offset` | `0` |
| `register` | `0` |
| `JALR` | `0` |
| `next_pc_high` | `[0, 0, 0]` |
| `next_pc_low` | `0` |
| `unmasked_low_byte` | `0` |
| `μ` | `0` |

---

# LT Chip

The  chip constrains an indicator bit for the less-than relation, signed or unsigned. If the `invert` flag is set, it inverts the result.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `DWordHHW` | The left operand |
| `rhs` | `DWordHHW` | The right operand |
| `signed` | `Bit` | whether to interpret `lhs` and `rhs` as signed integers (1) or not (0) |
| `invert` | `Bit` | Whether to invert the result |

### Output

| Name | Type | Description |
|------|------|-------------|
| `res` | `Bit` | The result |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `lhs_sub_rhs` | `DWordHL` | $`lhs` - `rhs`$ |
| `lhs_msb` | `Bit` | The most significant bit of `lhs` |
| `rhs_msb` | `Bit` | The most significant bit of `rhs` |
| `lt` | `Bit` | Whether $`lhs` < `rhs`$, taking `signed` into account |

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

## Constraints

We first constrain that all inputs are range checked and all variables correspond to their definition. For the defining constraint of `lt`, [lt:c:lt], observe that it is a choice between two options, depending on the input flag `signed`. In the case of unsigned comparison, we simply need `unsigned_lt`, indicating that a wraparound (carry bit) modulo `2^64` is needed to go from `rhs` to `lhs` via addition. For the case of signed comparison, we first need some case analysis.

We split `a < b` into four disjoint cases, conditioned on the sign of `a` and `b`. Recall that the sign of a number in two's complement can be read off from the MSB, being `1` for a negative number and `0` for a positive one. For this analysis, we denote the MSB of `a` as `A` and the MSB of `b` as `B`. The four disjoint cases then become:

+ `dash(A) and B and (a < b)` + `A and dash(B) and (a < b)` + `A and B and (a < b)` + `dash(A) and dash(B) and (a < b)`

The first case is evidently false, while the second case simplifies to `A and dash(B)`. For the third and fourth case, observe that when `A = B`, the `<` relation is preserved by the modular correspondence between `[-2^(31), 2^(31))` and `[0, 2^(64))`. Importantly, this modular correspondence is merely a reinterpretation of the bits or values of `a` and `b`, due to the representation in two's complement. Hence, we can introduce the value `C = `unsigned_lt``, that accurately represents the relation `a < b` when `A = B`.

Combining our three remaining cases, we obtain the boolean formula `A dash(B) or A B C or dash(A) dash(B) C`. Since the cases are disjoint, this can be computed with the binary-valued polynomial `P(A, B, C) = A (1 - B) + A B C + (1 - A) (1 - B) C`.

The polynomial `P` can be simplified to a total degree of two. We claim that the polynomial `Q(A, B, C) = A (1 - B) + A C + (1 - B) C` is, for the purposes of this chip, equivalent to `P`. An exhaustive check shows that `P(A, B, C) != Q(A, B, C)` only for the triple `(A, B, C) = (1, 0, 1)`. This is, however, impossible due to the correctness of `ADD`. In more detail, if we let `s` be the (range-checked) difference `a - b` (so the equivalent of the `lhs_sub_rhs` column), and `x'` denote the most significant word of a variable `x`, we need `c dot 2^32 + a' = b' + s' + `carry[0]``, by the definition of `carry`. However, the left hand side of this is at least `3 dot 2^31`, as `(A, C) = (1, 1)`, and the right hand side is at most `(2^31 - 1) + (2^32 - 1) + 1 = 3 dot 2^31 - 1`. Therefore, we can use `Q` to constrain `lt` when `signed = 1`.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LT-C1` | `IS_HALF[lhs[1]]` | μ |
| `LT-C2` | `IS_HALF[rhs[1]]` | μ |
| `LT-C3` | `IS_BIT<signed>` |  |
| `LT-C4` | `IS_BIT<invert>` |  |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LT-C5` | `MSB16[lhs_msb; lhs[2]]` | μ |
| `LT-C6` | `MSB16[rhs_msb; rhs[2]]` | μ |
| `LT-C7` | `lt` = `signed` dot (A (1 - B) + A C + (1 - B) C) + (1 - `signed`) dot `unsigned_lt` |  |
| | _polynomial:_ `lt - signed * (lhs_msb * (1 - rhs_msb) + lhs_msb * carry[1] + (1 - rhs_msb) * carry[1]) - (1 - signed) * unsigned_lt = 0` | |
| `LT-C8` | `res` = `lt` xor `invert` |  |
| | _polynomial:_ `res + 2 * lt * invert - lt - invert = 0` | |

And then we constrain the subtraction, taking care of the remaining range checking not yet covered by the assumptions or the `MSB16` lookup.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `LT-C9.i` | i ∈ [0, 1] | `IS_BIT<carry[i]>` |  |
| `LT-C10.i` | i ∈ [0, 3] | `IS_HALF[lhs_sub_rhs[i]]` | μ |

The chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LT-C11` | `ALU[[res, 0]; lhs::DWordWL, rhs::DWordWL, ⧼LT⧽ + 32 * signed + 64 * invert]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `lhs` | `0` |
| `rhs` | `0` |
| `signed` | `0` |
| `invert` | `0` |
| `res` | `0` |
| `lhs_sub_rhs` | `0` |
| `lhs_msb` | `0` |
| `rhs_msb` | `0` |
| `lt` | `0` |
| `μ` | `0` |

## Potential optimizations

- Split the chip into a signed and an unsigned chip, making the unsigned version cheaper.

---

# EQ Chip

The  chip is an ALU chip that compares two values and outputs a bit indicating whether they are equal or not. It optionally inverts the result if the `invert` flag is set.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `a` | `DWordWL` | The first input |
| `b` | `DWordWL` | The second input |
| `invert` | `Bit` | Whether to invert the result |

### Output

| Name | Type | Description |
|------|------|-------------|
| `res` | `Bit` | The result |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `diff` | `DWordHL` | The difference `a - b` |
| `eq` | `Bit` | The bit indicating `a == b` |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `EQ-A1.i` | i ∈ [0, 1] | `IS_WORD[a[i]]` |
| `EQ-A2.i` | i ∈ [0, 1] | `IS_WORD[b[i]]` |

## Constraints

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `EQ-C1.i` | i ∈ [0, 3] | `IS_HALF[diff[i]]` | μ |
| `EQ-C2` |  | `IS_BIT<invert>` |  |
| `EQ-C3` |  | `SUB<diff::DWordWL; a, b>` |  |
| `EQ-C4` |  | `ZERO[eq; diff[0] + diff[1] + diff[2] + diff[3]]` | μ |
| `EQ-C5` |  | `res` = `eq` xor `invert` |  |
| | | _polynomial:_ `res + 2 * eq * invert - eq - invert = 0` | |
| `EQ-C6` |  | `ALU[[res, 0]; a, b, ⧼EQ⧽ + 64 * invert]` | -μ |

## Padding

The chip can be padded with the following values:

| Column | Padding value |
|--------|---------------|
| `a` | `0` |
| `b` | `0` |
| `invert` | `0` |
| `res` | `0` |
| `diff` | `0` |
| `eq` | `0` |
| `μ` | `0` |

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
| `MUL-C1` |  | `IS_BIT<lhs_signed>` |  |
| `MUL-C2` |  | `IS_BIT<rhs_signed>` |  |
| `MUL-C3.i` | i ∈ [0, 3] | `IS_HALF[lhs[i]]` | μ_sum |
| `MUL-C4.i` | i ∈ [0, 3] | `IS_HALF[rhs[i]]` | μ_sum |
| `MUL-C5` |  | `SIGN<lhs_is_negative; lhs[3], lhs_signed>` |  |
| `MUL-C6` |  | `SIGN<rhs_is_negative; rhs[3], rhs_signed>` |  |
| `MUL-C7.i` | i ∈ [0, 3] | `IS_HALF[lo[i]]` | μ_sum |
| `MUL-C8.i` | i ∈ [0, 3] | `IS_HALF[hi[i]]` | μ_sum |
| `MUL-C9.i` | i ∈ [0, 3] | `IS_B20[carry[i]]` | μ_sum |

### Product

[mul:c:raw_product] defines `raw_product` in terms of the (sign extended) input values `lhs` and `rhs`.

| Tag | Range | Description |
|-----|-------|-------------|
| `MUL-C10.i` | i ∈ [0, 3] | `raw_product[i]` = sum_(`k`=0)^1 2^(16k) sum_(`j`=0)^(2i+k) `lhs_ext[j]` dot `rhs_ext[2i+k-j]` |
| | | _polynomial:_ `Σ_k = 0^1 2^(16 * k) * Σ_j = 0^2 * i + k lhs_ext[j] * rhs_ext[2 * i + k - j] - raw_product[i] = 0` |

### Lookup

The  chip contributes the following to the lookup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MUL-C11` | `ALU[lo::DWordWL; lhs::DWordWL, rhs::DWordWL, ⧼MUL⧽ + 32 * lhs_signed + 64 * rhs_signed]` | -μ_lo |
| `MUL-C12` | `ALU[hi::DWordWL; lhs::DWordWL, rhs::DWordWL, ⧼MUL⧽ + 32 * lhs_signed + 64 * rhs_signed + 128]` | -μ_hi |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `lhs` | `0` |
| `lhs_signed` | `0` |
| `rhs` | `0` |
| `rhs_signed` | `0` |
| `lo` | `0` |
| `hi` | `0` |
| `lhs_is_negative` | `0` |
| `rhs_is_negative` | `0` |
| `raw_product` | `0` |
| `μ_lo` | `0` |
| `μ_hi` | `0` |

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

## Constraints

First, we range-check all inputs.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C1.i` | i ∈ [0, 3] | `IS_HALF[n[i]]` | μ_sum |
| `DVRM-C2.i` | i ∈ [0, 3] | `IS_HALF[d[i]]` | μ_sum |
| `DVRM-C3` |  | `IS_BIT<signed>` |  |

From the ISA, we gather five requirements for the `DIV[U][W]` and `REM[U][W]` instructions:

enum.item([ _For both signed and unsigned division, except in the case of_ overflow, _it holds that ``n` = `q` `d` + `r``._ ]), enum.item([ _`DIV` and `DIVU` perform [...] signed and unsigned integer division [...] rounding towards zero._ ]), enum.item([ _For `REM`, the sign of a nonzero [remainder] equals the sign of the [numerator]._ ]), enum.item([ In case of _division-by-zero_, ``r` = `n`` and ``q` = 2^64-1` (unsigned) or ``q` = -1` (signed). ]), enum.item([ In case of _overflow_, ``q` = `n`` and ``r` = 0` ]), where _overflow_ occurs when ``n` = -2^(63)` and ``d` = -1` (and, hence, ``signed` = 1`), and _division-by-zero_ indicates that ``d` = 0`. In the following, we list the constraints associated with the  chip, and explain how these together enforce all five of these requirements.

### R3: Sign remainder equals sign numerator

We start with R3, which is straightforwardly asserted by constraint [dvrm:c:sign_r_equals_sign_n].

| Tag | Description |
|-----|-------------|
| `DVRM-C4` | `r` eq.not 0 => `sign_r` = `sign_n` |
| | _polynomial:_ `Σ_i = 0^3 r[i] * (sign_r - sign_n) = 0` |

### R2: rounding towards zero

R2 states that "_[in] signed and unsigned integer division [the quotient is] round[ed] towards zero._" In other words, + the sign of ``n`-`qd`` must match that of `n` (unless ``qd` = `n``), and + `|`n`-`qd`|  < |`d`|` (unless ``d` = 0`).

Leveraging R1 , we can rewrite these as + the sign of ``r`` must match that of `n` (unless ``r` = 0`), and + `|`r`|  < |`d`|` (unless ``d` = 0`).

Focusing on the first statement, we observe that this trivially holds when ``signed` = 0`, while R3 deals with the case that ``signed` = 1`. The second statement is enforced by [dvrm:c:abs_r_lt_abs_d]. [dvrm:c:abs_r_if_negative] and [dvrm:c:abs_r_if_nonnegative] (resp. [dvrm:c:abs_d_if_negative] and [dvrm:c:abs_d_if_nonnegative]) are included to ensure that `abs_r` (resp. `abs_d`) is the absolute values of `r` (resp. `d`).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C5` |  | `ALU[[1 - div_by_zero, 0]; abs_r, abs_d, ⧼LT⧽]` | μ_sum |
| `DVRM-C6` |  | sign_r ⇒ `NEG<abs_r; r>` |  |
| `DVRM-C7.i` | i ∈ [0, 1] | not`sign_r` => `abs_r` = `r` |  |
| | | _polynomial:_ `(1 - sign_r) * (abs_r[i] - (r::DWordWL)[i]) = 0` | |
| `DVRM-C8` |  | sign_d ⇒ `NEG<abs_d; d>` |  |
| `DVRM-C9.i` | i ∈ [0, 1] | not`sign_d` => `abs_d` = `d` |  |
| | | _polynomial:_ `(1 - sign_d) * (abs_d[i] - (d::DWordWL)[i]) = 0` | |

### R5: overflow

The ISA requires that ``q` = `n`` and ``r` = 0` in the event of overflow (i.e., when ``n` = -2^63` and ``d` = -1`). We note that the second half of this requirement is already satisfied by R2: since ``d` = -1 != 0`, R2 requires that `|`r`| < |`d`| = 1`, to which ``r` = 0` is the only satisfying value.

We moreover find that R1 can be leveraged to enforce the correct value of `q`. While ``n` = `qd` + `r`` (R1) does _not_ hold in the case of overflow, the relation ``n` = |`q`|`d` + `r`` _does_. We moreover note that the 64-bit _signed_ two's complement representation of `-2^63` is identical to the 64-bit _unsigned_ representation of `|-2^63| = 2^63`. As such, by interpreting `q` as an unsigned integer when ``overflow` = 1`, it follows that R1 will enforce ``q` = `0x80...00``.

In summary, in case of overflow R2 enforces that ``r` = 0`. Moreover it suffices to interpret `q` as unsigned integer ([dvrm:c:sign_q]); R1 will ensure it contains the correct value.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `DVRM-C10` | `sign_q` = `signed` dot (1- `overflow`) |  |
| | _polynomial:_ `signed * (1 - overflow) - sign_q = 0` | |
| `DVRM-C11` | `ZERO[overflow; n[0] + n[1] + n[2] + (n[3] - 2^15 * sign_n) + (1 - sign_n) + (65535 - d[0]) + (65535 - d[1]) + (65535 - d[2]) + (65535 - d[3])]` | μ_sum |

We highlight [dvrm:c:overflow]. Recall that the `overflow` flag should be set if and only if (i) ``signed` = 1`, (ii) ``n` = `0x80...00``, and (iii) ``d` = `0xFF...FF``. These requirements are equivalent to the state where: $ forall i in [0, 3]:&& 65535 - `d`_i &= 0,\ forall i in [0, 2]:&& `n`_i &= 0,\ && `n`_3 - 2^15 dot `sign_n` &= 0,\ && 1 - `sign_n` &= 0,\ $ where ``signed` = 1` follows from the last equality. The requirement is phrased in this way, because the left-hand sides of the above expressions are `>= 0` by construction. Given that the sum of these expressions does not exceed `2^19` (and thus never wraps in the field), we can now say that the `overflow` bit should be set to `1` if and only if their sum evaluates to `0`. The `ZERO` lookup guarantees this to be the case.

### R1: $#`n` = #`qd` + #`r`$

Rewriting R1, we find the constraint `not`overflow` => `n` - `r` = `qd``.

Since `n`, `d`, `q` and `r` are all 64-bit integers, we must assert this equality `mod 2^128`, rather than `mod 2^64`. To this end, we introduce `extended_n_sub_r` and leverage the `MUL` chip to verify that it is equal to ``qd` mod 2^128` using constraints [dvrm:c:mul_lower] and [dvrm:c:mul_upper]; [dvrm:c:q_range] is included to uphold assumption [mul:c:rhs].

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C12` |  | `ALU[n_sub_r::DWordWL; d::DWordWL, q::DWordWL, ⧼MUL⧽ + 32 * signed + 64 * sign_q]` | μ_sum |
| `DVRM-C13` |  | `ALU[extension_n_sub_r::DWordWL; d::DWordWL, q::DWordWL, ⧼MUL⧽ + 32 * signed + 64 * sign_q + 128]` | μ_sum |
| `DVRM-C14.i` | i ∈ [0, 3] | `IS_HALF[q[i]]` | μ_sum |

It now remains to enforce that `extended_n_sub_r` is the _signed_ 128-bit representation of ``n`-`r``. Here, we introduce `extended_n` and `extended_r`. By their definition, these variables contain the signed 128-bit representations of `n` and `r`. The `carry` variable has been defined such that it mimics those in the `ADD` chip, except that here we add two `QuadHL`s rather than two `DWordHL`, thus needing four carry bits instead of two. With this in place, [dvrm:c:n_sub_r] (mimicking [add:c:carry]) ensures `extended_n_sub_r` must contain the correct value.

Lastly, observe that ``n` - `r` in (-2^64, 2^64)`, _regardless_ of the value of `signed`. Moreover, note that the upper halves of the 128-bit representations of all values in this range are either `0xFFFFFFFF` (negative) or `0x00000000` (non-negative). This means that we do not need to store all 128 bits of `extended_n_sub_r`. Rather, we need only store the lower 64-bits, and a separate bit (`sign_n_sub_r`) indicating whether the top limbs are all-ones or all-zeroes. The prover is free to select the value for `sign_n_sub_r`; only one of the two will fit the proof.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C15.i` | i ∈ [0, 3] | `IS_BIT<carry[i]>` |  |
| `DVRM-C16.i` | i ∈ [0, 3] | `IS_HALF[r[i]]` | μ_sum |
| `DVRM-C17.i` | i ∈ [0, 3] | `IS_HALF[n_sub_r[i]]` | μ_sum |
| `DVRM-C18` |  | `IS_BIT<sign_n_sub_r>` |  |

### R4: division-by-zero

R4 requires that ``q` = 2^64-1` (unsigned) or `-1` (signed) and ``r` = n` when ``d` = 0`. Recalling R1, we see that ``n` = `q` `d` + `r` = `r`` when ``d` = 0`, already enforces the latter. Next, we note that, in two's complement, the _unsigned_ value `2^64-1` and _signed_ value `-1` are both represented by the bit string `0xFFFFFFFF`. Hence, only [dvrm:c:q_if_div_by_zero] is required to completely constrain R4; [dvrm:c:div_by_zero] just ensures the `div_by_zero` flag is set when ``d` = 0`.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `DVRM-C19.i` | i ∈ [0, 3] | `div_by_zero` => `q[i]` = 65535 |  |
| | | _polynomial:_ `div_by_zero * (q[i] - 65535) = 0` | |
| `DVRM-C20` |  | `ZERO[div_by_zero; d[0] + d[1] + d[2] + d[3]]` | μ_sum |

### Other

The following constraints are included to enforce the values of `sign_n`, `sign_r` and `sign_d` are correct.

| Tag | Description |
|-----|-------------|
| `DVRM-C21` | `SIGN<sign_n; n[3], signed>` |
| `DVRM-C22` | `SIGN<sign_r; r[3], signed>` |
| `DVRM-C23` | `SIGN<sign_d; d[3], signed>` |

### Output

Lastly, this chip contributes the following to the lookup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `DVRM-C24` | `ALU[q::DWordWL; n::DWordWL, d::DWordWL, ⧼DIVREM⧽ + 32 * signed]` | -μ_q |
| `DVRM-C25` | `ALU[r::DWordWL; n::DWordWL, d::DWordWL, ⧼DIVREM⧽ + 32 * signed + 128]` | -μ_r |

## Padding

To pad the  table, we use the following data, representing the unsigned division `frac(0, 0, style: "horizontal")`:

| Column | Padding value |
|--------|---------------|
| `n` | `0` |
| `d` | `0` |
| `signed` | `0` |
| `q` | `0` |
| `r` | `0` |
| `div_by_zero` | `1` |
| `overflow` | `0` |
| `abs_r` | `0` |
| `abs_d` | `0` |
| `n_sub_r` | `0` |
| `sign_n_sub_r` | `0` |
| `sign_n` | `0` |
| `sign_d` | `0` |
| `sign_q` | `0` |
| `sign_r` | `0` |
| `μ_q` | `0` |
| `μ_r` | `0` |

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
| `μ_ARE_BYTES` | `BaseField` |  |
| `μ_IS_HALF` | `BaseField` |  |
| `μ_IS_B20` | `BaseField` |  |
| `μ_HWSL` | `BaseField` |  |

*Note*: This table contains one row for every possible value of `(X, Y, Z)`. As such, it has length `2^8 dot 2^8 dot 2^4 = 2^(20)`.

We use the ALU operation descriptors from [decode] to identify the operations in the `BYTE_ALU` interaction. Since each of the three columns is only `2^16` rows long, they can be combined in a single `2^20` column (with room to spare).

## Lookup

This chip adds the following interactions to the lookup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `BITWISE-C1` | `BYTE_ALU[AND; ⧼AND⧽, X, Y]` | -μ_AND |
| `BITWISE-C2` | `BYTE_ALU[OR; ⧼OR⧽, X, Y]` | -μ_OR |
| `BITWISE-C3` | `BYTE_ALU[XOR; ⧼XOR⧽, X, Y]` | -μ_XOR |
| `BITWISE-C4` | `MSB8[MSB8; X]` | -μ_MSB8 |
| `BITWISE-C5` | `MSB16[MSB16; X + 256 * Y]` | -μ_MSB16 |
| `BITWISE-C6` | `ZERO[ZERO; X + 256 * Y + 65536 * Z]` | -μ_ZERO |
| `BITWISE-C7` | `ARE_BYTES[X, Y]` | -μ_ARE_BYTES |
| `BITWISE-C8` | `IS_HALF[X + 256 * Y]` | -μ_IS_HALF |
| `BITWISE-C9` | `IS_B20[X + 256 * Y + 65536 * Z]` | -μ_IS_B20 |
| `BITWISE-C10` | `HWSL[[SLL, SLLC]; X + 256 * Y, Z]` | -μ_HWSL |

## Notes/Optimizations

The following ideas may prove to be optimizations for the  chip: + Drop `MSB8` column, and instead define the `MSB8` lookup as `MSB8<X> := MSB16[256X]`. Note: currently, `MSB8` also implicity range checks the input `X` (the lookup fails if `X` is not a `Byte`). This optimization should only be executed when all chips leveraging `MSB8` do _not_ need this implicit range check. + Place the 16-bit (`AND`, `OR`, `XOR`, `MSB16`, etc.) and 20-bit (`HWSL`, `IS_B20`, `ZERO`) lookups in separate tables.

---

# BYTEWISE Chip

The  chip is an ALU chip that decomposes the input `DWordWL` values into bytes and performs a `BITWISE` operation pairwise (AND, OR, XOR). The `BITWISE` lookup inherently performs a range check, so no further constraints are necessary.

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `a` | `DWordBL` | The first input |
| `b` | `DWordBL` | The second input |
| `op` | `Byte` | The operation to perform |

### Output

| Name | Type | Description |
|------|------|-------------|
| `res` | `DWordBL` | The result |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` |  |

## Constraints

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `BYTEWISE-C1.i` | i ∈ [0, 7] | `BYTE_ALU[res[i]; op, a[i], b[i]]` | μ |
| `BYTEWISE-C2` |  | `ALU[res::DWordWL; a::DWordWL, b::DWordWL, op]` | -μ |

## Padding

The chip can be padded with the following values:

| Column | Padding value |
|--------|---------------|
| `a` | `0` |
| `b` | `0` |
| `op` | `0` |
| `res` | `0` |
| `μ` | `0` |

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
address_add := [base_address[0] + i + 1 - 2^32 * carry[i], base_address[1] + carry[i]]
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

Some of the assumptions can be checked with only arithmetic constraints, so we provide these below.

| Tag | Description |
|-----|-------------|
| `MEMW-C1` | `IS_BIT<write2>` |
| `MEMW-C2` | `IS_BIT<write4>` |
| `MEMW-C3` | `IS_BIT<write8>` |
| `MEMW-C4` | `IS_BIT<write2 + write4 + write8>` |

Our assumptions do not explicitly cover any range checks for the `is_register` and `value` columns, as these are not necessary for the correctness of this chip in isolation. Still, these properties are necessary for the consistency of the system as a whole, and therefore we document it here, keeping the type information as a reading help.

## Constraints

Depending on the values of `write2`, `write4` and `write8`, the addresses following `base_address` need to be constructed. Rather than computing these in full (which would require the later addresses to be instantiated), it suffices to know the `carry`: the bit indicating whether ``base_address`_0 + t >= 2^32`, i.e., whether adding `t in [1, 7]` to `base_address` requires a carry from the lower to the upper limb. Note that it is safe for the prover to chose these bits: additions for which this bit is not correctly set will yield an address where either the lower or upper limb is out of bounds. As such, the constructed address will not match any existing memory tokens, which are only initialized for correctly formatted and range-checked doublewords (see [memory]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW-C5` |  | `IS_BIT<μ_read>` |  |
| `MEMW-C6` |  | `IS_BIT<μ_write>` |  |
| `MEMW-C7` |  | `IS_BIT<μ_sum>` |  |
| `MEMW-C8` |  | `w2` => `μ_sum` |  |
| | | _polynomial:_ `w2 * (1 - μ_sum) = 0` | |
| `MEMW-C9.i` | i ∈ [0, 6] | `IS_BIT<carry[i]>` |  |
| `MEMW-C10` |  | `ALU[[1, 0]; old_timestamp[0], timestamp, ⧼LT⧽]` | μ_sum |
| `MEMW-C11` |  | `ALU[[1, 0]; old_timestamp[1], timestamp, ⧼LT⧽]` | w2 |
| `MEMW-C12.i` | i ∈ [2, 3] | `ALU[[1, 0]; old_timestamp[i], timestamp, ⧼LT⧽]` | w4 |
| `MEMW-C13.i` | i ∈ [4, 7] | `ALU[[1, 0]; old_timestamp[i], timestamp, ⧼LT⧽]` | write8 |

As long as `timestamp` is properly range-checked, the presence of `old_timestamp` in the memory argument automatically ensures it is appropriately range checked (this assumes no external entities provide negative multiplicities without range checking the timestamp). This ensures the assumptions for `LT` are satisfied.

There is no need to check that the additions do not overflow, as our address calculations are not performed modulo `2^64` here, and any overflow will result in an address without matching initialization.

The chip adds the following tuples to the lookup argument, to effectuate that part of the memory argument.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW-CM14` |  | `memory[is_register, base_address, old_timestamp[0], old[0]]` | μ_sum |
| `MEMW-CM15` |  | `memory[is_register, base_address, timestamp, value[0]]` | -μ_sum |
| `MEMW-CM16` |  | `memory[is_register, address_add[0], old_timestamp[1], old[1]]` | w2 |
| `MEMW-CM17` |  | `memory[is_register, address_add[0], timestamp, value[1]]` | -w2 |
| `MEMW-CM18.i` | i ∈ [2, 3] | `memory[is_register, address_add[i - 1], old_timestamp[i], old[i]]` | w4 |
| `MEMW-CM19.i` | i ∈ [2, 3] | `memory[is_register, address_add[i - 1], timestamp, value[i]]` | -w4 |
| `MEMW-CM20.i` | i ∈ [4, 7] | `memory[is_register, address_add[i - 1], old_timestamp[i], old[i]]` | write8 |
| `MEMW-CM21.i` | i ∈ [4, 7] | `memory[is_register, address_add[i - 1], timestamp, value[i]]` | -write8 |

This chip contributes the following to the lookup argument:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW-CO22` | `MEMW[old; is_register, base_address, value, timestamp, write2, write4, write8]` | -μ_read |
| `MEMW-CO23` | `MEMW[is_register, base_address, value, timestamp, write2, write4, write8]` | -μ_write |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `is_register` | `0` |
| `base_address` | `0` |
| `value` | `0` |
| `timestamp` | `0` |
| `write2` | `0` |
| `write4` | `0` |
| `write8` | `0` |
| `old` | `0` |
| `carry` | `0` |
| `old_timestamp` | `0` |
| `μ_read` | `0` |
| `μ_write` | `0` |

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

Some of the assumptions can be checked with only arithmetic constraints, so we provide these below.

| Tag | Description |
|-----|-------------|
| `MEMW_A-C1` | `IS_BIT<write2>` |
| `MEMW_A-C2` | `IS_BIT<write4>` |
| `MEMW_A-C3` | `IS_BIT<write8>` |
| `MEMW_A-C4` | `IS_BIT<write2 + write4 + write8>` |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW_A-C9` | `IS_HALF[base_address[0] + write2 + 3 * write4 + 7 * write8]` | μ_sum |
| `MEMW_A-C10` | `IS_BIT<μ_read>` |  |
| `MEMW_A-C11` | `IS_BIT<μ_write>` |  |
| `MEMW_A-C12` | `IS_BIT<μ_sum>` |  |
| `MEMW_A-C13` | `w2` => `μ_sum` |  |
| | _polynomial:_ `w2 * (1 - μ_sum) = 0` | |
| `MEMW_A-C14` | `ALU[[1, 0]; old_timestamp, timestamp, ⧼LT⧽]` | μ_sum |

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW_A-CM15` |  | `memory[is_register, base_address::DWordWL, old_timestamp, old[0]]` | μ_sum |
| `MEMW_A-CM16` |  | `memory[is_register, base_address::DWordWL, timestamp, value[0]]` | -μ_sum |
| `MEMW_A-CM17` |  | `memory[is_register, base_address::DWordWL + 1::DWordWL, old_timestamp, old[1]]` | w2 |
| `MEMW_A-CM18` |  | `memory[is_register, base_address::DWordWL + 1::DWordWL, timestamp, value[1]]` | -w2 |
| `MEMW_A-CM19.i` | i ∈ [2, 3] | `memory[is_register, base_address::DWordWL + i::DWordWL, old_timestamp, old[i]]` | w4 |
| `MEMW_A-CM20.i` | i ∈ [2, 3] | `memory[is_register, base_address::DWordWL + i::DWordWL, timestamp, value[i]]` | -w4 |
| `MEMW_A-CM21.i` | i ∈ [4, 7] | `memory[is_register, base_address::DWordWL + i::DWordWL, old_timestamp, old[i]]` | write8 |
| `MEMW_A-CM22.i` | i ∈ [4, 7] | `memory[is_register, base_address::DWordWL + i::DWordWL, timestamp, value[i]]` | -write8 |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW_A-CO23` | `MEMW[old; is_register, base_address::DWordWL, value, timestamp, write2, write4, write8]` | -μ_read |
| `MEMW_A-CO24` | `MEMW[is_register, base_address::DWordWL, value, timestamp, write2, write4, write8]` | -μ_write |

### Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `is_register` | `0` |
| `base_address` | `0` |
| `value` | `0` |
| `timestamp` | `0` |
| `write2` | `0` |
| `write4` | `0` |
| `write8` | `0` |
| `old` | `0` |
| `old_timestamp` | `0` |
| `μ_read` | `0` |
| `μ_write` | `0` |

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
old_timestamp := [old_timestamp_lo, timestamp[1]]::DWordWL
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
| `MEMW_R-C2.i` | i ∈ [0, 1] | `memory[1, [(2 * address + i)::Word, 0], old_timestamp, old[i]]` | μ_sum |
| `MEMW_R-C3.i` | i ∈ [0, 1] | `memory[1, [(2 * address + i)::Word, 0], timestamp, val[i]]` | -μ_sum |

This chip can either just write (``μ_write` = 1`), or both read and write (``μ_read` = 1`) in the same cycle. It must be asserted that at most one of these two options is selected:

| Tag | Description |
|-----|-------------|
| `MEMW_R-C4` | `IS_BIT<μ_read>` |
| `MEMW_R-C5` | `IS_BIT<μ_write>` |
| `MEMW_R-C6` | `IS_BIT<μ_sum>` |

Lastly, this chip contributes the following interactions to the logup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW_R-C7` | `MEMW[[old[0], old[1], 0, 0, 0, 0, 0, 0]; 1, [(2 * address)::Word, 0], [val[0], val[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | -μ_read |
| `MEMW_R-C8` | `MEMW[1, [(2 * address)::Word, 0], [val[0], val[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | -μ_write |

### Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `address` | `0` |
| `timestamp` | `0` |
| `val` | `0` |
| `old` | `0` |
| `old_timestamp_lo` | `0` |
| `μ_read` | `0` |
| `μ_write` | `0` |

## Notes/optimizations

The following ideas may prove to be optimizations for the // chip: - `MEMB` chip that does a one-byte write to remove old_timestamp from here (uncertain tradeoffs) - Adding `μ_sum`/`w2`/`w4`/`write8` multiplicities to the `IS_HALF` lookups may make some GKR things faster if there are known zeroes. - For the register fast-path, one may upgrade the `IS_HALF` check to an `IS_B20` check for extended range at the cost of looking through a larger table.

---

# LOAD Chip

The  chip provides functionality to read values from memory and sign-extend them where appropriate. It delegates low-level memory handling to the `MEMW` chip ([memw]).

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `base_address` | `DWordWL` | The base address to read from, gets offset by $[0, 7]$, depending on how big the access is |
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
| `LOAD-A2.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

## Constraints

The chip delegates the actual memory interaction to the `MEMW` chip, and ensures correctness of the requested sign/zero extension. The output `res` is correctly range-checked as long as the memory contents are.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `LOAD-C1` |  | `IS_BIT<signed>` |  |
| `LOAD-C2` |  | `IS_BIT<read2>` |  |
| `LOAD-C3` |  | `IS_BIT<read4>` |  |
| `LOAD-C4` |  | `IS_BIT<read8>` |  |
| `LOAD-C5` |  | `IS_BIT<read2 + read4 + read8>` |  |
| `LOAD-C6` |  | `read2` + `read4` + `read8` => `μ` |  |
| | | _polynomial:_ `(read2 + read4 + read8) * (1 - μ) = 0` | |
| `LOAD-C7` |  | `MEMW[res; 0, base_address, res::BaseField[8], timestamp, read2, read4, read8]` | μ |
| `LOAD-C8` |  | `MSB8[sign_bit; res[0]]` | read1 |
| `LOAD-C9` |  | `MSB8[sign_bit; res[1]]` | read2 |
| `LOAD-C10` |  | `MSB8[sign_bit; res[3]]` | read4 |
| `LOAD-C11.i` | i ∈ [4, 7] | !`read8` => `res`_i = `signed` dot `sign_bit` dot 255 |  |
| | | _polynomial:_ `(1 - read8) * (res[i] - signed * sign_bit * 255) = 0` | |
| `LOAD-C12.i` | i ∈ [2, 3] | !(`read4` + `read8`) => `res`_i = `signed` dot `sign_bit` dot 255 |  |
| | | _polynomial:_ `(1 - read4 - read8) * (res[i] - signed * sign_bit * 255) = 0` | |
| `LOAD-C13` |  | !(`read2` + `read4` + `read8`) => `res`_1 = `signed` dot `sign_bit` dot 255 |  |
| | | _polynomial:_ `(1 - read2 - read4 - read8) * (res[1] - signed * sign_bit * 255) = 0` | |

The chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LOAD-C14` | `MEMOP[res::DWordWL; timestamp, base_address, 0::DWordWL, 2 * signed + 4 * read2 + 8 * read4 + 16 * read8]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `base_address` | `0` |
| `timestamp` | `0` |
| `read2` | `0` |
| `read4` | `0` |
| `read8` | `0` |
| `signed` | `0` |
| `res` | `0` |
| `sign_bit` | `0` |
| `μ` | `0` |

---

# STORE Chip

The  chip provides functionality to store a value to memory. It decomposes a `DWord` into bytes and delegates low-level memory handling to the `MEMW` chip ([memw]).

## Variables

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `base_address` | `DWordWL` | The base address to write to, gets offset by $[0, 7]$, depending on how big the access is |
| `timestamp` | `DWordWL` | The timestamp at which this memory access is said to occur |
| `write2` | `Bit` | Whether to write exactly 2 bytes |
| `write4` | `Bit` | Whether to write exactly 4 bytes |
| `write8` | `Bit` | Whether to write exactly 8 bytes |
| `value` | `DWordBL` | The value to store |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `write1` | `Bit` | Whether to write exactly 1 byte |

**Definition of `write1`:**
```
write1 := μ - write2 - write4 - write8
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `STORE-A1.i` | i ∈ [0, 1] | `IS_WORD[base_address[i]]` |
| `STORE-A2.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

## Constraints

The chip delegates the actual memory interaction to the `MEMW` chip, and ensures the values are proper bytes.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `STORE-C1` |  | `IS_BIT<μ>` |  |
| `STORE-C2` |  | `IS_BIT<write2>` |  |
| `STORE-C3` |  | `IS_BIT<write4>` |  |
| `STORE-C4` |  | `IS_BIT<write8>` |  |
| `STORE-C5` |  | `IS_BIT<write2 + write4 + write8>` |  |
| `STORE-C6` |  | `write2` + `write4` + `write8` => `μ` = 1 |  |
| | | _polynomial:_ `(write2 + write4 + write8) * (1 - μ) = 0` | |
| `STORE-C7.i` | i ∈ [0, 7] | μ ⇒ `IS_BYTE<value[i]>` |  |
| `STORE-C8` |  | `MEMW[0, base_address, value, timestamp, write2, write4, write8]` | μ |

The chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `STORE-C9` | `MEMOP[0::DWordWL; timestamp, base_address, value::DWordWL, 1 + 4 * write2 + 8 * write4 + 16 * write8]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `base_address` | `0` |
| `timestamp` | `0` |
| `write2` | `0` |
| `write4` | `0` |
| `write8` | `0` |
| `value` | `0` |
| `μ` | `0` |

---

# About ECALL

ECALLs provide system-level functionalities to the guest program.

When `ECALL` is executed, it is assumed that: - register `A7` contains the system call number

- the arguments are located in registers `A0`-`A6`, and - the return value is written to `A0`, where `A0`-`A7` are symbolic names for the registers `x10`-`x17`

## ECALL number overview

We provide a list of supported ECALL numbers. Negative numbers (represented as 2s complement 64-bit numbers), are used for our own custom accelerators/extensions.

/ 64: `write` ([commit]) / 93: `exit` ([halt]) / -1: `SHA256` ([sha256]) / -2: `KECCAK` ([keccak]) / -3: `ECSM` ([ecsm])

---

# HALT Chip

## Variables

The  chip leverages  variable, spanning  columns and leverages  interactions:

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which to halt the program |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | The `next_pc` value the CPU wrote during the instruction HALT was invoked |

## Assumptions

It is assumed the input is range checked:

| Tag | Range | Description |
|-----|-------|-------------|
| `HALT-A1.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

## Constraints

The  chip: + makes sure register `x10` (containing the exit code) equals `0` ([halt:c:read_zero_exit_code]), + writes `0` to all other registers ([halt:c:zeroize_registers_lo]/[halt:c:zeroize_registers_hi]), and + sets `pc` equal to `1` ([halt:c:consume_pc], [halt:c:emit_pc]). Note that the writes performed by all these interactions --- except for the `pc` --- are accompanied by the timestamp `2^64-1`; the maximum timestamp. This prevents any other operation involving memory from being executed hereafter. The `pc` is consumed and re-emitted at the same timestamp to enable padding rows for the CPU. This means that the verifier will have to know the final timestamp at which a CPU padding `pc` was written to be able to balance the final LogUp.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `HALT-C1.i` | i ∈ [1, 9] | `MEMW[1, (2 * i)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C2` |  | `MEMW[0::BaseField[8]; 1, (2 * 10)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C3.i` | i ∈ [11, 31] | `MEMW[1, (2 * i)::DWordWL, 0::BaseField[8], (2^64 - 1)::DWordWL, 1, 0, 0]` | 1 |
| `HALT-C4.i` | i ∈ [0, 1] | `memory[1, [2 * 255 + i, 0], [timestamp[0] + 1, timestamp[1]], pc[i]]` | 1 |
| `HALT-C5.i` | i ∈ [0, 1] | `memory[1, [2 * 255 + i, 0], [timestamp[0] + 1, timestamp[1]], [1, 0][i]]` | -1 |

[ Observe that --- in its current state --- this solution puts the burden of verifying the register cleanup on the verifier inside of the lookup argument. Alternatively, one could add 31 lookups to the "memory" table to remove the _known_ final tokens for the registers there. ])

### Lookup

In this VM, halting is considered equivalent to executing a `sys_exit`. Hence, this chip responds to `ECALL`s with system call number 93.

The HALT chip therefore contributes the following interaction to the lookup-argument:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `HALT-C6` | `ECALL[timestamp, 93::DWordWL]` | -1 |

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
| `COMMIT-C2` | `MEMW[[address[0], address[1], 0, 0, 0, 0, 0, 0]; 1, (2 * 11)::DWordWL, [address[0], address[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | first |
| `COMMIT-C3` | `MEMW[[count[0], count[1], 0, 0, 0, 0, 0, 0]; 1, (2 * 12)::DWordWL, [count[0], count[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | first |
| `COMMIT-C4` | `MEMW[[1, 0, 0, 0, 0, 0, 0, 0]; 1, (2 * 10)::DWordWL, [count[0], count[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | first |
| `COMMIT-C5` | `MEMW[[index, 0, 0, 0, 0, 0, 0, 0]; 1, (2 * 254)::DWordWL, [index + count::BaseField, 0, 0, 0, 0, 0, 0, 0], timestamp, 0, 0, 0]` | first |

*Note*: the observant reader will notice that [commit:c:read_index] casts `count` to a `BaseField`, potentiallly losing information. This is indeed correct. However, since it is practically impossible to commit more than `2^64-2^32` bytes in a single VM execution, it was decided to permit this.

Next, we read the `value` located at buffer address `address` and commit to it under the given `index`. This is only performed when we have not yet reached the `end` of the commit sequence.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `COMMIT-C6` | `MEMW[[value, 0, 0, 0, 0, 0, 0, 0]; 0, address, [value, 0, 0, 0, 0, 0, 0, 0], timestamp, 0, 0, 0]` | μ - end |
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

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `index` | `0` |
| `address` | `[0, 0, 0, 0]` |
| `address_incr` | `[1, 0, 0, 0]` |
| `count` | `[1, 0, 0, 0]` |
| `count_decr` | `[0, 0, 0, 0]` |
| `first` | `0` |
| `end` | `0` |
| `value` | `0` |
| `μ` | `0` |

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
| `SHA256-C1` |  | `MEMW[[(m_addr[0]::DWordWL)[0], (m_addr[0]::DWordWL)[1], 0, 0, 0, 0, 0, 0]; 1, (2 * 11)::DWordWL, [(m_addr[0]::DWordWL)[0], (m_addr[0]::DWordWL)[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | μ |
| `SHA256-C2.i` | i ∈ [0, 7], j ∈ [0, 3] | `IS_HALF[m_addr[i][j]]` | μ |
| `SHA256-C3.i` | i ∈ [1, 7] | `ADD<m_addr[i]::DWordWL; m_addr[0]::DWordWL, (8 * i)::DWordWL>` |  |
| `SHA256-C4.i` | i ∈ [0, 7] | `MEMW[[m[8 * i + 3], m[8 * i + 2], m[8 * i + 1], m[8 * i + 0], m[8 * i + 7], m[8 * i + 6], m[8 * i + 5], m[8 * i + 4]]; 0, m_addr[i]::DWordWL, [m[8 * i + 3], m[8 * i + 2], m[8 * i + 1], m[8 * i + 0], m[8 * i + 7], m[8 * i + 6], m[8 * i + 5], m[8 * i + 4]], timestamp, 0, 0, 1]` | μ |
| `SHA256-C5` |  | `MEMW[[(h_addr[0]::DWordWL)[0], (h_addr[0]::DWordWL)[1], 0, 0, 0, 0, 0, 0]; 1, (2 * 10)::DWordWL, [(h_addr[0]::DWordWL)[0], (h_addr[0]::DWordWL)[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | μ |
| `SHA256-C6.i` | i ∈ [0, 3], j ∈ [0, 3] | `IS_HALF[h_addr[i][j]]` | μ |
| `SHA256-C7.i` | i ∈ [1, 3] | `ADD<h_addr[i]::DWordWL; h_addr[0]::DWordWL, 8 * i::DWordWL>` |  |
| `SHA256-C8.i` | i ∈ [0, 3] | `MEMW[[h[8 * i + 3], h[8 * i + 2], h[8 * i + 1], h[8 * i + 0], h[8 * i + 7], h[8 * i + 6], h[8 * i + 5], h[8 * i + 4]]; 0, h_addr[i]::DWordWL, [out[8 * i + 3], out[8 * i + 2], out[8 * i + 1], out[8 * i + 0], out[8 * i + 7], out[8 * i + 6], out[8 * i + 5], out[8 * i + 4]], timestamp + 1::DWordWL, 0, 0, 1]` | μ |

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
| `SHA256-C13` |  | `SHA256ROUND[timestamp, [2^0 * h[3] + 2^8 * h[2] + 2^16 * h[1] + 2^24 * h[0], 2^0 * h[7] + 2^8 * h[6] + 2^16 * h[5] + 2^24 * h[4], 2^0 * h[11] + 2^8 * h[10] + 2^16 * h[9] + 2^24 * h[8], 2^0 * h[15] + 2^8 * h[14] + 2^16 * h[13] + 2^24 * h[12], 2^0 * h[19] + 2^8 * h[18] + 2^16 * h[17] + 2^24 * h[16], 2^0 * h[23] + 2^8 * h[22] + 2^16 * h[21] + 2^24 * h[20], 2^0 * h[27] + 2^8 * h[26] + 2^16 * h[25] + 2^24 * h[24], 2^0 * h[31] + 2^8 * h[30] + 2^16 * h[29] + 2^24 * h[28]], 0]` | μ |
| `SHA256-C14` |  | `SHA256ROUND[timestamp, last_round_out, 64]` | -μ |
| `SHA256-C15.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<out[i]>` |  |
| `SHA256-C16.i` | i ∈ [0, 7] | `ADD<[0, 2^0 * out[4 * i + 3] + 2^8 * out[4 * i + 2] + 2^16 * out[4 * i + 1] + 2^24 * out[4 * i + 0]]; [0, last_round_out[i]], [0, 2^0 * h[4 * i + 3] + 2^8 * h[4 * i + 2] + 2^16 * h[4 * i + 1] + 2^24 * h[4 * i + 0]]>` |  |

In this VM, we assign syscall number -1 to the  accelerator. The chip therefore contributes the following interaction to the lookup-argument:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHA256-C17` | `IS_BIT<μ>` |  |
| `SHA256-C18` | `ECALL[timestamp, (2^64 - 1)::DWordWL]` | -μ |

### Padding

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `h` | `0` |
| `h_addr` | `[0, 8, 16, 24]` |
| `m` | `0` |
| `m_addr` | `[0, 8, 16, 24, 32, 40, 48, 56]` |
| `out` | `0` |
| `last_round_out` | `0` |
| `μ` | `0` |

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
| `SHA256MSGSCHED-C1` | μ ⇒ `IS_BYTE<index - 16>` |  |
| `SHA256MSGSCHED-C2` | `SHA256_M[back2; timestamp, index - 2]` | μ |
| `SHA256MSGSCHED-C3` | `SHA256_M[back7; timestamp, index - 7]` | μ |
| `SHA256MSGSCHED-C4` | `SHA256_M[back15; timestamp, index - 15]` | μ |
| `SHA256MSGSCHED-C5` | `SHA256_M[back16; timestamp, index - 16]` | μ |

Then, we calculate the result. It suffices to check that the carry of adding four range-checked words into a range-checked word is not too big, following the logic from [add]. In this case, using the `IS_BYTE` constraint allows us to add multiple words together at the same time, without needing to store and range-check intermediate results.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHA256MSGSCHED-C6` |  | `ROTXOR[s0; back15, 2, 11, 3, 0]` | μ |
| `SHA256MSGSCHED-C7` |  | `ROTXOR[s1; back2, 3, 2, 10, 0]` | μ |
| `SHA256MSGSCHED-C8` |  | μ ⇒ `IS_BYTE<carry>` |  |
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
| `SHA256ROUND-C1.i` | i ∈ [0, 3] | `BYTE_ALU[a_and_b[i]; ⧼AND⧽, a[i], b[i]]` | μ |
| `SHA256ROUND-C2.i` | i ∈ [0, 3] | `BYTE_ALU[a_xor_b[i]; ⧼XOR⧽, a[i], b[i]]` | μ |
| `SHA256ROUND-C3.i` | i ∈ [0, 3] | `BYTE_ALU[c_and_a_xor_b[i]; ⧼AND⧽, c[i], a_xor_b[i]]` | μ |
| `SHA256ROUND-C4.i` | i ∈ [0, 3] | `BYTE_ALU[e_and_f[i]; ⧼AND⧽, e[i], f[i]]` | μ |
| `SHA256ROUND-C5.i` | i ∈ [0, 3] | `BYTE_ALU[not_e_and_g[i]; ⧼AND⧽, 255 - e[i], g[i]]` | μ |
| `SHA256ROUND-C6` |  | `SHA256_K[kval; index]` | μ |
| `SHA256ROUND-C7` |  | `SHA256_M[wval; timestamp, index]` | μ |
| `SHA256ROUND-C8` |  | `ROTXOR[S0; a::Word, 6, 9, 2, 1]` | μ |
| `SHA256ROUND-C9` |  | `ROTXOR[S1; e::Word, 9, 14, 6, 1]` | μ |

Then we constrain the addition for the new state, constraining additions with the same `IS_BYTE` trick as before.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHA256ROUND-C10.i` | i ∈ [0, 1] | `IS_HALF[out_a[i]]` | μ |
| `SHA256ROUND-C11` |  | μ ⇒ `IS_BYTE<carry_a>` |  |
| `SHA256ROUND-C12.i` | i ∈ [0, 1] | `IS_HALF[out_e[i]]` | μ |
| `SHA256ROUND-C13` |  | μ ⇒ `IS_BYTE<carry_e>` |  |

Finally, we chain the rounds together through the interactions.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHA256ROUND-C14` | `SHA256ROUND[timestamp, [a::Word, b::Word, c::Word, d, e::Word, f::Word, g::Word, h], index]` | -μ |
| `SHA256ROUND-C15` | `SHA256ROUND[timestamp, [out_a::Word, a::Word, b::Word, c::Word, out_e::Word, e::Word, f::Word, g::Word], index + 1]` | μ |

### Padding

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `a` | `0` |
| `b` | `0` |
| `c` | `0` |
| `d` | `0` |
| `e` | `0` |
| `f` | `0` |
| `g` | `0` |
| `h` | `0` |
| `index` | `0` |
| `out_a` | `0` |
| `out_e` | `0` |
| `a_and_b` | `0` |
| `a_xor_b` | `0` |
| `c_and_a_xor_b` | `0` |
| `e_and_f` | `0` |
| `not_e_and_g` | `0` |
| `kval` | `0` |
| `S0` | `0` |
| `S1` | `0` |
| `wval` | `0` |
| `μ` | `0` |

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
| `ROTXOR-C1.i` | i ∈ [0, 1] | `HWSL[[a0_left[i], a0_right[i]]; a[i], 16 - r0]` | μ |
| `ROTXOR-C2.i` | i ∈ [0, 1] | `HWSL[[a1_left[i], a1_right[i]]; (a0::WordHL)[i], r1]` | μ |
| `ROTXOR-C3.i` | i ∈ [0, 1] | `HWSL[[a2_left[i], a2_right[i]]; a[i], 16 - r2]` | μ |
| `ROTXOR-C4.i` | i ∈ [0, 1] | `a0[i]` = `a0_left[i]` + `a0_right[1 - i]` |  |
| | | _polynomial:_ `(a0::WordHL)[i] - a0_left[i] - a0_right[1 - i] = 0` | |
| `ROTXOR-C5.i` | i ∈ [0, 1] | `a1[i]` = `a1_left[i]` + `a1_right[1 - i]` |  |
| | | _polynomial:_ `(a1::WordHL)[i] - a1_left[i] - a1_right[1 - i] = 0` | |
| `ROTXOR-C6` |  | `a2[0]` = `a2_left[1]` + `a2_right[0]` |  |
| | | _polynomial:_ `(a2::WordHL)[0] - a2_left[1] - a2_right[0] = 0` | |
| `ROTXOR-C7` |  | `a2[1]` = `last_rot` dot `a2_left[0]` + `a2_right[1]` |  |
| | | _polynomial:_ `(a2::WordHL)[1] - last_rot * a2_left[0] - a2_right[1] = 0` | |

Then the bitwise XOR of the results.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ROTXOR-C8.i` | i ∈ [0, 3] | `BYTE_ALU[a01[i]; ⧼XOR⧽, a0[i], a1[i]]` | μ |
| `ROTXOR-C9.i` | i ∈ [0, 3] | `BYTE_ALU[out[i]; ⧼XOR⧽, a01[i], a2[i]]` | μ |

And finally contribute to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `ROTXOR-C10` | `ROTXOR[out::Word; a::Word, r0, r1, r2, last_rot]` | -μ |

### Padding

| Column | Padding value |
|--------|---------------|
| `a` | `0` |
| `r0` | `0` |
| `r1` | `0` |
| `r2` | `0` |
| `last_rot` | `0` |
| `out` | `0` |
| `a0_left` | `0` |
| `a0_right` | `0` |
| `a1_left` | `0` |
| `a1_right` | `0` |
| `a2_left` | `0` |
| `a2_right` | `0` |
| `a0` | `0` |
| `a1` | `0` |
| `a2` | `0` |
| `a01` | `0` |
| `μ` | `0` |

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

---

# KECCAK Accelerator

The  chip applies the keccak permutation `kappa` to a given memory range; other aspects of keccak hashing (such as repeated permutation invocation, input padding and state initialization) fall outside the scope of this accelerator.

This permutation `kappa: FF_2^1600 -> FF_2^1600` operates on 1600 bits and is composed of 24 applications of round-permutation `Lambda: FF_2^1600 times NN -> FF_2^1600`, where the additional parameter is the round constant. `Lambda` is defined as the composition `iota compose chi compose pi compose rho compose theta`, where only `iota` depends on the round constant.

The keccak accelerator comprises two chips: a core chip that interacts with the memory --- loading the input and writing the output, and a round chip that applies the round permutation.

## Core chip

### Columns

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which the permutation is performed |
| `addr` | `DWordBL` | memory address storing the first bit of the state |
| `input_state` | `[['Byte', 8], 5][5]` | state at the start of executing the permutation |

### Output

| Name | Type | Description |
|------|------|-------------|
| `output_state` | `[['Byte', 8], 5][5]` | state after executing the permutation |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `state_ptr` | `['DWordHL', 5][5]` | memory addresses storing the entire state |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

### Constraints

In this VM, we assign syscall number -2 to the  accelerator. The chip therefore contributes the following interaction to the lookup-argument:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `KECCAK-C1` | `ECALL[timestamp, (2^64 - 2)::DWordWL]` | -μ |

The address containing the state to be permuted is passed in as argument `A0 = x10`. The following constraints describe that this address is read into `addr` ([keccak:c:read_addr]), from which `state_ptr` --- the collection of pointers to all lanes of the state --- is derived ([keccak:c:state_ptr]). The state is then read into `input_state`, while the `output_state` is written back to the indicated address ([keccak:c:load_store_state]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `KECCAK-C2` |  | `MEMW[addr; 1, (2 * 10)::DWordWL, addr, timestamp, 1, 0, 0]` | μ |
| `KECCAK-C3.i` | x ∈ [0, 4], y ∈ [0, 4] | `ADD<state_ptr[x][y]::DWordWL; addr::DWordWL, (8 * (5 * y + x))::DWordWL>` |  |
| `KECCAK-C4.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 3] | `IS_HALF[state_ptr[x][y][z]]` | μ |
| `KECCAK-C5.i` | x ∈ [0, 4], y ∈ [0, 4] | `MEMW[input_state[x][y]; 0, state_ptr[x][y]::DWordWL, output_state[x][y], timestamp, 0, 0, 1]` | μ |

Lastly, the input state is pushed to the Keccak-round function, while the output after 24 rounds is taken off the bus:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `KECCAK-C6` | `KECCAK[timestamp, 0, input_state]` | μ |
| `KECCAK-C7` | `KECCAK[timestamp, 24, output_state]` | -μ |

### Padding

The  table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `addr` | `0` |
| `input_state` | `0` |
| `output_state` | `0` |
| `state_ptr` | `8 * [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24]]` |
| `μ` | `0` |

## Round chip

### Columns

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which the permutation is performed |
| `round` | `BaseField` | index of the permutation round |
| `start` | `[['Byte', 8], 5][5]` | state at the start of executing the permutation |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `Cxz` | `[['Byte', 8], 4][5]` | $xor_(i=0)^(y+2) `start[x,i,z]`$ |
| `Cxz_left` | `['Byte', 8][5]` | the left-rotated component of `rotated_Cxz` |
| `Cxz_right` | `['Bit', 4][5]` | the right-rotated component of `rotated_Cxz` (which is a single bit) |
| `Dxz` | `['Byte', 8][5]` | $`Cxz[`\(`x` - 1) mod 5`,y,z]` xor `rotated_Cxz[`\(`x` + 1) mod 5`,y,z]`$ |
| `theta` | `[['Byte', 8], 5][5]` | $theta(`start`)$, the state after applying $theta$. |
| `rot_left` | `[['Byte', 8], 5][5]` | the left-rotated component of $`theta[x,y]` <<< `rnc`$ |
| `rot_right` | `[['Byte', 8], 5][5]` | the right-rotated component of $`theta[x,y]` <<< `rnc`$ |
| `chi_ANDs` | `[['Byte', 8], 5][5]` | $(`pi[`\(x+1) mod 5`,y,z]` xor 255) times.o `pi[`\(x + 2) mod 5`,y,z]`$ |
| `chi` | `[['Byte', 8], 5][5]` | $(chi compose pi compose rho compose theta)(`start`)$; the state after applying $chi$ |
| `rc` | `Byte[8]` | round constants |
| `iota` | `Byte[8]` | state update following from step $iota$. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `rotated_Cxz` | `['Byte', 8][5]` | $`Cxz[x,`3`,z]` <<< 1$ |
| `out` | `[['Byte', 8], 5][5]` | state at the end of executing the permutation |
| `rho` | `[['Byte', 8], 5][5]` | $(rho compose theta)(`start`)$; the state after applying $rho$ |
| `pi` | `[['Byte', 8], 5][5]` | $(pi compose rho compose theta)(`start`)$; the state after applying $pi$ |

**Definition of `rotated_Cxz`:**
```
rotated_Cxz := Cxz_left[x][z] + Cxz_right[x][3]
rotated_Cxz := Cxz_left[x][z]
rotated_Cxz := Cxz_left[x][z] + Cxz_right[x][0]
rotated_Cxz := Cxz_left[x][z]
rotated_Cxz := Cxz_left[x][z] + Cxz_right[x][1]
rotated_Cxz := Cxz_left[x][z]
rotated_Cxz := Cxz_left[x][z] + Cxz_right[x][2]
rotated_Cxz := Cxz_left[x][z]
```

**Definition of `out`:**
```
out := iota[z]
out := chi[x][y][z]
out := chi[x][y][z]
out := chi[x][y][z]
```

**Definition of `rho`:**
```
rho := (1 - rbc[x][y][0]) * (1 - rbc[x][y][1]) * (rot_left[x][y][z] + rot_right[x][y][(z - 2) mod 8]) + rbc[x][y][0] * (1 - rbc[x][y][1]) * (rot_left[x][y][(z - 2) mod 8] + rot_right[x][y][(z - 4) mod 8]) + (1 - rbc[x][y][0]) * rbc[x][y][1] * (rot_left[x][y][(z - 4) mod 8] + rot_right[x][y][(z - 6) mod 8]) + rbc[x][y][0] * rbc[x][y][1] * (rot_left[x][y][(z - 6) mod 8] + rot_right[x][y][z])
```

**Definition of `pi`:**
```
pi := rho[(x + 3 * y) mod 5][x][z]
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

`start` contains the state to which the permutation should be applied. Its three-dimensional array mimics the specification's three-dimensional state

and orders the bits as prescribed.

Rho rotates every lane by a rotation offset in `[0, 64)`. These offsets are identical for every round.

We decompose each offset in three components: the lower nibble (4 bits) are represented by `rnc`, while the upper two bits are represented by as `Bit`s in `rbc`. That is, ``rho_offset[x][y]` = `rnc[x][y]` + 16 dot `rbc[x][y][0]` + 32 dot `rbc[x][y][1]``.

### Constraints

The following constraints ensure that `theta` captures the state after applying the first subpermutation of the round-permutation: `theta`. Note here that `Cxz_left` and `Cxz_right` do have to be range-checked; it cannot be assumed that this implicitly follows from [keccak:c:Dxz] combined with `rotated_Cxz`'s definition.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `KECCAK_RND-C1.i` | x ∈ [0, 4], z ∈ [0, 7] | `BYTE_ALU[Cxz[x][0][z]; ⧼XOR⧽, start[x][0][z], start[x][1][z]]` | μ |
| `KECCAK_RND-C2.i` | x ∈ [0, 4], y ∈ [2, 4], z ∈ [0, 7] | `BYTE_ALU[Cxz[x][y - 1][z]; ⧼XOR⧽, Cxz[x][y - 2][z], start[x][y][z]]` | μ |
| `KECCAK_RND-C3.i` | x ∈ [0, 4], z ∈ [0, 3] | `HWSL[[(Cxz_left[x]::DWordHL)[z], Cxz_right[x][z]::Half]; (Cxz[x][3]::DWordHL)[z], 1]` | μ |
| `KECCAK_RND-C4.i` | x ∈ [0, 4], z ∈ [0, 7] | μ ⇒ `IS_BYTE<Cxz_left[x][z]>` |  |
| `KECCAK_RND-C5.i` | x ∈ [0, 4], z ∈ [0, 3] | `IS_BIT<Cxz_right[x][z]>` |  |
| `KECCAK_RND-C6.i` | x ∈ [0, 4], z ∈ [0, 7] | `BYTE_ALU[Dxz[x][z]; ⧼XOR⧽, Cxz[(x - 1) mod 5][3][z], rotated_Cxz[(x + 1) mod 5][z]]` | μ |
| `KECCAK_RND-C7.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 7] | `BYTE_ALU[theta[x][y][z]; ⧼XOR⧽, start[x][y][z], Dxz[x][z]]` | μ |

Next, we constrain that `rho` captures the state after applying subpermutation `rho`. Note here as well that `rot_left` and `rot_right` do have to be range-checked; it cannot be assumed that this implicitly follows from later constraints.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `KECCAK_RND-C8.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 3] | `HWSL[[(rot_left[x][y]::DWordHL)[z], (rot_right[x][y]::DWordHL)[z]]; (theta[x][y]::DWordHL)[z], rnc[x][y]]` | μ |
| `KECCAK_RND-C9.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 7] | μ ⇒ `IS_BYTE<rot_left[x][y][z]>` |  |
| `KECCAK_RND-C10.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 7] | μ ⇒ `IS_BYTE<rot_right[x][y][z]>` |  |

Observe that the lane-permutation performed by `pi` is absorbed in `pi`'s definition. The next permutation that is constrained in `chi`:

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `KECCAK_RND-C11.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 7] | `BYTE_ALU[chi_ANDs[x][y][z]; ⧼AND⧽, 255 - pi[(x + 1) mod 5][y][z], pi[(x + 2) mod 5][y][z]]` | μ |
| `KECCAK_RND-C12.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 7] | `BYTE_ALU[chi[x][y][z]; ⧼XOR⧽, pi[x][y][z], chi_ANDs[x][y][z]]` | μ |

Lastly, the round constants are added to one of the lanes in the state. `iota` contains the updated lane. In the definition of `out`, the output of `chi` and `iota` is combined to construct the output of the permutation.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `KECCAK_RND-C13.i` | z ∈ [0, 7] | `BYTE_ALU[iota[z]; ⧼XOR⧽, chi[0][0][z], rc[z]]` | μ |

Lastly, the round chip contributes the following interactions to the lookup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `KECCAK_RND-C14` | `KECCAK[timestamp, round, start]` | -μ |
| `KECCAK_RND-C15` | `KECCAK[timestamp, round + 1, out]` | μ |
| `KECCAK_RND-C16` | `KECCAK_RC[rc; round]` | -μ |

### Notes/potential optimizations

- one does not have to repeat `addr` in `state_ptr`; this saves 4 columns and 4 `IS_HALF` checks. - step `rho` does not need to be applied to `state[0][0]`; its has a zero-shift. This saves 16 columns and 4 `HWSL` interactions. - when the output of `HWSL` are `Byte`s mapped as `Half`s, we find that out of every four output bytes, at least one is zero. Since `rnc` is constant, [keccak:c:rho_rotation] makes those zero-bytes show up in `rot_left` and `rot_right` at constant locations. This means 96 columns can be removed from the chip at no cost. Likewise, 96 `IS_BYTE` interactions can be dropped from [keccak:c:range_rot_left] and [keccak:c:range_rot_right]. - the shift-constants are equivalent to `1 mod 16` for `(`x`, `y`) = (1, 0)` and `-1 mod 16` for `(2, 3)`. This means that for those lanes it suffices to constrain `rot_left`/`rot_right` as `Bit`s rather than `Byte`s, saving an additional 8 `IS_BYTE` interactions. - ``rc[2]` = `rc[4]` = `rc[5]` = `rc[6]` = 0`. As such, those elements need not be stored in `rc`, and need not be XORed into the state in the `iota`-step. This saves 8 columns and 4 `XOR_BYTE` interactions. - when executed in large volumnes, `KECCAK_RND` could benefit from having a three-way XOR lookup table. With this in place, the 80 interactions in [keccak:c:theta_cxz_start] and [keccak:c:theta_cxz] could be dropped. Likewise, 80 columns could be removed from the chip (a \~5% savings).

## Round constant lookup

### Columns

We provide the round constants through a short precomputed lookup table: .

### Input

| Name | Type | Description |
|------|------|-------------|
| `round` | `BaseField` |  |
| `RC` | `Byte[8]` | round constants for the given `round` |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` |  |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `KECCAK_RC-C1` | `KECCAK_RC[RC; round]` | -μ |

---

# ECSM Accelerator

## Theory behind Elliptic Curves

An elliptic curve `E(a, b, p)` in _short Weierstrass_ form has parameters `a,b in FF_p` for some prime `p` with `4a^3+27b^2 eq.not 0`, and coordinates `(x, y) in FF_p^2` satisfying the equation $ y^2=x^3+a x+b. $

Additionally, there is the _point at infinity_, `⁠`, which has no native short-Weierstrass representation. It acts as the identity element (zero) in the group: given non-zero curve point `P`, it holds that $

$

The negation of curve point `P = (x_P, y_P)` is constructed as `-P := (x_P, -y_P)`. Naturally, `P + (-P) = `.

The addition of points `P, Q` distinguishes three cases. For `x_P eq.not x_Q`, one uses $ (x_R, y_R) := (lambda^2 - x_P - x_Q, lambda (x_P - x_R) - y_P) $ with `lambda = frac((y_Q - y_P), (x_Q - x_P), style: "horizontal")`. When `x_P = x_Q` and `y_P eq.not - y_Q`, one instead uses `lambda = frac(3x_P^2, 2y_P, style: "horizontal")`. The remaing case that `(x_P, y_P) = (x_Q, -y_Q)` corresponds with `Q = -P`; the addition results in ``.

An addition operation gives rise to an algorithm for scalar multiplication. Given curve point `P` and scalar `k`, the multiple `k times P` can trivially be computed as `P + P + ... + P`. This accelerator instead leverages the _double-and-add_ ) technique, which utilizes only `O(log(k))` additions for the full multiplication.

The purpose of this accelerator is to speed up the scalar multiplication `k times G` for scalar `k in [1, N)` and point `G in E(0, b, p) without {}` with `p in [2^248, 2^256)`. In particular, the accelerator supports the curve ``secp256k1` = E(0, 7, 2^256-2^32 - 977)`. This accelerator leverages _double-and-add_, executing the multiplication in `O(log(k))` doublings and `O(w_H (k)) = O(log(k))` additions, where `w_H (dot)` denotes the hamming-weight of a bitstring.

## Overview

The accelerator comprises three chips: - *`ECSM` (Elliptic Curve Scalar Multiply)*; this chip is responsible for loading inputs `x_G` and `k` from memory, reconstructing `y_G`, dispatching a double-and-add sequence request to the `ECDAS` chip, and writing the result point `x_R` back to memory. - *`ECDAS` (Elliptic Curve Double/Add Sequence)* is responsible for the consecutive doubling/adding the provided point to itself, ultimately arriving at `k times G`. - *`EC_SCALAR`* serves `k` bit-by-bit to the `ECDAS` chip to inform the flow of the double-and-add sequence.

## ECSM <ecsm-sm>

The  (Elliptic Curve Scalar Multiply) chip is generic over the constants - `b`, the second curve coefficient, - `p`, the prime field modulus, and - `N`, the order of the curve group. To support scalar multiplication over different curves, one chip instance should be created for each curve.

The chip is triggered by executing `ECALL`, with the ECALL-number is set to `-3`. The chip expects - `x10` to contain the address where `x_R := (k times G)_x` is to be stored, - `x11` to contain the address at which the least significant byte of `x_G` is to be found, - `x12` to contain the address at which the least significant byte of `k` is to be found, where it is assumed that `x_G, x_R` and `k` are provided as little-endian.

### Columns

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which this operation is executed |
| `addr_xG` | `DWordWL` | address at which `x`-coordinate of start point `G` is stored |
| `addr_k` | `DWordWL` | address at which scalar `k` is stored |
| `addr_xR` | `DWordWL` | address to which the `x`-coordinate of result point `R` is to be written |

### Output

| Name | Type | Description |
|------|------|-------------|
| `xR` | `U256BL` | $(`k` times `G`)_x$ |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `yR` | `U256BL` | $(`k` times `G`)_y$ |
| `k` | `U256BL` | `k` |
| `len_k` | `Byte` | Position of `k`'s most significant 1-bit |
| `xG` | `U256BL` | $x_G$ |
| `yG` | `U256BL` | $y_G$ |
| `x2` | `U256BL` | $x_G^2$ |
| `q0` | `U256BL` | quotient for computing `x2` |
| `c0` | `BaseField[64]` | carries for computing `x2` |
| `q1` | `Byte[33]` | quotient for computing `yG` |
| `c1` | `BaseField[64]` | carries for computing `yG` |
| `k_sub_N` | `U256HL` | $`k`- `N` mod 2^256$ |
| `xR_sub_p` | `U256HL` | $x_R - `p` mod 2^256$ |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `c2` | `Bit[8]` | carries for computing $`N` + `k_sub_N`$ |
| `c3` | `Bit[8]` | carries for computing $`P` + `xR_sub_p`$ |
| `XG` | `Byte[64]` | zero-extension of `xG` |
| `YG` | `Byte[64]` | zero-extension of `yG` |
| `X2` | `Byte[64]` | zero-extension of `x2` |
| `Q0` | `Byte[64]` | zero-extension of `q` |
| `Q1` | `Byte[64]` | zero-extension of `q1` |
| `B` | `Byte[64]` | zero-extension of `b` |
| `P` | `Byte[64]` | zero-extension of `p` |

**Definition of `c2`:**
```
c2 (when iter=['i', 0]) := 2^-32 * ((N::U256WL)[i] + (k_sub_N::U256WL)[i] - (k::U256WL)[i])
c2 (when iter=['i', 1, 7]) := 2^-32 * ((N::U256WL)[i] + (k_sub_N::U256WL)[i] + c2[i - 1] - (k::U256WL)[i])
```

**Definition of `c3`:**
```
c3 (when iter=['i', 0]) := 2^-32 * ((p::U256WL)[i] + (xR_sub_p::U256WL)[i] - (xR::U256WL)[i])
c3 (when iter=['i', 1, 7]) := 2^-32 * ((p::U256WL)[i] + (xR_sub_p::U256WL)[i] + c3[i - 1] - (xR::U256WL)[i])
```

**Definition of `XG`:**
```
XG (when iter=['i', 0, 31]) := xG[i]
XG (when iter=['i', 32, 63]) := 0
```

**Definition of `YG`:**
```
YG (when iter=['i', 0, 31]) := yG[i]
YG (when iter=['i', 32, 63]) := 0
```

**Definition of `X2`:**
```
X2 (when iter=['i', 0, 31]) := x2[i]
X2 (when iter=['i', 32, 63]) := 0
```

**Definition of `Q0`:**
```
Q0 (when iter=['i', 0, 31]) := q0[i]
Q0 (when iter=['i', 32, 63]) := 0
```

**Definition of `Q1`:**
```
Q1 (when iter=['i', 0, 32]) := q1[i]
Q1 (when iter=['i', 33, 63]) := 0
```

**Definition of `B`:**
```
B (when iter=['i', 0, 31]) := b[i]
B (when iter=['i', 32, 63]) := 0
```

**Definition of `P`:**
```
P (when iter=['i', 0, 31]) := p[i]
P (when iter=['i', 32, 63]) := 0
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

### Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `ECSM-A1` |  | $(#`addr_xG` mod 2^32) + 24 < 2^32$ |
| `ECSM-A2` |  | $(#`addr_k` mod 2^32) + 31 < 2^32$ |
| `ECSM-A3` |  | $(#`addr_xR` mod 2^32) + 24 < 2^32$ |

### Constraints

#### Interactions

This chip is triggered by an `ECALL` with the opcode indicating this chip:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `ECSM-C1` | `IS_BIT<μ>` |  |
| `ECSM-C2` | `ECALL[timestamp, [2^32 - 3, 2^32 - 1]]` | -μ |

#### Read `xG`

Once triggered, it loads register `x11` to see where `x_G` is stored in memory ([ec:c:read_addr_xG]) and subsequently load `x_G` in ([ec:c:read_xG]). Assumption [ec:a:addr_xG_alignment] ensures no overflows happen when incrementing the address in [ec:c:read_xG]. Note: `xG` is assumed to be range checked, since they're read from memory.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C3` |  | `MEMW[[addr_xG[0], addr_xG[1], 0, 0, 0, 0, 0, 0]; 1, (2 * 11)::DWordWL, [addr_xG[0], addr_xG[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | μ |
| `ECSM-C4.i` | i ∈ [0, 3] | `MEMW[[xG[8 * i + 0], xG[8 * i + 1], xG[8 * i + 2], xG[8 * i + 3], xG[8 * i + 4], xG[8 * i + 5], xG[8 * i + 6], xG[8 * i + 7]]; 0, addr_xG + (8 * i)::DWordWL, [xG[8 * i + 0], xG[8 * i + 1], xG[8 * i + 2], xG[8 * i + 3], xG[8 * i + 4], xG[8 * i + 5], xG[8 * i + 6], xG[8 * i + 7]], timestamp, 0, 0, 1]` | μ |

#### Constrain `Gy`

With `x_G` read and range checked, we direct our attention to `y_G`. Rather than reading it from memory, the prover provides it as a witness and proves it to be correct. In particular, the chip enforces the relations $ x_G^2 - `x2` - q_0 dot p &= 0,\ y_G^2 - x_G dot `x2` - b + (p - q_1)p &= 0\ $ where non-negative `q_0` and `q_1` are prover-provided witnesses. Note that these are equivalent to $

y_G^2 &equiv x_G dot `x2` + b  mod p\ $ which combine to `y_G^2 equiv x_G^3 + b mod p`. Rewriting the two relations, we get $ q_0 &= (x_G^2 - `x2`) dot p^(-1),\ q_1 &= (y_G^2 - x_G dot `x2`-b) dot p^(-1) + p. $ Using the fact that `x_G, y_G, `x2` in [0, p)`, we find that `q_0 in [0, p)` and `q_1 in [0, 2p)`. We therefore restrict the choice of quotients to `q_0 in [0, 2^256)` and `q_1 in [0, 2^257)`.

Below, we enforce the first of the two sub-relations. We emphasize here that [ec:c:c0_63_is_zero] is required to ensure the sum evaluates to `0`, rather than just `0 mod 2^256`. The constraints [ec:c:c0_0] and [ec:c:c0_i], as well as the magic number `8160` in [ec:c:range_c0] are discussed in [ecsm]-limb_carry.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C5.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<x2[i]>` |  |
| `ECSM-C6.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<q0[i]>` |  |
| `ECSM-C7` |  | 2^8 dot `c0`_0 = `XG`_0 dot `XG`_0 - `X2`_0 - `Q0`_0 dot `P`_0 |  |
| | | _polynomial:_ `XG[0] * XG[0] - X2[0] - Q0[0] * P[0] - 2^8 * c0[0] = 0` | |
| `ECSM-C8.i` | i ∈ [1, 63] | 2^8 dot `c0`_i = `c0`_(i-1) - `X2`_i + sum_(j=0)^i `XG`_j dot `XG`_(i-j) - `Q0`_j dot `P`_(i-j) |  |
| | | _polynomial:_ `(c0[i - 1] - 2^8 * c0[i] - X2[i]) + Σ_j = 0^i (XG[j] * XG[i - j] - Q0[j] * P[i - j]) = 0` | |
| `ECSM-C9` |  | `c0`_63 = 0 |  |
| | | _polynomial:_ `c0[63] = 0` | |
| `ECSM-C10.i` | i ∈ [0, 62] | `IS_HALF[c0[i] + 8160]` | μ |

Next, we restrict the witness pair `(y_G, `q1`)`. Note there that [ec:c:c1_0] and [ec:c:c1_i] multiply `B` by `μ` to simplify the padding; there are no other side-effects to this since ``μ` = 1` on non-padding rows ([ec:c:mu_isbit]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C11.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<yG[i]>` |  |
| `ECSM-C12.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<q1[i]>` |  |
| `ECSM-C13` |  | `IS_BIT<q1[32]>` |  |
| `ECSM-C14` |  | 2^8 dot `c1`_0 = `YG`_0 dot `YG`_0 - `X2`_0 dot `XG`_0 - `μ` dot `B`_0 + `P`_0 dot `P`_0 - `Q1`_0 dot `P`_0 |  |
| | | _polynomial:_ `YG[0] * YG[0] + P[0] * P[0] - X2[0] * XG[0] - μ * B[0] - Q1[0] * P[0] - 2^8 * c1[0] = 0` | |
| `ECSM-C15.i` | i ∈ [1, 63] | 2^8 dot `c1`_i = `c1`_(i-1) - `μ` dot `B`_i + sum_(j=0)^i (`YG`_j dot `YG`_(i-j) + `P`_j dot `P`_(i-j) - `X2`_j dot `XG`_(i-j) - `Q1`_j dot `P`_(i-j)) |  |
| | | _polynomial:_ `(c1[i - 1] - 2^8 * c1[i] - μ * B[i]) + Σ_j = 0^i (YG[j] * YG[i - j] + P[j] * P[i - j] - X2[j] * XG[i - j] - Q1[j] * P[i - j]) = 0` | |
| `ECSM-C16` |  | `c1`_63 = 0 |  |
| | | _polynomial:_ `c1[63] = 0` | |
| `ECSM-C17.i` | i ∈ [0, 62] | `IS_HALF[c1[i] + 16319]` | μ |

#### Read and verify `k`

After reading `addr_k` from `x12` ([ec:c:read_addr_k]), we read `k` from this address ([ec:c:load_k]). Similar to `addr_xG`, assumption [ec:a:addr_k_alignment] ensures the address offsets in [ec:c:load_k] do not overflow the lower limb. To prevent the point at infinity from showing up during the scalar multiplication, we require that ``k` < `N``. This is achieved by requiring that the addition ``N` + (`k` - `N`)` overflows `mod 2^256` ([ec:c:k_lt_N]). Additionally, [ec:c:k_gt_0] ensures that ``k` > 0`, preventing a case where ``k` times `G` = `.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C18` |  | `MEMW[[addr_k[0], addr_k[1], 0, 0, 0, 0, 0, 0]; 1, (2 * 12)::DWordWL, [addr_k[0], addr_k[1], 0, 0, 0, 0, 0, 0], timestamp, 1, 0, 0]` | μ |
| `ECSM-C19.i` | i ∈ [0, 3] | `MEMW[[k[8 * i + 0], k[8 * i + 1], k[8 * i + 2], k[8 * i + 3], k[8 * i + 4], k[8 * i + 5], k[8 * i + 6], k[8 * i + 7]]; 0, addr_k + (8 * i)::DWordWL, [k[8 * i + 0], k[8 * i + 1], k[8 * i + 2], k[8 * i + 3], k[8 * i + 4], k[8 * i + 5], k[8 * i + 6], k[8 * i + 7]], timestamp, 0, 0, 1]` | μ |
| `ECSM-C20.i` | i ∈ [0, 15] | `IS_HALF[k_sub_N[i]]` | μ |
| `ECSM-C21.i` | i ∈ [0, 6] | μ ⇒ `IS_BIT<c2[i]>` |  |
| `ECSM-C22` |  | `μ` => `c2`_7 = 1 |  |
| | | _polynomial:_ `μ * (1 - c2[7]) = 0` | |
| `ECSM-C23` |  | `ZERO[k[0] + k[1] + k[2] + k[3] + k[4] + k[5] + k[6] + k[7] + k[8] + k[9] + k[10] + k[11] + k[12] + k[13] + k[14] + k[15] + k[16] + k[17] + k[18] + k[19] + k[20] + k[21] + k[22] + k[23] + k[24] + k[25] + k[26] + k[27] + k[28] + k[29] + k[30] + k[31]]` | μ |

#### Subroutine

With point `G` and scalar `k` fully constructed, we delegate bit-by-bit serving of the scalar `k` to the `EC_SCALAR` chip. Here, we capture the index of the most significant 1-bit of `k` in `len_k`. Note: if the prover decides to capture a lesser significant bit here, the LogUp will not balance, as the skipped bits will never taken off the bus. Next, we interact with the `ECDAS` chip, providing `G` both as the accumulator, and increment ([ec:c:start_double_add]); we specifically instruct the chip to start with a _double_-operation. After completing its double-and-add sequence, the result is captured in `R` ([ec:c:receive_double_add]).

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `ECSM-C24` | `SERVE_K[timestamp, addr_k::DWordWL, 31]` | μ |
| `ECSM-C25` | `BIT[timestamp, len_k]` | -μ |
| `ECSM-C26` | `ECDAS[timestamp, xG, yG, xG, yG, len_k - 1, 0]` | μ |
| `ECSM-C27` | `ECDAS[timestamp, xR, yR, xG, yG, -1, 0]` | -μ |

#### Range check `xR`

Before storing `x_R`, it is verified that `x_R in [0, p)`. To this end, witness ``xR_sub_p` := `xR` - p mod 2^256` is added to `p`; if the addition sums to `xR` and overflows `mod 2^256`, it must hold that ``xR` < p`. The addition is constrained by requiring that `c3` are bits ([ec:c:range_c3]); an overflow occurs if and only if ``c3[7]` = 1` ([ec:c:xR_addition_overflows]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C28.i` | i ∈ [0, 15] | `IS_HALF[xR_sub_p[i]]` | μ |
| `ECSM-C29.i` | i ∈ [0, 6] | μ ⇒ `IS_BIT<c3[i]>` |  |
| `ECSM-C30` |  | `μ` => `c3`_7 = 1 |  |
| | | _polynomial:_ `μ * (1 - c3[7]) = 0` | |

#### Write `xR`

We read `addr_xR` from register `x10` ([ec:c:load_addrR]), and subsequently write `xR` to this address ([ec:c:write_xR]). Note that the `timestamp` on both memory accesses is offset to allow `addr_xR` to equal `addr_xG` and thus for `x_R` to overwrite `x_G` in memory. Similar to `addr_xG` and `addr_k`, it is assumed that the addition of the small offsets will not overflow the lower limb of `addr_xR` ([ec:a:addr_xR_alignment]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECSM-C31` |  | `MEMW[[addr_xR[0], addr_xR[1], 0, 0, 0, 0, 0, 0]; 1, (2 * 10)::DWordWL, [addr_xR[0], addr_xR[1], 0, 0, 0, 0, 0, 0], timestamp + 1::DWordWL, 1, 0, 0]` | μ |
| `ECSM-C32.i` | i ∈ [0, 3] | `MEMW[0, addr_xR + (8 * i)::DWordWL, [xR[8 * i + 0], xR[8 * i + 1], xR[8 * i + 2], xR[8 * i + 3], xR[8 * i + 4], xR[8 * i + 5], xR[8 * i + 6], xR[8 * i + 7]], timestamp + 2::DWordWL, 0, 0, 1]` | μ |

### Padding

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `addr_xG` | `0` |
| `addr_k` | `0` |
| `addr_xR` | `0` |
| `xR` | `0` |
| `yR` | `0` |
| `k` | `0` |
| `len_k` | `0` |
| `xG` | `0` |
| `yG` | `0` |
| `x2` | `0` |
| `q0` | `0` |
| `c0` | `0` |
| `q1` | `p` |
| `c1` | `0` |
| `k_sub_N` | `0` |
| `xR_sub_p` | `0` |
| `μ` | `0` |

## ECDAS chip <ecdas>

The  chip (_Elliptic Curve Double-and-Add Sequence_) is responsible for accelerating the addition of two curve points, or the doubling of a single curve point. More specifically, given curve points `A` (accumulator) and `G` (generator), and selector bit `op`, it performs the mapping $ (A, G) mapsto cases( (A + A, &G) &text("if") `op` = 0, (A + G, &G) &text("if") `op` = 1 $

Recall that the addition of two curve points `A, B` is treated differently based on three cases:

enum.item[`x_A eq.not x_B`], enum.item[`x_A eq x_B` and `y_A eq.not -y_B`, or], enum.item[`x_A eq x_B` and `y_A eq -y_B`] where _double_ may encounter the last two cases, while _add_ may encounter all three. Cases 2 and 3 may, for specific inputs, evaluate to ``: a point that has no native short-Weierstrass representation. Therefore, the  and  chips were designed to avoid this case. To see how, note that  + is the sole chip that can "activate" the  chip by issuing an `ECDAS` lookup, + enforces that `G` and the initial `A` do not equal ``, and + ensures `k in [1, N)`, where `N` denotes the order of the curve. This combined yields that neither doubling `A` or adding `A + G` can produce ``:

*Double.* For `2A` to equal ``, the curve must have _even_ order. Since the order of the `secp256k1` curve is _odd_, such a point does not exist.

*Add.* If `A + G = `, then `A = -G =  - G = r N G - G` for some `r >= 0`. Because  initializes `A = G eq.not `, it must hold that `r >= 1`. Furthermore, the restriction that `k <= N-1` ensures `r <= 1`. Hence, `A = (N-1)G`. Since `N-1` is the maximal value of `k`, the previous round producing `A = (N-1)G` was the last round of this scalar multiplication. This means that now `round` is negative, which will fail constraint [ecdas:c:range_round].

### Columns

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | Timestamp at which the operation is executed. |
| `xG` | `U256BL` | $x_`G`$ |
| `yG` | `U256BL` | $y_`G`$ |
| `xA` | `U256BL` | $x_`A`$ |
| `yA` | `U256BL` | $y_`A`$ |
| `round` | `Byte` | scaling round |
| `op` | `Bit` | whether to double (0) or add (1) |

### Output

| Name | Type | Description |
|------|------|-------------|
| `xR` | `U256BL` | $x_`R`$ |
| `yR` | `U256BL` | $y_`R`$ |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `λ` | `U256BL` | `λ` |
| `q0` | `Byte[33]` | quotient used to constrain `λ` |
| `c0` | `BaseField[64]` | carries used to constrain `λ` |
| `q1` | `Byte[33]` | quotient used to constrain `xR` |
| `c1` | `BaseField[64]` | carries used to constrain `xR` |
| `q2` | `Byte[33]` | quotient used to constrain `yR` |
| `c2` | `BaseField[64]` | carries used to constrain `yR` |
| `next_op` | `Bit` | `op`-flag for the next iteration |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `XG` | `U512BL` | zero-extension of `xG` |
| `YG` | `U512BL` | zero-extension of `yG` |
| `XA` | `U512BL` | zero-extension of `xA` |
| `YA` | `U512BL` | zero-extension of `yA` |
| `XR` | `U512BL` | zero-extension of `xR` |
| `YR` | `U512BL` | zero-extension of `yR` |
| `Λ` | `U512BL` | zero-extension of `λ` |
| `Q0` | `U512BL` | zero-extension of `q0` |
| `Q1` | `U512BL` | zero-extension of `q1` |
| `Q2` | `U512BL` | zero-extension of `q2` |
| `P` | `U512BL` | zero-extension of `p` |
| `R` | `U512BL` | zero-extension of `r` |

**Definition of `XG`:**
```
XG (when iter=['i', 0, 31]) := xG[i]
XG (when iter=['i', 32, 63]) := 0
```

**Definition of `YG`:**
```
YG (when iter=['i', 0, 31]) := yG[i]
YG (when iter=['i', 32, 63]) := 0
```

**Definition of `XA`:**
```
XA (when iter=['i', 0, 31]) := xA[i]
XA (when iter=['i', 32, 63]) := 0
```

**Definition of `YA`:**
```
YA (when iter=['i', 0, 31]) := yA[i]
YA (when iter=['i', 32, 63]) := 0
```

**Definition of `XR`:**
```
XR (when iter=['i', 0, 31]) := xR[i]
XR (when iter=['i', 32, 63]) := 0
```

**Definition of `YR`:**
```
YR (when iter=['i', 0, 31]) := yR[i]
YR (when iter=['i', 32, 63]) := 0
```

**Definition of `Λ`:**
```
Λ (when iter=['i', 0, 31]) := λ[i]
Λ (when iter=['i', 32, 63]) := 0
```

**Definition of `Q0`:**
```
Q0 (when iter=['i', 0, 32]) := q0[i]
Q0 (when iter=['i', 33, 63]) := 0
```

**Definition of `Q1`:**
```
Q1 (when iter=['i', 0, 32]) := q1[i]
Q1 (when iter=['i', 33, 63]) := 0
```

**Definition of `Q2`:**
```
Q2 (when iter=['i', 0, 32]) := q2[i]
Q2 (when iter=['i', 33, 63]) := 0
```

**Definition of `P`:**
```
P (when iter=['i', 0, 31]) := p[i]
P (when iter=['i', 32, 63]) := 0
```

**Definition of `R`:**
```
R (when iter=['i', 0, 32]) := r[i]
R (when iter=['i', 33, 63]) := 0
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

### Constraints

First, the chips receives the input for this double/add step:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `ECDAS-C1` | `ECDAS[timestamp, xA, yA, xG, yG, round, op]` | -μ |
| `ECDAS-C2` | `IS_BIT<μ>` |  |
| `ECDAS-C3` | μ ⇒ `IS_BYTE<round>` |  |

#### Operation switching

The `op`-flag determines whether `R := 2A` (0) or `R:= A+G` (1). This chip introduces a set of constraints that properly constrains `R` depending on this flag. To illustrate how this is achieved, we split addition up in three relations: $ lambda &equiv (y_G - y_A)/(x_G - x_A) &&mod p,\ x_R &equiv lambda^2 - x_A - x_G &&mod p,\ y_R &equiv lambda (x_A - x_R) - y_A &&mod p.\ $ Introducing the non-negative witnesses `q'_0, q'_1` and `q_2`, we can convert these relations into $ lambda (x_G - x_A) - y_G + y_A + (`r` - q'_0) p &= 0,\ lambda^2 - x_A - x_G - x_R + (`r` - q'_1) p &= 0,\ lambda (x_A - x_R) - y_A - y_R + (`r` - q_2) p &= 0,\ $ for some `r in NN` to be fixed later.

Special attention should be paid to the first relation: if `x_A = x_G`, `lambda` can be chosen freely. By design, this situation cannot occur.

Observe that this would require either `A = G` or `A = -G`. With the latter situation previously ruled out, only the first remains. For `A = (r N + 1) G` for some `r in NN` and `N` the order of the curve, all cases with `r>0` can be ruled out since  verifies that the scalar `k < N`. The remaining case `A=G` is the intial state pushed onto the LogUp by  ([ec:c:start_double_add]), with `op`-flag set to `0` (_double_), not `add`. Hence, this situation cannot occur. ]

We rewrite the relations to find $ q'_0 &= `r` + p^(-1) dot (lambda (x_G - x_A) - y_G + y_A),\ q'_1 &= `r` + p^(-1) dot (lambda^2 - x_A - x_G - x_R),\ q_2  &= `r` + p^(-1) dot (lambda (x_A - x_R) - y_A - y_R)\ $ from which we can conclude that `q'_0, q_2 in (`r`-p, `r`+p)` and `q'_1 in (`r`, `r` + p)`. When doubling, only the formulae for `lambda` and `x_R` are different: $ lambda &equiv (3x_A^2)/(2y_A) &&mod p,\ x_R &equiv lambda^2 - 2x_A &&mod p.\ $ Introducing non-negative witnesses `q''_0` and `q''_1`, we convert these into $ 2lambda y_A - 3x_A^2 + (`r` - q''_0) p &= 0,\ lambda^2 - 2x_A - x_G - x_R + (`r` - q''_1) p &= 0.\ $

Special attention should be paid to the first relation: if `y_A = 0`, `lambda` can again be chosen freely. As previously established, `y_A != 0` for all points on the `secp256k1` curve. Hence, this situation will not occur. ] Reordering yields $ q''_0 &= `r` + p^(-1) dot (2lambda y_A - 3x_A^2 ),\ q''_1 &= `r` + p^(-1) dot (lambda^2 - 2x_A - x_G - x_R ).\ $ where `q''_0 in (`r`-3p, `r` + 2p)`, and `q''_1 = (`r`, `r` + p)`. We can now leverage the `op`-flag to merge the relations for `lambda` and `x_R` into $

lambda^2 - x_A - x_G - x_R + (1-`op`) (x_G - x_A) + (`r` - q_1) p &= 0\ $ which yields $ q_0 &= `r` + p^(-1) dot (`op` dot ((x_G - x_A)lambda - y_G + y_A) + (1-`op`) (2lambda y_A - 3x_A^2)),\ q_1 &= `r` + p^(-1) dot ((lambda^2 - x_A - x_G - x_R + (1-`op`) (x_G - x_A)).\ $ with `q_0 in (r-3p, r+2p)` and `q_1 in (r, r+p)`. By setting `r := 3p`, we ensure `q_0 in (0, 5p), q_1 in (3p, 4p)` and `q_2 in (2p, 4p)` are non-negative for all inputs.

#### Constraining $lambda$

We start by establishing the relation $

$

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECDAS-C4.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<λ[i]>` |  |
| `ECDAS-C5.i` | i ∈ [0, 32] | μ ⇒ `IS_BYTE<q0[i]>` |  |
| `ECDAS-C6` |  | 2^8 dot `c0`_0 = `op` dot (`Λ`_0 dot (`XG`_0 - `XA`_0) + `YA`_0 - `YG`_0) + (1 - `op`) dot (2 dot `Λ`_0 dot `YA`_0 - 3 dot `XA`_0 dot `XA`_0) + `R`_0 dot `P`_0 - `Q0`_0 dot `P`_0 = 0 |  |
| | | _polynomial:_ `2^8 * c0[0] + Q0[0] * P[0] - R[0] * P[0] - op * (Λ[0] * (XG[0] - XA[0]) + YA[0] - YG[0]) - (1 - op) * (2 * Λ[0] * YA[0] - 3 * XA[0] * XA[0]) = 0` | |
| `ECDAS-C7.i` | i ∈ [1, 63] | 2^8 dot `c0`_i = `c0`_(i-1) + `op` dot (`YA`_i - `YG`_i) + sum_(j=0)^i `op` dot `Λ`_j dot (`XG`_(i-j) - `XA`_(i-j)) + (1 - `op`) dot (2 dot `Λ`_j dot `YA`_(i-j) - 3 dot `XA`_j dot `XA`_(i-j)) + `R`_j dot `P`_(i-j) - `Q0`_j dot `P`_(i-j) |  |
| | | _polynomial:_ `2^8 * c0[i] - c0[i - 1] - op * (YA[i] - YG[i]) - Σ_j = 0^i (op * Λ[j] * (XG[i - j] - XA[i - j]) + (1 - op) * (2 * Λ[j] * YA[i - j] - 3 * XA[j] * XA[i - j]) + (R[j] * P[i - j] - Q0[j] * P[i - j])) = 0` | |
| `ECDAS-C8` |  | `c0`_63 = 0 |  |
| | | _polynomial:_ `c0[63] = 0` | |
| `ECDAS-C9.i` | i ∈ [0, 62] | `IS_HALF[c0[i] + 32636]` | μ |

#### Constraining $x_R$

Secondly, we establish $ lambda^2 - x_A - x_G - x_R - (1-`op`) (x_A - x_G) + (`r` - q_1) p &= 0 $

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECDAS-C10.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<xR[i]>` |  |
| `ECDAS-C11.i` | i ∈ [0, 32] | μ ⇒ `IS_BYTE<q1[i]>` |  |
| `ECDAS-C12` |  | 2^8 dot `c1`_0 = `Λ`_0 dot `Λ`_0 - `XA`_0 - `XG`_0 - `XR`_0 - (1-`op`) (`XA`_0 - `XG`_0) + `R`_0 dot `P`_0 - `Q1`_0 dot `P`_0 |  |
| | | _polynomial:_ `Λ[0] * Λ[0] + R[0] * P[0] - Q1[0] * P[0] - XA[0] - XG[0] - XR[0] - (1 - op) * (XA[0] - XG[0]) - 2^8 * c1[0] = 0` | |
| `ECDAS-C13.i` | i ∈ [1, 63] | 2^8 dot `c1`_i = `c1`_(i-1) - `XA`_i - `XG`_i - `XR`_i - (1- `op`) (`XA`_i - `XG`_i) + sum_(j=0)^i `Λ`_j dot `Λ`_(i-j) + `R`_j dot `P`_(i-j) - `Q1`_j dot `P`_(i-j) |  |
| | | _polynomial:_ `c1[i - 1] - 2^8 * c1[i] - XA[i] - XG[i] - XR[i] - (1 - op) * (XA[i] - XG[i]) - Σ_j = 0^i (Q1[j] * P[i - j] - R[j] * P[i - j] - Λ[j] * Λ[i - j]) = 0` | |
| `ECDAS-C14` |  | `c1`_63 = 0 |  |
| | | _polynomial:_ `c1[63] = 0` | |
| `ECDAS-C15.i` | i ∈ [0, 62] | `IS_HALF[c1[i] + 8161]` | μ |

#### Constraining $y_R$

Third, $ lambda (x_A - x_R) - y_A - y_R + (`r` - q_2) p &= 0 $ is constrained:

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `ECDAS-C16.i` | i ∈ [0, 31] | μ ⇒ `IS_BYTE<yR[i]>` |  |
| `ECDAS-C17.i` | i ∈ [0, 32] | μ ⇒ `IS_BYTE<q2[i]>` |  |
| `ECDAS-C18` |  | 2^8 dot `c2`_0 = `Λ`_0 dot (`XA`_0 - `XR`_0) - `YA`_0 - `YR`_0 + `R`_0 dot `P`_0 - `Q2`_0 dot `P`_0 |  |
| | | _polynomial:_ `Λ[0] * (XA[0] - XR[0]) + R[0] * P[0] - Q2[0] * P[0] - YA[0] - YR[0] - 2^8 * c2[0] = 0` | |
| `ECDAS-C19.i` | i ∈ [1, 63] | 2^8 dot `c2`_i = `c2`_(i-1) - `YA`_i - `YR`_i + sum_(j=0)^i `Λ`_j dot (`XA`_(i-j) - `XR`_(i-j)) + `R`_j dot `P`_(i-j) - `Q2`_j dot `P`_(i-j) |  |
| | | _polynomial:_ `c2[i - 1] - 2^8 * c2[i] - YA[i] - YR[i] - Σ_j = 0^i (Q2[j] * P[i - j] - R[j] * P[i - j] - Λ[j] * (XA[i - j] - XR[i - j])) = 0` | |
| `ECDAS-C20` |  | `c2`_63 = 0 |  |
| | | _polynomial:_ `c2[63] = 0` | |
| `ECDAS-C21.i` | i ∈ [0, 62] | `IS_HALF[c2[i] + 16320]` | μ |

Lastly, the updated accumulator is sent out for the next step to be processed ([ecdas:c:send]). To determine whether the next step should be an addition or doubling, the `next_op` bit is provided as witness by the prover. Setting this bit to 1 can only be done in active rows ([ecdas:c:next_op_implies_mu]), when the current ``op` = 0` (double), and does require the scalar bit in this position to be set ([ecdas:c:receive_next_op]).

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `ECDAS-C22` | `IS_BIT<next_op>` |  |
| `ECDAS-C23` | `BIT[timestamp, round]` | -next_op |
| `ECDAS-C24` | `op` = 1 => `next_op` = 0 |  |
| | _polynomial:_ `op * next_op = 0` | |
| `ECDAS-C25` | `next_op` = 1 => `μ` = 1 |  |
| | _polynomial:_ `next_op * (1 - μ) = 0` | |
| `ECDAS-C26` | `ECDAS[timestamp, xR, yR, xG, yG, round - 1 - next_op, next_op]` | μ |

### Padding

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `xG` | `0` |
| `yG` | `0` |
| `xA` | `0` |
| `yA` | `0` |
| `round` | `0` |
| `op` | `0` |
| `xR` | `0` |
| `yR` | `0` |
| `λ` | `0` |
| `q0` | `r` |
| `c0` | `0` |
| `q1` | `r` |
| `c1` | `0` |
| `q2` | `r` |
| `c2` | `0` |
| `next_op` | `0` |
| `μ` | `0` |

## EC-Scalar

### Columns

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which to serve the constant |
| `ptr` | `DWordWL` | pointer to the first byte of the scalar |
| `offset` | `Byte` | index of limb |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `limb_bits` | `Bit[8]` | bit-decomposition of the limb being read |
| `last_limb` | `Bit` | whether this is the last limb to read |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `limb` | `Byte` | limb being read |

**Definition of `limb`:**
```
limb := Σ_i = 0^7 2^i * limb_bits[i]
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

### Assumptions

This chip makes an assumption:

| Tag | Range | Description |
|-----|-------|-------------|
| `EC_SCALAR-A1` |  | $#`ptr` + #`offset`$ does not overflow the bottom limb |

### Constraints

The chip starts by extracting the input information from the bus when its multiplicity is set.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `EC_SCALAR-C1` | `SERVE_K[timestamp, ptr, offset]` | -μ |
| `EC_SCALAR-C2` | `IS_BIT<μ>` |  |

Next, it reads `limb` from address ``ptr` + `offset``. Note that the read-timestamp is offset by `1` to prevent a collision with read of `k` performed by . Since `limb` is reconstructed from `limb_bits`, it is ensured those are in fact bits.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `EC_SCALAR-C3` |  | `MEMW[[limb, 0, 0, 0, 0, 0, 0, 0]; 0, ptr + [offset::Word, 0], [limb, 0, 0, 0, 0, 0, 0, 0], timestamp + 1::DWordWL, 0, 0, 0]` | μ |
| `EC_SCALAR-C4.i` | i ∈ [0, 7] | `IS_BIT<limb_bits[i]>` |  |

For each `limb_bit` that is set, an `BIT`-interaction is sent on the bus, to inform the double-and-add sequence on the  chip. To prevent interactions from occurring in padding rows, an active limb bit requires a non-zero multiplicity.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `EC_SCALAR-C5.i` | i ∈ [0, 7] | `BIT[timestamp, 8 * offset + i]` | limb_bits[i] |
| `EC_SCALAR-C6.i` | i ∈ [0, 7] | `limb_bits`_i = 1 => `μ` = 1 |  |
| | | _polynomial:_ `limb_bits[i] * (1 - μ) = 0` | |

Unless this was the `last_limb` (i.e., ``offset` = 0`), we recurse on serving the previous limb.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `EC_SCALAR-C7` | `IS_BIT<last_limb>` |  |
| `EC_SCALAR-C8` | `last_limb` = 1 => `μ` = 1 |  |
| | _polynomial:_ `last_limb * (1 - μ) = 0` | |
| `EC_SCALAR-C9` | `last_limb` = 1 => `offset` = 0 |  |
| | _polynomial:_ `last_limb * offset = 0` | |
| `EC_SCALAR-C10` | `SERVE_K[timestamp, ptr, offset - 1]` | μ - last_limb |

`last_limb` is a witness provided by the prover, which, technically, could be kept at `0` when ``offset` = 0`. However, that would require an additional `2^64` table entries to balance out the LogUp bus. Since this is assumed infeasible, the prover is constrained to set `last_limb` appropriately.

### Padding

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `ptr` | `0` |
| `offset` | `0` |
| `limb_bits` | `[0, 0, 0, 0, 0, 0, 0, 0]` |
| `last_limb` | `0` |
| `μ` | `0` |

## Notes / optimizations

- To utilize the  /  chips for different curves, consider introducing a lookup table for the curve-constants `a`, `b`, `p`, `r` and `N`, and look them up when a scalar multiplication selects them. The selection procedure could be done through the `ECALL` number; the  chip would accept multiple numbers, setting an internal "curve-selector" field accordingly. - Transitioning from `U256BL`s to `U256HL`s would roughly halve the number of columns in both the  and  chips. This would likely require increasing the sizes of the carries from 16 to 24 bits. Since the carries need to be range checked, one would have to investigate whether - it would be possible to perform a 24-bit range-check lookup, - one could set up a 24-bit range-check table. This could be as narrow as two columns. - have some hybrid version, where there is a native lookup table for x-bits, and a dynamic table for outliers (high carries are not encountered frequently).

## Discussing the carries <ecsm-limb_carry>

To constrain `x2` and `y_G` in , and `lambda`, `x_R` and `y_R` in , we use (variations of) the same technique: - multiplications are performed limb-by-limb, - a set of carry-limbs is used to exchange the underflow/overflow from one limb to another, and - the carry limbs are range constrained to ensure only one output value is possible.

We now explore this carry-technique and provide some proofs.

### Lemma 1

Let `V in NN` and `A,M in [0, V)`. For `i >= 1`, we define $ r_i &:= A (V-1) + M sum_(j=1)^i (V-1)^2 = i M(V-1)^2 + A(V-1),\ v_i &:= r_i + c_(i-1) mod V,\ c_i &:= V^(-1) (r_i + c_(i-1) - v_i),\ c_0 &:= 0 $ It holds that $ c_i = i M(V-1) + A - M - delta_(M<A) $ where kronecker delta `delta_x` equals `1` if `x` is true, and `0` otherwise.

For `i = 1`, we find that $ r_1 &= M(V - 1)^2 + A(V-1) \ &= M(V^2-2V) + (A-delta) V + delta V + M - A \ v_1 &equiv delta V + M - A mod V\ c_1 &= V^(-1) (M(V^2 - 2V) + (A-delta) V)\ &= M(V-2) + A-delta $ Suppose the statement to hold for arbitrary `i >= 1`. We find that $ d_(i+1) &= (i+1)M(V-1)^2 + A(V-1)\ v_(i+1) &equiv (i+1)M(V^2 - 2V) + (i+1)M + A V - A + i M(V-2) + (i-1)M + A-delta &&mod V\ &equiv (i+1)M(V^2 - 2V) + (A + i M - delta)V + delta (V-1) &&mod V\ &equiv delta (V-1) &&mod V\ c_(i+1) &= V^(-1) dot ((i+1)M(V^2 - 2V) + V(A + i M - delta))\ &= (i+1)M(V - 2) + A + i M - delta $ `qed`

### Corollary 1

Let `L` be a number of limbs, `b` be the number of bits per limb, `M in [0, L)` the number of multiplications in the formula, and `A in [0, L)` the number of additions. The maximum value of the carry is $ L M (2^b-1) + A - M - delta_(M < A) $

Applying the corollary to the relations $ x_G^2 - `x2` - q_0 dot p &= 0,\ y_G^2 - x_G dot `x2` - b + (p - q_1)p &= 0,\

lambda^2 - x_A - x_G - x_R + (1-`op`) (x_G - x_A) + (`r` - q_1) p &= 0,\ lambda (x_A - x_R) - y_A - y_R + (`r` - q_2) p &= 0.\ $ We find that the carries for sixteen 8-bit limbs are in the range $ (1): [-8160, 8159]\ (2): [-16319, 16318]\ (3): [-32636, 24477]\ (4): [-8161, 16318]\ (5): [-16320, 16318]\ $