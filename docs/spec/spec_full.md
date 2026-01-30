# Lambda VM Specification

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

> **Note:** properly link/refer to the logup spec

So long as we can properly constrain temporal integrity (that is, no memory operation can consume future tokens), this "balancing" act of tokens can be integrated (with sufficient domain separation) into the existing LogUp argument: consuming a token corresponds to a "receive" and emitting a new token is a "send".

## Temporal integrity

> **Note:** Properly link/refer to the LT chip

To ensure temporal integrity, every memory operation needs to be constrained for the newly emitted token to have a strictly greater timestamp than the consumed token. This raises the question of how to represent timestamps and cleanly perform this check, as over a finite field the “less than” relation is ill-defined (though it is common and natural to consider it as the less than relation over the natural lift of the field into the integers). We choose to represent timestamps as machine words, using the existing `LT` chip ([lt]) functionality for comparisons.

- Clean definition of “less-than”, using the already existing `LT` functionality in the ALU - Harder to perform increments, needing extra constraints beyond field arithmetic - But this can be alleviated by providing a precomputed column that has a fixed increment per CPU row ][ - Comparison is more annoying, but can work by: - Decomposition into a machine word and chip interaction with the LT chip - Bit decomposition and comparison constraints - Range-checking the difference to be sufficiently small w.r.t. the field characteristic. - Increments and basic arithmetic operations are cheap ] ]

> **Note:** reference to CPU chip/timestamp column and MEMW chip

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

We identify a few alternatives that would achieve the desired initialization/finalization functionalities, and consider their respective trade-offs.

_"Free-zero" initialization_

Zero-initialization could be achieved by allowing the `MEMW` chip to output a zero without consuming a token from the lookup argument. This would in turn be made secure by finalization consuming at most one token per address: if an address is initialized more than once, the proof cannot be finalized. - This requires fewer pages (and hence tables) for zero-initialization. - But it comes at a cost of added complexity in the `MEMW `chip, and likely some extra columns to handle this. Keeping track of initialized addresses, and potentially having to initialize only some of the bytes in a word-read may make bookkeeping challenging. - This is an alternative form of sparse initialization (see below), so it is incompatible with paged finalization. Paged finalization can be made into a compatible sparse form by adding a bit-checked multiplicity column.

_Sparse initialization/finalization_

One or more STARK tables (depending on the amount of memory used) consisting of `(address, value)` columns are introduced, where for zero-initialization, `value` can be constant zero. Transition constraints ensure that `address` is strictly increasing, enforcing the "at most once" property; `value` is range-checked to consist of bytes. Similar to paged finalization, an additional `timestamp` column is added, containing the final timestamp each address was accessed. This table is then further used to contribute to the LogUp sum as with any other interactions. - The transition constraints can be chosen to only apply on finalization, as at-most-once finalization is enough to ensure consistency. - Sparse initialization is incompatible with paged finalization, see also the remark under free-zero initialization above. - This would require transition constraints, which currently are not needed elsewhere in the VM design - Additionally, for memory use exceeding the capacity of a single initialization/finalization table, some form of transition constraint between tables is needed - Alternatively, transition constraints could potentially be avoided by more integration into the LogUp system, but this could turn out more costly in practice - This is compatible with the above "free zero" initialization - Since a prover-committed address column is needed (rather than a precomputed one), the number of required columns increases. - As an optimization, the address column could potentially be used simultaneously for initialization and finalization - Sparse initialization/finalization reduces the cost for sparse memory access patterns, where only a few addresses would be accessed per page. Most programs and compilers should however favor a memory locality that makes paged initialization/finalization comparable. ]

### Register initialization/finalization

> **Note:** Properly link/reference ECALL/HALT chip

The initial and final state of registers can be entirely known by the verifier, since the relevant initialization values are either zero, or embedded in the ELF, and the final values can be set to a known value by the HALT ecall. As additionally, the number of registers is small, the verifier can directly add the required balancing terms to the LogUp sum.

## Notes and considerations

- Register reads and writes may interact within a single cycle, so a correct and fixed ordering needs to be ensured - Correctness of initialization and completeness of finalization need to be ensured

## Future topics of interest

- Optimize memory systems after determining factual bottlenecks (e.g. taking inspiration from Twist and Shout, or other recent research)

---

# Variables

While this VM operates on 64-bit words, the proving system's base field has fewer than `2^64` elements available and thus cannot represent all words natively. To this end, we introduce the concept of "variables" as an abstraction layer on top of the VM's field elements. The following table lists all variable types used in this VM.

columns: (auto, 1fr, auto), inset: 7pt, align: (top+left, top+left, top+center, ), table.header([*Name*], [*Description*], [*\*]), ..for type in config.variables.types { ([], [], [.subtypes.len()]) },

---

# IS_BIT Template

box( inset: (left: 4pt, right: 4pt), outset: (top: 4pt, bottom: 4pt), radius: 2pt, fill: luma(230), raw(code)) }

Barring exceptional cases, this template is used to assert that a variable of type `Bit` assumes a valid value under some condition.

## Interface

The  constraint template has the following interface:

where `cond` is any value described by an expression _of degree at most `1`_. Note that  can be used to denote the _unconditional_ application of the  template to `X`.

## Variables

The  template operates on two variables: `cond` and `X`:

## Constraints

It takes only one constraint to enforce that `X` must be either `0` or `1` whenever ``cond` eq.not 0`:

*Note*: - In case of _unconditional_ template application, `cond` can be dropped from the constraint, simplifying it to ``X` (1- `X`) = 0`. - As described earlier, the `cond` variable must be describable by a degree-1 (i.e., linear) expression. This is to make sure that [isbit:c:isbit]'s expression has degree at most 3.

## Proof of correctness

If `cond` is `0`, [isbit:c:isbit] is trivially satisfied: `X` can assume any value and the polynomial constraint will evaluate to `0` regardless. When ``cond` eq.not 0`, it follows that the statement can only be proven when ``X` (1-`X`) equiv 0 mod p`, with `p` the modulus of the field. Because `BaseField` is a prime field, this equality is only satisfied if either ``X` equiv 0 mod p` or `1-`X` equiv 0 mod p`. Hence, it is proven that when ``cond` eq.not 0`, [isbit:c:isbit] is only satisfied if ``X` in {0, 1}`. 

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `X` | `BaseField` | Value for which to assert that it lies in the range ${0, 1}$. |

### Condition

| Name | Type | Description |
|------|------|-------------|
| `cond` | `BaseField` | Whether the constraint should be applied ($eq.not 0$) or not ($0$). |

### all

| Tag | Description |
|-----|-------------|
| `IS_BIT-C1` | `cond` => `X` (1-`X`) = 0 |
| | _polynomial:_ `cond * X * (1 - X) = 0` |

---

# ADD/SUB Template

box( inset: (left: 4pt, right: 4pt), outset: (top: 4pt, bottom: 4pt), radius: 2pt, fill: luma(230), raw(code)) }

## Notation

The  constraint template has the following interface:

where `cond` is any value described by an expression _of degree at most `1`_.

### 

For ease of notation, we moreover introduce the  constraint template. Its interface

maps onto the  template as

It constrains that ``diff` = `lhs` - `rhs` mod 2^64` when the expression `cond` is non-zero. As with ,  can be used to denote the _unconditional_ application of the template.

## Variables

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `ADD-A1.i` | i ∈ [0, 1] | `IS_WORD[lhs[i]]` |
| `ADD-A2.i` | i ∈ [0, 1] | `IS_WORD[rhs[i]]` |
| `ADD-A3.i` | i ∈ [0, 1] | `IS_WORD[sum[i]]` |

## Constraints

This template introduces the following constraints

## Columns

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

### all

| Tag | Range | Description |
|-----|-------|-------------|
| `ADD-C1.i` | i ∈ [0, 1] | cond ⇒ `IS_BIT<carry[i]>` |

---

# DECODE Table

All `RV64IMC` instruction are to be decoded to a format that can be interpreted by the VM. This section outlines the decoding table being used in the VM. For reasons of efficiency, data in this table is significantly compressed. Since reasoning about this compressed form is needlessly complex, the `decode (uncompressed)` section presents the same table in uncompressed form, and explains how to decode `RV64IM` assembly instructions to it. Instructions on how to compress the uncompressed table to form the compressed decode table, can be derived from the `packed_decode` variable provided below.

## Columns

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

The  table is comprised of  variables that are expressed using  columns:

## Padding

The  table must be padded to a length that is a power of two. Empty rows with the following content can be added to achieve this:

Note that this row sets the `EBREAK` flag. Given that `CPU` asserts that `EBREAK = 0` (see [cpu:c:ebreak_traps]), using this "padding-instruction" would immediately make the CPU table unprovable. Note moreover that the `pc` is set to `7`. This value is the _smallest odd number_ (i.e., not reachable during regular execution) that is more than _`4`_ (i.e., the max `pc`-increment) greater than _`1`_ (i.e., the `pc`-value used in the [additional instruction] referred to by `CPU`-padding lines).

## Decoding

For the purposes of explaining decoding, we decompress 's `packed_decode` variable into its constituent variables. Note that the below table is _not_ used in practice: it is solely used for the purposes of this explanation.

We will illustrate how each instruction should be expressed in this (uncompressed) decoding table. The columns of the accompanying table represent the following: - *`operation`*: the assembly operation being encoded. - *`op-flag`*: which of the "`ALU` selector flags" operation flags to set. Each operation sets exactly one. - *`w_instr`*, *`signed`*: whether to set the `word_instr` and `signed` flags, respectively. - *other*: the other flags that should be set or variables that should be given specific values.

For the purpose of brevity and readability, the table uses the following rules-of-thumb: + `rd`, `rs1`, `rs2`, and `imm` are mapped to the values provided by the instruction; when a value is not specified by an instruction it defaults to `0`. + `read_register1`, `read_register2` and `write_register` are set to `1` when respectively ``rs1` != 0`, ``rs2` != 0`, or  ``rd` != 0`. + Any flag that is not listed is set to `0`, with the exception of the `c_type` flag. *The `c_type` flag is set independently of the below table*, as explained next.

Further clarification is provided in the notes following the table.

### C-type instructions

The `RV64C` extension for compressed instructions specifies that \~50% of all instructions can be represented using a 16-bit instruction (rather than 32-bits), saving \~25% in code size. This execution of assembly code is _not_ agnostic to an instruction's compression state; after executing a compressed instruction, the `pc` should be incremented by `2` rather than `4`. To indicate an instruction is provided in compressed form, the `c_type` flag is introduced. *This flag should be set to `1` whenever the decoded instruction is provided in compressed form and `0` otherwise.*

/// Add a reference to one or more notes following this table. super("[" + refs.pos().map(r => ref(r)).join(",") + "]") }

show figure: set block(breakable: true)

figure(table( columns: (auto, auto, 40pt, 40pt, 1fr, 15pt), stroke: 0pt, inset: (right: .5em), align: (left, right, center, center, left, right), fill: (_, y) => if calc.odd(y) and y <= lines.len() { luma(245) } else { white }, table.header([*Operation*], [*op-flag*], [*`w_instr`*], [*`signed`*], [*other*], []), table.hline(stroke: 1.5pt), table.vline(x: 1, start: 1, end: lines.len() + 1, stroke: .5pt), ..lines.flatten(), table.hline(stroke: 1.5pt), table.footer([*Operation*], [*op-flag*], [*`w_instr`*], [*`signed`*], [*other*]), ), caption: [Decoding table] }

// OP-IMM ([`ADDI[W]   rd, rs1, imm`], [`ADD`], [`[W]`], [], [], []), ([`SLTI[U]   rd, rs1, imm`], [`SLT`], [], [.not`[U]`], [], []), ([`ANDI      rd, rs1, imm`], [`AND`], [], [], [], []), ([`ORI       rd, rs1, imm`], [`OR`],   [], [], [], []), ([`XORI      rd, rs1, imm`], [`XOR`], [], [], [], []), ([`SLLI[W]   rd, rs1, imm`], [`SHIFT`], [`[W]`], [], [], []), ([`SRLI[W]   rd, rs1, imm`], [`SHIFT`], [`[W]`], [], [`mp_selector`], []), ([`SRAI[W]   rd, rs1, imm`], [`SHIFT`], [`[W]`], [1], [`mp_selector`], []), // OP ([`ADD[W]    rd, rs1, rs2`], [`ADD`], [`[W]`], [], [], []), ([`SUB[W]    rd, rs1, rs2`], [`SUB`], [`[W]`], [], [], []), ([`SLT[U]    rd, rs1, rs2`], [`SLT`], [], [.not`[U]`], [], []), ([`AND       rd, rs1, rs2`], [`AND`], [], [], [], []), ([`OR        rd, rs1, rs2`], [`OR`], [], [], [], []), ([`XOR       rd, rs1, rs2`], [`XOR`], [], [], [], []), ([`SLL[W]    rd, rs1, rs2`], [`SHIFT`], [`[W]`], [], [], []), ([`SRL[W]    rd, rs1, rs2`], [`SHIFT`], [`[W]`], [], [`mp_selector`], []), ([`SRA[W]    rd, rs1, rs2`], [`SHIFT`], [`[W]`], [1], [`mp_selector`], []), // OP - M ([`MUL[W]    rd, rs1, rs2`], [`MUL`], [`[W]`], [1], [`mp_selector`], []), ([`MULH      rd, rs1, rs2`], [`MUL`], [], [1], [`mp_selector`, `muldiv_selector`], []), ([`MULHU     rd, rs1, rs2`], [`MUL`], [], [], [`muldiv_selector`], []), ([`MULHSU    rd, rs1, rs2`], [`MUL`], [], [1], [`muldiv_selector`], []), ([`DIV[U][W] rd, rs1, rs2`], [`DIVREM`], [`[W]`], [.not`[U]`], [], []), ([`REM[U][W] rd, rs1, rs2`], [`DIVREM`], [`[W]`], [.not`[U]`], [`muldiv_selector`], []), // LUI/AUIPC ([`LUI       rd, imm`], [`ADD`], [], [], [], []), ([`AUIPC     rd, imm`], [`ADD`], [], [], [`rs1 := x255`], []), ([`JAL       rd, imm`], [`JALR`], [], [], [`rs1 := x255`], []), // Branching ([`JALR      rd, rs1, imm`], [`JALR`], [], [], [], []), ([`BEQ      rs1, rs2, imm`], [`BEQ`], [], [], [], []), ([`BNE      rs1, rs2, imm`], [`BEQ`], [], [], [`mp_selector`], []), ([`BLT[U]   rs1, rs2, imm`], [`BLT`], [], [.not`[U]`], [], []), ([`BGE[U]   rs1, rs2, imm`], [`BLT`], [], [.not`[U]`], [`mp_selector`], []), // LOAD ([`LD        rd, rs1, imm`], [`LOAD`], [], [], [`mem_8B`], []), ([`LW[U]     rd, rs1, imm`], [`LOAD`], [], [.not`[U]`], [`mem_4B`], []), ([`LH[U]     rd, rs1, imm`], [`LOAD`], [], [.not`[U]`], [`mem_2B`], []), ([`LB[U]     rd, rs1, imm`], [`LOAD`], [], [.not`[U]`], [], []), // STORE ([`SD       rs1, rs2, imm`], [`STORE`], [], [], [`mem_8B`], []), ([`SW       rs1, rs2, imm`], [`STORE`], [], [], [`mem_4B`], []), ([`SH       rs1, rs2, imm`], [`STORE`], [], [], [`mem_2B`], []), ([`SB       rs1, rs2, imm`], [`STORE`], [], [], [], []), // ECALL/EBREAK ([`ECALL`], [`ECALL`], [], [], [``rs1` := `x17``], []), ([`EBREAK`], [`EBREAK`], [], [], [], []), // FENCE ([`FENCE`], [`ADD`], [], [], [], []),

// Construct a note that can be referenced through `lbl` show figure: (it) => align(left, []) [ ] }

#### Notes

We note the following about the above decoding table:

enum.item( referenceable_note( "note_word_instr", [`word_instr`: `[W]` indicates that ``word_instr` = 1` for the `W`-variant of the operation, and `0` for the non-`W`-variant.] ), enum.item( referenceable_note( "note_signed", [`signed`: .not`[U]` indicates that ``signed` = 1` for the *non-`U`*-variant of the operation, and `0` for the `U`-variant.] ), enum.item( referenceable_note( "note-lui", [`LUI`: this operation loads the 20-bit `imm` in the upper bits of `rd`. Observe that this can be represented using `ADDI rd, x0, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[12:32]` of `imm` and extending it to 64 bits.*] ), enum.item( referenceable_note( "note-auipc", [`AUIPC`: this operation adds the 20-bit immediate to the upper bits of `pc` and stores the result in `rd`. Given that the `pc` is stored in `x255`, this operation can be represented using `ADDI rd, x255, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[12:32]` of `imm` and extending it to 64 bits.*] ), enum.item( referenceable_note( "note-jal", [`JAL`: this operation stores ``pc` + 4` in `rd` and adds two times the sign-extended 20-bit immediate to the `pc`. Note that this can be represented using `JALR rd, x255, imm`. As such, *we expect the decoding to take care of writing the immediate in bit range `[1:21]` of `imm` and extending it to 64 bits; the least significant bit should always be 0.*] ), enum.item( referenceable_note( "note-ecall", [`ECALL`: "On RISC-V a system call has its own instruction: `ECALL`. [...] A7 [= register `x17`] contains the system call number." [[source]] ] ), enum.item( referenceable_note( "note-fence", [`FENCE`: currently, the VM interprets this operation as `ADDI x0 x0 0`; a no-op.]

## One more instruction <cpu-padding-decode-row>

In addition to decoding all instructions provided in the ELF and adding a corresponding entry to the  table, one must include an entry that has ``pc` = 1` and every other variable set to `0`. Note that this will never conflict with any entry in the ELF, since it has an odd `pc` value.

This entry is used to pad the `CPU` table. More details on this matter are provided in the `CPU` chip.

---

# CPU Chip

## Columns

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
| `rv1_sign_bit` | `Bit` | The sign bit of `rv1` if seen as a 32-bit word |
| `arg1` | `DWordBL` | The extended version of `rv1`, depending on `word_instr` |
| `arg2_sign_bit` | `Bit` | The sign bit of `arg2` if seen as a 32-bit word |
| `arg2` | `DWordBL` | A multiplexed version of `rv2` and `imm`, to be used as second argument to ALU calls |
| `res_sign_bit` | `Bit` | The sign bit of `res`, if seen as a 32-bit word |
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

The `CPU` chip is comprised of  variables that are expressed using  columns:

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `CPU-A1` |  | At most one ALU selector flag is 1 by the decoding, and every other flag is 0. |
| `CPU-A2` |  | When `STORE + LOAD + BEQ + BLT = 0`, either `rs2 = 0` or `imm = 0` should be enforced by the decoding. This is needed for `arg2`. |

## Constraints

First, we perform a decoding lookup for the current PC.

| Tag | Description |
|-----|-------------|
| `CPU-C1` | `DECODE[pc, imm, packed_decode]` |

> **Note:** All casts for interactions will have to be reviewed once other chip interfaces stabilise

### Range checks

> **Note:** Make sure we argue for every column here

> **Note:** is `rvd` still sufficiently constrained? (can also be done through the memory argument like `pc`?)

We constrain all columns to have the appropriate ranges. The flags and register indices looked up from the decoding need to be checked, as they are communicated through the interaction in a packed form. In contrast, we know ahead of time that decoding will ensure proper range checks for `pc` and `imm`. Similarly, since `next_pc` will propagate through the memory argument and be looked up in the instruction decoding on the next cycle, it is forced to be in the correct range. For the auxiliary columns, we need to check the limbs of `arg1`, `arg2`, and `res`. The ranges of the other auxiliary columns are enforced through later constraints.

| Tag | Range | Description |
|-----|-------|-------------|
| `CPU-CR2` |  | `IS_BIT<read_register1>` |
| `CPU-CR3` |  | `IS_BIT<read_register2>` |
| `CPU-CR4` |  | `IS_BIT<write_register>` |
| `CPU-CR5` |  | `IS_BIT<memory_2bytes>` |
| `CPU-CR6` |  | `IS_BIT<memory_4bytes>` |
| `CPU-CR7` |  | `IS_BIT<memory_8bytes>` |
| `CPU-CR8` |  | `IS_BIT<c_type_instruction>` |
| `CPU-CR9` |  | `IS_BIT<signed>` |
| `CPU-CR10` |  | `IS_BIT<mp_selector>` |
| `CPU-CR11` |  | `IS_BIT<muldiv_selector>` |
| `CPU-CR12` |  | `IS_BIT<word_instr>` |
| `CPU-CR13` |  | `IS_BIT<ADD>` |
| `CPU-CR14` |  | `IS_BIT<SUB>` |
| `CPU-CR15` |  | `IS_BIT<SLT>` |
| `CPU-CR16` |  | `IS_BIT<AND>` |
| `CPU-CR17` |  | `IS_BIT<OR>` |
| `CPU-CR18` |  | `IS_BIT<XOR>` |
| `CPU-CR19` |  | `IS_BIT<SHIFT>` |
| `CPU-CR20` |  | `IS_BIT<JALR>` |
| `CPU-CR21` |  | `IS_BIT<BEQ>` |
| `CPU-CR22` |  | `IS_BIT<BLT>` |
| `CPU-CR23` |  | `IS_BIT<LOAD>` |
| `CPU-CR24` |  | `IS_BIT<STORE>` |
| `CPU-CR25` |  | `IS_BIT<MUL>` |
| `CPU-CR26` |  | `IS_BIT<DIVREM>` |
| `CPU-CR27` |  | `IS_BIT<ECALL>` |
| `CPU-CR28` |  | `IS_BIT<EBREAK>` |
| `CPU-CR29` |  | `IS_BYTE[rs1]` |
| `CPU-CR30` |  | `IS_BYTE[rs2]` |
| `CPU-CR31` |  | `IS_BYTE[rd]` |
| `CPU-CR32.i` | i ∈ [0, 7] | `IS_BYTE[arg1[i]]` |
| `CPU-CR33.i` | i ∈ [0, 7] | `IS_BYTE[arg2[i]]` |
| `CPU-CR34.i` | i ∈ [0, 7] | `IS_BYTE[res[i]]` |

### ALU

The ALU functionality is then obtained through judicious dispatching to the corresponding chips.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CA35` |  | ADD + LOAD + STORE ⇒ `ADD<res::DWordWL; arg1::DWordWL, arg2::DWordWL>` |  |
| `CPU-CA36` |  | SUB + BEQ ⇒ `SUB<res::DWordWL; arg1::DWordWL, arg2::DWordWL>` |  |
| `CPU-CA37` |  | `LT[res[0]; arg1::DWordWL, arg2::DWordWL, signed]` | SLT + BLT |
| `CPU-CA38.i` | i ∈ [1, 7] | `SLT` + `BLT` => `res[i]` = 0 |  |
| | | _polynomial:_ `(SLT + BLT) * res[i] = 0` | |
| `CPU-CA39.i` | i ∈ [0, 7] | `AND_BYTE[res[i]; arg1[i], arg2[i]]` | AND |
| `CPU-CA40.i` | i ∈ [0, 7] | `OR_BYTE[res[i]; arg1[i], arg2[i]]` | OR |
| `CPU-CA41.i` | i ∈ [0, 7] | `XOR_BYTE[res[i]; arg1[i], arg2[i]]` | XOR |
| `CPU-CA42` |  | `SHIFT[res::DWordHL; arg1::DWordHL, arg2[0], mp_selector, signed, word_instr]` | SHIFT |
| `CPU-CA43` |  | JALR ⇒ `ADD<res::DWordWL; pc, (2 * c_type_instruction + 4 * (1 - c_type_instruction))::DWordWL>` |  |
| `CPU-CA44` |  | `MUL[res; arg1, signed, arg2, mp_selector, muldiv_selector]` | MUL |
| `CPU-CA45` |  | `DVRM[res; arg1, arg2, signed, muldiv_selector]` | DIVREM |

### Memory

The interactions with the memory, both for register loading and storing, as for `LOAD` and `STORE` instructions are handled. Note that since registers need no byte-addressing, we store them in the memory argument with `Word` limbs. The timestamps are ensured to be disjoint for disjoint memory locations. One consequence of that is that `next_pc` is written at `timestamp + 1` to ensure the access is disjoint with the `pc` read into `rv1` as part of the `AUIPC` instruction.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `CPU-CM46` |  | `MEMW[rv1; 1, 2 * rs1, rv1, timestamp + 0, 1, 0, 0]` | read_register1 |
| `CPU-CM47.i` | i ∈ [0, 2] | `!read_register1` => `rv1[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register1) * rv1[i] = 0` | |
| `CPU-CM48` |  | `MEMW[rv2; 1, 2 * rs2, rv2, timestamp + 1, 1, 0, 0]` | read_register2 |
| `CPU-CM49.i` | i ∈ [0, 2] | `!read_register2` => `rv2[i]` = 0 |  |
| | | _polynomial:_ `(1 - read_register2) * rv2[i] = 0` | |
| `CPU-CM50` |  | `MEMW[1, 2 * rd, rvd, timestamp + 2, 1, 0, 0]` | write_register |
| `CPU-CM51` |  | `LOAD[rvd; 0, res, timestamp + 0, memory_2bytes, memory_4bytes, memory_8bytes, signed]` | LOAD |
| `CPU-CM52` |  | `MEMW[0, res, rv2, timestamp + 1, memory_2bytes, memory_4bytes, memory_8bytes]` | STORE |
| `CPU-CM53` |  | `MEMW[pc; 1, 2 * 255, next_pc, timestamp + 1, 1, 0, 0]` | 1 - pad |

### System

The interactions with the wider system.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CS54` | `!EBREAK` |  |
| | _polynomial:_ `1 - EBREAK = 0` | |
| `CPU-CS55` | `ECALL[timestamp, rv1::DWordWL]` | ECALL |

### Input and output to the ALU

We constrain `arg1`, `arg2` and `rvd` to correspond to the wanted values, including the appropriate sign/zero extension, depending on `word_instr`.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CE56` | (`rv1_sign_bit` or `arg2_sign_bit` or `res_sign_bit`) => `word_instr` |  |
| | _polynomial:_ `(rv1_sign_bit + arg2_sign_bit + res_sign_bit) * (1 - word_instr) = 0` | |
| `CPU-CE57` | `MSB16[rv1_sign_bit; rv1[1]]` | word_instr |
| `CPU-CE58` | `arg1[:4]` = `rv1[:2]` |  |
| | _polynomial:_ `(arg1::DWordWL)[0] - (rv1::DWordWL)[0] = 0` | |
| `CPU-CE59` | `arg1[4:]` = `rv1[2]` dot (1 - `word_instr`) + (2^(32) - 1) dot `rv1_sign_bit` dot `signed` |  |
| | _polynomial:_ `(arg1::DWordWL)[1] - (1 - word_instr) * rv1[2] - signed * rv1_sign_bit * (2^32 - 1) = 0` | |
| `CPU-CE60` | `MSB16[arg2_sign_bit; rv2[1]]` | word_instr |
| `CPU-CE61` | `arg2[:4]` = (1 - `STORE` - `LOAD`) dot `rv2[:2]` + (1 - `BEQ` - `BLT`) dot `imm[0]` |  |
| | _polynomial:_ `(arg2::DWordWL)[0] - (1 - STORE - LOAD) * (rv2::DWordWL)[0] - (1 - BEQ - BLT) * imm[0] = 0` | |
| `CPU-CE62` | `arg2[4:]` = (1 - `STORE` - `LOAD`) dot ((1 - `word_instr`) dot `rv2[2]` + `signed` dot `arg2_sign_bit` dot (2^(32) - 1)) + (1 - `BEQ` - `BLT`) dot `imm[1]` |  |
| | _polynomial:_ `(arg2::DWordWL)[1] - (1 - STORE - LOAD) * (1 - word_instr) * rv2[2] - (1 - STORE - LOAD) * signed * arg2_sign_bit * (2^32 - 1) - (1 - BEQ - BLT) * imm[1] = 0` | |
| `CPU-CE63` | `MSB8[res_sign_bit; res[3]]` | word_instr |
| `CPU-CE64` | `!LOAD` => `rvd[0]` = `res[:4]` |  |
| | _polynomial:_ `(1 - LOAD) * (rvd[0] - (res::DWordWL)[0]) = 0` | |
| `CPU-CE65` | `!LOAD` => `rvd[1]` = (1 - `word_instr`) dot `res[4:]` + `res_sign_bit` dot (2^(32) - 1) |  |
| | _polynomial:_ `(1 - LOAD) * (rvd[1] - (1 - word_instr) * (res::DWordWL)[1] - res_sign_bit * (2^32 - 1)) = 0` | |

### Other constraints

> **Note:** proper ref to IsZero/IsEqual

For [cpu:c:is_equal], refer to the logic of IsZero or IsEqual, in combination with the subtraction of [cpu:c:sub].

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `CPU-CO66` | `ZERO[is_equal; res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7]]` | BEQ |
| `CPU-CO67` | `branch_cond` = `JALR` or (`BLT` and (`res` xor `invert`)) or (`BEQ` and (`is_equal` xor `invert`)) |  |
| | _polynomial:_ `-branch_cond + JALR + res[0] * (1 - mp_selector) * BLT + (1 - res[0]) * mp_selector * BLT + is_equal * (1 - mp_selector) * BEQ + (1 - is_equal) * mp_selector * BEQ = 0` | |
| `CPU-CO68` | `BRANCH[next_pc; pc, imm[0], arg1::DWordWL, JALR]` | branch_cond |
| `CPU-CO69` | `ADD<next_pc; pc, (2 * c_type_instruction + 4 * (1 - c_type_instruction))::DWordWL>` |  |

> **Note:** Document the choice to not have a multiplicity column here for padding

## Padding

The CPU can be padded with the following values, which have a corresponding row in the DECODE table, at the _odd_ address 1, only reachable through a HALT ecall.

This approach minimizes the number of dependent lookups, increasing only multiplicities in the DECODE table and the IS_BYTE lookup.

---

# SHIFT Chip

## Interface

The  chip has the following interface:

``` // param in: the value being shifted // param shift: the number of bits to shift `in` by // param direction: whether to shift left (0) or right (1) // param signed: whether to interpret `in` as a signed (1) or unsigned (0) integer // param word_instr: whether to execute the SLL/SR* (0) or SLLW/SR*W (1) instruction // out shifted: the resulting value SHIFT[shifted: DWord; in: DWord, shift: Byte, direction: Bit, signed: Bit, word_instr: Bit] ``` In other words, the  chip is designed to constrain that $

$ $

$ Here, `<<` and `>>` denote the _logical_ left and right shift operations, while `>>>` denotes the _arithmetic_ right shift operation.

## Columns

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
| `limb_shift` | `Bit[4]` | One-hot vector indicating whether $floor.l `shift` / 16 floor.r equiv i mod s$, where $s = 2$ when $`word_instr` = 1$ and $4$ otherwise. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `extension` | `Half` | sign extension of `in`. |
| `left` | `Bit` | Whether to perform a left-shift. |
| `right` | `Bit` | Whether to perform a right-shift. |
| `intra_limb_left` | `DWordHL` | `in << (shift % 16)` if `left` |
| `intra_limb_right` | `DWordHL` | `in >>> (shift % 16)` if `right` and `signed`;\ `in >> (shift % 16)` if `right` and `!signed` |
| `shifted` | `DWordHL` | $`in <</>>/>>>` (`shift` mod 32 dot (2 - `word_instr`))$ |

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
shifted := left * Σ_j = 0^i limb_shift[j] * intra_limb_left[i - j] + right * (Σ_j = 0^3 - i limb_shift[j] * intra_limb_right[i + j] + extension * Σ_j = 3 - i^3 limb_shift[j])
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

The `SHIFT` chip is comprised of  variables that are expressed using  columns:

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `SHIFT-A1.i` | i ∈ [0, 3] | `IS_HALFWORD[in[i]]` |
| `SHIFT-A2` |  | `IS_BYTE[shift]` |
| `SHIFT-A3` |  | `IS_BIT<direction>` |
| `SHIFT-A4` |  | `IS_BIT<signed>` |
| `SHIFT-A5` |  | `IS_BIT<word_instr>` |

## Explanation

This chip has a rather complex design as a result of designing it to fit in as few columns possible. We briefly discuss the intricacies of the design, attempting to illustrate its correctness.

The chip's design revolves around a two-phase shifting process: 1. shift `in` by `x := `shift` mod 16` bits, 2. shift that result by `(`shift`-x) mod 64` (or `mod 32` if ` `word_instr` = 1`). The intermediate value representing the state between the two phases is stored in the scratch variables `X` and `Y`. The definition of `shifted` describes how one can combine the `X`, `Y` and `extension` variables to construct the output value as described using `Half`-limbs. The output variable `out` is equivalent to `shifted`, but expressed using `Word`-limbs.

In the following, we cover how these two phases were designed to complement one another. Here, we start with discussing the _logical_ left/right shift operations only; the modifications required to compute the _arithmetic_ right shift will be discussed at the end.

### First phase

We zoom in on the first step. Here, we make use of the two lookup operations - ``HWSL[x: Half, y: B4]` := (`x` `<<` `y`) mod 2^16` (short for "HalfWord Shift Left"), and - ``HWSLC[x: Half, y: B4]` := `x` `>>` (16-`y`)` (short for "HalfWord Shift Left's Carry") Note here that one can use these two lookups to compute `out: Half[4] := in << y` as: $

$ as long as ``y` < 16`. Observing that ``HWSL[x,` 16-`y]` = (`x` `<<` (16-`y`)) mod 2^16`, and ``HWSLC[x,` 16-`y]` = `x` `>>` `y`` for ``y` in [1, 15]`, one can also use these lookups to compute `out := in >> y` as $

$ as long as `0 < `y` < 16`.

Observe now that the values being looked up are (almost) independent from the direction of the shift: only the shift-amount varies slightly. When we now define $

(16-`shift`) mod 16 & "when shifting right" ), $ it only takes some rearranging and combining of the values ``X[`i`] := HWSL[in[`i`], bit_shift]`` and ``Y[`i`] := HWSLC[in[`i`], bit_shift]`` to form the limbs of ``in <</>> shift` mod 16`. In the remaining case that ``right` = 1` and ``shift` = 0 mod 16`, the limbs of ``in <</>> shift` mod 16` simply match those of `in`.

### Second phase

Since we're operating on 16-bit limbs, all the limbs in ``in <</>> shift`` must also occur somewhere in ``in <</>> shift` mod 16`. The number of full-limbs we still need to shift is determined by the fifth and sixth least significant bit of `shift`. With `limb_shift` containing a unary decoding of the integer represented by these two bits, we find that the intermediate value needs to be shifted over by `i` limbs (to the `left` or `right`) when ``limb_shift[`i`]` = 1`. These things combined yield `shifted`'s definition.

Of course, when ``word_instr` = 1` and, thus, only ``shift` mod 32` should be considered, the bit-mask for the lookup constraining `limb_shift` is adjusted appropriately (see [shift:c:limb_shift_lookup]).

### Arithmetic right shift

Lastly, we discuss the case of performing the _arithmetic_ right shift. Here, `extension` is constrained to contain a repetition of `in`'s most significant bit. Copies of this variable are used for any full limbs shifted in when ``right` = `signed` = 1`. Moreover, `X[4]` contains a copy of `extension` shifted over by the right number of bits, to allow the construction of ``in >>> shift` mod 16` as the appropriate intermediate.

## Constraints

First, we constrain `bit_shift` based on whether we are left or right-shifting. [shift:c:zbs] makes sure `zbs` is set to `1` if and only if `bit_shift = 0`. This flag is used to indicate the special case that ``right` = 1` and ``shift` = 0 mod 16`.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C3` | `AND_BYTE[bit_shift; shift, 15]` | left |
| `SHIFT-C4` | `AND_BYTE[bit_shift; 2^8 - shift, 15]` | right |
| `SHIFT-C5` | `IsZero<zbs; bit_shift>` | μ |

Next, we shift the limbs of `in` left and right by the appropriate amount, storing the results in `X` and `Y` respectively. When `zbs = 1`, the output cannot be used to compose ``in >>/>>> shift` mod 16`. To resolve this, we override `Y[i] := in[i]` and `X[i] := 0` in this case.

The case of `left`-shifting and ``bit_shift` = 0` will be used for padding rows. To prevent unnecessary lookups in padding rows, we override ``X[i]` := `in[i]`` and ``Y[i]` := 0` here.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C6.i` | i ∈ [0, 3] | `HWSL[X[i]; in[i], bit_shift]` | 1 - zbs |
| `SHIFT-C7.i` | i ∈ [0, 3] | `zbs` => `X[i]` = `in[i]` dot `left` |  |
| | | _polynomial:_ `zbs * (X[i] - in[i] * left) = 0` | |
| `SHIFT-C8` |  | `HWSL[X[4]; extension, bit_shift]` | 1 - zbs |
| `SHIFT-C9` |  | `zbs` => `X[4]` = 0 |  |
| | | _polynomial:_ `zbs * X[4] = 0` | |
| `SHIFT-C10.i` | i ∈ [0, 3] | `HWSLC[Y[i]; in[i], bit_shift]` | 1 - zbs |
| `SHIFT-C11.i` | i ∈ [0, 3] | `zbs` => `Y[i]` = `in[i]` dot `right` |  |
| | | _polynomial:_ `zbs * (Y[i] - in[i] * right) = 0` | |

### Full-limb shifting

Next, we constrain that `limb_shift` is a proper unary encoding of the fifth (and sixth if ``word_instr` = 0`) bit of `shift`. For this to be the case, three requirements must be satisfied: + *unary(0)*: ``limb_shift[`i`]` in {0, 1}` for `i in [0, 3]`, + *unary(1)*: ``limb_shift[`i`]` = 1` for exactly one `i`, and + *proper encoding*: ``limb_shift[`i`]` = 1 <=> 1/16 (`shift &` (48-32 dot `word_instr`)) = i` The first requirement is enforced by constraint [shift:c:limb_shift_is_bit]. To construct a constraint for the second and third requirement, observe that $ 1/16 dot (`shift &` (48-32 dot `word_instr`)) in cases( {0, 1, 2, 3} &"if" `word_instr` = 0, {0, 1} &"if" `word_instr` = 1 $ Observe moreover that, assuming *unary(0)*, the expression $ 1/16 dot (1 + sum_(i=0)^3 (16i-1) dot `limb_shift[`i`]`) $ can evaluate to `i` if and only if ``limb_shift[`i`]` = 1`, while the others are `0`. This means that the relation $ 1 + sum_(i=0)^3 (16i-1) dot `limb_shift[`i`]` = `shift &` (48-32 dot `word_instr`) $ enforces both *unary(1)* and *proper encoding*. This is the exact relation [shift:c:limb_shift_lookup] enforces.

Hereafter, one must only check that `out` is the proper cast of `shifted` into a `DWordWL`.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `SHIFT-C12.i` | i ∈ [0, 3] | `IS_BIT<limb_shift[i]>` |  |
| `SHIFT-C13` |  | `AND_BYTE[(1 - limb_shift[0]) + 15 * limb_shift[1] + 31 * limb_shift[2] + 47 * limb_shift[3]; shift, 48 - 32 * word_instr]` | μ |
| `SHIFT-C14.i` | i ∈ [0, 1] | `out[:2]` = `shifted[:4]` |  |
| | | _polynomial:_ `out[i] - (shifted::DWordWL)[i] = 0` | |

### Miscellaneous

*Note*: `is_negative` is not used when `signed = 0`. As such, there is no problem with it being unconstrained in this case.

### Lookups

This chip adds the following interaction to the lookup.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C15` | `SHIFT[out; in, shift, direction, signed, word_instr]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

### is_negative

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `SHIFT-C2` | `MSB16[is_negative; in[3]]` | signed |

### left_flag

| Tag | Description |
|-----|-------------|
| `SHIFT-C1` | `direction` => `μ` = 1 |
| | _polynomial:_ `direction * (1 - μ) = 0` |

---

# BRANCH Chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `pc` | `DWordWL` | The current pc, used as base address when `!JALR` |
| `offset` | `Word` | The offset from the base address to jump to |
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
next_pc_unmasked (when iter=0) := 2^16 * next_pc_high[0] + 2^8 * next_pc_low[1] + unmasked_low_byte[0]
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

The `BRANCH` chip is comprised of  variables that are expressed using  columns:

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `BRANCH-A1.i` | i ∈ [0, 1] | `pc` is range checked, `IS_WORD[pc[i]]` |
| `BRANCH-A2` |  | `offset` is range checked, `IS_WORD[offset]` |
| `BRANCH-A3.i` | i ∈ [0, 1] | `register` is range checked, `IS_WORD[register[i]]` |
| `BRANCH-A4` |  | `IS_BIT<JALR>` |

## Constraints

> **Note:** Check correspondence with CPU for passing in `offset` as word or dword

We constrain `next_pc` to be ``base_address` + `offset``, where `base_address` equals `pc` when ``JALR` = 0` and `register` otherwise.

The range checks on `unmasked_low_byte` and `next_pc_low[0]` are performed implicitly by the `AND_BYTE` lookup.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `BRANCH-C1` |  | 1 - JALR ⇒ `ADD<next_pc_unmasked; pc, offset::DWordWL>` |  |
| `BRANCH-C2` |  | JALR ⇒ `ADD<next_pc_unmasked; register, offset::DWordWL>` |  |
| `BRANCH-C3` |  | `IS_BYTE[next_pc_low[1]]` | μ |
| `BRANCH-C4` |  | `AND_BYTE[next_pc_low[0]; unmasked_low_byte[0], 254]` | μ |
| `BRANCH-C5.i` | i ∈ [0, 2] | `IS_HALFWORD[next_pc_high[i]]` | μ |

This chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `BRANCH-C6` | `BRANCH[next_pc; pc, offset, register, JALR]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

---

# MEMW Chip

## Columns

### Input

| Name | Type | Description |
|------|------|-------------|
| `is_register` | `Bit` | Whether the address represents a register index |
| `base_address` | `DWordWL` | The base address to read/write from/to, gets offset by $[0, 7]$, depending on how big the access is |
| `value` | `BaseField[8]` | The values to store in memory. For regular memory, these should be (up to) 8 range-checked `Byte`s; registers are stored as two range-checked `Word`s |
| `timestamp` | `DWordWL` | The timestamp at which this memory access is said to occur |
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
| `address_add` | `DWordHL[7]` | `address_add[i] = base_address + i + 1` |
| `old_timestamp` | `DWordWL[8]` | The timestamp at which the address was last accessed |

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

The `MEMW` chip is comprised of  variables that are expressed using  columns:

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `MEMW-A1.i` | i ∈ [0, 1] | `IS_WORD[base_address[i]]` |
| `MEMW-A2` |  | `IS_BIT<write2>` |
| `MEMW-A3` |  | `IS_BIT<write4>` |
| `MEMW-A4` |  | `IS_BIT<write8>` |
| `MEMW-A5` |  | `IS_BIT<write2 + write4 + write8>` |
| `MEMW-A6.i` | i ∈ [0, 1] | `IS_WORD[timestamp[i]]` |

Our assumptions do not explicitly cover any range checks for the `is_register` and `value` columns, as these are not necessary for the correctness of this chip in isolation. These properties are necessary for the consistency of the system as a whole, and therefore we document it here, keeping the type information as a reading help.

## Constraints

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `MEMW-C1` |  | `IS_BIT<μ_sum>` |  |
| `MEMW-C2` |  | `w2` => `μ_sum` |  |
| | | _polynomial:_ `w2 * (1 - μ_sum) = 0` | |
| `MEMW-C3` |  | `ADD<address_add[0]::DWordWL; base_address, 1>` | w2 |
| `MEMW-C4.i` | i ∈ [1, 2] | `ADD<address_add[i]::DWordWL; base_address, i + 1>` | w4 |
| `MEMW-C5.i` | i ∈ [3, 6] | `ADD<address_add[i]::DWordWL; base_address, i + 1>` | write8 |
| `MEMW-C6.i` | i ∈ [0, 6], j ∈ [0, 3] | `IS_HALFWORD[address_add[i][j]]` |  |
| `MEMW-C7` |  | `LT[1; old_timestamp[0], timestamp, 0]` | μ_sum |
| `MEMW-C8` |  | `LT[1; old_timestamp[1], timestamp, 0]` | w2 |
| `MEMW-C9.i` | i ∈ [2, 3] | `LT[1; old_timestamp[i], timestamp, 0]` | w4 |
| `MEMW-C10.i` | i ∈ [4, 7] | `LT[1; old_timestamp[i], timestamp, 0]` | write8 |

As long as `timestamp` is properly range-checked, the presence of `old_timestamp` in the memory argument automatically ensures appropriate range checking (as long as no external entities provide negative multiplicities without range checking the timestamp). This ensures the assumptions for `LT` are satisfied.

We additionally check that the address does not overflow for more significant bytes of the access.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW-CR11` | `LT[1; base_address, address_add[0]::DWordWL, 0]` | write2 |
| `MEMW-CR12` | `LT[1; base_address, address_add[2]::DWordWL, 0]` | write4 |
| `MEMW-CR13` | `LT[1; base_address, address_add[6]::DWordWL, 0]` | write8 |

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

This chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `MEMW-CO22` | `MEMW[old; is_register, base_address, value, timestamp, write2, write4, write8]` | μ_read |
| `MEMW-CO23` | `MEMW[is_register, base_address, value, timestamp, write2, write4, write8]` | μ_write |

## Future optimization ideas

- Fast path for aligned memory access where all bytes have the same old timestamp - MEMB chip that deals does a one-byte write to remove old_timestamp from here (uncertain tradeoffs) - Compute `base_address[1] + 1` once and have high words of `address_add` as Words - Improve overflow trapping somehow so we don't need `LT` (could tie into previous one by checking carry bit of the +1) - Adding `μ_sum`/`w2`/`w4`/`write8` multiplicities to the `IS_HALFWORD` lookups may make some GKR things faster if there are known zeroes.

---

# LT Chip

## Columns

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

The `LT` chip is comprised of  variables that are expressed using  columns:

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `LT-A1` |  | `IS_WORD[lhs[0]]` |
| `LT-A2` |  | `IS_WORD[rhs[0]]` |
| `LT-A3` |  | `IS_BIT<signed>` |

We assume the inputs `lhs`, `rhs` and `signed` are partially range checked.

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
| `LT-C4` | `IS_HALFWORD[lhs[1]]` | μ |
| `LT-C5` | `IS_HALFWORD[rhs[1]]` | μ |

And then we constrain the subtraction, taking care of the remaining range checking not yet covered by the assumptions or the `MSB16` lookup.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `LT-C6.i` | i ∈ [0, 1] | `IS_BIT<carry[i]>` |  |
| `LT-C7.i` | i ∈ [0, 3] | `IS_HALFWORD[lhs_sub_rhs[i]]` | μ |

The chip contributes the following to the lookup argument.

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `LT-C8` | `LT[lt; lhs::DWordWL, rhs::DWordWL, signed]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

---

# MUL Chip

## Columns

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

The `MUL` chip is comprised of  variables that are expressed using  columns:

`mat(delim: , top; bottom)` }

## Assumptions

| Tag | Range | Description |
|-----|-------|-------------|
| `MUL-A1.i` | i ∈ [0, 3] | `IS_HALF[lhs[i]]` |
| `MUL-A2.i` | i ∈ [0, 3] | `IS_HALF[rhs[i]]` |

The following range checks are assumed to be performed/enforced outside of this chip:

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

## Notes

- `lo` and `hi` are stored in `DWordHL`s (rather than `DWordWL`s) because of their values being range checked. Since it is not required that both `μ_lo` and `μ_hi` are non-zero at the same time, one cannot safely assume their range to be checked elsewhere.

As an optimization, one might be able to use a `DWordWL` and `DWordHL` to store `lo` and `hi`, where one would decide which to store in which based on the multiplicities `μ_lo` and `μ_hi`; the value sent into the lookup could then be assumed range-checked by the other side of the relation. This optimization was not included at this moment because of its negative impact on the readability and verifiability of the chip.

---

# DVRM Chip

//  chip = load_chip("src/dvrm.toml", config)

*placeholder chapter: WIP*

---

# LOAD Chip

## Columns

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

The `LOAD` chip is comprised of  variables that are expressed using  columns:

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
| `LOAD-C9` | `LOAD[res::DWordWL; base_address, timestamp, read2, read4, read8]` | -μ |

## Padding

The table can be padded to the next power of two with the following value assignments:

---

# ECALL Chips

##  chip

### Columns

The  chip leverages  variable, spanning  columns:

### Assumptions

It is assumed the input is range checked:

### Constraints

The  chip: + makes sure register `x10` (containing the exit code) equals `0` ([halt:c:read_zero_exit_code]), + writes `0` to all other registers ([halt:c:zeroize_registers_lo]/[halt:c:zeroize_registers_hi]), and + sets `pc` equal to `1` ([halt:c:pc]). Note that the writes performed by all these interactions are accompanied by the timestamp `2^64-1`; the maximum timestamp. This prevents any other operation involving memory from being executed hereafter.

[ Observe that --- in its current state --- this solution puts the burden of verifying the register cleanup on the verifier inside of the lookup argument. Alternatively, one could add 31 lookups to the "memory" table to remove the _known_ final tokens for the registers there. ])

#### Lookup

The HALT chip contributes the following interaction to the lookup-argument:

*Note*: [`93` is the system call number corresponding to `sys_exit`.]

### Padding

This chip should only contain a single row. Given that `2^0 = 1`, this chip does not need to be padded. As such, no padding is defined.

---

# BITWISE Chips

## Columns

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
| `ZERO` | `Bit` | whether $`X` = 0 and `Y` = 0$ |
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
| `μ_HWSLC` | `BaseField` |  |

The  chip is comprised of  variables that are expressed using  columns. Of these, the _input_ and _output_ variables ( in total) are precomputed.

*Note*: This table contains one row for every possible value of `(X, Y, Z)`. As such, it has length `2^8 dot 2^8 dot 2^4 = 2^(20)`.

## Lookup

This chip adds the following interactions to the lookup:

## Areas of Optimization

The following ideas may prove to be optimizations for the  chip: + Extend `IS_BYTE[X]` to `ARE_BYTES[X, Y]`, such that two bytes are range checked at once. When only a single check is required, one can still execute `IS_BYTE[X] := ARE_BYTES[X, 0]`. + Drop `MSB8` column, and instead define the `MSB8` lookup as `MSB8<X> := MSB16[256X]`. Note: currently, `MSB8` also implicity range checks the input `X` (the lookup fails if `X` is not a `Byte`). This optimization should only be executed when all chips leveraging `MSB8` do _not_ need this implicit range check. + Place the 16-bit (`AND`, `OR`, `XOR`, `MSB16`, `ZERO`, etc.) and 20-bit (`HWSL`, `HWSLC`, `IS_B20`) lookups in separate tables. + Combine `HWSL` and `HWSLC` into a single lookup (see also \).

## Constraints

### contributions

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `BITWISE-C1` | `AND_BYTE[AND; X, Y]` | -μ_AND |
| `BITWISE-C2` | `OR_BYTE[OR; X, Y]` | -μ_OR |
| `BITWISE-C3` | `XOR_BYTE[XOR; X, Y]` | -μ_XOR |
| `BITWISE-C4` | `MSB8[MSB8; X]` | -μ_MSB8 |
| `BITWISE-C5` | `MSB16[MSB16; X + 256 * Y]` | -μ_MSB16 |
| `BITWISE-C6` | `ZERO[ZERO; X + 256 * Y]` | -μ_ZERO |
| `BITWISE-C7` | `IS_BYTE[X]` | -μ_IS_BYTE |
| `BITWISE-C8` | `IS_HALF[X + 256 * Y]` | -μ_IS_HALF |
| `BITWISE-C9` | `IS_B20[X + 256 * Y + 65536 * Z]` | -μ_IS_B20 |
| `BITWISE-C10` | `HWSL[SLL; X + 256 * Y, Z]` | -μ_HWSL |
| `BITWISE-C11` | `HWSLC[SLLC; X + 256 * Y, Z]` | -μ_HWSLC |