# Two VMs for recursion: what we built, what broke, and what we learned

*Implementation of the 3MI Field VM (#971) and Recursion (#943) specifications, as merged on October 2. The work is closed; this document is meant as input for the spec. Proposed Typst text for the open sections is in `1_proposed_spec_text.md`.*

## The short version
 
We built the recursion the spec describes:
- the **Lambda VM (RISC-V)** runs the binary half, v_bin: hashing, Fiat–Shamir, Merkle paths and proof of work;
- the **Field VM** runs the field half, v_fld: constraints, composition, DEEP and FRI folds;
- both are **proven together in one proof**, bound by a pseudo-`PAGE`: the spec's *split verification*.

It runs end to end, from an Ethereum block (a 10-transfer ethrex test block) to a uniform recursion tower at ~111 bits.

Getting there meant three kinds of work:
- filling in several parts the spec leaves as TODO;
- changing a few things it states;
- closing soundness holes that only appear once the two halves really run apart.

One finding sums up the design question: **in this two-VM design, the field VM is not the bottleneck; the binary VM is.**
- The Field VM next to algebraic hash chips verifies an epoch proof as fast as a dedicated recursion machine.
- With the hashes on RISC-V, as the spec places them, the same work is ~25× slower.

The document is in three parts:
- **Part I** follows the spec as written;
- **Part II** covers what the spec leaves as TODO and how we made it work;
- **Part III** covers the experimental branches: a binary-field prover for the hash half, *split proving* across proof systems with Flock (the section the spec leaves as TODO), and BLAKE3.

A word on names. Lambda's own recursion machine is the **LFM** (Lambda Field Machine): one machine with write-once memory and its own RPX hash chips. It is the baseline throughout. In the implementation it is also the language in which the verifier is first written before being cut into the two halves.

---

## Part I — Following the spec

### 1. The Field VM on its own

The first step was the Field VM as written:
- the single `FMA d == a·b + c` instruction over the cubic extension of Goldilocks;
- registers `ZERO`, `PC` and N general-purpose ones;
- both addressing modes;
- input and output hinting;
- the halt self-loop at PC 0, with execution starting at PC 1.

Assembler, executor, AIR and prover follow the text closely.

A few pieces the spec does not define had to be decided along the way (the `MEM` table, how it is bound to the statement, hint values). Part II describes them.

One change was needed in our own framework. The out-mux reads the next row's registers, and the framework only opened the LogUp accumulator at the next row. Every other next-row read was silently checked against zero until those columns were opened as well.

### 2. Running a real verifier on it

The next step was real work: the field half of verifying one epoch proof of the Lambda VM, with RPX as the hash. The existing epoch verifier was translated into Field VM programs, one FMA per field operation, with inversions as hints that the FMA checks.

The first version used 11× the cells of the same arithmetic in the LFM. The gap was structural:
- the explicit `MEM` table;
- per-row decoding;
- every argument being an extension element.

An optimization campaign closed most of it, taking proving from 9.6 s to 4.5 s (2.8 s with degree 3 at blowup 2) and the gap from ~11× to ~3×:
- N = 2 registers instead of 5, since generated code works through memory;
- chained power-of-two segments instead of one padded table;
- a chunked `DECODE`;
- operand fusion.

Two ideas did not pay off: a register-free row variant (only 13% of rows could use it) and a third memory argument.

### 3. The hash half: where the trouble started

The field half is half the verifier. Two versions of the hash half were built:
- **Option A, not the spec:** the Field VM plus the LFM's RPX hash chips in one proof. It proves an epoch's verification **on par with the LFM itself** (within ±1% over 10 interleaved runs).
- **Option B, the spec:** the Field VM plus a RISC-V program for the hash half. The first version was not viable: a software RPX permutation costs ~73,000 RISC-V cycles, and one production epoch needs 82,323 of them. That is 6.07 G cycles and an extrapolated ~24 h of proving, about 11,000× the LFM.

Making option B usable took two changes:
- an **RPX precompile** in the Lambda VM;
- replacing the interpreter that ran the cut program with **native code compiled for each proof shape**. In the binary half, the precompile and the compiled code brought the cost to 55–85 cycles per hash.

Option B then verified an epoch in ~199 s, still about 25× the LFM, while option A stayed at parity. That gap never went away, and it is the main design lesson below.

### 4. Recursion: the tower that would not shrink

The Recursion chapter defines `verify*`. It was built as described:
- one program holding the base verifier and the recursive verifier, both run at every level;
- every assertion of the first multiplied by 1 − b and every one of the second by b, with b² = b asserted ungated;
- `comm(x)` committing to the base instance and `comm(y)` to the program.

Each level is a single proof containing both VMs, as the spec's split verification describes; how the two VMs share data is in Part II.

**It did not converge.** Verifying one epoch cost 337× the epoch's own cycles, so every level was bigger than the one it verified. What brought it below 1×:

| Change | Cost of verifying one epoch |
|---|---:|
| Starting point: one Merkle tree and one FRI per table, interpreted v_bin | 337× |
| Native compiled v_bin | 35.5× |
| One Merkle tree per phase over all trace tables, and one FRI | 11.5× |
| No keccak tables in the recursive guest (they were 26% of all hash permutations) | 2.38× |
| Epochs of 2²³ cycles | **0.615×** |

Two of these deserve a comment:
- **Sharding.** Sharding tables made the cost grow with the epoch size, so a bigger epoch did not help until shards were gone.
- **Batched commitments depart from the spec,** which commits to every table individually (`recursion.typ:347–349`). Program tables (`DECODE`, `FIELD_VM_DECODE`, the address maps) keep their own trees, so the program identity is still a list of roots; trace tables share one tree per phase. The same change on a mainnet block gave −73% proof size and −75% verification time.

Two more problems came from uniformity:
- **A straight-line verifier depends on the proof shape** (table heights, query counts), so its program id is constant only at a fixed point. The construction emits `verify'` for a shape, proves a level, emits again from that proof, and repeats until the id repeats.
- **Shapes that depend on data break that fixed point.** Three causes appeared:
  - branches on query bits;
  - the number of memory pages the guest touches;
  - a multiplication table that deduplicates operands, while the inactive branch multiplied arbitrary values by its zero gate.

  Fixes: branch-free code; reused guest memory; multiplying by the gate with a mask.

The tower converged: three levels with one program id, at 7.96 M cycles at Q = 101 and grinding 28, and 7.69 M at Q = 110 once the program tables shared one preprocessed tree. (Q is the number of FRI queries. At blowup 4 each query adds about 2 bits of security and grinding adds its own bits, so Q = 110 with 20 bits of grinding gives ~128 bits, and Q = 80 with 32 bits gives ~111.)

### 5. From an Ethereum block

A tower needs a base. A block's proof is a bundle of epoch proofs plus a global proof, and both branches of `verify'` run at every level, so verifying the whole bundle inside `verify'` would make every level pay for it. Instead, an **anchor** program verifies the bundle once, and `verify'` with b = 0 verifies the anchor's proof.

Two more spec issues showed up here:
- **The chapters disagree on the commitment domain.** The Recursion chapter backs stdout with `PAGE` tables (`recursion.typ:359–394`); the Memory chapter calls that domain unpaged, and the Streaming chapter forbids `PAGE` there. The implementation binds stdout through the COMMIT bus, whose term the verifier recomputes.
- **The node that verifies two proofs sets the security ceiling.** At Q = 80 it sits at 0.973 × 2²³; Q = 110 would need ~10.8 M cycles.

End to end on `ethrex_10_transfers`, a 10-transfer test block from ethrex (CPU server with 96 cores):
- from the block to the first recursive proof, ~37 min;
- each further level, ~7.5 min;
- every level 8.16 M cycles and ~97 GB of memory, at ~111 bits;
- one program id at every level; the final verifier accepts level k and rejects k + 1.

For comparison, the same tower with the LFM's hash chips instead of RISC-V (option A) proves a level in ~25–45 s with 15–29 GB, at Q = 110, end to end from a mainnet block.

### What Part I says about the two-VM design

Measured on the same CPU server, verifying the same epoch proof:

| Field half | Hash half | Prove |
|---|---|---:|
| LFM | LFM (one machine) | ~8 s |
| **Field VM** | LFM's RPX chips | **on par** (±1%) |
| **Field VM** | **RISC-V**, as in the spec | **~199 s** (~25×) |

In the tower the gap is ~10× per level, with ~4× the memory.

**The Field VM is not the problem.** Inside a monolithic RISC-V verifier, field arithmetic is 70–80% of the cycles; DEEP reconstruction alone is 60–74%. Moving it to the Field VM is exactly right, and the Field VM handles it as well as a dedicated machine.

What stays on RISC-V is expensive for reasons that have little to do with fields:
- hashing through a precompile;
- byte order: big-endian commitments cost ~300 cycles per Merkle level without a byte-reverse instruction;
- decoding the proof: 41% of a byte-native binary half;
- rebuilding table shapes: 24% of a byte-native binary half.

**The binary half needs its own design.** Either:
- a hash, precompile ABI and encodings chosen for it (little-endian, bulk absorb, zero-copy proof access, shapes embedded in the verifier); or
- a spec that lets the hash half run on algebraic hash chips next to the Field VM.

---

## Part II — About the TODOs

These are the parts the spec leaves open or as TODO. Each one turned out to be needed for a sound proof that converges.

### 6. Split verification in one proof: how the two VMs share data

**Spec:** the Split verification section (`recursion.typ:145–239`) has the two halves share a record and be proven as one program. The input, private input and record "are made available to the VMs by means of `PAGE` tables" (`recursion.typ:345–346`), but nothing says how the Field VM reads them. (Proving the halves *separately* is split proving, the TODO at `recursion.typ:240`; that is Part III, §12.)

**Options.** Robin pointed out that the side-by-side Field VM and Lambda VM case "can still be a single proof". For the shared data there were two proposals:
- one commitment serving as both the Field VM's `MEM` and the RISC-V memory initialization. The binary VM would then have to rebuild the field version of the whole input to check it;
- after Robin talked with Erik: an extension-field version for the Field VM and a byte version for the binary VM, in the same proof, bound through LogUp.

**Taken:** the second, as a **pseudo-`PAGE`**: 24 bytes per cell, three little-endian limbs, mapped to Field VM addresses by a preprocessed `FADDR`, placed in page-aligned private-input pages. Two additions:
- **A second direction, `BACK`,** for values the binary half computes for the Field VM (challenges, query bits). The proposal covered only the shared input.
- **A constraint that each limb is below p.** Without it, the bytes of v + p pass as v; a test shows it.

**What it surfaced.** Running both halves in one proof exposed soundness holes in our first implementation that nothing in the spec prevents. Each one was confirmed by a test that builds the forged proof: the proof was accepted before the fix and is rejected after it, which also shows the test is not vacuous. The holes:
- query bits decomposed from x + p instead of x, which picks different query positions for the same field element;
- the fourth lane of shared hints, of `BACK` cells and of published words left unchecked when a four-element RPX digest met a three-coordinate cell;
- program constants not bound;
- a hint the Field VM decomposed that the binary half never saw;
- the binary program's identity not bound;
- limbs encoded as v + p.

### 7. The communication record

**Spec:** the record "primarily consists of the Fiat–Shamir challenges", and the overview is left to "update once @verification is complete" (`recursion.typ:329–334`).

**What we found.** The holes in §6 share a cause: **the record must contain every value one half assumes and the other produces, each with one enforced encoding.** Challenges alone are not enough.

**Options.** Let the record be whatever the cut makes cross, or define it by meaning.

**Taken:** first the cut, which gave tens of thousands of intermediate cells and tied both halves to one compiler. Then a record where only named values cross, in a layout computed from the proof shape alone:
- **from v_bin to v_fld:** the Fiat–Shamir challenges, in transcript order, and the query bits;
- **from v_fld to v_bin:** the register LDE and the statement words, which v_bin hashes.

Values both halves can compute from the shared input are computed by both. The emitted verifier is checked against the layout every time it is built.

On a Q = 80 level, 6,129 named cells cross instead of 32,220 cut-chosen ones, and the level costs 6.8% fewer cycles. This is what lets either half be rewritten or audited on its own.

### 8. The verification chapter

**Spec:** `verifier.typ` is a TODO ("sigma protocol, layout the various steps; Fiat–Shamir transformation").

**Taken:** one order of transcript steps, followed by both the native verifier and the in-VM one. The record in §7 is derived from it. The proposed text lists the steps.

### 9. The level counter

**Spec:** soundness of the construction "is not yet proven"; a recursion-level counter is suggested in an aside (`recursion.typ:131–137`).

**Taken:** required. Each level publishes `level = 1 + b · level_child`, and the final verifier accepts level k and rejects k + 1. Ten tampers are rejected, among them b = 2, b flipped, wrong commitments, a forged child level and another program of the same shape.

### 10. Field VM details

**`MEM`.** The spec asks for "a committed table containing the memory as well as the multiplicities" (`field_vm.typ:16–27`) without defining it.
- *Options:* preprocessed or constrained addresses.
- *Taken, confirmed by Robin:* a chip with one row per address; uniqueness by constraint (`ADDR₀ = 0`, `ADDR' = ADDR + 1`), because with LogUp alone a prover could commit two rows with one address; backfilled by the prover, so write-once in execution.
- *Public cells:* a flag sends them on a bus the verifier closes itself.

**Hint values.** The spec leaves them as an implementation detail (`field_vm.typ:147`).
- An output hint with `imm₀ ≠ 0` is solved.
- A single linear input hint (as in `INV`) is solved.
- Anything else comes from the host.
- Output hints on `ZERO` and hint collisions are rejected by the executor.

**Smaller choices:**
- the extension polynomial w³ − 2;
- `DECODE` packed as 11 base-field values instead of extension elements, since the X² coefficient was always zero;
- N = 2 registers;
- segmented execution.

---

## Part III — Experimental implementations

### 11. A binary-field prover for the hash half (Binius64)

The Recursion chapter names Flock as a possible prover for the hash half, so the first experiment measured what a binary-field prover does with this hash load.
- **Binius64 and keccak:** 65,536 keccak-f permutations in 2.1–3.0 s. An epoch needs ~70k compressions, so the hash half would take ~2.3–3.2 s, against ~12 minutes for keccak on the RISC-V VM. That is similar to, or a little better than, the LFM's whole epoch, before coupling the two proofs or verifying a binary-field proof in the next layer.
- **Flock and BLAKE3:** Flock was 6–7× faster than Binius64 on the same hash, proving 65,536 BLAKE3 compressions in 0.14 s.

### 12. Split proving across proof systems (Flock)

This is the spec's *split proving*, the section left as TODO (`recursion.typ:240`). Robin described it as "the road towards splitting the two parts up entirely, and through that enabling Flock".

**The idea.** Prove each half in the proof system that suits it:
- the field half in a STARK over Goldilocks, with the Field VM;
- the hash half in Flock, a prover over a binary field that is very fast at bitwise hashes such as BLAKE3.

Two proofs per step instead of one.

**One step, in order.**

```text
   verifier of one epoch proof
              │  cut
      ┌───────┴────────┐
      ▼                ▼
   v_fld             v_bin
   Field VM          Flock circuit
   (Goldilocks)      (binary field)
      │                │
      ▼                ▼
   proof A          proof B
      │                │
      └─── shared values published in both ───┘
                       │
                       ▼
         the next level checks they are equal
         and verifies both proofs
```

1. **Cut.** The verifier is cut as in Part I: field operations to v_fld, hashing, bits and Merkle work to v_bin.
2. **Proof A.** The Field VM runs v_fld and proves it, as a STARK over Goldilocks.
3. **Proof B.** v_bin is written as a Flock circuit and proved:
   - all 79,343 BLAKE3 compressions of an epoch;
   - the glue around them, as Flock gates: Merkle-path selects, packing and unpacking, bit decompositions with their check that the value is below p, reversed digests;
   - prover hints as private witness.

   BLAKE3 is the 6-round variant our executor used, with Flock patched to match.
4. **Publish.** Every value both halves use becomes a public input of both proofs.
5. **Check.** Someone checks that A and B published the same values and verifies both proofs. In the first version a host did the comparison (together with one keccak-f permutation, since Flock has no keccak gate). In the tower, the next level does it.

**What has to be bound, and why it is so much.** Two kinds of values cross:
- **The record:** values one half computes and the other uses (challenges, query bits, results). For one epoch this was ~94k cells.
- **The proof data both halves read:** ~100k cells. The Field VM folds FRI with the opened values of the proof being verified, but only Flock can check those values against their Merkle roots. So the opened values and Merkle siblings must reach both proofs. They cannot be cut, because they are the proof itself.

Together that is 194,345 cells per step. Moving work to the side that owns it cut the record:
- assertions to Flock;
- bit decompositions to the Field VM;
- each half its own constants.

The derived values dropped from ~94k to ~15–25k cells. **The ~94,153 cells of proof data stay:** that is the floor of this design.

**One step, measured** (a real BLAKE3 epoch):
- Field VM, 0.59 s; Flock, 0.50 s; total **1.10 s**;
- against 5.05 s for the LFM with RPX, ~4.6× faster in prove time (1.4–1.7× once the circuit build is counted).

**Recursion.** The next level verifies both proofs: the Field VM proof with our verifier, and the Flock proof with Flock's verifier replayed in the circuit. Flock's verification costs ~166k BLAKE3 compressions, and 93% of them go to digesting the published values. Built this way, the tower is sound up to height 2.

**It does not converge.** The published interface grows with every level: 112,804 cells at level 1 (after the cuts), 972,491 at level 2. A level verifies the previous level's two proofs, and the data it must share between its own two halves is now the previous proofs' data, published values included. So the interface feeds itself.

The obvious fix is to bind the shared values with one random linear combination instead of publishing them. It is cheap in Goldilocks, but costs ~20 billion constraints per challenge in Flock, where a Goldilocks multiply-add is ~20–30k Boolean constraints.

**Conclusion:** split proving across fields needs a binding of the shared values whose cost does not grow with them and is cheap in both fields. Until then, split verification in one proof (Part II, §6) is the construction that converges.

### 13. BLAKE3

**6 or 7 rounds.** The code used a 6-round BLAKE3 variant, so the standard 7-round BLAKE3 was compared against it:
- **In the base layer** (proving a block), the seventh round is practically free: +0.4% prove and +3% verify on a mainnet block. BLAKE3 stays ~12% faster than keccak there.
- **Inside a circuit over Goldilocks** (the LFM wrap), the seventh round costs **+17% prove**: every round is cells, 620 M → 695 M.
- **In Flock,** the 6-round variant was used, with Flock patched to match.

So, if recursion moves to a binary-field prover, standard BLAKE3 has no downside. In a field circuit, 6 rounds save ~17% at the price of a non-standard hash whose security margin is not quantified.

**In the one-proof tower.** The last experiment asked whether the tower of Part I could use BLAKE3 instead of RPX:
- It runs: three levels with one program id at Q = 1 (4.42 M cycles per level).
- At Q = 4 it needs 7.42 M cycles and 117 GB, and it only fits in memory by recomputing low-degree extensions. A BLAKE3 compression is a 3,268-column table; an RPX permutation, ~330.
- In a binary half generated from our word-lane representation, BLAKE3 cost ~4× RPX. Two compiler fusions cut that 31%, not enough for Q = 80.

Over plain bytes the picture flips: a Rust binary half over bytes, with a bulk-absorb call, was 4.4× cheaper than RPX in a proxy measurement. **The hash, its precompile calling convention and the digest representation have to be chosen together.** With the RPX tower already at ~111 bits, the BLAKE3 line was closed here.

---

## What we would put in the spec

Each row is a concrete change proposed for the spec. The Typst text for it, ready to paste, and the evidence behind it are in the attached `1_proposed_spec_text.md`, in the section listed.

| Proposal | Where in the spec | Section of `1_proposed_spec_text.md` |
|---|---|---|
| Write the verification chapter as an ordered list of transcript steps, from which the record follows | `verifier.typ` (empty, TODO) | §1 |
| Define the communication record as every value that crosses between the halves, named, with producer, consumer, type and a count from the proof shape | `recursion.typ:329–334` (overview TODO) | §2 |
| Say how the two VMs share data when both halves are proven in one proof: the pseudo-`PAGE`, in both directions, with canonical limbs | `recursion.typ`, Split verification (new subsection) | §3, first block |
| State what split proving needs: a binding between the two proofs whose cost does not grow with what crosses | `recursion.typ:240–241` (Split proving TODO) | §3, second block |
| Make the level counter required, and say how b is constrained | `recursion.typ:131–137` | §4 |
| Give every crossing value one enforced encoding (bits, limbs, whole digests, base-field values) | `recursion.typ`, record section | §9 |
| Define the `MEM` table (chip, unique addresses, write-once) and how public cells are bound | `field_vm.typ:16–27` | §7, §8 |
| Allow one tree per phase over all trace tables and one FRI, with program tables committed separately | `recursion.typ:347–349` | §11 |
| Separate instance and program commitments, list what identifies a split program, and require a shape fixed point for a uniform verifier | `recursion.typ:341–357` | §12 |
| Replace "typically shrinks" with a measurable inequality, `C_verify(L) < 2^L` | `recursion.typ:80–85` | §14 |
| Resolve the commitment-domain conflict by binding stdout through the COMMIT bus | `recursion.typ:359–394`, `memory.typ`, `streaming.typ` | §10 |
| Smaller points: the extension polynomial, hint values, restricting the ISA, segments, N as a parameter | `field_vm.typ` | §5, §6, §13 |
| Use one name for `FIELD_VM_DECODE` and fix the index typos in `field_vm.toml` | `recursion.typ:344`, `field_vm.toml` | — (editorial) |

## Open points for the spec

- **Where the hash half runs.** On RISC-V, as the spec places it, it is ~25× the cost of algebraic hash chips next to the Field VM (Part I). Keeping it there needs a binary half designed for it; the alternative is to allow algebraic hash chips.
- **Batched trace commitments.** Uniform recursion converged only with one tree per phase and one FRI. Whether that becomes a requirement is open.
- **Proof shapes and the commitment domain.** A straight-line verifier needs a fixed proof shape; whether recursion is meant over fixed or arbitrary shapes is open. So is whether the commitment domain is paged or unpaged, where the chapters disagree.
- **Binding for split proving.** Proving the halves separately (with Flock) needs a cross-field binding whose cost does not grow with what crosses; which construction the spec intends is open.
