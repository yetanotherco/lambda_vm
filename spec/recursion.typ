#import "/book.typ": book-page, et, aside

#show: book-page("recursion.typ")


// Outline
#let binaryVM = raw("binaryVM")
#let fieldVM = raw("fieldVM")


#let functionSpace = $PP$
#let program = $bb(p)$
#let inputSpace = $II$
#let input = $bb(i)$
#let instanceSpace = $XX$
#let instance = $bb(x)$
#let instance2 = $bb(y)$
#let witnessSpace = $WW$
#let witness = $bb(w)$
#let proofSpace = $bb(Pi)$
#let proof = $bb(pi)$
#let prove = $italic("p")$
#let verify = $italic("v")$
#let commit = $italic("c")$
#let one = $bb(1)$
#let zero = $bb(0)$
#let function = $cal(F)$
#let relation = $cal(R)$

#show math.equation.where(block: false): box

= Notation

Let $functionSpace := {function: inputSpace times witnessSpace mapsto BB}$ denote the set of functions mapping input-witness pairs $(input; witness) in inputSpace times witnessSpace$ to a boolean ${ zero, one } in BB$.
Let relation $relation subset.eq functionSpace times inputSpace =: instanceSpace$ denote the set of all succesfully terminating program instances, i.e., all function-input instances $(program, input) in functionSpace times inputSpace$ for which there exists a witness $witness in witnessSpace$ such that $program\(input; witness) = one$.

Let there furthermore exist _proving system_ $(prove, verify)$ with prover $prove in { function: instanceSpace times witnessSpace mapsto proofSpace }$ and verifier $verify in { function: instanceSpace times proofSpace mapsto BB}$ such that
$
forall (instance, witness) in relation times witnessSpace &: PP[verify\(instance, prove\(instance; witness)) = one | instance(witness) = one] = 1 \
forall instance in instanceSpace without relation,  forall proof in proofSpace &: PP[verify\(instance, proof) = one] < epsilon
$
with $epsilon$ negligibly small.
That is: any valid proof for a terminating program verifiers successfully, while the probability of any proof verifying a unsuccesfully-terminating program is negligible.

= Proof recursion
In our application, the prover wishes to convince the verifier that for some public program-input instance $instance = (program, input) in instanceSpace$ they know a private witness $witness in witnessSpace$ such that $program\(input; witness) = one$.
To this end, the prover uses $prove\(program, input; witness) = prove\(instance; witness)$ to generate proof $proof$ and sends this to the verifier.
They then use $verify(instance, proof)$ to check that the proof is valid, convincing them of the prover's claim.

When we observe that $verify in functionSpace$, we can now let the prover compute $prove\(verify, instance; prove\(instance; witness)) = proof'$ and send this proof for the verifier to $verify((verify, instance), proof')$, proving that they _know a proof attesting that $instance$ is in the relation_.
This concept, colloquially known as _proof recursion_, can be applied recursively.
This is often beneficial for _succint_ proving systems where proof size (and verification time) typically shrinks as the level of recursion increases.
The technique is mostly useful in settings where the extra time spent by the prover is outweighed by the time saved by the verifier(s), e.g., a computationally constrained verifier, or multiple verifiers.

== Proof traceability
Importantly, the final recursive proof should be _tied_ to both the original instance $instance$, as well as the entire stack of verifiers used along the way. 
Without this, the final verifier cannot verify that the received proof attests to the original claim.
We exemplify this in the following triple-nested example:
$
&prove\([verify, instance'']; prove\([verify, instance']; prove\([verify, instance]; prove\(instance; witness))) = proof'''\
&verify(instance''', proof''') in BB
$
which requires $instance''' = [verify, instance''] = [verify, [verify, instance']] = [verify, [verify, [verify, instance]]]$: the original instance, as well as the full stack of verification functions used during recursion.

It is undesirable for the instance to grow as the level of recursion increases.
To this end, one can construct the modified proving system $(prove', verify')$ such that
$
forall (instance, witness) in relation times witnessSpace &: PP[verify'\(commit\(instance), prove'\(commit\(instance); instance, witness)) = one | instance(witness) = one] = 1 \
forall instance in instanceSpace without relation,  forall proof in proofSpace &: PP[verify'\(commit\(instance), proof) = one] < epsilon
$
where $commit\(dot)$ denotes a constant-size cryptographic commitment of the provided value.
Importantly, this allows the instance to be constant size.
It does, however, trade instance size for computation time, as the verifier now has to (pre)compute the $n$th nested commitment to verify an $n$-deep recursion.

#et(
  "design a setup such that the validators does not have to track the entire verification stack, i.e., if a verifier accepts the top level proof for the instance, that must mean that the instance's program was either 1) itself, or 2) the guest (= base level). The tricky thing here is that you'd have to somehow bypass the validator code containing the hash-root of a commitment of itself (which you should not be able to do with cryptographic hash functions)"
)

= Operation-specific verification
#let scratch = $bb(s)$

To verify a proof, several checks of different types need to be performed.
For the purposes of this discussion, we distinguish two types of checks: 
those that rely primarily on binary arithmetic, and those relying on field arithmetic.

Emulating either type of arithmetic on a VM designed for the other, typically incurs significant performance overhead.
Yet, recursive proving heavily relies on both types.
With the aim of bypassing a performance penalty, we introduce a field arithmetic-oriented mini-VM (henceforth referred to as the _field-VM_),
which will act as a _co-processor_ to the established specified binary arithmetic-oriented VM (henceforth referred to as _binary-VM_).
Since both VMs are proven using the same proof system, a unified proof can be produced for the parallel execution of both VMs.

The introduction of this split requires the verification algorithm be split as well.
In the process of verifying proofs of the current proof system (`DEEP-FRI` + `LogUp`), results of binary arithmetic are used to verify field arithmetical constraints --- e.g., field challenges extracted from binary hash outputs --- and vice-versa --- e.g., hashing merkle leafs containing field elements during FRI-query proof verification.
This implies that some form of communication between both VMs is required.

This architecture solves this by introducing a prover-hinted _communication record_ accessible to both VMs.
In practice, this record will primarily contain values being reinterpreted --- from $FF$ to $ZZ_(2^N)$ and vice-versa --- during verification.
The two halves of the split verification algorithm should be designed to verify the record: for each value on the record, one of the VMs _verifies_ the value to be correct, while the other _assumes_ the value to be correct and resumes the verification algorithm under this assumption.

To ensure this verification happens correctly, both verification algorithms must align on the interpretation of each value on the proof-record pair.
To this end, the dimensions of the record must be determined at _algorithm design-time_ and parametrized in terms of the proof only.
Then, both verification algorithm halves should be designed to agree on the interpretation of the proof and communication record, irrespective of the provided proof.

Note that, as part of check correctness of a proof-of-split-verification, the verifier must now verify that the VMs were given 1) the same proof and communication record, and 2) a synchronized algorithm pair; otherwise the prover could cheat.

#aside("Coupling")[
  As observed, both verification halves must be synchronized to correctly verify a proof.
  This implies that some coupling between both halves must exist.
  This design utilizes little coupling in the VM design, instead forcing the guest programs to solve synchronization, as a result introducing the coupling there. 
  
  This no-coupling VM design permits one of the two halves to transition to a different proof system (e.g., moving to Flock #footnote(link("https://eprint.iacr.org/2026/1329", "Flock: Fast Proving for Batch Boolean Computations. src: https://eprint.iacr.org/2026/1329")) to accelerate hash-verification) while incurring as little design overhead as possible.
]

// #let bool = $#`B`$
// #let field = $#`F`$
// #let equal = $#`E`$
// #let consistency = $#`C`$


// - Let $verify_bool || verify_field  := verify$ denote the decomposed verifier.
// $
//   prove\([verify_bool || verify_field, instance]; [proof, scratch]) = proof'
// $
// $
// v'(instance, proof) := verify([verify_bool || verify_field, instance], proof)
// $
// $
//   verify\([verify_bool || verify_field, instance], proof')\
//   // &=verify'\(instance, proof')\
//   &=verify\([verify_bool, instance], proof') times verify\([verify_field, instance], proof')\
// $
// $
//   &prove\([verify', [verify_bool || verify_field, instance]]; [proof', scratch'])\
//   &=prove\([verify_bool || verify_field || verify_consistency, [verify, instance]]; proof')\
//   // &=prove\([verify_bool\([verify, instance], dot) times verify_field\([verify, instance], dot); proof')\
//   &=[
//     prove\([verify_bool, [verify, instance]]; proof'),
//     prove\([verify_field, [verify, instance]]; proof'),
//     prove\([verify_consistency, [verify, instance]]; proof')
//   ]\
//   &= [proof'_bool, proof'_field]\
//   &= proof''\
//   &\ \
//   &prove\([verify, [verify, instance]]; proof')\
//   &=prove\([verify_bool || verify_field, [verify, instance]]; proof')\
//   &=[
//     prove\([verify_bool, [verify, instance]]; proof'),
//     prove\([verify_field, [verify, instance]]; proof')
//   ]\
//   &= [proof'_bool, proof'_field]\
//   &= proof''\
//   &\ \
//   &verify\([verify, [verify, instance]], proof'')\
//   &=verify\([verify_bool || verify_field, [verify, instance]], [proof'_bool, proof'_field])\
//   &=verify\([verify_bool, instance], proof'_bool) times verify\([verify_field, instance], proof'_field)\
// $
// ---
// - $prove\((verify_bool, instance); proof) -> proof_bool$
// - $prove\((verify_field, instance); proof) -> proof_field$
// - $verify\(((verify_bool, instance),(verify_field, instance)), (proof_bool, proof_field)) $
// ---
// $
// &verify'\((verify_bool, verify_field, instance), (proof_bool, proof_field)) \
// &= verify((verify_bool, instance), proof_bool) times verify\((verify_field, instance), proof_field)
// &\ \
// &overline(prove)\((verify'_bool, verify'_field, (verify_bool, verify_field, instance)); (proof_bool, proof_field))\
// &= (
//   prove\((verify'_bool, (verify_bool, verify_field, instance)); (proof_bool, proof_field)),
//   prove\((verify'_field, (verify_bool, verify_field, instance)); (proof_bool, proof_field))
//   )\
// &= (proof^1_bool, proof^1_field)
// &\ \
// &verify'\((verify'_bool, verify'_field, (verify_bool, verify_field, instance)), (proof'_bool, proof'_field))
// $
// - $prove\((verify'_field, (verify_bool, verify_field, instance)); (proof_bool, proof_field)) -> proof'_field$
// - $verify'\((verify'_bool, verify'_field, (verify_bool, verify_field, instance)), (proof'_bool, proof'_field))$
// ---
// - $prove\((verify'_bool, (verify'_bool, verify'_field, (verify_bool, verify_field, instance))); (proof'_bool, proof'_field)) -> proof''_bool$
// - $prove\((verify'_field, (verify'_bool, verify'_field, (verify_bool, verify_field, instance))); (proof'_bool, proof'_field)) -> proof''_field$
// - $verify\((verify'_bool, verify'_field, (verify'_bool, verify'_field, (verify'_bool, verify'_field, (verify_bool, verify_field, instance)))), (proof''_bool, proof''_field))$
// ---
// - $prove\((verify_bool, ((verify_bool, instance),(verify_field, instance))); (proof_bool, proof_field)) -> proof_bool'$
// - $prove\((verify_field, ((verify_bool, instance),(verify_field, instance))); (proof_bool, proof_field)) -> proof_field'$
// - $verify\(((verify_bool, ((verify_bool, instance),(verify_field, instance))), (verify_field, ((verify_bool, instance),(verify_field, instance)))), (proof_0', proof_1'))$


// - typically, verification algorithms reinterpret data based on the field.
// - expand proof to include prover-provided "scratch space", 
// - commit to this "expanded proof"
// - have both VMs use the same expanded proof to 
//   - verify programs must be tuned such that all values in the scratch space are
//     - checked by one of the two VMs and
//     - leveraged by other VM to speed up verification.
// - 

// - specific verify programs.



= Theory applied
Applying these observations and design requirements to this VM, we present the following design

- separate field-VM (@field-VM) with its own `DECODE` table (@field-decode).

== Split Verification Algorithm(s)

=== Verification of guest program proof
#let FRI = raw("FRI")
#let DEEP = raw("DEEP")
#let LogUp = raw("LogUp")
#let challenges = $bb(C)$
#let table_commitments = $cal(C)_cal(T)$
#let logup_commitments = $cal(C)_cal(L)$
#let DEEP_commitments = $cal(C)_cal(D)$
#let DEEP_openings = $cal(O)_cal(D)$
#let FRI_folding_commitments = $cal(C)_cal(F)$
#let FRI_query_openings = $cal(O)_cal(F)$
#let proof = $bb(pi)$
#let expanded_proof = $proof^*$
#let fs = $#`FiatShamir`$

Proof contents:
- #table_commitments: the commitments to all AIR-tables, 
- #logup_commitments: the commitments to the #LogUp columns, 
- #DEEP_commitments: the #DEEP commitments, 
- #DEEP_openings: the #DEEP openings, 
- #FRI_folding_commitments: the #FRI folding commitments, and 
- #FRI_query_openings: the #FRI query openings.

#figure(image("figures/DEEP-FRI_verification.svg", height: 75%))

On communcation record:
- all the challenges: lincomb, segment, DEEP coordinate, LogUp, folding & query

Native verification steps:
- binaryVM:
  - [B] Derive lincomb challenges from table commitments + public input
  - [B] Derive segment challenges from quotient commitments + table commitments + public input
  - [B] Derive DEEP point from segment + quotient + table commitments + public input
  - [B] Derive LogUp challenges from table commitments + public input
  - [B] verify LogUp opening proofs
  - [B] derive folding challenges from (everything before)
  - [B] derive FRI-query challenges
  - [B] verify query proofs
- fieldVM:
  - [F] verify opened LogUp sums
  - [F] verify low-degreeness of FRI output
  - [F] verify query opening validity.
  - [F] verify DEEP quotient/segmenting using DEEP-point

== Verification of verification-proof
TODO

// = L0 proof
// Let $proof\(g,x) := (#table_commitments, #DEEP_commitments, #DEEP_openings, #FRI_folding_commitments, #FRI_query_openings)$ denote a proof produced by the prover for program $g$ on public input $x$, with 
// - #table_commitments the commitments to all AIR-tables, 
// - #DEEP_commitments the #DEEP commitments, 
// - #DEEP_openings the #DEEP openings, 
// - #FRI_folding_commitments the #FRI folding commitments, and 
// - #FRI_query_openings the #FRI query openings.


// = Verifying an L0 proof
// Let $#fs\(proof) -> challenges$ denote the deterministic map producing the challenges corresponding to a given proof.
// We construct an _expanded proof_ $#expanded_proof := (proof, #fs\(proof)) = (proof, #`prog_comm`, challenges)$ containing the original proof, a commitment to the original guest program (including public parameters), and the challenges required for verification.

// The program commitment #`prog_comm` can be a commitment to the public information of a specific proof, e.g., the hash of the commitments to the `DECODE` table(s) and all public `PAGE` tables.

// Next, let us define two verification programs:

// ```
// func verify_L0_binary(proof: Proof, prog_comm, challenges) -> Proof:
//     commit(prog_comm)                               # through printing to stdout
//     assert challenges == fiatShamir(proof)
//     assert verify_FRI_query_proofs(proof, challenges)

// func verify_L0_field(proof: Proof, _prog_comm, challenges) -> Proof:
//     assert verify_DEEP_openings(proof, challenges)
//     assert verify_FRI_folding(proof, challenges)
//     assert verify_FRI_output_is_low_degree(proof, challenges)
//     assert verify_LogUp_equals_zero(proof, challenges)

// func proof_L0_verification(prog_comm, proof: Proof) -> DoubleProof:
//     challenges = fiatShamir_risc5VM(proof)
//     input_commitment = commit((prog_comm, proof, challenges))    
//     proof0: Proof = risc5VM.prove(verify_L0_binary, input_commitment)
//     proof1: Proof = fieldVM.prove(verify_L0_field, input_commitment)
//     return (input_commitment, proof0, proof1)
// ```

// = Verifying an L1 proof

// ```
// func verify_L1_binary(proof: Proof, prog_comm, challenges) -> Proof:
//     commit(prog_comm)                               # through printing to stdout
//     assert challenges.c0 == fiatShamir_risc5VM(proof)
//     assert challenges.c1 == fiatShamir_fieldVM(proof)
//     assert verify_FRI_query_proofs(proof.p0, challenges.c0)
//     assert verify_FRI_query_proofs(proof.p1, challenges.c1)

// func verify_L1_field(proof, prog_comm, challenges) -> Proof:
//     assert verify_DEEP_openings(proof.p0, challenges.c0)
//     assert verify_DEEP_openings(proof.p1, challenges.c1)
//     assert verify_FRI_folding(proof.p0, challenges.c0)
//     assert verify_FRI_folding(proof.p1, challenges.c1)
//     assert verify_FRI_output_is_low_degree(proof.p0, challenges.c0)
//     assert verify_FRI_output_is_low_degree(proof.p1, challenges.c1)

//     # compute verifier contribution to the risc5VM's LogUp
//     vc = compute_commitment_contribution(challenges.c0, prog_comm)
//     assert verify_LogUp_equals_zero(proof.p0 + vc, challenges.c0)
//     assert verify_LogUp_equals_zero(proof.p1, challenges.c1)

// func proof_L1_verification(_prog_comm, proof: DoubleProof) -> DoubleProof:
//     c0 = fiatShamir_risc5VM((proof.input_comm, proof.p0))
//     c1 = fiatShamir_fieldVM((proof.input_comm, proof.p1))
//     challenges = (c0, c1)

//     input_commitment = commit((_prog_comm, proof, challenges))    
//     proof0: Proof = risc5VM.prove(verify_L1_binary, input_commitment)
//     proof1: Proof = fieldVM.prove(verify_L1_field, input_commitment)
//     return (input_commitment, proof0, proof1)
// ```





// = Recursion
// - proof system generates proof
// - proof is still quite large
// - rather than verify the proof itself, have the prover generate proof that the verification of the first proof succeeds, where this new proof is smaller than the first.
// - repeat until the desired proof size is reached
// - at the end, verify this "recursed" proof.
//   - this is commonly called "proof recursion"

// - one important aspect, is that the _recursed proof_ should be tied to the original, base proof.

// = Recursion components
// Three different configurations
// + prove_guest_program(guest_program) -> proof
// + prove_single_proof_verification(proof) -> double_proof
// + prove_double_proof_verification(double_proof) -> double_proof

// == Proving a guest program
// -> take guest program
// > generate proof

// contents of proof:
// - table commitments
// - DEEP commitments
// - DEEP openings
// - FRI folding commitments
// - FRI query openings (= node content + merkle path)

// == Proving the verification of a proof
// - expand proof to proof_with_challenges
// - commit to proof_with_challenges (e.g., as PAGES tables)
// - binaryVM runs program "verify_binary", with commitment as instance and (proof, challenges) as witness
//     - commits to the `commitment` by printing it to `stdout`
//     - verifies that:
//         - FRI query proofs are valid
//         - challenges are correctly derived from the proof transcript
// - fieldVM runs program "verify_field" with commitment as instance and (proof, challenges) as witness:
//     - verifies that:
//         - DEEP opening is valid
//         - FRI folding was done correctly.
// - generate two proofs, with a *shared commitment to the memory init/fini of the commitment*
//     -> (shared_commitment, proof_binary_vm, proof_field_vm)

// == Proving the verification of a double-proof
// - expand proofs to proof_with_challenges
// - commit to proof_with_challenges (e.g., as PAGES tables)
// - binaryVM runs program "verify_binary", with commitment as instance and (proof, challenges) as witness
//     - commits to the `commitment` by printing it to `stdout`
//     - verifies that:
//         - FRI query proofs are valid
//         - challenges are correctly derived from the proof transcript
// - fieldVM runs program "verify_field" with commitment as instance and (proof, challenges) as witness:
//     - verifies that:
//         - DEEP opening is valid
//         - FRI folding was done correctly.





// // Keys
// #let ProverKey = $KK$
// #let VerifKey = $VV$

// // Spaces
// #let instanceSpace = $XX$
// #let witnessSpace = $WW$
// #let outSpace = $BB$
// #let hashOutSpace = $HH$
// #let proofSpace = $Pi$

// Let $PP: XX times WW mapsto BB$ denote the collection of guest programs mapping a (public) _instance_ $x in XX$ and (private) witness

// - L0: proof $arrow.l$ prove(guest_program, input)
// - L1: (proof0, proof1) $arrow.l$ prove(verify_proof(proof))
// - L2+: (proof0, proof1) $arrow.l$ prove(verify_proof(proof))

// Level 0:
// - instance: program ELF, public inputs
// - witness: private inputs

// Proof L0:
// - setup:
//   - turn ELF, public inputs into DECODE table
//   - comm = commit to DECODE table
// - prover:
//   - proof $arrow.l$ prove(comm, witness)


// prover:
// - runs prove()

// Level 0:
// $
//   text("program space: ") 
//   && PP &:&& instanceSpace times witnessSpace &&mapsto outSpace\
//   text("preprocessor space: ") 
//   && PP PP &:&& PP times instanceSpace &&mapsto ProverKey times VerifKey\
//   text("L0 prover: ") 
//   && #`p` &in&& ProverKey times witnessSpace &&mapsto proofSpace\
//   text("L0 verifier: ") 
//   && #`v` &in&& VerifKey times proofSpace &&mapsto outSpace
// $

// Level 1:
// $
//   text("program: ") 
//   && p' &in&& [hashOutSpace] times [VerifKey times proofSpace times witnessSpace'] &&mapsto outSpace\
//   text("preprocessor: ") 
//   && #`pp`' &:&& p' times hashOutSpace &&mapsto ProverKey' times VerifKey'\
//   text("L1 prover: ") 
//   && #`p`' &in&& ProverKey' times [VerifKey times proofSpace times witnessSpace'] &&mapsto proofSpace' := proofSpace times proofSpace\
//   text("L1 verifier: ") 
//   && #`v`' &in&& VerifKey' times proofSpace' &&mapsto outSpace
// $

// Level 2 - $inf$:
// $
//   text("program: ") 
//   && p_2 &in&& [hashOutSpace] times [VerifKey' times proofSpace' times witnessSpace'] &&mapsto outSpace\
//   text("preprocessor: ") 
//   && #`pp`_2 &:&& p_2 times hashOutSpace &&mapsto ProverKey' times VerifKey'\
//   text("L2 prover: ") 
//   && #`p`_2 &in&& ProverKey' times [VerifKey' times proofSpace' times witnessSpace'] &&mapsto proofSpace'\
//   text("L2 verifier: ") 
//   && #`v`_2 &in&& VerifKey' times proofSpace' &&mapsto outSpace
// $

// s.t. $(#`h`, #`vk`, #`π`) mapsto #`H` (#`vk`) = #`h`  text("and") #`verify` (#`vk`, #`π`) = 1$


// $
//   #`program<`XX #`>` (WW) mapsto BB
// $
