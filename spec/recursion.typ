#import "/book.typ": book-page, et, aside

#show: book-page("recursion.typ")


// Outline
#let binaryVM = raw("binaryVM")
#let fieldVM = raw("fieldVM")


#let functionSpace = $cal(F)$
#let verifierSpace = $cal(V)$
#let privateFunctionSpace = $hat(cal(F))$
#let program = $f$
#let inputSpace = $II$
#let input = $bb(i)$
#let instanceSpace = $XX$
#let instanceCommitmentSpace = $CC$
#let instance = $bb(x)$
#let instance2 = $bb(y)$
#let witnessSpace = $WW$
#let witness = $bb(w)$
#let proofSpace = $bb(Pi)$
#let proof = $bb(pi)$
#let prove = $italic("p")$
#let verify = $italic("v")$
#let commit(x) = $overline(#x)$
#let comm(x) = $commit(#x)$
#let one = $bb(1)$
#let zero = $bb(0)$
#let function = $bb(f)$
#let relation = $cal(R)$
#let iff = $arrow.double.l.r$
#let implies = $arrow.double.r$
#let prob = $PP$

#show math.equation.where(block: false): box

= Notation
Let $BB := { zero, one }$ denote the boolean set and let 
$functionSpace := {f: inputSpace times witnessSpace mapsto BB}$ denote 
the set of functions mapping the (public) input space $inputSpace$ and (private) 
witness space $witnessSpace$ to this set.
We use $instanceSpace := functionSpace times inputSpace = {instance: witnessSpace mapsto BB}$ 
to denote the set of functions with the public input "baked in"; 
elements in this set are henceforth referred to as _function instances_, or simply _instances_.
We then define $relation subset.eq instanceSpace$ 
as the set of all _solvable instances_, 
i.e., all instances $instance in instanceSpace$
for which there exists a witness $witness in witnessSpace$ such that 
$instance\(witness) = one$.
Lastly, we introduce the commitment function $c: instanceSpace mapsto instanceCommitmentSpace$.
To simplify notation, we use $commit(instance) = c(instance)$.

We now assume the existence of _proving system_ $(prove, verify)$ with 
prover $prove: instanceSpace times witnessSpace mapsto proofSpace$ and 
verifier $verify: instanceCommitmentSpace times proofSpace mapsto BB$ such that
$
forall (instance, witness) in relation times witnessSpace &: prob[verify\(commit(instance), prove\(instance; witness)) = one | instance(witness) = one] = 1 \
forall instance in instanceSpace without relation,  forall proof in proofSpace &: prob[verify\(commit(instance), proof) = one] < epsilon
$
with $epsilon$ negligibly small and $proofSpace$ the proof space.
That is: any valid proof for a solvable instance verifies successfully, 
while the probability of any proof verifying an unsolvable instance is negligible.

Translating this to the purposes of this VM, a prover wishes to convince the verifier 
that for some agreed upon program ($program in functionSpace$) and specified public input ($input in inputSpace$),
they know a private input ($witness in witnessSpace$) such that the program terminates successfully 
(i.e., $(program, input) in relation$).
To this end, the prover uses $prove\((program, input); witness) = prove\(instance; witness)$ 
to construct some proof $proof in proofSpace$ and sends this to the verifier.
They then use $verify(comm(instance), proof)$ to check that the proof is valid, 
convincing them of the prover's claim.

= Proof recursion
Now observe that the verifier $verify$ is itself a function in 
$verifierSpace := {hat(f): instanceSpace times proofSpace mapsto BB} subset.eq functionSpace$.
This means that we can use $prove$ to prove that the verification of a proof $proof$ for a given instance $instance$ succeeds:
$
  &prove\(verify(comm(instance), dot); proof) = proof', text("and")
  &verify(comm(verify(comm(instance), dot)), proof') = one.
$
This new proof $proof'$ thus attests to _the existence of a proof $proof$ that 
satisfies the verifier on the given instance $instance$_.

This concept, colloquially known as _proof recursion_, can be applied repeatedly.
The technique is specifically beneficial for _succint_ proving systems where proof size
typically shrinks (and verification time therefore reduces) as the level of recursion increases.
The technique is mostly useful in settings where the extra time spent by the prover
is outweighed by the time saved by the verifier(s), 
e.g., a computationally constrained verifier, or multiple verifiers.

= Resolving growing instance complexity
While recursive proving leads to a decrease in proof size, this is naively traded off
against an increase in instance complexity.
Looking at a depth-two recursive proof,
$
  &prove\(verify(comm(verify(comm(instance), dot)), dot); proof') = proof'', text("and")\
  &verify(comm(verify(comm(verify(comm(instance), dot)), dot)), proof'') = one.
$
we see that the verifier first the verifier first has to derive the commitment
$comm(verify(comm(verify(comm(instance), dot)), dot))$
from the given base instance $instance$ before verifying the proof.
This increase in verifier computation is undesirable and should be avoided.

A solution to this, is to leverage the following variation to the verification algorithm:
$
  verify': instanceCommitmentSpace^2 times {0, 1} times proofSpace: (c_0, c_1, b, proof) mapsto
  cases(
    verify(c_0, proof) &text("if") b=0,
    verify(c_1(c_0, c_1, dot), proof) &text("if") b=1
  )
$
where it is assumed that $comm(function(x_1, x_2, dot))$ can be easily
constructed from $comm(function), comm(x_1)$, and $comm(x_2)$.
By choosing $c_0 = commit(instance)$ and $c_1 = commit(verify')$, the prover can then prove
the base case by selecting $b=0$, and set $b=1$ during further recursion.
Then, when presented with depth-n proof $proof^((n))$ and base instance $instance$, 
the verifier executes
$
  verify'(commit(instance), commit(verify'), 1, proof^((n)))
  &= verify(verify'(commit(instance), commit(verify'), dot), proof^((n)))\
  &= verify(verify(verify'(commit(instance), commit(verify'), dot dot), dot), proof^((n)))\
  &= verify(verify(verify(dots.c(v(commit(instance), dot), dot), dots.c), dot), dot), proof^((n))).
$
In other words, we have constructed a verifier $verify'$ which can only verify 
the desired base case, or a proof it produced itself.
This means that with successful verification of the ultimate proof $proof^((n))$, 
it is also guaranteed that $verify'$ must have been used at every step in the proof recursion.
This solution moreover reduces the verifier overhead on parsing the instance to a minimum, 
as both $comm(instance)$ and $comm(verify')$ can typically be precomputed.

#aside([$comm(verify')$ absorption])[
Note that $commit(verify')$ must be provided to $verify'$ as a _parameter_;
absorbing it into $verify'$ would imply an object containing a cryptographic commitment of itself, 
which is theoretically impossible.
]

#et("illustrate that there comes a termination point, i.e., a proof cannot prove itself.")
#et("note shakiness of recursion")

= Split processing
#let record = $bb(r)$
In practice, we find that the set of operations utilized for verification differs vastly from
those typically performed by guest programs.
Specifically, verification primarily involves hashing and (extension) field arithmetic, 
where especially the second is absent in typical guest programs.

Emulating field arithmetic on the a binary arithmetic-oriented VM, typically
incurs significant computational overhead.
With the aim of avoiding this performance penalty, we introduce a field 
arithmetic-oriented mini-VM (henceforth referred to as the _field-VM_),
which will act as a _co-processor_ to the established _binary-VM_.
Since both VMs are proven using the same proof system, a unified proof can be 
produced for the parallel execution of both VMs.

The introduction of this split allows the verification algorithm to be split in two halves,
with each VM performing the computations it is fastest at.
The two halves cannot work independently, however.
In the process of verifying proofs of the current proof system (`DEEP-FRI` + `LogUp`),
results of binary arithmetic are used to verify field arithmetical constraints 
--- e.g., field challenges extracted from binary hash outputs --- 
and vice-versa --- e.g., hashing merkle leafs containing field elements during FRI-query proof verification.
This implies that some form of communication between both VMs is required.

This architecture enables the required communications by introducing a 
prover-hinted _communication record_ $record$ accessible to both VMs.
In practice, this record will primarily contain values being reinterpreted 
--- from $FF$ to $ZZ_(2^64)$ and vice-versa --- during verification.
The two halves of the split verification algorithm are adapted to leverage
the record: for each value on the record, one of the VMs _verifies_ the value to be correct, 
while the other _assumes_ its correctness and resumes verification under this assumption.

To ensure correct verification, both verification-algorithm halves must align
on the interpretation of each value on the proof-record pair.
To this end, the dimensions of the record must be determined at _verification algorithm design-time_ 
and parametrized in terms of the proof only.
Then, both verification algorithm halves can be given the same logic to interpret the record, 
effectively synchronizing their interpretation.

#aside("Coupling")[
  As observed, both verification halves must be synchronized to correctly verify a proof.
  This implies that some coupling between both halves must exist.
  This design utilizes little coupling in the VM design, instead forcing 
  the guest programs to solve synchronization, as a result introducing the coupling there. 
  
  This no-coupling VM design permits one of the two halves to transition to a 
  different proof system (e.g., moving to Flock 
  #footnote(link(
    "https://eprint.iacr.org/2026/1329", 
    "Flock: Fast Proving for Batch Boolean Computations. src: https://eprint.iacr.org/2026/1329"
  )) 
  to accelerate hash-verification) while incurring as little design overhead as possible.
]

In theory, any division of tasks between the two VMs would work.
Yet, it is expected that some division will be more performant than others.
Below, we provide a division that, in theory, is expected to achieve solid performance:

*Record $record$.*
The record contains all challenges the prover derived using Fiat-Shamir.

*Tasks for $verify'_b\(c_0, c_1, b, proof, record)$:*
+ assert that $b in {0, 1}$,
+ verify challenges on record $record$ according to Fiat-Shamir,
+ verify the various opening proofs;
    - if $b=0$: 
        verify binary-VM DECODE table (@decode) query opening against $c_0$
    - if $b=1$:
        verify binary-VM DECODE table (@decode) query opening against $c_(1,b)$ and
        verify field-VM DECODE table (@field-decode) query opening against $c_(1,f)$
+ `COMMIT` to $c_0$ and $c_1$ (see @commit)

*Tasks $verify'_f\(c_0, c_1, b, proof, record)$:*
+ verify LogUp openings sum to zero,
    - if $b=1$, use $c_0$ and $c_1$ to complete the `COMMIT` balance.
+ verify `DEEP` evaluation
+ verify `FRI` folding
+ verify `FRI` output low degreeness check.

*Prover.*
The prover performs the following steps:
$
   proof &arrow.l prove(instance, witness)\
   proof' &arrow.l prove(verify'_b || verify'_f, (commit(instance), [commit(verify'_b), commit(verify'_f)], 0, proof, record))\
   proof^((n)) &arrow.l prove(verify'_b || verify'_f, (commit(instance), [commit(verify'_b), commit(verify'_f)], 1, proof^((n-1)), record))
$

*Ultimate verification.*
$verify'(commit(instance), [commit(verify'_b), commit(verify'_f)], 1, proof^((n-1))) =? one$



// // #let bool = $#`B`$
// // #let field = $#`F`$
// // #let equal = $#`E`$
// // #let consistency = $#`C`$


// // - Let $verify_bool || verify_field  := verify$ denote the decomposed verifier.
// // $
// //   prove\([verify_bool || verify_field, instance]; [proof, scratch]) = proof'
// // $
// // $
// // v'(instance, proof) := verify([verify_bool || verify_field, instance], proof)
// // $
// // $
// //   verify\([verify_bool || verify_field, instance], proof')\
// //   // &=verify'\(instance, proof')\
// //   &=verify\([verify_bool, instance], proof') times verify\([verify_field, instance], proof')\
// // $
// // $
// //   &prove\([verify', [verify_bool || verify_field, instance]]; [proof', scratch'])\
// //   &=prove\([verify_bool || verify_field || verify_consistency, [verify, instance]]; proof')\
// //   // &=prove\([verify_bool\([verify, instance], dot) times verify_field\([verify, instance], dot); proof')\
// //   &=[
// //     prove\([verify_bool, [verify, instance]]; proof'),
// //     prove\([verify_field, [verify, instance]]; proof'),
// //     prove\([verify_consistency, [verify, instance]]; proof')
// //   ]\
// //   &= [proof'_bool, proof'_field]\
// //   &= proof''\
// //   &\ \
// //   &prove\([verify, [verify, instance]]; proof')\
// //   &=prove\([verify_bool || verify_field, [verify, instance]]; proof')\
// //   &=[
// //     prove\([verify_bool, [verify, instance]]; proof'),
// //     prove\([verify_field, [verify, instance]]; proof')
// //   ]\
// //   &= [proof'_bool, proof'_field]\
// //   &= proof''\
// //   &\ \
// //   &verify\([verify, [verify, instance]], proof'')\
// //   &=verify\([verify_bool || verify_field, [verify, instance]], [proof'_bool, proof'_field])\
// //   &=verify\([verify_bool, instance], proof'_bool) times verify\([verify_field, instance], proof'_field)\
// // $
// // ---
// // - $prove\((verify_bool, instance); proof) -> proof_bool$
// // - $prove\((verify_field, instance); proof) -> proof_field$
// // - $verify\(((verify_bool, instance),(verify_field, instance)), (proof_bool, proof_field)) $
// // ---
// // $
// // &verify'\((verify_bool, verify_field, instance), (proof_bool, proof_field)) \
// // &= verify((verify_bool, instance), proof_bool) times verify\((verify_field, instance), proof_field)
// // &\ \
// // &overline(prove)\((verify'_bool, verify'_field, (verify_bool, verify_field, instance)); (proof_bool, proof_field))\
// // &= (
// //   prove\((verify'_bool, (verify_bool, verify_field, instance)); (proof_bool, proof_field)),
// //   prove\((verify'_field, (verify_bool, verify_field, instance)); (proof_bool, proof_field))
// //   )\
// // &= (proof^1_bool, proof^1_field)
// // &\ \
// // &verify'\((verify'_bool, verify'_field, (verify_bool, verify_field, instance)), (proof'_bool, proof'_field))
// // $
// // - $prove\((verify'_field, (verify_bool, verify_field, instance)); (proof_bool, proof_field)) -> proof'_field$
// // - $verify'\((verify'_bool, verify'_field, (verify_bool, verify_field, instance)), (proof'_bool, proof'_field))$
// // ---
// // - $prove\((verify'_bool, (verify'_bool, verify'_field, (verify_bool, verify_field, instance))); (proof'_bool, proof'_field)) -> proof''_bool$
// // - $prove\((verify'_field, (verify'_bool, verify'_field, (verify_bool, verify_field, instance))); (proof'_bool, proof'_field)) -> proof''_field$
// // - $verify\((verify'_bool, verify'_field, (verify'_bool, verify'_field, (verify'_bool, verify'_field, (verify_bool, verify_field, instance)))), (proof''_bool, proof''_field))$
// // ---
// // - $prove\((verify_bool, ((verify_bool, instance),(verify_field, instance))); (proof_bool, proof_field)) -> proof_bool'$
// // - $prove\((verify_field, ((verify_bool, instance),(verify_field, instance))); (proof_bool, proof_field)) -> proof_field'$
// // - $verify\(((verify_bool, ((verify_bool, instance),(verify_field, instance))), (verify_field, ((verify_bool, instance),(verify_field, instance)))), (proof_0', proof_1'))$


// // - typically, verification algorithms reinterpret data based on the field.
// // - expand proof to include prover-provided "scratch space", 
// // - commit to this "expanded proof"
// // - have both VMs use the same expanded proof to 
// //   - verify programs must be tuned such that all values in the scratch space are
// //     - checked by one of the two VMs and
// //     - leveraged by other VM to speed up verification.
// // - 

// // - specific verify programs.



// = Theory applied
// Applying these observations and design requirements to this VM, we present the following design

// - separate field-VM (@field-VM) with its own `DECODE` table (@field-decode).

// == Split Verification Algorithm(s)

// === Verification of guest program proof
// #let FRI = raw("FRI")
// #let DEEP = raw("DEEP")
// #let LogUp = raw("LogUp")
// #let challenges = $bb(C)$
// #let guestProgramCommitment = $cal(C)_cal(G)$
// #let tableCommitments = $cal(C)_cal(T)$
// #let logupCommitments = $cal(C)_cal(L)$
// #let logupOpenings = $cal(O)_cal(L)$
// #let quotientCommitments = $cal(C)_cal(Q)$
// #let deepCommitments = $cal(C)_cal(D)$
// #let deepOpenings = $cal(O)_cal(D)$
// #let friFoldingCommitments = $cal(C)_cal(F)$
// #let friQueryOpenings = $cal(O)_cal(F)$
// #let proof = $bb(pi)$
// #let expanded_proof = $proof^*$
// #let fs = $#`FiatShamir`$

// #grid(
//   columns: (1fr, auto),
//   column-gutter: 1em,
//   [
//     Proof contents:
//     - #tableCommitments: the commitments to all AIR-tables, 
//     - #logupCommitments: the #LogUp commitments, 
//     - #logupOpenings: the #LogUp openings, 
//     - #deepCommitments: the #DEEP commitments, 
//     - #deepOpenings: the #DEEP openings, 
//     - #friFoldingCommitments: the #FRI folding commitments, and 
//     - #friQueryOpenings: the #FRI query openings.

//     *Native verification*:

//     input:
//         - proof
//         - public commitment (i.e., program + public input)
//     verification steps:
//       - derive lincomb challenges
//       - derive segment challenges
//       - derive DEEP point
//       - derive LogUp challenges
//       - verify LogUp opening proofs
//       - verify LogUp openings sum to zero,
//       - derive folding challenges
//       - verify low-degreeness of FRI output,
//       - derive FRI-query challenges,
//       - verify FRI-query proofs,
//       - verify folding was done correctly,
//       - verify DEEP quotient/segmenting using DEEP-point.

//     *Split verification steps*:
//     - communcation record:
//         - all the challenges: lincomb, segment, DEEP point, LogUp, folding & query
//     - binaryVM: verify
//       - recorded lincomb challenges,
//       - recorded segment challenges,
//       - recorded DEEP point,
//       - recorded LogUp challenges,
//       - LogUp opening proofs,
//       - recorded folding challenges,
//       - recorded FRI-query challenges, and
//       - FRI-query proofs.
//     - fieldVM: verify
//       - LogUp openings sum to zero,
//       - low-degreeness of FRI output,
//       - query opening are valid,
//       - DEEP quotient/segmenting using DEEP-point.

//   ],
//   figure(image("figures/DEEP-FRI_verification.svg", height: 90%))
// )

// == Transformation
// - COMMIT to any public input
//     -> this forces the verifier in the next-layer to include it in verifying this proof.
// - use 

// == Verification of verification-proof
// *Native verification steps*:
// - public input:
//     - commitment of guest program + public parameters
// - private input:
//     - guest program + public parameters
//     - proof that guest program in R
// - steps:
//     - commit to guest program: COMMIT to commitment.
//     - _all of the above_, where
//         - openings of guest program table are verified against that commitment



// #let get = $arrow.l$
// #let FS = $#`FiatShamir`$

// == Verify base
// Input:
// - instance:
//     - #guestProgramCommitment: commitment to guest program.
// - proof:
//     - #tableCommitments: commitments to all AIR-tables, 
//     - #logupCommitments: #LogUp commitments, 
//     - #logupOpenings: #LogUp openings, 
//     - #quotientCommitments: quotient commitments, 
//     - #deepCommitments: #DEEP commitments, 
//     - #deepOpenings: #DEEP openings, 
//     - #friFoldingCommitments: #FRI folding commitments, and 
//     - #friQueryOpenings: #FRI query openings.

// Steps:
// + derive linear combination challenges,
// + derive segment challenges,
// + derive DEEP point,
// + derive LogUp challenges,
// + verify LogUp opening proofs,
// + verify LogUp openings sum to zero,
// + derive folding challenges,
// + verify low-degreeness of FRI output,
// + derive FRI-query challenges,
// + verify FRI-query proofs,
// + verify folding was done correctly,
// + verify DEEP quotient/segmenting using DEEP-point.


// - what needs to be done to verify a base proof, (see verification)
// - what extra needs to be done to do this usiing the split verifier,
// - what extra needs to be done to _prove_ this verification.

// - what needs to be done to verify a recursive proof,
// - what extra needs to be done to do this using the split verifier,
// - what extra needs to be done to _prove_ this verification.

// = L0 proof
// Let $proof\(g,x) := (#tableCommitments, #DEEP_commitments, #DEEP_openings, #FRI_folding_commitments, #FRI_query_openings)$ denote a proof produced by the prover for program $g$ on public input $x$, with 
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
