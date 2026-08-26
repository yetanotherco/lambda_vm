#import "/book.typ": book-page, et, aside

#show: book-page("recursion.typ")

// Spaces and instances
#let (functionSpace, function) = ($cal(F)$, $f$)
#let (inputSpace, input) = ($II$, $bb(i)$)
#let (instanceSpace, instance) = ($XX$, $bb(x)$)
#let (witnessSpace, witness) = ($WW$, $bb(w)$)
#let (proofSpace, proof) = ($bb(Pi)$, $bb(pi)$)

#let (commitmentSpace, commitment) = ($cal(C)$, $bb(c)$)
#let commit(x) = $overline(#x)$
#let comm(x) = $commit(#x)$

#let program = $f$
#let relation = $cal(R)$

#let verifierSpace = $cal(V)$
#let (prove, verify) = ($italic("p")$, $italic("v")$)

// Mathematical symbols
#let (zero, one) = ($bb(0)$, $bb(1)$)
#let iff = $arrow.double.l.r$
#let implies = $arrow.double.r$
#let prob = $PP$

#show math.equation.where(block: false): box

= Notation
Let $BB := { zero, one }$ denote the boolean set and let 
$functionSpace := {function: inputSpace times witnessSpace mapsto BB}$ denote 
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
Lastly, we introduce the instance commitment function $c: instanceSpace mapsto commitmentSpace$.
Note that this commitment scheme does not involve randomness; it is a determistic scheme.
Randomness is typically required to make a commitment _hiding_.
For the purposes of this discussion, we are not concerned with this property, 
as the function will only be used for committing to public information. 
To simplify notation, we henceforth use $commit(instance)$ to represent the commitment $c(instance)$ of $instance$.

We now assume the existence of _proof system_ $(prove, verify)$ with 
prover $prove: instanceSpace times witnessSpace mapsto proofSpace$ and 
verifier $verify: commitmentSpace times proofSpace mapsto BB$ such that
$
forall (instance, witness) in relation times witnessSpace 
&: prob[verify\(commit(instance), prove\(instance; witness)) = one | instance(witness) = one] = 1 \
forall instance in instanceSpace without relation,  forall proof in proofSpace 
&: prob[verify\(commit(instance), proof) = one] < epsilon
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
$verifierSpace := {hat(f): commitmentSpace times proofSpace mapsto BB} subset.eq functionSpace$.
This means that we can use $prove$ to prove that the verification of a proof $proof$ 
for a given instance $instance$ succeeds:
$
  &prove\(verify(comm(instance), dot); proof) = proof', text("and")
  &verify(comm(verify(comm(instance), dot)), proof') = one.
$
This new proof $proof'$ thus attests to _the existence of a proof $proof$ that 
satisfies the verifier on the given instance $instance$_.

This concept, colloquially known as _proof recursion_, can be applied repeatedly.
The technique is specifically beneficial for _succinct_ proving systems where proof size
typically shrinks (and verification time reduces) as the level of recursion increases.
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
  verify': commitmentSpace^2 times {0, 1} times proofSpace: (commitment_0, commitment_1, b, proof) mapsto
  cases(
    verify(commitment_0, proof) &text("if") b=0,
    verify(commitment_1(commitment_0, commitment_1, dot), proof) &text("if") b=1
  )
$
where it is assumed that $comm(function(x_1, x_2, dot))$ can be easily
constructed from $comm(function), comm(x_1)$, and $comm(x_2)$.
By choosing $commitment_0 = commit(instance)$ and $commitment_1 = commit(verify')$, the prover can then prove
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

Our architecture enables the required communications by introducing a 
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

*Tasks for $verify'_b\(commitment_0, commitment_1, b, proof, record)$:*
+ assert that $b in {0, 1}$,
+ verify challenges on record $record$ according to Fiat-Shamir,
+ verify the various opening proofs;
    - if $b=0$: 
        verify binary-VM DECODE table (@decode) query opening against $commitment_0$
    - if $b=1$:
        verify binary-VM DECODE table (@decode) query opening against $commitment_(1,b)$ and
        verify field-VM DECODE table (@field-decode) query opening against $commitment_(1,f)$
+ `COMMIT` to $commitment_0$ and $commitment_1$ (see @commit)

*Tasks $verify'_f\(commitment_0, commitment_1, b, proof, record)$:*
+ verify LogUp openings sum to zero,
    - if $b=1$, use $commitment_0$ and $commitment_1$ to complete the `COMMIT` balance.
+ verify `DEEP` evaluation
+ verify `FRI` folding
+ verify `FRI` output low degreeness check.

*Prover.*
The prover performs the following steps:
$
   proof &arrow.l prove(instance, witness)\
   proof' &arrow.l prove(verify'_b || verify'_f, (commit(instance), [commit(verify'_b), commit(verify'_f)], 0, proof, record))\
   proof^((i)) &arrow.l prove(verify'_b || verify'_f, (commit(instance), [commit(verify'_b), commit(verify'_f)], 1, proof^((i-1)), record))
$

*Final verification.*
$verify'(commit(instance), [commit(verify'_b), commit(verify'_f)], 1, proof^((n))) =? one$
