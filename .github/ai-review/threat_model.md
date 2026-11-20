# yetanotherco/lambda_vm

RISC-V zkVM: execution trace generation, a STARK prover and verifier, recursion and
continuations, and formal models (Lean / Clean) under `formal/`.

Priorities, soundness above all:
- Underconstrained columns or constraints: a value the prover can choose freely that the
  constraints should pin down.
- Missing range checks (limbs, bytes, carries, multiplicities).
- LogUp / bus errors: wrong multiplicities, unbalanced sends and receives, a bus term that a
  malicious prover can cancel.
- Integrity of preprocessed and committed tables (a table the verifier trusts but does not fix).
- Padding rows that have side effects (bus interactions, memory writes) or skip constraints.
- The verifier accepting an invalid proof; missing checks on proof shape or public inputs.
- Continuation / recursion linking errors (state handed from one segment or layer to the next
  without being bound).
- Panics or unbounded work in the verifier on a malformed proof, including allocations or
  recursion sized by proof-supplied counts, lengths or nesting.
- Formal models that diverge from the implementation, or prove a weaker statement than claimed.
