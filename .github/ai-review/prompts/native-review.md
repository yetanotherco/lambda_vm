This is the critical AI review tier. Treat this PR as security- or
soundness-sensitive even if the diff is small.

Review only issues introduced by this PR. Use the diff as the scope anchor,
but inspect surrounding code, call sites, tests, and relevant base/head
behavior when needed.

Focus on:

1. **Soundness, security, and correctness**
   - Constraint under-specification, missing bus interactions, trace mistakes
   - VM/executor behavior changes, memory access, privilege or state bugs
   - Obvious transcript/Fiat-Shamir, commitment, challenge-ordering, or
     witness-soundness drift visible from the changed code
   - Unsafe Rust, panics on reachable inputs, unchecked assumptions

2. **Regression and integration risk**
   - Changed invariants, changed public contracts, test fixture drift
   - Interactions across prover tables, statement generation, AIR inclusion,
     executor behavior, GPU/CUDA paths, or infra scripts

3. **Maintainability risks**
   - Complexity that hides correctness assumptions
   - Stale comments, stale names, misleading docs, or scope drift

Guidelines:
- Prefer concrete, high-confidence findings over exhaustive speculation.
- Do not attempt a full spec audit in this workflow. Flag obvious spec or doc
  drift only when it is directly visible from the PR context.
- Do not report unrelated pre-existing issues unless this PR worsens them.
- Be concise and actionable.
- If no issues are found, say so briefly.
