This is the standard AI review tier. Review this PR seriously and report
concrete issues that should be addressed before merge.

Review only issues introduced by this PR. Use the diff as the scope anchor.
Do not perform a full spec audit and do not report unrelated pre-existing issues.

Focus on:

1. **Correctness and regressions**
   - Logic errors, edge cases, changed invariants, incorrect error handling
   - VM, prover, memory, bus, trace, and constraint behavior affected by the diff

2. **Tests and observability**
   - Missing tests for new behavior or fixed edge cases
   - Tests whose names/assertions no longer match the behavior

3. **Simplicity and maintainability**
   - Unnecessary complexity, duplicated logic, avoidable abstractions
   - Stale comments, stale names, misleading doc comments, or scope drift

Guidelines:
- Prefer fewer, higher-confidence findings.
- Do not suggest micro-optimizations or cosmetic rewrites.
- Be concise and actionable.
- Include concrete file and line references when possible.
- If no issues are found, say so briefly.
