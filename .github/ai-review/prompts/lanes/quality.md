Review this PR for code-health issues introduced by the changed code:
simplification, duplication, naming, and test coverage. Report real, actionable
improvements with concrete file:line references — no low-signal churn.

Focus on:

- simplification: unnecessarily complex or clever code that could be clearer;
  avoidable abstractions and indirection introduced by the change
- duplication: logic repeated by this change that should be shared
- naming and comments: names, comments, or doc comments that no longer match the
  behavior or scope after this change; stale docs left behind
- tests: changed behavior with no test; edge cases likely to regress; tests
  whose names, fixtures, or assertions no longer match the implementation

Useful cosmetic rewrites are welcome when they make the changed code, names,
comments, or docs easier to understand. Do not request broad rewrites, churn, or
premature abstractions.
