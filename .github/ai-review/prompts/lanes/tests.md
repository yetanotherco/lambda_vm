Review this PR for missing or stale tests.

Focus on:

- changed behavior without a test
- edge cases that are likely to regress
- tests whose names, fixtures, or assertions no longer match the implementation
- prover, executor, trace, bus, and constraint changes that need targeted tests
- docs or comments that imply behavior not covered by tests

Do not ask for broad test rewrites. Prefer targeted tests tied to the changed
behavior.
