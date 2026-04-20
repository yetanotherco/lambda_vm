# Investigation: Uniform Degree-2 Constraints

## Summary

Reducing max constraint degree from 3 to 2 would eliminate the composition polynomial
decomposition (decompose_d2 + extend), saving 2 iFFT(N) + 2 FFT(2N) per table.

## Degree-3 Constraints Found

**39 total instances** across 14 constraint types, falling into 3 patterns:

### Pattern 1: Conditional IS_BIT — `selector * x * (1 - x) = 0`
- **24 instances**: CPU (12), BRANCH (4), LOAD (7), LT (1)
- Cannot reduce without adding committed columns (the carry/variable is virtual)
- Each reduction adds 1 column + 1 degree-2 constraint per instance
- **Cost**: 24 extra columns across 4 tables, 24 extra constraints

### Pattern 2: Conditional product — `selector * a * b = 0`
- **5 instances**: CPU (BranchCondConstraint, Arg1Upper, Arg2Upper, RvdUpper), SHIFT (10)
- Same reduction approach: commit `aux = selector * a`, enforce `aux * b` at degree 2
- **Cost**: ~15 extra columns across 3 tables, ~15 extra constraints

### Pattern 3: LogUp batching — `c * fp_a * fp_b` (degree 3)
- **Every LogUp table** (via LookupBatchedTermConstraint)
- Unbatching (1 interaction per aux column) reduces degree to 2
- **Cost**: aux columns increase from ceil(N/2) to N per table (~55 → ~110 total)

## Trade-off Analysis

### Savings from degree 2
- Eliminates decompose_d2 + extend: **saves 2 iFFT(N) + 2 FFT(2N) per table**
- For 12 tables: ~24 N-point FFT equivalents saved
- At N=2^19: ~24 * 10ms = ~240ms saved

### Cost of degree 2

**Option A: Only fix base constraints (Patterns 1+2), keep LogUp at degree 3**
- Doesn't achieve uniform degree 2 — LogUp still forces degree 3
- No decomposition savings
- Not viable as standalone change

**Option B: Also unbatch LogUp (all 3 patterns)**
- Extra aux columns: ~55 more (from ~55 to ~110)
- Each extra column requires: FFT for LDE, Merkle hashing, constraint evaluation
- Extra FFTs: ~55 * (iFFT(N) + FFT(2N)) = ~55 * 30ms = ~1650ms ADDED
- Extra Merkle hashing: ~55 * N * 0.5us = ~550ms ADDED
- Total cost: ~2200ms added vs ~240ms saved
- **Net: ~2000ms SLOWER**

**Option C: Add committed columns for Patterns 1+2 only**
- ~39 extra columns, ~39 extra constraints
- Doesn't help with LogUp (still degree 3)
- No FFT savings (LogUp keeps degree 3)
- **Net: strictly worse (more columns, same degree)**

## Conclusion

**Uniform degree 2 is NOT cost-effective.** The LogUp batching constraint
(LookupBatchedTermConstraint) is degree 3 by design, and unbatching it
adds far more columns (and thus FFT/Merkle cost) than the decomposition
FFTs it would save.

The correct optimization path for composition polynomial handling is
**MMCS + shared FRI** (commit at N points in a shared tree), not
constraint degree reduction.

## Alternative: Mixed-Degree Evaluation (stwo approach)

Instead of uniform degree 2, evaluate constraints at their NATURAL degree:
- Degree-2 constraints evaluated on the 2N LDE domain (existing)
- Degree-3 constraints evaluated on a 3N domain (or handled via the
  existing decomposition)
- Each group produces its own quotient polynomial part

This avoids penalizing degree-2 constraints with the degree-3 decomposition.
However, it requires significant changes to the constraint evaluator and
composition polynomial handling. Worth investigating as a separate effort.
