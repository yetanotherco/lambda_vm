Review this PR for soundness-sensitive issues visible from the changed code and
nearby context.

Focus on:

- under-constrained values, missing constraints, and incorrect selectors
- missing or incorrect bus interactions
- trace assignment mistakes and witness assumptions
- inconsistent prover/verifier behavior
- AIR inclusion or statement-generation drift
- obvious transcript, commitment, or challenge-ordering drift visible from the
  changed code

This is not a full spec audit. Report only issues with concrete evidence in the
diff or surrounding code.
