# Team-lead ruling: R1f(b) fixture route

For keccak-emitter, 2026-07-30. Answers the blocker in
`lfm-r1f-handoff.md` §3b. Supersedes any earlier phrasing.

## RULING: route 2 — the machine consumes BYTES. But for a better reason than
## the one you gave.

You argued route 2 on collision-avoidance and caching. Both true, both
secondary. The decisive argument is **fidelity**: bytes are how recursion
actually receives a proof in production.

The RV64 recursion guest does not get a `ContinuationProof`. It gets a blob in
private input and reads it zero-copy through rkyv (`StarkProofView`'s
`Owned | Archived` split exists precisely for that, and the continuation path
verifies in place from archived bytes). So an LFM arena filler whose input is a
byte blob is the direct analogue of the guest's reader, and any divergence
between them is a *meaningful* signal. An arena filler that consumed an
in-memory `ContinuationProof` would be testing a path production does not have.

That also disposes of route 1 on the merits, not just on collision risk: adding
a `pub(crate)` accessor would let `lfm` reach into a structure the real
recursion path never sees.

## Use the EXISTING blob, do not invent a format

Do not design a fixture encoding. The repo already produces exactly the bytes
the recursion guest consumes, and there is already a dump path for capturing
them to a file. Look for:

- the continuation guest-input encoder (grep for `encode_continuation_guest_input`
  or the encoder used by `prover/src/recursion.rs` to build the guest's private
  input — it embeds the supplied roots, which you will need);
- the dump test that writes such a blob to disk (grep `test_dump_recursion_input`
  and the `RECURSION_DUMP_PRESET` / `INNER_ELF` / `INNER_INPUT` / `EPOCH_LOG2`
  environment knobs; earlier campaign work used exactly this to produce fixed
  blobs for measurement).

If that machinery exists and is reachable, your fixture is a captured blob plus
a small checked-in note recording the knobs used to produce it. If it turns out
to be `#[ignore]`d, env-gated, or otherwise not directly usable, say so and
propose the smallest thing that reuses the same ENCODER rather than a new one —
the encoder is what must not drift, the harness around it is incidental.

Keep the blob small: a two-epoch continuation over an existing tiny test ELF
(`fibonacci`/`empty`), not ethrex. You need real structure, not real workload.
If the smallest honest blob is still large enough to be awkward in git, put it
under the scratchpad and have the test regenerate-or-load, with the regeneration
path exercised rather than the checked-in bytes.

## Consequence for slice (a)

The arena filler's input type is therefore `&[u8]` (or the archived view over
it), not `MultiProofView`. Its job is: read the archived blob, pull out the
roots and the openings for one query, and lay them out as arena words for the
emitter. Where the guest's reader and your filler disagree about layout, the
guest is right.

## Standing note

You have now stopped twice on decisions that were genuinely mine, and both times
the stop was correct. Do not let this ruling make you more reluctant to decide
things yourself — the standing decisions still say implement-and-flag when the
call is yours. This one was not.
