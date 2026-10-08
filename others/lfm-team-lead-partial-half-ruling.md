# Team-lead ruling: partial-half appends (handoff §10 last item)

For keccak-emitter, before starting §2 item 1. Written 2026-07-29 ~21:05Z.

## The reframe that dissolves most of it

Append boundaries are NOT machine-visible. The production transcript's hasher
sees only the CONCATENATED byte stream between two finalize points (samples);
`append_bytes` boundaries have no representation in the digest input. All
lengths are shape-static in a straight-line program. Therefore the emitter's
correct unit of packing is the SEGMENT (sample-to-sample), not the append:
concatenate every absorbed byte rendering in the segment at emit time, then
chunk the whole segment into halves. "A partial half in the middle of a
segment that the NEXT APPEND must continue into" cannot occur — the emitter
already holds the next append's bytes when it packs.

Segment prefixes are safe by construction: each segment after the first
starts with the 32-byte reversed digest (a multiple of 4), and the reversed-
digest words already exist on the bus.

## What genuinely remains, and the ruling

The residual problem is CONSTANT/DYNAMIC MISALIGNMENT: a constant of length
≢ 0 (mod 4) — e.g. the 27-byte `LAMBDAVM_STARK_STATEMENT_V3` tag — shifts a
following DYNAMIC value (a root word, a count) so one half mixes constant
bytes with dynamic bytes. The `stream_half + pad_const` trick covers this
only when the dynamic side's overlapping bytes are known-zero; in general it
needs a byte-level splice of the dynamic value at a constant offset
(re-aligning u32 halves by s∈{1,2,3} bytes ⇒ byte extraction, BitDec-32 per
affected half or a byte-table route; constant volume, and it only occurs in
the STATEMENT-ABSORB leg — a few dozen halves per proof, not in FRI/Merkle
traffic).

RULING — the predecessor's recommendation, adopted with the reframe:
1. R1d NOW: `append` accepts whole halves only, loud assert, plus the test
   that the assert fires (pin the limitation, don't leave it latent). This is
   sufficient for the entire FRI-verifier scope: digests, field elements and
   u64 renderings are all 4-byte multiples.
2. Document IN THE EMITTER (doc comment): the segment-level concatenation
   argument above, so nobody reintroduces per-append packing; and that the
   statement-absorb leg will add a `splice_misaligned(constant_prefix_len,
   dynamic_halves)` helper (BitDec-based, constant offset, tiny volume) when
   that leg is built — an extension point, not a redesign.
3. Do NOT build the splice now. Scope discipline: no verifier leg needs it
   yet, and its design should be reviewed against the real statement stream.

If anything in §2 contradicts this ruling, the ruling wins; report the
contradiction.
