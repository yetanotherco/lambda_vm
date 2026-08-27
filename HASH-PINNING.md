# HASH PINNING — what this branch pins, and what it cannot pin yet

One of four sibling branches (`hash-blake3`, `hash-poseidon`, `hash-rpo`,
`hash-rpx`), all cut from `73ee2a64` so the comparison is like-for-like. That
commit already carries every hash tenant plus the generic `LFM_HASH` socket;
`blake3-full-mmcs` remains the campaign trunk and is not one of these four.

**This branch's hash: Poseidon-original — UNSHIPPABLE, priced reference only (621 cells/compression).**

## ⚠ There are TWO hash planes, and only one of them can be pinned today

| plane | what it is | where it lives | pinnable now? |
|---|---|---|---|
| **1 — commitment / Fiat–Shamir** | the hash the HOST commits proofs under, and therefore the hash the in-VM verifier must replay | `crypto/stark/src/config.rs`: `DefaultStarkHash`, `CommitmentHash` | **keccak and BLAKE3 ONLY** |
| **2 — the `LFM_HASH` socket** | the permutation the machine's own hash chip proves | `HasherKind` | **all five** |

✓ VERIFIED: `CommitmentHash` has exactly two variants, `Keccak256` and
`Blake3`, and `DefaultStarkHash = Blake3StarkHash`. **There is no RPO, RPX or
Poseidon `StarkHash` backend, so "committed under RPX" cannot currently be
expressed in the type system at all** — not selected wrongly, not
expressible.

## What that means per branch

- **`hash-blake3`** — plane 1 and plane 2 agree already; it is the status quo
  and its block numbers are the measured record (104.2 min / 358.2 GiB /
  36.9 MB).
- **`hash-rpo`, `hash-rpx`, `hash-poseidon`** — plane 2 is available; **plane 1
  is not**. A block produced on these branches today would be committed under
  **BLAKE3** regardless of the branch name.

⚠ **That is why this branch does not flip `HasherKind::default()`.** Doing so
was tried and measured on a scratch worktree: it drifts every registry
`program_id` (the hasher tag is folded into it) and fails **18 tests**, all
`program_id` drift, requiring a full regeneration of `LFM_REGISTRY` via
`compute_lfm_registry`. `registry.rs` governs that explicitly — *"a drift
failure is investigated, never re-blessed to silence the test"* — and
`compute_lfm_registry` adds *"changing it here is a re-blessing of the whole
table … a second hasher becomes additional rows, never a silent replacement"*.

**That cost buys nothing until plane 1 exists**, and it would leave a branch
that looks ready and is not. So it is deliberately not done, rather than done
and caveated.

## What the real pin becomes, and when

Once the production `crypto/stark` migration lands, pinning IS one small commit
per branch — a `DefaultStarkHash` type alias, the matching `HasherKind`
default, and a registry regeneration.

★ **And the migration is written ONCE, not four times.** ✓ VERIFIED by reading
all eight exhaustive `WrapHash` matches: every one keys on digest/absorb
**SHAPE**, not on a named hash. All three algebraic candidates are
**state 12 / rate 8 / capacity 4 / digest 4 felts = ONE cell** against BLAKE3's
two-cell 32-byte digest, so `WrapDigest` can depend on the shape and the
migration commits only to "algebraic family". The single per-member item —
`CommitmentHash` plus its `commitment_hash_tag` — is one enum arm and one tag
byte, additive, and moves no existing root.

## Comparison discipline

⚠ **The four branches are NOT four equally-grounded numbers.** BLAKE3 has a
measured block record; the other three have projections from measured chips and
a measured census. Every table must keep marking which cells are measured and
which are extrapolated.

Measured cells per compression, one instrument, no flags:
**RPX 325 · RPO 445 · Poseidon 621 · BLAKE3 4,946.**
Poseidon is **UNSHIPPABLE** (broken family, eprint 2026/306 and 2026/1692) and
appears only because the numbers were asked for.

Full working: `thoughts/shared/block-compression/RPO-LANE.md` on `rpo-migration`.
