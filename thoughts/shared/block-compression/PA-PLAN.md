# PA-PLAN — the RV64 prover commits with BLAKE3

**Scoping record.** Read-only pass; no builds run, no edits made.
**Ground:** worktree `/Users/maurofab/workspace/lambda_vm-blake3-impl`, branch
`blake3-real-hash` @ `bad2d97d`. **Date:** 2026-08-12.

**What P-a is:** move the RV64 STARK prover's commitment hash from keccak256 to
BLAKE3 across all four domains together — Merkle trees, FRI-layer trees,
Fiat–Shamir transcript, grinding PoW — because the LFM wrap re-derives the inner
proof's transcript and every domain must be hosted-recomputable.

**Decisions already taken (Mauro, 2026-08-12), folded in throughout:**
1. **6-round BLAKE3 is the target** ("to see if this works"). 7r stays buildable
   via the existing feature; the primary host implementation is the in-repo
   reduced-round compression, and the `blake3` crate stays dev-only as the 7r
   compression anchor. See §1.5, and §1.6 for the structural consequence.
2. **The CUDA kernels are a pre-authorized parallel workstream**, not a tail
   stage. Kernel list, oracle and start condition are in §6.1; it appears as
   track **G** in the stage table.

**The one decision still owed** is §1.6: bare cv-chain vs standard chunk tree.
It blocks track G's chaining loop and Stage 5's emitter, so it wants answering
before either commits.

Claims are ✓ VERIFIED (read the code, cited) / ? INFERRED / ✗ UNVERIFIED.
Everything marked ✓ below was read in this pass, not inherited.

> **Provenance note.** Four delegated sweeps (host backends; transcript +
> grinding; guest/fixture/CI blast radius; GPU + emitters + D0 collision) were
> launched and none returned before this was written — the same failure
> D0-DESIGN.md records. Every load-bearing claim here was therefore read
> directly by the author, and inherited citations are marked as such and were
> re-checked (two were stale — see §2.1 and §5). §8 lists what is genuinely
> still open.

---

## 0. Verdict — read this before scheduling anything

**The prover-side switch is M. The thing that makes it worth doing is L, it is a
change to the MACHINE, not to the prover, and PLAN.md currently prices the whole
item off the prover half.**

PLAN.md's rationale for putting P-a first says step 2's `StarkHash`
parameterization "makes it a second config instance rather than surgery"
(`PLAN.md:97-98`). That is true of `crypto/stark` and false of the payoff. Three
findings, in order of how much they move the schedule:

1. **The 4.06× requires a chip the machine does not have.** The hash matrix
   priced hosting at 4,946 base-equivalent cells per compression at **rate 8**
   (64-byte message block). That is `blake3_chip`/`LFM_BLAKE3` — the *general*
   BLAKE3 compression, taking `h`, all 16 message words, counter, `block_len`
   and flags from columns (`blake3_chip.rs:504-525`). ✓ VERIFIED that chip is
   **not a machine chip**: `LfmColumnGroups` has ten groups and none is blake3
   (`compiler.rs:112-123`); `LFM_BLAKE3` occurs only inside `blake3_probe.rs`
   (`:141 .with_name("LFM_BLAKE3")`), a standalone measurement instrument.
   The machine's live blake3 is the `LFM_HASH` **socket arm**, which pins
   `h = IV`, pins counter/`block_len`/flags, and uses 8 of 16 message words
   (`blake3_socket.rs:740-757`, `:725-732`). Hosting a wide-leaf absorption on
   the socket runs at an amortized **rate 4**, not 8
   (`epoch_verify.rs:428-456`: `LFM_HASH_RATE_FELTS = 4`, "**this is 4.25×
   worse than keccak's 17**", `:437`). Promoting `LFM_BLAKE3` to a registered,
   program-callable group — layout, preprocessed prefix, eDSL emitters,
   registry rows, admission-validator coverage — is the campaign's real P-a
   cost and it is soundness-bearing.

2. **The socket alternative is probably disqualified on digest width, not on
   cost.** The socket truncates to four output words = one cell = **128 bits**
   (`blake3_socket.rs:268-271`, `OUT_WINDOW = HASH_DIGEST_FELTS`;
   `hash.rs:23`), against the RV64 proof's current `Commitment = [u8;32]`
   (`config.rs:18-19`) and `KeccakDigest = [Cell;2]` in the guest
   (`edsl`, used at `sub_proof.rs:228`, `fri.rs:314`). A 128-bit Merkle node is
   a 64-bit collision bound. That is a decision for Mauro, but it is a
   *security* decision, and it should not be taken as a side effect of picking
   a cheaper hosting route.

3. **Under `cuda`, a blake3 `StarkHash` instance cannot be written at all.**
   ✓ VERIFIED `config.rs:116-122`: with `feature = "cuda"`, `StarkHash::Batched`
   additionally requires `KeccakTreeBackend`. The step-0 H3 guard landed and
   bound seven GPU tree entries to that marker (`gpu_lde.rs:701, 758, 802, 884,
   1120, 1171, 1573`). This is the guard working exactly as designed — and it
   means the GPU regression window is not a runtime fallback but a compile-time
   fork that P-a must decide explicitly.

**Recommended shape.** Byte-oriented BLAKE3 **at 6 rounds** (decided — §1.5),
256-bit digests, one family across all four domains, hosted by a promoted
`LFM_BLAKE3`. It keeps the existing leaf byte encoding untouched, keeps
`Commitment = [u8;32]` and the rkyv wire format byte-identical, and is the only
option consistent with the 4.06× the campaign is planning against.

**One question needs answering before Stage 1 commits an API** (§1.6): at 6
rounds nothing in the world can recompute our hashes anyway, so standard
BLAKE3's chunk tree — 1024-byte chunks, per-chunk counter, flag schedule — buys
no interop and costs ~6% extra compressions plus a state machine in all nine
CUDA kernels *and* in the wrap emitter. Recommend a bare cv-chain over 64-byte
blocks instead, same construction at both round counts. This is the one decision
that blocks other people's work.

**Effort, honestly:** whole item **L** (multi-week). Stages 1–3 (the prover) are
M and land behind a config with keccak still default. Stage 5 (chip promotion +
emitter switch) is L and is the center of mass. Stage 4 (guest leg) is L and is
gated on merging an unmerged branch. The CUDA kernels (track G) are M and run in
parallel from now.

---

## 1. Host backends

### 1.1 What exists

✓ VERIFIED the `blake3` crate is **dev-only, in `prover` alone**:
`prover/Cargo.toml:48` — `blake3 = { version = "1.8.5", default-features =
false, features = ["std","pure"] }`, sitting after `[dev-dependencies]`
(`:38`), with `:44-45` stating the intent: "The external anchor for
`lfm::blake3` at 7 rounds and for the `LFM_HASH` BLAKE3 socket … Test-only on
purpose." `crypto/crypto` and `crypto/stark` have **no** blake3 dependency
(grep over all `Cargo.toml`: the only two hits are `prover/Cargo.toml:22` and
`:48`).

✓ VERIFIED the round-generic host implementation is `prover/src/lfm/blake3.rs`:

| item | location | shape |
|---|---|---|
| `blake3_compress_rounds(h, m, t, block_len, flags, rounds)` | `blake3.rs:125-148` | fully general compression, `rounds` a runtime argument, `u32` words |
| `BLAKE3_STANDARD_ROUNDS = 7` / `BLAKE3_SIX_ROUNDS = 6` | `blake3.rs:59, 63` | |
| `BLAKE3_ROUNDS` | `blake3.rs:83, 85` | `7` unless `feature = "blake3-6round"`, then `6` |
| `BLAKE3_IV`, `BLAKE3_MSG_PERMUTATION` | `blake3.rs:46, 52` | |
| `CANONICAL_VECTORS` + `CANONICAL_OUT_7ROUND` | `blake3.rs:198-462` | 10 KAT vectors across `block_len` 18–64, both round counts |

✓ VERIFIED `blake3-6round` is declared **only** at `prover/Cargo.toml:22`
(`blake3-6round = []`), enabled by nothing, off by default. The house already
treats a split round count as a shipping hazard: `blake3_socket.rs:215` asserts
`SOCKET_ROUNDS == BLAKE3_ROUNDS` with the comment that a second `cfg` pair here
"is a silent pricing lie: the probe would measure one hash and the machine would
use another."

There is **no byte-oriented `&[u8] -> [u8;32]` blake3 on the host** outside the
dev-only crate — `blake3.rs` stops at the compression function, and
`blake3_socket.rs` is cell-oriented and fixed at a 36-byte one-block message
(`BLOCK_LEN_LFMC = 36`, `:262`; `FLAGS_LFMC = 0x0B`, `:256`; `COUNTER_LFMC = 0`,
`:265`).

### 1.2 The crate-layering problem, and the fix

The Merkle backends live in `crypto/crypto/src/merkle_tree/backends/`; the
round-generic compression lives in `prover`, which depends on `crypto`. A
backend in `crypto` therefore **cannot** call `prover`'s blake3.

**With 6 rounds decided, there is only one way out.** The `blake3` crate is
7-round only, so it cannot implement the pipeline's primary arm at all:

- **(a) `blake3` as a real dependency of `crypto/crypto`.** ✗ **Ruled out by the
  round-count decision** — it has no 6-round mode. It stays where it is
  (`prover`, dev-only) as the **compression-level KAT anchor for the 7r arm**,
  which is exactly the role `prover/Cargo.toml:44-45` already assigns it.
- **(b) Sink the compression core down. REQUIRED.** Move
  `prover/src/lfm/blake3.rs`'s `blake3_compress_rounds` + `BLAKE3_IV` +
  `BLAKE3_MSG_PERMUTATION` + `CANONICAL_VECTORS`/`CANONICAL_OUT_7ROUND` into
  `crypto/crypto` (`hash/blake3/`), re-export upward so `lfm` keeps its current
  API, and move the `blake3-6round` feature with it. ✓ Safe by construction: the
  LFM chip, the socket and the new backend then share **one** compression
  function — which is what `blake3_socket.rs:203-215` says the tree already
  depends on, and the only way the CUDA kernels (§6.1) and the wrap emitter can
  be checked against the same reference.

✓ VERIFIED the 6-round implementation the backend will call already exists and
is the one the chip's trace filler uses. `blake3_compress_rounds(h, m, t,
block_len, flags, rounds)` (`blake3.rs:125-148`) takes `rounds` as a **runtime
argument**; `blake3_compress_6round` (`:108-115`) is the fixed-6 wrapper. The
chip fills its trace through the value interpretation of the same dataflow —
`ValueFlow`'s `input_h`/`input_v12`/`add3` at `blake3_chip.rs:650-665` read
`self.h[i]` / `self.v12[j]` / `self.m[m_idx]` — so host filler and backend hash
identically by sharing one function, not by agreeing.

⚠ Note `make lint` does **not** build `blake3-6round` (recorded at
`thoughts/shared/lfm-real-hash/phase2-report.md:446, 520`). Moving the feature
into a lower crate widens that blind spot — the feature must be added to the
Makefile's combination matrix in the same change.

### 1.3 What the two backends must implement

✓ VERIFIED the contract, `crypto/crypto/src/merkle_tree/traits.rs`:

```rust
pub trait IsMerkleTreeBackend {                       // :11-32
    type Node: PartialEq + Eq + Clone + Sync + Send;
    type Data: Sync + Send;
    fn hash_data(leaf: &Self::Data) -> Self::Node;    // :16
    fn hash_leaves(..) -> Vec<Self::Node>;            // :20, defaulted
    fn hash_new_parent(a: &Self::Node, b: &Self::Node) -> Self::Node;  // :31
}
pub trait IsStreamingLeafBackend<F>: IsMerkleTreeBackend {            // :47-59
    fn hash_bytes(data: &[u8]) -> Self::Node;                          // :54
    fn hash_data_from_slices(a: &[FE<F>], b: &[FE<F>]) -> Self::Node;  // :58
}
```

The contract is **byte-oriented** (`:52-58`: `hash_bytes` "Equals `hash_data`
applied to the elements `data` encodes"). That is a direct fit for standard
blake3 and a poor one for any felt-absorbing variant — a second, independent
argument for the byte-oriented choice.

✓ VERIFIED the leaf **encoding does not move**: `leaves_bit_reversed_grouped<E,
B>` (`commitment.rs:55-110`) and `commit_bit_reversed_with<E, B>`
(`commitment.rs:175-190`) are already backend-generic and serialize
`rows_per_leaf` bit-reversed rows column-by-column big-endian into a reused
buffer, then call `B::hash_bytes(buf)` (`:94`). Only the `keccak_*`-named
wrappers pin the alias (`commitment.rs:123, 137, 148, 165`). **Consequence: D0's
S1 wide-leaf spec item does not apply to P-a.** The RV64 leaf keeps the exact
byte layout it has today; what binds opening width remains the explicit I3 check
the verifier already performs (`verifier.rs:204-213`), not the hash.

### 1.4 The Pair/Batched two-element invariant

✓ VERIFIED the invariant is documented at `config.rs:93-106` and pinned by a
test that asserts `<Batched>::hash_data(&vec![a,b]) ==
<Pair>::hash_data(&[a,b])` over three vectors (`tests/commitment_tests.rs:110-121`),
plus a second test that the streaming routes agree with `hash_data`
(`:124-152`).

✓ VERIFIED it is load-bearing, not decorative, and the reason is asymmetric:
the prover builds FRI-layer trees with `Pair` (`fri/mod.rs:105`) and the
verifier authenticates those same openings with `Batched`
(`verifier.rs:736`, `verify_merkle_path::<H::Batched<FieldExtension>>`). The
verifier never uses `H::Pair` at all.

**How the blake3 instance honours it: one family, both sides.** Define
`Blake3Batched<F>` and `Blake3Pair<F>` over the *same* serialize-then-`hash_bytes`
routine, with `Pair::hash_data(&[a,b])` implemented as
`hash_bytes(a.be ‖ b.be)` — literally the two-element case of the batched path.
Do not prove that two independently-written encodings coincide; make them one
function. Then keep the existing invariant test and add the blake3 arm to it.

### 1.5 Round count — DECIDED: 6 rounds

**Mauro, 2026-08-12: 6-round is the target ("to see if this works"); 7r stays
buildable.** All pipeline numbers below are 6r.

The round count stays a **compile-time** knob, matching what exists:
`BLAKE3_ROUNDS` (`blake3.rs:83-85`). A backend generic over a `const ROUNDS:
usize` would let one build produce two hashes and is exactly the failure
`blake3_socket.rs:203-215` was written to prevent. So: one `Blake3StarkHash`
whose backends call the crate-global `BLAKE3_ROUNDS`, and the feature moves the
whole tree at once.

⚠ **Do not invert the feature's polarity.** `blake3-6round` currently means
"6 instead of the default 7" (`blake3.rs:83-85`), and A6R-signoff /
`ORCHESTRATION.md:45` record the ratified framing as "7-round instantiated
baseline, 6 behind the feature". Flipping the flag's sense would silently change
what every existing measurement and report means by "default". Keep the name and
the polarity; make the campaign **build with `--features blake3-6round`** and add
it to the Makefile's lint/test matrix — `make lint` does not cover it today
(`thoughts/shared/lfm-real-hash/phase2-report.md:446, 520`), and PLAN.md:177
already lists "blake3-6round OFF by default (+16% if forgotten)" as a live build
trap. It is now a trap on the P-a pipeline too.

### 1.6 ★ What 6 rounds does to the leaf CONSTRUCTION (new decision surface)

The round-count decision has a structural consequence that is easy to miss, and
it makes the work *smaller*.

✓ VERIFIED the interop position, A6R-signoff `:104-106`: "7-round parent merges
are bit-compatible with published BLAKE3, so an external verifier can recompute
a tree. **6-round merges are computed by nothing else in the world.**"

At 6 rounds there is therefore **no external verifier to be compatible with** —
and standard BLAKE3's chunk-tree machinery (1024-byte chunks, per-chunk counter
`t`, the `CHUNK_START`/`CHUNK_END`/`PARENT`/`ROOT` flag schedule) exists purely
for interop and parallelism, not for security. Keeping it at 6r buys nothing and
costs three times over:

- ~6% extra compressions (one parent per 16 block compressions — §5, the
  overhead the hash matrix does not model),
- a chunk-tree state machine in each of the nine CUDA kernels (§6.1),
- the same state machine again in the wrap's eDSL emitter (§4.6), where every
  flag/counter case is emitted cells.

**Recommendation: define the RV64 leaf/parent hash as a bare cv-chain over
64-byte blocks** — `cv₀ = IV`, `cv_{i+1} = compress(cv_i, block_i, t=0,
block_len, flags)` with one domain constant per role and the length bound into
the final block — and use **the same construction for both round counts**, with
the `blake3` crate anchoring the *compression function* at 7r rather than the
full hash. That is already how the socket is anchored
(`prover/Cargo.toml:44-45`: `blake3::hash(a ‖ b ‖ "LFMC")` is a one-block call),
and `CANONICAL_VECTORS` (`blake3.rs:198-462`) already KATs the compression at
both round counts across `block_len` 18–64.

⚠ The cost of this recommendation: the 7r arm stops being a *tree-compatible*
BLAKE3. If the point of keeping 7r buildable is "an external party can recompute
our commitments", then the 7r arm must keep the standard chunk tree and the two
arms are **two constructions**, not one knob — which roughly doubles the kernel
and emitter work. **This is a question for Mauro and it should be answered before
Stage 1 commits an API**, because §6.1's kernel agent needs to know which
structure it is building.

### 1.7 ★ DRAFT SPEC — the RV64 byte hash (`Blake3Chain`)

**Status: DRAFT.** Implemented at Stage 1 as the working default, per Mauro's
standing decision to proceed on the cv-chain; **formally pending ratification**.
This subsection is what §1.6 said had to exist before Stage 1 committed an API,
and it is the reference track G's chaining loop and Stage 5's emitter build to.

**Scope.** This is the *host, byte-oriented* hash the RV64 prover's Merkle
leaves, Merkle parents, FRI-layer leaves, transcript and grinding are built
from. It is **not** the LFM-native cell-oriented layer specified in
`commit-spec/COMMIT.md`, which chains `LFML_row` over cells inside the machine.
The two are different domains that happen to share a compression function.

#### 1.7.1 The construction

`Blake3Chain(M)` for a byte string `M`, at the crate-global round count
`BLAKE3_ROUNDS`:

```
n      = max(1, ceil(|M| / 64))                 # blocks; the empty message is ONE block
m_i    = bytes [64i, 64i+64) of M, zero-padded to 64, read as 16 LE u32 words
L      = |M| - 64·(n-1)                         # 0 when |M| = 0; 1..=64 otherwise
F_i    = (CHUNK_START if i = 0 else 0) | (CHUNK_END | ROOT if i = n-1 else 0)
         # CHUNK_START = 1, CHUNK_END = 2, ROOT = 8

cv_0     = BLAKE3_IV
cv_{i+1} = compress(cv_i, m_i, t = 0, block_len = 64, flags = F_i)[0..8]   for i < n-1
digest   = compress(cv_{n-1}, m_{n-1}, t = 0, block_len = L, flags = F_{n-1})[0..8]
```

The digest is those low 8 output words written **little-endian** = 32 bytes.
`t = 0` on every block; the chaining value is never reset.

In one sentence: **standard BLAKE3 restricted to a single chunk that never
ends.**

#### 1.7.2 The five properties it was designed for

- **P1 — the crate anchor is maximal.** For `|M| ≤ 1024` at 7 rounds this is
  bit-for-bit `blake3::hash(M)`. Standard BLAKE3's first chunk *is* this chain
  (t = 0, that flag schedule), and a message of at most one chunk has the
  chunk's output as the root, so `ROOT` lands on the same compression. The
  entire 0..=1024-byte range is therefore a known-answer test against the
  official crate with **no oracle, no JSON and no transcription in between** —
  the strongest external anchor available to any construction at this layer, and
  strictly stronger than the compression-only anchor §1.6 assumed.
- **P2 — a 64-byte message degenerates to exactly the parent form.** One block,
  first and last, so `flags = 0x0B`, `block_len = 64`, `h = IV`, `t = 0`. That is
  precisely `blake3_hash_merkle_parent` (`kernels/blake3.cu:222`) and
  `merkle_parent` (`tests/blake3_reference/mod.rs`). Note that is the *parent*
  claim. The two-element **leaf** invariant of `config.rs:93-106` is separate and
  easier — two Goldilocks elements are 16 bytes — and it holds **by
  construction** for a different reason: `Pair` and `Batched` are the same
  generic backend over the same digest, so `Pair::hash_data(&[a,b])` and
  `Batched::hash_data(&vec![a,b])` are the same 16 bytes through the same
  function. What P2 adds is that the parent layer needs no separate definition:
  it is this same hash at a 64-byte message.
- **P3 — the divergence is stated, not discovered.** Above 1024 bytes this is
  **not** standard BLAKE3: the standard would start chunk 1 (`t = 1`, `cv = IV`)
  and build a chunk tree over chunk CVs. We keep one unbounded chunk. This buys
  the ~6% of extra parent compressions §1.6 priced, and keeps the emitter and the
  nine CUDA kernels free of a chunk-tree state machine. The 7r arm remains a
  *compression-level* anchor, not a tree-compatible BLAKE3 — the cost §1.6
  already named and recommended accepting.
- **P4 — the framing is injective.** `(n, L)` determines `|M|`, and the blocks
  are `M` zero-padded, so distinct messages give distinct compression-input
  sequences. Two messages of different length never share a chain: they differ in
  `L`, or in `n` (hence in which block carries `CHUNK_END|ROOT`), or in block
  content. Padding introduces no cross-length collision.
- **P5 — parents are construction-independent.** A parent's message is one block,
  so bare cv-chain and chunk tree agree on it bit-for-bit. Whatever §1.6 is
  eventually ratified as, every parent hash in the tree is unchanged.

#### 1.7.3 Design forks, and how each is resolved

| # | fork | resolution | status |
|---|---|---|---|
| **F1** | `t` = block counter, or 0 throughout? | **0 throughout.** A block counter diverges from the crate at the second block and would cost P1 — the ≤1KiB anchor — for nothing: `t` carries no security here, it is the chunk index of a construction that has one chunk. | ✗ OPEN for ratification |
| **F2** | keep the `CHUNK_START`/`CHUNK_END`/`ROOT` schedule, or drop it for one constant? | **Keep.** Dropping it saves one selector in the emitter and breaks both P1 and P2 — and P2 is the invariant `config.rs` requires. The emitter cost is three constants selected by first/last, not a state machine. | ✗ OPEN for ratification |
| **F3** | domain-separate leaves from parents? | **No — inherit keccak's posture exactly.** ✓ VERIFIED the live keccak configuration does not separate them either: `hash_new_parent_bytes` (`field_element_vector.rs:74-92`) is the digest of the two concatenated 32-byte nodes, and an 8-element leaf is the digest of the same 64 bytes (`:217-227`). P-a therefore *inherits* this property rather than introducing it, and the argument that covers keccak's tree covers this one unchanged. Changing it is a change to both hashes, not to blake3. | ✗ OPEN — carried, not new |
| **F4** | 128-bit socket-style digest, or 256-bit? | **256-bit** (Mauro, decided). §0's finding 2 is why: 128 bits is a 64-bit collision bound, and it also keeps `Commitment = [u8; 32]` and the rkyv wire format byte-identical. | ✓ DECIDED |

#### 1.7.4 The KAT schedule

What the vectors have to discriminate, and the length at which each becomes
visible. Both round counts, every row.

| # | input | what it pins |
|---|---|---|
| K1 | `|M| = 0` | the empty message is ONE block with `block_len = 0`, not zero blocks |
| K2 | `|M| ∈ 1..=63` | `block_len` is the true length; the tail is zero-padded |
| K3 | `|M| = 64` | **P2** — one block, `0x0B`, and equality with the parent form |
| K4 | `|M| = 65` | the first chain step: block 0 loses `CHUNK_END\|ROOT`, block 1 gains it |
| K5 | `|M| = 128` | an exact multiple of 64 does not emit a spurious empty final block |
| K6 | `|M| ∈ {192, 256, 1024}` | the interior blocks carry `flags = 0` |
| K7 | `|M| = 1088` | **P3** — the first length past one chunk, where we leave the standard |

At **7 rounds**, K1–K6 are all specified by reference to an external artifact:
each equals `blake3::hash(M)` from the official crate (P1). They are checked that
way in the tests rather than transcribed, so there is nothing to mistype. K7 is
specified as `≠ blake3::hash(M)` — the negative control for P3, without which
"we implement the single-chunk chain" would be unfalsifiable.

At **6 rounds** the vectors are generated from this construction and committed as
a table. §1.6 said no external artifact exists at 6 rounds; **that turns out to
be too pessimistic**, and the correction matters because it is the campaign's
weakest provenance link.

✓ **Cross-checked, 2026-08-14.** #903's Python oracle
(`thoughts/blake3/blake3-oracle/blake3_ref.py`) is a full standard-BLAKE3
implementation with the round count as a parameter — `blake3_hash(data, out_len,
rounds)`. Two facts make it usable as an independent reference here:

1. At `rounds = 7` it reproduces the official `blake3` package (1.0.9)
   bit-for-bit at every length checked, **including multi-chunk lengths** (1088,
   2048). So the oracle is standard BLAKE3, pinned from outside, not just at the
   compression level but at the tree level.
2. Standard BLAKE3 over a message of at most one chunk *is* this construction
   (P1) — at any round count, since P1's argument is structural and does not
   mention the round function.

So `blake3_hash(m, 32, 6)` is an independent computation of `Blake3Chain` at 6
rounds for every `|m| ≤ 1024`. It was run over all twelve KAT messages: **the
eleven at `|m| ≤ 1024` all match**, and **1088 differs** — which is P3 confirmed
from the other side, and is strictly more than the 7-round negative control
gives. The 7r control says "we are not the standard at 7 rounds"; this says the
divergence is *the chunking*, because a reference that is standard at 6 rounds
too still parts from us at exactly the chunk boundary.

That leaves the 6-round table cross-checked by two implementations sharing no
code, over the whole range the prover actually hashes in. It is still not a
*published* vector — nothing published computes this — but "regression pin only"
would now understate it.

⚠ **Fragility to record.** The oracle survives only as `__pycache__` bytecode
(`blake3_ref.cpython-314.pyc`) in an untracked directory; the `.py` source is
gone, as is `canonical_6round_vectors.json`. The cross-check was run by loading
the bytecode directly. The digests below are therefore the durable record of the
result — re-running it depends on an artifact one `git clean` removes.

#### 1.7.5 The committed 6-round vectors

Message of length `n` is bytes `37i + 11 (mod 256)`, `i` in `0..n` — the same
generator the existing compression-level anchor uses. Digests are
`Blake3Chain` at **6 rounds**, hex, in the byte order the digest has on the
wire. Live copy: `CHAIN_KAT_6ROUND` in
`crypto/crypto/src/hash/blake3/chain.rs`, asserted by
`six_round_chain_matches_the_committed_table`. The `oracle` column is the
independent cross-check described above.

| len | digest | oracle @6r |
|---|---|---|
| 0 | `3C3BBB1F335A31EA86464B651C0206FC81D33262AE00EA1A65F3D1D04AFAEFC9` | agrees |
| 1 | `2A50E45B8921F9EFA008D9F39F7165600CF48A7F0E859C2122E3CCB6B9677EE5` | agrees |
| 31 | `C38BF62F506040B2600273778D281B8943621E2B8A9F59E2379F8FD7E5C85125` | agrees |
| 63 | `C373F51A5EB8B27EA05BB1F6F4E62E924FF4D8A279F0D05AFA5CD519391D6389` | agrees |
| 64 | `5900A1E398BB2BF6D3BA7F1A29197B79C86B71AD2C2631F4AC736C82DB043CB5` | agrees |
| 65 | `53953FCADC39B8623901AF7B534F2F6933E312F50299331334E6C0A7C9DBC2BE` | agrees |
| 127 | `9E0DD8168D199A04590C2CBA439B270776E42715D518F68655E56692483E505E` | agrees |
| 128 | `5CAFFC8784E817BBBA991B2108C26A3DFDF804245EF63AE1040A3C34F1B362FF` | agrees |
| 192 | `399D6B9ADEB2F88450775F773E9DEC08836C135713C2C5DD09F4CECEB0ED3888` | agrees |
| 256 | `FBCAB3699A4959FA37190E98CA5142DDBC88330F2E7D12335DB9C6C8881A0B87` | agrees |
| 1024 | `F395E7E2150363B6D200487515425B0204EEA424072183B701176ECCBE0FFE1B` | agrees |
| 1088 | `B4738EDE77A6EC166EE97667118D4793CBF2B08B45AAC7C6D52943B5D298C688` | **differs** — P3, as designed |

#### 1.7.6 What Stage 1 built against this spec

- `crypto::hash::blake3::chain::Blake3Chain` — the construction as a `digest`
  hasher, so it drops into the Merkle backends and (at Stage 3) the transcript.
- `BatchBlake3Backend` / `PairBlake3Backend` — the *same* two generic backends
  the keccak aliases are, with the digest swapped. P2 is therefore structural:
  the two families are one function, not two encodings shown to agree.
- `stark::config::Blake3StarkHash` + `CommitmentHash::Blake3`, non-`cuda` only.
- Oracles: the 7-round anchor over all 1025 lengths; the P3 divergence control;
  the parent-form check at both round counts; streaming-split agreement; the
  committed table and its distinctness control; the two-element invariant with a
  blake3 arm; a commit→open→verify round trip with a negative control.

**Not** built, and why: a full prove→verify under `Blake3StarkHash`. `fri/` is
not parameterized over the configuration (§4.1), so the prover would build
keccak FRI trees and the verifier check them with blake3. That is Stage 2.

---

## 2. Transcript

### 2.1 What `DefaultTranscript` actually is

✓ VERIFIED `crypto/crypto/src/fiat_shamir/default_transcript.rs`. It is a thin
`digest::Digest` wrapper, not a bespoke sponge:

- `use crate::hash::platform_keccak::PlatformKeccak256 as Keccak256;` (`:3`),
  `use digest::Digest;` (`:5`)
- `pub struct DefaultTranscript<F: HasDefaultTranscript> { hasher: Keccak256, … }` (`:31-32`)
- doc `:19` — "Keccak-sponge Fiat-Shamir transcript with a Plonky3-style duplex
  output buffer"
- squeeze (`:76-78`): `result_hash = hasher.finalize_reset(); hasher.update(result_hash)`
- `append_bytes` → `hasher.update(new_bytes)` (`:113-118`)
- `append_field_element` → `element.stream_bytes(&mut |b| self.hasher.update(b))` (`:121-125`)
- `state()` → `hasher.clone().finalize().into()` (`:128-129`)

✓ VERIFIED `IsTranscript` has five methods
(`fiat_shamir/is_transcript.rs:7-26`): `append_field_element`, `append_bytes`,
`state() -> [u8;32]`, `sample_field_element`, `sample_u64`. `IsStarkTranscript`
adds `sample_z_ood*` (`:28+`), whose bodies are defaults over
`sample_field_element`.

✓ VERIFIED the transcript is already injectable into the RV64 prove/verify path:
`multi_prove(… transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> +
Clone + Send) …)` at **`prover.rs:3055-3068`**, mirrored by
`multi_verify` at **`verifier.rs:1191-1200`** and `multi_verify_archived` at
**`:1211-1214`** (both `impl IsStarkTranscript<..> + Clone`).
⚠ Note these three line numbers correct D0-DESIGN.md §2, which cites
`prover.rs:3032-3044` / `verifier.rs:1219-1231` — stale after the intervening
commits, and it misses `multi_verify_archived` entirely. The in-place rkyv
verify path takes a transcript too, so a blake3 transcript has **two** verifier
entry points to satisfy, not one.

**So `Blake3Transcript` is the smallest piece of P-a.** The honest change is to
make `DefaultTranscript` generic over `D: Digest + Clone` and add a
`Blake3Transcript` alias — every method body is already hash-agnostic. Estimated
S.

### 2.2 The design decision the brief asks for: LFMT vs bytes

This is a real fork, and it is **not** the same fork as D0's. State it plainly:

| | **bytes-oriented blake3 transcript** | **B1 / LFMT construction** |
|---|---|---|
| what it is | `DefaultTranscript` with blake3 in place of keccak; 256-bit state; absorbs 64 B per compression | `blake3::hash(state ‖ operand ‖ "LFMT")` truncated to 128 bits (`blake3_socket.rs:245-254`); absorbs one cell (4 felts) per step |
| wrap hosts it with | promoted `LFM_BLAKE3` (does not exist as a chip yet) | the **existing** `LFM_HASH` socket arm |
| rate | 8 felts / compression | 4 felts / compression |
| state / challenge entropy | 256-bit state | 128-bit state, `squeeze_ext` takes 3 of 4 lanes ⇒ **96-bit challenges** (D0-DESIGN.md §3 item 3, unanalysed at production query counts) |
| `append_bytes` | native | ⚠ no byte-level absorb; needs a padding-and-length-bound byte→cell convention, specified not improvised |
| `state() -> [u8;32]` | native | ✗ no equivalent — and `state()` is what seeds grinding (§3) |

**How much does it matter?** Less than the leaf decision, and the numbers say
so: the transcript/spine is 2,667 of keccak's 118,080 permutations = **2.3%**
of the hash bill (`others/lfm-hash-matrix-scope.md:208`, ✓ read). So the
transcript should **follow** the leaf/tree decision rather than drive it.

**The reason to keep one family across all four domains is not elegance, it is
width.** If the leaves use the general chip and the transcript uses the socket,
the wrap's emitted program carries **both** AIRs — two hash tables, two sets of
preprocessed columns, and the tower re-absorbs both traces. PLAN.md's own D1
census already found that `LFM_HASH` dominates the tower's leaf bill at 57%
(`PLAN.md:151-153`); adding a second hash table makes that worse, not better.

**Recommendation: bytes-oriented, one family.** Take LFMT only if the socket
route wins the §1 digest-width decision, in which case all four domains go
socket-shaped together.

---

### 2.3 ★ Two riders come due at Stage 3

✓ VERIFIED `others/lfm-migration-riders.md` (read in full). It lists changes that
are "cheap-to-free if they ride the transcript/hash rebuild … and not worth a
proof-breaking change on their own", with an explicit admission rule: an entry
belongs there only if "the migration has to touch that code anyway" (`:63-67`).
**P-a Stage 3 is that migration for both entries.** They were written for the
ecosystem hash migration; nothing in them is LFM-specific.

**Rider 1 — constant-consumption challenge sampling (`:7-18`). Adopt it.**
`sample_field_element` loops on rejection, and "a straight-line machine cannot
follow a data-dependent consumption schedule, so the LFM transcript replay
encodes the no-rejection schedule and is **unprovable for a transcript that ever
rejects**" (`SOUNDNESS.md` §6.3, cited at `:12-15`). Cost of the fix is
completeness only, bounded `< 10^-6` per proof at production draw counts.

⚠ **This does not go away by switching to blake3.** Rejection sampling lives in
the *field* layer, not the hash layer — D0-DESIGN.md §3 item 2 traces it to
`extensions_goldilocks.rs:575-581` calling the base sampler three times, each an
unbounded `loop` at `goldilocks.rs:548-555`. A byte-oriented `Blake3Transcript`
inherits it unchanged. So P-a either adopts the rider or ships a blake3 RV64
transcript that carries the same standing unprovability restriction into every
future wrap — having just paid the proof-breaking cost that would have removed
it. **Adopted, for the BLAKE3 configuration only** (`TranscriptHash::
CANDIDATES_PER_COORDINATE`); keccak keeps the unbounded schedule so existing
proofs do not move.

⚠ **"The cheapest item in the whole plan" understates the cost.** A coordinate
draws `n = 2` candidates where the unbounded schedule draws ~1, so **challenge
sampling consumes twice the squeeze bytes** — a cubic-extension element goes
from ~3 candidates (0.75 squeezes) to exactly 6 (1.5). The transcript is 2.3% of
the hash bill, so it is small in the prover, but it lands in the recursion guest,
which replays every challenge and is exactly what the recursion campaign has been
optimizing. `n = 1` would be free — it is today's modal cost — but leaves a
≈ 2⁻³² per-coordinate tail, one proof in a few hundred thousand, which is not
negligible enough to call the schedule fixed. `n = 2` puts the tail at ≈ 2⁻⁶⁴.

⚠ **The fallback may NOT be a modular reduction.** Reducing an out-of-range
candidate mod `p` is free and total, but biases challenges by ≈ 2⁻³² per draw;
over ~10⁴ draws that is ~2⁻¹⁹ of statistical distance, which would *dominate*
the ~92 proven bits SECURITY-LEVELS establishes. The implementation draws on
instead, keeping the distribution exactly uniform. Failing instead would make
challenge sampling fallible on the verifier's replay path — an `Option` return
through every caller, and a panic risk where the no-prod-panic policy forbids one.

**Rider 2 — statement pad (`:20-61`). ✓ RE-DERIVED, and the premise below was
wrong: the arithmetic does NOT move, so P-a is not this rider's forcing
function.** The continuation-epoch statement encodes to `207 + L` bytes
(`L = |public_output|`, one byte per COMMIT); the inherited cursor shift is
`(3 + L) mod 4`, so Phase-A root absorbs land misaligned and need splicing —
"2 roots × 8 halves × T tables … at T = 24 that is 384 `BitDec` + ~13k `BALU`
rows per proof", ~0.2% of instructions but "low single-digit percent of the
machine's fixed trace floor" (`:51-57`). Zero for the ~1-in-4 workloads whose
`L` lands on a boundary.

✓ VERIFIED **the `mod 4` is the machine's half width, not the sponge's rate.**
`epoch_statement_cursor_is_three_plus_output_len` (`machine_tests.rs:2200`)
asserts the shift modulo `keccak_host::BYTES_PER_HALF`, and that constant is
**4** (`keccak_host.rs:15`) — the eDSL packs absorbed bytes into 4-byte halves.
The rate appears only in `padded_len` / `num_blocks`, i.e. in how many
compressions an absorb costs, never in where a root lands. Since 4 divides both
136 and 64 and the message bytes are identical either way, **the shift and the
splice cost are invariant under keccak → blake3**. The earlier claim that "the
shift arithmetic is absorb-granularity-specific and P-a moves the granularity"
conflated the sponge rate with the half width.

What *does* move is the compression count for the same absorb: keccak takes
`floor(n/136) + 1` permutations (2 for `n = 207`), `Blake3Chain` takes
`ceil(n/64)` compressions (4). More compressions, each ≈ 1/13.7 the cell cost, so
≈ 6.9× cheaper for the statement absorb — but that is the §5 census, not this
rider.

⚠ **"One-byte pad" is a misnomer.** `L` is workload-determined, so a fixed byte
cannot align anything; the pad has to be the 0–3 bytes that take `207 + L` to
the next multiple of 4. The file's own second correction implies this, but its
title does not.

**✗ OPEN — for Mauro, not forced by P-a.** Under the riders file's own admission
rule an entry belongs there only if "the migration has to touch that code
anyway" (`:63-67`), and Stage 3 does not: the statement encoder is hash-agnostic
and its arithmetic is unchanged. The natural host is **Stage 5**, which does
rewrite these emitters, or Stage 6, which is the proof-breaking moment the tag
bump wants. The numbers to decide on are above; the cost of waiting is that
~3-in-4 workloads keep paying ~0.2% of epoch-verify instructions.

Note the file carries three self-corrections on rider 2 (`:33-49`), including
that the `16R` term is dead because `runtime_page_ranges` is always empty for
continuation epochs. It is not re-imported above.

## 3. Grinding

✓ VERIFIED `crypto/stark/src/grinding.rs` in full. Two-layer keccak PoW over
`digest::Digest`:

```
inner  = Keccak256( PREFIX(8) ‖ seed(32) ‖ grinding_factor(1) )   // :80-90, 41 bytes
valid  = u64be( Keccak256( inner(32) ‖ nonce_be(8) )[..8] ) < 2^(64-gf)  // :66-76, 40 bytes
PREFIX = 0x0123456789abcded                                        // :6
```

`is_valid_nonce(seed: &[u8;32], nonce: u64, grinding_factor: u8)` (`:21`) is
seeded by `transcript.state()`.

**What P-a needs here is small and mostly mechanical.**

- Both hashed inputs are 41 and 40 bytes — one keccak block each, and equally
  **one blake3 compression each**. The wrap-hosted re-check is therefore
  **2 compressions per proof**, i.e. cost-irrelevant either way. Grinding is not
  a reason to choose anything.
- Because the file is written against `digest::Digest` (`:2`) and takes/returns
  `[u8;32]`, swapping the hash is a type substitution with **no signature
  change** — `is_valid_nonce`'s seed stays `[u8;32]` because the blake3
  transcript's `state()` is also 32 bytes.
- ⚠ **Do not scope grinding out for RV64.** D0-DESIGN.md §3 item 1 recommends
  `grinding_factor: 0` — that is right for *LFM* proofs and wrong here.
  `MIN_PROOF_OPTIONS` sets `grinding_factor: 1` on the RV64 recursion presets
  (`prover/src/recursion.rs:39-45`, cited in D0-DESIGN.md §3), and grinding is
  part of the RV64 proof's claimed security budget. Port the PoW; do not delete
  it.
- The wrap must emit the two compressions. ✓ VERIFIED **it already does — there
  is no gap.** The search above looked in `epoch_verify.rs` and the check is not
  there; it lives in the challenge spine, `prover/src/lfm/epoch.rs`, which is
  where the transcript absorbs are and therefore the right place.
  `emit_grinding_check` (`:350-406`, called at `:514` exactly when a nonce is
  present) builds the inner keccak over `PREFIX ‖ state ‖ factor` and the outer
  over `inner ‖ nonce_be`, then bit-decomposes the digest's first lanes and
  asserts the top `factor` bits are zero — `is_valid_nonce`'s predicate, done as
  a bit decomposition plus zero assertions rather than a comparison, because the
  bound is a power of two. Its own doc states the stake: "the nonce is absorbed,
  so the query indices depend on it, and an unchecked nonce is a free re-roll of
  every query index at zero cost."

  ⚠ **Consequence for Stage 5, and it is a site §4.6 does not list.** The check
  reaches keccak through `edsl::keccak256` (`edsl.rs:439`) →
  `keccak256_absorb_all` (`:483`), which is a **sponge-framing** emitter: it
  loops over `num_blocks`/`BLOCK_HALVES` and splices `pad_half`, all of which
  encode keccak's 136-byte rate and `pad10*1`. A BLAKE3 port needs the framing
  rewritten to 64-byte blocks and the chain's length-in-final-block convention,
  not just a compression swap. Two compressions per proof, so the *cost* is
  irrelevant — the work is in the framing.

---

## 4. Blast radius

### 4.1 `crypto/stark` — the parameterization

The whole host commitment path takes the configuration. The main trace commits
through `<H::Batched<E> as IsStreamingLeafBackend<E>>::hash_bytes`
(`prover.rs:893`) and the verifier checks it through `H::Batched` (`verifier.rs:594,
598, 677, 684`). FRI commits its layer trees with `H::Pair`
(`fri/mod.rs:commit_phase_from_evaluations`, `fri/batched.rs:batched_commit_phase`)
and opens them through `query_phase`, also on `H`.

The join between those two families is what makes a proof verifiable: the
prover builds FRI layer trees with `H::Pair` and the verifier re-hashes each
opened pair with `H::Batched` (`verifier.rs:736`), so the `StarkHash`
two-element invariant — `Batched::hash_data(&vec![a, b]) ==
Pair::hash_data(&[a, b])` — is load-bearing rather than decorative. Naming one
`H` is what makes the two sides agree; a configuration that broke the invariant
would reject every honest proof at its first FRI query, loudly.

⚠ **The `cuda` fork covers FRI as well.** `StarkHash::Pair` carries the same
`KeccakTreeBackend` bound `Batched` does under `cuda`, because `gpu_lde`'s FRI
commit drives the whole commit phase on device and hashes every layer with the
keccak kernels, labelling the result with the backend type it was handed. So a
cuda build has no BLAKE3 configuration for FRI layers either, which is
consistent with `Blake3StarkHash` not existing under `cuda` at all (§4.5, R4).

The `FriLayerMerkleTree` / `FriLayerMerkleTreeBackend` aliases survive as the
*default* configuration's names — the `const _` assertions in `config.rs` pin
them to `KeccakStarkHash`'s members, and `math-cuda`'s parity tests build
reference trees with them.

### 4.2 The `CommitmentHash` tripwire

✓ VERIFIED `CommitmentHash` has one variant (`config.rs:63-67`) and the
crate-global `COMMITMENT_HASH` is pinned to it (`:71`), tied to
`KeccakStarkHash` by a `const _` assert (`:193-196`).

✓ VERIFIED the only external consumer is the H1 guard:
`build_artifacts_with_hasher` opens with an exhaustive `const _: () = match
stark::config::COMMITMENT_HASH { CommitmentHash::Keccak256 => () };`
(`registry.rs:158-160`), documented at `:136-151` as the thing that "cannot
compile until someone decides here what the artifacts should say."

⚠ **Note the guard's direction.** It fires when a `Blake3` *variant* is added,
and again when the *aliases* flip (because `COMMITMENT_HASH` describes the
aliases, not the active `H`). It does **not** catch "prover ran under a blake3
`H` while the global const still reads Keccak256" — the const is global, the
configuration is per-type. If P-a keeps keccak as the default alias while a
blake3 `H` exists (which is the plan), `COMMITMENT_HASH` becomes a
half-truth for the duration. Either make the guard read `H::COMMITMENT_HASH` at
the call site, or write down that the global const describes the *default*
configuration only.

### 4.3 Recursion guests — the hard external dependency

✓ VERIFIED the guest acceleration mechanism, and it is clean:
`crypto/crypto/src/hash/platform_keccak.rs:1-5` — "Keccak-256 implementation
selected per target: the `keccak_permute` precompile on the riscv64 guest,
plain software `sha3::Keccak256` on host. Wraps
`lambda_vm_syscalls::keccak::Keccak256` with the `digest` crate traits so it's a
drop-in replacement anywhere a `D: Digest` is expected (Merkle tree backends,
Fiat-Shamir transcript)." The riscv64 arm is at `:7-45`.

✓ VERIFIED the guest-side sponge is `syscalls/src/keccak.rs` — "High-level
Keccak-256 hasher backed by the lambda-vm `keccak_permute` precompile" (`:1-5`),
rate 136 (`:37`), domain byte 0x01 (`:43-44`).

✓ VERIFIED **there is no blake3 syscall**: `syscalls/src/` contains
`allocator.rs, ef_io.rs, entrypoint.rs, keccak.rs, lib.rs, random.rs,
syscalls.rs` (on `origin/main`), and grep for `Blake3|BLAKE3` over `syscalls/`
returns nothing.

⚠ **A silent-desync hazard sits directly on P-a's path.**
`platform_keccak.rs:14-21` carries a load-bearing invariant: "this adapter must
remain a PURE PASSTHROUGH … The TypeId specializations in
`crypto/crypto/src/merkle_tree/backends/field_element_vector.rs` bypass it and
drive the syscall sponge directly, on the assumption that both paths hash
identically. Adding ANY behavior here … silently desyncs the specialized
branches from the generic path — and the failure surfaces as **in-guest proof
rejection, not as a host test failure**." A blake3 backend must either avoid
that specialization or get its own, and the check is a guest run, not a host
test.

✓ VERIFIED **#903 is unmerged and is exactly the missing piece.** Commit
`35038501 feat(prover,executor): BLAKE3 6-round compression accelerator` lives
on `feat/blake3-accelerator` (local + `origin/`), is **not** on `origin/main`
and **not** on `blake3-real-hash` (`prover/src/tables/` here contains only
`keccak.rs, keccak_rc.rs, keccak_rnd.rs`). Its message states the deliverable:
syscall `u64::MAX-2`; ABI `x10 → 8-aligned 176-byte region, h[32] | m[64] |
t[8] | len,flags[8] | out[64]`; executor implementation; chip
`prover/src/tables/blake3.rs` at 3,219 main columns / 1,397 sends / ~5,316
cell-equivalents per compression ≈ 1/13.7 of a post-#889 keccak-f. ⚠ It is the
**6-round internal variant**, resting on the A6R assumption, "to be ratified in
the spec before production use."

**Consequence for staging:** the RV64 recursion track (#844/#845/#846/#847) runs
a guest verifier whose hashing is keccak-precompile-accelerated. Flip the
prover's hash without merging #903 and that guest computes blake3 in RV64
software — a large cycle regression in the exact place that campaign has been
optimizing. Stage 4 is therefore gated on merging #903, and #903 itself carries
an unratified round-count assumption.

### 4.4 program_id / ELF digests / fixtures / CI

- ✓ `registry.rs:128-171` derives artifacts through `commit_group` and the two
  `preprocessed_commitment` helpers, all of which commit through `stark`'s
  Merkle layer. **Every LFM registry root moves** when the aliases flip, and
  `program_id` is derived from `hasher` (`:130-134`). Regeneration is
  `cargo run --bin compute_lfm_registry --release`, under the standing policy
  that "a drift failure is investigated, never re-blessed to silence the test."
  ⚠ For P-a this is an *intended* move, so the re-bless is legitimate — say so
  in the commit, and keep the Test-hasher rows as the honest control.
- ✓ Checked-in `.bin` files are ethrex **block inputs**
  (`executor/tests/ethrex_{10_transfers,bench_4,empty_block,simple_tx}.bin`),
  not proof bytes — commitment-hash-independent. ? INFERRED no checked-in proof
  blobs exist; I found none via `git ls-files`.
- ✓ **CI census.** `.github/workflows/pr_main.yaml` has ten jobs — `lint:19`,
  `test-executor:51`, `test-cli:158`, `test:186` (the gate, `if: always()`),
  `test-disk-spill:223`, `test-stark-cuda-lib:282`, `build-prover-tests:309`,
  `test-prover:344`, `test-prover-comprehensive:436`, `seed-elf-cache:522`.
  **Nothing pins proof bytes**, and `cross_verify_vm.sh` is not wired into CI —
  it is operator-run, so Stage 6's positive control is a manual gate. Two jobs
  matter to P-a:
  - `test-cli:178` — "Run syscalls host tests (keccak differential vs sha3)". A
    blake3 syscall needs the twin of this differential, against the `blake3`
    crate as the reference.
  - `test-stark-cuda-lib:282` — compiles the `cuda` feature **on every PR**.
    This is what will enforce R4's discipline: the blake3 `StarkHash` instance
    must be `#[cfg(not(feature = "cuda"))]` or this job goes red. Useful, not
    an obstacle — it turns the GPU fork into a compile error a reviewer sees.
  GPU execution lives in the separate `gpu-tests.yml` workflow (consistent with
  the standing note that it runs on `merge_group` against the rented box).
- ✓ VERIFIED **continuation chaining binds NO commitment-hash-derived value
  across epochs.** The epoch N→N+1 carry is `reg_fini: Vec<u32>`, a plain
  register file (`continuation.rs:438-456`, consumed at `:1703` as the next
  epoch's `register_init`); the rest of the carry is the GlobalMemory LogUp bus,
  which is field elements. No Fiat-Shamir state crosses either — every epoch
  builds a fresh `DefaultTranscript` seeded from the statement alone, and the
  `elf_digest` in that statement is an *independent* `PlatformKeccak256` over raw
  ELF bytes that does not move when the commitment hash does. `EpochProof`'s one
  `Commitment` field, `l2g_root`, binds an epoch to the **global** proof inside
  the same bundle — both sides move together under a flip, so it is inert.

  ⚠ **The format surface is a pinned constant, not a chained root.**
  `static_zero_page_commitment` (`prover/src/tables/page.rs:411-430`) is a
  hardcoded per-blowup commitment sitting directly on `verify_global`'s
  continuation path, and it is deliberately never supplied via private input
  ("zero-init pages use a compile-time constant and are never listed",
  `recursion.rs:113-115`) — so it is compiled into the host verifier *and* baked
  into the recursion guest ELFs. It moves on the flip. Regenerate with
  `cargo run --bin compute_static_commitments --release`, under the standing
  policy that a drift failure is investigated, never re-blessed to silence a
  test — which that function's own doc states.

  ✓ VERIFIED, upgrading the `? INFERRED` above: no checked-in proof blobs exist.
  The LFM `FixtureArchive` is a regenerable `/tmp` cache, untracked.

  ✓ The rkyv wire format does **not** move. `Commitment` is `[u8; 32]` and
  `StarkHash::Node` is deliberately not an associated type precisely so a
  configuration change leaves `StarkProof`'s derives byte-identical
  (`stark/src/config.rs:20-21, 109-112`), so #845's in-place verify path is
  untouched by a flip.

### 4.5 GPU

✓ VERIFIED the compile-time fork described in §0.3. Additionally, ✓ the
tree-less entry the CPU-trees phase needs exists: `try_expand_columns_batched<F,
E>` (`gpu_lde.rs:430`) takes **no** backend parameter and builds **no** tree —
GPU does the LDE, host does leaves and tree.

⚠ ✓ VERIFIED `device_only_gate` (`gpu_lde.rs:195-215+`) is **entirely
hash-agnostic** — field tower, env disables, power-of-two `lde_size`, LDE and
barycentric thresholds, `!is_preprocessed`, contiguous offsets, uniform
zerofier. That is the hazard, not the relief: under blake3 it still evaluates
**true**, and device-only residency drops the host trace that CPU leaf hashing
needs. Its own doc (`:178-186`) says a violated precondition hits a
`host_trace_empty` "hard-abort … the prove aborts loudly". So during the
accept-CPU-trees phase it **must be forced false** under the blake3
configuration — otherwise blake3 GPU proving aborts rather than falling back.
⚠ `:188-194` adds a LOCKSTEP obligation: the gate must imply the runtime
dispatch checks, and "a fallback condition added to a dispatch without a mirror
here turns every gate-true table into a hard-abort". Forcing it false is safe in
that direction; adding a blake3 condition to a dispatch without mirroring it here
is not.

✓ VERIFIED in this pass (upgraded from inherited): `grep -ril blake3
crypto/math-cuda/` returns **nothing**. The eleven kernel sources are
`arith.cu, barycentric.cu, constraint_interp.cu, deep.cu, ext3.cuh, fri.cu,
goldilocks.cuh, inverse.cu, keccak.cu, logup.cu, ntt.cu` — `keccak.cu` is the
only hash. Everything hash-agnostic survives untouched (LDE/NTT, constraint
composition, barycentric, DEEP, FRI fold arithmetic, LogUp, inverse/arith).
The full kernel and wrapper inventory is in §6.1, where the parallel track needs
it.

### 4.6 The LFM wrap's hosted-verify emitters — where the 4× materializes

✓ VERIFIED the five emission sites, all keccak today:

| domain | emitter | site |
|---|---|---|
| leaf absorption | `emit_leaf_hash(b, shape, values) -> KeccakDigest` | `sub_proof.rs:245` |
| trace Merkle paths | `edsl::keccak_merkle_walk(b, leaf, bits, &opening.siblings)` | `sub_proof.rs:289` |
| FRI-layer paths | `edsl::keccak_merkle_walk(..)` | `fri.rs:564` |
| transcript | `keccak_absorb` / `keccak_absorb_rev` | `builder.rs:414, 426` |
| ★ register commitment | `edsl::keccak_leaf_hash` + `edsl::keccak_merkle_tree_root` | `programs.rs:1266, 1270` |

★ The fifth is different in kind from the other four: `emit_register_commitment`
**builds a whole Merkle tree in eDSL**, over the cross-epoch register carry,
rather than walking or absorbing one. `keccak_merkle_tree_root` has exactly one
caller, so a grep for `merkle_walk` misses it. Note the interaction with §4.4:
the carry itself is hash-free on the host path, but the LFM wrap feeds it
through the commitment hash *in-machine* — deliberately, per `RootCells::from_digest`
(`lfm/epoch.rs:118-125`): "computing it from those cells is what binds them."
A BLAKE3 twin of this emitter is therefore part of Stage 5, not optional.
⚠ Scope check: the emitter ships, but the assembled epoch verifier is
`#[cfg(test)]` and no `LfmProgramKind` reaches it yet (`programs.rs:1149-1154`).

and the LFM-native counterparts that exist today — `edsl::leaf_hash_pair` and
`edsl::merkle_walk` over **one-cell** digests, used by the fixture programs
(`programs.rs:638-648`), with the host mirror `fixture.rs:145
host_leaf_hash_pair`. `KeccakDigest` is `[Cell;2]` (`sub_proof.rs:228`,
`fri.rs:314`) against the native one-cell digest — the "half as many cells per
Merkle level" payoff, and the 128-bit digest question from §0.2, are the same
fact seen twice.

**This is the switch that produces the win, and it is the stage that needs the
new chip.** `edsl::merkle_walk` compresses through the *socket*
(`blake3_socket.rs:86, 586`), so reusing it verbatim buys the socket's rate 4
and its 128-bit digest. Emitting against a promoted `LFM_BLAKE3` needs new eDSL
emitters, which do not exist.

---

## 5. The numbers, and what I checked about them

The campaign is planning against "blake3-6r is 4.06×" (`CENSUS.md:143`). I
traced that number to source rather than inheriting it, because P-a is being
sequenced on it.

✓ VERIFIED the composition (`others/lfm-hash-matrix-scope.md:60-95`, A6R-signoff
`:108-133`, F7 `:10-30`):

```
keccak  epoch-verify total   11,165,806,868 base-equiv cells   (census of the EMITTED program)
        hash term             9,381,609,472  (84.02%)  = P 118,080 × 77,992 × 1.01871 padding
        residue + BITWISE     1,784,197,396  (15.98%)
blake3  hash term               967,402,978  = 195,593 compressions × 4,946 cells
        total                 2,751,600,246  → 4.06× at 6 rounds
                                             → 3.85× at 7 rounds (5,714 cells/compression)
```

**Three things worth knowing before quoting 4.06×:**

1. ✓ **The rate penalty IS already in the model — I checked, because it is the
   obvious way for a figure like this to be wrong.** Keccak absorbs 17 felts per
   permutation, a rate-8 candidate 8, so absorption-bound work costs more
   invocations. `others/lfm-hash-matrix-scope.md:191-210` measures the split
   rather than assuming it: keccak 115,413 legs (67,671 leaves + 47,742
   paths/FRI) vs rate-8 187,902 (140,160 + 47,742) = **1.63×**, not the 2.125×
   ceiling, because 41.4% of the bill is path/FRI work that is 1:1 at any rate.
   `P_candidate ∈ [190,569, 193,569]`. The keccak side reproduces the ledger
   exactly (115,413 = entry 10's legs figure), which is what makes the candidate
   side trustworthy. **The 4.06× survives this check.**

2. ⚠ **But it is a rate-8 figure, and rate 8 is the general chip.**
   `blake3_probe.rs:683` computes it as
   `query_permutations_at_rate(&l.verify, 8)`, and the comment at `:686-694`
   says exactly what that means: "Rate 8 is BLAKE3's own: its socket absorbs two
   cells of message per compression … It is NOT the field-native chain's rate —
   that is `epoch_verify::LFM_HASH_RATE_FELTS`, which is 4 because the chain
   absorbs one cell per step. The two were the same number while the sponge was
   a three-cell duplex, and this line used to say 'blake and field-native' on
   that basis; **they have since diverged**." Wide-leaf absorption through the
   socket — chained or tree-shaped — amortizes to 4 felts per compression.
   **DERIVED** (my arithmetic, not the repo's): back-solving the leaf term from
   the two measured points gives ≈3,248 (query, group) terms over ≈1.095M leaf
   felts, so rate 4 lands at ≈277k leaf permutations and `P ≈ 330,000` ≈ 1.72×
   the rate-8 count; folding in the socket arm's slightly narrower 2,964 main
   columns (`PLAN.md:167-169`) puts socket-hosted blake3 at roughly **3.3× at 6
   rounds**, not 4.06×.
   **This is free to settle exactly and should be Stage 0**: the closed form is
   already rate-parameterized (`epoch_verify.rs:484-541` —
   `blocks_at_rate`, `leaf_permutations_at_rate`,
   `fri_leaf_permutations_at_rate`, `query_permutations_at_rate`), so changing
   the `8` at `blake3_probe.rs:683` to `4` and re-running the ignored instrument
   prices the socket route with no proving.

3. ⚠ **Provenance caveats already on record, which P-a inherits.** Only the
   keccak row is a census of a real artifact; the blake3 row is
   `measured residue + measured BITWISE + hardcoded P × measured AIR width`,
   with `let p = 192_000u64;` at `blake3_probe.rs:711` never asserted against
   the instrument's own computed interval (F7.1). And every figure is for a
   16-cycle fibonacci fixture epoch at blowup 8 / 73 queries — **3.9× under a
   production-sized epoch** (F7.2), while `CENSUS.md` applies the ratio at
   blowup2/219q and blowup4/110q. The ratio is probably more portable than the
   absolute, but neither has been checked at the presets the campaign will use.

**The 6-round decision is the cheaper arm of the matrix**: 4,946 cells per
compression and **4.06×**, against 5,714 and 3.85× at 7 rounds — the +15.5%
per-compression / +5.5% epoch-column delta derived at A6R-signoff `:118-133`.
So the decision moves the plan's headline number the right way, and every figure
in this document is the 6r arm unless it says otherwise.

**A standard chunk tree would add ~6% the model does not show** (? INFERRED, my
arithmetic): above 1024 bytes it costs one parent compression per 16 block
compressions. The model's 1.01871 factor is `KECCAK_RND` chunking waste, a
different thing. **§1.6 argues this 6% should simply not be incurred** — at 6
rounds the chunk tree has no interop purpose, and #903's ABI exposes raw compress
(`h[32] | m[64] | t[8] | len,flags[8] | out[64]`), so a bare cv-chain is
directly buildable on the guest, the host, the device and the chip alike.

---

## 6. Staging

Keccak stays the default through Stage 5. Every stage has an oracle that can
fail.

**On the king gate — the brief pointed at the wrong one for P-a.** ✓ VERIFIED
`prover/tests/d0_king_gate.rs` proves and verifies **LFM** proofs (`lfm_prove` /
`lfm_verify` over `trivial_program`, `:36-39`), and its own header says it "is
the LFM-side counterpart of `scripts/cross_verify_vm.sh`, which does the same
for RV64 ELF proofs in both directions" (`:7-9`). **P-a's king gate is
`scripts/cross_verify_vm.sh REF_OLD REF_NEW`** (`:1-36`): builds `bin/cli` at
both refs in an isolated worktree and exchanges real VM proofs per ELF, both
directions.

Its polarity inverts at the flip, and that is the point:
- Stages 1–4 (keccak still default): cross-verify must **PASS** both directions
  — that is the proof the refactor is inert.
- Stage 6 (flip): cross-verify must **FAIL** both directions, and a same-ref
  blake3 round trip must pass. A passing cross-verify after the flip would mean
  the hash did not actually move.

### 6.1 ★ PARALLEL TRACK — the blake3 CUDA kernels

**Pre-authorized by Mauro as a parallel workstream, not a tail stage.** This is
the right call: it is the only part of P-a with no dependency on the machine-chip
work (Stage 5) or the guest work (Stage 4), and leaving it to the end is what
would create the GPU regression window described in §0.3/R4.

**Start condition — two options, and the earlier one is real.**

- **Earliest (can start immediately):** the kernels depend on the *compression
  function* and the *leaf byte layout*, both of which are already frozen and
  readable today — `blake3_compress_rounds` (`blake3.rs:125-148`) and
  `leaves_bit_reversed_grouped` (`commitment.rs:55-110`, which serializes
  `rows_per_leaf` bit-reversed rows column-by-column big-endian and hashes the
  buffer once). Neither moves in Stage 1. **An agent can begin on the device
  compression function plus the two simplest leaf kernels right now.**
- **Blocking on one answer:** the *chaining construction* — §1.6's open question
  (bare cv-chain vs standard chunk tree). The device compression function and
  the byte serialization are identical either way, so roughly 60% of the work is
  unblocked; the leaf-kernel chaining loop and the tail handling are not.
  **Dispatch now, scoped to the compression function + serialization + the
  level/tail compressors; hold the multi-block leaf chaining until §1.6 is
  answered.**

**The kernel list.** ✓ VERIFIED firsthand against `keccak.cu` — nine hash
kernels to mirror, one device helper to replace, one kernel that needs nothing:

| keccak kernel | line | blake3 mirror needed |
|---|---|---|
| `keccak_f1600` (device helper) | `:50` | → `blake3_compress` device fn, 6r, from `blake3.rs:125-148` |
| `keccak256_leaves_base_batched` | `:152` | yes |
| `keccak256_leaves_base_row_pair_batched` | `:196` | yes |
| `keccak256_leaves_ext3_batched` | `:237` | yes |
| `keccak_comp_poly_leaves_ext3` | `:277` | yes |
| `keccak_fri_leaves_ext3` | `:326` | yes |
| `keccak_merkle_level` | `:394` | yes (parent compressor) |
| `keccak_merkle_tail` | `:408` | yes (parent compressor) |
| `keccak256_leaves_base_row_major_row_pair` | `:473` | yes |
| `keccak256_leaves_base_row_major_row_pair_range` | `:511` | yes (column-subset variant) |
| `merkle_gather_paths` | `:433` | **none — hash-agnostic**, reusable as is |

✓ VERIFIED the Rust wrappers that need blake3 twins, `crypto/math-cuda/src/merkle.rs`:
`keccak_leaves_base:33`, `keccak_leaves_ext3:83`,
`build_merkle_tree_on_device:316`, `build_comp_poly_tree_from_slabs_dev:494`,
`build_comp_poly_tree_from_evals_ext3_keep:544`,
`build_fri_layer_tree_from_evals_ext3:564`. `gather_merkle_paths_dev:358` is
hash-agnostic and needs no twin. Note tree *building* is on-device too, not only
leaf hashing.

**The parity oracle.** ✓ VERIFIED the template already exists — mirror these
rather than inventing a harness: `crypto/math-cuda/tests/keccak_leaves.rs`,
`merkle_root_parity.rs`, `fri_layer_tree.rs`, `comp_poly_tree.rs`,
`merkle_tree.rs`, `merkle_gather.rs`. The blake3 versions assert device output
against **the host 6-round implementation** (`blake3_compress_rounds` at
`BLAKE3_SIX_ROUNDS`), which is the same reference the chip's trace filler uses
(§1.2) — so device, host backend and in-circuit chip are all checked against one
function. Seed the compression-level check with `CANONICAL_VECTORS` +
`CANONICAL_OUT_7ROUND` (`blake3.rs:198-462`), which cover `block_len` 18–64 at
both round counts; ⚠ there is no 6-round expected-output constant table beside
`CANONICAL_OUT_7ROUND`, so the 6r arm's KATs are pinned by the host
implementation only — **generating and committing a `CANONICAL_OUT_6ROUND` table
is a prerequisite for the kernel agent to have an independent oracle at all.**

**Ordering guard.** The existing keccak parity tests must stay green throughout
— keccak remains the default until Stage 6, and the cuda-feature job runs on
every PR (`pr_main.yaml:282`).

**Effort: M.** Nine kernels, but they are structurally uniform, the byte
serialization is shared with the CPU path, and the parity harness is a template
rather than new design. The risk is not difficulty, it is the §1.6 answer
arriving late and forcing the chaining loop to be rewritten.

### 6.2 Stage table

| # | stage | oracle | effort |
|---|---|---|---|
| **0** | **Price the fork before building.** Re-run the census at rate 4 vs rate 8 (`blake3_probe.rs:683`), and settle §0.2's digest-width question with Mauro. Zero proving. | the instrument's own printout; `p_lo ≤ p ≤ p_hi` asserted (fixes F7.1 in passing) | **S** |
| **1** | Sink the compression core into `crypto/crypto` (§1.2b); **generate and commit `CANONICAL_OUT_6ROUND`** (R13 — the 6r arm has no independent KAT table today, and it is the arm we are shipping); add `Blake3Batched`/`Blake3Pair` + `Blake3StarkHash`; keccak still the alias | `CANONICAL_VECTORS` × both round counts (7r against `CANONICAL_OUT_7ROUND`, `blake3.rs:407`, itself anchored to the `blake3` crate; 6r against the new table); the Pair/Batched invariant test (`commitment_tests.rs:110-121`) extended with the blake3 arm; `make lint` incl. `blake3-6round` | **M** |
| **2** | Thread `H` through `fri/` (§4.1, ~13 sites); prove+verify round trip under `Blake3StarkHash` behind config. Close the §4.4 continuation-chaining question here | same-ref blake3 round trip passes; `cross_verify_vm.sh` still passes keccak↔keccak both directions | **M** |
| **3** | `Blake3Transcript` (make `DefaultTranscript` generic over `D: Digest + Clone`); port grinding to blake3 (§3); **adopt rider 1 — constant-consumption sampling — and re-derive rider 2's cursor arithmetic under blake3's 64-byte block (§2.3)** | transcript KATs; a grinding KAT; honest-path control — blake3 proofs with `grinding_factor: 1` verify; rider 1 pinned by a test that the draw consumes a fixed candidate count | **M** (was S–M; the riders add scope but remove a standing restriction) |
| **4** | **Guest leg.** Merge #903 (`feat/blake3-accelerator`); add `platform_blake3.rs` mirroring `platform_keccak.rs`; audit the TypeId specialization (§4.3) | an in-guest verify of a blake3-committed proof, measured in cycles against the keccak baseline. ⚠ host tests cannot see this failure | **L** |
| **5** | **Promote `LFM_BLAKE3` to a machine chip group**; new eDSL emitters; switch the four emitter sites (§4.6); re-census | adversarial-debate review (new chip group = soundness surface, house rule); re-census against Stage 0's projection; tamper controls both directions + honest-path control | **L** |
| **G** | **PARALLEL: blake3 CUDA kernels** (§6.1). Nine kernels + six wrappers; keccak stays default throughout | mirrored parity tests vs the host 6r implementation; `CANONICAL_OUT_6ROUND` committed first; existing keccak parity tests stay green | **M** |
| **6** | **Flip:** default aliases, registry re-bless, and the GPU fork resolved — if track G has landed, `StarkHash`'s `cuda` `KeccakTreeBackend` bound (`config.rs:116-122`) comes off; if not, blake3 stays `cfg(not(cuda))` and GPU proving stays keccak-only | `cross_verify_vm.sh` fails both directions (positive control); same-ref blake3 round trip passes; full suite green; `compute_lfm_registry` re-blessed deliberately | **S** code / **M** judgement |

Round count is no longer a Stage-6 decision — 6r is decided (§1.5), which is why
Stage 6's judgement load drops from L to M. What remains open for Mauro is
**§1.6's construction question (bare cv-chain vs standard chunk tree)**, and that
one is needed *early*, before Stage 1 commits an API and before track G writes a
chaining loop.

Stages 1–3 are independent of 4 and 5 and can run in parallel with them. Stage 5
does not depend on Stage 4. **Track G runs alongside everything and gates
nothing except Stage 6's GPU fork** — its start condition is in §6.1, and the
unblocked ~60% can begin immediately. Stage 6 depends on 1–5.

---

## 7. Risk register

| # | risk | status |
|---|---|---|
| **R1 ★** | **The 4× needs a chip the machine does not have.** §0.1. Scoping P-a as a `crypto/stark` config instance under-prices it by the whole of Stage 5 | Pinned by nothing. **This is the finding that should change the schedule.** |
| **R2 ★** | **Socket route drops Merkle nodes to 128-bit** = 64-bit collision bound, on the production RV64 proof (§0.2) | Pinned by nothing. Security decision, needs Mauro |
| **R3** | **FRI is still keccak-concrete** (§4.1) — a blake3 `H` silently mixes hashes at the type level and rejects every honest proof at the first FRI query | Fails loudly at test time; no guard |
| **R4** | **cuda + blake3 does not compile** (§0.3) — deliberate, via the step-0 H3 guard. Track G (§6.1) is what retires it; until then blake3 must be `cfg(not(cuda))` and the PR-time cuda job (`pr_main.yaml:282`) enforces that | Guarded at compile time (`config.rs:116-122`, `gpu_lde.rs:701…`) |
| **R5** | **Guest desync is invisible to host tests** — the TypeId specialization bypass (`platform_keccak.rs:14-21`) surfaces as in-guest proof rejection only | Documented, not tested. Stage 4 needs a guest-run oracle |
| **R6** | **#903 unmerged and unratified** — the guest syscall P-a needs is on a side branch and is the 6-round A6R variant "to be ratified in the spec before production use" | Branch `feat/blake3-accelerator` |
| **R7** | **Domain separation across the four domains.** Keccak gets none today (leaves, parents, FRI leaves and transcript are all plain keccak over distinct byte shapes). Blake3 offers tags cheaply (the socket already does this: `TAG_LFMC/LFML/LFMT`, `blake3_socket.rs:227-254`) | ⚠ **Decide explicitly.** Adding tags is a strict improvement but changes the hash; inheriting "no separation" is defensible but should be a written choice, not an oversight |
| **R8** | **`COMMITMENT_HASH` becomes a half-truth** while a blake3 `H` coexists with keccak aliases (§4.2) | Guard exists but reads the global const, not `H` |
| **R9** | **Fixture/registry regeneration** — every LFM root moves; the re-bless is legitimate here but collides with the standing "never re-bless to silence" policy unless stated | `registry.rs:5-8` |
| **R10** | **Round-count split.** With 6r the target but `blake3-6round` OFF by default (`blake3.rs:83-85`) and `make lint` not building it, the pipeline's primary arm is the one CI never compiles. Sinking the core into `crypto/crypto` widens the blind spot | `blake3_socket.rs:215` asserts single-knob; Makefile matrix **must** be extended in Stage 1 |
| **R12 ★** | **6 rounds rests on an unratified assumption.** A6R-signoff `:104-106` — 6r "is computed by nothing else in the world"; #903's own commit message says the variant "rests on the named A6R assumption … to be ratified in the spec before production use". Mauro's framing is exploratory ("to see if this works"), which is a fine reason to build it and not a reason to skip the ratification | Recorded in `thoughts/blake3/blake3-chip/IMPLEMENTATION.md` per #903; spec ratification still owed |
| **R13** | **The 6r arm has no independent KAT table.** `CANONICAL_OUT_7ROUND` exists (`blake3.rs:407`); there is no `CANONICAL_OUT_6ROUND`, so 6-round expected outputs are pinned by the host implementation alone. Track G would then be checking a device port against the same code path it was derived from | **Blocking prerequisite for §6.1** — generate and commit the 6r table first |
| **R14** | **§1.6 unanswered blocks two workstreams.** The chaining construction determines the leaf kernel loop (track G) and the emitter's flag/counter cases (Stage 5). Answering it late forces rework in both | Needs Mauro; ~60% of track G is unblocked meanwhile |
| **R11** | **Collision with in-flight D0 steps 3–4.** Both P-a and D0 add `StarkHash` instances and both touch `config.rs`, `registry.rs` and the backends directory | See below |

### On R11 — how P-a and D0 avoid colliding

They want **different instances of the same trait**, which is the good case:
D0's is cell-oriented over `LfmWord` (LFML leaves / LFMC parents, 128-bit,
socket-hosted); P-a's is byte-oriented (256-bit, general-chip-hosted). Both
keep `Node = Commitment = [u8;32]`, so neither moves the wire format.

Three shared files need sequencing rather than merging: `config.rs` (both add a
`CommitmentHash` variant and an instance), `registry.rs:158` (the H1 guard's
exhaustive match breaks for whichever lands first), and
`merkle_tree/backends/`. **Recommendation: land P-a's Stage 1 §1.2b core sink
first** — D0's Blake3 backends can then be built on the same compression
function instead of a second one, which is the same argument
`blake3_socket.rs:203-215` makes about the probe and the socket.

⚠ D0's `d0_king_gate.rs` "must compile *unchanged* across the refs being
compared — that is itself the API-stability half of the test" (`:29-33`). P-a
Stage 1's crate move must not touch the API surface that file names
(`lfm_prove`, `lfm_verify`, `build_artifacts`, `LfmWord`, `MultiProof`).

---

## 8. What I did not close

Stated so the next pass does not assume coverage:

- **Continuation chaining** (§4.4) — whether any commitment-hash-derived value
  is bound across epochs. Fold into Stage 2.
- **Whether the wrap re-checks the inner grinding nonce today** (§3). If it does
  not, that is a pre-existing hosted-verify gap to file separately.
- **The rate-4 figure in §5.2 is my arithmetic**, marked DERIVED. Stage 0
  replaces it with the repo's own closed form at no cost.
- **§1.6's construction question is open, not unverified** — it needs a decision
  from Mauro, and it blocks track G's chaining loop and Stage 5's emitter.

Closed since the first draft: the CUDA kernel inventory (§4.5/§6.1) and
`device_only_gate` (§4.5) are now ✓ VERIFIED firsthand rather than inherited;
the CI census (§4.4) is complete.
