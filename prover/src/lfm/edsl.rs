//! eDSL libraries: transcript, Merkle and FRI expressed as ordinary Rust
//! that *emits instructions*. Host-side `for` loops unroll — nothing
//! loop-shaped reaches the machine; shapes (path depths, query counts,
//! domain parameters) are compile-time constants of the emitted program.
//!
//! The Fiat–Shamir transcript here is the machine side of the protocol loop and
//! is mirrored bit-exactly by `fixture::HostSponge`. Since option B1 it is a
//! **compress chain**, specified in
//! `thoughts/shared/lfm-real-hash/transcript-spec/`; see [`SpongeVar`].

use crate::tables::types::FE;

use super::builder::{Bit, Cell, DigestVal, Ext, Felt, LfmBuilder};

/// The advance marker `"SQZ0"`, read as one little-endian `u32`.
///
/// Lane 0 of the constant cell a squeeze advances with. It distinguishes an
/// advance operand from an absorbed digest as **defence in depth only** — the
/// load-bearing absorb/squeeze separation is that the operation sequence is a
/// compile-time constant of the program (see [`SpongeVar`]).
pub const SQUEEZE_MARK: u32 = u32::from_le_bytes(*b"SQZ0");

/// The Fiat–Shamir transcript over `LFM_HASH`: a **compress chain**, state = 1
/// cell.
///
/// ```text
/// absorb(c)        state ← T(state, c)                       1 step
/// absorb2(c0, c1)  state ← T(T(state, c0), c1)               2 steps
/// squeeze()        out = state ; state ← T(state, SQ(i))     1 step
/// ```
///
/// where `T` is one `LFM_HASH` two-to-one step in the TRANSCRIPT domain
/// ([`LfmBuilder::transcript_step`]) and `SQ(i) = [SQUEEZE_MARK, i, 0, 0]`.
/// Squeeze outputs BEFORE advancing.
///
/// # Why a chain and not a sponge
///
/// This replaced an overwrite-rate duplex over a 3-cell permutation (option B1,
/// ratified 2026-08-11). A chain over a collision-resistant compression is the
/// textbook Fiat–Shamir transcript and needs no assumption beyond the one the
/// hash already carries — no T-sponge theorem, no capacity argument. It needs no
/// *permutation* either, which is what lets the machine's real hash have a
/// compress socket and no permute socket at all. Being public-coin is what makes
/// this legitimate: every absorbed value is a public commitment and every
/// squeezed value a public challenge, so there is no secret for a capacity to
/// protect.
///
/// The state is one cell = 128 bits, so ~**64-bit collision resistance** by the
/// birthday bound. That is `HASH_DIGEST_FELTS = 4` speaking, not this
/// construction: the digest already had that bound.
///
/// # Why the squeeze counter, and why it is free
///
/// The eDSL fully unrolls, so `i` is a compile-time constant and `SQ(i)` is a
/// program constant pinned by `program_id` — a constant cell was going to be
/// emitted either way, and this one carries a counter. What it buys: without it
/// a run of consecutive squeezes iterates ONE fixed public non-injective map,
/// whose functional graph an adversary can precompute — the structure the
/// FSE-2014 T-sponge attacks on GLUON-64 exploit. With it every step is a
/// different map and no single functional graph exists.
///
/// ⚠ **Squeeze runs still lose entropy, and the bound scales with the query
/// count.** A run of `k` consecutive squeezes shrinks the reachable state by
/// `−log₂ α_k` bits, `α_k ~ 2/k`: 1.7 bits at `k = 4`, 7 at `k = 256`, 15 at
/// `k = 2^16`. The counter does not change those numbers (composing distinct
/// random maps obeys the same recursion) — it removes the attack structure. The
/// FRI query loop squeezes once per query with no absorb between, so **its run
/// length IS the query count**. A program whose runs exceed `k = 2^16` must
/// revisit the analysis in the transcript spec §4.2; below that the 64-bit
/// collision bound above dominates and this changes nothing.
#[derive(Clone, Copy, Debug)]
pub struct SpongeVar {
    state: Cell,
    /// The next squeeze's index — host-side bookkeeping, so it appears in the
    /// program only as the constant it selects.
    squeeze_index: u32,
}

impl SpongeVar {
    pub fn new(b: &mut LfmBuilder) -> Self {
        SpongeVar {
            state: b.felt_const(FE::zero()).as_cell(),
            squeeze_index: 0,
        }
    }

    /// Absorb one cell: one transcript step against the current state.
    pub fn absorb(&mut self, b: &mut LfmBuilder, c: Cell) {
        self.state = b
            .transcript_step(self.state.as_digest(), c.as_digest())
            .as_cell();
    }

    /// Absorb a cell of four arbitrary FIELD ELEMENTS.
    ///
    /// Data enters the transcript the same way it enters a Merkle tree: through
    /// the LEAF encoding. The cell is hashed to a digest in the `"LFML"` domain
    /// and that digest is absorbed, so the chain binds the data up to the leaf
    /// hash's collision resistance.
    ///
    /// ⚠ **Use this for DATA and [`SpongeVar::absorb`] for DIGESTS.** Absorbing
    /// raw field elements would hand the socket lanes that are not `u32`, which
    /// under the machine's real hash is not a preference but an unprovable row
    /// (obligation O1). A transcript that absorbs commitments needs `absorb`; one
    /// that absorbs polynomial coefficients, evaluations or any other field data
    /// needs this.
    pub fn absorb_felts(&mut self, b: &mut LfmBuilder, c: Cell) {
        let acc = leaf_chain_start(b);
        let d = b.leaf(acc, c);
        self.absorb(b, d.as_cell());
    }

    /// Absorb two cells, in order. Two steps, not one: the chain takes one
    /// operand per step, and the ORDER is what the transcript binds.
    pub fn absorb2(&mut self, b: &mut LfmBuilder, c0: Cell, c1: Cell) {
        self.absorb(b, c0);
        self.absorb(b, c1);
    }

    /// The chain's state cell as it stands, WITHOUT advancing — what
    /// `AlgebraicTranscript::state` serialises, and what grinding seeds from.
    ///
    /// Observation only: nothing here absorbs or squeezes, so a later step sees
    /// the state this returned. That is the same contract
    /// `DefaultTranscript::state` has on the byte side, where production
    /// finalizes a CLONE of the hasher rather than the hasher.
    pub fn state(&self) -> Cell {
        self.state
    }

    /// Squeeze one cell: the current state, then advance past it with `SQ(i)`.
    ///
    /// Output-then-advance rather than advance-then-output, so no squeezed
    /// value is ever the state a later step absorbs into.
    pub fn squeeze_cell(&mut self, b: &mut LfmBuilder) -> Cell {
        let out = self.state;
        // `SQ(i)`, interned like every other program constant — one `LFM_CONST`
        // row per distinct squeeze index, and nothing else.
        let sq = b.digest_const([
            FE::from(u64::from(SQUEEZE_MARK)),
            FE::from(u64::from(self.squeeze_index)),
            FE::zero(),
            FE::zero(),
        ]);
        self.state = b.transcript_step(self.state.as_digest(), sq).as_cell();
        self.squeeze_index += 1;
        out
    }

    /// Squeeze an ext challenge: lanes 0–2 of a squeezed cell.
    pub fn squeeze_ext(&mut self, b: &mut LfmBuilder) -> Ext {
        let c = self.squeeze_cell(b);
        let [l0, l1, l2, _] = b.unpack(c);
        b.pack_ext(l0, l1, l2)
    }

    /// Squeeze `nbits` index bits: the canonical bit decomposition of lane 0
    /// of a squeezed cell (masking to a power-of-two bound, so no rejection
    /// loop — the convention the RV64 verifier's query sampling already uses).
    pub fn squeeze_bits(&mut self, b: &mut LfmBuilder, nbits: usize) -> Vec<Bit> {
        let c = self.squeeze_cell(b);
        let [l0, _, _, _] = b.unpack(c);
        b.bit_dec(l0, nbits)
    }
}

/// Arena words one commitment DIGEST occupies under this builder's
/// configuration — two on a byte hash, ONE on an algebraic one.
///
/// ★ The single definition of that stride. Six sites used to spell it as the
/// literal `2`, three of them carrying a comment saying "two arena words per
/// sibling IS the digest's width — when the algebraic path lands this stride
/// follows the width rather than the literal". This is that.
pub fn digest_words(b: &LfmBuilder) -> u32 {
    match b.wrap_hash() {
        WrapHash::Algebraic => 1,
        _ => 2,
    }
}

/// Read one commitment digest out of an arena at `base`, consuming
/// [`digest_words`] words.
pub fn hint_digest(b: &mut LfmBuilder, arena: super::instr::ArenaId, base: u32) -> WrapDigest {
    let cells: Vec<Cell> = (0..digest_words(b))
        .map(|i| b.hint_word(arena, base + i))
        .collect();
    WrapDigest::from_cells(&cells)
}

/// Where a leaf chain starts: the zero cell.
///
/// ⚠ **This is a chain START, not a shape HEADER.** COMMIT.md §1.3 opens the
/// chain at `[LEAF_MARK, num_cols, kind, rows_per_leaf]` so that the leaf's
/// width and element kind are bound *inside* the hash — the whole point of that
/// construction. Nothing here has a width to bind: these leaves are fixed-shape
/// by the program that builds them, exactly as they were before the chain
/// existed. The commitment layer that hashes arbitrary-width openings supplies
/// the header instead of this, and it must, or its leaves bind no shape.
///
/// Interned like every other program constant, so a program's whole leaf traffic
/// costs one `LFM_CONST` row for this.
pub fn leaf_chain_start(b: &mut LfmBuilder) -> DigestVal {
    b.digest_const([FE::zero(); 4])
}

/// The Merkle LEAF digest of a pair of data cells — eight field elements.
///
/// **Two compressions, and the shape is the point:** the cells are absorbed in
/// order into one `"LFML"` chain, four felts per hash, each step chaining the
/// last. So a leaf's *data* never enters a compress as a digest, and a parent
/// never enters as data — which is what makes an internal node un-replayable as
/// a leaf regardless of the tree's depth (obligation O5, discharged by the tag
/// rather than by fixed depth).
///
/// It cost THREE while the accumulator was not in the message: each cell hashed
/// to its own leaf digest and an `"LFMC"` parent folded the two. Absorbing and
/// chaining in the same compression is what took leaf absorption from 2 felts
/// per hash to 4 (COMMIT.md §1.4.1), and leaf absorption is ~70% of a recursion
/// tower node's bill. The chain binds the cells' ORDER for free, where the fold
/// bound it through the parent's operand order.
///
/// Before either, this was `compress(cell0, cell1)` — one compression that
/// treated arbitrary field elements as if they were `u32` digest lanes. Under a
/// hash whose lanes must BE `u32` that is not merely undesirable, it is
/// unprovable, which is why FRI data could not be hashed at all before the leaf
/// mode existed.
pub fn leaf_hash_pair(b: &mut LfmBuilder, c0: Cell, c1: Cell) -> DigestVal {
    let acc = leaf_chain_start(b);
    let d0 = b.leaf(acc, c0);
    b.leaf(d0, c1)
}

/// Walk one Merkle authentication path. `bits` are the leaf-index bits
/// low-to-high (level 0 first): bit = 0 ⇒ the current node is the LEFT
/// child. Sibling digests come as (arena-hinted) cells; every hinted value
/// ends up inside a `compress`, which is what authenticates it.
pub fn merkle_walk(
    b: &mut LfmBuilder,
    leaf: DigestVal,
    bits: &[Bit],
    siblings: &[Cell],
) -> DigestVal {
    assert_eq!(bits.len(), siblings.len(), "one sibling per level");
    let mut current = leaf;
    for (bit, sibling) in bits.iter().zip(siblings) {
        let (left, right) = b.select(*bit, current.as_cell(), *sibling);
        current = b.compress(left.as_digest(), right.as_digest());
    }
    current
}

// ===================== production keccak Merkle =====================

/// A 32-byte digest as it lives in the machine: two words of four `u32` halves
/// each, half `h` carrying digest bytes `4h..4h+4`.
///
/// ★ **One type for both production hashes, and that is load-bearing rather
/// than a coincidence.** `keccak256`'s digest is the state's first 32 bytes and
/// `Blake3Chain`'s is `out[0..8]` little-endian, so under both hashes digest
/// byte `j` is byte `j % 4` of half `j / 4`. Every arena stride, every
/// `siblings: Vec<...>` width and every proof shape is therefore identical
/// across the two, which is what lets the constructions below be parameterized
/// on the hash rather than duplicated per hash. The `LFM_HASH` socket would
/// NOT have this property — its digest is 128 bits, one cell.
/// The largest number of cells any wrap digest occupies — two, for the byte
/// hashes' 32 bytes.
pub const MAX_DIGEST_CELLS: usize = 2;

/// A commitment digest as it lives in the machine, carrying its own WIDTH.
///
/// ★ **The width is data, not a constant, and that is the whole point.** A byte
/// hash's digest is 32 bytes — two words of four `u32` halves. An **algebraic**
/// hash's digest is four field elements: **ONE cell**. Both flow through the
/// same walks, comparisons and openings, so the shape travels with the value
/// rather than being written into every call site.
///
/// Before this, the type was `[Cell; 2]` and five root comparisons plus two
/// Merkle walks each spelled the count out as `d[0]`, `d[1]`. Those are the
/// sites a one-cell digest would have silently half-worked at.
///
/// Derefs to `[Cell]`, so `d[0]`, `d.len()` and `d.iter()` keep working and the
/// forty-odd sites that only carry a digest around are untouched.
#[derive(Debug, Clone, Copy)]
pub struct WrapDigest {
    cells: [Cell; MAX_DIGEST_CELLS],
    len: u8,
}

impl WrapDigest {
    /// A two-cell digest — the byte hashes' 32 bytes.
    pub fn from_pair(a: Cell, b: Cell) -> Self {
        WrapDigest {
            cells: [a, b],
            len: 2,
        }
    }

    /// A ONE-cell digest — an algebraic hash's four felts.
    ///
    /// The unused slot repeats the cell rather than holding a sentinel: nothing
    /// reads past `len`, and a repeated handle cannot be mistaken for a real
    /// second word the way a zero cell could.
    pub fn from_cell(c: Cell) -> Self {
        WrapDigest {
            cells: [c, c],
            len: 1,
        }
    }

    /// A digest of the same width as `cells`, from a slice.
    ///
    /// Used by the Merkle walks, which rebuild a digest cell by cell after
    /// selecting each against its sibling on the level's bit.
    pub fn from_cells(cells: &[Cell]) -> Self {
        assert!(
            (1..=MAX_DIGEST_CELLS).contains(&cells.len()),
            "a digest is one or two cells, got {}",
            cells.len()
        );
        let mut out = [cells[0]; MAX_DIGEST_CELLS];
        out[..cells.len()].copy_from_slice(cells);
        WrapDigest {
            cells: out,
            len: cells.len() as u8,
        }
    }

    /// The cells this digest actually occupies.
    pub fn cells(&self) -> &[Cell] {
        &self.cells[..self.len as usize]
    }
}

impl core::ops::Deref for WrapDigest {
    type Target = [Cell];
    fn deref(&self) -> &[Cell] {
        self.cells()
    }
}

/// A 32-byte keccak digest. See [`WrapDigest`].
pub type KeccakDigest = WrapDigest;

/// A 32-byte BLAKE3 digest. See [`WrapDigest`].
pub type Blake3Digest = WrapDigest;

/// Halves in a 32-byte digest.
pub const DIGEST_HALVES: usize = 8;

/// The eight halves of a BYTE digest, ready to be streamed into another hash.
///
/// ⛔ **NOT hash-agnostic, whatever an earlier version of this comment said.**
/// It indexes `d[1]`, and an algebraic `WrapDigest` is ONE cell whose second
/// slot REPEATS the first ([`WrapDigest::from_cell`]) — so on that arm this
/// would return `lo ‖ lo`: eight plausible halves, four of them a duplicate,
/// with nothing to notice. The assert below is what makes that loud instead.
///
/// ✓ VERIFIED unreachable on the algebraic arm today: the only caller is
/// `parent_stream`, inside [`WrapHash::hash_pair`]'s `Some(byte_hash)` arm. The
/// assert is for the NEXT caller, since the stale claim is exactly what would
/// invite one. The felt view of an algebraic digest is
/// `epoch::RootCells::byte_halves`, which renders each felt big-endian rather
/// than pretending the cells are `u32` lanes.
pub fn digest_halves(b: &mut LfmBuilder, d: WrapDigest) -> [Felt; DIGEST_HALVES] {
    assert_eq!(
        d.cells().len(),
        DIGEST_HALVES / 4,
        "digest_halves is the BYTE digest's view; an algebraic digest's felts \
         are not `u32` halves and go through RootCells::byte_halves"
    );
    let lo = b.unpack(d[0]);
    let hi = b.unpack(d[1]);
    core::array::from_fn(|h| if h < 4 { lo[h] } else { hi[h - 4] })
}

/// The Merkle LEAF hash of a row pair, in the production commitment layout.
///
/// `values` is `evaluations ‖ evaluations_sym` — the two bit-reversed rows the
/// leaf covers, each written column by column. Every element is a base field
/// element rendered as its canonical `u64` in BIG-endian bytes
/// (`FieldElement<GoldilocksField>::stream_bytes`), so each costs one
/// [`super::transcript_replay::felt_be_halves`]: one `LFM_BITDEC` row and 64
/// `LFM_BALU` rows. The hash itself is `keccak256` over `8 · values.len()`
/// bytes.
///
/// ## The byteswapping is NOT what this costs — measured, against expectation
///
/// A `c`-column table gives `2c` elements, so `2c` decompositions and `128c`
/// ALU rows against only `⌈(16c + 1) / 136⌉` permutations. On row counts the
/// byteswapping looks overwhelming, which is what the R1f handoff predicted.
/// That reading is wrong: rows of different chips are not comparable units. An
/// `LFM_BALU` row carries 4 non-preprocessed columns, while one permutation
/// expands into 24 `KECCAK_RND` rounds of 1480 columns — so in main-trace cells
/// a permutation costs 113 byteswaps, and the hash term dominates at every
/// table width. `machine_tests::keccak_merkle_opening_cost` measures it and
/// asserts the inequality holds.
///
/// The swap is real work regardless, and it is not avoidable by pre-swapping in
/// the arena: the same opened values are consumed as FIELD ELEMENTS by the FRI
/// algebra and as BYTES by this hash, so something has to connect the two
/// representations, and only the machine can do it in a way the proof binds.
pub fn keccak_leaf_hash(b: &mut LfmBuilder, values: &[Felt]) -> KeccakDigest {
    WrapHash::Keccak.leaf_hash(b, values)
}

/// The byte stream a Merkle LEAF hashes, and the shape it covers.
///
/// Split out of [`WrapHash::leaf_hash`] so the group's boundary is explicit
/// data rather than a length the hash call happens to receive: the batched
/// (mixed-height, per-matrix) leaf the MMCS lane needs adds a header to
/// exactly this, and can do so without re-cutting the emitter.
///
/// Every element is a base field element rendered as its canonical `u64` in
/// BIG-endian bytes (`FieldElement<GoldilocksField>::stream_bytes`), so each
/// costs one [`super::transcript_replay::felt_be_halves`]: one `LFM_BITDEC` row
/// and 64 `LFM_BALU` rows. The byte content is the hash's input and is
/// identical under both wrap hashes — only the framing above it differs.
pub fn leaf_stream(b: &mut LfmBuilder, values: &[Felt]) -> (Vec<Felt>, usize) {
    use super::keccak_host::BYTES_PER_HALF;
    use super::transcript_replay::felt_be_halves;

    assert!(!values.is_empty(), "a leaf covers at least one column");
    let mut stream = Vec::with_capacity(2 * values.len());
    for v in values {
        stream.extend(felt_be_halves(b, *v));
    }
    let len_bytes = BYTES_PER_HALF * stream.len();
    (stream, len_bytes)
}

/// Walk one Merkle authentication path under the PRODUCTION hash.
///
/// This is the keccak counterpart of [`merkle_walk`], and the two are not
/// interchangeable: `merkle_walk` compresses with `LFM_HASH`/`TestPermutation`,
/// the deliberately non-cryptographic Milestone-C placeholder, so it can only
/// ever authenticate the Milestone-C fixture tree. Production trees are keccak
/// throughout, and this is the walk that authenticates them.
///
/// `bits` are the leaf index low-to-high, level 0 first; `bit = 0` means the
/// current node is the LEFT child, matching `verify_merkle_path_from_leaf_hash`
/// (`index % 2 == 0 ⇒ hash(current, sibling)`).
///
/// ## The parent step
///
/// `hash_new_parent(l, r) = keccak(l ‖ r)` — 64 bytes, no domain separation and
/// no ordering flag, so the ordering is carried entirely by the index bit. 64
/// bytes sits inside one 136-byte rate block, so a level is exactly ONE
/// permutation. Per level the machine pays two `Select`s (a digest is two words
/// and both must swap on the same bit), four `Unpack`s and that permutation.
pub fn keccak_merkle_walk(
    b: &mut LfmBuilder,
    leaf: KeccakDigest,
    bits: &[Bit],
    siblings: &[KeccakDigest],
) -> KeccakDigest {
    WrapHash::Keccak.merkle_walk(b, leaf, bits, siblings)
}

/// The production Merkle PARENT hash: `keccak(left ‖ right)`.
///
/// `hash_new_parent` streams the two 32-byte nodes into one digest with no
/// domain separation and no ordering flag, so 64 bytes sit inside a single
/// 136-byte rate block and a parent is exactly ONE permutation.
///
/// This is the step [`keccak_merkle_walk`] performs once per level after its
/// `Select`, and the step a whole-tree build performs once per internal node
/// with no `Select` at all — a tree's child ORDER is known when the program is
/// emitted, so there is no bit to swap on. Keeping the two callers on one
/// primitive is what makes "the walk and the build hash the same way" a
/// property of the code rather than of a comment.
pub fn keccak_hash_pair(
    b: &mut LfmBuilder,
    left: KeccakDigest,
    right: KeccakDigest,
) -> KeccakDigest {
    WrapHash::Keccak.hash_pair(b, left, right)
}

/// The 16 halves a BYTE-hash Merkle PARENT hashes: `left ‖ right`, 64 bytes.
///
/// ⛔ **Byte arms only, and this doc used to say "hash-agnostic".** It is
/// reached solely from [`WrapHash::hash_pair`]'s `Some(byte_hash)` arm, and it
/// goes through [`digest_halves`], which is the byte digest's view — an
/// algebraic digest has no `u32` halves to stream. The algebraic parent is
/// `algebraic_hash_pair`, a compress over cells, and it builds no stream at all.
///
/// Across the TWO byte hashes it genuinely is uniform: `hash_new_parent`
/// streams the two 32-byte nodes with no domain separation and no ordering
/// flag under either. What differs is what 64 bytes COSTS — one keccak
/// permutation (inside the 136-byte rate) and one BLAKE3 compression (exactly
/// one 64-byte block). Both are one invocation, which is why the parent layer
/// is 1:1 across that switch and the whole win there is per-compression cost.
/// ⚖ One invocation is also what an algebraic parent costs (8 felts fills the
/// rate exactly), so the LEVEL count is uniform across all three — it is only
/// the byte STREAM that is not.
fn parent_stream(b: &mut LfmBuilder, left: WrapDigest, right: WrapDigest) -> Vec<Felt> {
    let left_halves = digest_halves(b, left);
    let right_halves = digest_halves(b, right);
    let mut stream = Vec::with_capacity(2 * DIGEST_HALVES);
    stream.extend(left_halves);
    stream.extend(right_halves);
    stream
}

/// Build a whole Merkle TREE bottom-up and return its root.
///
/// The counterpart of [`keccak_merkle_walk`]: the walk authenticates ONE leaf
/// against a root it is given, this CONSTRUCTS the root from every leaf. A
/// derivation needs the second — there is no root to authenticate against,
/// producing it is the point.
///
/// Cost is `leaves − 1` permutations on top of the leaves' own, so a tree over
/// `L` leaves is `2L − 1` permutations in total.
///
/// ## Power-of-two leaves
///
/// `MerkleTree::build_from_hashed_leaves` runs `complete_until_power_of_two`
/// first, which pads by REPEATING the last leaf. This asserts a power of two
/// instead of emitting that padding: leaf counts here are shape (an LDE row
/// count over `ROWS_PER_LEAF`), so a non-power-of-two is a caller bug rather
/// than a case to handle, and emitting duplicate-leaf padding no production
/// commitment can reach would be dead program text.
pub fn keccak_merkle_tree_root(b: &mut LfmBuilder, leaves: &[KeccakDigest]) -> KeccakDigest {
    WrapHash::Keccak.merkle_tree_root(b, leaves)
}

// ===================== the wrap's commitment hash =====================

/// Which byte hash the wrap's commitment layer runs.
///
/// ★ **A parameter, not a switch.** The whole construction layer below —
/// leaves, parents, path walks, whole-tree builds — is written ONCE and
/// dispatches only at the two places a hash actually appears: hashing a byte
/// stream, and hashing a 64-byte parent. That is possible because
/// [`WrapDigest`] is the same shape under both, so no call site's types move.
///
/// Two consequences worth stating because they are the reason for the shape:
///
/// - A batched (mixed-height, per-matrix) leaf adds ONE construction, not one
///   per hash — which is what the MMCS lane's emitter step needs.
/// - Switching the wrap's hash is a value flowing through the emitters, so an
///   honest-path control (keccak still proves and verifies through the same
///   code) is a real control rather than a different code path.
///
/// [`Keccak`](WrapHash::Keccak) is the **unset** value, not the production one
/// — see [`WrapHash::production`] and the header of `programs.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WrapHash {
    #[default]
    Keccak,
    Blake3,
    /// ★ **One variant for the whole ALGEBRAIC family**, and that is the point.
    ///
    /// RPO, RPX and Poseidon share this arm because they share everything the
    /// EMITTER cares about: state 12, rate 8, capacity 4, and a **one-cell
    /// four-felt digest** against the byte hashes' two-cell 32-byte one. Which
    /// permutation the emitted socket rows actually prove is `HasherKind`,
    /// chosen when the AIR set is built — an orthogonal axis. So a fourth
    /// algebraic candidate needs no emitter work at all.
    ///
    /// ⚠ The byte-stream constructions (`hash_bytes`, `hash_bytes_with_rev`)
    /// have **no meaning here**: what they hash IS felts, serialised only
    /// because the incumbent hashes are byte-oriented. On this arm the
    /// serialisation is deleted at the call site rather than reimplemented.
    Algebraic,
}

impl WrapHash {
    /// ★ The wrap hash that matches the host's commitment configuration.
    ///
    /// **Any wrap program that AUTHENTICATES a host proof must hash the way the
    /// host committed.** Its Merkle walks re-derive roots the host built and its
    /// leaf hashes reproduce leaves the host hashed; under the wrong hash none
    /// of that reconstructs, and the leg is verifying nothing.
    ///
    /// ⚠ **It does not fail as a wrong digest.** The walk produces a root
    /// matching nothing, the leg then inverts a difference that should have been
    /// non-zero, and the executor reports `DivByZero` at some address. That is
    /// the diagnostic signature of a leg left on the default — and it is why
    /// this is a named function rather than a constant each site spells out,
    /// because the failure names neither the hash nor the site.
    ///
    /// The match is exhaustive on purpose: a third commitment hash cannot be
    /// added without deciding what the in-machine verifier does about it, the
    /// same tripwire `commitment_hash_tag` carries for program identity.
    ///
    /// **Not for programs that are ABOUT a hash.** The R1b/R1c/R1d keccak
    /// instruments and `program_id_program_source` name their hash directly and
    /// must keep doing so.
    pub const fn production() -> Self {
        match crate::hash_pin::BLOCK_COMMITMENT_HASH {
            stark::config::CommitmentHash::Keccak256 => WrapHash::Keccak,
            stark::config::CommitmentHash::Blake3 => WrapHash::Blake3,
            // ★ Three commitment hashes, ONE emitter arm. The permutation is
            // `HasherKind`, an orthogonal axis chosen when the AIR set is built.
            stark::config::CommitmentHash::Rpo256
            | stark::config::CommitmentHash::Rpx256
            | stark::config::CommitmentHash::Poseidon => WrapHash::Algebraic,
        }
    }

    /// This configuration's BYTE-stream hash, when it has one.
    ///
    /// ★ `None` on the algebraic arm is a FACT about that arm, not an error to
    /// report. A byte-stream hash returns a 32-byte digest — two cells — and an
    /// algebraic digest is four felts, one cell; there is no value of the return
    /// type that would be correct. The "rev" half is worse still: a BYTE
    /// reversal is not an operation on a field element at all.
    ///
    /// ⚠ **This replaces an emit-time panic, and the replacement is the point.**
    /// While the byte constructions lived on [`WrapHash`] itself they needed an
    /// algebraic arm, and that arm could only panic. The panic was recorded as an
    /// INTERIM whose exit was "the algebraic path becomes UNABLE to reach a
    /// byte-digest API, and then the panic is DELETED rather than kept as a
    /// belt" — a retained defensive panic behind a type-level impossibility being
    /// exactly the dead production panic this repo's no-production-panic policy
    /// exists to prevent. Moving the constructions to [`ByteWrapHash`] is that
    /// exit: an algebraic configuration cannot name them, so there is nothing
    /// left to defend against.
    ///
    /// Every caller matches on this, and the `None` arm IS that caller's
    /// algebraic implementation — so the exhaustiveness that used to be an
    /// assertion is now the control flow.
    pub const fn byte_hash(self) -> Option<ByteWrapHash> {
        match self {
            WrapHash::Keccak => Some(ByteWrapHash::Keccak),
            WrapHash::Blake3 => Some(ByteWrapHash::Blake3),
            WrapHash::Algebraic => None,
        }
    }

    /// The Merkle LEAF hash of a row pair, in the production commitment layout.
    ///
    /// See [`leaf_stream`] for what the bytes are and what they cost; this adds
    /// only the hash. The rate penalty the campaign prices lands here and
    /// nowhere else: a leaf absorbs 136 bytes per keccak permutation against 64
    /// per BLAKE3 compression.
    pub fn leaf_hash(self, b: &mut LfmBuilder, values: &[Felt]) -> WrapDigest {
        match self.byte_hash() {
            None => Self::algebraic_leaf_hash(b, values),
            Some(h) => {
                let (stream, len_bytes) = leaf_stream(b, values);
                h.hash_bytes(b, &stream, len_bytes)
            }
        }
    }

    /// ★ The ALGEBRAIC parent: one `compress` row, one cell out.
    ///
    /// The socket primitive `b.compress` already exists and is gated; this is
    /// the wrap-world name for it. Under RPO the compress domain is zero, so
    /// this is literally `Rpo256::merge` and matches
    /// `algebraic_commit`'s `parent` on the host.
    fn algebraic_hash_pair(b: &mut LfmBuilder, left: WrapDigest, right: WrapDigest) -> WrapDigest {
        debug_assert_eq!(left.len(), 1, "an algebraic digest is one cell");
        debug_assert_eq!(right.len(), 1, "an algebraic digest is one cell");
        WrapDigest::from_cell(
            b.compress(left[0].as_digest(), right[0].as_digest())
                .as_cell(),
        )
    }

    /// ★ The ALGEBRAIC leaf: the rate-8 OVERWRITE DUPLEX.
    ///
    /// Eight fresh felts per permutation, against the socket leaf chain's four —
    /// the convention this lane priced at 25% of the aggregation program. Each
    /// block overwrites the two rate cells and carries the capacity cell from
    /// the previous permutation, which is exactly `MODE_P`: three cells in,
    /// three out, already in the frozen bus contract.
    ///
    /// ⚠ Every constant comes from the ONE rule
    /// (`algebraic_commit::leaf_capacity`), never restated here — the capacity
    /// is program data under `MODE_P`, so a second definition of it is a root
    /// the host cannot reproduce. This mirrors
    /// `algebraic_commit::sponge_leaf` step for step.
    fn algebraic_leaf_hash(b: &mut LfmBuilder, values: &[Felt]) -> WrapDigest {
        use super::algebraic_commit::leaf_capacity;

        let zero = b.felt_const(FE::zero());
        let zero_cell = b.digest_const([FE::zero(); 4]).as_cell();

        // Four felts per cell, the tail zero-padded.
        let cells: Vec<Cell> = values
            .chunks(4)
            .map(|c| {
                let lanes: [Felt; 4] = core::array::from_fn(|i| c.get(i).copied().unwrap_or(zero));
                b.pack_word(lanes)
            })
            .collect();

        // The capacity carries the padding flag and the leaf domain.
        let mut cap = b.digest_const(leaf_capacity(values.len())).as_cell();
        if cells.is_empty() {
            // An empty leaf never permutes — the digest is the initial rate,
            // which is zero. `sponge_leaf` returns the same.
            return WrapDigest::from_cell(zero_cell);
        }

        let mut digest = zero_cell;
        for block in cells.chunks(2) {
            let rate0 = block[0];
            let rate1 = block.get(1).copied().unwrap_or(zero_cell);
            let out = b.permute([rate0, rate1, cap]);
            digest = out[0];
            cap = out[2];
        }
        WrapDigest::from_cell(digest)
    }

    /// The production Merkle PARENT hash: `hash(left ‖ right)`.
    ///
    /// One invocation under every hash — 64 bytes fits inside keccak's 136-byte
    /// rate, IS exactly one BLAKE3 block, and an algebraic parent's two digest
    /// cells fill the rate-8 sponge exactly. ⚠ Three arms, not two: the byte
    /// ones build a stream through `parent_stream`, the algebraic one
    /// compresses cells and builds no stream. This is the step
    /// [`WrapHash::merkle_walk`] performs once per level after its `Select`, and
    /// the step a whole-tree build performs once per internal node with no
    /// `Select` at all: a tree's child ORDER is known when the program is
    /// emitted. Keeping the two callers on one primitive is what makes "the walk
    /// and the build hash the same way" a property of the code.
    pub fn hash_pair(self, b: &mut LfmBuilder, left: WrapDigest, right: WrapDigest) -> WrapDigest {
        match self.byte_hash() {
            None => Self::algebraic_hash_pair(b, left, right),
            Some(h) => {
                let stream = parent_stream(b, left, right);
                h.hash_bytes(b, &stream, 2 * COMMITMENT_BYTES)
            }
        }
    }

    /// Walk one Merkle authentication path under the production hash.
    ///
    /// `bits` are the leaf index low-to-high, level 0 first; `bit = 0` means the
    /// current node is the LEFT child, matching
    /// `verify_merkle_path_from_leaf_hash` (`index % 2 == 0 ⇒ hash(current,
    /// sibling)`). Per level: two `Select`s (a digest is two words and both must
    /// swap on the same bit), four `Unpack`s and one hash invocation.
    ///
    /// A loop over the siblings it is given, with no arity or height assumption
    /// beyond the one-sibling-per-level assert — deliberately, so a batched path
    /// that injects at mixed heights extends this rather than replacing it.
    ///
    /// Not interchangeable with [`merkle_walk`], which compresses with
    /// `LFM_HASH`/`TestPermutation`, the deliberately non-cryptographic
    /// Milestone-C placeholder.
    pub fn merkle_walk(
        self,
        b: &mut LfmBuilder,
        leaf: WrapDigest,
        bits: &[Bit],
        siblings: &[WrapDigest],
    ) -> WrapDigest {
        assert_eq!(bits.len(), siblings.len(), "one sibling per level");
        let mut current = leaf;
        for (bit, sibling) in bits.iter().zip(siblings) {
            // ★ EVERY cell of the digest swaps on the SAME bit — a loop rather
            // than two hard-coded halves, so a ONE-cell algebraic digest costs
            // ONE select per level where a byte digest costs two.
            debug_assert_eq!(
                current.len(),
                sibling.len(),
                "a node and its sibling must be the same width"
            );
            let mut left = [current[0]; MAX_DIGEST_CELLS];
            let mut right = [current[0]; MAX_DIGEST_CELLS];
            for k in 0..current.len() {
                let (l, r) = b.select(*bit, current[k], sibling[k]);
                left[k] = l;
                right[k] = r;
            }
            let n = current.len();
            current = self.hash_pair(
                b,
                WrapDigest::from_cells(&left[..n]),
                WrapDigest::from_cells(&right[..n]),
            );
        }
        current
    }

    /// Build a whole Merkle TREE bottom-up and return its root.
    ///
    /// The counterpart of [`WrapHash::merkle_walk`]: the walk authenticates ONE
    /// leaf against a root it is given, this CONSTRUCTS the root from every
    /// leaf. A derivation needs the second — there is no root to authenticate
    /// against, producing it is the point. Cost is `leaves − 1` parent hashes.
    ///
    /// `MerkleTree::build_from_hashed_leaves` runs `complete_until_power_of_two`
    /// first, which pads by REPEATING the last leaf. This asserts a power of two
    /// instead of emitting that padding: leaf counts here are shape (an LDE row
    /// count over `ROWS_PER_LEAF`), so a non-power-of-two is a caller bug, and
    /// emitting duplicate-leaf padding no production commitment can reach would
    /// be dead program text.
    pub fn merkle_tree_root(self, b: &mut LfmBuilder, leaves: &[WrapDigest]) -> WrapDigest {
        assert!(!leaves.is_empty(), "a tree has at least one leaf");
        assert!(
            leaves.len().is_power_of_two(),
            "leaf counts are shape and must be a power of two; production would \
             pad by repeating the last leaf and no caller here needs that"
        );
        let mut level = leaves.to_vec();
        while level.len() > 1 {
            level = level
                .chunks_exact(2)
                .map(|pair| self.hash_pair(b, pair[0], pair[1]))
                .collect();
        }
        level[0]
    }
}

/// Bytes in a commitment / Merkle node.
pub const COMMITMENT_BYTES: usize = 32;

/// Assert two words are equal, lane by lane (2 unpacks + 4 lowered asserts).
pub fn assert_word_eq(b: &mut LfmBuilder, x: Cell, y: Cell) {
    let yl = b.unpack(y);
    assert_word_eq_lanes(b, x, &yl);
}

/// Assert a word equals four already-unpacked lanes (hoist the reference
/// word's unpack out of a loop — e.g. one root compared per query).
/// ★ Assert a whole DIGEST equals a root's lanes — the digest-shaped form of
/// [`assert_word_eq_lanes`].
///
/// Five sites used to spell this as two indexed calls, `root[0]`/`root[1]`,
/// which writes the digest's cell COUNT into every one of them. Looping here
/// concentrates that count in one place, so a digest of a different width
/// (an algebraic root is ONE cell of four felts, not two of 32 bytes) changes
/// this function rather than five call sites.
///
/// Behaviour is identical for a two-cell digest; this is a refactor, not a
/// change, and the gates that covered those five sites still cover it.
pub fn assert_digest_eq_lanes(b: &mut LfmBuilder, d: WrapDigest, lanes: &[[Felt; 4]]) {
    assert_eq!(
        d.len(),
        lanes.len(),
        "a digest and the lanes it is compared against must have the same width"
    );
    for (cell, word) in d.iter().zip(lanes.iter()) {
        assert_word_eq_lanes(b, *cell, word);
    }
}

pub fn assert_word_eq_lanes(b: &mut LfmBuilder, x: Cell, y_lanes: &[Felt; 4]) {
    let xl = b.unpack(x);
    for i in 0..4 {
        b.assert_eq(xl[i], y_lanes[i]);
    }
}

/// `scale · Π factors[i]^{bits[i]}` — one Select + one Mul per bit. Used to
/// derive domain points (and their inverses) from query-index bits; the
/// factors are program constants, so nothing here touches an arena.
pub fn pow_bits(b: &mut LfmBuilder, bits: &[Bit], factors: &[FE], scale: FE) -> Felt {
    assert_eq!(bits.len(), factors.len());
    let mut acc = b.felt_const(scale);
    for (bit, factor) in bits.iter().zip(factors) {
        let one = b.felt_const(FE::one());
        let f = b.felt_const(*factor);
        let (chosen, _) = b.select(*bit, one.as_cell(), f.as_cell());
        acc = b.mul(acc, Felt(chosen.0));
    }
    acc
}

/// `Σ_i coeffs[i]·α^i` over ext, coeffs given low-to-high (base cells are
/// valid ext operands). One `MulAdd` per coefficient — the Horner shape.
pub fn horner_ext(b: &mut LfmBuilder, alpha: Ext, coeffs_low_to_high: &[Ext]) -> Ext {
    let mut iter = coeffs_low_to_high.iter().rev();
    let mut acc = *iter.next().expect("at least one coefficient");
    for c in iter {
        acc = b.emul_add(acc, alpha, *c);
    }
    acc
}

/// One unnormalized FRI fold — our production convention exactly:
/// `(lo + hi) + inv_x·ζ·(lo − hi)` (the missing ½ is absorbed into the
/// terminal polynomial).
pub fn fri_fold(b: &mut LfmBuilder, lo: Ext, hi: Ext, zeta: Ext, inv_x: Felt) -> Ext {
    let sum = b.eadd(lo, hi);
    let diff = b.esub(lo, hi);
    let zd = b.emul(zeta, diff);
    let scaled = b.emul_base(zd, inv_x);
    b.eadd(sum, scaled)
}

/// A configuration that hashes a BYTE STREAM — the two incumbents, and only
/// them.
///
/// ★ It exists so that "this hash has a 32-byte digest" is a TYPE rather than a
/// runtime check. [`WrapHash::byte_hash`] is the only way in, and it cannot
/// produce one from an algebraic configuration — so the byte-stream
/// constructions below are unreachable from the algebraic path by construction,
/// which is what let the emit-time panic that used to guard them be deleted
/// rather than retained.
///
/// The Merkle constructions (`leaf_hash`, `hash_pair`, `merkle_walk`) stay on
/// [`WrapHash`] and are NOT split, because they are shape-generic: they take and
/// return [`WrapDigest`], which carries its own width. Only the byte-STREAM
/// entries have no algebraic meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteWrapHash {
    Keccak,
    Blake3,
}

impl ByteWrapHash {
    /// The hash of a byte stream supplied as `u32`-half felts.
    ///
    /// Both hashes take the SAME packing — four bytes per felt, little-endian
    /// ([`super::keccak_host::pack_stream`]) — because a BLAKE3 message word is
    /// itself a little-endian `u32` of four consecutive message bytes. So a
    /// stream built for one is a stream for the other, and only the framing
    /// above it changes: 136-byte rate blocks with `pad10*1` against 64-byte
    /// blocks with zero padding and an explicit `block_len`.
    pub fn hash_bytes(self, b: &mut LfmBuilder, stream: &[Felt], len_bytes: usize) -> WrapDigest {
        match self {
            ByteWrapHash::Keccak => {
                let d = keccak256(b, stream, len_bytes);
                WrapDigest::from_pair(d[0], d[1])
            }
            ByteWrapHash::Blake3 => {
                let d = blake3_256(b, stream, len_bytes);
                WrapDigest::from_pair(d[0], d[1])
            }
        }
    }

    /// [`ByteWrapHash::hash_bytes`] returning BOTH the digest and its byte-REVERSED
    /// form — what `DefaultTranscript::sample()` returns and re-absorbs.
    ///
    /// Free under both hashes, for the same reason: the bus recomposes each
    /// `u32` lane from four byte columns, so the reversal is a second `Linear`
    /// over the same columns (`layout::keccak::REV_ADDR0`,
    /// `layout::blake3::REV_ADDR0`).
    pub fn hash_bytes_with_rev(
        self,
        b: &mut LfmBuilder,
        stream: &[Felt],
        len_bytes: usize,
    ) -> (WrapDigest, [Cell; 2]) {
        match self {
            ByteWrapHash::Keccak => {
                let (d, rev) = keccak256_with_rev(b, stream, len_bytes);
                (WrapDigest::from_pair(d[0], d[1]), rev)
            }
            ByteWrapHash::Blake3 => {
                let (d, rev) = blake3_256_with_rev(b, stream, len_bytes);
                (WrapDigest::from_pair(d[0], d[1]), rev)
            }
        }
    }
}

// ============ the configured hash, as free functions ============
//
// Every construction below reads `b.wrap_hash()`. Call sites take no hash
// parameter, so there is no site to forget at the flip — and the emitters that
// must NOT follow the configuration (`programs::emit_program_id`, the R1c
// instruments) keep naming `keccak256` and friends directly, which makes a grep
// for the pinned hash in `lfm/` return exactly the deliberate exceptions.

/// [`ByteWrapHash::hash_bytes`] under a hash the CALLER has already established
/// is a byte hash.
///
/// ⚠ It takes the hash rather than reading `b.wrap_hash()`, and that asymmetry
/// with the other free functions here is deliberate: the rest are total over
/// every configuration, this one is not. A caller reaches it by matching on
/// [`WrapHash::byte_hash`], which is where its algebraic case is handled — so
/// the parameter is the proof that the case was handled.
pub fn wrap_hash_bytes(
    b: &mut LfmBuilder,
    h: ByteWrapHash,
    stream: &[Felt],
    len_bytes: usize,
) -> WrapDigest {
    h.hash_bytes(b, stream, len_bytes)
}

/// [`ByteWrapHash::hash_bytes_with_rev`], on the same terms as
/// [`wrap_hash_bytes`].
pub fn wrap_hash_bytes_with_rev(
    b: &mut LfmBuilder,
    h: ByteWrapHash,
    stream: &[Felt],
    len_bytes: usize,
) -> (WrapDigest, [Cell; 2]) {
    h.hash_bytes_with_rev(b, stream, len_bytes)
}

/// [`WrapHash::leaf_hash`] under the builder's configured hash.
pub fn wrap_leaf_hash(b: &mut LfmBuilder, values: &[Felt]) -> WrapDigest {
    let h = b.wrap_hash();
    h.leaf_hash(b, values)
}

/// [`WrapHash::hash_pair`] under the builder's configured hash.
pub fn wrap_hash_pair(b: &mut LfmBuilder, left: WrapDigest, right: WrapDigest) -> WrapDigest {
    let h = b.wrap_hash();
    h.hash_pair(b, left, right)
}

/// [`WrapHash::merkle_walk`] under the builder's configured hash.
pub fn wrap_merkle_walk(
    b: &mut LfmBuilder,
    leaf: WrapDigest,
    bits: &[Bit],
    siblings: &[WrapDigest],
) -> WrapDigest {
    let h = b.wrap_hash();
    h.merkle_walk(b, leaf, bits, siblings)
}

/// [`WrapHash::merkle_tree_root`] under the builder's configured hash.
pub fn wrap_merkle_tree_root(b: &mut LfmBuilder, leaves: &[WrapDigest]) -> WrapDigest {
    let h = b.wrap_hash();
    h.merkle_tree_root(b, leaves)
}

// ===================== small arithmetic predicates =====================

/// `1` if `x == 0`, else `0`, for an `x` the caller knows is below `2^nbits`.
///
/// Hint-free by necessity. The textbook `1 − x·x⁻¹` needs an inverse witness,
/// and the only witness channel this machine has is an arena — whose standing
/// rule is that everything hinted is transitively hash-authenticated, which an
/// inverse is not. So the DECOMPOSITION is the witness: `LFM_BITDEC` proves the
/// bits really are `x`'s, and `Π (1 − bᵢ)` is one exactly when every bit is
/// zero. The result is a product of booleans, so it is boolean by construction
/// and legal as a `Select` bit.
///
/// Cost: one `LFM_BITDEC` row plus `2·nbits − 1` `LFM_BALU` rows. The caller's
/// bound is load-bearing — bits above `nbits` are not decomposed into cells, so
/// a larger `x` would be reported zero on its low bits alone.
pub fn is_zero_bounded(b: &mut LfmBuilder, x: Felt, nbits: usize) -> Bit {
    let bits = b.bit_dec(x, nbits);
    let one = b.felt_const(FE::one());
    let mut acc = one;
    for bit in bits {
        let complement = b.sub(one, bit.as_felt());
        acc = b.mul(acc, complement);
    }
    Bit(acc.0)
}

/// `1` if `x == 2^nbits − 1`, else `0`, for an `x` below `2^nbits`.
///
/// The dual of [`is_zero_bounded`] and cheaper by the complements: `Π bᵢ`.
pub fn is_all_ones_bounded(b: &mut LfmBuilder, x: Felt, nbits: usize) -> Bit {
    let bits = b.bit_dec(x, nbits);
    let one = b.felt_const(FE::one());
    let mut acc = one;
    for bit in bits {
        acc = b.mul(acc, bit.as_felt());
    }
    Bit(acc.0)
}

// ============================== keccak256 ==============================

/// `keccak256` over a byte stream supplied as `u32`-half felts (four bytes
/// each, little-endian — see [`super::keccak_host::pack_stream`]). Returns the
/// 32-byte digest as two machine words of halves.
///
/// Shapes are compile-time, as everywhere in this machine: `len_bytes` fixes
/// the block count and the padding positions, so `pad10*1` is emitted as
/// interned program CONSTANTS rather than computed. A different length is a
/// different program with a different digest — which is the straight-line
/// discipline working as intended, not a limitation to route around.
///
/// The digest is the state's first 32 bytes = halves 0..7 = words 0 and 1 of
/// the state's word representation, which is exactly `PlatformKeccak256`'s
/// output byte order (byte `j` = byte `j % 4` of half `j / 4`).
pub fn keccak256(b: &mut LfmBuilder, stream: &[Felt], len_bytes: usize) -> [Cell; 2] {
    let (state, _) = keccak256_absorb_all(b, stream, len_bytes, false);
    [state[0], state[1]]
}

/// `keccak256`, additionally returning the byte-REVERSED digest — the value the
/// production `DefaultTranscript::sample()` both returns as the challenge and
/// re-absorbs as the next segment's prefix.
///
/// `sample()` is byte-for-byte identical before and after #841, so this is
/// independent of which transcript revision the caller targets.
pub fn keccak256_rev(b: &mut LfmBuilder, stream: &[Felt], len_bytes: usize) -> [Cell; 2] {
    let (_, rev) = keccak256_absorb_all(b, stream, len_bytes, true);
    rev.expect("requested")
}

/// `keccak256` returning BOTH digests — plain and byte-reversed — off the one
/// keccak row that produces them.
///
/// The transcript replay needs both at once and they are not interchangeable:
/// the reversed digest is what `sample()` returns and re-absorbs, while
/// candidates are read off the PLAIN digest (the reversal and the big-endian
/// candidate read cancel — see [`super::keccak_host::candidate_from_state`]).
pub fn keccak256_with_rev(
    b: &mut LfmBuilder,
    stream: &[Felt],
    len_bytes: usize,
) -> ([Cell; 2], [Cell; 2]) {
    let (state, rev) = keccak256_absorb_all(b, stream, len_bytes, true);
    ([state[0], state[1]], rev.expect("requested"))
}

/// `Σ_i 2^i · bits[i]`, bits low-to-high — the value a bit decomposition stands
/// for. Horner from the top: one `MulAdd` per bit after the first.
pub fn bits_to_felt(b: &mut LfmBuilder, bits: &[Bit]) -> Felt {
    let two = b.felt_const(FE::from(2u64));
    let mut iter = bits.iter().rev();
    let mut acc = iter.next().expect("at least one bit").as_felt();
    for bit in iter {
        acc = b.mul_add(acc, two, bit.as_felt());
    }
    acc
}

// ============================== Blake3Chain ==============================

/// `Blake3Chain` over a byte stream supplied as `u32`-half felts — the same
/// packing [`keccak256`] takes, because a BLAKE3 message word IS a
/// little-endian `u32` of four consecutive message bytes.
///
/// Returns the 32-byte digest as two machine words. The digest is the
/// compression's output words 0 and 1 (`out[0..8]` little-endian), so reading it
/// costs nothing — and the CHAINING VALUE of the next block is those same two
/// words, which is why multi-block messages need no repacking between blocks.
///
/// Shapes are compile-time, as everywhere in this machine: `len_bytes` fixes the
/// block count, the `block_len` of the final block and the whole flag schedule,
/// all of which are emitted as interned program CONSTANTS. The schedule itself
/// is read from [`crypto::hash::blake3::chain`] rather than restated here — see
/// [`blake3_absorb_all`].
pub fn blake3_256(b: &mut LfmBuilder, stream: &[Felt], len_bytes: usize) -> Blake3Digest {
    blake3_absorb_all(b, stream, len_bytes, false).0
}

/// [`blake3_256`], returning the byte-REVERSED digest instead — the value the
/// production `DefaultTranscript::sample()` both returns as the challenge and
/// re-absorbs as the next segment's prefix.
pub fn blake3_256_rev(b: &mut LfmBuilder, stream: &[Felt], len_bytes: usize) -> [Cell; 2] {
    blake3_absorb_all(b, stream, len_bytes, true)
        .1
        .expect("requested")
}

/// [`blake3_256`] returning BOTH digests — plain and byte-reversed — off the one
/// compression that produces them.
pub fn blake3_256_with_rev(
    b: &mut LfmBuilder,
    stream: &[Felt],
    len_bytes: usize,
) -> (Blake3Digest, [Cell; 2]) {
    let (digest, rev) = blake3_absorb_all(b, stream, len_bytes, true);
    (digest, rev.expect("requested"))
}

/// The `Blake3Chain` framing, emitted.
///
/// Three differences from [`keccak256_absorb_all`], each a place a port goes
/// wrong silently, so each is named:
///
/// - **No padding constants.** BLAKE3 zero-pads. There is no `pad10*1`, no
///   `pad_half`, and nothing analogous to keccak's `stream[g] + pad` merge —
///   a half past the message is the shared zero cell, full stop.
/// - **`block_len` is data, not shape-implied.** The final block carries the
///   true byte count where keccak encoded the same information positionally in
///   its pad. Compile-time known, so it is an interned constant — but it must be
///   *emitted*.
/// - **Flags are a three-value schedule, not a constant.** First / interior /
///   last, with a single-block message carrying both ends at once
///   (`CHUNK_START | CHUNK_END | ROOT`), which is exactly the parent form.
///
/// ★ The schedule is [`chain::block_flags`] / [`chain::block_len_of`] /
/// [`chain::num_blocks`] — the host hasher's own, hoisted into `crypto` for
/// this caller. A second statement of "which block carries `CHUNK_START`" is the
/// single most likely way for an in-machine hash to differ from the host's by
/// one compression's flags, and that difference is a valid proof of the wrong
/// digest.
fn blake3_absorb_all(
    b: &mut LfmBuilder,
    stream: &[Felt],
    len_bytes: usize,
    want_rev: bool,
) -> (Blake3Digest, Option<[Cell; 2]>) {
    use super::blake3::BLAKE3_IV;
    use super::blake3::chain::{BLOCK_LEN, block_flags, block_len_of, num_blocks};
    use super::keccak_host::{BYTES_PER_HALF, num_stream_halves};

    assert_eq!(
        stream.len(),
        num_stream_halves(len_bytes),
        "stream must hold exactly ceil(len_bytes / 4) halves"
    );

    /// `u32` halves in one 64-byte BLAKE3 block: the message words `m[0..16]`.
    const HALVES_PER_BLOCK: usize = BLOCK_LEN / BYTES_PER_HALF; // 16
    /// Machine words the message occupies, four `u32` lanes each.
    const MESSAGE_WORDS: usize = HALVES_PER_BLOCK / 4; // 4

    let zero = b.felt_const(FE::zero());
    let blocks = num_blocks(len_bytes);

    // The initial chaining value: `BLAKE3_IV` as two interned constants. `h` is
    // `u32` words 0..8, so word 0 is `IV[0..4]` and word 1 is `IV[4..8]`.
    let iv_word = |b: &mut LfmBuilder, w: usize| -> Cell {
        b.digest_const(core::array::from_fn(|l| {
            FE::from(u64::from(BLAKE3_IV[4 * w + l]))
        }))
        .as_cell()
    };
    let mut h: [Cell; 2] = [iv_word(b, 0), iv_word(b, 1)];
    let mut rev: Option<[Cell; 2]> = None;

    for block in 0..blocks {
        // Half `h` of this block is half `block * 16 + h` of the message; both a
        // block (64 bytes) and a half (4 bytes) divide evenly, so the two
        // indexings line up with no straddling. Halves past the message are the
        // zero cell — the zero padding, with nothing to splice.
        let message: [Cell; MESSAGE_WORDS] = core::array::from_fn(|w| {
            let lane = |l: usize| {
                let g = block * HALVES_PER_BLOCK + 4 * w + l;
                if g < stream.len() { stream[g] } else { zero }
            };
            b.pack_word([lane(0), lane(1), lane(2), lane(3)])
        });

        // `(t_lo, t_hi, block_len, flags)`. `t = 0` at every block: the
        // construction is a single chunk that never ends (PA-PLAN §1.7, F1).
        let params = b
            .digest_const([
                FE::zero(),
                FE::zero(),
                FE::from(u64::from(block_len_of(block, len_bytes))),
                FE::from(u64::from(block_flags(block, blocks))),
            ])
            .as_cell();

        let out = if block + 1 == blocks && want_rev {
            let (out, rev_words) = b.blake3_compress_rev(h, message, params);
            rev = Some(rev_words);
            out
        } else {
            b.blake3_compress(h, message, params)
        };
        // The next block's chaining value is `out[0..8]` — output words 0 and 1
        // — and so is the digest when this was the last block. One and the same,
        // which is what the truncation to `[out[0], out[1]]` says.
        h = [out[0], out[1]];
    }

    (WrapDigest::from_pair(h[0], h[1]), rev)
}

fn keccak256_absorb_all(
    b: &mut LfmBuilder,
    stream: &[Felt],
    len_bytes: usize,
    want_rev: bool,
) -> ([Cell; 13], Option<[Cell; 2]>) {
    use super::keccak_host::{num_blocks, num_stream_halves, pad_half};
    use super::layout::keccak::{BLOCK_HALVES, BLOCK_WORDS, NUM_WORDS};

    assert_eq!(
        stream.len(),
        num_stream_halves(len_bytes),
        "stream must hold exactly ceil(len_bytes / 4) halves"
    );

    let zero = b.felt_const(FE::zero());
    let mut state: [Cell; NUM_WORDS] = [zero.as_cell(); NUM_WORDS];
    let mut rev: Option<[Cell; 2]> = None;

    for block in 0..num_blocks(len_bytes) {
        // Half `h` of this block is half `block * BLOCK_HALVES + h` of the
        // padded message; both the rate (136 bytes) and a half (4 bytes) divide
        // evenly, so the two indexings line up with no straddling across blocks.
        let halves: Vec<Felt> = (0..BLOCK_HALVES)
            .map(|h| {
                let g = block * BLOCK_HALVES + h;
                let pad = pad_half(len_bytes, g);
                match (g < num_stream_halves(len_bytes), pad) {
                    // Entirely inside the message.
                    (true, 0) => stream[g],
                    // Straddles the end: the stream half's high bytes are zero
                    // by the packing convention, so adding merges the padding in
                    // without carrying.
                    (true, p) => {
                        let c = b.felt_const(FE::from(p));
                        b.add(stream[g], c)
                    }
                    // Entirely padding (possibly all-zero).
                    (false, p) => b.felt_const(FE::from(p)),
                }
            })
            .collect();

        // 34 halves into 9 words; the last word's top two slots are the unused
        // half slots the chip pins to zero.
        let block_words: [Cell; BLOCK_WORDS] = core::array::from_fn(|w| {
            let lane = |l: usize| {
                let h = 4 * w + l;
                if h < BLOCK_HALVES { halves[h] } else { zero }
            };
            b.pack_word([lane(0), lane(1), lane(2), lane(3)])
        });

        let last = block + 1 == num_blocks(len_bytes);
        if last && want_rev {
            let (next, rev_words) = b.keccak_absorb_rev(state, block_words);
            state = next;
            rev = Some(rev_words);
        } else {
            state = b.keccak_absorb(state, block_words);
        }
    }

    (state, rev)
}
