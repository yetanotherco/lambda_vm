//! Host-side arena filler: real proof BYTES → LFM arena words.
//!
//! Input is the guest's wire-format blob, not an in-memory bundle — see
//! [`super::proof_fixture`] for why that fidelity matters. Everything here reads
//! the archived view in place, exactly as the recursion guest does.
//!
//! ## The packing rule this module exists to enforce
//!
//! An arena is a vector of `u32` words, NOT a byte stream. Every field must be
//! packed into its OWN halves; concatenating fields and packing afterwards lets
//! a field whose length is not a multiple of four shift every field behind it.
//! That bug cost real debugging time in R1e and it is silent — the halves count
//! still comes out right, only the values are wrong.

use crypto::merkle_tree::proof::verify_merkle_path_from_leaf_hash;
use math::field::element::FieldElement;
use stark::config::{BatchedMerkleTreeBackend, Commitment};

use crate::tables::types::GoldilocksField;

use super::keccak_host::pack_stream;
use super::proof_fixture::FixtureArchive;
use super::word::{LfmWord, base_word};

type FE = FieldElement<GoldilocksField>;

/// The Merkle backend the main trace is committed under — the production alias,
/// not a locally chosen equivalent, so a backend change reaches this module.
type MainBackend = BatchedMerkleTreeBackend<GoldilocksField>;

/// Halves in one 32-byte commitment.
pub const ROOT_HALVES: usize = 8;

/// The main-trace Merkle roots an epoch's sub-proofs commit to, in air order.
///
/// These are the roots Phase A absorbs and, more importantly for R1f, the roots
/// a Merkle opening is authenticated AGAINST. They come straight off the proof.
///
/// NOTE for the Phase-A leg: the verifier also absorbs each air's PREPROCESSED
/// commitment, and that one does NOT live in the proof — it comes from the AIR
/// set (`air.precomputed_commitment()`), which means replaying Phase A over a
/// real proof needs the epoch's AIRs rebuilt, not just its bytes. Out of scope
/// here and flagged rather than papered over.
pub fn epoch_main_roots(archive: &FixtureArchive, epoch: usize) -> Vec<Commitment> {
    let bundle = &archive.guest_input().bundle;
    assert!(
        epoch < bundle.num_epochs(),
        "epoch {epoch} out of range ({} epochs)",
        bundle.num_epochs()
    );
    let proofs = bundle.epoch_proof(epoch);
    (0..proofs.len())
        .map(|i| *proofs.get(i).lde_trace_main_merkle_root())
        .collect()
}

/// Number of sub-proofs (tables) in an epoch.
pub fn epoch_num_tables(archive: &FixtureArchive, epoch: usize) -> usize {
    archive.guest_input().bundle.epoch_proof(epoch).len()
}

pub fn num_epochs(archive: &FixtureArchive) -> usize {
    archive.guest_input().bundle.num_epochs()
}

/// Bytes epoch `epoch` committed — the statement's `public_output` field.
pub fn epoch_public_output(archive: &FixtureArchive, epoch: usize) -> &[u8] {
    archive.guest_input().bundle.epoch_public_output(epoch)
}

/// Packs commitments into arena halves, each root into its OWN eight halves.
pub fn roots_to_halves(roots: &[Commitment]) -> Vec<FE> {
    let mut out = Vec::with_capacity(roots.len() * ROOT_HALVES);
    for root in roots {
        let halves = pack_stream(root);
        debug_assert_eq!(halves.len(), ROOT_HALVES);
        out.extend(halves);
    }
    out
}

/// Wraps packed halves as arena words.
pub fn halves_to_arena(halves: Vec<FE>) -> Vec<LfmWord> {
    halves.into_iter().map(base_word).collect()
}

/// A 32-byte commitment as the two machine words a keccak digest occupies:
/// four `u32` halves per word, half `h` = bytes `4h..4h+4` little-endian.
///
/// This is NOT [`super::word::pack_digest`]'s layout. That one packs four FULL
/// felts, which is the `LFM_HASH` (Milestone-C) digest; a keccak digest lives on
/// the bus as eight `u32` halves and must be handed to the chip that way.
pub fn commitment_words(c: &Commitment) -> [LfmWord; 2] {
    let halves = pack_stream(c);
    debug_assert_eq!(halves.len(), ROOT_HALVES);
    [
        [halves[0], halves[1], halves[2], halves[3]],
        [halves[4], halves[5], halves[6], halves[7]],
    ]
}

// ==================== one query's main-trace opening ====================

/// One FRI query's MAIN-trace opening, in the form the machine consumes it.
///
/// This is the input to [`crate::lfm::edsl::keccak_merkle_walk`] and the thing
/// R1f authenticates: a real row pair from a real continuation-epoch proof,
/// against that proof's own committed root.
///
/// ## What the verifier does with these fields
///
/// `Verifier::verify_opening_pair` hashes `evaluations ‖ evaluations_sym` into
/// one leaf and folds it up `merkle_path` at index `iota`. The pair is one leaf
/// because `ROWS_PER_LEAF = 2`: a query opens a value and its symmetric
/// counterpart, which are the two bit-reversed rows `2·iota` and `2·iota+1`, so
/// a single path authenticates both.
pub struct MainTraceOpening {
    /// The committed root, read off the proof — the oracle for the whole leg.
    pub root: Commitment,
    /// `evaluations ‖ evaluations_sym` in hash order: the row pair written
    /// column by column, each element rendered big-endian by the leaf hasher.
    pub values: Vec<FE>,
    /// Where `evaluations_sym` starts — i.e. the table's column count.
    pub num_columns: usize,
    /// Sibling digests, LEAF LEVEL FIRST. That is the order
    /// `verify_merkle_path_from_leaf_hash` consumes them in: it walks the vector
    /// forwards while shifting the index right, so element 0 pairs with the
    /// index's least significant bit. (`Proof`'s doc comment describes the
    /// reverse; the code is what this mirrors.)
    pub siblings: Vec<Commitment>,
}

impl MainTraceOpening {
    /// Reads query `query` of sub-proof `table` in epoch `epoch`.
    pub fn extract(
        archive: &FixtureArchive,
        epoch: usize,
        table: usize,
        query: usize,
    ) -> MainTraceOpening {
        let bundle = &archive.guest_input().bundle;
        assert!(epoch < bundle.num_epochs(), "epoch {epoch} out of range");
        let proofs = bundle.epoch_proof(epoch);
        assert!(table < proofs.len(), "table {table} out of range");
        let proof = proofs.get(table);
        assert!(
            query < proof.deep_poly_openings_len(),
            "query {query} out of range ({} openings)",
            proof.deep_poly_openings_len()
        );
        let opening = proof.deep_poly_opening(query).main_trace_polys();
        let evaluations = opening.evaluations();
        let sym = opening.evaluations_sym();
        assert_eq!(
            evaluations.len(),
            sym.len(),
            "a row pair's two rows must have the same width"
        );
        MainTraceOpening {
            root: *proof.lde_trace_main_merkle_root(),
            num_columns: evaluations.len(),
            values: evaluations.iter().chain(sym.iter()).cloned().collect(),
            siblings: opening.merkle_path().to_vec(),
        }
    }

    /// Path length = tree depth = the number of index bits the walk consumes.
    pub fn depth(&self) -> usize {
        self.siblings.len()
    }

    /// The leaf hash, computed by the PRODUCTION hasher on the production
    /// split — literally the call `verify_opening_pair` makes.
    pub fn leaf_hash(&self) -> Commitment {
        MainBackend::hash_data_from_slices(
            &self.values[..self.num_columns],
            &self.values[self.num_columns..],
        )
    }

    /// Whether production's own path check accepts this opening at `index`.
    pub fn verifies_at(&self, index: usize) -> bool {
        verify_merkle_path_from_leaf_hash::<MainBackend>(
            &self.siblings,
            &self.root,
            index,
            self.leaf_hash(),
        )
    }

    /// Every leaf index at which this opening authenticates.
    ///
    /// ## Why a search, and why that is honest
    ///
    /// The index is the FRI query challenge `iota`, and it is NOT in the proof —
    /// the verifier derives it from the transcript, which needs the epoch's
    /// statement and its AIR set, neither of which a byte blob carries (the
    /// preprocessed commitments come from `air.precomputed_commitment()`). Since
    /// the path, the leaf and the root are all fixed by the proof, the index is
    /// nonetheless determined by them, so recovering it by exhaustion asks the
    /// proof rather than inventing an answer — and the oracle doing the asking
    /// is production's `verify_merkle_path_from_leaf_hash`, not a local model.
    ///
    /// The result is a LIST because a degenerate tree has several: a table
    /// whose trace is mostly padding commits identical rows, so identical
    /// leaves sit under identical subtrees and many indices verify. Any opening
    /// used for an index-tamper vector must have exactly one — otherwise
    /// "flip an index bit" is not a tamper at all. Callers assert that.
    ///
    /// Costs `2^depth` path walks; fine at the fixture's depths, not a
    /// mechanism anything but a fixture should use.
    pub fn indices_that_verify(&self) -> Vec<usize> {
        (0..(1usize << self.depth()))
            .filter(|i| self.verifies_at(*i))
            .collect()
    }

    /// The leaf's field elements as arena words: one base word each, since the
    /// machine byteswaps them itself (they are full felts, not `u32` halves).
    pub fn leaf_arena(&self) -> Vec<LfmWord> {
        self.values.iter().copied().map(base_word).collect()
    }

    /// The sibling digests as arena words, two per level, leaf level first.
    pub fn sibling_arena(&self) -> Vec<LfmWord> {
        self.siblings.iter().flat_map(commitment_words).collect()
    }

    /// The committed root as arena words.
    pub fn root_arena(&self) -> Vec<LfmWord> {
        commitment_words(&self.root).to_vec()
    }
}

/// Host mirror of the machine's walk, returning the root it reaches.
///
/// Production's checker returns a bool, so it cannot supply the root a TAMPERED
/// input folds to — which a coherent forgery needs (the forged run must claim a
/// root consistent with its own inputs, or it fails in-machine before the
/// interesting check). Built from the production parent hash, so the only thing
/// local about it is the loop.
pub fn walk_to_root(leaf: Commitment, index: usize, siblings: &[Commitment]) -> Commitment {
    use crypto::merkle_tree::traits::IsMerkleTreeBackend;
    let mut node = leaf;
    let mut index = index;
    for sibling in siblings {
        node = if index.is_multiple_of(2) {
            MainBackend::hash_new_parent(&node, sibling)
        } else {
            MainBackend::hash_new_parent(sibling, &node)
        };
        index >>= 1;
    }
    node
}

// ==================== the cross-epoch L2G binding ====================

/// Each epoch's own committed L2G table root, in epoch order.
///
/// The left-hand side of `verify_l2g_commitment_binding_view`: epoch `i`'s
/// `EpochProof::l2g_root`, which that epoch's own proof commits to.
pub fn epoch_l2g_roots(archive: &FixtureArchive) -> Vec<Commitment> {
    let bundle = &archive.guest_input().bundle;
    (0..bundle.num_epochs())
        .map(|i| bundle.epoch_l2g_root(i))
        .collect()
}

/// The global proof's first `count` sub-proof main-trace roots — the right-hand
/// side of the same binding.
///
/// The global proof carries one L2G sub-proof per epoch FIRST, then
/// GLOBAL_MEMORY, so sub-proof `i` is epoch `i`'s L2G table. Production also
/// checks `final_proof.len() >= epoch_l2g_roots.len()`; here that is structural,
/// since a machine program compiled for `n` epochs reads exactly `n` roots and
/// this function panics rather than short-reading.
pub fn global_l2g_roots(archive: &FixtureArchive, count: usize) -> Vec<Commitment> {
    let global = archive.guest_input().bundle.global_proof();
    assert!(
        global.len() >= count,
        "the global proof has {} sub-proofs, need {count}",
        global.len()
    );
    (0..count)
        .map(|i| *global.get(i).lde_trace_main_merkle_root())
        .collect()
}

/// Commitments as arena words, two per root, in order.
pub fn commitments_to_arena(roots: &[Commitment]) -> Vec<LfmWord> {
    roots.iter().flat_map(commitment_words).collect()
}

// ==================== the attestation's program id ====================

/// The inner ELF bytes the guest input carries.
pub fn inner_elf(archive: &FixtureArchive) -> &[u8] {
    archive.guest_input().inner_elf.as_slice()
}

/// The supplied DECODE preprocessed root.
pub fn decode_commitment(archive: &FixtureArchive) -> Commitment {
    archive.guest_input().decode_commitment
}

/// The supplied per-page genesis roots, `(base, commitment)`.
///
/// ⚠ EMPTY for the `fibonacci` fixture — that guest touches no data pages — so
/// any test that only uses the fixture leaves the page path unexercised. Drive
/// it with a synthetic shape rather than treating it as covered.
pub fn page_commitments(archive: &FixtureArchive) -> Vec<(u64, Commitment)> {
    archive
        .guest_input()
        .page_commitments
        .iter()
        .map(|p| (p.0.to_native(), p.1))
        .collect()
}

// ============ the cross-epoch REGISTER boundary ============

/// Epoch `i`'s `(register_init, reg_fini)` — the pair
/// `register::compute_precomputed_commitment_with_fini` turns into that epoch's
/// preprocessed REGISTER commitment.
///
/// INIT is the VERIFIER's derivation, never a bundled value: epoch 0's comes
/// from the inner ELF's entry point and every later epoch's is the previous
/// epoch's `reg_fini`. That is the whole point of the chaining obligation, so
/// reading INIT off the proof here would quietly test a different mechanism —
/// the walk below is the same one `verify_continuation_archived` performs.
pub fn register_boundary(archive: &FixtureArchive, epoch: usize) -> (Vec<u32>, Vec<u32>) {
    let bundle = &archive.guest_input().bundle;
    assert!(
        epoch < bundle.num_epochs(),
        "epoch {epoch} out of range ({} epochs)",
        bundle.num_epochs()
    );
    let elf = executor::elf::Elf::load(inner_elf(archive)).expect("the inner ELF must load");
    let mut init = crate::tables::register::register_init_from_entry_point(elf.entry_point);
    for i in 0..epoch {
        init = bundle.epoch_reg_fini(i).expect("reg_fini deserializes");
    }
    let fini = bundle.epoch_reg_fini(epoch).expect("reg_fini deserializes");
    (init, fini)
}
