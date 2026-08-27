//! Assembly — the per-table challenge replay that turns the legs into a
//! verifier.
//!
//! Every leg so far took its challenges as ARENA WORDS: `emit_sub_proof` hints
//! `γ` and `ζ`, `declare_fri` hints the folding challenges, and the query index
//! arrives as a hinted felt that the walk decomposes. That is fine for a
//! differential against production, which supplies the true values, and it is
//! fatal in a verifier: a prover who chooses `γ` chooses the DEEP fold, one who
//! chooses `ζ_k` chooses the FRI fold, and one who chooses `ι` chooses which
//! rows are ever opened. This module is where those words stop being data.
//!
//! ## What it replays
//!
//! `verifier.rs`'s per-table body, in production's order and nothing else:
//!
//! - the FORK (`:1263-1266`) — clone the shared post-Phase-A transcript, then a
//!   domain separator `idx.to_le_bytes()` when the epoch has more than one
//!   table;
//! - the aux root (`:1269-1271`) and the bus contribution `L` (`:1274-1276`);
//! - Round 2 (`:1380-1404`): sample `β`, then absorb the composition root;
//! - Round 3 (`:1411-1434`): sample `z`, then absorb the two pruned OOD blocks
//!   COLUMN-major, then the claimed composition parts;
//! - Round 4 (`:1445-1504`): sample `γ`; then per committed FRI layer sample
//!   `ζ_k` and absorb root `k`; then `ζ_C` if and only if the codeword folds;
//!   then every terminal coefficient; then grinding; then the query indices.
//!
//! The interleaving in Round 4 is the part no leg-side test could catch, and it
//! is load-bearing in both directions: a `ζ_k` sampled after its own layer root
//! is a challenge the prover can answer, and a layer root that is never absorbed
//! leaves the query indices independent of the codeword they index.
//!
//! ## Why the values are returned as CELLS
//!
//! The point of the module is that there is exactly ONE cell per value and both
//! consumers read it. `L` is absorbed here and summed by the LogUp closure; the
//! OOD block cells are absorbed here and folded by the constraint leg and by
//! DEEP; the composition parts likewise; the layer roots are absorbed here and
//! compared against the FRI walk. Nothing is hinted twice, which is the
//! assembly obligation (`others/lfm-assembly-obligations.md`, OPEN 3) stated as
//! a construction rather than as a rule to remember.
//!
//! ## Zero-rejection, one level up
//!
//! `sample_z_ood_with_domain_params` REJECTS a `z` that lands in the trace
//! domain or on the LDE coset and draws again. A straight-line program cannot,
//! so [`emit_z_ood`] draws once and CONSTRAINS both rejection predicates to be
//! false — the same disposition as the sampler's canonicity guard, and the same
//! completeness-only cost (`SOUNDNESS.md` §6.3).

use crate::tables::types::{FE, FEE, GoldilocksExtension};

use super::builder::{Bit, Ext, Felt, LfmBuilder};
use super::fri::FriShape;
use super::layout::keccak::DIGEST_WORDS;
use super::transcript_replay::{ByteString, TranscriptReplay};

/// The grinding prefix, `crypto/stark/src/grinding.rs`'s `PREFIX`.
const GRINDING_PREFIX: [u8; 8] = 0x0123_4567_89ab_cded_u64.to_be_bytes();

/// A commitment root as the machine holds it: eight `u32` lanes, produced ONCE.
///
/// Both consumers of a root — the transcript absorb and the Merkle comparison —
/// want a different view of the same 32 bytes, and a root that was hinted twice
/// (or unpacked twice) would let those views drift. So there is one unpack per
/// root and every consumer reads its lanes.
///
/// The three constructors are the three SOURCES a root can have, which is
/// assembly ledger entry 7's whole content: program text ([`Self::constant`]),
/// an in-machine derivation ([`Self::from_digest`]) or the proof's arena
/// ([`Self::hint`]). Which one a given commitment may use is a property of what
/// the commitment is a function of, not a convenience.
#[derive(Clone)]
pub struct RootCells {
    /// The root as the configuration's digest cells — TWO on a byte hash, ONE on
    /// an algebraic one.
    digest: super::edsl::WrapDigest,
    /// Those cells unpacked, ONE ENTRY PER CELL, hoisted so every consumer reads
    /// the same lanes.
    ///
    /// ⚠ A `Vec` rather than `[_; DIGEST_WORDS]`, and that is the whole shape
    /// change: the array wrote the byte digest's cell COUNT into the type, so an
    /// algebraic root — one cell of four felts, not two of 32 bytes — could not
    /// be one of these. `edsl::assert_digest_eq_lanes` already zips a digest
    /// against these and asserts the widths match, so it works at either width
    /// unchanged once this stops being fixed at two.
    ///
    /// ⚠ On the algebraic arm these lanes are FULL FELTS, not `u32` halves.
    /// Nothing may hand them to a byte-stream absorb; [`RootCells::absorb`]
    /// exists so no caller has to know which it is holding.
    pub lanes: Vec<[Felt; 4]>,
}

impl RootCells {
    /// Arena words one root occupies — the configuration's digest cell count.
    ///
    /// Read from the emitter's own `WrapDigest` shape rather than restated, so
    /// the machine side and `proof_arena`'s writer cannot disagree about it.
    pub fn words_per_root(b: &LfmBuilder) -> u32 {
        match b.wrap_hash() {
            super::edsl::WrapHash::Algebraic => 1,
            _ => DIGEST_WORDS as u32,
        }
    }

    /// Read a root out of an arena at `base` and hoist its unpack.
    ///
    /// ⚠ Consumes [`RootCells::words_per_root`] words, not always two: an
    /// algebraic root is ONE arena word of four felts. Callers that advance a
    /// cursor across several roots must step by the same function.
    pub fn hint(b: &mut LfmBuilder, arena: super::instr::ArenaId, base: u32) -> Self {
        let n = Self::words_per_root(b);
        let words: Vec<_> = (0..n).map(|i| b.hint_word(arena, base + i)).collect();
        Self::from_cells(b, &words)
    }

    /// A root already in hand as the configuration's digest cells.
    fn from_cells(b: &mut LfmBuilder, cells: &[super::builder::Cell]) -> Self {
        RootCells {
            digest: super::edsl::WrapDigest::from_cells(cells),
            lanes: cells.iter().map(|c| b.unpack(*c)).collect(),
        }
    }

    /// The root as the configuration's digest cells — what a comparison against
    /// a computed digest takes.
    pub fn digest(&self) -> super::edsl::WrapDigest {
        self.digest
    }

    /// Absorb this root into a transcript.
    ///
    /// ★ The reason `halves()` is gone: nine call sites spelled
    /// `t.append_halves(&root.halves())`, which writes "a root is eight `u32`
    /// halves" into every one of them. It is one cell of four felts on an
    /// algebraic hash, and the absorb is FREE there.
    pub fn absorb(&self, b: &mut LfmBuilder, t: &mut TranscriptReplay) {
        t.append_root_cells(b, self.digest.cells());
    }

    /// A BYTE-hash root from its eight `u32` halves — the instruments' shape.
    ///
    /// ⚠ Byte arm only by construction: eight halves IS the byte digest's
    /// layout, and a caller holding them is by that fact not on the algebraic
    /// arm. Kept for the hash-pinned instruments and the tests that build a root
    /// half by half; production reads roots through [`RootCells::hint`].
    pub fn from_halves(b: &mut LfmBuilder, halves: &[Felt]) -> Self {
        assert_eq!(
            halves.len(),
            4 * DIGEST_WORDS,
            "a byte root is eight halves"
        );
        let cells: Vec<_> = halves
            .chunks(4)
            .map(|c| b.pack_word([c[0], c[1], c[2], c[3]]))
            .collect();
        Self::from_cells(b, &cells)
    }

    /// The lanes flattened, in order.
    ///
    /// ⚠ These are `u32` HALVES on a byte hash and FULL FELTS on an algebraic
    /// one, so this is meaningful only to a caller that already knows which it
    /// is holding — which is why the transcript absorbs go through
    /// [`RootCells::absorb`] instead. It exists for the hash-pinned instruments
    /// and for tests that publish or compare a root's cells directly.
    pub fn lanes_flat(&self) -> Vec<Felt> {
        self.lanes.iter().flatten().copied().collect()
    }

    /// [`RootCells::absorb`] where the segment cursor is not 4-byte aligned.
    pub fn absorb_misaligned(&self, b: &mut LfmBuilder, t: &mut TranscriptReplay) {
        t.append_root_cells_misaligned(b, self.digest.cells());
    }

    /// A root that is PROGRAM TEXT — its eight halves interned as constants.
    ///
    /// Admissible only for a commitment that is a function of the proof OPTIONS
    /// and nothing else, because a program constant is part of program identity:
    /// interning a root derived from per-proof data (an ELF, a register file)
    /// would give the machine one program per proof instead of one per epoch
    /// SHAPE. The two that qualify are BITWISE and KECCAK_RC
    /// (`bitwise::preprocessed_commitment(options)`,
    /// `tables::keccak_rc::preprocessed_commitment(options)`), plus PAGE's
    /// zero-init root, which every zero-initialised page shares.
    ///
    /// Half `h` is bytes `4h..4h+4` little-endian —
    /// `proof_arena::commitment_words`' layout, which is how a keccak digest
    /// reaches the chip.
    pub fn constant(b: &mut LfmBuilder, root: &[u8; 4 * 4 * DIGEST_WORDS]) -> Self {
        if b.wrap_hash() == super::edsl::WrapHash::Algebraic {
            // ★ The same 32 bytes read as the FOUR CANONICAL FELTS an algebraic
            // backend serialised them from — one interned cell, not eight
            // interned halves. The conversion is the backend's own, not a
            // second spelling of it.
            let cell = super::algebraic_commit::commitment_to_digest(root);
            let d = b.digest_const(cell);
            return Self::from_cells(b, &[d.as_cell()]);
        }
        let halves: Vec<Felt> = root
            .chunks(4)
            .map(|c| {
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(c);
                b.felt_const(FE::from(u64::from(u32::from_le_bytes(bytes))))
            })
            .collect();
        let cells: Vec<_> = (0..DIGEST_WORDS)
            .map(|w| {
                let lane = |j: usize| halves[4 * w + j];
                b.pack_word([lane(0), lane(1), lane(2), lane(3)])
            })
            .collect();
        Self::from_cells(b, &cells)
    }

    /// A root the machine COMPUTED — the derivation's two digest words, unpacked
    /// once so every consumer reads the same lanes.
    ///
    /// This is REGISTER's source: the commitment is a function of the previous
    /// epoch's `reg_fini`, and computing it from those cells is what binds them
    /// (`programs::emit_register_commitment`). A hinted REGISTER root would leave
    /// the register boundary — the carried commit index among it — a free arena
    /// word.
    pub fn from_digest(b: &mut LfmBuilder, digest: super::edsl::WrapDigest) -> Self {
        Self::from_cells(b, digest.cells())
    }
}

/// The shape of one sub-proof's challenge replay. Every field is a program
/// constant: shape, never proof data.
#[derive(Clone, Debug)]
pub struct TableChallengeShape {
    /// Position in the epoch's table list — the fork's domain separator.
    pub index: usize,
    /// How many sub-proofs the epoch has. Production skips the separator
    /// entirely at one table (`verifier.rs:1264`), so this changes the bytes.
    pub num_tables: usize,
    /// Whether the sub-proof carries an aux (LogUp) trace root.
    pub has_aux_root: bool,
    /// Whether the sub-proof carries a bus contribution `L`.
    pub has_contribution: bool,
    /// `log2` of the trace length.
    pub log2_trace_length: u32,
    /// `log2` of the blowup factor.
    pub log2_blowup: u32,
    /// `ProofOptions::coset_offset`.
    pub coset_offset: FE,
    /// `(width, height)` of the current-row OOD block, as the proof carries it.
    pub ood_current_dims: (usize, usize),
    /// `(width, height)` of the pruned next-row OOD block.
    pub ood_next_dims: (usize, usize),
    /// Composition-poly parts — `air.composition_poly_degree_bound / N`.
    pub num_parts: usize,
    /// The FRI shape, which fixes how many `ζ`s are drawn and in what order the
    /// layer roots are absorbed.
    pub fri: FriShape,
    /// `ProofOptions::grinding_factor`. Zero means no nonce at all.
    pub grinding_factor: u8,
    /// `ProofOptions::fri_number_of_queries`.
    pub num_queries: usize,
}

impl TableChallengeShape {
    /// `log2` of the LDE domain.
    pub fn log2_lde_length(&self) -> u32 {
        self.log2_trace_length + self.log2_blowup
    }

    /// Bits one query index carries — `sample_u64(lde_length >> 1)`
    /// (`verifier.rs:138-141`), so one bit narrower than the domain, which is
    /// exactly the Merkle depth the walk consumes.
    pub fn index_bits(&self) -> usize {
        self.log2_lde_length() as usize - 1
    }

    fn check(&self) {
        assert!(
            self.index < self.num_tables,
            "the table index must be in range"
        );
        assert_eq!(
            self.fri.log2_lde_length,
            self.log2_lde_length(),
            "the FRI shape and the trace shape must describe one domain"
        );
        assert_eq!(
            self.fri.num_queries, self.num_queries,
            "the query count is one shape, declared once"
        );
        assert!(self.num_parts > 0, "a composition polynomial has parts");
    }
}

/// The proof-carried cells one table's replay absorbs.
///
/// These are the caller's cells, hinted once and handed here — never re-hinted.
/// The struct is the assembly join surface: the same values go on to the
/// constraint leg, the DEEP fold, the FRI walk and the LogUp closure.
pub struct TableAbsorbs<'a> {
    /// The aux trace root, present exactly when the AIR has an aux trace.
    pub aux_root: Option<&'a RootCells>,
    /// The bus contribution `L`. The LogUp closure sums THIS cell.
    pub contribution: Option<Ext>,
    /// The composition polynomial's committed root.
    pub composition_root: &'a RootCells,
    /// The current-row OOD block, ROW-major as the proof carries it
    /// (`width · height` cells).
    pub ood_current: &'a [Ext],
    /// The pruned next-row OOD block, row-major.
    pub ood_next: &'a [Ext],
    /// The claimed composition parts at `z^P`.
    pub parts: &'a [Ext],
    /// The committed FRI layer roots, in fold order.
    pub fri_roots: &'a [RootCells],
    /// The terminal polynomial's coefficients, low-to-high.
    pub fri_coeffs: &'a [Ext],
    /// The grinding nonce, present exactly when `grinding_factor > 0`.
    ///
    /// Carried as a FELT, so a nonce at or above `p` cannot be expressed. That
    /// is a completeness restriction and not a soundness one — such a nonce
    /// yields no LFM proof, never a wrong verdict — and it is unreachable in
    /// practice: the prover searches nonces upward from zero, so reaching `p`
    /// would mean grinding 64 bits.
    pub nonce: Option<Felt>,
}

/// One table's challenges, as the cells the verification legs consume.
pub struct TableChallenges {
    /// The constraint-coefficient base. Production expands `β⁰ .. β^{n−1}` and
    /// splits the run into transition then boundary coefficients.
    pub beta: Ext,
    /// The OOD point.
    pub z: Ext,
    /// The DEEP batching challenge.
    pub gamma: Ext,
    /// `ζ₀ .. ζ_C`, or empty when the codeword never folds.
    pub zetas: Vec<Ext>,
    /// Per query, the index bits low-to-high — `index_bits()` of them.
    ///
    /// Bits, never a felt: production draws `sample_u64(lde >> 1)`, whose
    /// output is `nbits` bits by construction, and the walk consumes bits. A
    /// felt would readmit the standalone driver's aliasing (ledger entry 5),
    /// where `ι` and `ι + 2^(n−1)` are the same query.
    pub iota_bits: Vec<Vec<Bit>>,
}

/// Fork the shared transcript for table `index` — `verifier.rs:1263-1266`.
///
/// The clone emits nothing: the shared prefix's keccak rows were emitted once,
/// when Phase A ran, and every fork carries the same cells for them. The
/// separator is a program constant because the table index is shape.
pub fn fork_table(shared: &TranscriptReplay, index: usize, num_tables: usize) -> TranscriptReplay {
    assert!(index < num_tables, "the table index must be in range");
    let mut fork = shared.clone();
    if num_tables > 1 {
        fork.append_const_bytes(&(index as u64).to_le_bytes());
    }
    fork
}

/// `z ∉ trace domain ∪ LDE coset`, drawn once and constrained.
///
/// Production loops until both predicates fail
/// (`is_transcript.rs:61-74`). This machine draws one `z` and proves the two
/// non-memberships, which is the same accepted set — a `z` production would
/// have rejected makes the program unprovable rather than making it accept.
///
/// Both predicates are equalities over `z^N`, so one `N`-power chain
/// (`log2_trace_length` extension squarings) serves both, and each
/// non-equality is one extension division: `1/(a − b)` is provable exactly when
/// `a ≠ b`, the idiom `assert_canonical` uses on the candidate halves.
pub fn emit_z_ood(
    b: &mut LfmBuilder,
    t: &mut TranscriptReplay,
    shape: &TableChallengeShape,
) -> Ext {
    let z = t.sample_ext(b);
    assert_z_outside_domains(b, z, shape);
    z
}

/// The two non-memberships alone, so they can be driven with a chosen `z`.
///
/// A transcript-derived `z` is generic with overwhelming probability, so the
/// guard is unreachable from [`emit_z_ood`] — the reason it lives in its own
/// function is that `the_z_guard_rejects_a_point_in_either_domain` can then
/// feed it the points production would have rejected.
pub fn assert_z_outside_domains(b: &mut LfmBuilder, z: Ext, shape: &TableChallengeShape) {
    assert_z_outside_domains_raw(
        b,
        z,
        shape.log2_trace_length,
        shape.log2_blowup,
        shape.coset_offset,
    );
}

/// [`assert_z_outside_domains`] from the three domain parameters directly —
/// the batched spine draws per-table `z`s without a per-table
/// [`TableChallengeShape`] to hand over.
pub fn assert_z_outside_domains_raw(
    b: &mut LfmBuilder,
    z: Ext,
    log2_trace_length: u32,
    log2_blowup: u32,
    coset_offset: FE,
) {
    // z^N by repeated squaring; N = 2^log2_trace_length.
    let mut z_pow_trace = z;
    for _ in 0..log2_trace_length {
        z_pow_trace = b.emul(z_pow_trace, z_pow_trace);
    }
    let one = b.ext_const(&FEE::one());
    assert_ne_ext(b, z_pow_trace, one);

    // (z^N)^blowup against coset_offset^lde — the offset power is a program
    // constant because the domain is shape.
    let mut z_pow_lde = z_pow_trace;
    for _ in 0..log2_blowup {
        z_pow_lde = b.emul(z_pow_lde, z_pow_lde);
    }
    let offset_pow = coset_offset
        .pow(1u64 << (log2_trace_length + log2_blowup))
        .to_extension::<GoldilocksExtension>();
    let offset_pow = b.ext_const(&offset_pow);
    assert_ne_ext(b, z_pow_lde, offset_pow);
}

/// Constrain `a ≠ b` by exhibiting `(a − b)⁻¹`.
///
/// `Div` is constrained as `OUT · B = A`, which for `A = 1` has no witness at
/// `B = 0`: the program is unprovable exactly when the two are equal.
fn assert_ne_ext(b: &mut LfmBuilder, x: Ext, y: Ext) {
    let d = b.esub(x, y);
    let one = b.ext_const(&FEE::one());
    let _ = b.ediv(one, d);
}

/// Verify the grinding nonce — `grinding::is_valid_nonce`.
///
/// Two keccaks: the inner hash over `PREFIX ‖ state ‖ factor` (41 bytes) and
/// the outer over `inner ‖ nonce_be` (40 bytes).
///
/// The predicate is `u64::from_be_bytes(digest[..8]) < 2^(64 − g)` — "the top
/// `g` bits of the digest's first eight bytes, read big-endian, are zero". Those
/// eight bytes are lanes 0 and 1 of the digest's first WORD, and a lane is four
/// bytes LITTLE-endian, so byte `i` of the big-endian run is bit-range
/// `[8·(i mod 4), 8·(i mod 4) + 8)` of lane `i / 4`. The check is therefore a
/// bit decomposition of at most two lanes plus a run of zero assertions — no
/// comparison and no 64-bit arithmetic, because the bound is a power of two.
///
/// Skipping the check would not merely be untidy: the nonce is absorbed, so
/// the query indices depend on it, and an unchecked nonce is a free re-roll of
/// every query index at zero cost.
pub(super) fn emit_grinding_check(
    b: &mut LfmBuilder,
    // `seed` carries its own width, so an algebraic seed (ONE cell) needs no
    // change here.
    seed: super::edsl::WrapDigest,
    // The nonce as a FELT rather than as its two big-endian halves: the halves
    // are the byte hash's rendering of it, and the algebraic arm needs the value.
    nonce: Felt,
    factor: u8,
) {
    assert!(
        (1..=64).contains(&factor),
        "a grinding factor is in 1..=64 (grinding.rs:22-25), got {factor}"
    );

    let Some(byte_hash) = b.wrap_hash().byte_hash() else {
        emit_algebraic_grinding_check(b, seed, nonce, factor);
        return;
    };

    let nonce_halves = nonce_halves(b, nonce);
    let mut inner = ByteString::new();
    inner.push_const(&GRINDING_PREFIX);
    let mut seed_halves = Vec::with_capacity(8);
    for w in seed.iter() {
        seed_halves.extend_from_slice(&b.unpack(*w));
    }
    inner.push_halves(&seed_halves);
    inner.push_const(&[factor]);
    let inner_hash = inner.wrap_hash(b, byte_hash);

    let mut outer = ByteString::new();
    let mut inner_halves = Vec::with_capacity(8);
    for w in inner_hash.iter() {
        inner_halves.extend_from_slice(&b.unpack(*w));
    }
    outer.push_halves(&inner_halves);
    outer.push_halves(&nonce_halves);
    let digest = outer.wrap_hash(b, byte_hash);

    // The zero bits, as `(byte, bit-within-byte)` pairs of the big-endian run:
    // `factor / 8` whole leading bytes, then the top `factor % 8` bits of the
    // next one.
    let whole = factor as usize / 8;
    let rest = factor as usize % 8;
    let mut wanted: Vec<(usize, usize)> = Vec::with_capacity(factor as usize);
    for byte in 0..whole {
        wanted.extend((0..8).map(|bit| (byte, bit)));
    }
    wanted.extend((8 - rest..8).map(|bit| (whole, bit)));

    let lanes = b.unpack(digest[0]);
    let zero = b.felt_const(FE::zero());
    let mut decomposed: [Option<Vec<Bit>>; 2] = [None, None];
    for (byte, bit) in wanted {
        let lane = byte / 4;
        let bits = match &decomposed[lane] {
            Some(bits) => bits,
            None => {
                decomposed[lane] = Some(b.bit_dec(lanes[lane], 32));
                decomposed[lane].as_ref().expect("just decomposed")
            }
        };
        let v = Felt(bits[8 * (byte % 4) + bit].addr());
        b.assert_eq(v, zero);
    }
}

/// The ALGEBRAIC arm of [`emit_grinding_check`] — two permutations and one bit
/// decomposition, against the byte arm's two full sponge hashes.
///
/// ★ **Both preimages are already felts, so neither hash pays a byte encoding.**
/// `is_valid_nonce` is `H(H(PREFIX ‖ seed ‖ factor) ‖ nonce_be)`, and under
/// `AlgebraicDigest` the host reaches those bytes through `felts_from_bytes` —
/// which reads 8-byte big-endian groups. The seed is
/// `AlgebraicTranscript::state()`, i.e. its four state felts written canonically
/// big-endian, so `felts_from_bytes` recovers exactly those four felts; the
/// inner digest is likewise four canonical felts; and the nonce's eight
/// big-endian bytes are the nonce. The same cancellation a commitment root gets
/// in the transcript, twice more.
///
/// ⚠ A nonce at or above `p` reduces — on BOTH sides, since the host's
/// `felts_from_bytes` calls `FE::from(u64)` exactly as this does — so the two
/// agree. It remains the completeness restriction [`TableAbsorbs::nonce`]
/// records, not a disagreement introduced here.
///
/// Every constant comes from the host's own rules: the byte→felt grouping from
/// `felts_from_bytes`, the rate/capacity split and the padding flag from
/// `single_block_leaf_cells`. The asserts pin the LANE PLACEMENT those rules
/// imply, so the packs below are the rule rather than a second copy of it.
fn emit_algebraic_grinding_check(
    b: &mut LfmBuilder,
    seed: super::edsl::WrapDigest,
    nonce: Felt,
    factor: u8,
) {
    use super::algebraic_commit::{felts_from_bytes, single_block_leaf_cells};

    assert_eq!(seed.len(), 1, "an algebraic transcript state is ONE cell");

    // ---- inner: H(PREFIX ‖ seed ‖ factor), 41 bytes → six felts, one block.
    // The seed's 32 bytes are left ZERO here: those four felts are machine
    // cells, and leaving them zero is what lets the asserts below prove the
    // constants occupy the other two slots and nothing else.
    let mut inner_bytes = [0u8; 41];
    inner_bytes[..GRINDING_PREFIX.len()].copy_from_slice(&GRINDING_PREFIX);
    inner_bytes[40] = factor;
    let inner_felts = felts_from_bytes(&inner_bytes);
    assert_eq!(inner_felts.len(), 6, "41 bytes is six big-endian groups");
    assert_eq!(
        &inner_felts[1..5],
        &[FE::zero(); 4],
        "the seed must occupy felts 1..5 exactly, or the packs below are wrong"
    );
    let inner_cells = single_block_leaf_cells(&inner_felts);
    assert_eq!(
        inner_cells[0],
        [inner_felts[0], FE::zero(), FE::zero(), FE::zero()],
        "rate cell 0 must carry the prefix felt and three seed felts"
    );
    assert_eq!(
        inner_cells[1],
        [FE::zero(), inner_felts[5], FE::zero(), FE::zero()],
        "rate cell 1 must carry the fourth seed felt then the factor felt"
    );

    let s = b.unpack(seed[0]);
    let zero = b.felt_const(FE::zero());
    let prefix = b.felt_const(inner_felts[0]);
    let factor_felt = b.felt_const(inner_felts[5]);
    let inner_rate0 = b.pack_word([prefix, s[0], s[1], s[2]]);
    let inner_rate1 = b.pack_word([s[3], factor_felt, zero, zero]);
    let inner_cap = b.digest_const(inner_cells[2]);
    let inner = b.permute([inner_rate0, inner_rate1, inner_cap.as_cell()])[0];

    // ---- outer: H(inner ‖ nonce_be), 40 bytes → five felts, one block.
    // ★ The inner digest cell IS rate cell 0 — its four lanes are felts 0..4 of
    // this preimage — so nothing repacks it.
    let outer_felts = felts_from_bytes(&[0u8; 40]);
    assert_eq!(outer_felts.len(), 5, "40 bytes is five big-endian groups");
    let outer_cells = single_block_leaf_cells(&outer_felts);
    assert_eq!(
        outer_cells[1],
        [FE::zero(); 4],
        "the nonce must be felt 4, i.e. lane 0 of rate cell 1"
    );
    let outer_rate1 = b.pack_word([nonce, zero, zero, zero]);
    let outer_cap = b.digest_const(outer_cells[2]);
    let digest = b.permute([inner, outer_rate1, outer_cap.as_cell()])[0];

    // ---- the check: the first eight BIG-endian bytes of the digest, read as a
    // `u64`, must be below `2^(64 − factor)`. Those eight bytes are lane 0's
    // canonical `u64` (`cell_to_bytes` writes lane 0 first), so this is "the top
    // `factor` bits of lane 0 are zero" — and `bit_dec` enforces canonicity as
    // part of the row, which is exactly the `canonical()` the host applies.
    let [lane0, _, _, _] = b.unpack(digest);
    let bits = b.bit_dec(lane0, 64);
    for bit in bits.iter().rev().take(factor as usize) {
        b.assert_eq(Felt(bit.addr()), zero);
    }
}

/// `nonce.to_be_bytes()` as the two `u32` halves the transcript absorbs.
///
/// The transcript reads halves as four LITTLE-endian bytes, so the big-endian
/// rendering is the felt's two halves in reversed ORDER, each byte-swapped —
/// which is exactly what `felt_be_halves` produces.
pub(super) fn nonce_halves(b: &mut LfmBuilder, nonce: Felt) -> [Felt; 2] {
    super::transcript_replay::felt_be_halves(b, nonce)
}

/// Replay one table's rounds 2 to 4 against a FORKED transcript.
///
/// `t` must be the fork ([`fork_table`]), not the shared transcript. Returns
/// the challenges as cells; every absorbed value came from the caller.
pub fn emit_table_challenges(
    b: &mut LfmBuilder,
    t: &mut TranscriptReplay,
    shape: &TableChallengeShape,
    absorbs: &TableAbsorbs<'_>,
) -> TableChallenges {
    shape.check();
    assert_eq!(
        absorbs.aux_root.is_some(),
        shape.has_aux_root,
        "the aux root's presence is shape"
    );
    assert_eq!(
        absorbs.contribution.is_some(),
        shape.has_contribution,
        "the contribution's presence is shape"
    );
    assert_eq!(
        absorbs.ood_current.len(),
        shape.ood_current_dims.0 * shape.ood_current_dims.1,
        "the current-row OOD block must match its declared dimensions"
    );
    assert_eq!(
        absorbs.ood_next.len(),
        shape.ood_next_dims.0 * shape.ood_next_dims.1,
        "the next-row OOD block must match its declared dimensions"
    );
    assert_eq!(absorbs.parts.len(), shape.num_parts, "one cell per part");
    assert_eq!(
        absorbs.fri_roots.len(),
        shape.fri.num_committed(),
        "one root per committed FRI layer"
    );
    assert_eq!(
        absorbs.fri_coeffs.len(),
        shape.fri.num_terminal_coeffs(),
        "the terminal polynomial's coefficient count is shape"
    );
    assert_eq!(
        absorbs.nonce.is_some(),
        shape.grinding_factor > 0,
        "a nonce exists exactly when grinding is on"
    );

    // ---- Phase C and the contribution bind, inside the fork.
    if let Some(root) = absorbs.aux_root {
        root.absorb(b, t);
    }
    if let Some(l) = absorbs.contribution {
        append_ext_cell(b, t, l);
    }

    // ---- Round 2: β, then the composition root.
    let beta = t.sample_ext(b);
    absorbs.composition_root.absorb(b, t);

    // ---- Round 3: z, then both OOD blocks COLUMN-major, then the parts.
    let z = emit_z_ood(b, t, shape);
    for (dims, block) in [
        (shape.ood_current_dims, absorbs.ood_current),
        (shape.ood_next_dims, absorbs.ood_next),
    ] {
        let (width, height) = dims;
        for col in 0..width {
            for row in 0..height {
                append_ext_cell(b, t, block[row * width + col]);
            }
        }
    }
    for part in absorbs.parts {
        append_ext_cell(b, t, *part);
    }

    // ---- Round 4: γ, the interleaved FRI commit phase, then the queries.
    let gamma = t.sample_ext(b);

    let mut zetas = Vec::with_capacity(shape.fri.num_committed() + 1);
    for root in absorbs.fri_roots {
        // Sample FIRST, absorb SECOND — a ζ drawn after its own layer root is a
        // challenge the prover answers rather than one that binds them.
        zetas.push(t.sample_ext(b));
        root.absorb(b, t);
    }
    if shape.fri.total_folds() > 0 {
        zetas.push(t.sample_ext(b));
    }
    for c in absorbs.fri_coeffs {
        append_ext_cell(b, t, *c);
    }

    if let Some(nonce) = absorbs.nonce {
        let seed = t.state(b);
        emit_grinding_check(b, seed, nonce, shape.grinding_factor);
        // `append_felt` IS `append_halves(&felt_be_halves(nonce))` on the byte
        // arm — the same two halves this used to build by hand — and on the
        // algebraic arm it is the free lane-0 cell.
        t.append_felt(b, nonce);
    }

    let iota_bits = (0..shape.num_queries)
        .map(|_| t.sample_u64_pow2(b, shape.index_bits()))
        .collect();

    TableChallenges {
        beta,
        z,
        gamma,
        zetas,
        iota_bits,
    }
}

/// Rebuild the full OOD grid from the two pruned blocks the proof carries.
///
/// The in-machine analogue of `ood::reconstruct_ood_full`
/// (`crypto/stark/src/ood.rs`), and the seam between the spine and the two legs
/// that fold the grid: the same cells [`emit_table_challenges`] absorbed
/// column-major come back here as `num_eval_points × num_total_cols` rows, which
/// is the shape `constraints::emit_analyzed` reads `Op::Var{offset, col}` out of
/// and the shape `deep::emit_deep_invariants` sums.
///
/// ## The zeros are program text, not arena data
///
/// A pruned next-row entry is reconstructed as ZERO by the real verifier — no
/// transition constraint reads a pruned column at the next row, and DEEP pairs
/// those positions with zero coefficients. Emitting the pooled zero constant
/// makes the pruning part of the program rather than a property of the supplied
/// arena, which is the standing decision ("next-row pruning likewise, because the
/// verifier reconstructs an undeclared column as ZERO"). The permissive
/// direction is the dangerous one: a machine that hinted a value into a pruned
/// slot would fold a frame the real verifier cannot see.
///
/// This emits no instruction beyond interning that zero — it is cell plumbing,
/// which is the point. The blocks are the transcript's own cells, so there is no
/// second copy of the grid for a prover to disagree with.
pub fn emit_reconstruct_ood(
    b: &mut LfmBuilder,
    deep: &super::deep::DeepShape,
    current: &[Ext],
    next: &[Ext],
) -> Vec<Vec<Ext>> {
    let width = deep.num_total_cols;
    let mask_width = deep.next_row_cols.len();
    let next_rows = deep
        .num_eval_points
        .checked_sub(deep.step_size)
        .expect("the OOD grid is at least the current-row block");
    assert_eq!(
        current.len(),
        deep.step_size * width,
        "the current-row block is step_size × num_total_cols"
    );
    assert_eq!(
        next.len(),
        next_rows * mask_width,
        "the next-row block is (num_eval_points − step_size) × |next_row_cols|"
    );

    let zero = b.felt_const(FE::zero()).as_ext();
    let mut rows = Vec::with_capacity(deep.num_eval_points);
    for r in 0..deep.step_size {
        rows.push(current[r * width..(r + 1) * width].to_vec());
    }
    for r in 0..next_rows {
        let mut row = vec![zero; width];
        for (m, &col) in deep.next_row_cols.iter().enumerate() {
            assert!(
                col < width,
                "a next-row column must index into the trace: {col} against {width}"
            );
            row[col] = next[r * mask_width + m];
        }
        rows.push(row);
    }
    rows
}

/// Absorb an extension cell the way `append_field_element` streams it: three
/// coordinates, each eight big-endian bytes.
pub(super) fn append_ext_cell(b: &mut LfmBuilder, t: &mut TranscriptReplay, v: Ext) {
    let coords = b.unpack(v.as_cell());
    t.append_ext(b, [coords[0], coords[1], coords[2]]);
}

/// Constrain `v < 2^32` — the felt-width guard assembly ledger entry 1 owes.
///
/// An LFM arena is untyped felts; production's `reg_fini` and `register_init` are
/// `Vec<u32>`, and that TYPE is the whole enforcement on their side. So without
/// this the machine's accepted set is strictly wider than production's: a prover
/// could derive the REGISTER preprocessed commitment over a boundary value no
/// production epoch can hold.
///
/// One `BitDec` plus one recomposition: `bit_dec` exposes the low 32 bits, and
/// asserting `v` equals their weighted sum has no witness above `2^32 − 1`. The
/// same idiom `emit_output_bytes` uses on the public-output halves, and for the
/// same reason.
///
/// Deliberately at the ASSEMBLY call site rather than inside
/// `programs::emit_register_commitment`: the isolated derivation's width gap is
/// pinned by a guard test that asserts the hazard still exists
/// (`machine_tests::the_derivation_extends_a_non_u32_register_value_demonstrating_
/// hazard`), which is correct — an isolated derivation binds nothing, and closing
/// the gap is what assembly is for.
pub fn assert_u32(b: &mut LfmBuilder, v: Felt) {
    let bits = b.bit_dec(v, 32);
    let two = b.felt_const(FE::from(2u64));
    let mut acc = Felt(bits[31].addr());
    for k in (0..31).rev() {
        let bit = Felt(bits[k].addr());
        acc = b.mul_add(acc, two, bit);
    }
    b.assert_eq(v, acc);
}

/// `β⁰ .. β^{n−1}` — production's `compute_alpha_powers(&beta, n)`, which the
/// quotient fold consumes as transition coefficients then boundary ones.
///
/// Derived in-machine from the ONE `β` the transcript produced, for the reason
/// `constraints::emit_alpha_powers` exists: a hinted power run is a prover's
/// free choice of every constraint coefficient.
pub fn emit_beta_powers(b: &mut LfmBuilder, beta: Ext, n: usize) -> Vec<Ext> {
    super::constraints::emit_alpha_powers(b, beta, n)
}

/// The public output as one cell per BYTE, derived from the `u32` halves the
/// statement absorbed.
///
/// The output has two consumers with incompatible shapes: the statement absorb
/// wants four-byte halves, and the COMMIT-bus target folds one term per byte
/// (`logup::emit_commit_bus_target`). Two arenas would be two claims about the
/// same string — the assembly obligation's fifth instance. So the halves are
/// the arena, and the bytes are DERIVED here.
///
/// Each half costs one `BitDec` and one `MulAdd`, and the recomposition assert
/// is what makes it a range check as well: a hinted half at or above `2^32`
/// cannot equal the sum of the four bytes read out of its low 32 bits, so the
/// program is unprovable rather than absorbing one string and folding another.
///
/// A trailing partial half is masked the same way the transcript masks it: only
/// `len_bytes` bytes are returned, and the unused high bytes of the last half
/// are pinned to zero — without that a prover could absorb one string while the
/// length prefix claimed another.
pub fn emit_output_bytes(b: &mut LfmBuilder, halves: &[Felt], len_bytes: usize) -> Vec<Felt> {
    assert_eq!(
        halves.len(),
        len_bytes.div_ceil(4),
        "one half per four output bytes, the last one partial"
    );
    let mut out = Vec::with_capacity(len_bytes);
    let zero = b.felt_const(FE::zero());
    for (h, half) in halves.iter().enumerate() {
        let bits = b.bit_dec(*half, 32);
        let bytes: Vec<Felt> = (0..4)
            .map(|k| super::edsl::bits_to_felt(b, &bits[8 * k..8 * k + 8]))
            .collect();
        // half = Σ byteₖ·2^{8k}, which pins the half to its four bytes AND to
        // the range `[0, 2^32)`.
        let mut acc = bytes[3];
        for k in (0..3).rev() {
            let shift = b.felt_const(FE::from(256u64));
            acc = b.mul_add(acc, shift, bytes[k]);
        }
        b.assert_eq(*half, acc);

        for (k, byte) in bytes.into_iter().enumerate() {
            if 4 * h + k < len_bytes {
                out.push(byte);
            } else {
                b.assert_eq(byte, zero);
            }
        }
    }
    out
}
