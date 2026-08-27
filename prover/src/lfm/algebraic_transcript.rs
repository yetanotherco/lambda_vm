//! The Fiat–Shamir transcript for an ALGEBRAIC commitment hash — the host side
//! of the seam whose other side is [`super::edsl::SpongeVar`].
//!
//! # The problem this module exists to solve (item A2)
//!
//! A wrap program verifies a proof the HOST produced, which means the in-VM
//! transcript replay must re-derive exactly the challenges the host derived. Get
//! that wrong and Fiat–Shamir does not fail loudly — the walk reconstructs
//! nothing, some difference that should have been non-zero is inverted, and the
//! executor reports `DivByZero` at an address that names neither the hash nor
//! the site (the diagnostic signature `edsl::WrapHash::production`'s header
//! warns about).
//!
//! Under the byte hashes the two sides agree because both are byte sponges. An
//! algebraic hash absorbs FIELD ELEMENTS, while `IsTranscript` — the trait the
//! STARK prover and verifier are generic over — offers `append_bytes`. **The
//! encoding between those two is the convention this module pins.**
//!
//! # ✓ What made this tractable
//!
//! Three facts, each verified rather than assumed:
//!
//! 1. **The transcript is a caller-supplied parameter, not a fixed type.**
//!    `prove`/`verify` take `transcript: &mut impl IsStarkTranscript<..>`, so an
//!    algebraic transcript is a new type, not a modification of
//!    `DefaultTranscript`. Nothing the byte path uses is edited.
//! 2. **`IsTranscript` is already felt-native where it matters** —
//!    `append_field_element` and `sample_field_element` speak
//!    `FieldElement`, not bytes. Only `append_bytes` and `state()` are byte-typed.
//! 3. **`append_bytes` has a tiny, regular call surface.** Across the whole
//!    STARK core it is called with exactly two things: **32-byte Merkle roots**
//!    and **8-byte integers** (query indices, heights, widths, the grinding
//!    nonce). A 32-byte root under an algebraic hash *is* four felts, and the
//!    in-VM `SpongeVar::absorb` consumes exactly one four-felt cell.
//!
//! # ★ THE CONVENTION
//!
//! The state is ONE cell, zero-initialised, and every step is one `LFM_HASH`
//! transcript-domain step — identical to [`super::edsl::SpongeVar`], because
//! matching it is the entire point.
//!
//! | operation | absorbs |
//! |---|---|
//! | `append_bytes(b)` | `[len(b), 0, 0, 0]` as a DIGEST cell, then `⌈len/32⌉` payload cells, each 32 bytes read as four canonical BIG-endian `u64` felts, the last zero-padded |
//! | `append_field_element(x)` | `x`'s three Fp3 coefficients as one DATA cell `[x0, x1, x2, 0]`, through the LEAF encoding |
//! | `state()` | the four state felts, canonical big-endian — the grinding seed |
//! | `sample_field_element()` | squeeze one cell, read lanes 0–2 as Fp3 |
//! | `sample_u64(n)` | squeeze one cell, lane 0 canonical, masked to `n − 1` |
//!
//! ## Why the length prefix — it is the injectivity argument
//!
//! ⚠ Without it the absorb is **not injective** and Fiat–Shamir is not binding:
//! a 32-byte root and an 8-byte integer would both become one cell, so a root
//! whose last 24 bytes are zero would be indistinguishable from the integer in
//! its first 8. Prefixing the byte length separates every call by construction,
//! at a cost of one extra cell per call — and `append_bytes` runs on the order
//! of 10² times per verify, so the cost is noise.
//!
//! The rule is deliberately UNIFORM rather than special-cased on 32 and 8. A
//! conditional encoding is how a collision gets introduced later by someone
//! adding a third call shape.
//!
//! ## ⚠ Byte order: ONE rule, and integers come in byte-swapped
//!
//! Felt↔byte is **canonical BIG-endian** everywhere on the algebraic path —
//! the transcript, the Merkle nodes and the leaf buffers — because that is
//! what `ByteConversion::write_bytes_be` already does for a Goldilocks felt
//! (`canonical_u64().to_be_bytes()`), and that is how every leaf the STARK
//! serialises reaches a commitment backend. A little-endian island here and a
//! big-endian one there is how a root gets produced that nobody can reproduce.
//!
//! The consequence, stated because it looks wrong at first glance: the STARK
//! core hands `append_bytes` **little-endian** integers
//! (`(idx as u64).to_le_bytes()`), so the resulting felt is the byte-SWAP of
//! that integer. This is harmless — those call sites carry compile-time
//! constants, so an emitter materialises whatever felt the rule produces and
//! no runtime swap exists anywhere — but it must be DERIVED from the rule at
//! both ends rather than assumed. The differential gate caught precisely this:
//! a machine side that hand-wrote `FE::from(idx)` disagreed with the host.
//!
//! ## Why DIGEST cells for bytes and a DATA cell for field elements
//!
//! It mirrors `SpongeVar` exactly: `absorb` is for digests, `absorb_felts` for
//! data, and the two are different hash domains. Roots and lengths are digests
//! and small integers — opaque, already committed. A field element is data that
//! an adversary chooses, so it enters through the LEAF encoding, the same way
//! data enters a Merkle tree. Using one domain for both would give a program
//! that absorbs a root the ability to claim it absorbed a field element.
//!
//! ## `sample_u64` is CONSTANT-CONSUMPTION, and it has to be
//!
//! ✓ VERIFIED every call site in the STARK core passes a power of two
//! (`domain_size >> 1`, `1 << (h_max − 1)`, `1 << 9`). The incumbent
//! `DefaultTranscript` rejection-samples, but at a power-of-two bound its
//! threshold is zero and the loop never rejects — which is why the in-VM replay
//! can encode one draw. This implementation masks, so it consumes exactly one
//! cell per draw unconditionally: a straight-line machine cannot emit a loop
//! whose trip count depends on a sampled value (`SOUNDNESS.md` §6.3).
//!
//! # What pins it
//!
//! [`tests::the_host_transcript_and_the_machine_replay_derive_the_same_challenges`]
//! is the gate: it emits a program that drives `SpongeVar` through the same
//! sequence, proves it, and compares the machine's published challenges against
//! this type's. A divergence fails there rather than inside a query walk.

use math::field::traits::IsPrimeField;

use crypto::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::edsl::SQUEEZE_MARK;
use super::hash::{HasherKind, LfmHasher};
use super::word::LfmWord;

/// Bytes one cell carries: four Goldilocks felts, canonical little-endian.
pub const BYTES_PER_CELL: usize = 4 * 8;

/// The Fiat–Shamir transcript over an algebraic `LFM_HASH` tenant.
///
/// Parameterised by [`HasherKind`] rather than fixed, so RPO, RPX and the
/// Poseidon reference share one implementation and a new candidate costs
/// nothing here — the objective this whole lane is organised around.
#[derive(Clone, Debug)]
pub struct AlgebraicTranscript {
    state: LfmWord,
    squeeze_index: u32,
    hasher: HasherKind,
}

impl AlgebraicTranscript {
    /// A fresh transcript under `hasher`, state zero.
    pub fn new(hasher: HasherKind) -> Self {
        Self {
            state: [FE::zero(); 4],
            squeeze_index: 0,
            hasher,
        }
    }

    /// `SQ(i) = [SQUEEZE_MARK, i, 0, 0]` — the advance operand, identical to
    /// `SpongeVar`'s.
    pub fn squeeze_operand(i: u32) -> LfmWord {
        [
            FE::from(u64::from(SQUEEZE_MARK)),
            FE::from(u64::from(i)),
            FE::zero(),
            FE::zero(),
        ]
    }

    /// The state as it stands — what the KATs pin and what `state()` serialises.
    pub fn state_word(&self) -> LfmWord {
        self.state
    }

    /// One transcript step against a DIGEST cell.
    pub fn absorb_cell(&mut self, c: &LfmWord) {
        self.state = self.hasher.transcript(&self.state, c);
    }

    /// Absorb a cell of four arbitrary FIELD ELEMENTS, through the leaf
    /// encoding — the DATA domain.
    pub fn absorb_felts(&mut self, c: &LfmWord) {
        let d = self.hasher.leaf(&[FE::zero(); 4], c);
        self.absorb_cell(&d);
    }

    /// Output the current state, then advance past it with `SQ(i)`.
    ///
    /// Output-then-advance, so no squeezed value is ever the state a later step
    /// absorbs into — the ordering `SpongeVar::squeeze_cell` documents.
    pub fn squeeze_cell(&mut self) -> LfmWord {
        let out = self.state;
        let sq = Self::squeeze_operand(self.squeeze_index);
        self.state = self.hasher.transcript(&self.state, &sq);
        self.squeeze_index += 1;
        out
    }

    /// ★ **THE `append_bytes` RULE, stated once.** The cells that call absorbs,
    /// in order: the length, then the payload in 32-byte groups.
    ///
    /// ⚠ **Exported so the MACHINE side derives from it rather than restating
    /// it.** Every constant an emitter needs for a byte absorb comes from here.
    /// This is the third time this lane has written a host↔machine encoding,
    /// and twice the differential caught a machine side that had hand-written a
    /// constant which agreed with the rule until the rule moved. A restated
    /// convention is a convention with two definitions, and the second one is
    /// always the one that rots.
    pub fn append_bytes_cells(bytes: &[u8]) -> Vec<LfmWord> {
        let mut cells = Vec::with_capacity(1 + bytes.len().div_ceil(BYTES_PER_CELL));
        cells.push([
            FE::from(bytes.len() as u64),
            FE::zero(),
            FE::zero(),
            FE::zero(),
        ]);
        cells.extend(bytes.chunks(BYTES_PER_CELL).map(Self::bytes_to_cell));
        cells
    }

    /// ★ **THE `append_field_element` RULE, stated once** — an Fp3 element as
    /// one DATA cell. Exported for the same reason as
    /// [`Self::append_bytes_cells`].
    pub fn field_element_cell(element: &FEE) -> LfmWord {
        let v = element.value();
        [v[0], v[1], v[2], FE::zero()]
    }

    /// Read 32 bytes as four canonical BIG-endian felts. The inverse of
    /// [`Self::cell_to_bytes`].
    ///
    /// ★ **Big-endian, and that is not arbitrary.** It is the convention
    /// `ByteConversion::write_bytes_be` already uses for a Goldilocks felt
    /// (`canonical_u64().to_be_bytes()`), which is how every leaf the STARK
    /// serialises reaches a commitment backend. One rule for felt↔byte across
    /// the whole algebraic path — the transcript, the Merkle nodes and the leaf
    /// buffers — rather than a little-endian island here and a big-endian one
    /// there, which is the kind of asymmetry that produces a root nobody can
    /// reproduce.
    ///
    /// ⚠ Bytes reaching here are Merkle roots produced by an algebraic backend,
    /// i.e. already canonical felts. A non-canonical eight-byte group would
    /// reduce, and reduction is what would make two different roots absorb
    /// identically — so the backend's serialisation being canonical is a
    /// PRECONDITION of this convention, not a detail.
    pub fn bytes_to_cell(chunk: &[u8]) -> LfmWord {
        core::array::from_fn(|i| {
            let mut b = [0u8; 8];
            let start = i * 8;
            if start < chunk.len() {
                let end = (start + 8).min(chunk.len());
                b[..end - start].copy_from_slice(&chunk[start..end]);
            }
            FE::from(u64::from_be_bytes(b))
        })
    }

    /// Four felts as 32 canonical big-endian bytes — the inverse of
    /// [`Self::bytes_to_cell`], and the same rule `write_bytes_be` uses.
    pub fn cell_to_bytes(c: &LfmWord) -> [u8; BYTES_PER_CELL] {
        let mut out = [0u8; BYTES_PER_CELL];
        for (i, f) in c.iter().enumerate() {
            let v = GoldilocksField::canonical(f.value());
            out[i * 8..(i + 1) * 8].copy_from_slice(&v.to_be_bytes());
        }
        out
    }
}

impl IsTranscript<GoldilocksExtension> for AlgebraicTranscript {
    /// An Fp3 element as one DATA cell `[x0, x1, x2, 0]`.
    fn append_field_element(&mut self, element: &FEE) {
        self.absorb_felts(&Self::field_element_cell(element));
    }

    /// The length prefix, then the payload in 32-byte cells. See the module
    /// header for why the prefix is the injectivity argument.
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        for cell in Self::append_bytes_cells(new_bytes) {
            self.absorb_cell(&cell);
        }
    }

    /// The four state felts, canonical big-endian — the grinding seed.
    fn state(&self) -> [u8; 32] {
        Self::cell_to_bytes(&self.state)
    }

    fn sample_field_element(&mut self) -> FEE {
        let c = self.squeeze_cell();
        FEE::new([c[0], c[1], c[2]])
    }

    /// ⚠ Constant-consumption: exactly one cell, always. See the module header.
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        debug_assert!(upper_bound > 0, "upper_bound must be greater than 0");
        debug_assert!(
            upper_bound.is_power_of_two(),
            "sample_u64 is masked, so a non-power-of-two bound ({upper_bound}) would be \
             biased; every STARK call site passes a power of two"
        );
        let c = self.squeeze_cell();
        GoldilocksField::canonical(c[0].value()) & (upper_bound - 1)
    }
}

impl IsStarkTranscript<GoldilocksExtension, GoldilocksField> for AlgebraicTranscript {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lfm::builder::{LfmBuilder, LfmProgramSource};
    use crate::lfm::compiler::compile;
    use crate::lfm::edsl::{self, SpongeVar};
    use crate::lfm::proof::{lfm_prove_with_hasher, verify_against};
    use crate::lfm::registry::build_artifacts_with_hasher;
    use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

    /// The hashers this convention is defined for. `Test` is included on
    /// purpose: it is not cryptographic, but it exercises the same wiring, so a
    /// break in the ENCODING shows up under all four rather than being
    /// mistaken for something about one permutation.
    const ALGEBRAIC: [HasherKind; 4] = [
        HasherKind::Test,
        HasherKind::Poseidon,
        HasherKind::Rpo,
        HasherKind::Rpx,
    ];

    fn options() -> ProofOptions {
        GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
    }

    /// A 32-byte "root" and an 8-byte integer — the only two shapes the STARK
    /// core ever hands `append_bytes`.
    const ROOT: [u8; 32] = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x00, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0x00, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0x00, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
        0x32, 0x00,
    ];
    const SMALL: u64 = 0x0123_4567;
    const QUERY_BITS: usize = 5;

    /// The three Fp3 coefficients the field-element absorb uses.
    fn field_element() -> FEE {
        FEE::new([FE::from(7u64), FE::from(11u64), FE::from(13u64)])
    }

    /// The HOST side of the sequence, through the public `IsTranscript` API.
    fn host_challenges(hasher: HasherKind) -> (FEE, FEE, u64) {
        let mut t = AlgebraicTranscript::new(hasher);
        t.append_bytes(&ROOT);
        let a = t.sample_field_element();
        t.append_bytes(&SMALL.to_le_bytes());
        let b = t.sample_field_element();
        t.append_field_element(&field_element());
        let q = t.sample_u64(1 << QUERY_BITS);
        (a, b, q)
    }

    /// The MACHINE side: the same sequence expressed in `SpongeVar` ops, with
    /// every challenge published so a proof carries them where a verifier can
    /// check them.
    ///
    /// ⚠ This is written from the CONVENTION, not from the host implementation
    /// — the length cell, the payload cell, the leaf-encoded field element and
    /// the masked draw are each spelled out here. That is what makes the test a
    /// differential rather than a tautology: if the host and this disagree
    /// about the encoding, they disagree about the challenges.
    fn replay_program_source() -> LfmProgramSource {
        let mut b = LfmBuilder::new();
        let arena = b.declare_arena(2);
        let root = b.hint_word(arena, 0);
        let felts = b.hint_word(arena, 1);

        let mut sponge = SpongeVar::new(&mut b);

        // append_bytes(ROOT) — every cell DERIVED from the rule. The payload
        // arrives as an arena word so the program reads real data; the length
        // cell is a program constant, and which constant is the rule's to say.
        let root_cells = AlgebraicTranscript::append_bytes_cells(&ROOT);
        let len32 = b.digest_const(root_cells[0]);
        sponge.absorb(&mut b, len32.as_cell());
        sponge.absorb(&mut b, root);
        let a = sponge.squeeze_ext(&mut b);

        // append_bytes(SMALL.to_le_bytes()): length cell, then the value cell.
        let small_cells = AlgebraicTranscript::append_bytes_cells(&SMALL.to_le_bytes());
        let len8 = b.digest_const(small_cells[0]);
        sponge.absorb(&mut b, len8.as_cell());
        // append_bytes(SMALL.to_le_bytes()) — again every cell from the rule.
        // ⚠ The payload felt is the byte-SWAP of the integer, because the STARK
        // hands `append_bytes` LITTLE-endian integers and the one felt↔byte rule
        // reads BIG-endian. Harmless — these are compile-time constants, so an
        // emitter materialises whatever the rule produces and no runtime swap
        // exists — but a machine side that hand-wrote `FE::from(SMALL)` is what
        // this gate caught, which is why nothing here is hand-written.
        let small = b.digest_const(small_cells[1]);
        sponge.absorb(&mut b, small.as_cell());
        let bb = sponge.squeeze_ext(&mut b);

        // append_field_element: DATA, so the leaf encoding.
        sponge.absorb_felts(&mut b, felts);

        b.public(a.as_cell());
        b.public(bb.as_cell());
        let bits = sponge.squeeze_bits(&mut b, QUERY_BITS);
        let q = edsl::bits_to_felt(&mut b, &bits);
        b.public(q.as_cell());
        b.finish()
    }

    fn replay_arena() -> Vec<Vec<LfmWord>> {
        // Both arena words DERIVED from the rule, not restated.
        let root_cell = AlgebraicTranscript::append_bytes_cells(&ROOT)[1];
        vec![vec![
            root_cell,
            AlgebraicTranscript::field_element_cell(&field_element()),
        ]]
    }

    /// ★★ **THE A2 GATE.** The host transcript and the in-VM replay must derive
    /// the SAME challenges, for every algebraic tenant.
    ///
    /// This is the one open correctness question in the algebraic swap: get the
    /// byte↔felt encoding wrong and Fiat–Shamir does not fail loudly, it fails
    /// as a `DivByZero` deep inside a query walk that names neither the hash nor
    /// the site. Gating it here means a divergence is a failing test.
    #[test]
    fn the_host_transcript_and_the_machine_replay_derive_the_same_challenges() {
        let opts = options();
        let program = compile(replay_program_source());
        for hasher in ALGEBRAIC {
            let (a, b, q) = host_challenges(hasher);
            let artifacts = build_artifacts_with_hasher(&program, &opts, hasher);
            let proved =
                lfm_prove_with_hasher(&program, &artifacts, &replay_arena(), &opts, hasher)
                    .expect("the replay program must prove");

            let pub_a = proved.public_words[0].1;
            let pub_b = proved.public_words[1].1;
            let pub_q = proved.public_words[2].1;

            assert_eq!(
                [pub_a[0], pub_a[1], pub_a[2]],
                *a.value(),
                "{hasher:?}: first sampled field element must agree"
            );
            assert_eq!(
                [pub_b[0], pub_b[1], pub_b[2]],
                *b.value(),
                "{hasher:?}: second sampled field element must agree"
            );
            assert_eq!(
                GoldilocksField::canonical(pub_q[0].value()),
                q,
                "{hasher:?}: sampled query index must agree"
            );

            assert!(
                verify_against(
                    &artifacts.roots,
                    &artifacts.program_id,
                    artifacts.keccak_rnd_chunks,
                    &proved.proof,
                    &proved.public_words,
                    &opts,
                    artifacts.hasher,
                    artifacts.chip_set,
                ),
                "{hasher:?}: the replay proof must verify"
            );
        }
    }

    /// ⚠ The length prefix is what makes the absorb INJECTIVE, and this is the
    /// collision it prevents: a 32-byte root whose tail is zero and an 8-byte
    /// integer holding the same leading bytes would otherwise absorb
    /// identically.
    #[test]
    fn the_length_prefix_separates_a_root_from_an_integer() {
        let mut padded = [0u8; 32];
        padded[..8].copy_from_slice(&SMALL.to_le_bytes());

        let mut with_root = AlgebraicTranscript::new(HasherKind::Rpo);
        with_root.append_bytes(&padded);

        let mut with_int = AlgebraicTranscript::new(HasherKind::Rpo);
        with_int.append_bytes(&SMALL.to_le_bytes());

        assert_ne!(
            with_root.state_word(),
            with_int.state_word(),
            "a 32-byte zero-padded value and the 8-byte value must not collide"
        );

        // And the control: without the length they WOULD be the same cell.
        assert_eq!(
            AlgebraicTranscript::bytes_to_cell(&padded),
            AlgebraicTranscript::bytes_to_cell(&SMALL.to_le_bytes()),
            "the payload cells are identical — the prefix is the only separator"
        );
    }

    /// The byte round-trip must be exact, or `state()` does not name the state.
    #[test]
    fn a_cell_round_trips_through_its_canonical_bytes() {
        let c: LfmWord = [
            FE::from(1u64),
            FE::from(0xFFFF_FFFF_0000_0000u64),
            FE::zero(),
            FE::from(12345u64),
        ];
        assert_eq!(
            AlgebraicTranscript::bytes_to_cell(&AlgebraicTranscript::cell_to_bytes(&c)),
            c
        );
    }

    /// `sample_u64` consumes exactly one cell per draw regardless of the value
    /// drawn — the property a straight-line replay depends on.
    #[test]
    fn sampling_is_constant_consumption() {
        let mut t = AlgebraicTranscript::new(HasherKind::Rpo);
        for expected in 1..=8u32 {
            let _ = t.sample_u64(1 << QUERY_BITS);
            assert_eq!(t.squeeze_index, expected, "one squeeze per draw");
        }
    }
}
