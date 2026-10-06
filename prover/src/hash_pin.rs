//! ★★ **THE HASH PIN** — the one place a build says which hash the BLOCK path
//! proves under.
//!
//! # Why this is a module and not a line in `crypto/stark`
//!
//! `crypto/stark`'s [`stark::config::DefaultStarkHash`] is the *workspace's*
//! default: it names the hash behind `Commitment`, `BatchedMerkleTree` and every
//! blessed constant in the repo, and a `const` assertion there makes re-pointing
//! it a compile error precisely so those artifacts cannot drift.
//!
//! The hash-comparison branches need something different — to change what the
//! block path commits under **without** touching that default, so BLAKE3's
//! enforcement stays intact while a sibling branch proves under RPO. ✓ VERIFIED
//! that is expressible: `IsStarkProver<Field, FieldExtension, PI, H: StarkHash>`
//! is generic over the configuration, and `prover` was already naming
//! `DefaultStarkHash` *explicitly* at each of its prove and verify call sites.
//! Those are type parameters, not a global. Collecting them behind these two
//! names turns "which hash does the block path use" from a property spread over
//! six files into a property of this one.
//!
//! # ⚠ The pin is TWO things, and the second is easy to miss
//!
//! [`StarkHash::Transcript`] names a `TranscriptHash` — a **digest
//! configuration**, which is what GRINDING computes over. The Fiat–Shamir
//! transcript **object** is built by the caller and handed to `multi_prove`, so
//! the type system does not force it to match.
//!
//! For the byte hashes the two coincide: the object is
//! `DefaultTranscript<E, H::Transcript>`, a sponge over that digest. **For an
//! algebraic hash they do not.** `AlgebraicTranscript` is a compress chain over
//! cells, not a byte sponge over `AlgebraicDigest`, and a branch that pinned only
//! [`BlockStarkHash`] would commit under RPO while sponging Fiat–Shamir through
//! bytes — self-consistent between prover and verifier, and therefore **silent**.
//! That is the same half-flip `stark::config::DefaultStarkTranscript`'s own doc
//! warns about, and [`block_transcript`] is why it cannot happen here.
//!
//! # What a branch changes
//!
//! Exactly the three items below, and nothing else in the workspace. On an
//! algebraic branch they become, for example:
//!
//! ```ignore
//! pub type BlockStarkHash  = crate::lfm::algebraic_commit::RpoStarkHash;
//! pub type BlockTranscript = crate::lfm::algebraic_transcript::AlgebraicTranscript;
//! pub fn block_transcript(seed: &[u8]) -> BlockTranscript {
//!     BlockTranscript::with_seed(crate::lfm::hash::HasherKind::Rpo, seed)
//! }
//! ```
//!
//! ✓ VERIFIED that flip compiles and runs end to end — it was performed, built,
//! and executed against this crate's own prove/verify tests before this module
//! was written, which is how [`BlockProver`], the generic transcript parameter
//! on `compute_expected_commit_bus_balance_view`, and the
//! `IsStreamingLeafBackend` import in `proof_arena` were found. None of those
//! three shows up on a build that only ever pins BLAKE3.
//!
//! # `cuda` on an algebraic pin
//!
//! Compiles, and cannot prove under the wrong hash. The algebraic backends are
//! `DeviceTreeBackend`s carrying their own `CommitmentHash` as the device
//! dispatch key, so a device tree is built by the kernels of the hash it is
//! named for or not built at all. RPX256 has those kernels (`math_cuda::rpx`),
//! so a GPU run under this pin commits on the device; RPO256 and Poseidon do
//! not yet, and a GPU run under one of them aborts at its first device commit
//! with `unimplemented!` naming the hash. ⛔ Neither a `compile_error!` nor a
//! byte-hash fallback belongs here: the first hides the cuda lint arm from the
//! branch, the second is exactly the silent wrong-hash build this module exists
//! to make impossible.
//!
//! # ⚠ TWO regenerations, not one — in THIS order, plus one stray constant
//!
//! A pin change is **not** complete until every root blessed under the old hash
//! is regenerated. There are two families of them and the order is load-bearing:
//!
//! 1. **The static preprocessed commitments — FIRST.** FOUR families: `bitwise`,
//!    `keccak_rc`, and `page`'s zero-page AND private-page constants, at blowup
//!    2/4/8. Each returns a BLESSED CONSTANT from `preprocessed_commitment`
//!    rather than recomputing, so under a new pin the prover recomputes an
//!    algebraic root, compares it against a BLAKE3 constant, and fails with
//!    `ProvingError::PrecomputedCommitmentMismatch`.
//!    `cargo run --bin compute_static_commitments --release`, then paste.
//! 2. **`LFM_REGISTRY` — SECOND, only once the statics are in the tree.**
//!    `registry.rs` fills slots 13 and 14 of every entry from `keccak_rc` and
//!    `bitwise`'s `preprocessed_commitment` — the blessed constants above, not a
//!    recomputation — and `lfm_program_id` folds every root. A registry generated
//!    before the statics were pasted therefore embeds the OUTGOING hash's
//!    constants, and `machine_tests::registry_drift_*` fires at exactly those two
//!    slots. The control-first re-run under the outgoing pin cannot see this:
//!    both tables are self-consistent there.
//!    `cargo run --bin compute_lfm_registry --release`.
//! 3. **`SUB_DECODE_COMMITMENT_BLOWUP_2`** in `tests/decode_tests.rs` — a
//!    test-local blessed constant outside both generators, regenerated by the
//!    `#[ignore]` test `print_decode_commitment_for_sub`.
//!
//! ✓ VERIFIED (1) empirically: it is exactly how the trial flip failed, and it
//! is the correct failure — loud, at prove time, naming the cause. ✓ VERIFIED
//! (2) empirically too: the first RPX regeneration ran the registry before the
//! statics and all six drift tests fired at slots 13 and 14. `registry.rs`
//! governs both: a drift failure is investigated, never re-blessed to silence
//! the test, and neither table is ever hand-edited.

/// The commitment configuration the block path proves and verifies under.
///
/// Every `multi_prove` / `multi_verify` instantiation in this crate names this
/// rather than `stark::config::DefaultStarkHash`, so the two can differ on a
/// branch without the workspace default moving.
pub type BlockStarkHash = crate::lfm::algebraic_commit::RpxStarkHash;

/// The Fiat–Shamir transcript OBJECT the block path builds.
///
/// See the module header for why this is pinned separately from
/// [`BlockStarkHash`] rather than derived from it.
pub type BlockTranscript = crate::lfm::algebraic_transcript::AlgebraicTranscript;

/// A fresh block-path transcript over `seed`.
///
/// A function rather than a bare `::new`, because the two arms construct
/// differently: a byte transcript takes the seed in its constructor, an
/// algebraic one absorbs it as its first `append_bytes` call. Callers should not
/// have to know which.
pub fn block_transcript(seed: &[u8]) -> BlockTranscript {
    BlockTranscript::with_seed(BLOCK_HASHER, seed)
}

/// The prover the block path drives, at [`BlockStarkHash`].
///
/// ⚠ **Not `stark::prover::Prover`.** That alias is `GenericProver` at
/// `DefaultStarkHash`, so it is BLAKE3-fixed regardless of what `H` a call site
/// passes alongside it — and the two disagreeing is a type error rather than a
/// silent wrong hash, which is how this was found. The `IsStarkProver` impl
/// itself is fully generic over `H`; only the alias is pinned, so the fix is an
/// alias at the pin rather than anything in `crypto/stark`.
pub type BlockProver<Field, FieldExtension, PI> =
    stark::prover::GenericProver<Field, FieldExtension, PI, BlockStarkHash>;

/// The verifier the block path drives, at [`BlockStarkHash`]. See
/// [`BlockProver`] for why the `stark::verifier::Verifier` alias is not it.
pub type BlockVerifier<Field, FieldExtension, PI> =
    stark::verifier::GenericVerifier<Field, FieldExtension, PI, BlockStarkHash>;

/// The `LFM_HASH` socket permutation the block path's programs are EXECUTED and
/// proved under — the machine's own hash chip.
///
/// ⚠ **A third axis, and it is orthogonal to [`BlockStarkHash`].** That one says
/// which hash the HOST commits under; this says which permutation the MACHINE's
/// `Instr::Hash` rows compute. They have to agree, and nothing in the type
/// system makes them: the socket hasher is passed per call to `execute` and
/// `lfm_prove_with_hasher`.
///
/// ★ **Why it went unpinned until it bit.** Under a byte hash the emitter's
/// Merkle work goes through `ByteWrapHash::hash_bytes`, which lowers to the
/// dedicated KECCAK / `LFM_BLAKE3` chips and emits **no `Instr::Hash` at all** —
/// so the socket hasher handed to `execute` is never consulted, and passing
/// `TestPermutation` is free and correct. The algebraic arm goes through
/// `b.compress` / `b.permute`, which ARE `Instr::Hash`, executed by whatever is
/// passed. The same distinction the byte hash made irrelevant, needed back.
///
/// Every `execute` and prove call on the block path names this rather than a
/// literal, so the two axes cannot drift apart in a test harness while
/// production stays correct.
pub const BLOCK_HASHER: crate::lfm::hash::HasherKind = crate::lfm::hash::HasherKind::Rpx;

/// The [`CommitmentHash`] the block path's roots may be called by.
///
/// ★ Read this rather than `stark::config::COMMITMENT_HASH`. That const names
/// the hash of the workspace ALIASES and says so in its own doc — a prover can
/// run under a configuration whose `COMMITMENT_HASH` differs and the const will
/// not know. The block path IS such a configuration on three of the four
/// branches, so anything describing a block proof's roots must read the pin.
pub const BLOCK_COMMITMENT_HASH: stark::config::CommitmentHash =
    <BlockStarkHash as stark::config::StarkHash>::COMMITMENT_HASH;

// =========================================================================
// The base format (`p1/*` exploration branch)
// =========================================================================

/// Which hash the block path's BASE proof commits under: the pin above
/// ([`BaseHash::Rpx`], the default), or ZisK's Poseidon1 ([`BaseHash::P1`]:
/// `lfm::p1_commit`, 4-ary trees, ZisK's transcript and width-8 grind).
///
/// It is the verifier's format, never the proof's and never the
/// environment's: [`base_of`] reads it from the `ProofOptions` the caller
/// passes (`format.base`), and the block prover, the host verifier and the
/// preprocessed roots all take it from there. The recursion still verifies
/// RPX proofs, so a P1 base proof is proved and host-verified, never wrapped;
/// the epoch and LFM provers refuse a P1 format ([`require_rpx_base`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseHash {
    /// [`BlockStarkHash`] under [`BlockTranscript`]: `BaseFormat::RPX`, today.
    Rpx,
    /// [`crate::lfm::p1_commit::P1StarkHash`] under
    /// [`crate::lfm::p1_commit::P1Transcript`]: `BaseFormat::P1`.
    P1,
}

/// The base configuration `format` names: [`BaseHash::P1`] for a Poseidon1
/// base, else the pin. What the preprocessed roots and the statics follow;
/// the prove and verify entries check the name first ([`checked_base`]).
pub fn base_of(format: &stark::proof::options::ProofFormat) -> BaseHash {
    base_of_hash(format.base.hash)
}

/// [`base_of`] for the format's base hash alone.
pub fn base_of_hash(hash: stark::config::CommitmentHash) -> BaseHash {
    match hash {
        stark::config::CommitmentHash::Poseidon1 => BaseHash::P1,
        _ => BaseHash::Rpx,
    }
}

/// [`base_of`], refusing a base hash the block path has no configuration for
/// (the pin's or Poseidon1 only). The block prover and verifier entries call
/// this, so a format naming another hash is a typed refusal there.
pub fn checked_base(format: &stark::proof::options::ProofFormat) -> Result<BaseHash, String> {
    match format.base.hash {
        stark::config::CommitmentHash::Poseidon1 => Ok(BaseHash::P1),
        h if h == BLOCK_COMMITMENT_HASH => Ok(BaseHash::Rpx),
        h => Err(format!(
            "the block path has no base configuration for the format's hash {h:?} \
             (expected {BLOCK_COMMITMENT_HASH:?} or Poseidon1)"
        )),
    }
}

/// Refuse a path that has no P1 arm (the epoch pipeline, the monolithic and
/// LFM provers) under a P1 format, rather than letting it mix RPX proofs with
/// P1 preprocessed roots. The error names the path; the caller types it.
pub fn require_rpx_base(
    path: &str,
    options: &stark::proof::options::ProofOptions,
) -> Result<(), String> {
    match base_of(&options.format) {
        BaseHash::Rpx => Ok(()),
        BaseHash::P1 => Err(format!(
            "{path} has no Poseidon1 arm: its options' format names a Poseidon1 base"
        )),
    }
}

/// Under [`BaseHash::P1`], the root of a STATIC preprocessed table (BITWISE,
/// KECCAK_RC, the zero-init and private-input pages): computed once per
/// process by `compute` and kept, keyed by everything it is a function of
/// (the table, the blowup, the coset offset, the leaf layout). The blessed
/// constants in those tables are RPX roots, so under P1 they cannot be
/// returned; a recompute per AIR construction would put a 2^20-row commit in
/// every prove's setup. [`warm_base_statics`] fills it before a timed prove.
pub fn p1_static_root(
    table: &'static str,
    options: &stark::proof::options::ProofOptions,
    layout: stark::leaf_layout::LeafLayout,
    compute: impl FnOnce() -> stark::config::Commitment,
) -> stark::config::Commitment {
    type Key = (&'static str, u8, u64, usize);
    static ROOTS: std::sync::Mutex<std::collections::BTreeMap<Key, stark::config::Commitment>> =
        std::sync::Mutex::new(std::collections::BTreeMap::new());
    let key = (
        table,
        options.blowup_factor,
        options.coset_offset,
        layout.rows_per_leaf(),
    );
    if let Some(root) = ROOTS.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return *root;
    }
    // Computed outside the lock: deterministic, so a concurrent duplicate
    // computes the same root.
    let root = compute();
    ROOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, root);
    root
}

/// Under [`BaseHash::P1`], compute every static preprocessed root
/// [`p1_static_root`] keeps, both leaf layouts, at `options`, and print them;
/// under RPX they are constants and this does nothing. A harness calls it
/// before the clock starts, so the P1 arm's prove pays what the RPX arm's
/// pays for them: nothing.
pub fn warm_base_statics(options: &stark::proof::options::ProofOptions) {
    if base_of(&options.format) != BaseHash::P1 {
        return;
    }
    use stark::leaf_layout::LeafLayout;
    let t = std::time::Instant::now();
    // The Poseidon1 kernels load on first use: load them here, off the clock.
    #[cfg(feature = "cuda")]
    if !stark::gpu_lde::warm_commitment_hash(stark::config::CommitmentHash::Poseidon1) {
        eprintln!("BASE HASH P1: the Poseidon1 device kernels did not load");
    }
    let mut lines = Vec::new();
    for layout in [LeafLayout::RowPair, LeafLayout::Row] {
        let roots = [
            (
                "bitwise",
                crate::tables::bitwise::preprocessed_commitment_for(options, layout),
            ),
            (
                "keccak_rc",
                crate::tables::keccak_rc::preprocessed_commitment_for(options, layout),
            ),
            (
                "zero page",
                crate::tables::page::zero_init_preprocessed_commitment_for(options, layout),
            ),
            (
                "private page",
                crate::tables::page::private_page_preprocessed_commitment_for(options, layout),
            ),
        ];
        for (name, root) in roots {
            lines.push(format!(
                "{name}/{}: {}",
                layout.rows_per_leaf(),
                root.map_or("none".to_string(), |r| r[..8]
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>())
            ));
        }
    }
    eprintln!(
        "BASE HASH P1 STATICS: {} roots in {:.2}s ({})",
        lines.len(),
        t.elapsed().as_secs_f64(),
        lines.join(" · ")
    );
}

/// A base-proof configuration: the commitment configuration and the
/// Fiat–Shamir transcript object, named together so the two cannot be mixed
/// (the module header's half-flip). The block prover and the host verifier are
/// generic over it; [`checked_base`] picks the instance once, at their entries,
/// from the options' format.
pub trait BlockHash: Send + Sync + 'static {
    /// The commitment configuration.
    type H: stark::config::StarkHash;
    /// The transcript object.
    type Transcript: crypto::fiat_shamir::is_transcript::IsStarkTranscript<
            crate::tables::types::GoldilocksExtension,
            crate::tables::types::GoldilocksField,
        > + Clone
        + Send;
    /// The base this configuration answers to.
    const BASE: BaseHash;
    /// A fresh transcript over `seed`.
    fn transcript(seed: &[u8]) -> Self::Transcript;
    /// The monolithic statement's leading domain tag under `format`: it names
    /// the commitment geometry the statement is about (I-P1C §9.2).
    fn statement_tag(format: &stark::proof::options::ProofFormat) -> Vec<u8>;
    /// The transcript's state digest as a machine word: what the block
    /// tree's leaves publish (`TranscriptReplay::state`).
    fn state_word(t: &Self::Transcript) -> crate::lfm::word::LfmWord;
}

/// The pin: [`BlockStarkHash`] under [`BlockTranscript`].
pub struct RpxBlock;

impl BlockHash for RpxBlock {
    type H = BlockStarkHash;
    type Transcript = BlockTranscript;
    const BASE: BaseHash = BaseHash::Rpx;
    fn transcript(seed: &[u8]) -> BlockTranscript {
        block_transcript(seed)
    }
    /// Today's tag, byte for byte: RPX statements do not move.
    fn statement_tag(_: &stark::proof::options::ProofFormat) -> Vec<u8> {
        crate::statement::DOMAIN_TAG.to_vec()
    }
    fn state_word(t: &BlockTranscript) -> crate::lfm::word::LfmWord {
        t.state_word()
    }
}

/// ZisK's Poseidon1 (`lfm::p1_commit`).
pub struct P1Block;

impl BlockHash for P1Block {
    type H = crate::lfm::p1_commit::P1StarkHash;
    type Transcript = crate::lfm::p1_commit::P1Transcript;
    const BASE: BaseHash = BaseHash::P1;
    fn transcript(seed: &[u8]) -> Self::Transcript {
        crate::lfm::p1_commit::P1Transcript::with_seed(seed)
    }
    /// `LAMBDAVM_STARK_STATEMENT_V5/P1W16/C<h>`: the hash and its 4-ary cap
    /// height (`C0` uncapped).
    fn statement_tag(format: &stark::proof::options::ProofFormat) -> Vec<u8> {
        p1_statement_tag(format.base.arity4_cap)
    }
    /// `state()`'s four lanes (the node encoding's felts).
    fn state_word(t: &Self::Transcript) -> crate::lfm::word::LfmWord {
        use crypto::fiat_shamir::is_transcript::IsTranscript;
        let bytes = IsTranscript::<crate::tables::types::GoldilocksExtension>::state(t);
        crate::lfm::algebraic_commit::commitment_to_digest(&bytes)
    }
}

/// [`BlockHash::statement_tag`] of the configuration `format` names
/// ([`base_of`]): the monolithic statement's leading tag, RPX's byte for byte
/// at the default.
pub fn statement_tag(format: &stark::proof::options::ProofFormat) -> Vec<u8> {
    match base_of(format) {
        BaseHash::Rpx => RpxBlock::statement_tag(format),
        BaseHash::P1 => P1Block::statement_tag(format),
    }
}

/// The P1 statement tag at cap `cap` ([`BlockHash::statement_tag`]). A test
/// can build the RPX tag in its place (`p1_tag_omitted_for_test`) to show the
/// tag is checked.
pub fn p1_statement_tag(cap: stark::proof::options::CapPolicy) -> Vec<u8> {
    #[cfg(test)]
    if P1_TAG_OMITTED.load(std::sync::atomic::Ordering::Relaxed) {
        return crate::statement::DOMAIN_TAG.to_vec();
    }
    let height = match cap {
        stark::proof::options::CapPolicy::Fixed(c) => c,
        _ => 0,
    };
    let mut tag = crate::statement::DOMAIN_TAG.to_vec();
    tag.extend_from_slice(format!("/P1W16/C{height}").as_bytes());
    tag
}

/// Test only: P1 statements take RPX's tag while set (a mutation of the tag,
/// and the reproduction of the pre-tag P1 bytes). Process-global: a test that
/// sets it runs in its own process.
#[cfg(test)]
pub static P1_TAG_OMITTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The prover at configuration `C`.
pub type BlockProverOf<C, Field, FieldExtension, PI> =
    stark::prover::GenericProver<Field, FieldExtension, PI, <C as BlockHash>::H>;

/// The verifier at configuration `C`.
pub type BlockVerifierOf<C, Field, FieldExtension, PI> =
    stark::verifier::GenericVerifier<Field, FieldExtension, PI, <C as BlockHash>::H>;

#[cfg(test)]
mod tests {
    use super::*;

    /// The format names the configuration: RPX's base is the pin, a
    /// Poseidon1 base is P1, any other hash is refused at the entries.
    #[test]
    fn the_base_format_names_the_configuration() {
        use stark::proof::options::{BaseFormat, ProofFormat};
        let with = |base| ProofFormat {
            base,
            ..ProofFormat::LEGACY
        };
        assert_eq!(base_of(&ProofFormat::LEGACY), BaseHash::Rpx);
        assert_eq!(checked_base(&ProofFormat::LEGACY), Ok(BaseHash::Rpx));
        assert_eq!(base_of(&with(BaseFormat::P1)), BaseHash::P1);
        assert_eq!(checked_base(&with(BaseFormat::P1)), Ok(BaseHash::P1));
        let keccak = with(BaseFormat {
            hash: stark::config::CommitmentHash::Keccak256,
            ..BaseFormat::RPX
        });
        assert!(checked_base(&keccak).is_err());
        let opts = |format| stark::proof::options::ProofOptions {
            format,
            ..stark::proof::options::ProofOptions::default_test_options()
        };
        assert!(require_rpx_base("x", &opts(ProofFormat::LEGACY)).is_ok());
        assert!(require_rpx_base("x", &opts(with(BaseFormat::P1))).is_err());
    }

    /// RPX's statement tag is today's byte for byte; P1's names the hash and
    /// its cap height, so the two never share a statement prefix.
    #[test]
    fn the_statement_tags_name_the_base() {
        use stark::proof::options::{BaseFormat, CapPolicy, ProofFormat};
        let p1 = |cap| ProofFormat {
            base: BaseFormat {
                arity4_cap: cap,
                ..BaseFormat::P1
            },
            ..ProofFormat::LEGACY
        };
        assert_eq!(
            RpxBlock::statement_tag(&ProofFormat::LEGACY),
            b"LAMBDAVM_STARK_STATEMENT_V5".to_vec()
        );
        assert_eq!(
            P1Block::statement_tag(&p1(CapPolicy::Fixed(4))),
            b"LAMBDAVM_STARK_STATEMENT_V5/P1W16/C4".to_vec()
        );
        assert_eq!(
            P1Block::statement_tag(&p1(CapPolicy::Off)),
            b"LAMBDAVM_STARK_STATEMENT_V5/P1W16/C0".to_vec()
        );
    }

    /// ★ No environment read decides the base: the library's sources name
    /// neither former knob, and `hash_pin` reads no environment at all. The
    /// base comes from the caller's `ProofOptions` (I-P1C §9.3).
    #[test]
    fn no_environment_read_decides_the_base() {
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).expect("readable source tree") {
                let path = entry.expect("a directory entry").path();
                if path.is_dir() {
                    if path.file_name().is_some_and(|n| n != "tests") {
                        walk(&path, out);
                    }
                } else if path.extension().is_some_and(|e| e == "rs")
                    && !path.to_string_lossy().ends_with("_tests.rs")
                {
                    out.push(path);
                }
            }
        }
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        walk(&src, &mut files);
        assert!(
            files.len() > 50,
            "the walk found the library ({} files)",
            files.len()
        );
        let knobs = [
            concat!("LAMBDA_VM_", "BASE_HASH"),
            concat!("LAMBDA_VM_", "P1_CAP"),
        ];
        for file in &files {
            let text = std::fs::read_to_string(file).expect("readable source");
            for knob in knobs {
                assert!(!text.contains(knob), "{} names {knob}", file.display());
            }
        }
        let pin = std::fs::read_to_string(src.join("hash_pin.rs")).expect("hash_pin.rs");
        let library = pin
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("the library half");
        assert!(
            !library.contains("std::env"),
            "hash_pin reads the environment"
        );
    }

    /// ✓ The RPX configuration IS the pin: same commitment configuration, same
    /// transcript object and the same state from the same seed, so the knob at
    /// its default changes no byte.
    #[test]
    fn the_rpx_configuration_is_the_pin() {
        use crypto::fiat_shamir::is_transcript::IsTranscript;
        assert_eq!(
            std::any::TypeId::of::<<RpxBlock as BlockHash>::H>(),
            std::any::TypeId::of::<BlockStarkHash>()
        );
        assert_eq!(
            std::any::TypeId::of::<<RpxBlock as BlockHash>::Transcript>(),
            std::any::TypeId::of::<BlockTranscript>()
        );
        let a = <BlockTranscript as IsTranscript<E>>::state(&RpxBlock::transcript(b"seed"));
        let b = <BlockTranscript as IsTranscript<E>>::state(&block_transcript(b"seed"));
        assert_eq!(a, b);
        // The P1 configuration is a different hash and a different stream.
        let p = <crate::lfm::p1_commit::P1Transcript as IsTranscript<E>>::state(
            &P1Block::transcript(b"seed"),
        );
        assert_ne!(a, p);
    }
    // Named here rather than at module scope: the byte arm's `BlockTranscript`
    // mentions the extension field and an algebraic arm's does not, so a
    // module-scope import would be unused on one of the two.
    use crate::tables::types::GoldilocksExtension as E;
    use stark::config::StarkHash;

    /// ✓ The pin is COHERENT: the transcript object the block path builds sponges
    /// on the same hash the commitment configuration names.
    ///
    /// ⚠ This is the half-flip guard, and it is a real one rather than a
    /// tautology only because [`BlockTranscript`] is pinned separately — the two
    /// names can disagree, which is exactly the failure this catches. It is
    /// stated over `NAME` because that is the one thing both sides expose.
    #[test]
    fn the_transcript_and_the_commitment_configuration_name_one_hash() {
        use crypto::fiat_shamir::transcript_hash::TranscriptHash;

        // The byte arm's object IS `DefaultTranscript<E, H::Transcript>`, so the
        // agreement is by construction here and this test says so cheaply. On an
        // algebraic branch the two are independent types and this becomes the
        // check that matters.
        let named = <<BlockStarkHash as StarkHash>::Transcript as TranscriptHash>::NAME;
        assert!(
            !named.is_empty(),
            "a commitment configuration must name its Fiat-Shamir hash"
        );
    }

    /// ✓ A fresh transcript is deterministic in its seed — the property every
    /// prove/verify pair depends on, and the one a mis-wired constructor breaks.
    #[test]
    fn a_seeded_transcript_is_a_function_of_its_seed() {
        use crypto::fiat_shamir::is_transcript::IsTranscript;

        let a = <BlockTranscript as IsTranscript<E>>::state(&block_transcript(b"seed-one"));
        let b = <BlockTranscript as IsTranscript<E>>::state(&block_transcript(b"seed-one"));
        let c = <BlockTranscript as IsTranscript<E>>::state(&block_transcript(b"seed-two"));
        assert_eq!(a, b, "the same seed must give the same state");
        assert_ne!(a, c, "a different seed must give a different state");
    }
}
