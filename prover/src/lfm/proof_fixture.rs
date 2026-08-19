//! Real continuation-proof BYTES for the machine to consume.
//!
//! R1f's premise: everything before this point ran on synthetic or
//! self-generated data. This module produces an actual two-epoch continuation
//! proof in exactly the encoding the RV64 recursion guest receives.
//!
//! ## Why bytes, and why THESE bytes
//!
//! The guest never sees a `ContinuationProof`. It gets a blob in private input
//! and reads it zero-copy through rkyv. So a machine-side reader whose input is
//! a byte blob is the direct analogue of the guest's reader, and a disagreement
//! between the two is a meaningful signal; a reader that consumed an in-memory
//! `ContinuationProof` would be exercising a path production does not have.
//!
//! The encoding is therefore NOT invented here. It is
//! [`crate::recursion::encode_continuation_guest_input`] — the same encoder the
//! guest's blob comes from — so the fixture cannot drift from production without
//! the encoder itself changing.
//!
//! ## Why not the existing dump test
//!
//! `tests::recursion_smoke_test::test_dump_recursion_input` produces exactly
//! these bytes, but it is `#[ignore]`d as a diagnostic, is driven by five
//! environment variables, and writes to a fixed `/tmp` path. None of that is
//! usable from a deterministic unit test. This module calls the same two public
//! functions it calls — `prove_continuation` then
//! `encode_continuation_guest_input` — and nothing else, so the ENCODER (the
//! part that must not drift) is shared while the harness around it is not.

use std::path::{Path, PathBuf};

use stark::proof::options::ProofOptions;

use crate::recursion::MIN_PROOF_OPTIONS;

/// Inner guest whose execution the fixture proves. `fibonacci` rather than
/// `empty`: the fixture needs enough cycles to actually split into two epochs,
/// and `empty` collapses to a single (monolithic-style) one.
pub const FIXTURE_INNER_ELF: &str = "fibonacci";

/// Epoch size, as `log2(cycles)`.
///
/// Measured, not guessed: this guest runs **15 cycles** — the fixture passes no
/// private input, so `n` reads as 0 and the loop body never executes — which an
/// 8-cycle epoch splits into two and a 16-cycle one does not. A single-epoch
/// fixture would defeat the point, since the whole target is a CONTINUATION,
/// and `continuation_fixture_generates_two_epochs` is the canary for it.
///
/// ⚠ **The cycle count is a property of the compiled ELF, not of the guest
/// source.** `bench_vs/lambda/fibonacci` has no dependencies, so nothing in this
/// workspace moves it — but the pinned nightly and the sysroot do, and a
/// codegen change of two instructions is enough to cross an epoch boundary at
/// this size. If the canary reports one epoch, re-measure rather than guess:
/// run the ELF to completion under `Executor::resume_with_limit` and count the
/// logs, one per cycle, then set this to a `log2` strictly below the count.
///
/// Blob sizes for the record: 947,340 bytes at the two epochs this selects,
/// against 309,084 for the single epoch a 16-cycle one collapses to.
pub const FIXTURE_EPOCH_LOG2: u32 = 3;

/// Proof options the fixture is proved under: the `min` preset, which is the
/// cheapest to generate. It is explicitly NOT a secure parameter set — this
/// fixture exists to exercise byte layout and Merkle structure, not to stand in
/// for a production proof's security.
pub fn fixture_options() -> ProofOptions {
    MIN_PROOF_OPTIONS
}

/// Repository root, derived from this crate's manifest directory.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("prover/ has a parent")
        .to_path_buf()
}

/// Reads a recursion-suite guest ELF (built by `make compile-recursion-elfs`).
pub fn read_inner_elf() -> Vec<u8> {
    let path = workspace_root().join(format!(
        "executor/program_artifacts/recursion/{FIXTURE_INNER_ELF}.elf"
    ));
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read {} — run `make compile-recursion-elfs`: {e}",
            path.display()
        )
    })
}

/// Proves the fixture continuation and encodes the guest blob.
///
/// Returns `(blob, num_epochs)`. The epoch count is read before encoding
/// because the encoder consumes the bundle.
pub fn generate() -> (Vec<u8>, usize) {
    let elf = read_inner_elf();
    let opts = fixture_options();
    let bundle = crate::continuation::prove_continuation(&elf, &[], FIXTURE_EPOCH_LOG2, &opts)
        .expect("fixture continuation must prove");
    let num_epochs = bundle.num_epochs();
    let blob = crate::recursion::encode_continuation_guest_input(bundle, &elf, &opts)
        .expect("fixture blob must encode");
    (blob, num_epochs)
}

/// Loads the cached blob, generating and caching it when absent.
///
/// Proving is slow enough that regenerating per test is not viable, but a
/// checked-in binary is worse: it can drift from the encoder silently. So the
/// cache lives outside the repository and the GENERATION path is what tests
/// exercise on a cold cache.
///
/// ## ⚠ The blob is NOT reproducible — measured, and it constrains callers
///
/// Two `generate()` calls on identical inputs produce blobs that differ in
/// ~65k of 587k bytes, and the difference is SEMANTIC, not archive padding:
/// some sub-proofs commit to different roots, which moves the Fiat-Shamir
/// challenges, which opens different leaves. (`machine_tests::
/// fixture_generation_is_not_reproducible` is the standing evidence.)
///
/// So **nothing derived from a specific blob may be pinned as a constant** —
/// not a query index, not a leaf value, not a root. Pin SHAPE (column counts,
/// tree depths), which is stable, and recover per-blob values from the blob.
/// R1f's `R1F_SHAPE` and its recovered leaf index are built that way; a pinned
/// index would have broken on the very next cold cache.
///
/// The write is atomic (temp file then rename) because the cache is shared by
/// tests that run in parallel and one of them regenerates it: without the
/// rename a reader can observe a half-written blob, and since blobs differ,
/// "it was fine last time" proves nothing.
/// ★ The cache key for everything that changes these bytes INCOMPATIBLY.
///
/// The cache lives in the shared temp directory and is keyed on the inner ELF
/// and the epoch size — neither of which says anything about the proof format.
/// Every format-moving branch on one machine therefore wrote its blob over
/// everyone else's, and the next branch to read it got bytes its own
/// `rkyv::access` could not validate. That surfaces as
/// `"fixture blob must validate"` in whichever test read first — a phantom
/// failure with no relationship to the branch under test.
///
/// It is worse than a wasted debugging hour: the stash-and-rerun control used
/// to separate "pre-existing failure" from "my change broke it" SHARES this
/// cache, so a poisoned blob makes both arms fail and the control cannot tell
/// them apart. An exoneration taken against a stale blob proves nothing.
///
/// So the key names the two axes that actually move the bytes: the statement
/// encoding version and the commitment hash.
pub fn cache_format_key() -> String {
    let tag = std::str::from_utf8(crate::statement::DOMAIN_TAG).unwrap_or("stmt");
    format!("{tag}-{:?}", stark::config::COMMITMENT_HASH)
}

/// Whether `bytes` still look like a blob this build can read.
///
/// Only the wire prefix — the magic and version [`crate::encode_recursion_input`]
/// writes. Deliberately cheap: this is the last line of defence for a truncated
/// or foreign file, not the format check. Separating format-incompatible builds
/// is [`cache_format_key`]'s job, because two branches can share this prefix and
/// still disagree about everything after it.
fn prefix_is_readable(bytes: &[u8]) -> bool {
    bytes.len() > crate::RECURSION_INPUT_PREFIX_LEN
        && bytes[0..4] == crate::RECURSION_INPUT_MAGIC
        && u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
            == crate::RECURSION_INPUT_VERSION
}

pub fn load_or_generate(cache: &Path) -> Vec<u8> {
    if let Ok(bytes) = std::fs::read(cache) {
        // A stale or foreign blob REGENERATES rather than erroring downstream:
        // the failure it used to cause named a validation site, never the cache.
        if prefix_is_readable(&bytes) {
            return bytes;
        }
        eprintln!(
            "fixture cache at {} is not readable by this build — regenerating",
            cache.display()
        );
    }
    let (blob, _) = generate();
    write_cache(cache, &blob);
    blob
}

/// Publishes `blob` at `cache` atomically, so a concurrent reader sees either
/// the old complete blob or the new one, never a torn prefix.
pub fn write_cache(cache: &Path, blob: &[u8]) {
    if let Some(dir) = cache.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let staging = cache.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&staging, blob).is_ok() {
        let _ = std::fs::rename(&staging, cache);
    }
}

/// Checks the blob carries the recursion input's magic prefix — i.e. that it is
/// the guest's wire format and not some other encoding.
pub fn has_recursion_prefix(blob: &[u8]) -> bool {
    blob.len() > crate::RECURSION_INPUT_PREFIX_LEN
        && blob.starts_with(&crate::RECURSION_INPUT_MAGIC)
}

/// An opened fixture blob, holding the aligned bytes the archived view borrows
/// from.
///
/// Mirrors `recursion::verify_continuation_and_attest`'s decode exactly: strip
/// the magic/version prefix, re-align if the host `Vec` is not on rkyv's
/// alignment (guest slices are aligned by construction; host ones carry no such
/// guarantee), then `rkyv::access` with validation. The owning struct exists
/// because the archived view borrows from the aligned buffer.
pub struct FixtureArchive {
    aligned: rkyv::util::AlignedVec<{ crate::RECURSION_INPUT_ALIGN }>,
}

impl FixtureArchive {
    pub fn open(blob: &[u8]) -> Self {
        let archive_bytes = crate::recursion_archive_bytes(blob)
            .expect("fixture blob must carry the recursion magic and version");
        let mut aligned = rkyv::util::AlignedVec::new();
        aligned.extend_from_slice(archive_bytes);
        Self { aligned }
    }

    /// The validated archived guest input.
    pub fn guest_input(&self) -> &crate::recursion::ArchivedContinuationGuestInput {
        rkyv::access::<crate::recursion::ArchivedContinuationGuestInput, rkyv::rancor::Error>(
            &self.aligned,
        )
        .expect("fixture blob must validate")
    }
}
