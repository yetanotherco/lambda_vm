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
/// Measured, not guessed: this guest yields ONE epoch at `log2` 6, 8 and 10, and
/// two at 4 — so it runs somewhere between 17 and 64 cycles and only a 16-cycle
/// epoch splits it. A single-epoch fixture would defeat the point, since the
/// whole target is a CONTINUATION.
///
/// Blob sizes for the record: 310,212 bytes at one epoch, 587,188 at two.
pub const FIXTURE_EPOCH_LOG2: u32 = 4;

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
pub fn load_or_generate(cache: &Path) -> Vec<u8> {
    if let Ok(bytes) = std::fs::read(cache) {
        return bytes;
    }
    let (blob, _) = generate();
    if let Some(dir) = cache.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(cache, &blob);
    blob
}

/// Checks the blob carries the recursion input's magic prefix — i.e. that it is
/// the guest's wire format and not some other encoding.
pub fn has_recursion_prefix(blob: &[u8]) -> bool {
    blob.len() > crate::RECURSION_INPUT_PREFIX_LEN
        && blob.starts_with(&crate::RECURSION_INPUT_MAGIC)
}
