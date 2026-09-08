//! Throwaway probe (untracked): do the upstream EEST Amsterdam benchmark
//! fixtures execute on the guest this branch pins?
//!
//! Inputs are `statelessInputBytes` lifted verbatim out of
//! `tests-zkevm-benchmark@v0.8.2` blockchain tests, so nothing is rebuilt host
//! side. Run with:
//!   cargo test --manifest-path tooling/ethrex-tests/Cargo.toml --release \
//!     --test eest_amsterdam_probe -- --nocapture

use ethrex_guest_program::crypto::NativeCrypto;
use ethrex_guest_program::l1::run_stateless_guest;
use std::sync::Arc;

const BIN_DIR: &str = "../../eest_amsterdam_bins";

#[test]
fn eest_amsterdam_fixtures_validate_natively() {
    // The .bin files are downloaded/generated artifacts, not committed: a checkout
    // without them skips rather than fails, so this probe can live in the tree.
    if !std::path::Path::new(BIN_DIR).is_dir() {
        println!("{BIN_DIR} absent — skipping");
        return;
    }
    let mut entries: Vec<_> = std::fs::read_dir(BIN_DIR)
        .expect("bin dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "no .bin found in {BIN_DIR}");

    for path in entries {
        let inputs = std::fs::read(&path).expect("read fixture");
        let schema_id = u16::from_be_bytes([inputs[0], inputs[1]]);
        let started = std::time::Instant::now();
        let output = run_stateless_guest(&inputs, Arc::new(NativeCrypto));
        let elapsed = started.elapsed();
        let name = path.file_name().unwrap().to_string_lossy();
        println!(
            "{:<60} in={:>9} B schema={:#06x} out={} B ok={} chain_id={} native={:?}",
            &name[..name.len().min(60)],
            inputs.len(),
            schema_id,
            output.len(),
            output.get(32).copied().unwrap_or(255),
            u64::from_le_bytes(output[33..41].try_into().unwrap()),
            elapsed,
        );
        assert_eq!(output.len(), 43, "{name}: unexpected output length");
        assert_eq!(output[32], 1, "{name}: native stateless validation failed");
    }
}

/// What the retired real-block fixture does on this branch's guest: the rkyv
/// archive's first two bytes are not the Amsterdam schema id, so the input is
/// rejected and the guest commits an all-zero result. It does NOT error out —
/// which is exactly why a stale fixture would silently benchmark nothing.
///
/// Reads a copy kept outside `executor/tests`, since that directory now holds the
/// regenerated (valid) fixture for the same block. Skips when the copy is absent.
#[test]
fn legacy_rkyv_real_block_is_rejected_silently() {
    let path = format!("{BIN_DIR}/legacy_rkyv_real_block.bin");
    if !std::path::Path::new(&path).is_file() {
        println!("{path} absent — skipping");
        return;
    }
    let inputs = std::fs::read(&path).expect("legacy fixture");
    let output = run_stateless_guest(&inputs, Arc::new(NativeCrypto));
    println!(
        "legacy fixture: {} B, first two bytes {:#04x}{:02x}, output {} B, ok={}, all_zero={}",
        inputs.len(),
        inputs[0],
        inputs[1],
        output.len(),
        output[32],
        output.iter().all(|b| *b == 0),
    );
    assert_eq!(output.len(), 43);
    assert_eq!(output[32], 0, "legacy input must not validate");
}
