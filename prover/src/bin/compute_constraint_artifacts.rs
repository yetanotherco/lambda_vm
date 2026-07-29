//! Serializes every production table's constraint artifact to disk, so the
//! constraints exist as DATA rather than only as compiled code.
//!
//! Run with:
//!     cargo run --bin compute_constraint_artifacts --release -- <out_dir>
//!
//! Writes `<out_dir>/<TABLE>.bin` per table plus a `MANIFEST.txt` recording each
//! artifact's size, and prints the size table (the recursion machine's
//! program-length budget).
//!
//! Capture is a build-time operation: it hash-conses the whole constraint body,
//! which is exactly what a guest must not do. That is the point of writing the
//! result down — see `stark::constraint_ir::artifact`.
//!
//! The artifacts are NOT proof-options dependent (pinned by
//! `artifacts_are_invariant_across_proof_options`), so one file per table
//! covers every blowup factor.
//!
//! ⚠️  These bytes are not an oracle. Nothing about a serialized artifact proves
//! it matches the compiled folder — only
//! `prover/src/tests/constraint_artifact_tests.rs` does, by evaluating the
//! deserialized artifact against the folders on random frames. Regenerating
//! these files does not bless a constraint change.

use std::path::PathBuf;

use lambda_vm_prover::test_utils::production_airs;
use stark::constraint_ir::ConstraintArtifact;
use stark::proof::options::GoldilocksCubicProofOptions;

fn main() {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "constraint_artifacts".to_string())
        .into();

    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid");

    std::fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", out_dir.display()));

    let mut manifest = String::from(
        "# Per-table constraint artifacts. Sizes in bytes.\n\
         # table  constraints  nodes  base_consts  ext_consts  bytes\n",
    );
    let mut total_bytes = 0usize;
    let mut total_nodes = 0usize;

    println!(
        "{:<12} {:>7} {:>9} {:>7} {:>7} {:>10}",
        "table", "constr", "nodes", "bconst", "econst", "bytes"
    );

    for (label, air) in production_airs(&options) {
        let artifact = ConstraintArtifact::capture(&*air);
        artifact
            .validate_against(&*air)
            .unwrap_or_else(|e| panic!("[{label}] artifact rejected against its own AIR: {e}"));
        let bytes = artifact
            .to_bytes()
            .unwrap_or_else(|e| panic!("[{label}] serialize failed: {e}"));

        let path = out_dir.join(format!("{label}.bin"));
        std::fs::write(&path, &bytes)
            .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));

        println!(
            "{:<12} {:>7} {:>9} {:>7} {:>7} {:>10}",
            label,
            artifact.roots.len(),
            artifact.nodes.len(),
            artifact.base_consts.len(),
            artifact.ext_consts.len(),
            bytes.len()
        );
        manifest.push_str(&format!(
            "{label} {} {} {} {} {}\n",
            artifact.roots.len(),
            artifact.nodes.len(),
            artifact.base_consts.len(),
            artifact.ext_consts.len(),
            bytes.len()
        ));
        total_bytes += bytes.len();
        total_nodes += artifact.nodes.len();
    }

    manifest.push_str(&format!("TOTAL - {total_nodes} - - {total_bytes}\n"));
    let manifest_path = out_dir.join("MANIFEST.txt");
    std::fs::write(&manifest_path, &manifest)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", manifest_path.display()));

    println!(
        "\ntotal: {total_nodes} nodes, {total_bytes} bytes ({:.1} KiB)\nwritten to {}",
        total_bytes as f64 / 1024.0,
        out_dir.display()
    );
}
