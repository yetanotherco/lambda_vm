//! Prints static `(bitwise, keccak_rc)` preprocessed-table commitments for
//! a fixed set of `blowup_factor` values. The output is pasted into the
//! `static_commitment` match bodies in
//! `prover/src/tables/{bitwise,keccak_rc}.rs`. The
//! `static_commitments_tests` test suite pins the values so any drift in
//! the AIR or FFT pipeline is caught at test time.
//!
//! Run with:
//!     cargo run --bin compute_static_commitments --release
//!
//! ⚠️  Do not run this just to silence a failing drift test — see the
//! "Regenerating" section on `static_commitment` in `bitwise.rs` and
//! `keccak_rc.rs` for when it's actually appropriate to bless new bytes.

use lambda_vm_prover::tables::{STATIC_BLOWUP_FACTORS, bitwise, keccak_rc};
use stark::config::Commitment;
use stark::proof::options::GoldilocksCubicProofOptions;

fn format_commitment(commitment: &Commitment) -> String {
    let mut out = String::from("[\n");
    for chunk in commitment.chunks(8) {
        out.push_str("            ");
        for (i, byte) in chunk.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&format!("0x{byte:02x},"));
        }
        out.push('\n');
    }
    out.push_str("        ]");
    out
}

fn main() {
    println!(
        "// Paste these match arms into the `static_commitment` match body\n\
         // in `prover/src/tables/{{bitwise,keccak_rc}}.rs`.\n"
    );

    for &blowup in STATIC_BLOWUP_FACTORS {
        let options = match GoldilocksCubicProofOptions::with_blowup(blowup) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("skipping blowup={blowup}: {e}");
                continue;
            }
        };

        let bitwise = bitwise::compute_preprocessed_commitment(&options);
        let keccak_rc = keccak_rc::compute_preprocessed_commitment(&options);

        println!(
            "// blowup_factor = {blowup}\n\
             // ---- bitwise:\n        \
             {blowup} => Some({bitwise_fmt}),\n\
             // ---- keccak_rc:\n        \
             {blowup} => Some({keccak_fmt}),\n",
            bitwise_fmt = format_commitment(&bitwise),
            keccak_fmt = format_commitment(&keccak_rc),
        );
    }
}
