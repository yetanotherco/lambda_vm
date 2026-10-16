//! Prints static `(bitwise, keccak_rc, zero_page)` preprocessed-table commitments
//! for a fixed set of `blowup_factor` values. The output is pasted into the
//! `static_commitment` match bodies in `prover/src/tables/{bitwise,keccak_rc}.rs`
//! and the `static_zero_page_commitment` match body in `prover/src/tables/page.rs`.
//! The `static_commitments_tests` test suite pins the values so any drift in
//! the AIR or FFT pipeline is caught at test time.
//!
//! Run with:
//!     cargo run --bin compute_static_commitments --release
//!
//! ⚠️  Do not run this just to silence a failing drift test — see the
//! "Regenerating" section on `static_commitment` in `bitwise.rs` /
//! `keccak_rc.rs` and `static_zero_page_commitment` in `page.rs` for when
//! it's actually appropriate to bless new bytes.

use lambda_vm_prover::tables::{STATIC_BLOWUP_FACTORS, bitwise, keccak_rc, page};
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
        "// Paste these match arms into the `static_commitment` match bodies\n\
         // in `prover/src/tables/{{bitwise,keccak_rc}}.rs` and the\n\
         // `static_zero_page_commitment` match body in `prover/src/tables/page.rs`.\n"
    );

    let zero_page_config = page::PageConfig::zero_init(0);

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
        let zero_page = page::compute_precomputed_commitment(&zero_page_config, &options);
        let private_page = page::compute_offset_only_commitment(&options);

        println!(
            "// blowup_factor = {blowup}\n\
             // ---- bitwise:\n        \
             {blowup} => Some({bitwise_fmt}),\n\
             // ---- keccak_rc:\n        \
             {blowup} => Some({keccak_fmt}),\n\
             // ---- zero_page:\n        \
             {blowup} => Some({zero_page_fmt}),\n\
             // ---- private_page (OFFSET only):\n        \
             {blowup} => Some({private_page_fmt}),\n",
            bitwise_fmt = format_commitment(&bitwise),
            keccak_fmt = format_commitment(&keccak_rc),
            zero_page_fmt = format_commitment(&zero_page),
            private_page_fmt = format_commitment(&private_page),
        );
    }
}
