//! Prints static `(bitwise, keccak_rc, zero_page)` preprocessed-table commitments
//! for a fixed set of `blowup_factor` values. The output is pasted into the
//! `static_commitment` match bodies in `prover/src/tables/{bitwise,keccak_rc}.rs`
//! and the `static_zero_page_commitment` match body in `prover/src/tables/page.rs`.
//!
//! Run with:
//!     cargo run --bin compute_static_commitments --release

use lambda_vm_prover::tables::{bitwise, keccak_rc, page};
use stark::config::Commitment;
use stark::proof::options::GoldilocksCubicProofOptions;

const STATIC_BLOWUP_FACTORS: &[u8] = &[2, 4, 8, 16, 32];

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

    let zero_page_config = page::PageConfig::zero_init(0, page::DEFAULT_PAGE_SIZE);

    for &blowup in STATIC_BLOWUP_FACTORS {
        let options = match GoldilocksCubicProofOptions::with_blowup(blowup) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("skipping blowup={blowup}: {e}");
                continue;
            }
        };

        let bitwise_c = bitwise::preprocessed_commitment(&options);
        let keccak_rc_c = keccak_rc::preprocessed_commitment(&options);
        let zero_page = page::compute_precomputed_commitment(&zero_page_config, &options);

        println!(
            "// blowup_factor = {blowup}\n\
             // ---- bitwise:\n        {blowup} => Some({bitwise}),\n\
             // ---- keccak_rc:\n        {blowup} => Some({keccak_rc}),\n\
             // ---- zero_page:\n        {blowup} => Some({page}),\n",
            bitwise = format_commitment(&bitwise_c),
            keccak_rc = format_commitment(&keccak_rc_c),
            page = format_commitment(&zero_page),
        );
    }
}
