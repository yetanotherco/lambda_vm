//! Prints the static preprocessed-table commitments — FOUR families: `bitwise`,
//! `keccak_rc`, and `page`'s zero-page and private-page (OFFSET-only) constants
//! — for a fixed set of `blowup_factor` values. The output is pasted into the
//! `static_commitment` match bodies in `prover/src/tables/{bitwise,keccak_rc}.rs`
//! and the `static_zero_page_commitment` / `static_private_page_commitment`
//! match bodies in `prover/src/tables/page.rs`.
//! The `static_commitments_tests` test suite pins the values so any drift in
//! the AIR or FFT pipeline is caught at test time.
//!
//! Run with:
//!     cargo run --bin compute_static_commitments --release
//!
//! `--layout row` prints the ONE-ROW (S2) twins instead — the same columns
//! committed with one LDE row per leaf — for `STATIC_BLOWUP_FACTORS_ONE_ROW`;
//! they are pasted into the `*_one_row` match bodies next to each constant
//! and pinned by the one-row drift tests. `--layout pair` (the default) is
//! the output above, unchanged.
//!
//! ⚠ On a hash-pin change run this FIRST and paste before `compute_lfm_registry`:
//! the registry embeds these constants (slots 13 and 14 of every entry, and
//! `program_id` folds them), so a registry generated before the paste carries
//! the outgoing hash's statics and the drift gate catches it.
//!
//! ⚠️  Do not run this just to silence a failing drift test — see the
//! "Regenerating" section on `static_commitment` in `bitwise.rs` /
//! `keccak_rc.rs` and the two `page.rs` constants for when it's actually
//! appropriate to bless new bytes. A hash-pin change is one such time, and it
//! regenerates all four families together (`prover/src/hash_pin.rs`).

use lambda_vm_prover::tables::{
    STATIC_BLOWUP_FACTORS, STATIC_BLOWUP_FACTORS_ONE_ROW, bitwise, keccak_rc, page,
};
use stark::config::Commitment;
use stark::leaf_layout::LeafLayout;
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

/// `--layout pair|row` (default `pair`). Anything else aborts: a typo must
/// not print the other layout's constants under this one's name.
fn layout_arg() -> LeafLayout {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => LeafLayout::RowPair,
        [flag, value] if flag == "--layout" => match value.as_str() {
            "pair" => LeafLayout::RowPair,
            "row" => LeafLayout::Row,
            other => panic!("--layout must be `pair` or `row`, got `{other}`"),
        },
        other => panic!("usage: compute_static_commitments [--layout pair|row], got {other:?}"),
    }
}

fn main() {
    let layout = layout_arg();
    let blowups = match layout {
        LeafLayout::RowPair => STATIC_BLOWUP_FACTORS,
        LeafLayout::Row => STATIC_BLOWUP_FACTORS_ONE_ROW,
    };
    println!("// leaf layout: {layout:?}");
    // The one-row twins go into the `*_one_row` functions beside each constant.
    let suffix = if layout.is_one_row() { "_one_row" } else { "" };
    println!(
        "// Paste these match arms into the `static_commitment{suffix}` match bodies\n\
         // in `prover/src/tables/{{bitwise,keccak_rc}}.rs` and the\n\
         // `static_zero_page_commitment{suffix}` / `static_private_page_commitment{suffix}`\n\
         // match bodies in `prover/src/tables/page.rs`.\n"
    );

    let zero_page_config = page::PageConfig::zero_init(0);

    for &blowup in blowups {
        let options = match GoldilocksCubicProofOptions::with_blowup(blowup) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("skipping blowup={blowup}: {e}");
                continue;
            }
        };

        let bitwise = bitwise::compute_preprocessed_commitment_with(&options, layout);
        let keccak_rc = keccak_rc::compute_preprocessed_commitment_with(&options, layout);
        let zero_page =
            page::compute_precomputed_commitment_with(&zero_page_config, &options, layout);
        let private_page = page::compute_offset_only_commitment_with(&options, layout);

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
