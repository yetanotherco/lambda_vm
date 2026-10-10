//! D-HASH stage 0: one WHIR chain verifier's EMITTED census under RPX (today)
//! and under the Poseidon1 width-16 geometry (rate-12 leaves and transcript,
//! 4-ary walks and cap roots; [`LfmBuilder::with_p1w16_census`]).
//!
//! The census geometry's program is not executable (each width-16 hash is a
//! stand-in row), so only its ROW counts are read. Cells: every chip at
//! `main + 3·aux` (the W12 chip's `621 = 612 + 3·3` convention), the HASH chip
//! at its hasher's width: RPX 325 (its census), the width-16 chip measured by
//! `p1w16_chip`: 348 compact, 812 rule.
//!
//! cargo test --release -p lambda-vm-prover --lib lfm::p1w16_census_tests -- --nocapture

use multilinear::whir_chain::GrindBits;

use super::airs::{LfmChipCells, lfm_chip_census_with_hasher};
use super::builder::LfmBuilder;
use super::edsl::WrapHash;
use super::hash::HasherKind;
use super::p1w16_chip::Layout;
use super::whir_chain::ChainShape;
use super::whir_chain_tests::{chain_program, chain_program_from};

fn shape(n: usize) -> ChainShape {
    let mut config = crate::multilinear_prove::chain_config(&[(1, n.min(27))]);
    config.log_blowup = 2;
    config.num_queries = 112;
    config.grind = GrindBits {
        folding: 0,
        ood: 0,
        query: 20,
    };
    ChainShape::new(&config, n)
}

fn rows(census: &[LfmChipCells], name: &str) -> u64 {
    census
        .iter()
        .filter(|c| c.name == name)
        .map(|c| c.real_rows)
        .sum()
}

/// Real-row cells of every chip but HASH, at `main + 3·aux`.
fn non_hash_cells(census: &[LfmChipCells]) -> u64 {
    census
        .iter()
        .filter(|c| c.name != "LFM_HASH" && c.name != "LFM_RANGE")
        .map(|c| c.real_rows * (c.main_cols + 3 * c.aux_cols) as u64)
        .sum()
}

const CHIPS: [&str; 8] = [
    "LFM_CONST",
    "LFM_BALU",
    "LFM_XALU",
    "LFM_SELECT",
    "LFM_BITDEC",
    "LFM_HASH",
    "LFM_LANES",
    "LFM_HINT",
];

/// The default emission at n = 27, rate 1/4, Q 112, as this branch's base
/// (37d819add) emits it. D-RATE's census of the WHIR pipeline's chain
/// (600baed87, `drate_census.out`) agrees on every chip but HINT (36,905
/// there): that base's `whir_chain` differs. That the census plumbing leaves
/// the default emission alone is carried by `whir_chain_tests`' closed-form
/// and execution tests, which run the same emitter.
#[test]
fn the_default_chain_census_is_unchanged() {
    let census = lfm_chip_census_with_hasher(&chain_program(&shape(27)), HasherKind::Rpx);
    let got: Vec<u64> = CHIPS.iter().map(|c| rows(&census, c)).collect();
    assert_eq!(got, vec![66, 23184, 52076, 37968, 791, 19367, 37659, 36891]);
}

/// The census geometry moves HASH, SELECT and HINT only, and by the closed
/// forms of a 4-ary walk and rate-12 blocks.
#[test]
fn p1w16_chain_census() {
    let rpx_width = {
        let c = lfm_chip_census_with_hasher(&chain_program(&shape(22)), HasherKind::Rpx);
        let h = c
            .iter()
            .find(|c| c.name == "LFM_HASH")
            .expect("a HASH chip");
        h.main_cols + 3 * h.aux_cols
    };
    assert_eq!(rpx_width, 325, "RPX's census width");
    println!("\n== D-HASH stage 0: emitted chain census, rate 1/4, Q 112, grind 0/0/20 ==");
    for n in 22..=27usize {
        let s = shape(n);
        let base = lfm_chip_census_with_hasher(&chain_program(&s), HasherKind::Rpx);
        let p1 = lfm_chip_census_with_hasher(
            &chain_program_from(
                &s,
                LfmBuilder::new()
                    .with_wrap_hash(WrapHash::legacy())
                    .with_p1w16_census(),
            ),
            HasherKind::Rpx,
        );
        for (label, c) in [("rpx  ", &base), ("p1w16", &p1)] {
            let line: Vec<String> = CHIPS
                .iter()
                .map(|n| format!("{}={}", &n[4..], rows(c, n)))
                .collect();
            println!("CENSUS n={n} {label} {}", line.join(" "));
        }
        for chip in [
            "LFM_CONST",
            "LFM_BALU",
            "LFM_XALU",
            "LFM_BITDEC",
            "LFM_LANES",
        ] {
            assert_eq!(rows(&base, chip), rows(&p1, chip), "n={n}: {chip} moved");
        }
        let (h0, h1) = (rows(&base, "LFM_HASH"), rows(&p1, "LFM_HASH"));
        let other0 = non_hash_cells(&base);
        let other1 = non_hash_cells(&p1);
        let rpx_cells = other0 + h0 * rpx_width as u64;
        let compact = other1 + h1 * Layout::Compact.cells_per_permutation() as u64;
        let rule = other1 + h1 * Layout::Rule.cells_per_permutation() as u64;
        println!(
            "CELLS n={n} hash {h0}->{h1} ({:.3}) select {}->{} hint {}->{} | non-hash {other0}->{other1} | \
             chain cells rpx {rpx_cells} p1w16-compact {compact} ({:.3}) p1w16-rule {rule} ({:.3})",
            h1 as f64 / h0 as f64,
            rows(&base, "LFM_SELECT"),
            rows(&p1, "LFM_SELECT"),
            rows(&base, "LFM_HINT"),
            rows(&p1, "LFM_HINT"),
            compact as f64 / rpx_cells as f64,
            rule as f64 / rpx_cells as f64,
        );
    }
}
