//! The PROTO lane's P1 on the STARK pipeline (`LAMBDA_VM_GAP_P1_STARK`, base
//! epochs at blowup 2) against the tables whose preprocessed roots are STATIC:
//! under `one_row = auto` each one's layout is resolved from its widths, and
//! a resolved layout with no shipped root is a proving error — so the
//! knob-on arm can reach a block run only if every one of them has a root.

use std::sync::Arc;

use stark::leaf_layout::{LeafLayout, table_leaf_layout};
use stark::proof::options::{OneRowMode, ProofOptions};
use stark::traits::AIR;

use crate::recursion::Preset;
use crate::tables::page::PageConfig;
use crate::tables::{bitwise, keccak_rc};
use crate::test_utils::{E, F, create_bitwise_air, create_keccak_rc_air};
use crate::zf_format::ZfFormat;

/// The STARK pipeline's base options at `preset`: the production format with
/// `one_row = auto`, which #985 compiles in and every STARK arm sets.
fn stark_pipeline(preset: Preset) -> ProofOptions {
    ZfFormat {
        one_row: OneRowMode::Auto,
        ..ZfFormat::DEFAULT
    }
    .options(preset.options())
}

type Air = Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>>;

/// Every table whose preprocessed root is a shipped constant, built as the
/// production AIR sets build it, at the height it has in the block: BITWISE's
/// full table, KECCAK_RC's 24 rounds padded to 32, and a `DEFAULT_PAGE_SIZE`
/// page of the global proof (zero-init and private-input).
fn static_root_tables(opts: &ProofOptions) -> Vec<(&'static str, usize, Air)> {
    let bitwise: Air = Box::new(create_bitwise_air(opts).with_lazy_preprocessed_columns(
        bitwise::lazy_commitment(opts),
        bitwise::NUM_PRECOMPUTED_COLS,
        Arc::new(bitwise::preprocessed_columns),
    ));
    let keccak_rc: Air = Box::new(create_keccak_rc_air(opts).with_lazy_preprocessed_columns(
        keccak_rc::lazy_commitment(opts),
        keccak_rc::NUM_PRECOMPUTED_COLS,
        Arc::new(keccak_rc::preprocessed_columns),
    ));
    let zero_page: Air = Box::new(crate::continuation::global_memory_air(
        opts,
        &PageConfig::zero_init(0x10_0000),
        None,
    ));
    let private_page: Air = Box::new(crate::continuation::global_memory_air(
        opts,
        &PageConfig {
            page_base: 0x20_0000,
            init_values: None,
            is_private_input: true,
        },
        None,
    ));
    let page_rows = crate::tables::page::DEFAULT_PAGE_SIZE;
    vec![
        ("BITWISE", 1 << 20, bitwise),
        ("KECCAK_RC", 32, keccak_rc),
        ("zero-init page", page_rows, zero_page),
        ("private-input page", page_rows, private_page),
    ]
}

/// The layouts `auto` resolves the static-root tables to, and whether each
/// has a root there.
fn resolved(preset: Preset) -> Vec<(&'static str, LeafLayout, bool)> {
    let opts = stark_pipeline(preset);
    static_root_tables(&opts)
        .into_iter()
        .map(|(name, rows, air)| {
            let layout = table_leaf_layout(air.as_ref(), rows);
            (
                name,
                layout,
                air.precomputed_commitment_for(layout).is_some(),
            )
        })
        .collect()
}

/// ★ The control: at blowup 4 the rule resolves what the record's census
/// reads for these tables (gap/census-stark, `GAPB STARK_FRI … one_row=`):
/// BITWISE row pairs, KECCAK_RC one row, every global-proof page row pairs —
/// each with a shipped root.
#[test]
fn at_blowup_4_auto_resolves_the_records_layouts() {
    use LeafLayout::{Row, RowPair};
    assert_eq!(
        resolved(Preset::Blowup4),
        vec![
            ("BITWISE", RowPair, true),
            ("KECCAK_RC", Row, true),
            ("zero-init page", RowPair, true),
            ("private-input page", RowPair, true),
        ]
    );
}

/// ★ P1: at blowup 2 (219 queries) `auto` still puts KECCAK_RC on one row and
/// the rest on row pairs, and every one of them has a root — KECCAK_RC's
/// through the blowup-2 twin, the only static root P1 needs.
#[test]
fn at_blowup_2_every_static_root_table_has_a_root_for_its_layout() {
    use LeafLayout::{Row, RowPair};
    assert_eq!(
        resolved(Preset::Blowup2),
        vec![
            ("BITWISE", RowPair, true),
            ("KECCAK_RC", Row, true),
            ("zero-init page", RowPair, true),
            ("private-input page", RowPair, true),
        ]
    );
}

/// The blowup-2 twin is the recompute of its columns at that layout, and is
/// not the row-pair root.
#[test]
fn the_keccak_rc_blowup_2_one_row_twin_is_its_recompute() {
    let opts = stark_pipeline(Preset::Blowup2);
    let recomputed = keccak_rc::compute_preprocessed_commitment_with(&opts, LeafLayout::Row);
    assert_eq!(keccak_rc::static_commitment_one_row(2), Some(recomputed));
    assert_eq!(
        keccak_rc::preprocessed_commitment_for(&opts, LeafLayout::Row),
        Some(recomputed)
    );
    assert_ne!(recomputed, keccak_rc::preprocessed_commitment(&opts));
    assert_ne!(Some(recomputed), keccak_rc::static_commitment_one_row(4));
}

/// ★ P1 end to end on the production paths (a box test: it proves a full VM
/// trace with the 2^20-row BITWISE table): a VM proof at blowup 2 under the
/// STARK pipeline's format — `cap = auto`, `fri = dp`, `one_row = auto`, so
/// KECCAK_RC on one row through its blowup-2 twin — round-trips, and the
/// blowup-4 verifier rejects it.
#[test]
fn a_vm_proof_round_trips_at_blowup_2_under_the_stark_pipeline_format() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_mul_8");
    let p1 = stark_pipeline(Preset::Blowup2);
    let vm_proof = crate::prove_with_options(&elf_bytes, &p1, &Default::default())
        .expect("the fixture must prove at blowup 2 under the STARK pipeline's format");
    assert!(
        crate::verify_with_options(&vm_proof, &elf_bytes, &p1, None, None)
            .expect("honest verify must not error"),
        "an honest blowup-2 VM proof must verify"
    );
    let one_row_tables = vm_proof
        .proof
        .proofs
        .iter()
        .filter(|p| {
            p.deep_poly_openings[0]
                .composition_poly
                .evaluations_sym
                .is_empty()
        })
        .count();
    println!(
        "GAP PROTO P1 VM blowup 2 / {} q: {one_row_tables} of {} tables one-row",
        p1.fri_number_of_queries,
        vm_proof.proof.proofs.len()
    );
    assert!(one_row_tables > 0, "auto must put some table on one row");
    assert!(
        !crate::verify_with_options(
            &vm_proof,
            &elf_bytes,
            &stark_pipeline(Preset::Blowup4),
            None,
            None
        )
        .unwrap_or(false),
        "the blowup is a verifier constant: the blowup-4 verifier must reject"
    );
}
