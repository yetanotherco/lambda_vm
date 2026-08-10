//! Cross-check every production table's declared OOD transition window against
//! the next-row read set derived from its captured constraint IR.
//!
//! [`stark::traits::AIR::trace_ood_next_row_columns`] declares which full-width
//! `[main | aux]` columns a transition constraint reads at the *next* row. The
//! verifier opens every trace column at `z` but prunes the `g·z` (next-row)
//! opening down to exactly that declared set, reconstructing ZERO for every
//! other column at the next row (see `stark::ood`). A constraint that reads a
//! next-row column the declaration omits is therefore fed zero there — a silent
//! soundness/completeness bug.
//!
//! For every VM table the window is the hand-synced `AirWithBuses` override
//! (empty, or exactly the LogUp accumulator column); it deliberately ignores the
//! wrapped constraint set, which could legally read the next row. The only guard
//! against that declaration drifting from the constraints is a test — the
//! `debug_assert`s that would otherwise catch it are compiled out under the
//! `--release` test profile this repo uses. This is that test: it derives the
//! true read set from the captured [`stark::constraint_ir::ConstraintProgram`]
//! (which runs the wrapped constraint set AND the LogUp emission through one
//! CaptureBuilder) and validates the declaration against it, so the check tracks
//! the real constraints rather than a copy of the declaration.
//!
//! It only CONSTRUCTS AIRs (no program execution, no ELF), so it runs anywhere.
//! The table list mirrors the enumeration in `constraint_program_tests.rs` — the
//! canonical per-table `create_*_air` constructors from `test_utils`; there is no
//! ELF-free registry to iterate (`VmAirs::air_refs` needs a real ELF plus
//! preprocessed-commitment builds), so a new table must be added here.

use stark::proof::options::GoldilocksCubicProofOptions;
use stark::traits::AIR;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::*;

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

/// Assert an AIR's declared next-row window equals / covers the IR-derived read
/// set.
///
/// * `derived ⊆ declared` for every AIR — the soundness direction: a derived
///   column missing from the declaration is pruned to zero at the next row.
/// * exact equality when `exact` — every `AirWithBuses` should declare
///   *precisely* the accumulator column (or nothing); over-declaration only
///   bloats the `g·z` opening, but for these AIRs the window is exactly known.
fn assert_ood_window_matches_ir(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    exact: bool,
    label: &str,
) {
    let (main, aux) = air.trace_layout();

    let mut declared = air.trace_ood_next_row_columns();
    declared.sort_unstable();
    declared.dedup();

    // The production capture (lazy OnceLock behind the AIR): the wrapped
    // constraint set spliced ahead of the LogUp suffix, so a next-row read by
    // ANY constraint — base or LogUp — is in the derived set.
    let derived = air.constraint_program().next_row_trace_reads(main);

    for &c in &derived {
        assert!(
            c < main + aux,
            "[{label}] derived next-row column {c} out of concatenated width {main}+{aux}"
        );
        assert!(
            declared.contains(&c),
            "[{label}] a transition constraint reads full-width column {c} at the next row, but \
             it is absent from trace_ood_next_row_columns() = {declared:?}; the verifier prunes \
             that g·z opening to ZERO — soundness bug"
        );
    }

    if exact {
        assert_eq!(
            derived, declared,
            "[{label}] declared next-row window {declared:?} is not exactly the IR-derived read \
             set {derived:?}: over-declaration bloats every g·z opening"
        );
    }
}

/// Every production table AIR declares an OOD transition window equal to the
/// next-row read set derived from its captured constraint IR. All VM tables are
/// `AirWithBuses`, whose window is exactly the accumulator column (or empty), so
/// equality is asserted for each.
#[test]
fn all_table_windows_match_captured_ir() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    assert_ood_window_matches_ir(&create_cpu_air(&opts), true, "CPU");
    assert_ood_window_matches_ir(&create_dma_air(&opts), true, "DMA");
    assert_ood_window_matches_ir(&create_bitwise_air(&opts), true, "BITWISE");
    assert_ood_window_matches_ir(&create_lt_air(&opts), true, "LT");
    assert_ood_window_matches_ir(&create_shift_air(&opts), true, "SHIFT");
    assert_ood_window_matches_ir(&create_eq_air(&opts), true, "EQ");
    assert_ood_window_matches_ir(&create_bytewise_air(&opts), true, "BYTEWISE");
    assert_ood_window_matches_ir(&create_store_air(&opts), true, "STORE");
    assert_ood_window_matches_ir(&create_cpu32_air(&opts), true, "CPU32");
    assert_ood_window_matches_ir(&create_memw_air(&opts), true, "MEMW");
    assert_ood_window_matches_ir(&create_memw_aligned_air(&opts), true, "MEMW_A");
    assert_ood_window_matches_ir(&create_memw_register_air(&opts), true, "MEMW_R");
    assert_ood_window_matches_ir(&create_load_air(&opts), true, "LOAD");
    assert_ood_window_matches_ir(&create_decode_air(&opts), true, "DECODE");
    assert_ood_window_matches_ir(&create_mul_air(&opts), true, "MUL");
    assert_ood_window_matches_ir(&create_dvrm_air(&opts), true, "DVRM");
    assert_ood_window_matches_ir(&create_branch_air(&opts), true, "BRANCH");
    assert_ood_window_matches_ir(&create_halt_air(&opts), true, "HALT");
    assert_ood_window_matches_ir(&create_commit_air(&opts), true, "COMMIT");
    assert_ood_window_matches_ir(&create_page_air(&opts, 0x1000), true, "PAGE");
    assert_ood_window_matches_ir(&create_register_air(&opts), true, "REGISTER");
    assert_ood_window_matches_ir(&create_keccak_air(&opts), true, "KECCAK");
    assert_ood_window_matches_ir(&create_keccak_rnd_air(&opts), true, "KECCAK_RND");
    assert_ood_window_matches_ir(&create_keccak_rc_air(&opts), true, "KECCAK_RC");
    assert_ood_window_matches_ir(&create_ecsm_air(&opts), true, "ECSM");
    assert_ood_window_matches_ir(&create_ecdas_air(&opts), true, "ECDAS");
    assert_ood_window_matches_ir(&create_hint_air(&opts), true, "HINT");
}
