//! End-to-end regression tests for unconstrained multiplicities: MUL's
//! `μ_lo`/`μ_hi` and DVRM's `μ_q`/`μ_r` must be bounded and non-negative, and
//! SHIFT's `zbs` must be a bit.
//!
//! **MUL / DVRM.** Both chips send every range check (and DVRM its MUL/LT/ZERO checks) with
//! multiplicity `μ_a + μ_b`, the row's total lookup count. Before the fix neither
//! column was constrained, so a row with `μ_a = +1, μ_b = −1` sent no checks at
//! all while still receiving one `a` lookup: its result was free. The stray `−1`
//! on the `b` tuple was cancelled by an honest copy of the row with
//! `μ_a = 0, μ_b = +1`, whose checks fire once and pass. Both forgeries below
//! verified against `main` (ffc4ac19). The fix range-checks each μ to a halfword
//! (`IS_HALF[μ]`, weighted by μ), so `μ_b = −1` has no receiver.
//!
//! **SHIFT.** The five HWSL senders fire with `1 − zbs`, and on a μ = 0 padding
//! row nothing pinned `zbs`: `zbs = 2` made that −1, so the padding row provided
//! exactly the HWSL tuples a forged real row needed (`1 << 1 = 0` verified
//! against `main`). The fix bit-constrains `zbs` (and μ).
//!
//! Each guest computes a value with one real instruction and commits it; the
//! malicious prover claims a different value and reshapes the chip's rows. It
//! also rebalances every BITWISE receiver multiplicity it can (those are free
//! main-trace columns, as in `page_offset_forgery_poc`), so the only thing left
//! to reject the proof is the new constraint itself.
//!
//! Run under production proof options.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;

use crate::statement::{StatementKind, absorb_statement};
use crate::tables::bitwise::{cols as bw_cols, row_index as bw_row_index};
use crate::tables::trace_builder::Traces;
use crate::tables::types::{BusId, FE, GoldilocksExtension as Ext, GoldilocksField as Base};
use crate::tables::{dvrm, mul};
use crate::test_utils::{E, asm_elf_bytes};
use crate::{MaxRowsConfig, VmAirs, VmProof};

use executor::elf::Elf;
use executor::vm::execution::Executor;

fn opts() -> ProofOptions {
    crate::GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// One forgery scenario: a guest whose single `lhs ∘ rhs` instruction yields
/// `honest`, and the chip rewrite that makes it yield `forged` instead.
struct Scenario {
    guest: &'static str,
    lhs: u64,
    rhs: u64,
    honest: u64,
    forged: u64,
    forge_rows: fn(&mut Traces, &Scenario),
}

const MUL: Scenario = Scenario {
    guest: "poc_mul_commit",
    lhs: 3,
    rhs: 5,
    honest: 15,
    forged: 99,
    forge_rows: forge_mul_rows,
};

const DIVU: Scenario = Scenario {
    guest: "poc_divu_commit",
    lhs: 7,
    rhs: 2,
    honest: 3,
    forged: 100,
    forge_rows: forge_dvrm_rows,
};

const SLL: Scenario = Scenario {
    guest: "poc_sll_commit",
    lhs: 1,
    rhs: 1,
    honest: 2,
    forged: 0,
    forge_rows: forge_shift_rows,
};

/// Which deviations the malicious prover applies.
#[derive(Clone, Copy)]
struct Forge {
    /// Rewrite the execution so the CPU claims `lhs ∘ rhs = forged`.
    claim: bool,
    /// Rewrite the chip's rows so they answer the forged claim.
    rows: bool,
}

fn craft_proof(s: &Scenario, forge: Forge) -> Result<VmProof, stark::prover::ProvingError> {
    let elf_bytes = asm_elf_bytes(s.guest);
    let options = opts();
    let program = Elf::load(&elf_bytes).expect("ELF load");
    let mut logs = Executor::new(&program, vec![])
        .expect("executor construction")
        .run()
        .expect("run")
        .logs;

    if forge.claim {
        let idx = logs
            .iter()
            .position(|l| l.src1_val == s.lhs && l.src2_val == s.rhs && l.dst_val == s.honest)
            .expect("arithmetic log not found");
        logs[idx].dst_val = s.forged;
        // The following `sd t1, 0(sp)` stores the result: keep it consistent.
        logs[idx + 1..]
            .iter_mut()
            .find(|l| l.src2_val == s.honest)
            .expect("SD of the result not found")
            .src2_val = s.forged;
    }

    let mut traces = Traces::from_elf_and_logs(
        &program,
        &logs,
        &MaxRowsConfig::default(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("trace build");

    if forge.rows {
        (s.forge_rows)(&mut traces, s);
    }

    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &program,
        &options,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::Monolithic,
        &elf_bytes,
        &traces.public_output_bytes,
        &table_counts,
        num_private_input_pages,
        &runtime_page_ranges,
        options.fri_final_poly_log_degree,
    );

    let proof = Prover::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut transcript,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )?;

    Ok(VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
        public_output: traces.public_output_bytes.clone(),
        num_private_input_pages,
    })
}

/// Whether `interactions` range-check column `col`, weighted by itself.
fn bounds_column(interactions: &[BusInteraction], col: usize) -> bool {
    interactions.iter().any(|i| {
        i.is_sender
            && i.bus_id == BusId::IsHalfword as u64
            && matches!(i.multiplicity, Multiplicity::Column(m) if m == col)
            && matches!(
                i.values.as_slice(),
                [BusValue::Packed { start_column, packing: Packing::Direct }] if *start_column == col
            )
    })
}

/// Add `delta` to the BITWISE receiver multiplicity `mu_col` at halfword `half`.
fn bump_bitwise(traces: &mut Traces, mu_col: usize, half: u64, delta: FE) {
    assert!(half < 1 << 16, "only halfwords have a BITWISE row");
    let bw = &mut traces.bitwise.main_table;
    let row = bw_row_index((half & 0xFF) as u8, (half >> 8) as u8, 0);
    let old = *bw.get(row, mu_col);
    bw.set(row, mu_col, old + delta);
}

/// The two-row rewrite shared by both chips, on one chip instance `t`:
///
/// * row A (the honest row answering the claim, `μ_a = 1, μ_b = 0`) gets the
///   forged result written by `write_forged` and `μ_b = −1`;
/// * row B (an all-zero padding row) becomes an honest copy with
///   `μ_a = 0, μ_b = 1`.
///
/// Returns the row-B copy's index so the caller can rebalance flag-gated sends.
/// If the chip range-checks its μ columns, also rebalances those IS_HALF sends
/// wherever a BITWISE row exists (`−1` has none — that is the fix).
fn forge_two_rows(
    traces: &mut Traces,
    table: fn(&mut Traces) -> &mut TraceTable<Base, Ext>,
    find_a: impl Fn(&TraceTable<Base, Ext>, usize) -> bool,
    (mu_a, mu_b, num_cols): (usize, usize, usize),
    bounded: bool,
    write_forged: impl Fn(&mut TraceTable<Base, Ext>, usize),
) -> usize {
    let t = table(traces);
    let rows = t.num_rows();
    let a = (0..rows)
        .find(|&r| find_a(t, r))
        .expect("honest row not found");
    assert_eq!(*t.get_main(a, mu_a), FE::one());
    assert_eq!(*t.get_main(a, mu_b), FE::zero());
    let b = (0..rows)
        .find(|&r| (0..num_cols).all(|c| *t.get_main(r, c) == FE::zero()))
        .expect("an all-zero padding row is needed for the honest copy");

    for c in 0..num_cols {
        let v = *t.get_main(a, c);
        t.set_main(b, c, v);
    }
    t.set_main(b, mu_a, FE::zero());
    t.set_main(b, mu_b, FE::one());
    write_forged(t, a);
    t.set_main(a, mu_b, FE::zero() - FE::one());

    if bounded {
        // IS_HALF[μ] | μ sends: B's μ_b 0 -> 1 adds IS_HALF[1] once. A's μ_b
        // 0 -> −1 adds IS_HALF[p−1] with weight −1, which has no BITWISE row.
        bump_bitwise(traces, bw_cols::MU_IS_HALF, 1, FE::one());
    }
    b
}

/// MUL: `lo(3 * 5)` forged to 99 (`μ_lo = 1, μ_hi = −1`).
fn forge_mul_rows(traces: &mut Traces, s: &Scenario) {
    use mul::cols;
    let bounded = bounds_column(&mul::bus_interactions(), cols::MU_HI);
    let forged = s.forged;
    let b = forge_two_rows(
        traces,
        |t| &mut t.muls[0],
        |t, r| {
            *t.get_main(r, cols::LHS_0) == FE::from(s.lhs)
                && *t.get_main(r, cols::RHS_0) == FE::from(s.rhs)
                && *t.get_main(r, cols::MU_LO) == FE::one()
        },
        (cols::MU_LO, cols::MU_HI, cols::NUM_COLUMNS),
        bounded,
        |t, a| t.set_main(a, cols::LO_0, FE::from(forged)),
    );

    // Row B's two MSB16 sign lookups are gated by the sign flags, not by μ, so
    // the copy duplicates them. Rebalance the BITWISE MSB16 receiver.
    for (half, flag) in [
        (cols::LHS_3, cols::LHS_SIGNED),
        (cols::RHS_3, cols::RHS_SIGNED),
    ] {
        let t = &traces.muls[0];
        let (half, mult) = (t.get_main(b, half).to_raw(), *t.get_main(b, flag));
        bump_bitwise(traces, bw_cols::MU_MSB16, half, mult);
    }
}

/// DVRM: `7 / 2` (unsigned) forged to 100 (`μ_q = 1, μ_r = −1`). Unsigned, so
/// the copy duplicates no SIGNED/SIGN_R/SIGN_D-gated sends.
fn forge_dvrm_rows(traces: &mut Traces, s: &Scenario) {
    use dvrm::cols;
    let bounded = bounds_column(&dvrm::bus_interactions(), cols::MU_R);
    let forged = s.forged;
    let b = forge_two_rows(
        traces,
        |t| &mut t.dvrms[0],
        |t, r| {
            *t.get_main(r, cols::N_0) == FE::from(s.lhs)
                && *t.get_main(r, cols::D_0) == FE::from(s.rhs)
                && *t.get_main(r, cols::MU_Q) == FE::one()
        },
        (cols::MU_Q, cols::MU_R, cols::NUM_COLUMNS),
        bounded,
        |t, a| t.set_main(a, cols::Q_0, FE::from(forged)),
    );
    let t = &traces.dvrms[0];
    for flag in [cols::SIGNED, cols::SIGN_R, cols::SIGN_D] {
        assert_eq!(
            *t.get_main(b, flag),
            FE::zero(),
            "scenario must stay unsigned"
        );
    }
}

/// SHIFT: `1 << 1` forged to 0. Row A (the real op) gets `X[0] = 0`, so
/// `out = 0` and its five HWSL tuples are no longer honest; padding row B gets
/// `zbs = 2, in[0] = 1, bit_shift = 1`, so its five HWSL senders fire with
/// `1 − zbs = −1` on exactly A's tuples. With μ = 0 every other B lookup
/// vanishes. The HWSL bus then needs no BITWISE entry for A at all, so the
/// prover drops A's honest HWSL demand from the BITWISE multiplicities.
fn forge_shift_rows(traces: &mut Traces, s: &Scenario) {
    use crate::tables::shift::cols;
    let t = &mut traces.shifts[0];
    let rows = t.num_rows();
    let a = (0..rows)
        .find(|&r| {
            *t.get_main(r, cols::IN_0) == FE::from(s.lhs)
                && *t.get_main(r, cols::SHIFT_AMOUNT) == FE::from(s.rhs)
                && *t.get_main(r, cols::MU) == FE::one()
        })
        .expect("honest SLL row not found");
    assert_eq!(*t.get_main(a, cols::X_0), FE::from(s.honest));
    let b = (0..rows)
        .find(|&r| *t.get_main(r, cols::MU) == FE::zero() && r != a)
        .expect("a padding row is needed");

    // A's honest HWSL demand: [in[i], bit_shift] for i in 0..4 and the extension
    // half `65535·is_negative`, each read at BITWISE row (half, bit_shift).
    let bit_shift = t.get_main(a, cols::BIT_SHIFT).to_raw();
    let ext = 65535 * t.get_main(a, cols::IS_NEGATIVE).to_raw();
    let halves: Vec<u64> = cols::IN
        .iter()
        .map(|&c| t.get_main(a, c).to_raw())
        .chain([ext])
        .collect();

    t.set_main(a, cols::X_0, FE::zero());
    t.set_main(a, cols::OUT_0, FE::from(s.forged));
    t.set_main(b, cols::ZBS, FE::from(2u64));
    t.set_main(b, cols::IN_0, FE::from(s.lhs));
    t.set_main(b, cols::BIT_SHIFT, FE::from(bit_shift));

    let bw = &mut traces.bitwise.main_table;
    for half in halves {
        let row = bw_row_index((half & 0xFF) as u8, (half >> 8) as u8, bit_shift as u8);
        let old = *bw.get(row, bw_cols::MU_HWSL);
        bw.set(row, bw_cols::MU_HWSL, old - FE::one());
    }
}

fn verifier_accepts(s: &Scenario, proof: &VmProof) -> bool {
    crate::verify_with_options(proof, &asm_elf_bytes(s.guest), &opts(), None, None)
        .expect("verify must not error")
}

fn u64_output(proof: &VmProof) -> u64 {
    u64::from_le_bytes(proof.public_output[..8].try_into().expect("8-byte output"))
}

/// Honest proving of the scenario verifies and commits the honest result.
fn assert_honest_verifies(s: &Scenario) {
    let proof = craft_proof(
        s,
        Forge {
            claim: false,
            rows: false,
        },
    )
    .expect("honest proving");
    assert_eq!(u64_output(&proof), s.honest);
    assert!(verifier_accepts(s, &proof), "honest proof must verify");
}

/// Did a crafted proof get accepted? A prover refusal also counts as rejected.
fn crafted_accepted(s: &Scenario, forge: Forge) -> bool {
    match craft_proof(s, forge) {
        Ok(proof) => {
            assert_eq!(
                u64_output(&proof),
                s.forged,
                "public output must be the forged value"
            );
            verifier_accepts(s, &proof)
        }
        Err(_) => false,
    }
}

#[test]
fn mul_honest_control_verifies() {
    assert_honest_verifies(&MUL);
}

/// Negative control: the CPU claims 99 but the MUL rows are left honest.
#[test]
fn mul_unanswered_forged_claim_is_rejected() {
    assert!(!crafted_accepted(
        &MUL,
        Forge {
            claim: true,
            rows: false
        }
    ));
}

/// Regression: `3 * 5 = 99` via `μ_lo = 1, μ_hi = −1` (verified on ffc4ac19).
#[test]
fn mul_negative_mu_hi_forgery_is_rejected() {
    assert!(
        !crafted_accepted(
            &MUL,
            Forge {
                claim: true,
                rows: true
            }
        ),
        "FORGERY: 3 * 5 = 99 was accepted — μ_lo/μ_hi are not bounded"
    );
}

#[test]
fn dvrm_honest_control_verifies() {
    assert_honest_verifies(&DIVU);
}

/// Negative control: the CPU claims 7 / 2 = 100 but the DVRM rows are left honest.
#[test]
fn dvrm_unanswered_forged_claim_is_rejected() {
    assert!(!crafted_accepted(
        &DIVU,
        Forge {
            claim: true,
            rows: false
        }
    ));
}

/// Regression: `7 / 2 = 100` via `μ_q = 1, μ_r = −1` (verified on ffc4ac19).
#[test]
fn dvrm_negative_mu_r_forgery_is_rejected() {
    assert!(
        !crafted_accepted(
            &DIVU,
            Forge {
                claim: true,
                rows: true
            }
        ),
        "FORGERY: 7 / 2 = 100 was accepted — μ_q/μ_r are not bounded"
    );
}

#[test]
fn shift_honest_control_verifies() {
    assert_honest_verifies(&SLL);
}

/// Negative control: the CPU claims 1 << 1 = 0 but the SHIFT rows are left honest.
#[test]
fn shift_unanswered_forged_claim_is_rejected() {
    assert!(!crafted_accepted(
        &SLL,
        Forge {
            claim: true,
            rows: false
        }
    ));
}

/// Regression: `1 << 1 = 0` via a `zbs = 2` padding row (verified on ffc4ac19).
#[test]
fn shift_zbs_two_padding_row_forgery_is_rejected() {
    assert!(
        !crafted_accepted(
            &SLL,
            Forge {
                claim: true,
                rows: true
            }
        ),
        "FORGERY: 1 << 1 = 0 was accepted — zbs is not bit-constrained"
    );
}
