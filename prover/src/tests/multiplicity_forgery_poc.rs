//! End-to-end regression tests for unconstrained multiplicities: MUL's
//! `μ_lo`/`μ_hi` and DVRM's `μ_q`/`μ_r` must be bounded and non-negative, and
//! SHIFT's `zbs` and LOAD's `μ` must be bits.
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
//! **LOAD.** LOAD never deduplicates, but μ was pinned to 1 only when a
//! read2/4/8 flag was set, so a byte-load row's μ was free. A `μ = −1 / μ = 1`
//! pair of LBU rows with the same `res[0]` cancels on MEMORY/MEMW (neither
//! carries `sign_bit`) while netting `MSB8[res[0]] → 0` minus `→ 1`, which
//! cancels a real LB row's forged sign bit (`lb 0x05 = 0xFF..05` verified
//! against `main`). The fix bit-constrains μ.
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
use executor::vm::logs::Log;

fn opts() -> ProofOptions {
    crate::GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// One forgery scenario: a guest whose target instruction (picked out of the
/// logs by `is_target`) yields `honest`, and the chip rewrite that makes it
/// yield `forged` instead.
struct Scenario {
    guest: &'static str,
    lhs: u64,
    rhs: u64,
    honest: u64,
    forged: u64,
    is_target: fn(&Log, &Scenario) -> bool,
    forge_rows: fn(&mut Traces, &Scenario),
}

/// An `lhs ∘ rhs = honest` arithmetic instruction.
fn arith_log(l: &Log, s: &Scenario) -> bool {
    l.src1_val == s.lhs && l.src2_val == s.rhs && l.dst_val == s.honest
}

/// The `lb t1, 8(sp)` log: the only instruction with a non-zero base register
/// that writes the honest byte (`li t0, 5` reads `zero`, `sb`/`sd` write none).
fn lb_log(l: &Log, s: &Scenario) -> bool {
    l.src1_val != 0 && l.dst_val == s.honest
}

const MUL: Scenario = Scenario {
    guest: "poc_mul_commit",
    lhs: 3,
    rhs: 5,
    honest: 15,
    forged: 99,
    is_target: arith_log,
    forge_rows: forge_mul_rows,
};

const DIVU: Scenario = Scenario {
    guest: "poc_divu_commit",
    lhs: 7,
    rhs: 2,
    honest: 3,
    forged: 100,
    is_target: arith_log,
    forge_rows: forge_dvrm_rows,
};

const SLL: Scenario = Scenario {
    guest: "poc_sll_commit",
    lhs: 1,
    rhs: 1,
    honest: 2,
    forged: 0,
    is_target: arith_log,
    forge_rows: forge_shift_rows,
};

const LB: Scenario = Scenario {
    guest: "poc_lb_commit",
    lhs: 0,
    rhs: 0,
    honest: 5,
    forged: 0xFFFF_FFFF_FFFF_FF05,
    is_target: lb_log,
    forge_rows: forge_load_rows,
};

/// Which deviations the malicious prover applies.
#[derive(Clone, Copy)]
struct Forge {
    /// Rewrite the execution so the CPU claims the target yields `forged`.
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
            .position(|l| (s.is_target)(l, s))
            .expect("target instruction log not found");
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

/// LOAD: `lb` of the byte 0x05 forged to sign-extend as if negative
/// (`0xFFFF_FFFF_FFFF_FF05`).
///
/// * Row A (the real LB, `μ = 1, signed = 1`) gets `res[1..8] = 0xFF` and
///   `sign_bit = 1`, which the extension constraints accept. Its MSB8 send
///   `(0x05, 1)` is a tuple BITWISE has no row for.
/// * Padding rows B and C become an LBU-shaped pair (`signed = 0`, so the
///   extension constraints pin `res[1..8] = 0` whatever `sign_bit` is) with the
///   same `res[0] = 0x05`: B has `μ = −1, sign_bit = 1`, C has `μ = 1,
///   sign_bit = 0`. Their MEMORY and MEMW tuples carry no `sign_bit` and cancel
///   each other; their MSB8 sends net `(0x05, 0) − (0x05, 1)`, which cancels
///   A's bogus send and re-supplies the honest one. BITWISE is untouched.
/// * The MEMW read answering A carries `res` as both `old` and `value`. For a
///   1-byte read only byte 0 is bound to a memory token, so its bytes 1..8 are
///   free and are rewritten to match A.
fn forge_load_rows(traces: &mut Traces, s: &Scenario) {
    use crate::tables::load::cols;
    use crate::tables::memw_aligned::cols as ma;
    const FF: u64 = 0xFF;
    let byte = s.honest;

    let t = &mut traces.loads[0];
    let rows = t.num_rows();
    let a = (0..rows)
        .find(|&r| {
            *t.get_main(r, cols::MU) == FE::one()
                && *t.get_main(r, cols::SIGNED) == FE::one()
                && [cols::READ2, cols::READ4, cols::READ8]
                    .iter()
                    .all(|&c| *t.get_main(r, c) == FE::zero())
                && *t.get_main(r, cols::RES[0]) == FE::from(byte)
        })
        .expect("honest LB row not found");
    assert_eq!(*t.get_main(a, cols::SIGN_BIT), FE::zero());
    let ts = [cols::TIMESTAMP_0, cols::TIMESTAMP_1].map(|c| *t.get_main(a, c));
    let padding: Vec<usize> = (0..rows)
        .filter(|&r| (0..cols::NUM_COLUMNS).all(|c| *t.get_main(r, c) == FE::zero()))
        .take(2)
        .collect();
    let [b, c] = padding[..] else {
        panic!("two all-zero padding rows are needed for the cancelling pair");
    };

    for &col in &cols::RES[1..] {
        t.set_main(a, col, FE::from(FF));
    }
    t.set_main(a, cols::SIGN_BIT, FE::one());
    for (row, mu, sign_bit) in [
        (b, FE::zero() - FE::one(), FE::one()),
        (c, FE::one(), FE::zero()),
    ] {
        t.set_main(row, cols::RES[0], FE::from(byte));
        t.set_main(row, cols::MU, mu);
        t.set_main(row, cols::SIGN_BIT, sign_bit);
    }

    let m = traces.memw_aligneds.iter_mut().find_map(|m| {
        (0..m.num_rows())
            .find(|&r| {
                *m.get_main(r, ma::IS_REGISTER) == FE::zero()
                    && *m.get_main(r, ma::MU_READ) == FE::one()
                    && *m.get_main(r, ma::TIMESTAMP_0) == ts[0]
                    && *m.get_main(r, ma::TIMESTAMP_1) == ts[1]
            })
            .map(|r| (m, r))
    });
    let (m, r) = m.expect("the MEMW_A read answering the LB not found");
    assert_eq!(*m.get_main(r, ma::VALUE[0]), FE::from(byte));
    for i in 1..8 {
        m.set_main(r, ma::OLD[i], FE::from(FF));
        m.set_main(r, ma::VALUE[i], FE::from(FF));
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

#[test]
fn load_honest_control_verifies() {
    assert_honest_verifies(&LB);
}

/// Negative control: the CPU claims `lb 0x05 = 0xFF..05` but the LOAD rows are
/// left honest.
#[test]
fn load_unanswered_forged_claim_is_rejected() {
    assert!(!crafted_accepted(
        &LB,
        Forge {
            claim: true,
            rows: false
        }
    ));
}

/// Regression: `lb` of 0x05 sign-extended to `0xFFFF_FFFF_FFFF_FF05` via a
/// `μ = −1 / μ = 1` byte-load pair that swaps A's MSB8 tuple.
#[test]
fn load_negative_mu_sign_extension_forgery_is_rejected() {
    assert!(
        !crafted_accepted(
            &LB,
            Forge {
                claim: true,
                rows: true
            }
        ),
        "FORGERY: lb 0x05 = 0xFFFFFFFFFFFFFF05 was accepted — LOAD's μ is not bit-constrained"
    );
}
