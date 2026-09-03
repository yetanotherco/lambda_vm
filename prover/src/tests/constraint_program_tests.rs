//! Combined-program differential tests: every production table's CAPTURED
//! constraint program — the [`stark::traits::AIR::constraint_program`] the
//! GPU interpreter will consume — is interpreted and compared bit-for-bit
//! against the compiled folders on random off-trace frames.
//!
//! This is the interpreter-side counterpart of the folder coverage in
//! `constraint_set_tests_*` (base bodies only) and
//! `lookup::logup_single_source_tests` (synthetic LogUp layouts only):
//! here each table's REAL bus-interaction layout runs through capture with
//! its base constraints spliced ahead of the LogUp suffix, so multi-pair
//! layouts, every production `Multiplicity` variant, and `idx_base > 0`
//! LogUp emission are all exercised on the interpreter path — which has no
//! production caller until the GPU lands, and therefore no other safety net.
//!
//! The folders are the oracle: they are the production prove/verify path,
//! independently pinned by the prove→verify suites and cross-version
//! verification.

use math::field::element::FieldElement;
use stark::constraint_ir::{eval_program, eval_program_verifier};
use stark::frame::Frame;
use stark::proof::options::GoldilocksCubicProofOptions;
use stark::table::TableView;
use stark::traits::{AIR, TransitionEvaluationContext};

use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::*;

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type Fp = FieldElement<Gl>;
type Fp3 = FieldElement<Ext3>;

const TRIALS: usize = 200;

/// Deterministic SplitMix64 (no `rand` dependency).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn fp3(&mut self) -> Fp3 {
        Fp3::new([
            Fp::from(self.next_u64()),
            Fp::from(self.next_u64()),
            Fp::from(self.next_u64()),
        ])
    }
}

/// The differential for one production AIR: capture the combined program
/// once via the production entry point, then assert on random two-step
/// frames that interpreting it matches the compiled folders — prover side
/// (`eval_program` vs `compute_transition_prover`) and verifier side
/// (`eval_program_verifier` vs `compute_transition`).
fn check_air(air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>, label: &str) {
    let n = air.context().num_transition_constraints;
    let num_base = air.num_base_transition_constraints();
    let (n_main, n_aux) = air.trace_layout();

    // The production capture (lazy OnceLock behind the AIR).
    let prog = air.constraint_program();
    assert_eq!(prog.roots.len(), n, "[{label}] one root per constraint");
    // Release-safe exact-once backstop: root id 0 is the reserved base-zero
    // sentinel, and no production constraint is identically zero — a root
    // left at the sentinel means its constraint_idx was never emitted
    // (e.g. a double-emit/skip typo), which the debug-only EmitTracker
    // would miss in a release test build.
    for (i, &root) in prog.roots.iter().enumerate() {
        assert_ne!(root, 0, "[{label}] constraint {i} was never captured");
    }

    let mut rng = SplitMix64(0xBADC_0FFE ^ label.len() as u64);
    for trial in 0..TRIALS {
        // Random two-step prover frame shaped like this table.
        let mk_step = |rng: &mut SplitMix64| {
            let main: Vec<Fp> = (0..n_main).map(|_| Fp::from(rng.next_u64())).collect();
            let aux: Vec<Fp3> = (0..n_aux).map(|_| rng.fp3()).collect();
            TableView::new(vec![main], vec![aux])
        };
        let frame = Frame::<Gl, Ext3>::new(vec![mk_step(&mut rng), mk_step(&mut rng)]);
        let challenges = vec![rng.fp3(), rng.fp3()]; // [z, alpha]
        let alphas: Vec<Fp3> = (0..air.max_bus_elements() + 2).map(|_| rng.fp3()).collect();
        let offset = rng.fp3();

        let ctx = TransitionEvaluationContext::new_prover(
            frame.as_row_frame(),
            &challenges,
            &alphas,
            &offset,
        );

        // --- prover side: folder vs interpreter ---
        let mut f_base = vec![Fp::zero(); num_base];
        let mut f_ext = vec![Fp3::zero(); n];
        air.compute_transition_prover(&ctx, &mut f_base, &mut f_ext);

        let mut i_base = vec![Fp::zero(); num_base];
        let mut i_ext = vec![Fp3::zero(); n];
        eval_program(prog, &ctx, &mut i_base, &mut i_ext);

        for c in 0..num_base {
            assert_eq!(
                f_base[c], i_base[c],
                "[{label}] prover folder vs interpreter, base constraint {c}, trial {trial}"
            );
        }
        for c in num_base..n {
            assert_eq!(
                f_ext[c], i_ext[c],
                "[{label}] prover folder vs interpreter, ext constraint {c}, trial {trial}"
            );
        }

        // --- verifier side: embed the frame into the extension ---
        let embed = |step: &TableView<Gl, Ext3>| -> TableView<Ext3, Ext3> {
            let main: Vec<Fp3> = (0..n_main)
                .map(|c| step.get_main_evaluation_element(0, c).to_extension())
                .collect();
            let aux: Vec<Fp3> = (0..n_aux)
                .map(|c| *step.get_aux_evaluation_element(0, c))
                .collect();
            TableView::new(vec![main], vec![aux])
        };
        let vframe: Frame<Ext3, Ext3> = Frame::new(vec![
            embed(frame.get_evaluation_step(0)),
            embed(frame.get_evaluation_step(1)),
        ]);
        let vctx = TransitionEvaluationContext::<Gl, Ext3>::new_verifier(
            &vframe,
            &challenges,
            &alphas,
            &offset,
        );

        let v_folder = air.compute_transition(&vctx);
        let mut v_interp = vec![Fp3::zero(); n];
        eval_program_verifier(prog, &vctx, &mut v_interp);

        for c in 0..n {
            assert_eq!(
                v_folder[c], v_interp[c],
                "[{label}] verifier folder vs interpreter, constraint {c}, trial {trial}"
            );
        }
    }
}

#[test]
fn all_table_programs_match_folders() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    check_air(&create_cpu_air(&opts), "CPU");
    check_air(&create_memmove_air(&opts), "DMA");
    check_air(&create_bitwise_air(&opts), "BITWISE");
    check_air(&create_lt_air(&opts), "LT");
    check_air(&create_shift_air(&opts), "SHIFT");
    check_air(&create_eq_air(&opts), "EQ");
    check_air(&create_bytewise_air(&opts), "BYTEWISE");
    check_air(&create_store_air(&opts), "STORE");
    check_air(&create_cpu32_air(&opts), "CPU32");
    check_air(&create_memw_air(&opts), "MEMW");
    check_air(&create_memw_aligned_air(&opts), "MEMW_A");
    check_air(&create_memw_register_air(&opts), "MEMW_R");
    check_air(&create_load_air(&opts), "LOAD");
    check_air(&create_decode_air(&opts), "DECODE");
    check_air(&create_mul_air(&opts), "MUL");
    check_air(&create_dvrm_air(&opts), "DVRM");
    check_air(&create_branch_air(&opts), "BRANCH");
    check_air(&create_halt_air(&opts), "HALT");
    check_air(&create_commit_air(&opts), "COMMIT");
    check_air(&create_page_air(&opts, 0x1000), "PAGE");
    check_air(&create_register_air(&opts), "REGISTER");
    check_air(&create_keccak_air(&opts), "KECCAK");
    check_air(&create_keccak_rnd_air(&opts), "KECCAK_RND");
    check_air(&create_keccak_rc_air(&opts), "KECCAK_RC");
    check_air(&create_ecsm_air(&opts), "ECSM");
    check_air(&create_ecdas_air(&opts), "ECDAS");
    check_air(&create_hint_air(&opts), "HINT");
}
