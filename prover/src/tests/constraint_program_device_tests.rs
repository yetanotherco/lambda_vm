//! Device-lowering differential tests: every production table's captured
//! constraint program is lowered to its flat GPU form
//! ([`stark::constraint_ir::DeviceProgram`]) and the CPU walk over that flat
//! blob ([`eval_device_program`]) is compared bit-for-bit against the compiled
//! prover folder on random off-trace frames.
//!
//! This is the pre-GPU parity oracle: the CUDA kernel is a transliteration of
//! [`eval_device_program`], so a bit-for-bit match here pins the on-device
//! layout and control flow (op tags, `Var` packing, const side-tables, the
//! base/ext output split) against the production path — across every real
//! bus-interaction layout, `Multiplicity` variant, and LogUp suffix — *before*
//! any CUDA exists. It complements the synthetic all-ops coverage in
//! `stark::constraint_ir::device`'s own unit test.
//!
//! The folder (`compute_transition_prover`) is the oracle: it is the production
//! prove path, independently pinned by the prove→verify suites and
//! cross-version verification.

use math::field::element::FieldElement;
use stark::constraint_ir::{DeviceProgram, eval_device_program};
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

/// Extension element → raw `[u64; 3]` limbs (the device representation).
fn enc(x: &Fp3) -> [u64; 3] {
    let limbs = x.value();
    [*limbs[0].value(), *limbs[1].value(), *limbs[2].value()]
}

/// Device differential for one production AIR: lower the captured program once,
/// then assert on random two-step prover frames that the flat-blob walk matches
/// the compiled folder, base and extension constraints alike.
fn check_air_device(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    label: &str,
) {
    let n = air.context().num_transition_constraints;
    let num_base = air.num_base_transition_constraints();
    let (n_main, n_aux) = air.trace_layout();

    let prog = air.constraint_program();
    let dev = DeviceProgram::lower(prog);
    assert_eq!(dev.roots.len(), n, "[{label}] one root per constraint");
    assert_eq!(
        dev.num_base as usize, num_base,
        "[{label}] num_base preserved"
    );

    let mut rng = SplitMix64(0xDECA_FBAD ^ label.len() as u64);
    for trial in 0..TRIALS {
        let mk_step = |rng: &mut SplitMix64| {
            let main: Vec<Fp> = (0..n_main).map(|_| Fp::from(rng.next_u64())).collect();
            let aux: Vec<Fp3> = (0..n_aux).map(|_| rng.fp3()).collect();
            TableView::new(vec![main], vec![aux])
        };
        let frame = Frame::<Gl, Ext3>::new(vec![mk_step(&mut rng), mk_step(&mut rng)]);
        let challenges = vec![rng.fp3(), rng.fp3()]; // [z, alpha]
        let alphas: Vec<Fp3> = (0..air.max_bus_elements() + 2).map(|_| rng.fp3()).collect();
        let offset = rng.fp3();

        // Raw-limb inputs for the device walk, extracted from the frame.
        let main_raw: Vec<Vec<u64>> = (0..2)
            .map(|off| {
                let step = frame.get_evaluation_step(off);
                (0..n_main)
                    .map(|c| *step.get_main_evaluation_element(0, c).value())
                    .collect()
            })
            .collect();
        let aux_raw: Vec<Vec<[u64; 3]>> = (0..2)
            .map(|off| {
                let step = frame.get_evaluation_step(off);
                (0..n_aux)
                    .map(|c| enc(step.get_aux_evaluation_element(0, c)))
                    .collect()
            })
            .collect();
        let rap_raw: Vec<[u64; 3]> = challenges.iter().map(enc).collect();
        let alpha_raw: Vec<[u64; 3]> = alphas.iter().map(enc).collect();
        let off_raw = enc(&offset);

        // Oracle: the compiled prover folder.
        let ctx = TransitionEvaluationContext::new_prover(
            frame.as_row_frame(),
            &challenges,
            &alphas,
            &offset,
        );
        let mut f_base = vec![Fp::zero(); num_base];
        let mut f_ext = vec![Fp3::zero(); n];
        air.compute_transition_prover(&ctx, &mut f_base, &mut f_ext);

        // Device walk over the flat blob.
        let mut d_base = vec![0u64; num_base];
        let mut d_ext = vec![[0u64; 3]; n];
        eval_device_program(
            &dev,
            &main_raw,
            &aux_raw,
            &rap_raw,
            &alpha_raw,
            off_raw,
            &mut d_base,
            &mut d_ext,
        );

        for c in 0..num_base {
            assert_eq!(
                d_base[c],
                *f_base[c].value(),
                "[{label}] folder vs device, base constraint {c}, trial {trial}"
            );
        }
        for c in num_base..n {
            assert_eq!(
                d_ext[c],
                enc(&f_ext[c]),
                "[{label}] folder vs device, ext constraint {c}, trial {trial}"
            );
        }
    }
}

#[test]
fn all_table_programs_lower_and_match_folders() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    check_air_device(&create_cpu_air(&opts), "CPU");
    check_air_device(&create_dma_air(&opts), "DMA");
    check_air_device(&create_bitwise_air(&opts), "BITWISE");
    check_air_device(&create_lt_air(&opts), "LT");
    check_air_device(&create_shift_air(&opts), "SHIFT");
    check_air_device(&create_eq_air(&opts), "EQ");
    check_air_device(&create_bytewise_air(&opts), "BYTEWISE");
    check_air_device(&create_store_air(&opts), "STORE");
    check_air_device(&create_cpu32_air(&opts), "CPU32");
    check_air_device(&create_memw_air(&opts), "MEMW");
    check_air_device(&create_memw_aligned_air(&opts), "MEMW_A");
    check_air_device(&create_memw_register_air(&opts), "MEMW_R");
    check_air_device(&create_load_air(&opts), "LOAD");
    check_air_device(&create_decode_air(&opts), "DECODE");
    check_air_device(&create_mul_air(&opts), "MUL");
    check_air_device(&create_dvrm_air(&opts), "DVRM");
    check_air_device(&create_branch_air(&opts), "BRANCH");
    check_air_device(&create_halt_air(&opts), "HALT");
    check_air_device(&create_commit_air(&opts), "COMMIT");
    check_air_device(&create_page_air(&opts, 0x1000), "PAGE");
    check_air_device(&create_register_air(&opts), "REGISTER");
    check_air_device(&create_keccak_air(&opts), "KECCAK");
    check_air_device(&create_keccak_rnd_air(&opts), "KECCAK_RND");
    check_air_device(&create_keccak_rc_air(&opts), "KECCAK_RC");
    check_air_device(&create_ecsm_air(&opts), "ECSM");
    check_air_device(&create_ecdas_air(&opts), "ECDAS");
    check_air_device(&create_hint_air(&opts), "HINT");
}
