//! `LogReadOnlyRAP` carries a constraint program.
//!
//! The CUDA composition arm evaluates `AIR::constraint_program()` once main
//! and aux are device-resident; `LogReadOnlyRAP` (the AIR of the checked-in
//! S3/S2 proof vectors) had none, so every device-proved vector test panicked
//! before it compared a byte. The program is captured from the SAME
//! `LogReadOnlyRAPConstraints` body the CPU folders run, so the device
//! composes the same polynomials and the vector bytes cannot move.
//!
//! These CPU tests pin that equality on random two-row frames three ways:
//! the prover folder (the CPU prover's hot path) == the captured program under
//! the generic interpreter == the lowered device program under its host model
//! (`eval_device_program`, the CPU model of the GPU kernel).

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as E;
use math::field::goldilocks::GoldilocksField as F;

use crate::constraint_ir::{DeviceProgram, eval_device_program, eval_program};
use crate::examples::read_only_memory_logup::LogReadOnlyRAP;
use crate::frame::Frame;
use crate::proof::options::ProofOptions;
use crate::table::TableView;
use crate::traits::{AIR, TransitionEvaluationContext};

type Felt = FieldElement<F>;
type Ext = FieldElement<E>;

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn fp(&mut self) -> Felt {
        Felt::from(self.next_u64())
    }
    fn ext(&mut self) -> Ext {
        Ext::from_raw([self.fp(), self.fp(), self.fp()])
    }
}

fn limbs(x: &Ext) -> [u64; 3] {
    let v = x.value();
    [v[0].canonical(), v[1].canonical(), v[2].canonical()]
}

fn from_limbs(l: [u64; 3]) -> Ext {
    Ext::from_raw([Felt::from(l[0]), Felt::from(l[1]), Felt::from(l[2])])
}

fn air() -> LogReadOnlyRAP<F, E> {
    LogReadOnlyRAP::<F, E>::new(&ProofOptions::default_test_options())
}

#[test]
fn the_log_read_only_program_has_the_air_shape() {
    let air = air();
    let prog = air.constraint_program();
    assert_eq!(prog.roots.len(), air.num_transition_constraints());
    assert_eq!(prog.num_base, air.num_base_transition_constraints());
    assert_eq!(prog.num_base, 2, "continuity and single-value are base");
    // Cached: a second call hands back the same program.
    assert!(std::ptr::eq(prog, air.constraint_program()));
}

#[test]
fn the_log_read_only_program_equals_the_prover_folder_and_the_device_model() {
    let air = air();
    let prog = air.constraint_program();
    let dev = DeviceProgram::lower(prog);
    let n = air.num_transition_constraints();
    let nb = air.num_base_transition_constraints();
    let (main_w, aux_w) = air.trace_layout();

    let mut rng = SplitMix64(0x1F1C_D2D2_0000_0001);
    for trial in 0..500 {
        let main: Vec<Vec<Felt>> = (0..2)
            .map(|_| (0..main_w).map(|_| rng.fp()).collect())
            .collect();
        let aux: Vec<Vec<Ext>> = (0..2)
            .map(|_| (0..aux_w).map(|_| rng.ext()).collect())
            .collect();
        let rap = vec![rng.ext(), rng.ext()];
        let alphas: Vec<Ext> = Vec::new();
        let offset = Ext::zero();

        let steps: Vec<TableView<F, E>> = main
            .iter()
            .zip(aux.iter())
            .map(|(m, a)| TableView::<F, E>::new(vec![m.clone()], vec![a.clone()]))
            .collect();
        let frame = Frame::<F, E>::new(steps);
        let ctx =
            TransitionEvaluationContext::new_prover(frame.as_row_frame(), &rap, &alphas, &offset);

        // The CPU prover's path.
        let mut folder_base = vec![Felt::zero(); nb];
        let mut folder_ext = vec![Ext::zero(); n];
        air.compute_transition_prover(&ctx, &mut folder_base, &mut folder_ext);

        // The captured program, generic interpreter.
        let mut interp_base = vec![Felt::zero(); nb];
        let mut interp_ext = vec![Ext::zero(); n];
        eval_program(prog, &ctx, &mut interp_base, &mut interp_ext);
        assert_eq!(folder_base, interp_base, "base constraints, trial {trial}");
        assert_eq!(
            folder_ext[nb..],
            interp_ext[nb..],
            "ext constraints, trial {trial}"
        );

        // The lowered device program, host model of the GPU kernel.
        let main_raw: Vec<Vec<u64>> = main
            .iter()
            .map(|r| r.iter().map(|x| x.canonical()).collect())
            .collect();
        let aux_raw: Vec<Vec<[u64; 3]>> =
            aux.iter().map(|r| r.iter().map(limbs).collect()).collect();
        let rap_raw: Vec<[u64; 3]> = rap.iter().map(limbs).collect();
        let mut base_dev = vec![0u64; nb];
        let mut ext_dev = vec![[0u64; 3]; n];
        eval_device_program(
            &dev,
            &main_raw,
            &aux_raw,
            &rap_raw,
            &[],
            limbs(&offset),
            &mut base_dev,
            &mut ext_dev,
        );
        for c in 0..nb {
            assert_eq!(
                Felt::from(base_dev[c]),
                folder_base[c],
                "device base {c}, trial {trial}"
            );
        }
        for c in nb..n {
            assert_eq!(
                from_limbs(ext_dev[c]),
                folder_ext[c],
                "device ext {c}, trial {trial}"
            );
        }
    }
}
