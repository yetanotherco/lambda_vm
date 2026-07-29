//! Serialization bit-exactness: for every production table, the constraint
//! artifact SERIALIZED at build time and read back evaluates identically to the
//! compiled folders.
//!
//! `constraint_program_tests` already pins the in-memory capture against both
//! folders. This suite pins the extra hop that "constraints as data" adds — the
//! bytes:
//!
//! ```text
//!   capture → artifact → to_bytes → from_bytes → lift → evaluate
//! ```
//!
//! Every stage in that chain is a place a program can change meaning: a
//! truncated `u32` index, a constant that loses canonical form through raw
//! limbs, a metadata list that reorders, an op tag that decodes to a different
//! operation. None of it is visible to a test that only exercises the in-memory
//! program, which is why this suite runs the deserialized artifact — never the
//! captured object — against the folders.
//!
//! The folders are the oracle: they are the production prove/verify path,
//! independently pinned by the prove→verify suites and cross-version
//! verification. All three evaluation paths are checked against them:
//! `eval_program` (prover shape), `eval_program_verifier` (OOD shape, the
//! recursion path), and `eval_device_program` (the flat blob).

use math::field::element::FieldElement;
use stark::constraint_ir::{
    ConstraintArtifact, eval_device_program, eval_program, eval_program_verifier,
};
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

const TRIALS: usize = 100;

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

/// One production AIR's serialized-artifact differential. Returns the artifact's
/// measured size for the size report.
fn check_air_artifact(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    label: &str,
) -> ArtifactSize {
    let n = air.context().num_transition_constraints;
    let num_base = air.num_base_transition_constraints();
    let (n_main, n_aux) = air.trace_layout();

    let artifact = ConstraintArtifact::capture(air);
    artifact
        .validate_against(air)
        .unwrap_or_else(|e| panic!("[{label}] freshly captured artifact rejected: {e}"));

    // The wire hop. Everything below runs the DESERIALIZED artifact, so a
    // codec bug cannot hide behind the in-memory object.
    let bytes = artifact
        .to_bytes()
        .unwrap_or_else(|e| panic!("[{label}] serialize failed: {e}"));
    let artifact = ConstraintArtifact::from_bytes(&bytes)
        .unwrap_or_else(|e| panic!("[{label}] deserialize failed: {e}"));
    artifact
        .validate_against(air)
        .unwrap_or_else(|e| panic!("[{label}] deserialized artifact rejected: {e}"));

    let prog = artifact.program();
    let dev = artifact.device_program();

    // Structural identity against the capture the AIR itself would produce.
    let captured = air.constraint_program();
    assert_eq!(
        prog.nodes, captured.nodes,
        "[{label}] nodes changed on the wire"
    );
    assert_eq!(
        prog.dims, captured.dims,
        "[{label}] dims changed on the wire"
    );
    assert_eq!(
        prog.roots, captured.roots,
        "[{label}] roots changed on the wire"
    );
    assert_eq!(
        prog.num_base, captured.num_base,
        "[{label}] num_base changed"
    );
    assert_eq!(
        prog.base_consts, captured.base_consts,
        "[{label}] base constants changed on the wire"
    );
    assert_eq!(
        prog.ext_consts, captured.ext_consts,
        "[{label}] ext constants changed on the wire"
    );
    assert_eq!(prog.roots.len(), n, "[{label}] one root per constraint");

    // Release-safe exact-once backstop, as in `constraint_program_tests`: root
    // id 0 is the reserved base-zero sentinel and no production constraint is
    // identically zero, so a root still at the sentinel means that constraint
    // was never captured.
    for (i, &root) in prog.roots.iter().enumerate() {
        assert_ne!(root, 0, "[{label}] constraint {i} was never captured");
    }

    let mut rng = SplitMix64(0x5EED_1234 ^ label.len() as u64);
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

        let ctx = TransitionEvaluationContext::new_prover(
            frame.as_row_frame(),
            &challenges,
            &alphas,
            &offset,
        );

        // --- oracle: the compiled prover folder ---
        let mut f_base = vec![Fp::zero(); num_base];
        let mut f_ext = vec![Fp3::zero(); n];
        air.compute_transition_prover(&ctx, &mut f_base, &mut f_ext);

        // --- path 1: generic interpreter over the deserialized program ---
        let mut i_base = vec![Fp::zero(); num_base];
        let mut i_ext = vec![Fp3::zero(); n];
        eval_program(&prog, &ctx, &mut i_base, &mut i_ext);

        for c in 0..num_base {
            assert_eq!(
                f_base[c], i_base[c],
                "[{label}] prover folder vs serialized program, base constraint {c}, trial {trial}"
            );
        }
        for c in num_base..n {
            assert_eq!(
                f_ext[c], i_ext[c],
                "[{label}] prover folder vs serialized program, ext constraint {c}, trial {trial}"
            );
        }

        // --- path 2: flat device walk over the deserialized blob ---
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

        let mut d_base = vec![0u64; num_base];
        let mut d_ext = vec![[0u64; 3]; n];
        eval_device_program(
            &dev,
            &main_raw,
            &aux_raw,
            &rap_raw,
            &alpha_raw,
            enc(&offset),
            &mut d_base,
            &mut d_ext,
        );
        for c in 0..num_base {
            assert_eq!(
                d_base[c],
                *f_base[c].value(),
                "[{label}] prover folder vs serialized device blob, base constraint {c}, trial {trial}"
            );
        }
        for c in num_base..n {
            assert_eq!(
                d_ext[c],
                enc(&f_ext[c]),
                "[{label}] prover folder vs serialized device blob, ext constraint {c}, trial {trial}"
            );
        }

        // --- path 3: the verifier/OOD shape, i.e. the recursion path ---
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
        eval_program_verifier(&prog, &vctx, &mut v_interp);
        for c in 0..n {
            assert_eq!(
                v_folder[c], v_interp[c],
                "[{label}] verifier folder vs serialized program, constraint {c}, trial {trial}"
            );
        }
    }

    ArtifactSize {
        label: label.to_string(),
        constraints: n,
        nodes: artifact.nodes.len(),
        base_consts: artifact.base_consts.len(),
        ext_consts: artifact.ext_consts.len(),
        bytes: bytes.len(),
    }
}

/// One AIR's measured artifact size.
struct ArtifactSize {
    label: String,
    constraints: usize,
    nodes: usize,
    base_consts: usize,
    ext_consts: usize,
    bytes: usize,
}

/// Every production table's serialized artifact evaluates bit-identically to
/// the compiled folders, on the prover shape, the verifier/OOD shape, and the
/// flat device blob.
#[test]
fn all_table_artifacts_roundtrip_and_match_folders() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let airs = production_airs(&opts);
    assert_eq!(
        airs.len(),
        NUM_PRODUCTION_AIRS,
        "the production AIR list changed size; every per-table suite's coverage moved with it"
    );

    let mut sizes: Vec<ArtifactSize> = Vec::with_capacity(airs.len());
    for (label, air) in &airs {
        sizes.push(check_air_artifact(&**air, label));
    }

    report_sizes(&sizes);
}

/// Print the per-AIR and total artifact sizes — the recursion machine's
/// program-length budget — and hold a ceiling so runaway growth is noticed.
fn report_sizes(sizes: &[ArtifactSize]) {
    let total_bytes: usize = sizes.iter().map(|s| s.bytes).sum();
    let total_nodes: usize = sizes.iter().map(|s| s.nodes).sum();
    let total_constraints: usize = sizes.iter().map(|s| s.constraints).sum();

    println!("\nconstraint artifact sizes (blowup=2)");
    println!(
        "{:<12} {:>7} {:>9} {:>7} {:>7} {:>10}",
        "table", "constr", "nodes", "bconst", "econst", "bytes"
    );
    for s in sizes {
        println!(
            "{:<12} {:>7} {:>9} {:>7} {:>7} {:>10}",
            s.label, s.constraints, s.nodes, s.base_consts, s.ext_consts, s.bytes
        );
    }
    println!(
        "{:<12} {:>7} {:>9} {:>7} {:>7} {:>10}",
        "TOTAL", total_constraints, total_nodes, "", "", total_bytes
    );
    println!(
        "total: {total_nodes} nodes, {total_bytes} bytes ({:.1} KiB) across {} tables\n",
        total_bytes as f64 / 1024.0,
        sizes.len()
    );

    // A loose ceiling: this is a budget signal, not a tight assertion. It exists
    // so a change that multiplies the program size fails here instead of being
    // discovered when the recursion machine will not fit.
    const CEILING_BYTES: usize = 8 * 1024 * 1024;
    assert!(
        total_bytes < CEILING_BYTES,
        "total artifact size {total_bytes} exceeds the {CEILING_BYTES}-byte budget ceiling"
    );
}

/// The captured artifact does not depend on the proof options.
///
/// This is the premise behind leaving `ProofOptions` OUT of the artifact — if it
/// failed, one artifact per table would not be enough and the whole scheme would
/// need an artifact per (table, blowup) pair. `AirContext` carries the options
/// alongside the shape scalars, so the independence is worth pinning rather than
/// assuming from a reading of the constructor.
#[test]
fn artifacts_are_invariant_across_proof_options() {
    let opts2 = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let opts4 = GoldilocksCubicProofOptions::with_blowup(4).expect("blowup=4 valid");

    let airs2 = production_airs(&opts2);
    let airs4 = production_airs(&opts4);

    for ((label, a2), (_, a4)) in airs2.iter().zip(airs4.iter()) {
        let art2 = ConstraintArtifact::capture(&**a2);
        let art4 = ConstraintArtifact::capture(&**a4);
        assert_eq!(
            art2, art4,
            "[{label}] the constraint artifact differs between blowup 2 and 4; it would have to \
             be stored per (table, blowup) pair"
        );
        assert_eq!(
            art2.to_bytes().expect("serialize"),
            art4.to_bytes().expect("serialize"),
            "[{label}] artifact bytes differ across blowup factors"
        );
    }
}

/// An artifact captured from one table must not validate against another.
///
/// The suite above only ever shows the shape check ACCEPTING. Without this, a
/// `validate_against` that returned `Ok(())` unconditionally would pass
/// everything here.
#[test]
fn an_artifact_does_not_validate_against_a_different_table() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let airs = production_airs(&opts);

    let mut checked = 0usize;
    for (i, (label_i, air_i)) in airs.iter().enumerate() {
        let artifact = ConstraintArtifact::capture(&**air_i);
        for (j, (label_j, air_j)) in airs.iter().enumerate() {
            if i == j {
                continue;
            }
            if artifact.validate_against(&**air_j).is_ok() {
                // Two tables can legitimately share every shape scalar (several
                // are bus-only tables with identical layouts), so an accept is
                // only a failure when the programs actually differ.
                let prog_i = air_i.constraint_program();
                let prog_j = air_j.constraint_program();
                assert_eq!(
                    (&prog_i.nodes, &prog_i.roots),
                    (&prog_j.nodes, &prog_j.roots),
                    "[{label_i}] artifact validated against [{label_j}], whose constraint \
                     program is different — the shape check cannot tell them apart"
                );
            } else {
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "validate_against never rejected any cross-table pairing; the check is not live"
    );
}

/// A pre-captured program can be supplied to a production AIR and is used
/// without capture — the scoped verify-path unban, on real tables.
#[test]
fn production_airs_accept_a_precaptured_program() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    // Build the artifact from one instance, install it into a fresh one.
    let artifact = ConstraintArtifact::capture(&create_cpu_air(&opts));
    let air = create_cpu_air(&opts).with_precaptured(artifact.program());

    let supplied = air
        .precaptured_constraint_program()
        .expect("the supplied program must be visible on the guest-safe accessor");
    assert!(
        std::ptr::eq(air.constraint_program(), supplied),
        "constraint_program() must return the supplied program, not a fresh capture"
    );

    // And it still evaluates like the folder.
    let n = air.context().num_transition_constraints;
    let num_base = air.num_base_transition_constraints();
    let (n_main, n_aux) = air.trace_layout();
    let mut rng = SplitMix64(0x00A1_1CE5);
    for _ in 0..16 {
        let mk_step = |rng: &mut SplitMix64| {
            let main: Vec<Fp> = (0..n_main).map(|_| Fp::from(rng.next_u64())).collect();
            let aux: Vec<Fp3> = (0..n_aux).map(|_| rng.fp3()).collect();
            TableView::new(vec![main], vec![aux])
        };
        let frame = Frame::<Gl, Ext3>::new(vec![mk_step(&mut rng), mk_step(&mut rng)]);
        let challenges = vec![rng.fp3(), rng.fp3()];
        let alphas: Vec<Fp3> = (0..air.max_bus_elements() + 2).map(|_| rng.fp3()).collect();
        let offset = rng.fp3();
        let ctx = TransitionEvaluationContext::new_prover(
            frame.as_row_frame(),
            &challenges,
            &alphas,
            &offset,
        );

        let mut f_base = vec![Fp::zero(); num_base];
        let mut f_ext = vec![Fp3::zero(); n];
        air.compute_transition_prover(&ctx, &mut f_base, &mut f_ext);

        let mut i_base = vec![Fp::zero(); num_base];
        let mut i_ext = vec![Fp3::zero(); n];
        eval_program(supplied, &ctx, &mut i_base, &mut i_ext);

        assert_eq!(f_base, i_base);
        for c in num_base..n {
            assert_eq!(f_ext[c], i_ext[c]);
        }
    }
}
