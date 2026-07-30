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

/// The captured artifact does not depend on the trace length either.
///
/// Same failure mode as the proof-options axis, different variable: if anything
/// in a captured program folded a domain-size-dependent constant, artifacts
/// would multiply per epoch shape.
///
/// The axis is structurally absent — no AIR constructor takes a trace length —
/// so the only route by which one could reach the artifact is
/// `composition_poly_degree_bound(n)`, the single trace-length-dependent method
/// on the trait, whose value the artifact stores divided through by `n`. That
/// division is only sound if the bound is exactly linear, so this sweeps a wide
/// range of `n` per table rather than trusting the two probe points
/// `ConstraintArtifact::capture` checks. A table whose bound had any constant
/// term or any non-linearity would be misrepresented by the stored multiplier,
/// and would show up here.
#[test]
fn artifacts_are_invariant_across_trace_length() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let airs = production_airs(&opts);
    assert_eq!(airs.len(), NUM_PRODUCTION_AIRS);

    for (label, air) in &airs {
        let artifact = ConstraintArtifact::capture(&**air);
        let k = artifact.shape.composition_degree_multiplier as usize;

        for log_n in 4usize..=24 {
            let n = 1usize << log_n;
            assert_eq!(
                air.composition_poly_degree_bound(n),
                k * n,
                "[{label}] composition_poly_degree_bound is not k·n at n=2^{log_n}; the artifact \
                 stores only the linear coefficient, so a trace-length-dependent AIR would need \
                 an artifact per epoch shape"
            );
        }

        // Nothing else on the artifact can vary with the trace length, but pin
        // capture determinism so a future source of nondeterminism (map
        // iteration order in the constant tables, say) is caught here.
        let again = ConstraintArtifact::capture(&**air);
        assert_eq!(artifact, again, "[{label}] capture is not deterministic");
    }
}

/// The four PARAMETERIZED tables produce a different program per parameter
/// value.
///
/// `PAGE` / `GLOBAL_MEMORY` fold a page base into constant bus terms; the two
/// `L2G` tables fold an epoch label. This test does not assert that away — it
/// characterizes it, because it is a real property of the current constraints
/// and the recursion machine has to plan around it.
///
/// # The variation is NOT confined to constant VALUES
///
/// The obvious guess is that two parameter values give the same node array with
/// one constant swapped. That is what `PAGE` and `GLOBAL_MEMORY` do, and it is
/// wrong in general: the builder interns constants by value, so a parameter
/// whose value happens to already be in the constant table costs no new node,
/// while a fresh value appends one — which shifts every later node id and hence
/// the constraint ROOTS. `L2G_GLOBAL` at `epoch_label = 1` reuses the existing
/// `1` constant; at `epoch_label = 7` it appends. Same algebra, different node
/// count and different root ids.
///
/// This matters for the machine-side fix: "swap one constant per page" would be
/// a cheap patch and it is not available. Promoting the parameter to a runtime
/// uniform is, because the ALGEBRA is invariant — which is what the shape and
/// metadata assertions below pin.
#[test]
fn parameterized_airs_vary_per_parameter_value() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    // (label, artifact at parameter A, artifact at parameter B)
    let cases: Vec<(&str, ConstraintArtifact, ConstraintArtifact)> = vec![
        (
            "PAGE",
            ConstraintArtifact::capture(&create_page_air(&opts, 0x1000)),
            ConstraintArtifact::capture(&create_page_air(&opts, 0x9000)),
        ),
        (
            "GLOBAL_MEMORY",
            ConstraintArtifact::capture(&create_global_memory_air(&opts, 0x1000)),
            ConstraintArtifact::capture(&create_global_memory_air(&opts, 0x9000)),
        ),
        (
            "L2G_GLOBAL",
            ConstraintArtifact::capture(&crate::continuation::l2g_global_air(&opts, 1)),
            ConstraintArtifact::capture(&crate::continuation::l2g_global_air(&opts, 7)),
        ),
        (
            "L2G_MEMORY",
            ConstraintArtifact::capture(&crate::continuation::l2g_memory_air(&opts, 1)),
            ConstraintArtifact::capture(&crate::continuation::l2g_memory_air(&opts, 7)),
        ),
    ];

    println!("\nparameterized tables: how two parameter values differ");
    for (label, a, b) in &cases {
        assert_ne!(
            a, b,
            "[{label}] is documented as parameterized but two parameter values gave the same \
             artifact; either the parameter stopped reaching the IR or the test picked two \
             values that collide"
        );

        // Invariant: the ALGEBRA. Same widths, same constraint count, same
        // zerofier shapes, same degree bound — only the embedded parameter
        // moves. This is the property that makes the parameter promotable to a
        // runtime uniform.
        assert_eq!(a.shape, b.shape, "[{label}] shape must not vary");
        assert_eq!(a.meta, b.meta, "[{label}] metadata must not vary");
        assert_eq!(a.num_base, b.num_base, "[{label}] num_base must not vary");
        assert_eq!(
            a.roots.len(),
            b.roots.len(),
            "[{label}] constraint count must not vary"
        );

        // Variable: node count and root ids — an artifact of the builder's
        // hash-consing, not of the constraints. A parameter value already in the
        // constant table costs no new ConstBase node while a fresh one appends,
        // which shifts every later node id.
        //
        // MEASURED, and note the counts are not all +1: L2G_GLOBAL moves 1 node
        // for 1 constant, L2G_MEMORY moves 2 for 1. The second node is some
        // further CSE difference downstream of the reused constant (L2G_MEMORY
        // at epoch_label = 1 contributes the constant 0, which IS node id 0, so
        // expressions over it have more chance to coincide with existing ones) —
        // that specific explanation is inferred, not verified, so the bound
        // below is deliberately loose. What is being pinned is only that the
        // delta stays local rather than the algebra changing shape.
        let node_delta = a.nodes.len().abs_diff(b.nodes.len());
        let const_delta = a.base_consts.len().abs_diff(b.base_consts.len());
        assert!(
            node_delta <= 4 && const_delta <= 1,
            "[{label}] two parameter values changed the program by {node_delta} nodes and \
             {const_delta} constants — too much to be the parameter's own interned constant and \
             its enclosing ops; the variation is structural, not just parametric"
        );
        let roots_moved = a.roots != b.roots;

        println!(
            "  {label:<14} nodes {:>3} vs {:>3}   consts {:>2} vs {:>2}   roots moved: {}",
            a.nodes.len(),
            b.nodes.len(),
            a.base_consts.len(),
            b.base_consts.len(),
            roots_moved
        );
    }
    println!();
}

/// GLOBAL_MEMORY has a second, ENUMERABLE axis: private-input pages are built
/// non-preprocessed, which changes the artifact's SHAPE rather than a constant.
///
/// Worth separating from the parameter axis above because the two have very
/// different consequences. A page base is an arbitrary address, so its artifact
/// set is unbounded; `is_private_input` is a boolean, so GLOBAL_MEMORY simply has
/// two shape variants and both can be enumerated. This pins that the difference
/// is confined to the preprocessed-column fields and does not touch the program.
#[test]
fn global_memory_private_input_is_a_second_shape_not_a_second_program() {
    use crate::tables::page::PageConfig;
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    let elf_page = PageConfig::zero_init(PAGE_TEST_BASE);
    let mut private_page = PageConfig::zero_init(PAGE_TEST_BASE);
    private_page.is_private_input = true;

    let elf = ConstraintArtifact::capture(&crate::continuation::global_memory_air(
        &opts,
        &elf_page,
        Some([0u8; 32]),
    ));
    let private = ConstraintArtifact::capture(&crate::continuation::global_memory_air(
        &opts,
        &private_page,
        Some([0u8; 32]),
    ));

    // Same constraints, same metadata: the bus interactions depend only on the
    // page base, which is equal here.
    assert_eq!(elf.nodes, private.nodes, "the program must not vary");
    assert_eq!(elf.base_consts, private.base_consts);
    assert_eq!(elf.roots, private.roots);
    assert_eq!(elf.meta, private.meta);

    // The shape does vary, in exactly the preprocessed fields.
    assert!(elf.shape.is_preprocessed, "an ELF page is preprocessed");
    assert!(
        !private.shape.is_preprocessed,
        "a private-input page is not preprocessed — the verifier never recomputes its genesis \
         column from the ELF"
    );
    assert!(elf.shape.num_precomputed_columns > 0);
    assert_eq!(private.shape.num_precomputed_columns, 0);

    let mut normalized = private.shape.clone();
    normalized.is_preprocessed = elf.shape.is_preprocessed;
    normalized.num_precomputed_columns = elf.shape.num_precomputed_columns;
    assert_eq!(
        normalized, elf.shape,
        "the two variants must differ ONLY in the preprocessed-column fields"
    );
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
