//! The batched argue (D-BATCH B-2, `stark::multilinear_table::batched`) on the
//! tables a proof really argues: every VM table a program lights up, and the
//! W-LFM chips with their prepared opening and settled prefixes.
//!
//! Proving tests, so a box runs them:
//!
//! ```text
//! cargo test --release -p lambda-vm-prover --lib tests::batched_argue_tests -- --ignored --nocapture
//! ```
//!
//! The per-table path's entry points are followed step for step
//! (`multilinear_prove::prove_with_options_and_inputs` /
//! `verify_with_options`, `lfm::whir_proof::prove_traces_whir_opening` /
//! `verify_whir_checked`), with the argue swapped for the batched one: the
//! statement, the commitment and the openings are today's.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use math::field::element::FieldElement;
use multilinear::mle::Mle;
use multilinear::whir_chain::{ArgueFormat, ChainConfig};
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{
    self, ArgueShape, BatchedMultiProof, CommittedTable, CommittedTables, Prepared, TableLayout,
    TableStatement, argue_plan,
};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::statement;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::test_utils::{E, F};
use crate::{VmAirs, multilinear_prove};

type Tamper = fn(&mut BatchedMultiProof<F, E>);

/// One table's layout from its AIR and shape — both sides call this.
fn layout<'a>(
    air: &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
    (width, num_vars): (usize, usize),
) -> TableLayout<'a, F, E> {
    TableLayout::<F, E>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        width,
        num_vars,
        Uniforms::default(),
    )
    .expect("layout")
}

fn batched(mut config: ChainConfig) -> ChainConfig {
    config.format.argue = ArgueFormat::BATCHED;
    config
}

/// A program's every table proved under the batched argue and verified; the
/// tamper, when given, applied between. Returns the verdict and each table's
/// name.
fn vm_round_trip(
    name: &str,
    tamper: Option<Tamper>,
) -> (Result<(), multilinear::Error>, Vec<String>) {
    let elf_bytes = crate::test_utils::asm_elf_bytes(name);
    let program = Elf::load(&elf_bytes).expect("ELF");
    let result = Executor::new(&program, Vec::new())
        .expect("executor")
        .run()
        .expect("runs");
    let mut traces = Traces::from_elf_and_logs(
        &program,
        &result.logs,
        &MaxRowsConfig::default(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("traces");
    drop(result);
    let table_counts = traces.table_counts();
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let public_output = traces.public_output_bytes.clone();
    let options = ProofOptions::default_test_options();
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
    let mut pairs = airs.air_trace_pairs(&mut traces);
    let shapes: Vec<(usize, usize)> = pairs
        .iter()
        .map(|(_, trace, _)| {
            (
                trace.main_table.width,
                trace.main_table.height.trailing_zeros() as usize,
            )
        })
        .collect();
    let names: Vec<String> = pairs
        .iter()
        .map(|(air, _, _)| air.name().to_string())
        .collect();
    let table_num_vars: Vec<u8> = shapes.iter().map(|&(_, n)| n as u8).collect();
    let config = batched(multilinear_prove::chain_config(&shapes));

    let mut committed = Vec::with_capacity(pairs.len());
    for ((air, trace, _), &shape) in pairs.iter_mut().zip(&shapes) {
        let mut columns = trace.columns_main();
        committed.push(
            CommittedTable::from_layout(layout(*air, shape), |col| {
                core::mem::take(&mut columns[col as usize])
            })
            .expect("table"),
        );
    }
    drop(pairs);
    let refs = airs.air_refs();
    let plan = {
        let layouts: Vec<TableLayout<'_, F, E>> = refs
            .iter()
            .zip(&shapes)
            .map(|(air, &shape)| layout(*air, shape))
            .collect();
        let argue: Vec<ArgueShape> = layouts
            .iter()
            .map(|l| ArgueShape::of(&l.statement()))
            .collect();
        argue_plan(&argue, 27)
    };
    eprintln!(
        "batched argue {name}: {} tables, bins {:?}, n_max {}, D_max {}",
        names.len(),
        plan.bins,
        plan.num_vars,
        plan.degree
    );

    let digest = statement::elf_digest(&elf_bytes);
    let verdict = crate::with_whir_hash!(|H| {
        type Tr = DefaultTranscript<E, <H as multilinear::whir_hash::WhirHash>::Transcript>;
        let absorb = |t: &mut Tr| {
            multilinear_prove::absorb(
                t,
                &digest,
                &public_output,
                &table_counts,
                num_private_input_pages,
                &runtime_page_ranges,
                &table_num_vars,
                &config,
            )
        };
        let mut transcript = Tr::new(&[]);
        absorb(&mut transcript);
        let committed = CommittedTables::<_, _, H>::commit(committed, &config).expect("commit");
        let mut proof =
            multilinear_table::multi_prove_batched(&committed, &config, &mut transcript, None)
                .expect("the batched prover runs");
        if let Some(tamper) = tamper {
            tamper(&mut proof);
        }

        // The verifier's side: layouts, preprocessed copies and stacks rebuilt
        // from the AIRs and the shapes alone.
        let layouts: Vec<TableLayout<'_, F, E>> = refs
            .iter()
            .zip(&shapes)
            .map(|(air, &shape)| layout(*air, shape))
            .collect();
        let preprocessed: Vec<Vec<Mle<F>>> = refs
            .iter()
            .map(|air| {
                air.precomputed_columns()
                    .into_iter()
                    .map(|values| Mle::new(values).expect("column"))
                    .collect()
            })
            .collect();
        let statements: Vec<TableStatement<'_, F, E>> = layouts
            .iter()
            .zip(&preprocessed)
            .map(|(layout, cols)| layout.statement_with_preprocessed(cols))
            .collect();
        let sizes = [shapes.len()];
        let (group_layouts, domains) =
            multilinear_prove::stacks(&shapes, &sizes, &config).expect("stacks");
        let mut transcript = Tr::new(&[]);
        absorb(&mut transcript);
        let mut probe = transcript.clone();
        multilinear_table::absorb_roots::<E, _>(&mut probe, &proof.roots, &[]);
        let z: FieldElement<E> = probe.sample_field_element();
        let alpha: FieldElement<E> = probe.sample_field_element();
        let owed = crate::compute_commit_bus_offset(&public_output, 0, &z, &alpha)
            .expect("the commit offset");
        multilinear_table::multi_verify_batched::<_, _, _, H>(
            &proof,
            &statements,
            &group_layouts,
            &domains,
            &sizes,
            &owed,
            &config,
            &mut transcript,
            None,
        )
    });
    (verdict, names)
}

/// ★ Every table `test_keccak` and `all_instructions_64` light up, argued
/// batched and verified.
#[test]
#[ignore = "proves a program: a box run"]
fn every_vm_table_round_trips_batched() {
    let mut seen: Vec<String> = Vec::new();
    for name in ["test_keccak", "all_instructions_64"] {
        let (verdict, names) = vm_round_trip(name, None);
        verdict.unwrap_or_else(|e| panic!("{name}: {e:?}"));
        eprintln!("batched argue {name}: VERIFIED over {} tables", names.len());
        for n in names {
            if !seen.contains(&n) {
                seen.push(n);
            }
        }
    }
    eprintln!("batched argue: {} distinct VM tables: {seen:?}", seen.len());
}

/// The same program with one table's factor value, one bus output, or one
/// ladder half wrong: each refused.
#[test]
#[ignore = "proves a program: a box run"]
fn a_tampered_vm_proof_is_refused() {
    let tampers: [(&str, Tamper); 3] = [
        ("factor value", |p| {
            p.argue.factor_values[0][0] += FieldElement::<E>::one()
        }),
        ("bus output", |p| {
            let last = p.argue.bus_outputs.len() - 1;
            p.argue.bus_outputs[last].0 += FieldElement::<E>::one()
        }),
        ("ladder half", |p| {
            p.argue.gkr[0].layers[3].halves[0].q_hi += FieldElement::<E>::one()
        }),
    ];
    for (what, tamper) in tampers {
        let (verdict, _) = vm_round_trip("test_keccak", Some(tamper));
        assert!(verdict.is_err(), "{what}: a tampered proof verified");
        eprintln!("batched argue test_keccak: {what} tamper refused: {verdict:?}");
    }
}

/// ★ The W-LFM chips of `TrivialV0`, batched, with the prepared opening and
/// the prefixes it settles (policy B) — the openings shared with today's path.
#[test]
#[ignore = "proves a W-LFM program: a box run"]
fn the_w_lfm_chips_round_trip_batched() {
    use crate::lfm::executor::execute;
    use crate::lfm::hash::HasherKind;
    use crate::lfm::trace::build_traces_with_hasher;
    use crate::lfm::whir_proof::{
        WhirLfmHash, WhirLfmPlan, WhirLfmTranscript, WhirPrepped, absorb_whir_lfm_statement,
        airs_for, build_whir_artifacts, prep_whir_tables, prepared_check,
    };
    use crate::tables::types::FE as Fe;

    let _ungrinded = crate::lfm::whir_proof::test_grind::off();
    let options = ProofOptions::default_test_options();
    let program = crate::lfm::programs::trivial_program();
    let build = build_whir_artifacts(&program, &options, HasherKind::Rpx).expect("artifacts");
    let arenas = vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| Fe::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ];
    let exec = execute(&program, &arenas, &build.artifacts.hasher).expect("executes");
    let mut traces = build_traces_with_hasher(&program, &exec.records, build.artifacts.hasher);
    let airs = airs_for(&build.artifacts, &options);
    let WhirPrepped {
        tables,
        counts,
        config,
        settled,
    } = prep_whir_tables(&build, &airs, &mut traces, true).expect("prep");
    let config = batched(config);
    let mut transcript = WhirLfmTranscript::new(&[]);
    absorb_whir_lfm_statement(
        &mut transcript,
        &build.artifacts.program_id,
        &exec.public_words,
        &build.artifacts.table_num_vars,
        &config,
    );
    let committed = CommittedTables::<F, E, WhirLfmHash>::commit_grouped_settled(
        tables,
        &[counts.len()],
        &config,
        &settled,
    )
    .expect("commit");
    let borrowed = multilinear::stacking::borrow(&build.prepared.columns);
    let prepared = Prepared {
        commitment: &build.prepared.commitment,
        columns: &borrowed,
        at: &build.artifacts.prepared_at,
    };
    let proof = multilinear_table::multi_prove_batched(
        &committed,
        &config,
        &mut transcript,
        Some(prepared),
    )
    .expect("the batched prover runs");

    let verify = |proof: &BatchedMultiProof<F, E>| {
        let refs = airs.air_refs();
        let plan = WhirLfmPlan::build(&build.artifacts, &refs).expect("plan");
        let statements = plan.statements();
        let vconfig = batched(plan.config);
        let mut transcript = WhirLfmTranscript::new(&[]);
        absorb_whir_lfm_statement(
            &mut transcript,
            &build.artifacts.program_id,
            &exec.public_words,
            &build.artifacts.table_num_vars,
            &vconfig,
        );
        let mut probe = transcript.clone();
        multilinear_table::absorb_roots::<E, _>(
            &mut probe,
            &proof.roots,
            &build.artifacts.prepared_roots,
        );
        let z: FieldElement<E> = probe.sample_field_element();
        let alpha: FieldElement<E> = probe.sample_field_element();
        let expected = crate::lfm::proof::expected_public_balance(&exec.public_words, &z, &alpha)
            .expect("balance");
        eprintln!(
            "batched argue W-LFM: {} tables {:?}, settled {:?}",
            statements.len(),
            plan.names,
            settled
        );
        multilinear_table::multi_verify_batched_settled::<F, E, _, WhirLfmHash>(
            proof,
            &statements,
            &plan.group_layouts,
            &plan.group_domains,
            &plan.sizes(),
            &expected,
            &vconfig,
            &mut transcript,
            Some(prepared_check(&build.artifacts)),
            plan.policy.excludes_prefix(),
        )
    };
    verify(&proof).expect("the W-LFM batched proof verifies");
    let mut bad = proof.clone();
    bad.argue.factor_values[0][0] += FieldElement::<E>::one();
    assert!(verify(&bad).is_err(), "a tampered W-LFM factor value");
    eprintln!("batched argue W-LFM: VERIFIED; tamper refused");
}
