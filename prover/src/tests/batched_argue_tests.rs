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
use stark::multilinear_table::batched::{self, Where};
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
    let (verdict, names, _) = vm_round_trip_elf(
        name,
        &crate::test_utils::asm_elf_bytes(name),
        tamper,
        Where::Device,
    );
    (verdict, names)
}

/// A Rust guest's ELF, built by `make compile-programs-rust`.
fn rust_elf_bytes(name: &str) -> Vec<u8> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    std::fs::read(root.join(format!("executor/program_artifacts/rust/{name}.elf")))
        .unwrap_or_else(|e| panic!("{name}.elf — run `make compile-programs-rust`: {e}"))
}

/// [`vm_round_trip`] over an ELF in hand.
///
/// `at` is where the prover runs: [`Where::Device`] is what
/// `multi_prove_batched` does (the card where it takes the work), and
/// [`Where::Host`] the host reference. The proof's canonical bytes come back
/// too, so the two can be compared.
fn vm_round_trip_elf(
    name: &str,
    elf_bytes: &[u8],
    tamper: Option<Tamper>,
    at: Where,
) -> (Result<(), multilinear::Error>, Vec<String>, Vec<u8>) {
    let elf_bytes = elf_bytes.to_vec();
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
        let mut proof = batched::prove_batched_on(
            &committed,
            &config,
            &mut transcript,
            None,
            batched::ProverFaults::default(),
            at,
        )
        .expect("the batched prover runs");
        // serde writes each element canonically, so equal bytes are equal
        // values. The roots and the argue only: the openings grind, and a
        // parallel nonce search returns ANY valid nonce, so two honest proofs
        // part there (memory: grinding-nonce-nondeterminism) — after the argue.
        let bytes = serde_json::to_vec(&(&proof.roots, &proof.argue)).expect("canonical bytes");
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
        let verdict = multilinear_table::multi_verify_batched::<_, _, _, H>(
            &proof,
            &statements,
            &group_layouts,
            &domains,
            &sizes,
            &owed,
            &config,
            &mut transcript,
            None,
        );
        (verdict, bytes)
    });
    let (verdict, bytes) = verdict;
    (verdict, names, bytes)
}

/// The VM's AIR labels, as `lean_program_census::vm_airs` names them: what a
/// table's name reduces to (`CPU[0]` → `CPU`, `PAGE:0x…` → `PAGE`).
fn label(name: &str) -> String {
    name.split(['[', ':']).next().unwrap_or(name).to_string()
}

/// ★ Every VM table, argued batched and verified: the monolithic proofs of
/// programs that light them up between them, then every epoch of two
/// continuations (the epoch's L2G bookend, DECODE's prepared opening, two
/// commitment groups). The labels seen must cover every AIR of
/// `lean_program_census::vm_airs`.
#[test]
#[ignore = "proves programs: a box run"]
fn every_vm_table_round_trips_batched() {
    let mut seen: Vec<String> = Vec::new();
    let mut note = |names: Vec<String>| {
        for n in names {
            let l = label(&n);
            if !seen.contains(&l) {
                seen.push(l);
            }
        }
    };
    for name in [
        "test_keccak",
        "all_instructions_64",
        "test_ecsm",
        "test_lb_lh_8",
        "misalign_lw",
        "all_loadstore_32",
    ] {
        let (verdict, names) = vm_round_trip(name, None);
        verdict.unwrap_or_else(|e| panic!("{name}: {e:?}"));
        eprintln!("batched argue {name}: VERIFIED over {} tables", names.len());
        note(names);
    }
    let (verdict, names, _) =
        vm_round_trip_elf("hint_min", &rust_elf_bytes("hint_min"), None, Where::Device);
    verdict.unwrap_or_else(|e| panic!("hint_min: {e:?}"));
    eprintln!(
        "batched argue hint_min: VERIFIED over {} tables",
        names.len()
    );
    note(names);
    for (name, log) in [("sub", 2u32), ("test_ecsm", 10)] {
        let (epochs, names) = epochs_round_trip(name, log);
        eprintln!("batched argue {name} at 2^{log}: {epochs} epochs VERIFIED");
        note(names);
    }
    let want: Vec<String> = crate::tests::lean_program_census::vm_airs()
        .into_iter()
        .map(|(_, l)| l.to_string())
        .collect();
    let missing: Vec<&String> = want.iter().filter(|l| !seen.contains(l)).collect();
    eprintln!(
        "batched argue: {} distinct VM labels {seen:?}; census wants {}, missing {missing:?}",
        seen.len(),
        want.len()
    );
    assert!(
        missing.is_empty(),
        "VM tables never argued batched: {missing:?}"
    );
}

/// Every epoch of `name` at `2^log`, batched: `prove_epoch`'s statement,
/// tables, groups and DECODE opening, with the argue swapped, and
/// `verify_epoch_bookend`'s side. Returns the epoch count and every table name.
fn epochs_round_trip(name: &str, log: u32) -> (usize, Vec<String>) {
    use crate::continuation::{self, PreparedEpoch};
    use crate::multilinear_continuation::{
        absorb_epoch, decode_prepared_for, decode_table_index, epoch_groups, owed,
    };
    use crate::tables::local_to_global;
    use crate::tables::register;
    use crate::tables::trace_builder::DecodeArtifacts;

    let elf_bytes = crate::test_utils::asm_elf_bytes(name);
    let elf = Elf::load(&elf_bytes).expect("load");
    let opts = ProofOptions::default_test_options();
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let digest = statement::elf_digest(&elf_bytes);
    let mut carried = register::register_init_from_entry_point(elf.entry_point);
    let mut count = 0usize;
    let mut names: Vec<String> = Vec::new();
    crate::with_whir_hash!(|H| {
        type Tr = DefaultTranscript<E, <H as multilinear::whir_hash::WhirHash>::Transcript>;
        let pinned = decode_prepared_for::<H>(&elf, &elf_bytes).expect("DECODE's commitment");
        continuation::for_each_epoch(&elf, &[], log, &artifacts, |prepared, _| {
            let PreparedEpoch {
                register_init,
                label,
                mut traces,
                boundary,
                is_final,
                ..
            } = prepared;
            assert_eq!(register_init, carried, "epoch {label}: the register chain");
            crate::tables::bitwise::update_multiplicities(
                &mut traces.bitwise,
                &local_to_global::collect_bitwise_from_l2g(&boundary),
            );
            let reg_fini = register::fini_from_trace(&traces.register);
            let table_counts = traces.table_counts();
            let public_output = traces.public_output_bytes.clone();
            let build_airs = || {
                (
                    continuation::build_epoch_airs(
                        &elf,
                        &opts,
                        &[],
                        &table_counts,
                        &register_init,
                        &reg_fini,
                        is_final,
                        None,
                    ),
                    continuation::l2g_memory_air(&opts, label),
                )
            };
            let (airs, l2g_air) = build_airs();
            let mut l2g_trace = local_to_global::generate_local_to_global_trace(&boundary);
            let mut pairs = airs.air_trace_pairs(&mut traces);
            pairs.push((&l2g_air, &mut l2g_trace, &()));
            let air_list: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
                pairs.iter().map(|(air, _, _)| *air).collect();
            let decode_at = decode_table_index(&air_list).expect("DECODE");
            let shapes: Vec<(usize, usize)> = pairs
                .iter()
                .map(|(_, trace, _)| {
                    (
                        trace.main_table.width,
                        trace.main_table.height.trailing_zeros() as usize,
                    )
                })
                .collect();
            // The bookend by its census label: continuation AIRs carry no name
            // of their own (they print as `unknown`), and it is pushed last.
            names.extend(
                pairs[..pairs.len() - 1]
                    .iter()
                    .map(|(air, _, _)| air.name().to_string()),
            );
            names.push("L2G".to_string());
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
            let sizes = epoch_groups(committed.len());
            pinned
                .agrees_with(&config)
                .expect("DECODE's commitment agrees");
            let absorb = |t: &mut Tr| {
                absorb_epoch(
                    t,
                    &digest,
                    &public_output,
                    &table_counts,
                    label,
                    &table_num_vars,
                    &config,
                )
            };
            let mut transcript = Tr::new(&[]);
            absorb(&mut transcript);
            let committed = CommittedTables::<_, _, H>::commit_grouped(committed, &sizes, &config)
                .expect("commit");
            let borrowed = multilinear::stacking::borrow(&pinned.columns);
            let decode_columns = pinned.settled_at(decode_at);
            let proof = multilinear_table::multi_prove_batched(
                &committed,
                &config,
                &mut transcript,
                Some(pinned.opening(&borrowed, &decode_columns)),
            )
            .expect("the batched prover runs");

            // The verifier's side, from AIRs of its own.
            let (vairs, vl2g) = build_airs();
            let mut refs = vairs.air_refs();
            refs.push(&vl2g);
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
            let (group_layouts, domains) =
                multilinear_prove::stacks(&shapes, &sizes, &config).expect("stacks");
            let mut transcript = Tr::new(&[]);
            absorb(&mut transcript);
            let check = pinned.check(&decode_columns);
            let owed = owed(
                &public_output,
                &register_init,
                &proof.roots,
                check.roots,
                &transcript,
            )
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
                Some(check),
            )
            .unwrap_or_else(|e| panic!("{name} epoch {label}: {e:?}"));
            carried = reg_fini;
            count += 1;
            Ok(())
        })
    })
    .expect("the epochs prepare");
    assert!(count > 0, "{name}: no epoch");
    (count, names)
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

/// ★ D-BATCH B-3 on real tables: the prover on the card proves the host
/// reference's canonical bytes — the roots and the whole argue — for every
/// table of `test_keccak`, `test_ecsm` and `all_instructions_64`, and both
/// verify. The openings are not compared: they grind under the production
/// config, and the parallel nonce search makes any two runs part there.
///
/// On a cuda build the device path runs the ladder's layers and the fused
/// zerocheck on the card; on a host build it is the device path's fallbacks
/// against the reference.
#[test]
#[ignore = "proves programs: a box run"]
fn the_card_proves_the_host_references_bytes_on_real_tables() {
    for name in ["test_keccak", "test_ecsm", "all_instructions_64"] {
        let elf = crate::test_utils::asm_elf_bytes(name);
        let fused = multilinear::gpu_fused::fused_sessions();
        let (card, names, card_bytes) = vm_round_trip_elf(name, &elf, None, Where::Device);
        let fused = multilinear::gpu_fused::fused_sessions() - fused;
        let (host, _, host_bytes) = vm_round_trip_elf(name, &elf, None, Where::Host);
        card.unwrap_or_else(|e| panic!("{name} on the card: {e:?}"));
        host.unwrap_or_else(|e| panic!("{name} on the host: {e:?}"));
        assert!(
            card_bytes == host_bytes,
            "{name}: the card's argue is not the host reference's bytes"
        );
        eprintln!(
            "batched argue B-3 {name}: {} tables, {fused} fused sessions on the card, {} argue bytes == the host's",
            names.len(),
            card_bytes.len()
        );
    }
}
