//! The block argued under the batched format (`BlockFormat::argue` =
//! `ArgueFormat::Batched`): each group's tables in one argue on the group's
//! fork — a lockstep GKR per bin, one constraint sumcheck — beside the
//! per-table default. Round trips, the format read from the verifier's side
//! only, one refusal per way a group's argue can be wrong (each new check with
//! its mutation), the default's bytes, and the card against the host.

use crate::block_whir::{
    self, BlockFormat, BlockOptions, BlockWhirProof, Deviations, prove_block_whir_with,
    verify_block_whir_with,
};
use crate::tables::MaxRowsConfig;
use crate::test_utils::{E, asm_elf_bytes};
use crate::zf_format::ZfFormat;
use math::field::element::FieldElement;
use multilinear::whir_chain::{ArgueFormat, StackVars};
use stark::multilinear_block::ArgueDeviation;
use stark::multilinear_table::batched::{ProverFaults, VerifierChecks, Where};
use stark::proof::options::ProofOptions;

/// The production format; one group holds a small program whole.
fn one_group(argue: ArgueFormat) -> BlockFormat {
    BlockFormat {
        zf: ZfFormat::DEFAULT,
        group_polys: block_whir::BLOCK_GROUP_POLYS,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue,
    }
}

/// A stack of 2^10 and two polynomials a group: a small program spans several
/// groups.
fn many_groups(argue: ArgueFormat) -> BlockFormat {
    let mut zf = ZfFormat::DEFAULT;
    zf.whir_stack = StackVars::new(10).expect("a valid stack");
    BlockFormat {
        zf,
        group_polys: 2,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue,
    }
}

const BATCHED: ArgueFormat = ArgueFormat::BATCHED;
const PER_TABLE: ArgueFormat = ArgueFormat::PerTable;

fn options(max_rows: MaxRowsConfig, keccak_rnd_rows_log2: usize) -> BlockOptions {
    BlockOptions {
        max_rows,
        keccak_rnd_rows_log2,
        ecdas_rows_log2: block_whir::BLOCK_ECDAS_ROWS_LOG2,
        keccak_rows_log2: block_whir::BLOCK_KECCAK_ROWS_LOG2,
        ecsm_rows_log2: block_whir::BLOCK_ECSM_ROWS_LOG2,
        drop_levels: multilinear::whir_commit::TreeDrop::uniform(3),
        window_log2: None,
        stream_keccak_rnd: false,
        stream_memw_lt: false,
        drop_streamed_ops: false,
        layout_workers: 0,
        layout_ahead: Some(2),
        pack_rest_as_laid_out: false,
        // Every table packed on a card, so a box run walks the narrow path at a
        // test's size.
        narrow: stark::multilinear_block::Narrowing::Card { min_cells: 0 },
        upload_ahead: true,
        memlog: false,
        // The rest laid out in waves of 1 MiB: several at a test's size.
        finish_keccak_rnd_chunks: true,
        finish_cuts: true,
        finish_stream: true,
        rest_layout_bytes: Some(1 << 20),
        pack_finished: true,
        gpack: true,
        spill: crate::block_whir::BlockSpillPolicy::Off,
    }
}

fn prove_with(
    elf: &[u8],
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
) -> BlockWhirProof {
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        options,
        deviations,
    )
    .expect("prove")
    .0
}

fn prove(elf: &[u8], format: &BlockFormat, options: &BlockOptions) -> BlockWhirProof {
    prove_with(elf, format, options, &Deviations::default())
}

fn verify_with(
    proof: &BlockWhirProof,
    elf: &[u8],
    format: &BlockFormat,
    skip_prepared: bool,
    checks: VerifierChecks,
) -> bool {
    verify_block_whir_with(
        proof,
        elf,
        &ProofOptions::default_test_options(),
        format,
        skip_prepared,
        checks,
    )
    .unwrap_or(false)
}

fn verify(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_with(proof, elf, format, false, VerifierChecks::ALL)
}

/// A proof argued with `faults` in group `group`.
fn faulted(
    elf: &[u8],
    format: &BlockFormat,
    o: &BlockOptions,
    group: usize,
    faults: ProverFaults,
) -> BlockWhirProof {
    prove_with(
        elf,
        format,
        o,
        &Deviations {
            argue: ArgueDeviation {
                group: Some(group),
                faults,
                at: Where::Device,
            },
            ..Default::default()
        },
    )
}

/// A group of `proof` and the position in it of a table shorter than the
/// group's tallest, with variables and with constraints — what a fault on the
/// padding or on `ξ`'s prefix needs: at no variables the prefix and the suffix
/// are both empty, and with no constraint `ξ` weighs nothing.
fn group_with_a_short_table(
    proof: &BlockWhirProof,
    elf: &[u8],
    format: &BlockFormat,
) -> (usize, usize) {
    let frame = block_whir::block_frame(
        proof.statement(),
        elf,
        &ProofOptions::default_test_options(),
        format,
    )
    .expect("the statement's frame");
    let airs = frame.airs.air_refs();
    let constrained = |t: usize| {
        let (width, num_vars) = frame.shapes[t];
        crate::multilinear_prove::layout_of(airs[t], width, num_vars)
            .expect("a layout")
            .statement()
            .shape
            .num_roots()
            > 0
    };
    proof
        .groups
        .iter()
        .enumerate()
        .find_map(|(g, tables)| {
            let height = |t: u32| proof.table_num_vars[t as usize];
            let tallest = tables.iter().map(|&t| height(t)).max()?;
            tables
                .iter()
                .position(|&t| height(t) > 0 && height(t) < tallest && constrained(t as usize))
                .map(|position| (g, position))
        })
        .expect("a group with a short constrained table")
}

/// ★ The block round-trips under the batched argue at one group and at many,
/// carrying one argue a group and no per-table argument. The format is the
/// verifier's: a batched proof under the per-table format is refused, and the
/// reverse.
#[test]
fn a_block_argued_batched_proves_and_verifies() {
    let elf = asm_elf_bytes("sub");
    let proof = prove(
        &elf,
        &one_group(BATCHED),
        &options(MaxRowsConfig::default(), 16),
    );
    assert_eq!(proof.argues.len(), 1);
    assert!(proof.proof.tables.is_empty());
    assert!(verify(&proof, &elf, &one_group(BATCHED)));
    assert!(!verify(&proof, &elf, &one_group(PER_TABLE)));

    let elf = asm_elf_bytes("all_instructions_64");
    let o = options(MaxRowsConfig::small(), 16);
    let proof = prove(&elf, &many_groups(BATCHED), &o);
    assert!(proof.groups.len() >= 3, "{} groups", proof.groups.len());
    assert_eq!(proof.argues.len(), proof.groups.len());
    assert!(verify(&proof, &elf, &many_groups(BATCHED)));
    assert!(!verify(&proof, &elf, &many_groups(PER_TABLE)));
    // A cap the verifier does not hold bins the trees otherwise.
    let other_cap = BlockFormat {
        argue: ArgueFormat::Batched { bin_log_cells: 0 },
        ..many_groups(BATCHED)
    };
    assert!(!verify(&proof, &elf, &other_cap));

    let per_table = prove(&elf, &many_groups(PER_TABLE), &o);
    assert!(per_table.argues.is_empty());
    assert!(verify(&per_table, &elf, &many_groups(PER_TABLE)));
    assert!(!verify(&per_table, &elf, &many_groups(BATCHED)));
}

/// KECCAK_RND cut into many chunks, batched: many trees a bin. No VM table
/// reads a shifted row, so no table keeps a claim reduction: each reads its
/// columns at the sumcheck's point directly.
#[test]
fn keccak_chunks_argue_batched() {
    let elf = asm_elf_bytes("test_keccak");
    let format = many_groups(BATCHED);
    let proof = prove(&elf, &format, &options(MaxRowsConfig::small(), 3));
    assert!(proof.table_counts.keccak_rnd >= 2);
    let tables: usize = proof.argues.iter().map(|a| a.bus_outputs.len()).sum();
    assert_eq!(tables, proof.table_num_vars.len());
    assert!(
        proof
            .argues
            .iter()
            .flat_map(|a| &a.reduces)
            .all(Option::is_none),
        "a VM table kept a reduction"
    );
    assert!(verify(&proof, &elf, &format));
}

/// The prepared openings settle under the batched argue as under the
/// per-table one: the dense fixture verifies, two groups' openings swapped are
/// refused, and with the openings unchecked (the mutation) the swap verifies.
#[test]
fn prepared_openings_settle_under_the_batched_argue() {
    let elf = asm_elf_bytes("test_dense_pages");
    let o = options(MaxRowsConfig::default(), 16);
    let proof = prove(&elf, &one_group(BATCHED), &o);
    assert!(!proof.prepared.is_empty());
    assert!(verify(&proof, &elf, &one_group(BATCHED)));

    let format = many_groups(BATCHED);
    let mut proof = prove(&elf, &format, &o);
    assert!(proof.prepared.len() >= 2, "{} stacks", proof.prepared.len());
    assert!(verify(&proof, &elf, &format));
    proof.prepared.swap(0, 1);
    assert!(!verify(&proof, &elf, &format));
    assert!(verify_with(
        &proof,
        &elf,
        &format,
        true,
        VerifierChecks::ALL
    ));
}

/// ★ A group argued wrongly is refused: a cross-table weight off by `λ`, a
/// table claim absorbed off by one, `ξ`'s suffix for a short table, a short
/// table padded the `2^Δ` way, and the trees binned under another cap.
#[test]
fn a_batched_group_argued_wrongly_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups(BATCHED);
    let o = options(MaxRowsConfig::small(), 16);
    let honest = prove(&elf, &format, &o);
    assert!(verify(&honest, &elf, &format));
    let (g, short) = group_with_a_short_table(&honest, &elf, &format);
    let faults = [
        (
            "weight",
            ProverFaults {
                wrong_weight: Some(0),
                ..Default::default()
            },
        ),
        (
            "claim",
            ProverFaults {
                tamper_claim: Some(short),
                ..Default::default()
            },
        ),
        (
            "suffix of xi",
            ProverFaults {
                suffix_xi: Some(short),
                ..Default::default()
            },
        ),
        (
            "scaled padding",
            ProverFaults {
                scaled_padding: Some(short),
                ..Default::default()
            },
        ),
        (
            "bins",
            ProverFaults {
                bin_log_cells: Some(0),
                ..Default::default()
            },
        ),
    ];
    for (name, fault) in faults {
        let proof = faulted(&elf, &format, &o, g, fault);
        assert!(!verify(&proof, &elf, &format), "{name} in group {g}");
    }
}

/// A batched argue tampered in the proof is refused: a bus output moved, one
/// missing, a factor value moved, two groups' argues swapped, and a reduction
/// where the format has none. The last is admitted with the reduction's shape
/// unchecked (the mutation).
#[test]
fn a_tampered_batched_argue_is_refused() {
    let elf = asm_elf_bytes("test_keccak");
    let format = many_groups(BATCHED);
    let honest = prove(&elf, &format, &options(MaxRowsConfig::small(), 3));
    assert!(verify(&honest, &elf, &format));
    let tampered = |tamper: &dyn Fn(&mut BlockWhirProof)| {
        let mut proof = honest.clone();
        tamper(&mut proof);
        proof
    };
    let one = FieldElement::<E>::one();
    assert!(!verify(
        &tampered(&|p| p.argues[0].bus_outputs[0].0 += one),
        &elf,
        &format
    ));
    assert!(!verify(
        &tampered(&|p| {
            p.argues[0].bus_outputs.pop();
        }),
        &elf,
        &format
    ));
    assert!(!verify(
        &tampered(&|p| p.argues[1].factor_values[0][0] += one),
        &elf,
        &format
    ));
    assert!(!verify(&tampered(&|p| p.argues.swap(0, 1)), &elf, &format));
    // A reduction on a table that reads no shifted row (no VM table does).
    let extra = tampered(&|p| {
        p.argues[0].reduces[0] = Some(multilinear::claim_reduce::ReduceProof {
            sumcheck: multilinear::sumcheck::SumcheckProof { rounds: Vec::new() },
            column_values: Vec::new(),
        });
    });
    assert!(!verify(&extra, &elf, &format));
    let inert = VerifierChecks {
        reduce_shape: false,
        ..VerifierChecks::ALL
    };
    assert!(verify_with(&extra, &elf, &format, false, inert));
}

/// A batched group argued on another group's fork is refused: the index each
/// fork absorbs ties the argue to its group.
#[test]
fn a_batched_group_on_another_fork_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups(BATCHED);
    let swapped = prove_with(
        &elf,
        &format,
        &options(MaxRowsConfig::small(), 16),
        &Deviations {
            fork_of: Some(|g| g ^ 1),
            ..Default::default()
        },
    );
    assert!(swapped.groups.len() >= 2);
    assert!(!verify(&swapped, &elf, &format));
}

/// ★ The bus balance, summed over every group's batched outputs, binds the
/// block: a prover leaving one KECCAK_RND chunk out — every table it proves
/// consistent — is refused, and admitted with the balance unchecked (the
/// mutation): the block-wide sum is what refuses it.
#[test]
fn the_bus_balance_binds_the_batched_block() {
    let elf = asm_elf_bytes("test_keccak");
    let format = many_groups(BATCHED);
    let o = options(MaxRowsConfig::small(), 3);
    let honest = prove(&elf, &format, &o);
    let short = prove_with(
        &elf,
        &format,
        &o,
        &Deviations {
            omit_first_keccak_rnd: true,
            ..Default::default()
        },
    );
    assert_eq!(
        short.table_counts.keccak_rnd + 1,
        honest.table_counts.keccak_rnd
    );
    assert!(!verify(&short, &elf, &format));
    let inert = VerifierChecks {
        balance: false,
        ..VerifierChecks::ALL
    };
    assert!(verify_with(&short, &elf, &format, false, inert));
}

/// A program's traces, built once, KECCAK_RND split at 2^`keccak_rnd_log2`.
fn fixed_traces(
    name: &str,
    max_rows: &MaxRowsConfig,
    keccak_rnd_log2: usize,
) -> (
    executor::elf::Elf,
    Vec<u8>,
    crate::tables::trace_builder::Traces,
) {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let elf = asm_elf_bytes(name);
    let program = Elf::load(&elf).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    let mut traces = crate::tables::trace_builder::Traces::from_elf_and_logs(
        &program,
        &logs,
        max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("the traces build");
    block_whir::split_keccak_rnd(&mut traces, keccak_rnd_log2);
    (program, elf, traces)
}

/// `traces` with every table that has no preprocessed column sorted row by
/// row: the same multiset of rows, in an order no run can change. A trace
/// build is not reproducible between processes (some tables come out of a
/// `HashMap` in its order), so a proof's bytes can only be compared across
/// builds over canonical rows. No VM table reads another row and every bus is a
/// multiset, so the sorted traces are still a block's, and they verify.
fn canonical_rows(program: &executor::elf::Elf, traces: &mut crate::tables::trace_builder::Traces) {
    let counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        program,
        &ProofOptions::default_test_options(),
        false,
        &traces.page_configs,
        &counts,
        None,
        true,
        None,
        None,
        None,
    );
    for (air, trace, _) in airs.air_trace_pairs(traces) {
        if air.num_precomputed_columns() > 0 {
            continue;
        }
        let columns = trace.columns_main();
        let height = columns.first().map_or(0, Vec::len);
        let mut rows: Vec<Vec<u64>> = (0..height)
            .map(|r| columns.iter().map(|c| c[r].canonical()).collect())
            .collect();
        rows.sort_unstable();
        let sorted: Vec<Vec<FieldElement<crate::test_utils::F>>> = (0..columns.len())
            .map(|c| rows.iter().map(|row| FieldElement::from(row[c])).collect())
            .collect();
        trace.main_table = stark::table::Table::from_columns(sorted);
    }
}

/// The proof of fixed traces, non-streamed, under `deviations`.
fn prove_fixed(
    program: &executor::elf::Elf,
    elf: &[u8],
    traces: &mut crate::tables::trace_builder::Traces,
    format: &BlockFormat,
    o: &BlockOptions,
    deviations: &Deviations,
) -> BlockWhirProof {
    block_whir::prove_traces(
        program,
        elf,
        traces,
        &ProofOptions::default_test_options(),
        format,
        o,
        deviations,
        false,
        &|_, _| {},
        &mut Default::default(),
    )
    .expect("prove")
}

/// A digest of what the default format's proof carries: its multilinear
/// proof, its prepared openings and its partition, each serialized.
fn default_digest(proof: &BlockWhirProof) -> String {
    let mut bytes = Vec::new();
    for part in [
        rkyv::to_bytes::<rkyv::rancor::Error>(&proof.proof)
            .expect("serialize")
            .to_vec(),
        rkyv::to_bytes::<rkyv::rancor::Error>(&proof.prepared)
            .expect("serialize")
            .to_vec(),
        rkyv::to_bytes::<rkyv::rancor::Error>(&proof.groups)
            .expect("serialize")
            .to_vec(),
    ] {
        bytes.extend_from_slice(&(part.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&part);
    }
    crate::statement::elf_digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// ★ The per-table default keeps today's bytes: the default format's proof of
/// fixed traces over canonical rows ([`canonical_rows`]) — its multilinear
/// proof, its prepared openings and its partition — digests to what the block
/// prover before the batched format proves over the same rows, with a
/// deterministic grind. The batched argue's only mark on a default proof is
/// [`BlockWhirProof::argues`], empty.
///
/// The digests depend on the ELF bytes (the statement absorbs the program's
/// digest), and the guests' bytes differ between toolchains, so the expected
/// digests are not constants here: `BLOCK_DEFAULT_DIGESTS` carries them
/// (`name=hex;name=hex;…`), proved on the same machine and ELFs by the same
/// body at the base sha (the box gate does both). A record: at noepoch/whir @
/// 11e2de53d, on the laptop's ELFs (all_instructions_64 md5 9ded1481…,
/// test_dense_pages b753fbd8…, sub ceefc702…), the host proves 6e0bd41e…,
/// 3b7726b1…, ca028342…, the same in two runs, and this branch proves the same.
///
/// Run alone, with the grind deterministic (it is read once per process):
///
/// ```text
/// LAMBDA_VM_DETERMINISTIC_GRIND=1 BLOCK_DEFAULT_DIGESTS='all_instructions_64=…;test_dense_pages=…;sub=…' \
///     cargo test --release -p lambda-vm-prover --lib -- \
///     tests::block_batched_tests::the_default_format_keeps_todays_bytes --exact --ignored --nocapture
/// ```
#[test]
#[ignore = "needs LAMBDA_VM_DETERMINISTIC_GRIND=1 and BLOCK_DEFAULT_DIGESTS, run alone"]
fn the_default_format_keeps_todays_bytes() {
    assert!(
        crypto::grinding::deterministic(),
        "export LAMBDA_VM_DETERMINISTIC_GRIND=1"
    );
    let given = std::env::var("BLOCK_DEFAULT_DIGESTS")
        .expect("BLOCK_DEFAULT_DIGESTS: the base sha's digests, name=hex;…");
    let expected = |name: &str| -> String {
        given
            .split(';')
            .filter_map(|pair| pair.trim().split_once('='))
            .find(|(n, _)| *n == name)
            .map(|(_, hex)| hex.to_string())
            .unwrap_or_else(|| panic!("BLOCK_DEFAULT_DIGESTS has no digest for {name}"))
    };
    let cases = [
        (
            "all_instructions_64",
            many_groups(PER_TABLE),
            MaxRowsConfig::small(),
        ),
        (
            "test_dense_pages",
            one_group(PER_TABLE),
            MaxRowsConfig::default(),
        ),
        ("sub", many_groups(PER_TABLE), MaxRowsConfig::small()),
    ];
    for (name, format, max_rows) in cases {
        let (program, elf, mut traces) = fixed_traces(name, &max_rows, 16);
        canonical_rows(&program, &mut traces);
        let o = options(max_rows, 16);
        let proof = prove_fixed(
            &program,
            &elf,
            &mut traces,
            &format,
            &o,
            &Deviations::default(),
        );
        assert!(verify(&proof, &elf, &format), "{name}");
        assert!(proof.argues.is_empty(), "{name}");
        let digest = default_digest(&proof);
        println!("BLOCK DEFAULT DIGEST {name}: {digest}");
        assert_eq!(
            digest,
            expected(name),
            "{name}: the default format's bytes moved"
        );
    }
}

/// ★ N-2's card gate (box only): every group's batched argue on the card
/// proves the host reference's bytes, over one commitment — the same fixed
/// traces proved twice, non-streamed, with a deterministic grind, so the whole
/// proof is compared — and both verify. On a device the fused sessions, the
/// device rounds and the ladder's Gruen layers are counted, so the comparison
/// is not of the host with itself.
///
/// ```text
/// LAMBDA_VM_DETERMINISTIC_GRIND=1 cargo test --release -p lambda-vm-prover --features cuda --lib -- \
///     tests::block_batched_tests::the_card_argues_the_hosts_bytes --exact --ignored --nocapture
/// ```
#[test]
#[ignore = "box only: the card against the host"]
fn the_card_argues_the_hosts_bytes() {
    assert!(
        crypto::grinding::deterministic(),
        "export LAMBDA_VM_DETERMINISTIC_GRIND=1"
    );
    let device = multilinear::gpu::reserve_budget() > 0;
    // Tall enough for the card (trees of 2^14 input cells and up, fused rounds
    // past 2^7) and, under a stack of 2^18 and two polynomials a group, spread
    // over several groups.
    let mut zf = ZfFormat::DEFAULT;
    zf.whir_stack = StackVars::new(18).expect("a valid stack");
    let format = BlockFormat {
        zf,
        group_polys: 2,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue: BATCHED,
    };
    for name in ["all_instructions_64", "test_keccak", "test_ecsm"] {
        let (program, elf, mut traces) = fixed_traces(name, &MaxRowsConfig::default(), 16);
        let o = options(MaxRowsConfig::default(), 16);
        let mut run = |at: Where| {
            let fused = multilinear::gpu_fused::fused_sessions();
            let rounds = multilinear::gpu::sumcheck_rounds();
            let gruen = multilinear::gkr_gruen::gruen_layers();
            let proof = prove_fixed(
                &program,
                &elf,
                &mut traces,
                &format,
                &o,
                &Deviations {
                    argue: ArgueDeviation {
                        at,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            );
            assert!(verify(&proof, &elf, &format), "{name} {at:?}");
            (
                rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
                    .expect("serialize")
                    .to_vec(),
                proof.groups.len(),
                multilinear::gpu_fused::fused_sessions() - fused,
                multilinear::gpu::sumcheck_rounds() - rounds,
                multilinear::gkr_gruen::gruen_layers() - gruen,
            )
        };
        let host = run(Where::Host);
        let card = run(Where::Device);
        println!(
            "BLOCK BATCHED CARD {name}: groups {} · device {device} · fused sessions {} · device rounds {} · Gruen ladder layers {} · {} bytes · {}",
            card.1,
            card.2,
            card.3,
            card.4,
            card.0.len(),
            if host.0 == card.0 { "EQUAL" } else { "DIFFER" }
        );
        assert_eq!(host.2, 0, "{name}: the host reference ran no fused session");
        if device {
            assert!(card.2 > 0, "{name}: the card ran fused sessions");
            assert!(card.3 > 0, "{name}: the card ran rounds");
            assert!(card.4 > 0, "{name}: the ladder ran Gruen's rounds");
        }
        assert!(host.0 == card.0, "{name}: the card's bytes differ");
    }
}
