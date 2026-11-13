//! The lean BLAKE3 tables ([`super::super::blake3_lean`]) proved by the base's
//! own block prover ([`stark::multilinear_block`]), and timed on the card
//! beside the STARK prove of the same tables.
//!
//! Test-only, a child of the lean probe so it reads the probe's traces, AIRs
//! and timers as they are. The question it answers: what does one group of
//! lean BLAKE3 rows cost the block prover — commit, batched argue, opening —
//! against the per-table STARK the probe measured.
//!
//! # What is proved
//!
//! The probe's tables in its order (round chunks, head, byte table, memory
//! mirror), each a table of the block, cut into groups by
//! [`multilinear_block::block_groups`] at the base's
//! [`crate::block_whir::BLOCK_GROUP_POLYS`], under the base's pinned format
//! ([`BlockFormat::pinned`]: blowup 2, the batched argue, one chain a run of a
//! group's polynomials) and the base's commitment hash
//! ([`crate::hash_pin::BlockWhirHash`]). Phase A is
//! [`BlockCommitted::commit`] with the production block options' tree drop,
//! narrowing and upload-ahead; phase B is
//! [`multilinear_block::block_prove_in_order`] in group order, under the
//! phase-B overlap a run names.
//!
//! # What binds the preprocessed columns
//!
//! The verifier evaluates them ([`TableLayout::statement_with_preprocessed`]):
//! no prepared stack is committed or opened. The verifier's copies are taken
//! from the probe's traces, as the STARK probe takes its preprocessed roots
//! from them; a recursion that ships this would derive them from the program.
//! So the prover's time here carries no prepared opening.
//!
//! # The statement
//!
//! A tag and every table's shape, absorbed before any root, on both sides.
//! The memory mirror closes `LfmMem` on its own, so the tables owe a zero
//! balance.

use multilinear::mle::Mle;
use multilinear::whir_chain::ChainConfig;
use multilinear::whir_hash::WhirHash;
use stark::multilinear_block::{self, BlockCommitted, BlockProved, GroupStamps, PhaseBOverlap};
use stark::multilinear_table::batched::VerifierChecks;
use stark::multilinear_table::{CommittedTable, TableLayout, TableStatement};

use crate::block_whir::{BLOCK_GROUP_POLYS, BlockFormat, BlockOptions};
use crate::multilinear_prove::{layout_of, stacks};

use super::*;

type H = crate::hash_pin::BlockWhirHash;
type Sponge = <H as WhirHash>::Sponge;

const STATEMENT_TAG: &[u8] = b"LFM_B3R_BLOCK_PROBE_V1";

/// What both sides derive from the tables' shapes: never read from a proof.
struct Frame {
    /// `(main width, height in variables)` a table, in table order.
    shapes: Vec<(usize, usize)>,
    /// Tables a group.
    sizes: Vec<usize>,
    config: ChainConfig,
    /// The verifier's copy of each table's preprocessed columns.
    prep: Vec<Vec<Mle<F>>>,
}

/// Every table's trace, in [`Airs::refs`] order.
fn table_traces(t: &Traces) -> Vec<&TraceTable<F, E>> {
    t.rounds
        .iter()
        .chain([&t.head, &t.table, &t.mirror])
        .collect()
}

fn frame(airs: &Airs, t: &Traces) -> Result<Frame, String> {
    let refs = airs.refs();
    let traces = table_traces(t);
    if refs.len() != traces.len() {
        return Err(format!("{} AIRs for {} traces", refs.len(), traces.len()));
    }
    let mut shapes = Vec::with_capacity(refs.len());
    let mut prep = Vec::with_capacity(refs.len());
    for (air, trace) in refs.iter().zip(&traces) {
        let width = air.trace_layout().0;
        let rows = trace.num_rows();
        if trace.main_table.width != width || !rows.is_power_of_two() {
            return Err(format!(
                "{}: a trace of {} columns and {rows} rows for a {width}-column AIR",
                air.name(),
                trace.main_table.width
            ));
        }
        shapes.push((width, rows.trailing_zeros() as usize));
        let count = if air.is_preprocessed() {
            air.num_precomputed_columns()
        } else {
            0
        };
        prep.push(
            prefix_columns(trace, count)
                .into_iter()
                .map(|c| Mle::new(c).map_err(|e| format!("{}: {e:?}", air.name())))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    let config = BlockFormat::pinned().chain_config(&shapes);
    let sizes = multilinear_block::block_groups(&shapes, config.format.stack, BLOCK_GROUP_POLYS)
        .map_err(|e| format!("{e:?}"))?;
    Ok(Frame {
        shapes,
        sizes,
        config,
        prep,
    })
}

fn absorb_statement(transcript: &mut Sponge, shapes: &[(usize, usize)]) {
    transcript.append_bytes(STATEMENT_TAG);
    transcript.append_bytes(&(shapes.len() as u64).to_le_bytes());
    for &(width, num_vars) in shapes {
        transcript.append_bytes(&(width as u64).to_le_bytes());
        transcript.append_bytes(&(num_vars as u64).to_le_bytes());
    }
}

/// The block's tables, their columns copied out of the traces.
fn committed_tables<'a>(
    airs: &'a Airs,
    t: &Traces,
    frame: &Frame,
) -> Result<Vec<CommittedTable<'a, F, E>>, String> {
    let refs = airs.refs();
    let traces = table_traces(t);
    let made: Vec<Result<CommittedTable<'a, F, E>, String>> = {
        let one =
            |(air, trace, &(width, num_vars)): (DynAir<'a>, &TraceTable<F, E>, &(usize, usize))| {
                let mut columns = trace.columns_main();
                let layout = layout_of(air, width, num_vars).map_err(|e| format!("{e:?}"))?;
                CommittedTable::from_layout(layout, |col| {
                    core::mem::take(&mut columns[col as usize])
                })
                .map_err(|e| format!("{}: {e:?}", air.name()))
            };
        let items: Vec<_> = refs
            .into_iter()
            .zip(traces)
            .zip(&frame.shapes)
            .map(|((a, t), s)| (a, t, s))
            .collect();
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            items.into_par_iter().map(one).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            items.into_iter().map(one).collect()
        }
    };
    made.into_iter().collect()
}

/// Seconds of a block prove, by phase.
struct BlockSecs {
    commit: f64,
    phase_b: f64,
}

/// Phase A then phase B, the tables moved in.
fn prove_block<'a>(
    tables: Vec<CommittedTable<'a, F, E>>,
    frame: &Frame,
    overlap: PhaseBOverlap,
) -> Result<(BlockProved<F, E>, BlockSecs), String> {
    let options = BlockOptions::production();
    let mut transcript = <H as WhirHash>::sponge();
    absorb_statement(&mut transcript, &frame.shapes);
    let t = std::time::Instant::now();
    let block = BlockCommitted::commit::<H, Sponge>(
        tables,
        &frame.sizes,
        &frame.config,
        options.drop_levels,
        options.narrow,
        options.upload_ahead,
    )
    .map_err(|e| format!("commit: {e:?}"))?;
    let commit = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let order: Vec<usize> = (0..frame.sizes.len()).collect();
    let proved = multilinear_block::block_prove_in_order::<F, E, Sponge, H>(
        block.with_phase_b_overlap(overlap),
        &frame.config,
        &mut transcript,
        &[],
        &[],
        &|g| g,
        &multilinear_block::ArgueDeviation::default(),
        &|_| {},
        &order,
        None,
    )
    .map_err(|e| format!("prove: {e:?}"))?;
    let phase_b = t.elapsed().as_secs_f64();
    Ok((proved, BlockSecs { commit, phase_b }))
}

/// The host's verdict on a block proof, against the frame alone.
fn verify_block(airs: &Airs, frame: &Frame, proved: &BlockProved<F, E>) -> Result<(), String> {
    let (proof, argues, prepared, _, commit_ood) = proved;
    let refs = airs.refs();
    let layouts: Vec<TableLayout<'_, F, E>> = refs
        .iter()
        .zip(&frame.shapes)
        .map(|(air, &(width, num_vars))| layout_of(*air, width, num_vars))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{e:?}"))?;
    let statements: Vec<TableStatement<'_, F, E>> = layouts
        .iter()
        .zip(&frame.prep)
        .map(|(layout, prep)| layout.statement_with_preprocessed(prep))
        .collect();
    let (stacked, domains) =
        stacks(&frame.shapes, &frame.sizes, &frame.config).map_err(|e| format!("{e:?}"))?;
    let mut transcript = <H as WhirHash>::sponge();
    absorb_statement(&mut transcript, &frame.shapes);
    multilinear_block::block_verify_with::<F, E, Sponge, H>(
        proof,
        commit_ood,
        argues,
        prepared,
        &[],
        &statements,
        &stacked,
        &domains,
        &frame.sizes,
        &FEE::zero(),
        &frame.config,
        &mut transcript,
        false,
        VerifierChecks::ALL,
    )
    .map_err(|e| format!("{e:?}"))
}

/// Build, optionally corrupt, prove and verify through the block prover.
/// `Err` = the prover refused.
fn block_round_trip(
    ops: &[LeanOp],
    spec: Spec,
    mutate: impl FnOnce(&mut Traces),
    corrupt: impl FnOnce(&mut Frame, &mut BlockProved<F, E>),
) -> Result<bool, String> {
    let mut t = build_traces(ops, &spec);
    mutate(&mut t);
    let roots = placeholder_roots(&t);
    let airs = airs(&roots, &options(), spec.drops);
    let mut frame = frame(&airs, &t)?;
    let tables = committed_tables(&airs, &t, &frame)?;
    let (mut proved, _) = prove_block(tables, &frame, PhaseBOverlap::Off)?;
    corrupt(&mut frame, &mut proved);
    match verify_block(&airs, &frame, &proved) {
        Ok(()) => Ok(true),
        Err(e) => {
            println!("LVP REJECTED: {}", e.chars().take(160).collect::<String>());
            Ok(false)
        }
    }
}

/// The AIRs' STARK roots are not read on the block path: zeros.
fn placeholder_roots(t: &Traces) -> Roots {
    Roots {
        rounds: vec![[0u8; 32]; t.rounds.len()],
        head: [0u8; 32],
        table: [0u8; 32],
    }
}

// =========================================================================
// Laptop: the block path accepts the lean tables, and refuses forgeries
// =========================================================================

#[test]
fn the_lean_tables_prove_and_verify_through_the_block_prover() {
    let ops = probe_ops(64);
    let t = build_traces(&ops, &Spec::default());
    let airs = airs(&placeholder_roots(&t), &options(), Drops::default());
    let f = frame(&airs, &t).expect("a frame");
    assert_eq!(
        f.config.log_blowup, 1,
        "the pinned block format is blowup 2"
    );
    assert_eq!(
        f.sizes,
        vec![f.shapes.len()],
        "64 compressions are one group"
    );
    assert_eq!(
        block_round_trip(&ops, Spec::default(), |_| {}, |_, _| {}),
        Ok(true)
    );
}

/// The box's two shapes group as the job registers them: 2^17 compressions
/// one group, 2^19 four (the mirror's height from the S4 mix's estimate).
#[test]
fn the_card_shapes_group_as_registered() {
    for (log_n, groups) in [(17u32, 1usize), (19, 4)] {
        let (a, rows) = s4_shape_tables(log_n, lean::MAX_CHUNK_LOG);
        let shapes: Vec<(usize, usize)> = a
            .refs()
            .iter()
            .zip(&rows)
            .map(|(air, &r)| (air.trace_layout().0, r.trailing_zeros() as usize))
            .collect();
        let config = BlockFormat::pinned().chain_config(&shapes);
        let sizes =
            multilinear_block::block_groups(&shapes, config.format.stack, BLOCK_GROUP_POLYS)
                .expect("groups");
        assert_eq!(
            sizes.len(),
            groups,
            "2^{log_n}: groups {sizes:?} over {shapes:?}"
        );
    }
}

/// A wrong carry (consistent downstream): only the round constraint sees it,
/// so the block's constraint sumcheck must.
#[test]
fn a_wrong_carry_rejects_through_the_block_prover() {
    assert_not_accepted(
        "block: wrong add3 carry",
        block_round_trip(
            &probe_ops(12),
            at(3, 5, FaultKind::Carry3),
            |_| {},
            |_, _| {},
        ),
    );
}

/// A digest word in memory that the head did not write: the bus.
#[test]
fn a_wrong_digest_in_memory_rejects_through_the_block_prover() {
    assert_not_accepted(
        "block: wrong digest in memory",
        block_round_trip(
            &probe_ops(12),
            Spec::default(),
            |t| {
                let rows = t.mirror.num_rows();
                let r = (0..rows)
                    .find(|&r| *t.mirror.main_table.get(r, mirror::RECV_MULT) != FE::zero())
                    .expect("a digest row");
                let v = *t.mirror.main_table.get(r, mirror::V0 + 2);
                t.mirror.main_table.set_fe(r, mirror::V0 + 2, v + FE::one());
            },
            |_, _| {},
        ),
    );
}

/// The verifier's byte table differs from the committed one in one value: the
/// evaluated preprocessed columns are what binds the table.
#[test]
fn a_wrong_preprocessed_column_rejects_through_the_block_prover() {
    assert_not_accepted(
        "block: verifier's byte table differs",
        block_round_trip(
            &probe_ops(12),
            Spec::default(),
            |_| {},
            |f, _| {
                // The byte table sits after the round chunks and the head.
                let xor = f.prep.len() - 2;
                let mut col = f.prep[xor][table::X].evals().to_vec();
                col[7] += FE::one();
                f.prep[xor][table::X] = Mle::new(col).expect("same height");
            },
        ),
    );
}

/// One commit-time answer changed in the proof.
#[test]
fn a_tampered_commit_answer_rejects_through_the_block_prover() {
    assert_not_accepted(
        "block: tampered commit-time answer",
        block_round_trip(
            &probe_ops(12),
            Spec::default(),
            |_| {},
            |_, proved| {
                proved.4[0] = &proved.4[0] + &FEE::one();
            },
        ),
    );
}

// =========================================================================
// The card
// =========================================================================

fn overlap_of(name: &str) -> PhaseBOverlap {
    match name {
        "pinned" => PhaseBOverlap::Pinned,
        "on" => PhaseBOverlap::On,
        "off" => PhaseBOverlap::Off,
        other => panic!("LVP_OVERLAPS: unknown overlap {other} (pinned, on, off)"),
    }
}

fn group_line(label: &str, g: usize, s: &GroupStamps) -> String {
    format!(
        "LVP GROUP: {label} · group {g} · tables {} · polys {} · cells {:.1} M · upload {:.3} \
         paid {:.3} · commit {:.3} · argue {:.3} · encode {:.3} · open {:.3} · open paid {:.3} · \
         overlapped {} · argue reserved {:.2} GiB",
        s.tables,
        s.polys,
        s.cells as f64 / 1e6,
        s.upload_a,
        s.upload_paid,
        s.commit,
        s.argue,
        s.encode,
        s.open,
        s.open_paid,
        s.overlapped,
        s.argue_reserved as f64 / (1u64 << 30) as f64,
    )
}

/// One timed block prove of `traces`: the columns copied out first (untimed),
/// then phase A and phase B under the pool's high-water, then the host verify.
fn block_run(
    airs: &Airs,
    traces: &Traces,
    frame: &Frame,
    overlap: &str,
    label: &str,
    log_n: u32,
) -> f64 {
    let tables = committed_tables(airs, traces, frame).expect("the block's tables");
    let main: u64 = frame.shapes.iter().map(|&(w, v)| (w as u64) << v).sum();
    // The STARK arm's clock: from before the pool's trim to after it.
    eprintln!("LVP RUN BEGIN: arm block · overlap {overlap} · run {label} · log_n {log_n}");
    let start = std::time::Instant::now();
    let (r, peak) = pool_peak(|| prove_block(tables, frame, overlap_of(overlap)));
    let secs = start.elapsed().as_secs_f64();
    let (proved, phases) = r.expect("the block prove");
    let v = std::time::Instant::now();
    let verdict = verify_block(airs, frame, &proved);
    let vsecs = v.elapsed().as_secs_f64();
    for (g, s) in proved.3.iter().enumerate() {
        eprintln!(
            "{}",
            group_line(&format!("run {label} · overlap {overlap}"), g, s)
        );
    }
    let peak_s = peak.map_or("-".to_string(), |b| {
        format!(
            "{:.2} GiB · {:.2} B a main cell",
            b as f64 / (1u64 << 30) as f64,
            b as f64 / main as f64
        )
    });
    eprintln!(
        "LVP RESULT: arm block · overlap {overlap} · run {label} · log_n {log_n} · compressions {} · \
         groups {} · main cells {main} · prove {secs:.3} s (commit {:.3} · phase B {:.3}) · \
         µs a compression {:.3} · pool peak {peak_s} · verify {vsecs:.2} s · {}",
        1u64 << log_n,
        frame.sizes.len(),
        phases.commit,
        phases.phase_b,
        secs * 1e6 / (1u64 << log_n) as f64,
        match &verdict {
            Ok(()) => "verified".to_string(),
            Err(e) => format!("REJECTED {e}"),
        }
    );
    assert!(verdict.is_ok(), "a block proof must verify");
    secs
}

/// Every digest the head wrote is the op's 7-round compression, and every
/// parent and every leaf chain's last digest is the `blake3` crate's hash.
fn digests_are_blake3(ops: &[LeanOp], digests: &[[u32; 8]]) -> (usize, usize) {
    assert_eq!(ops.len(), digests.len());
    let mut parents = 0usize;
    let mut leaves = 0usize;
    let mut chain: Vec<u8> = Vec::new();
    for (op, d) in ops.iter().zip(digests) {
        assert_eq!(*d, op.digest(), "a head digest is the 7-round compression");
        if !op.felt_mode {
            assert_eq!(
                digest_bytes(d),
                *blake3::hash(&op.block).as_bytes(),
                "a parent's digest is blake3::hash(left ‖ right)"
            );
            parents += 1;
            continue;
        }
        if op.flags & CHUNK_START != 0 {
            chain.clear();
        }
        chain.extend_from_slice(&op.block[..op.block_len as usize]);
        if op.flags & ROOT != 0 {
            assert_eq!(
                digest_bytes(d),
                *blake3::hash(&chain).as_bytes(),
                "a leaf chain's last digest is blake3::hash of its bytes"
            );
            leaves += 1;
        }
    }
    (parents, leaves)
}

/// ★ The card (box, `--features cuda`, `LAMBDA_VM_BASE_SPLIT=1`): the lean
/// tables of 2^17 and 2^19 compressions through the block prover, the STARK
/// prove of the 2^17 tables in the same binary as the control.
///
/// Knobs (env): `LVP_LOGN` (default `17,19`), `LVP_RUNS` (default 3, after one
/// untimed warm-up of every arm), `LVP_OVERLAPS` (the block arms' phase-B
/// overlaps, default `pinned,on`), `LVP_STARK` (the STARK control's arms at
/// 2^17 and below, default `p1,b3`). Lines: `LVP CONFIG`, `LVP SHAPE`,
/// `LVP DIGESTS`, `LVP RUN BEGIN`, `LVP GROUP` (one a group a block run),
/// `LVP RESULT` (one a block run), `B3R S4 RESULT` (one a STARK run),
/// `LVP DONE`; under `LAMBDA_VM_BASE_SPLIT`, the batched argue's own
/// `BATCHED ARGUE …` lines, one set a group in group order.
#[test]
#[ignore]
fn lvp_the_lean_tables_through_the_block_prover_on_the_card() {
    let opts = options();
    let logs: Vec<u32> = env_list("LVP_LOGN", "17,19")
        .iter()
        .map(|s| s.parse().expect("LVP_LOGN: integers"))
        .collect();
    let runs: usize = std::env::var("LVP_RUNS")
        .ok()
        .map(|s| s.parse().expect("LVP_RUNS: an integer"))
        .unwrap_or(3);
    let overlaps = env_list("LVP_OVERLAPS", "pinned,on");
    for o in &overlaps {
        overlap_of(o);
    }
    let stark_arms = env_list("LVP_STARK", "p1,b3");
    let block_config = BlockFormat::pinned().chain_config(&[(1, 20)]);
    eprintln!(
        "LVP CONFIG: block blowup 2^{} · queries {} · stack 2^{} · argue {:?} · group {:?} · \
         round0 {:?} · group polys {BLOCK_GROUP_POLYS} · hash {} · split lines {} · STARK blowup {} \
         queries {} · cuda {} · rayon threads {} · runs {runs} · overlaps {} · stark arms {}",
        block_config.log_blowup,
        block_config.num_queries,
        block_config.format.stack.get(),
        block_config.format.argue,
        block_config.format.group,
        block_config.format.round0,
        core::any::type_name::<H>(),
        multilinear::whir_split::enabled(),
        opts.blowup_factor,
        opts.fri_number_of_queries,
        cfg!(feature = "cuda"),
        threads(),
        overlaps.join(","),
        stark_arms.join(","),
    );
    for log_n in logs {
        let n = 1usize << log_n;
        let spec = Spec::default();
        let b = std::time::Instant::now();
        let ops = mixed_ops(n, 0x5EED_0000 + u64::from(log_n));
        let traces = build_traces(&ops, &spec);
        let build_s = b.elapsed().as_secs_f64();
        let (parents, leaves) = digests_are_blake3(&ops, &traces.digests);
        eprintln!(
            "LVP DIGESTS: log_n {log_n} · {n} head digests = the 7-round compression · {parents} \
             parents and {leaves} leaf chains = blake3::hash"
        );
        let airs = airs(&placeholder_roots(&traces), &opts, Drops::default());
        let f = frame(&airs, &traces).expect("the frame");
        let main: u64 = f.shapes.iter().map(|&(w, v)| (w as u64) << v).sum();
        let mut at = 0usize;
        let groups: Vec<String> = f
            .sizes
            .iter()
            .map(|&size| {
                let names: Vec<String> = (at..at + size)
                    .map(|i| format!("{}@2^{}", air_name(airs.refs()[i]), f.shapes[i].1))
                    .collect();
                at += size;
                names.join("+")
            })
            .collect();
        eprintln!(
            "LVP SHAPE: log_n {log_n} · compressions {n} · tables {} · groups {} [{}] · main cells \
             {main} · {:.1} main cells a compression · queries {} · trace build {build_s:.2} s",
            f.shapes.len(),
            f.sizes.len(),
            groups.join(" | "),
            main as f64 / n as f64,
            f.config.num_queries,
        );
        let stark = log_n <= 17 && !stark_arms.is_empty();
        let stark_roots = stark.then(|| {
            (
                roots_under::<<P1Arm as Arm>::S>(&traces, &opts),
                roots_under::<<B3Arm as Arm>::S>(&traces, &opts),
            )
        });
        let stark_run = |arm: &str, label: &str| {
            let (p1, b3) = stark_roots.as_ref().expect("STARK roots");
            match arm {
                "p1" => s4_run::<P1Arm>(p1, &traces, &opts, label, log_n),
                "b3" => s4_run::<B3Arm>(b3, &traces, &opts, label, log_n),
                other => panic!("LVP_STARK: unknown arm {other}"),
            }
        };
        // Warm-ups: every arm once, untimed.
        if stark {
            for arm in &stark_arms {
                stark_run(arm, "warm");
            }
        }
        for o in &overlaps {
            block_run(&airs, &traces, &f, o, "warm", log_n);
        }
        for run in 1..=runs {
            let label = format!("{run}");
            if stark {
                for arm in &stark_arms {
                    stark_run(arm, &label);
                }
            }
            for o in &overlaps {
                block_run(&airs, &traces, &f, o, &label, log_n);
            }
        }
    }
    eprintln!("LVP DONE");
}
