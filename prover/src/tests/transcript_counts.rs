/// The Fiat-Shamir transcript's state machine, counted rather than run.
///
/// Mirrors `DefaultTranscript` exactly: an absorb is one `Update::update` and
/// invalidates the duplex output buffer; a squeeze is one `finalize_reset`
/// PLUS the chain-advance `update` of the 32 reversed bytes, which is itself a
/// counted absorb; `state()` is a finalize on a clone that neither resets nor
/// re-absorbs. Four candidates come out of one squeeze, an extension element
/// takes three, a `sample_u64` takes one.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptCounts {
    pub absorb_calls: u64,
    pub absorb_bytes: u64,
    pub squeezes: u64,
    pub state_finalizes: u64,
}

impl TranscriptCounts {
    /// Every finalize the transcript performs: one per squeeze, one per state.
    pub fn finalizes(&self) -> u64 {
        self.squeezes + self.state_finalizes
    }

    /// `transcript_absorbs` as A2a counts it: the APPEND calls only.
    ///
    /// `absorb_calls` here includes the chaining re-absorb a squeeze performs,
    /// which A2a attributes to squeezing rather than to an absorb anyone asked
    /// for. Subtracting it is A2a's own non-vacuity identity read backwards:
    /// `absorb_calls == transcript_absorbs + transcript_squeezes`.
    pub fn transcript_absorbs(&self) -> u64 {
        self.absorb_calls - self.squeezes
    }

    /// `transcript_squeezes` as A2a counts it: one per `sample()`.
    pub fn transcript_squeezes(&self) -> u64 {
        self.squeezes
    }
}

struct Sim {
    out_pos: usize,
    c: TranscriptCounts,
}

const SQUEEZE_LEN: usize = 32;

impl Sim {
    /// `DefaultTranscript::new(&[])` absorbs the empty seed, which is still one
    /// `update` call of zero bytes.
    fn new() -> Self {
        let mut s = Self {
            out_pos: SQUEEZE_LEN,
            c: TranscriptCounts::default(),
        };
        s.absorb(0);
        s
    }
    fn absorb(&mut self, n: u64) {
        self.out_pos = SQUEEZE_LEN;
        self.c.absorb_calls += 1;
        self.c.absorb_bytes += n;
    }
    fn squeeze(&mut self) {
        self.c.squeezes += 1;
        // `sample()` re-absorbs its own reversed output to advance the chain.
        self.c.absorb_calls += 1;
        self.c.absorb_bytes += 32;
        self.out_pos = 0;
    }
    fn next_u64(&mut self) {
        if self.out_pos + 8 > SQUEEZE_LEN {
            self.squeeze();
        }
        self.out_pos += 8;
    }
    /// A cubic-extension element: three base coordinates, each one candidate.
    /// Exact unless a candidate lands at or above the modulus, which is one
    /// extra draw with probability about 2^-32 per coordinate.
    fn sample_ext(&mut self) {
        for _ in 0..3 {
            self.next_u64();
        }
    }
    fn sample_u64(&mut self) {
        self.next_u64();
    }
    fn state(&mut self) {
        self.c.state_finalizes += 1;
    }
}

/// One table's transcript shape, read off the AIR and its layout.
#[derive(Clone, Copy, Debug)]
pub struct TableTranscriptShape {
    /// Height in variables.
    pub n: usize,
    /// GKR input-layer variables, `n + ceil_log2(interactions)`.
    pub m: usize,
    /// The batched sumcheck's degree, `max(shape.degree() + 1, 2)`.
    pub degree: usize,
    /// Committed factors, whose values `claim_reduce` absorbs.
    pub factors: usize,
    /// Main columns, whose values `claim_reduce` absorbs at the end.
    pub columns: usize,
}

/// One commitment group: the stack it landed on and how many columns it holds.
#[derive(Clone, Copy, Debug)]
pub struct GroupTranscriptShape {
    pub n_stack: usize,
    pub num_polys: usize,
    pub columns: usize,
}

fn drive_table(s: &mut Sim, t: &TableTranscriptShape) {
    // multilinear_table::verify
    s.absorb(24);
    s.absorb(24); // bus_output.0, .1
    // gkr::verify: m layers, layer i runs i degree-3 sumcheck rounds
    for i in 0..t.m {
        s.sample_ext(); // lambda
        for _ in 0..i {
            for _ in 0..3 {
                s.absorb(24);
            }
            s.sample_ext();
        }
        for _ in 0..4 {
            s.absorb(24); // p_lo, p_hi, q_lo, q_hi
        }
        s.sample_ext(); // c
    }
    for _ in 0..t.n {
        s.sample_ext(); // r
    }
    // constraint_argument::verify_core -> batch::verify
    for _ in 0..3 {
        s.absorb(24); // the three rule claims
    }
    s.sample_ext(); // the batching challenge
    for _ in 0..t.n {
        for _ in 0..t.degree {
            s.absorb(24);
        }
        s.sample_ext();
    }
    // claim_reduce::verify
    for _ in 0..t.factors {
        s.absorb(24);
    }
    s.sample_ext();
    for _ in 0..t.n {
        for _ in 0..2 {
            s.absorb(24);
        }
        s.sample_ext();
    }
    for _ in 0..t.columns {
        s.absorb(24);
    }
}

fn schedule(num_vars: usize, k: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut left = num_vars;
    while left > 0 {
        let take = k.min(left);
        out.push(take);
        left -= take;
    }
    out
}

fn drive_chain(s: &mut Sim, n_stack: usize, k: usize, queries: usize) {
    let sch = schedule(n_stack, k);
    let rounds = sch.len();
    for (r, &kr) in sch.iter().enumerate() {
        s.state(); // check_grind(folding)
        s.absorb(8);
        for _ in 0..kr {
            s.absorb(24);
            s.absorb(24); // degree-2 round polynomial
            s.sample_ext();
        }
        if r + 1 < rounds {
            s.absorb(32); // next_root
            s.sample_ext(); // z0
            s.absorb(24); // y0
            s.state(); // check_grind(ood)
            s.absorb(8);
            s.sample_ext(); // gamma
            s.state(); // check_grind(query)
            s.absorb(8);
        } else {
            s.absorb(24); // final_value
            s.state(); // check_grind(query)
            s.absorb(8);
        }
        for _ in 0..queries {
            s.sample_u64();
        }
    }
}

/// One whole proof's transcript: the statement, the roots, every table, every
/// commitment group's opening.
#[allow(clippy::too_many_arguments)]
pub fn transcript_counts(
    statement_absorbs: &[u64],
    tables: &[TableTranscriptShape],
    groups: &[GroupTranscriptShape],
    log_folding: usize,
    queries: usize,
    owed_probe: bool,
) -> TranscriptCounts {
    let mut s = Sim::new();
    for &n in statement_absorbs {
        s.absorb(n);
    }
    let roots: usize = groups.iter().map(|g| g.num_polys).sum();
    // `owed` runs the COMMIT bus's counterparty on a CLONE of the transcript:
    // every root again, then z and alpha. The clone is thrown away, so it moves
    // no challenge — but it is real hashing and a counter sees it. Only the
    // epoch path does this; the cross-epoch proof has no commit bus to owe.
    if owed_probe {
        let mut probe = Sim {
            out_pos: SQUEEZE_LEN,
            c: TranscriptCounts::default(),
        };
        for _ in 0..roots {
            probe.absorb(32);
        }
        probe.sample_ext();
        probe.sample_ext();
        s.c.absorb_calls += probe.c.absorb_calls;
        s.c.absorb_bytes += probe.c.absorb_bytes;
        s.c.squeezes += probe.c.squeezes;
    }
    // multi_verify: every root, then z, alpha, beta.
    for _ in 0..roots {
        s.absorb(32);
    }
    for _ in 0..3 {
        s.sample_ext();
    }
    for t in tables {
        drive_table(&mut s, t);
    }
    for g in groups {
        for _ in 0..g.columns {
            s.absorb(24);
        }
        s.sample_ext(); // the batching challenge
        for _ in 0..g.num_polys {
            drive_chain(&mut s, g.n_stack, log_folding, queries);
        }
    }
    s.c
}

// =========================================================================
// Driving it off the real AIRs
// =========================================================================

use crate::multilinear_continuation::{epoch_groups, global_groups};
use crate::multilinear_prove::{chain_config, stacks};
use crate::tables::trace_builder::DecodeArtifacts;
use executor::elf::Elf;
use multilinear::whir_chain::ChainConfig;
use stark::traits::AIR;

/// The tags, copied so the byte totals are exact. A drift here fails the
/// measured arm below, which is where it should fail.
const EPOCH_TAG: &[u8] = b"LAMBDAVM_MULTILINEAR_CONTINUATION_EPOCH_V1";
const GLOBAL_TAG: &[u8] = b"LAMBDAVM_MULTILINEAR_CONTINUATION_GLOBAL_V1";
/// `absorb_table_counts` writes one `u64` per sharded table kind.
const TABLE_COUNT_WORDS: usize = 14;

fn epoch_statement_absorbs(tag: &[u8], public_output: usize, table_num_vars: usize) -> Vec<u64> {
    let mut v = vec![tag.len() as u64, 32, 8, 8, public_output as u64];
    v.extend(std::iter::repeat_n(8u64, TABLE_COUNT_WORDS));
    v.extend([8, table_num_vars as u64, 8, 8, 8, 3]);
    v
}

fn global_statement_absorbs(pages: usize, table_num_vars: usize) -> Vec<u64> {
    let mut v = vec![GLOBAL_TAG.len() as u64, 32, 8, 8, 8];
    v.extend(std::iter::repeat_n(8u64, pages));
    v.extend([8, table_num_vars as u64, 8, 8, 8, 3]);
    v
}

/// One proof's table and group shapes, from the AIRs alone.
fn shapes_of_proof(
    pairs: &[(
        &dyn AIR<
            Field = crate::test_utils::F,
            FieldExtension = crate::test_utils::E,
            PublicInputs = (),
        >,
        usize,
        usize,
    )],
    sizes: &[usize],
    config: &ChainConfig,
) -> (Vec<TableTranscriptShape>, Vec<GroupTranscriptShape>) {
    let shapes: Vec<(usize, usize)> = pairs.iter().map(|&(_, w, v)| (w, v)).collect();
    let mut tables = Vec::with_capacity(pairs.len());
    for &(air, width, num_vars) in pairs {
        let layout = stark::multilinear_table::TableLayout::<
            crate::test_utils::F,
            crate::test_utils::E,
        >::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            width,
            num_vars,
            stark::multilinear_air::Uniforms::default(),
        )
        .expect("layout");
        let factors = layout
            .kinds()
            .iter()
            .filter(|k| k.source().is_some())
            .count();
        tables.push(TableTranscriptShape {
            n: num_vars,
            m: multilinear::logup::input_layer_vars(air.bus_interactions().len(), num_vars),
            degree: (layout.shape().degree() + 1).max(2),
            factors,
            columns: layout.slot_of().len(),
        });
    }
    let (layouts, _d) = stacks(&shapes, sizes, config).expect("stacks");
    let mut groups = Vec::with_capacity(layouts.len());
    let mut at = 0usize;
    for (layout, &size) in layouts.iter().zip(sizes) {
        groups.push(GroupTranscriptShape {
            n_stack: layout.n_stack(),
            num_polys: layout.num_polys(),
            columns: shapes[at..at + size].iter().map(|&(w, _)| w).sum(),
        });
        at += size;
    }
    (tables, groups)
}

/// Every proof of a continuation, as transcript shapes.
fn continuation_transcript_counts(
    elf_bytes: &[u8],
    inputs: &[u8],
    epoch_size_log2: u32,
    opts: &crate::ProofOptions,
    verbose: bool,
) -> (TranscriptCounts, usize, u64, u64) {
    let elf = Elf::load(elf_bytes).expect("load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let mut total = TranscriptCounts::default();
    let mut proofs = 0usize;
    let mut eager_calls = 0u64;
    let mut eager_bytes = 0u64;
    let add = |c: TranscriptCounts, total: &mut TranscriptCounts| {
        total.absorb_calls += c.absorb_calls;
        total.absorb_bytes += c.absorb_bytes;
        total.squeezes += c.squeezes;
        total.state_finalizes += c.state_finalizes;
    };

    let boundaries = crate::continuation::for_each_epoch(
        &elf,
        inputs,
        epoch_size_log2,
        &artifacts,
        |prepared, _| {
            let mut traces = prepared.traces;
            crate::tables::bitwise::update_multiplicities(
                &mut traces.bitwise,
                &crate::tables::local_to_global::collect_bitwise_from_l2g(&prepared.boundary),
            );
            let reg_fini = crate::tables::register::fini_from_trace(&traces.register);
            let table_counts = traces.table_counts();
            let public_output = traces.public_output_bytes.clone();
            // ⚠ `build_epoch_airs` eagerly computes the REGISTER preprocessed
            // commitment — a keccak Merkle build the multilinear verifier never
            // consumes, because `check_preprocessed` binds those columns
            // instead. It is not transcript work, but a keccak counter sees it,
            // so it is measured here rather than left to pollute the compare.
            crypto::hash_metrics::reset();
            let airs = crate::continuation::build_epoch_airs(
                &elf,
                opts,
                &[],
                &table_counts,
                &prepared.register_init,
                &reg_fini,
                prepared.is_final,
                None,
            );
            let eager = crypto::hash_metrics::snapshot();
            eager_calls += eager.absorb_calls;
            eager_bytes += eager.absorb_bytes;
            let l2g_air = crate::continuation::l2g_memory_air(opts, prepared.label);
            let mut l2g_trace =
                crate::tables::local_to_global::generate_local_to_global_trace(&prepared.boundary);
            let mut pairs = airs.air_trace_pairs(&mut traces);
            pairs.push((&l2g_air, &mut l2g_trace, &()));
            let triples: Vec<_> = pairs
                .iter()
                .map(|(air, t, _)| {
                    (
                        *air,
                        t.main_table.width,
                        t.main_table.height.trailing_zeros() as usize,
                    )
                })
                .collect();
            let shapes: Vec<(usize, usize)> =
                triples.iter().map(|&(_, w, v)| (w, v)).collect();
            let config = chain_config(&shapes);
            let sizes = epoch_groups(shapes.len());
            let (tables, groups) = shapes_of_proof(&triples, &sizes, &config);
            let c = transcript_counts(
                &epoch_statement_absorbs(EPOCH_TAG, public_output.len(), shapes.len()),
                &tables,
                &groups,
                config.log_folding,
                config.num_queries,
                true,
            );
            if verbose {
                let sum_m: usize = tables.iter().map(|t| t.m).sum();
                let gkr_rounds: usize = tables.iter().map(|t| t.m * (t.m - 1) / 2).sum();
                let sum_n: usize = tables.iter().map(|t| t.n).sum();
                let cols: usize = tables.iter().map(|t| t.columns).sum();
                let facs: usize = tables.iter().map(|t| t.factors).sum();
                let degs: usize = tables.iter().map(|t| t.n * t.degree).sum();
                let roots: usize = groups.iter().map(|g| g.num_polys).sum();
                let chain_rounds: usize = groups
                    .iter()
                    .map(|g| g.num_polys * g.n_stack.div_ceil(config.log_folding))
                    .sum();
                println!(
                    "epoch {:>2}: transcript_absorbs {:>8} transcript_squeezes {:>7} | absorb_calls {:>9} bytes {:>11} states {:>6}",
                    prepared.index,
                    c.transcript_absorbs(),
                    c.transcript_squeezes(),
                    c.absorb_calls,
                    c.absorb_bytes,
                    c.state_finalizes
                );
                println!(
                    "          tables {} sum_m {} gkr_rounds {} sum_n {} cols {} factors {} n*deg {} roots {} chain_rounds {} Q {}",
                    tables.len(), sum_m, gkr_rounds, sum_n, cols, facs, degs, roots, chain_rounds, config.num_queries
                );
            }
            add(c, &mut total);
            proofs += 1;
            Ok(())
        },
    )
    .expect("epochs");

    // The cross-epoch proof.
    let init_page_data = crate::tables::trace_builder::build_init_page_data(
        &crate::tables::trace_builder::build_initial_image_paged(&elf, inputs),
    );
    let num_private_input_pages = crate::tables::page::private_input_page_count(inputs);
    let page_bases = crate::continuation::touched_page_bases(&boundaries);
    let gm_configs = crate::continuation::global_memory_configs_from_init_page_data(
        &page_bases,
        &init_page_data,
        num_private_input_pages,
        true,
    );
    let mut final_state: crate::tables::global_memory::FiniStateMap =
        std::collections::HashMap::new();
    for epoch in &boundaries {
        for b in epoch.iter() {
            final_state.insert(
                b.address,
                crate::tables::global_memory::FiniState {
                    value: (b.fini.value & 0xFF) as u8,
                    epoch: b.fini.epoch,
                },
            );
        }
    }
    let l2g_airs: Vec<_> = (0..boundaries.len())
        .map(|i| {
            crate::continuation::l2g_global_air(
                opts,
                crate::tables::local_to_global::epoch_label(i as u64),
            )
        })
        .collect();
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|c| crate::continuation::global_memory_air(opts, c, None))
        .collect();
    let mut l2g_traces: Vec<_> = boundaries
        .iter()
        .map(|e| crate::tables::local_to_global::generate_local_to_global_trace(e.as_slice()))
        .collect();
    let mut gm_traces: Vec<_> = gm_configs
        .iter()
        .map(|c| crate::tables::global_memory::generate_global_trace(c, &final_state))
        .collect();
    let mut gpairs: Vec<crate::AirTracePair<'_>> = Vec::new();
    for (a, t) in l2g_airs.iter().zip(l2g_traces.iter_mut()) {
        gpairs.push((a, t, &()));
    }
    for (a, t) in gm_airs.iter().zip(gm_traces.iter_mut()) {
        gpairs.push((a, t, &()));
    }
    let gtriples: Vec<_> = gpairs
        .iter()
        .map(|(air, t, _)| {
            (
                *air,
                t.main_table.width,
                t.main_table.height.trailing_zeros() as usize,
            )
        })
        .collect();
    let gshapes: Vec<(usize, usize)> = gtriples.iter().map(|&(_, w, v)| (w, v)).collect();
    let gconfig = chain_config(&gshapes);
    let gsizes = global_groups(boundaries.len(), gm_configs.len());
    let (gtables, ggroups) = shapes_of_proof(&gtriples, &gsizes, &gconfig);
    let gc = transcript_counts(
        &global_statement_absorbs(page_bases.len(), gshapes.len()),
        &gtables,
        &ggroups,
        gconfig.log_folding,
        gconfig.num_queries,
        false,
    );
    if verbose {
        println!(
            "global  : transcript_absorbs {:>8} transcript_squeezes {:>7} | absorb_calls {:>9} bytes {:>11} states {:>6}",
            gc.transcript_absorbs(),
            gc.transcript_squeezes(),
            gc.absorb_calls,
            gc.absorb_bytes,
            gc.state_finalizes
        );
        // The cross-epoch proof's own shape line. It was missing, and that is
        // where a guest difference lands: the epoch rows are execution-derived
        // and matched across two ELFs to the unit, while this proof's
        // GLOBAL_MEMORY tables are built from the ELF's genesis image.
        let sum_m: usize = gtables.iter().map(|t| t.m).sum();
        let gkr_rounds: usize = gtables.iter().map(|t| t.m * (t.m - 1) / 2).sum();
        let sum_n: usize = gtables.iter().map(|t| t.n).sum();
        let cols: usize = gtables.iter().map(|t| t.columns).sum();
        let facs: usize = gtables.iter().map(|t| t.factors).sum();
        let degs: usize = gtables.iter().map(|t| t.n * t.degree).sum();
        let groots: usize = ggroups.iter().map(|g| g.num_polys).sum();
        let chain_rounds: usize = ggroups
            .iter()
            .map(|g| g.num_polys * g.n_stack.div_ceil(gconfig.log_folding))
            .sum();
        println!(
            "          tables {} sum_m {} gkr_rounds {} sum_n {} cols {} factors {} n*deg {} roots {} chain_rounds {} Q {}",
            gtables.len(),
            sum_m,
            gkr_rounds,
            sum_n,
            cols,
            facs,
            degs,
            groots,
            chain_rounds,
            gconfig.num_queries
        );
        // These AIRs carry no name, so the rows identify themselves: the first
        // `num_epochs` are the bookends in epoch order, the rest are the
        // global-memory tables in `page_bases` order, labelled by page base.
        println!("\n-- cross-epoch per-table census --");
        println!(
            "{:<22} {:>7} {:>10} {:>6} {:>5} {:>14}",
            "table", "width", "rows", "vars", "m", "cells"
        );
        let num_bookends = boundaries.len();
        for (i, ((_, w, v), t)) in gtriples.iter().zip(&gtables).enumerate() {
            let label = if i < num_bookends {
                format!("L2G[epoch {i}]")
            } else {
                format!("GM[page {:#012x}]", page_bases[i - num_bookends])
            };
            println!(
                "{label:<22} {w:>7} {:>10} {v:>6} {:>5} {:>14}",
                1usize << v,
                t.m,
                (*w as u64) << v
            );
        }
    }
    add(gc, &mut total);
    proofs += 1;
    (total, proofs, eager_calls, eager_bytes)
}

/// ★ The closed form against a MEASURED verify.
///
/// Run with `--features hash-metrics` and `LAMBDA_VM_WHIR_HASH=rpx`: the
/// algebraic Merkle backend and the RPX grind hash nothing through the counted
/// keccak wrapper, so `absorb_calls` isolates the Fiat-Shamir transcript —
/// which is still keccak, because every WHIR call site takes
/// `DefaultTranscript`'s defaulted hash parameter. That is the configuration
/// this arm exists to measure, and the one F5 says must not survive to D3.
#[test]
#[ignore]
fn the_transcript_closed_form_matches_a_measured_verify() {
    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "fibonacci".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let bytes = super::multilinear_bench_tests::elf_bytes(&name);
    let inputs = super::multilinear_bench_tests::input_bytes(&input);
    let opts = super::multilinear_bench_tests::options();
    println!(
        "\n== transcript closed form vs a measured verify: {name}, epoch 2^{epoch_size_log2} =="
    );

    let (predicted, proofs, eager_calls, eager_bytes) =
        continuation_transcript_counts(&bytes, &inputs, epoch_size_log2, &opts, true);

    let bundle = crate::multilinear_continuation::prove_continuation(
        &bytes,
        &inputs,
        epoch_size_log2,
        &opts,
    )
    .expect("prove");
    assert_eq!(bundle.num_epochs() + 1, proofs, "epoch count");
    crypto::hash_metrics::reset();
    let ok = crate::multilinear_continuation::verify_continuation(&bytes, &bundle, &opts)
        .expect("verify");
    assert!(ok, "the continuation must verify");
    let m = crypto::hash_metrics::snapshot();

    // Outside the transcript: `statement::elf_digest` is one keccak absorb of
    // the whole ELF plus one finalize, once per proof verified.
    let elf_calls = proofs as u64;
    let elf_bytes_total = proofs as u64 * bytes.len() as u64;
    println!(
        "\n{:<26} {:>12} {:>14} {:>10} {:>8}",
        "", "absorb_calls", "absorb_bytes", "squeezes", "states"
    );
    println!(
        "{:<26} {:>12} {:>14} {:>10} {:>8}",
        "closed form (transcript)",
        predicted.absorb_calls,
        predicted.absorb_bytes,
        predicted.squeezes,
        predicted.state_finalizes
    );
    println!(
        "{:<26} {:>12} {:>14}",
        "+ elf_digest, one/proof", elf_calls, elf_bytes_total
    );
    println!(
        "{:<26} {:>12} {:>14}   <- NOT transcript work",
        "+ eager REGISTER commit", eager_calls, eager_bytes
    );
    println!(
        "{:<26} {:>12} {:>14}",
        "MEASURED", m.absorb_calls, m.absorb_bytes
    );
    println!(
        "{:<26} total {} merkle {} merkle_nodes {} grinding {}",
        "measured finalizes", m.total, m.merkle, m.merkle_nodes, m.grinding
    );
    println!(
        "{:<26} {:>12} {:>14}",
        "difference",
        m.absorb_calls as i64 - (predicted.absorb_calls + elf_calls + eager_calls) as i64,
        m.absorb_bytes as i64 - (predicted.absorb_bytes + elf_bytes_total + eager_bytes) as i64,
    );
    #[cfg(feature = "hash-metrics")]
    {
        assert_eq!(
            m.absorb_calls,
            predicted.absorb_calls + elf_calls + eager_calls,
            "the closed form must predict every counted transcript absorb"
        );
        assert_eq!(
            m.absorb_bytes,
            predicted.absorb_bytes + elf_bytes_total + eager_bytes,
            "and every counted byte"
        );
    }
    #[cfg(not(feature = "hash-metrics"))]
    println!("\n(counters are compiled out; rerun with --features hash-metrics to assert)");
}

/// The block's predicted counts — what A2a's assert is scored against.
#[test]
#[ignore]
fn whir_transcript_counts_for_the_block() {
    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
    let input =
        std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_else(|_| "ethrex_mainnet_25368371".into());
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(21);
    let bytes = super::multilinear_bench_tests::elf_bytes(&name);
    let inputs = super::multilinear_bench_tests::input_bytes(&input);
    let opts = super::multilinear_bench_tests::options();
    println!("\n== transcript counts, {input}, epoch 2^{epoch_size_log2} ==");
    let (c, proofs, eager_calls, eager_bytes) =
        continuation_transcript_counts(&bytes, &inputs, epoch_size_log2, &opts, true);
    println!(
        "\nTOTAL over {proofs} proofs:\n  transcript_absorbs   {}\n  transcript_squeezes  {}\n  absorb_calls         {}  (= the two above, A2a's non-vacuity identity)\n  absorb_bytes         {}\n  state finalizes      {}  <- NEITHER bucket: `state()` is a finalize on a clone",
        c.transcript_absorbs(),
        c.transcript_squeezes(),
        c.absorb_calls,
        c.absorb_bytes,
        c.state_finalizes
    );
    println!(
        "eager REGISTER commitment, outside the transcript: {eager_calls} absorbs, {eager_bytes} bytes"
    );
    println!(
        "transcript finalizes (squeezes + states) = {}",
        c.finalizes()
    );
}
