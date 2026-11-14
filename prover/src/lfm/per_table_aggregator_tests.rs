//! The aggregation node over real child proofs: the child harvest the block
//! tree's tests share ([`real_child`], [`child_shape`], [`child_arena_words`]),
//! nodes emitted over STARK-proved epoch wraps and proved as the next level's
//! child, and the in-order pool the tree driver runs a level on.

use crate::tables::types::{FE, FEE};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::transcript_replay::TranscriptReplay;
use super::word::{LfmWord, base_word};

#[test]
fn the_aggregation_publish_profile_drops_only_diagnostics() {
    use super::epoch_tests::{Publishes, epoch_program_publishing, schema_words};

    let e = super::epoch_tests::real_epoch();
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);

    let run = |publishes| {
        let program = epoch_program_publishing(&e, true, publishes);
        let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
            .expect("the assembled verifier must execute under either profile");
        (program.instrs.len(), exec.public_words)
    };
    let (diag_instrs, diag) = run(Publishes::Diagnostic);
    let (agg_instrs, agg) = run(Publishes::Aggregation);

    // ---- the head: the pair, the id and the schema, identical cells.
    let head = 2 + 2 + schema_words(&e);
    assert_eq!(
        diag[..head],
        agg[..head],
        "the binding head must not move with the profile"
    );
    // ---- the tail: the closure's total, which both profiles publish LAST.
    //
    // ⚠ Against PRODUCTION'S OWN TARGET, one profile at a time — never by
    // comparing the two lists' last entries to each other. A published word is
    // an `(index, value)` pair, and the two indices cannot be equal: making the
    // lists different lengths is the entire point of R2. That was this
    // assertion's first form, and it failed on box A with the four field
    // elements matching exactly and only the indices differing (307 against
    // 144) — the emitter doing precisely what it should, caught by a test
    // asserting something it never meant.
    //
    // The oracle here is also stronger than the one it replaces: production's
    // COMMIT-bus balance, rather than "the other profile agrees with me".
    for (label, words) in [("Diagnostic", &diag), ("Aggregation", &agg)] {
        let (index, word) = words.last().expect("a profile publishes words");
        assert_eq!(
            *index as usize,
            words.len() - 1,
            "{label}: publish indices auto-increment, so the last word's index \
             is len-1"
        );
        assert_eq!(
            super::word::word_as_ext(word).expect("the bus total is ext"),
            e.expected_bus_balance,
            "{label}: the closure's total must end the list and reach \
             production's own COMMIT-bus target"
        );
    }
    assert_eq!(agg.len(), head + 1, "Aggregation is the head and the total");

    // ---- what was dropped, from the epoch's own shapes rather than from a
    // literal: per sub-proof the composition, one terminal per query, the
    // (beta, z, gamma) triple, the DEEP zetas and one index per query.
    let dropped: usize = e
        .tables
        .iter()
        .zip(&e.legs)
        .map(|(h, leg)| 1 + leg.verify.num_queries + 3 + h.zetas.len() + h.shape.num_queries)
        .sum();
    assert_eq!(
        diag.len(),
        agg.len() + dropped,
        "Aggregation must drop exactly the per-sub-proof diagnostics"
    );

    println!(
        "★ publish profile over {} sub-proofs: Diagnostic {} words / {diag_instrs} instrs, \\
         Aggregation {} words / {agg_instrs} instrs ({:.1}% of the words, {:.1}% of the \\
         instructions)",
        e.tables.len(),
        diag.len(),
        agg.len(),
        100.0 * agg.len() as f64 / diag.len() as f64,
        100.0 * agg_instrs as f64 / diag_instrs as f64,
    );
}

// ======================= the child harvest and the node ===================

/// One child LFM proof, production-accepted, harvested for emission (the
/// reader lives in [`super::harvest`], where the block tree's driver uses it
/// too).
pub(super) use super::harvest::HarvestedChild as RealChild;

/// Harvest a child from a proof PRODUCTION ACCEPTS. Panics loudly otherwise —
/// nothing downstream may read a proof the verifier would reject.
pub(super) fn real_child(
    artifacts: super::registry::LfmArtifacts,
    opts: crate::ProofOptions,
    proved: &super::proof::LfmProof,
) -> RealChild {
    real_child_timed(artifacts, opts, proved).0
}

/// [`real_child`], and the seconds its opening assert cost.
///
/// ⛔ THE ASSERT IS TIMED, NOT SKIPPED, AND THERE IS NO KNOB THAT SKIPS IT.
/// It is a complete host STARK verify of the proof produced moments earlier,
/// and it is what the harness's trustworthiness rests on — a child read from a
/// proof production would reject describes nothing. It belongs on a line of
/// its own rather than inside a `harvest` figure that then gets quoted as the
/// cost of harvesting.
pub(super) fn real_child_timed(
    artifacts: super::registry::LfmArtifacts,
    opts: crate::ProofOptions,
    proved: &super::proof::LfmProof,
) -> (RealChild, f64) {
    super::harvest::harvest_child_verified(artifacts, opts, proved)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// The child's shape, as the node's emitter reads it.
pub(super) fn child_shape(c: &RealChild) -> super::per_table_aggregator::ChildShape<'_> {
    super::per_table_aggregator::ChildShape {
        program_id: &c.artifacts.program_id,
        num_public_words: c.public_words.len(),
        fri_final_poly_log_degree: c.opts.fri_final_poly_log_degree,
        statement_tag: super::statement::LFM_STATEMENT_TAG,
        tables: c
            .tables
            .iter()
            .zip(&c.legs)
            .map(|(h, leg)| super::per_table_aggregator::ChildTable {
                challenge: &h.shape,
                verify: &leg.verify,
                analysis: &leg.analysis,
                precomputed_root: leg.precomputed_commitment.as_ref(),
            })
            .collect(),
    }
}

/// The child's arenas, in `declare_leg_arenas`' declaration order.
pub(super) fn child_arena_words(c: &RealChild) -> Vec<Vec<LfmWord>> {
    super::harvest::try_child_arena_words(c).unwrap_or_else(|e| panic!("{e}"))
}

/// The aggregation node over `children`, at any level and any arity.
pub(super) fn node_program(
    children: &[RealChild],
    layouts: &[super::per_table_aggregator::SchemaLayout],
    labels: &[&[u64]],
    label_range: (u64, u64),
    publishes: super::per_table_aggregator::NodePublishSet,
) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let shapes: Vec<_> = children.iter().map(child_shape).collect();
    super::per_table_aggregator::emit_node(
        &mut b,
        &super::per_table_aggregator::NodeInputs {
            children: &shapes,
            layouts,
            labels,
            label_range,
            publishes,
        },
    );
    compile(b.finish())
}

/// Name any sub-proof whose query sampler would be handed a zero bit width,
/// BEFORE emission reaches it.
///
/// `epoch::emit_table_challenges` samples each query index with
/// `sample_u64_pow2(shape.index_bits())`, and `index_bits()` is
/// `log2_trace_length + log2_blowup - 1`. A one-row trace at blowup 2 makes that
/// ZERO, and the sampler's own assert then fires deep inside emission with no
/// idea which table or which side of the tree it came from — which is exactly
/// how it surfaced on box A: a bare "nbits must be in 1..=32, got 0" with
/// nothing to attach it to.
///
/// A diagnostic, not a fix. If it fires, the shape is real and the question is
/// whether a one-row sub-proof should exist at all at this preset.
fn assert_samplable(label: &str, shapes: &[&super::epoch::TableChallengeShape]) {
    for (i, s) in shapes.iter().enumerate() {
        let lde = s.log2_trace_length + s.log2_blowup;
        assert!(
            lde >= 1,
            "{label} sub-proof {i}: log2_trace {} + log2_blowup {} = {lde}, so \
             index_bits() would be {} — the query sampler needs at least one bit",
            s.log2_trace_length,
            s.log2_blowup,
            lde as i64 - 1,
        );
    }
    let worst = shapes
        .iter()
        .enumerate()
        .min_by_key(|(_, s)| s.log2_trace_length + s.log2_blowup)
        .expect("a proof has sub-proofs");
    println!(
        "   {label}: {} sub-proofs, shallowest is #{} at log2_trace {} + log2_blowup {} = {} index bits",
        shapes.len(),
        worst.0,
        worst.1.log2_trace_length,
        worst.1.log2_blowup,
        worst.1.index_bits(),
    );
}

/// ★ THE LEAF NODE RUNS — the first aggregation node over real per-table wrap
/// proofs.
///
/// # What it is
///
/// `FAN_IN` epochs of a fixture continuation, each wrapped by the assembled
/// per-table epoch verifier under `Publishes::Aggregation`, and ONE emitted
/// program that verifies both wrap proofs and binds them: the shared attestation
/// id, the register fini→init seam, and each epoch's label pinned to its chain
/// position as a constant. The node then publishes the schema its own parent
/// will read.
///
/// # Why the tamper arms are three and not one
///
/// Each binding leg can fail on its own and a single arm would not tell them
/// apart. The register seam, the shared id and the label pin are three
/// independent claims about the relationship between two proofs that both
/// verify — and "both children verified" is exactly what a broken binding still
/// looks like. Each arm moves ONE published word in ONE child's arena and
/// nothing else, so what fails is named by which arm failed.
///
/// # Cost
///
/// Box tier and `#[ignore]`d: two epoch wraps proved (a wrap proof carries a
/// full LFM chip set), then a node program that verifies both. The suite gates
/// the pieces — `wrap_tests::the_fixture_epoch_wraps` proves one wrap,
/// `the_global_verifier_leg_runs_and_rejects_tampers` runs a per-table leg, and
/// `the_aggregation_publish_profile_drops_only_diagnostics` pins what a wrap
/// publishes — so this is the assembly rather than any of its parts.
#[test]
#[ignore = "box tier: proves FAN_IN epoch wraps and a node over them"]
fn the_leaf_node_verifies_and_binds_two_wraps() {
    use super::epoch_tests::Publishes;
    use super::per_table_aggregator::{FAN_IN, NodePublishSet, SchemaLayout};
    use super::program_census::build_artifacts_counted;
    use super::proof::lfm_prove;
    use std::time::Instant;

    let elf_bytes = super::proof_fixture::read_inner_elf();
    let inner = super::proof_fixture::fixture_options();
    let bundle = crate::continuation::prove_continuation(
        &elf_bytes,
        &[],
        super::proof_fixture::FIXTURE_EPOCH_LOG2,
        &inner,
    )
    .expect("the fixture continuation must prove");
    assert!(
        bundle.num_epochs() >= FAN_IN,
        "a fan-in-{FAN_IN} leaf needs {FAN_IN} epochs, the fixture has {}",
        bundle.num_epochs()
    );

    // ---- the children: one wrap per epoch, at the AGGREGATION publish set.
    let t = Instant::now();
    let mut children = Vec::with_capacity(FAN_IN);
    let mut layouts = Vec::with_capacity(FAN_IN);
    let mut labels = Vec::with_capacity(FAN_IN);
    let wrap_opts = super::proof::aggregation_wrap_options();
    let epoch_konsts = super::epoch_tests::EpochConstants::load(&elf_bytes, &inner, None)
        .expect("the inner ELF and its DECODE commitment must build once");
    for k in 0..FAN_IN {
        let e = super::epoch_tests::real_epoch_from_constants(&inner, &epoch_konsts, &bundle, k)
            .expect("every epoch must reconstruct from proofs alone");
        let out_halves = e.statement.public_output_len.div_ceil(4);
        // Pre-flight: the wrap program emits one query sampler per INNER
        // sub-proof, so a shape it cannot sample must be named here rather than
        // deep inside `epoch_program_publishing`.
        let inner_shapes: Vec<&super::epoch::TableChallengeShape> =
            e.tables.iter().map(|h| &h.shape).collect();
        assert_samplable(&format!("inner epoch {k}"), &inner_shapes);
        let program =
            super::epoch_tests::epoch_program_publishing(&e, true, Publishes::Aggregation);
        let arenas = super::epoch_tests::epoch_arena_words(&e, true);
        let artifacts =
            build_artifacts_counted(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
        let proved = lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
            .expect("the epoch wrap must prove at the aggregation preset");
        let layout = SchemaLayout::wrap(out_halves);
        layout.assert_covers(proved.public_words.len());
        layouts.push(layout);
        // The label is a pure function of the chain position, exactly as the
        // global proof's own AIR reconstruction derives it.
        labels.push([crate::tables::local_to_global::epoch_label(k as u64)]);
        children.push(real_child(artifacts, wrap_opts.clone(), &proved));
    }
    let label_refs: Vec<&[u64]> = labels.iter().map(|l| &l[..]).collect();
    let label_range = (labels[0][0], labels[FAN_IN - 1][0]);
    // S2: how many of each wrap's sub-proofs the node verifies at one-row
    // leaves (0 at the default format, all at `one_row = 1`, the AIR widths'
    // choice at `auto`). One parseable line per child for the box wrapper.
    for (k, c) in children.iter().enumerate() {
        let one_row = c
            .legs
            .iter()
            .filter(|l| l.verify.sub.layout.is_one_row())
            .count();
        println!(
            "ZFS2NODE child={k} {} sub_proofs={} one_row_legs={one_row}",
            crate::zf_format::ZfFormat::global().banner(),
            c.legs.len()
        );
    }
    println!(
        "   {FAN_IN} epoch wraps proved in {:.1}s, {} published words each, \
         {} sub-proofs each\n   RSS high-water AFTER the wrap proves: {:?} GiB",
        t.elapsed().as_secs_f64(),
        children[0].public_words.len(),
        children[0].tables.len(),
        super::wrap_tests::peak_rss_gib(),
    );

    // ---- the node. Same pre-flight on the CHILD side, so a zero bit width is
    // attributed to the wrap's own sub-proofs rather than to the inner epoch's.
    for (k, c) in children.iter().enumerate() {
        let shapes: Vec<&super::epoch::TableChallengeShape> =
            c.tables.iter().map(|h| &h.shape).collect();
        assert_samplable(&format!("child {k} (a wrap proof)"), &shapes);
    }
    let arenas: Vec<Vec<LfmWord>> = children.iter().flat_map(child_arena_words).collect();
    let node_layout = SchemaLayout::node(layouts[FAN_IN - 1].out_halves);

    // ---- ★ THE DIFFERENTIAL, on the same shape with the surface restored.
    //
    // Without this the only evidence that a leg derived its CHILD's challenges
    // is that the node executes — a leg on different challenges cannot
    // authenticate the child's walks, so execution implies agreement. That is
    // implication, and every other emitted verifier in this crate is held to a
    // value comparison against production's own replay. This is that comparison:
    // the pair each leg reaches, against the pair `verify_against_chunked`'s own
    // Phase A recovered host-side from the same proof.
    let t = Instant::now();
    let diagnostic = node_program(
        &children,
        &layouts,
        &label_refs,
        label_range,
        NodePublishSet::Diagnostic,
    );
    let exec_diag = execute(&diagnostic, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .expect("the diagnostic node must execute");
    let tail = node_layout.head + node_layout.schema_words();
    assert_eq!(
        exec_diag.public_words.len(),
        tail + 2 * FAN_IN,
        "the diagnostic node publishes the schema then one pair per child"
    );
    for (k, child) in children.iter().enumerate() {
        let got = |i: usize| {
            super::word::word_as_ext(&exec_diag.public_words[i].1).expect("an ext challenge")
        };
        assert_eq!(got(tail + 2 * k), child.z_alpha.0, "child {k}: the leg's z");
        assert_eq!(
            got(tail + 2 * k + 1),
            child.z_alpha.1,
            "child {k}: the leg's alpha"
        );
    }
    println!(
        "   ✓ differential: every leg reaches its child's OWN (z, alpha) \
         ({:.1}s, {} instructions)\n   RSS high-water AFTER the diagnostic arm: {:?} GiB",
        t.elapsed().as_secs_f64(),
        diagnostic.instrs.len(),
        super::wrap_tests::peak_rss_gib(),
    );

    // ---- the node a parent would verify.
    let t = Instant::now();
    let program = node_program(
        &children,
        &layouts,
        &label_refs,
        label_range,
        NodePublishSet::Aggregation,
    );
    println!(
        "   leaf node emitted in {:.1}s: {} instructions",
        t.elapsed().as_secs_f64(),
        program.instrs.len()
    );

    let t = Instant::now();
    let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .expect("★ the leaf node must execute");
    node_layout.assert_covers(exec.public_words.len());
    println!(
        "   ★ LEAF NODE EXECUTED in {:.1}s: {} published words (schema {} + head {})",
        t.elapsed().as_secs_f64(),
        exec.public_words.len(),
        node_layout.schema_words(),
        node_layout.head,
    );
    // ★ The reading the fan-in arithmetic needs. `peak_rss_gib` is `VmHWM`, a
    // PROCESS high-water mark that only ever rises, so the run's final figure
    // spans the wrap proves, the diagnostic arm and the node alike. Printing it
    // at each boundary turns one conflated number into a bound per phase: what
    // the node's own prove costs is at most the rise from here.
    println!(
        "   RSS high-water BEFORE the node prove: {:?} GiB",
        super::wrap_tests::peak_rss_gib()
    );

    // ---- the node's own proof, so the level above has something to verify.
    //
    // ⚠ THREE marks, not one. The mark above is taken BEFORE the artifact
    // build, which is not a bookkeeping call: it runs `lde_columns` +
    // `commit_lde_columns` over every chip group and builds the prep round — a
    // full commitment pass over the whole program. So a single "before the
    // prove" mark brackets the artifact build and the prove TOGETHER and cannot
    // say which of them costs what. These split it.
    // ⓘ This measurement runs ONE proof, so it is always a cache MISS and the
    // figure below is a real build — which is what it is here to price.
    let t = Instant::now();
    let artifacts = build_artifacts_counted(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
    println!(
        "   RSS high-water AFTER build_artifacts ({:.1}s): {:?} GiB",
        t.elapsed().as_secs_f64(),
        super::wrap_tests::peak_rss_gib()
    );
    let t = Instant::now();
    let proved =
        lfm_prove(&program, &artifacts, &arenas, &wrap_opts).expect("★ THE LEAF NODE MUST PROVE");
    let prove_secs = t.elapsed().as_secs_f64();
    println!(
        "   RSS high-water AFTER lfm_prove: {:?} GiB",
        super::wrap_tests::peak_rss_gib()
    );
    // ★ The census beside the measurement, so the two are never quoted apart.
    // The empty LFM machine costs 26,482,828 base-field-equivalent cells — the
    // 0-word row of `the_node_cost_model_is_measured` — so at fixture scale most
    // of a node's census is the machine's padding FLOOR rather than its
    // verification work, and a fixture node sits on a different part of the curve
    // from a production one.
    {
        let (main, aux) =
            super::airs::lfm_cell_counts_with_hasher(&program, crate::hash_pin::BLOCK_HASHER);
        let cells = main + 3 * aux;
        const EMPTY_MACHINE_CELLS: u64 = 26_482_828;
        println!(
            "   node census: {cells} cells, of which {} are the empty machine's \
             floor ({:.0}%) and {} are verification work",
            EMPTY_MACHINE_CELLS,
            100.0 * EMPTY_MACHINE_CELLS as f64 / cells as f64,
            cells.saturating_sub(EMPTY_MACHINE_CELLS),
        );
    }
    let t = Instant::now();
    assert!(
        super::proof::verify_against_artifacts(
            &artifacts,
            &proved.proof,
            &proved.public_words,
            &wrap_opts
        ),
        "the leaf node's proof must verify"
    );
    println!(
        "\n★ LEAF NODE PROVED AND VERIFIED\n   prove {prove_secs:.1}s\n   verify {:.2}s\n   \
         {} sub-proofs, {} published words\n   peak RSS {:?} GiB",
        t.elapsed().as_secs_f64(),
        proved.proof.proofs.len(),
        proved.public_words.len(),
        super::wrap_tests::peak_rss_gib(),
    );

    // ---- ONE TAMPER ARM PER BINDING LEG.
    //
    // Each moves a single published HALF in one child's publics arena — arena 0
    // of that child, eight halves per word — and nothing else. Both children's
    // proofs still verify on their own; what breaks is the relationship.
    let arena_of = |child: usize| -> usize {
        // Each child contributes `child_arena_words(child).len()` arenas, and its
        // publics arena is the first of them. Counted rather than assumed, so a
        // declaration-order change tampers the right arena or fails loudly.
        children[..child]
            .iter()
            .map(|c| child_arena_words(c).len())
            .sum()
    };
    let bump = |arenas: &mut Vec<Vec<LfmWord>>, arena: usize, word: usize| {
        let half = 8 * word;
        arenas[arena][half] =
            base_word(super::word::word_as_base(&arenas[arena][half]).expect("a half") + FE::one());
    };
    for (name, child, word) in [
        ("the register chain", 0usize, layouts[0].reg_fini(0)),
        ("the shared attestation id", 1usize, layouts[1].id(0)),
        ("the epoch label pin", 1usize, layouts[1].label(0)),
    ] {
        let mut tampered = arenas.clone();
        bump(&mut tampered, arena_of(child), word);
        assert!(
            execute(&program, &tampered, &crate::hash_pin::BLOCK_HASHER).is_err(),
            "moving {name} in child {child} must make the node unprovable"
        );
        println!("   ✓ tamper arm: {name} rejected");
    }
}

/// ★ A ZERO-BIT QUERY DRAW CONSUMES WHAT THE HOST CONSUMES.
///
/// # The defect this holds shut
///
/// A one-row trace at blowup 2 has a two-leaf LDE, so production's
/// `sample_query_indexes` calls `sample_u64(domain_size >> 1)` = `sample_u64(1)`
/// for every query of that table. Both host transcripts CONSUME before masking —
/// the byte arm draws one `next_sample_u64()`, the pinned algebraic arm squeezes
/// a cell — and return index 0. The emitted sampler refused `nbits = 0` outright,
/// so the epoch verifier could not be built over such an epoch at all; that is
/// what `the_leaf_node_verifies_and_binds_two_wraps` hit on box A, at inner epoch
/// 1's sub-proof #10.
///
/// # Why the obvious fix would have been the bug
///
/// "One index, so skip the sample" is wrong: the host does not skip it. An
/// emitter that skipped would be one squeeze short for that table and every
/// challenge after it in that fork would diverge.
///
/// # Why this test needs the follow-up draw
///
/// ⚠ A consumption desync here is invisible to a value differential ON the
/// table: every index is 0 whether the draw happened or not, and `iota_bits` is
/// the LAST thing `epoch::emit_table_challenges` samples, so nothing later in
/// that fork disagrees either. A green leaf test therefore proves nothing about
/// consumption — it proves only that no panic fired.
///
/// So this samples an extension element AFTER the zero-bit draw on both sides.
/// That element is the single observable that differs if the squeeze is missing,
/// and comparing it against the HOST's is what makes this emitter-host agreement
/// rather than emitter self-consistency.
///
/// ⓘ It lives here because lane A owns `transcript_replay.rs` only for this fix;
/// its natural home is beside the other transcript differentials.
#[test]
fn a_zero_bit_query_draw_consumes_what_the_host_does() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    const SEED: &[u8] = b"lane-A zero-bit query draw v0";

    // ---- the HOST, exactly as `sample_query_indexes` drives it at a one-row
    // table: one `sample_u64(1)`, then the next thing the transcript would give.
    let mut host = crate::hash_pin::block_transcript(SEED);
    let index = host.sample_u64(1);
    assert_eq!(index, 0, "a two-leaf domain has exactly one query index");
    let host_after: FEE = host.sample_field_element();

    // ---- the EMITTER, same seed, same sequence.
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let mut t = TranscriptReplay::new(SEED);
    let bits = t.sample_u64_pow2(&mut b, 0);
    assert!(bits.is_empty(), "a zero-bit draw yields no bits");
    let after = t.sample_ext(&mut b);
    b.public(after.as_cell());
    let program = compile(b.finish());
    let exec = execute(&program, &[], &crate::hash_pin::BLOCK_HASHER)
        .expect("a zero-bit query draw must emit and execute");

    assert_eq!(
        super::word::word_as_ext(&exec.public_words[0].1).expect("an ext"),
        host_after,
        "the draw AFTER a zero-bit query index must equal the host's — if the \
         emitter skipped the squeeze, this is the ONLY place it shows"
    );
}

// ============================== the stage cache ==============================

/// The named experiment, applied to every cached stage of a run.
///
/// ⛔ ONE mode decision, read once, applied everywhere. Two independent
/// cache-if-present decisions is how a harness comes to load one stage and prove
/// another, and the run then answers a question nobody asked: an unconditional
/// `A_BUNDLE` export once made a relaunch load a base an earlier run had saved,
/// so `prove_continuation` never ran and the run answered the CONTROL having been
/// launched as the TEST. `A_BUNDLE_MODE` names the mode; nothing infers it from
/// the filesystem.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum CacheMode {
    /// No caching: every stage proves. The default, and what CI runs.
    Off,
    /// Prove each stage and SAVE it. Refuses to overwrite an existing stage.
    Prove,
    /// LOAD each stage. Refuses to prove one whose cache is missing.
    Load,
}

impl CacheMode {
    /// `A_BUNDLE_MODE`, checked against whether a cache location was given.
    pub(super) fn from_env(located: bool) -> Self {
        match (located, std::env::var("A_BUNDLE_MODE").ok().as_deref()) {
            (false, _) => Self::Off,
            (true, None) => panic!(
                "a cache location is set but A_BUNDLE_MODE is not. Name the \
                 experiment — `prove` (prove each stage and save it) or `load` \
                 (load each saved stage) — so the harness cannot pick one from \
                 filesystem state"
            ),
            (true, Some("prove")) => Self::Prove,
            (true, Some("load")) => Self::Load,
            (true, Some(other)) => {
                panic!("A_BUNDLE_MODE must be `prove` or `load`, got `{other}`")
            }
        }
    }
}

/// ★ THE `prove` FIELD, SPLIT — `execute · fill · multi_prove`.
///
/// The per-node TIMING line prints the whole of `lfm_prove` as one number, and
/// the three phases inside it are three different machines: a single-threaded
/// interpreter, a parallel trace fill, and the only one that reaches the card.
/// A lever aimed at the wrong one reads zero.
///
/// ⓘ Printed HERE rather than beside the timers, for the stage's label — a
/// bare line is unattributable the moment two proofs are in flight. Silent on a
/// LOADED stage, because nothing proved: `take_prove_split` returns `None` and
/// there is nothing to attribute.
fn print_prove_split(label: &str) {
    if let Some(split) = super::proof::take_prove_split() {
        println!(
            "   {label} LFM PROVE: execute {:.2}s · fill {:.2}s · multi_prove {:.2}s{}",
            split.execute,
            split.fill,
            split.multi_prove,
            // ⓘ Absent when there was no wait, so a serial line is byte-identical
            // to every one already in the campaign's logs and the two arms diff
            // on their numbers rather than on their shape.
            if split.permit_wait > 0.0 {
                format!(" · permit wait {:.2}s", split.permit_wait)
            } else {
                String::new()
            },
        );
        // ⛔ A SECOND LINE, never extra fields on the first. Every log the
        // campaign has compared greps `LFM PROVE:` and reads its three numbers
        // positionally; widening that line would re-baseline every one of them.
        //
        // ⓘ `parallel 0 levels` beside a large `levels` is the reading this line
        // exists to make visible: it means every level was coalesced onto the
        // calling thread and the parallel path never ran, which a wall alone
        // would show only as a lever that did nothing.
        let e = split.exec;
        println!(
            "   {label} LFM EXEC: levels {} · parallel {} levels / {} hashes · \
             depth pass {:.2}s · setup {:.2}s · hash phase {:.2}s · apply {:.2}s · \
             residue {:.2}s · sum {:.2}s",
            e.levels,
            e.parallel_levels,
            e.parallel_hashes,
            e.depth_pass,
            e.setup,
            e.hash_phase,
            e.apply,
            e.residue,
            e.depth_pass + e.setup + e.hash_phase + e.apply + e.residue,
        );
        // ⓘ A THIRD line, for the two questions a wall cannot answer: how big
        // the thing `setup` touches is, and what width the pool actually gave.
        // `ns/perm` is measured on the coalesced levels of THIS proof — same
        // box, same load, no rayon — so `width` is a reading, not an argument.
        if e.serial_hashes > 0 {
            println!(
                "   {label} LFM EXEC WIDTH: records {:.0} MiB · {} serial hashes at \
                 {:.0} ns/perm · effective width {:.1}",
                e.record_bytes as f64 / (1u64 << 20) as f64,
                e.serial_hashes,
                e.secs_per_perm() * 1e9,
                e.effective_width(),
            );
        }
    }
}

/// Run `prove`, or substitute a cached proof, as `mode` says — and say which.
///
/// ★ WHY ONLY THE PROOF IS CACHED. A level's `LfmProof` is the one thing it
/// produces that cannot be re-derived cheaply: the program is a pure function of
/// its children's shapes and re-emits in seconds, and `build_artifacts` is a pure
/// function of the program. So the cache holds proofs and re-derives everything
/// else — which is also what makes a loaded arm's `L` the right number, because a
/// tree-builder carries the harvested children, not the emitters that made them.
///
/// ⇒ This is the precondition for pricing a LEVEL. Proving levels 0 and 1 in the
/// measuring process leaves their residue live and their peak already set, and a
/// `VmHWM` that never moves afterwards then reports a whole-process bound rather
/// than the level's cost. One arm per process is the fix, and skipping the
/// children's proves is what makes one arm per process affordable.
fn cached_stage(
    mode: CacheMode,
    path: Option<std::path::PathBuf>,
    label: &str,
    prove: impl FnOnce() -> super::proof::LfmProof,
) -> super::proof::LfmProof {
    type CachedProof = (super::proof::LfmStarkProof, Vec<(u32, LfmWord)>);
    let Some(p) = path.filter(|_| mode != CacheMode::Off) else {
        println!("   {label}: PROVED in-process, NOT cached");
        let proved = prove();
        print_prove_split(label);
        return proved;
    };
    match mode {
        CacheMode::Load => {
            assert!(
                p.exists(),
                "A_BUNDLE_MODE=load but {label}'s cache {} is absent: this would \
                 silently become a full prove, i.e. a different experiment",
                p.display()
            );
            let bytes = std::fs::read(&p).expect("the cached stage must read");
            let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(bytes.len());
            aligned.extend_from_slice(&bytes);
            let (proof, public_words) =
                rkyv::from_bytes::<CachedProof, rkyv::rancor::Error>(&aligned)
                    .expect("the cached stage must deserialize");
            println!(
                "   {label}: LOADED from {} ({} bytes) — NOT proved in this process",
                p.display(),
                bytes.len()
            );
            super::proof::LfmProof {
                proof,
                public_words,
            }
        }
        CacheMode::Prove => {
            assert!(
                !p.exists(),
                "A_BUNDLE_MODE=prove but {label}'s cache {} already exists: \
                 refusing to overwrite it, and refusing to silently load it \
                 instead",
                p.display()
            );
            let proved = prove();
            let cached: CachedProof = (proved.proof.clone(), proved.public_words.clone());
            let bytes =
                rkyv::to_bytes::<rkyv::rancor::Error>(&cached).expect("the stage must serialize");
            if let Some(dir) = p.parent() {
                std::fs::create_dir_all(dir).expect("the cache directory must exist");
            }
            std::fs::write(&p, &bytes).expect("the stage must persist");
            println!(
                "   {label}: PROVED and saved to {} ({} bytes)",
                p.display(),
                bytes.len()
            );
            print_prove_split(label);
            proved
        }
        CacheMode::Off => unreachable!("filtered above"),
    }
}

/// [`cached_stage`] for the base continuation, which is not an `LfmProof`.
///
/// ★ The pair of numbers a bundle cache buys is the measurement, not a
/// convenience: proving the base in-process leaves its residue live when a node
/// runs, so the node's measured carry-in is `L_children + L_base_residue`, while
/// a run that LOADS the bundle carries in `L_children` alone — which is what any
/// separately-written tree-builder would carry. The difference between the two IS
/// the base residue, and therefore how much of any high peak belongs to this
/// harness rather than to the tree. The deserialise path is production's own
/// (`bin/cli/src/main.rs:877-886`).
/// A stamped wall-clock window for a stretch no `TIMING` line covers.
///
/// ★★ WHY THESE EXIST. The pass-5 profile found **10.4 s at 3.8% GPU
/// utilisation** between the global L2G prove and the first wrap's device hold —
/// the largest contiguous idle block in the whole run — and NOTHING measured it.
/// It could only be decomposed by subtracting assumed terms, which is the exact
/// move this campaign keeps having to retract. Three stamps end that.
///
/// Reuses `LFM_CARD_TRACE` rather than adding a knob, because it is the same
/// job: hand an external sampler two wall-clock stamps to slice itself by.
fn stamped<T>(label: &str, f: impl FnOnce() -> T) -> T {
    if !super::device_permit::trace_enabled() {
        return f();
    }
    let t0 = stark::prove_split::epoch_secs();
    let t = std::time::Instant::now();
    let out = f();
    println!(
        "STAGE {label}: {:.2}s t=[{t0:.3},{:.3}]",
        t.elapsed().as_secs_f64(),
        stark::prove_split::epoch_secs(),
    );
    out
}

fn cached_bundle(
    mode: CacheMode,
    path: Option<std::path::PathBuf>,
    prove: impl FnOnce() -> crate::continuation::ContinuationProof,
) -> crate::continuation::ContinuationProof {
    let Some(p) = path.filter(|_| mode != CacheMode::Off) else {
        println!(
            "   base: PROVED in-process, NOT cached — carries in L_children + the base residue"
        );
        return prove();
    };
    match mode {
        CacheMode::Load => {
            assert!(
                p.exists(),
                "A_BUNDLE_MODE=load but {} does not exist: this would silently \
                 become a full base prove, i.e. a different experiment",
                p.display()
            );
            let bytes = std::fs::read(&p).expect("the cached bundle must read");
            let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(bytes.len());
            aligned.extend_from_slice(&bytes);
            let bundle = rkyv::from_bytes::<
                crate::continuation::ContinuationProof,
                rkyv::rancor::Error,
            >(&aligned)
            .expect("the cached bundle must deserialize");
            println!(
                "   base: LOADED from {} — carries in L_children ALONE, no base residue",
                p.display()
            );
            bundle
        }
        CacheMode::Prove => {
            assert!(
                !p.exists(),
                "A_BUNDLE_MODE=prove but {} already exists: refusing to overwrite \
                 a saved base, and refusing to silently load it instead",
                p.display()
            );
            let bundle = prove();
            // ⛔ HARNESS-ONLY, AND IT IS INSIDE THE `base:` NUMBER. Serialising
            // and writing a 19-epoch bundle is multi-second disk work that no
            // production pipeline does, and until this stamp existed it was
            // indistinguishable from proving.
            stamped("bundle cache write (HARNESS-ONLY)", || {
                let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&bundle)
                    .expect("the bundle must serialize");
                if let Some(dir) = p.parent() {
                    std::fs::create_dir_all(dir).expect("the cache directory must exist");
                }
                std::fs::write(&p, &bytes).expect("the bundle must persist");
            });
            println!(
                "   base: PROVED in-process and saved to {} — carries in \
                 L_children + the base residue",
                p.display()
            );
            bundle
        }
        CacheMode::Off => unreachable!("filtered above"),
    }
}

/// Where a stage of THIS run's tree caches, or `None` when `A_CACHE_DIR` is unset.
///
/// One directory, one file per stage, named for the stage rather than for the
/// run: a level-2 arm must load exactly the level-1 proofs an earlier arm saved,
/// and a name that encoded anything else would let two different trees share a
/// cache entry.
fn stage_path(dir: Option<&str>, stage: &str) -> Option<std::path::PathBuf> {
    dir.map(|d| std::path::Path::new(d).join(format!("{stage}.rkyv")))
}

/// Prove one aggregation node and hand it back as a CHILD of the next level.
///
/// The whole composition argument in one function: a node's proof is a plain
/// per-table `MultiProof`, so `real_child` reads it exactly as it reads a wrap's,
/// and the layout that describes it is `SchemaLayout::node`. Nothing about the
/// level appears here — which is what "the same emitter serves every level"
/// means operationally.
#[allow(clippy::too_many_arguments)]
fn prove_node_as_child(
    label: &str,
    children: &[RealChild],
    layouts: &[super::per_table_aggregator::SchemaLayout],
    labels: &[&[u64]],
    label_range: (u64, u64),
    out_halves: usize,
    opts: &crate::ProofOptions,
    mode: CacheMode,
    cache: Option<std::path::PathBuf>,
) -> (RealChild, super::per_table_aggregator::SchemaLayout) {
    use super::per_table_aggregator::NodePublishSet;

    let program = node_program(
        children,
        layouts,
        labels,
        label_range,
        NodePublishSet::Aggregation,
    );
    prove_node_program_as_child(label, &program, children, out_halves, opts, mode, cache)
}

/// [`prove_node_as_child`] for a caller that has ALREADY emitted the program.
///
/// ⛔ EMITTING IT TWICE IS NOT FREE. The tree driver builds a node's program to
/// take its census and its chip panel, and then called `prove_node_as_child`,
/// which emitted the identical program a second time — 7.4M instructions per
/// node, twice, at every one of 21 nodes. It was invisible because both halves
/// were correct and the only symptom was wall clock, which is exactly the
/// quantity we had agreed to treat as context.
///
/// ⇒ The census and the prove now share one program, and the split timings below
/// are what would have made the duplication visible in the first place: an
/// unattributed "60 s per node" cannot say whether it is emission, artifacts or
/// the prove.
#[allow(clippy::too_many_arguments)]
fn prove_node_program_as_child(
    label: &str,
    program: &LfmProgram,
    children: &[RealChild],
    out_halves: usize,
    opts: &crate::ProofOptions,
    mode: CacheMode,
    cache: Option<std::path::PathBuf>,
) -> (RealChild, super::per_table_aggregator::SchemaLayout) {
    use super::per_table_aggregator::SchemaLayout;
    use std::time::Instant;

    let wait_before = super::device_permit::waited_secs();
    let t = Instant::now();
    let arenas: Vec<Vec<LfmWord>> = children.iter().flat_map(child_arena_words).collect();
    let t_arenas = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let artifacts = super::program_census::build_artifacts_counted(
        program,
        opts,
        crate::hash_pin::BLOCK_HASHER,
    );
    let t_artifacts = t.elapsed().as_secs_f64();
    // ★ WHAT THE ARTIFACT BUILD SPENT QUEUEING rather than committing. Bracketed
    // and subtracted rather than taken-and-cleared: a counter that clears
    // couples every reader to every other one.
    let wait_artifacts = (super::device_permit::waited_secs() - wait_before).max(0.0);
    let wait_before = super::device_permit::waited_secs();
    let t = Instant::now();
    let proved = cached_stage(mode, cache, label, || {
        super::proof::lfm_prove(program, &artifacts, &arenas, opts)
            .expect("an aggregation node must prove")
    });
    let t_prove = t.elapsed().as_secs_f64();
    // ⛔ A LIVE MARK, NOT A HIGH-WATER — and within-run is not enough on its own.
    //
    // This line used to print `peak_rss_gib()`, a `VmHWM`. Read across leaf 0,
    // leaf 1 and an inner node it never moved, and the flatness was published as
    // "the tree does not grow per level". A high-water only rises: with the base
    // and four wraps already proved in the same process and no mark before leaf
    // 0, the reading bounds the WHOLE process and prices no level inside it —
    // level 1 included. Being a within-run comparison does not rescue it.
    // ⇒ The live figure plus `t=` is what prices a phase: `L` before, `L` after,
    // and an external sampler sliced to the window between the two stamps. One
    // arm per process (`A_CACHE_DIR` + `A_BUNDLE_MODE=load`) is what makes the
    // live figure mean the level rather than the run.
    println!("   {label}: {} instructions", program.instrs.len());
    mark(&format!("AFTER {label}"));
    let layout = SchemaLayout::node(out_halves);
    layout.assert_covers(proved.public_words.len());
    let t = Instant::now();
    let (child, t_verify) = real_child_timed(artifacts, opts.clone(), &proved);
    let t_harvest = t.elapsed().as_secs_f64();
    // ★ THE SPLIT, because "60 s per node" attributes nothing. Which half a
    // second cache layer would have to hold — the artifacts or the harvest —
    // is a different build depending on this line, and guessing it is how a
    // campaign builds the wrong cache.
    //
    // ★ And the harvest carries its OWN split, because the two halves belong to
    // different owners: `verify` is the harness's assert, `replay` is the
    // transcript walk a driver genuinely needs. Summed, the field prices a
    // phase no real pipeline has.
    // ★★ THE WAIT IS CONTAINED IN THE TWO FIELDS ABOVE, NOT ADDITIONAL TO THEM.
    // `build_artifacts` and `prove` are wall times and a queued worker's wait
    // sits inside whichever was running, which is why the same node reads 4.1 s
    // serial and 7.1 s at two siblings. Naming it here keeps the per-node
    // numbers comparable across arms AND closes the level's accounting: at K
    // workers the level's worker-seconds are K × wall, of which the permit's
    // `held` is the card, the waits are the queue, and the rest is host work.
    let wait_prove = (super::device_permit::waited_secs() - wait_before).max(0.0);
    let wait_total = wait_artifacts + wait_prove;
    println!(
        "   {label} TIMING: arenas {t_arenas:.1}s · build_artifacts {t_artifacts:.1}s \
         · prove {t_prove:.1}s · harvest {t_harvest:.1}s (verify {t_verify:.2} + replay {:.2}){}",
        t_harvest - t_verify,
        if wait_total > 0.0 {
            format!(
                " · permit wait {wait_total:.2}s (inside build_artifacts {wait_artifacts:.2} \
                 + prove {wait_prove:.2})"
            )
        } else {
            String::new()
        },
    );
    (child, layout)
}

/// ★ THE INNER NODE — a node whose children are NODE proofs.
///
/// # What this adds over the leaf gate
///
/// The leaf verifies wraps; this verifies leaves. Three things differ and each
/// is exercised here for the first time:
///
/// 1. **`SchemaLayout::node`** rather than `::wrap` — a different head (a node
///    has FAN_IN Phase A's and so no single `(z, α)`), a four-word label run
///    rather than two, and no trailing bus total.
/// 2. **A label RANGE per child**: each leaf carries the first and last label of
///    its subtree, and the inner node pins both ends of both. That is what makes
///    contiguity across sibling subtrees a consequence of the pins rather than a
///    separate check.
/// 3. **The L2G fold composing**: each leaf published a fold over ITS wraps'
///    roots, and the inner node folds those two folds. A single-child node folds
///    to identity and would not exercise `hash_pair` at this level, which is why
///    this needs two real leaves rather than one.
///
/// ⚠ Building this arm is what found the bug it now covers: a node used to
/// publish its fold as digest CELLS (four lanes in one word) while a wrap
/// publishes its root as lanes (one lane per word), and `emit_node_publishes`
/// reads `lanes[0]` of each published l2g word. A node child would have handed
/// ONE felt to `digest_from_lanes` where four are required. The fix removed the
/// asymmetry rather than parameterising it, so both layouts now read alike.
///
/// # Cost, and why the epoch requirement is asserted rather than worked around
///
/// A two-level tree at fan-in N needs N² epochs: N wraps per leaf, N leaves.
/// Padding the shortfall would mean a pad child with no epoch to belong to,
/// which breaks the register chain and the label pin — the same trade rejected
/// when the tree shape was priced. So this asserts the fixture is deep enough
/// and names the number if it is not.
#[test]
#[ignore = "box tier: proves FAN_IN^2 wraps, FAN_IN leaf nodes and one inner node"]
fn the_inner_node_verifies_two_leaf_nodes() {
    use super::epoch_tests::Publishes;
    use super::per_table_aggregator::{FAN_IN, SchemaLayout};
    use super::program_census::build_artifacts_counted;
    use super::proof::lfm_prove;
    use std::time::Instant;

    let elf_bytes = super::proof_fixture::read_inner_elf();
    let inner = super::proof_fixture::fixture_options();
    // ★ A LOCAL epoch size, NOT the shared `FIXTURE_EPOCH_LOG2`.
    //
    // A two-level tree needs FAN_IN^2 epochs and the shared constant selects two.
    // `epoch_size_log2` is a PARAMETER of `prove_continuation` (floor 2), so a
    // smaller epoch here yields more of them from the same guest without moving
    // the ground under the nine gates that already run against the shared
    // constant — every one of which would otherwise be re-baselined by a change
    // whose only purpose is to give THIS test more epochs.
    //
    // ⚠ Smaller epochs mean shallower tables, which is where the degenerate
    // shapes live. That is a feature: the one-row sub-proof and the one-leaf
    // Merkle tree were both found this way, and both are now gated in
    // milliseconds. If a third appears, it is a shape the emitter has to handle
    // and this is the cheapest place to find it.
    let epoch_log2 = super::proof_fixture::FIXTURE_EPOCH_LOG2 - 1;

    // ★★ ONE ARM PER PROCESS, which is what lets a LEVEL be priced.
    //
    // With `A_CACHE_DIR` set and `A_BUNDLE_MODE=load`, every stage below level 2
    // is read from disk instead of proved: the base, the four wraps and the two
    // leaf nodes. The inner node is then the only prove in the process, so `L`
    // before it is exactly what a tree-builder would carry — the bundle plus two
    // level-1 children — and the live marks around it price the level rather than
    // the run. An `A_BUNDLE_MODE=prove` arm writes that cache; the arms are
    // otherwise identical, and neither picks itself from filesystem state.
    //
    // ⚠ Loading a level-1 proof does NOT skip re-emitting its program: a node's
    // program is a pure function of its children's shapes and its `program_id`
    // must match the loaded proof, so the wrap harvests are still built (cheaply,
    // artifacts only) to re-emit the leaf programs. They are dropped before the
    // inner node proves, because they are not its children.
    let cache_dir = std::env::var("A_CACHE_DIR").ok();
    let mode = CacheMode::from_env(cache_dir.is_some());
    println!(
        "★ INNER NODE ARM: cache {mode:?}{}",
        match &cache_dir {
            Some(d) => format!(" in {d}"),
            None => String::new(),
        }
    );

    let bundle = cached_bundle(mode, stage_path(cache_dir.as_deref(), "bundle"), || {
        crate::continuation::prove_continuation(&elf_bytes, &[], epoch_log2, &inner)
            .expect("the fixture continuation must prove")
    });
    let needed = FAN_IN * FAN_IN;
    assert!(
        bundle.num_epochs() >= needed,
        "a two-level fan-in-{FAN_IN} tree needs {needed} epochs; at epoch_log2 \
         {epoch_log2} this guest gives {}. Lower `epoch_log2` further (its floor is \
         2) or use a longer guest — do NOT pad, a pad child belongs to no epoch \
         and breaks the register chain and the label pin",
        bundle.num_epochs()
    );

    let wrap_opts = super::proof::aggregation_wrap_options();
    let label_of = |k: usize| crate::tables::local_to_global::epoch_label(k as u64);

    // ---- level 0: one wrap per epoch, then level 1: one leaf per FAN_IN wraps.
    let t = Instant::now();
    let mut leaves = Vec::with_capacity(FAN_IN);
    let mut leaf_layouts = Vec::with_capacity(FAN_IN);
    let mut leaf_labels: Vec<[u64; 2]> = Vec::with_capacity(FAN_IN);
    let epoch_konsts = super::epoch_tests::EpochConstants::load(&elf_bytes, &inner, None)
        .expect("the inner ELF and its DECODE commitment must build once");
    for leaf in 0..FAN_IN {
        let mut wraps = Vec::with_capacity(FAN_IN);
        let mut wrap_layouts = Vec::with_capacity(FAN_IN);
        let mut wrap_labels: Vec<[u64; 1]> = Vec::with_capacity(FAN_IN);
        let mut out_halves = 0usize;
        for i in 0..FAN_IN {
            let k = leaf * FAN_IN + i;
            let e =
                super::epoch_tests::real_epoch_from_constants(&inner, &epoch_konsts, &bundle, k)
                    .expect("every epoch must reconstruct from proofs alone");
            out_halves = e.statement.public_output_len.div_ceil(4);
            let program =
                super::epoch_tests::epoch_program_publishing(&e, true, Publishes::Aggregation);
            let arenas = super::epoch_tests::epoch_arena_words(&e, true);
            let artifacts =
                build_artifacts_counted(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
            let proved = cached_stage(
                mode,
                stage_path(cache_dir.as_deref(), &format!("wrap-{leaf}-{i}")),
                &format!("wrap {k} (leaf {leaf}, child {i})"),
                || {
                    lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
                        .expect("the epoch wrap must prove")
                },
            );
            let layout = SchemaLayout::wrap(out_halves);
            layout.assert_covers(proved.public_words.len());
            wrap_layouts.push(layout);
            wrap_labels.push([label_of(k)]);
            wraps.push(real_child(artifacts, wrap_opts.clone(), &proved));
        }
        let refs: Vec<&[u64]> = wrap_labels.iter().map(|l| &l[..]).collect();
        let range = (
            label_of(leaf * FAN_IN),
            label_of(leaf * FAN_IN + FAN_IN - 1),
        );
        let (child, layout) = prove_node_as_child(
            &format!("leaf {leaf} (level 1)"),
            &wraps,
            &wrap_layouts,
            &refs,
            range,
            out_halves,
            &wrap_opts,
            mode,
            stage_path(cache_dir.as_deref(), &format!("leaf-{leaf}")),
        );
        println!(
            "   leaf {leaf}: {} published words, {} sub-proofs",
            child.public_words.len(),
            child.tables.len()
        );
        // ★ THE LEVEL-0 HARVESTS GO, and the trough is the point.
        //
        // A tree-builder proving level 2 holds its level-1 children and nothing
        // below them. Keeping the wraps alive here would inflate `L` by a whole
        // level that the real thing would have released, and a high-water could
        // not have shown the difference — only a live mark on either side of the
        // drop can. Emission borrowed their shapes; the leaf `RealChild` owns its
        // own artifacts, so nothing here is still borrowed.
        drop(wraps);
        drop(wrap_layouts);
        mark(&format!("AFTER dropping leaf {leaf}'s wrap harvests"));
        leaves.push(child);
        leaf_layouts.push(layout);
        leaf_labels.push([range.0, range.1]);
    }
    println!(
        "   {FAN_IN} leaf nodes over {needed} wraps in {:.1}s",
        t.elapsed().as_secs_f64()
    );

    // ---- level 2: the inner node, over NODE proofs.
    let refs: Vec<&[u64]> = leaf_labels.iter().map(|l| &l[..]).collect();
    let range = (leaf_labels[0][0], leaf_labels[FAN_IN - 1][1]);
    let out_halves = leaf_layouts[FAN_IN - 1].out_halves;
    let t = Instant::now();
    // ⛔ The inner node is NEVER cached — it is the measurement. `CacheMode::Off`
    // here regardless of the arm, so a `load` arm cannot accidentally read back a
    // level-2 proof and report the load as the level's cost.
    mark("BEFORE the inner node (this live figure is L for level 2)");
    let (inner_node, inner_layout) = prove_node_as_child(
        "the INNER node (level 2)",
        &leaves,
        &leaf_layouts,
        &refs,
        range,
        out_halves,
        &wrap_opts,
        CacheMode::Off,
        None,
    );
    println!(
        "\n★ INNER NODE PROVED AND VERIFIED (a node over {FAN_IN} NODE proofs)\n   \
         prove+harvest {:.1}s\n   {} published words, {} sub-proofs\n   \
         schema {} + head {}\n   peak RSS {:?} GiB",
        t.elapsed().as_secs_f64(),
        inner_node.public_words.len(),
        inner_node.tables.len(),
        inner_layout.schema_words(),
        inner_layout.head,
        super::wrap_tests::peak_rss_gib(),
    );

    // ---- the composition property, asserted rather than implied: an inner
    // node's published schema has the SAME shape as its children's, which is
    // what lets the level above it use the identical emitter.
    assert_eq!(
        inner_layout.total(),
        leaf_layouts[0].total(),
        "a node's published schema must not change with its level — that is what \
         makes the same emitter serve the level above"
    );
}

/// ★ A DEPTH-ZERO MERKLE WALK STILL BINDS THE LEAF TO THE ROOT.
///
/// # The shape
///
/// A one-row trace at blowup 2 has a two-leaf LDE, one row PAIR, and therefore a
/// Merkle tree with a single leaf and no levels — the leaf hash IS the root.
/// `SubProofShape::check` refused it outright (`merkle_depth >= 1`), which is
/// what `the_leaf_node_verifies_and_binds_two_wraps` hit once the query sampler
/// stopped refusing zero index bits.
///
/// # Why this is not "make the walk a no-op"
///
/// It is not a no-op and must not become one. Both sides do the same thing at
/// depth 0 and neither needs a special case:
///
/// - host `verify_merkle_path_from_leaf_hash` loops over an empty path and
///   returns `root_hash == hashed_value`;
/// - `emit_group_authentication` hashes the leaf, walks zero levels, and asserts
///   the result equals the committed root.
///
/// The compare is the entire binding, and it survives. **The rejection arm below
/// is what proves that** — if relaxing the shape check had let the walk skip its
/// root comparison, the honest arm would still pass and only this one would fail.
///
/// # Where the host enters
///
/// The root is not invented here: it is the leaf hash the emitter itself
/// computes, and `emit_leaf_hash`'s agreement with the host's backend is gated
/// separately by `algebraic_commit`'s leaf/parent differential. Composing the two
/// is what makes this emitter-versus-host rather than emitter-versus-itself, and
/// it is stated rather than assumed because the composition is the argument.
#[test]
fn a_depth_zero_walk_still_binds_leaf_to_root() {
    use super::sub_proof::{
        GroupCommitment, GroupOpening, GroupShape, emit_group_authentication, emit_leaf_hash,
    };

    const COLS: usize = 3;
    let shape = GroupShape {
        num_columns: COLS,
        is_ext: false,
    };
    let values: Vec<FE> = (0..shape.num_values() as u64)
        .map(|i| FE::from(7 * i + 1))
        .collect();

    // ---- the root, from the emitter's own leaf hash over those values.
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(shape.num_values() as u32);
    let cells: Vec<_> = (0..shape.num_values() as u32)
        .map(|i| b.hint_word(arena, i))
        .collect();
    let leaf = emit_leaf_hash(&mut b, shape, &cells);
    for cell in leaf.cells() {
        b.public(*cell);
    }
    let leaf_program = compile(b.finish());
    let leaf_arena: Vec<LfmWord> = values.iter().map(|v| base_word(*v)).collect();
    let leaf_exec = execute(
        &leaf_program,
        std::slice::from_ref(&leaf_arena),
        &crate::hash_pin::BLOCK_HASHER,
    )
    .expect("the leaf hash must execute");
    let root_words: Vec<LfmWord> = leaf_exec.public_words.iter().map(|(_, w)| *w).collect();

    // ---- the authentication at depth ZERO: no bits, no siblings.
    let build = || {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let a_vals = b.declare_arena(shape.num_values() as u32);
        let a_root = b.declare_arena(root_words.len() as u32);
        let vals: Vec<_> = (0..shape.num_values() as u32)
            .map(|i| b.hint_word(a_vals, i))
            .collect();
        let commitment = GroupCommitment::hint(&mut b, a_root, 0, shape);
        emit_group_authentication(
            &mut b,
            &commitment,
            &GroupOpening {
                values: vals,
                siblings: Vec::new(),
            },
            &[],
        );
        compile(b.finish())
    };
    let program = build();

    // ---- the honest arm: the leaf hash IS the root, so this must execute.
    execute(
        &program,
        &[leaf_arena.clone(), root_words.clone()],
        &crate::hash_pin::BLOCK_HASHER,
    )
    .expect("★ a one-leaf tree must authenticate against its own leaf hash");

    // ---- ★ THE REJECTION ARM — the one that proves the compare survives.
    let mut wrong = root_words.clone();
    wrong[0][0] += FE::one();
    assert!(
        execute(
            &program,
            &[leaf_arena, wrong],
            &crate::hash_pin::BLOCK_HASHER
        )
        .is_err(),
        "a depth-zero walk must still REJECT a root that is not the leaf hash — \
         if this passes, the walk stopped binding and the honest arm proves nothing"
    );
    println!(
        "   ✓ depth-0 walk: {} column pair binds to its root, and a moved root is rejected",
        COLS
    );
}

/// Both memory numbers at once — `(VmRSS, VmHWM)` in GiB, live and high-water.
///
/// ⚠ `VmHWM` alone cannot see a TROUGH, and the trough is half the model. A
/// high-water mark only rises, so a mark taken after a phase reports the largest
/// the process has EVER been, not what that phase left resident. That difference
/// decides whether the next phase's peak is `live + its own working set` or is
/// hidden under an earlier phase's mark entirely — and on this workload the two
/// diverge hard: box A measured `VmHWM` static at 72.50 GiB while `VmRSS`
/// oscillated between 19.98 and 58.13.
///
/// A `ps` sample of the LIVE figure once reached this lane as if it were a
/// high-water mark, and every derivation built on it was wrong by the gap
/// between them. Both are printed so that cannot recur.
fn rss_marks() -> (Option<f64>, Option<f64>) {
    let read = |key: &str| -> Option<f64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|l| l.starts_with(key))?;
        let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb / (1024.0 * 1024.0))
    };
    (read("VmRSS:"), read("VmHWM:"))
}

/// One labelled mark: `live` is what the next phase carries in, `high-water` is
/// what the process has ever held, `t` is the wall clock the sampler shares.
///
/// ⚠ `t` is UNIX-epoch seconds, not elapsed. An external VRAM sampler
/// (`nvidia-smi` at 10 Hz) has no view of this process's phases, so a shared
/// clock is the only thing that lets its trace be sliced to the NODE window —
/// without it a whole-run GPU peak gets attributed to whichever phase the reader
/// assumes, which is the same masking that made a process high-water read as one
/// tree level's cost.
fn mark(label: &str) {
    let (rss, hwm) = rss_marks();
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    println!("   MARK {label}: live {rss:?} GiB / high-water {hwm:?} GiB / t={t:.1}");
}

/// ★ THE TREE'S SHAPE COMES FROM THE EPOCH COUNT, not a constant.
///
/// Pure arithmetic — no proving, so it runs on every suite. It exists because
/// the shape was planned more than once against a number that was not the
/// block's. Block 25368371 is **23,935,913** cycles: 6 epochs at 2^22, **12 at
/// 2^21**. The retired **74,819,518** — the July prebuilt guest, ~3x dearer at
/// identical work — gives 18 and 36 instead, i.e. a tree two levels too deep. A
/// tree built to a constant answers whichever question that constant came from.
///
/// ⚠ **2^21 is the posture of record.** 2^22 does not fit a 32 GiB card at
/// blowup 4 — the resident per-table LDE + snapshot + tree roughly doubles from
/// 2^21 while the card does not, and it fails even at one table in flight. The
/// 2^22 rows below are kept as a CONTROL on the arithmetic, not as a posture.
///
/// The arities are asserted rather than the level count alone, because the
/// LEFTOVER RULE is the design decision: every node takes `1..=fan_in` children
/// and an odd level ends in a short node rather than carrying a proof upward to
/// a parent with mixed-shape children. Asserting only the depth would pass under
/// either rule.
#[test]
fn the_tree_shape_matches_the_epoch_count() {
    use super::per_table_aggregator::{tree_node_count, tree_shape};

    // The real block, both postures. Epoch counts are `ceil(cycles / 2^k)` for
    // block 25368371's **23,935,913** cycles: the guest built with fat LTO over
    // one codegen unit, the runtime's own memcpy and the Keccak final block
    // absorbed without its buffer.
    //
    // ⚠ THREE stale figures now, and they went stale the same way.
    //
    // 74,819,518 is the JULY prebuilt guest; thin-LTO and the accelerators made
    // it ~47% cheaper at identical work — same block, same keccak 10,478, same
    // ECSM 116. The stale figure still sits in a table in
    // `real_block_benchmark_selection.md`, with the correction in an addendum
    // BELOW it. Reading the table row and stopping gives 18/36 epochs and a
    // tree two levels too deep.
    //
    // 39,631,559 is the 2026-09-07 guest, and it is what THIS comment said
    // until #894. That bump cost another ~23% at identical work, so the 2^21
    // posture went from 19 epochs to 15 and the 2^22 posture from 10 to 8 —
    // one whole level off the fan-in-2 tree in both.
    //
    // 30,498,818 is the #894 guest. The guest build that followed (fat LTO,
    // the runtime memcpy, the leaner Keccak final block) cost 21.5% less at
    // identical work: 15 epochs became 12 at 2^21 and 8 became 6 at 2^22.
    // Nothing failed when either changed: every number below is a literal, so
    // the test went on passing while describing a block this repository no
    // longer proves. Any epoch count, tree shape or IDENTITY-line count derived
    // from an old figure is stale, and that includes saved run logs.
    for (epochs, fan_in, levels, nodes) in [
        (6usize, 2usize, 3usize, 6usize), // 2^22: 6 -> 3 -> 2 -> 1
        (6, 3, 2, 3),                     // 2^22: 6 -> 2 -> 1
        (12, 2, 4, 12),                   // 2^21: 12 -> 6 -> 3 -> 2 -> 1
        (12, 3, 3, 7),                    // 2^21: 12 -> 4 -> 2 -> 1
        (12, 4, 2, 4),                    // 2^21, fan-in 4 (the WHIR tree's opt-out): 12 -> 3 -> 1
        (12, 5, 2, 4), // 2^21, pure WHIR's default (the WHIR tree only): 12 -> 3 -> 1
    ] {
        let shape = tree_shape(epochs, fan_in);
        assert_eq!(
            shape.len(),
            levels,
            "levels at {epochs} epochs, fan-in {fan_in}"
        );
        assert_eq!(
            tree_node_count(&shape),
            nodes,
            "aggregator proofs at {epochs} epochs, fan-in {fan_in}"
        );
        // Every node takes 1..=fan_in, and each level consumes exactly what the
        // one below produced — the invariant a carried leftover would break.
        let mut below = epochs;
        for (i, level) in shape.iter().enumerate() {
            assert!(
                level.arities.iter().all(|a| (1..=fan_in).contains(a)),
                "level {i} has an arity outside 1..={fan_in}: {:?}",
                level.arities
            );
            assert_eq!(
                level.arities.iter().sum::<usize>(),
                below,
                "level {i} must consume exactly the {below} proofs beneath it"
            );
            below = level.nodes();
        }
        assert_eq!(below, 1, "the tree must close to a single root proof");
    }

    // A degenerate block is a legal shape, not an error: one epoch is already
    // the root and the interior is empty.
    assert!(tree_shape(1, 2).is_empty(), "one epoch needs no interior");

    // ★ The leftover rule is EXERCISED, not merely permitted: a shape that
    // never produced a short node would pass every assertion above while
    // testing nothing about wrap-versus-carry.
    let short_levels = |epochs: usize, fan_in: usize| -> usize {
        tree_shape(epochs, fan_in)
            .iter()
            .filter(|l| l.arities.iter().any(|a| *a < fan_in))
            .count()
    };

    // The shipped posture. ⚠ It is WEAKER than it was: at 19 epochs this was 3
    // short levels (19, 5 and 3 all left one over), and at 12 it is 1 — the
    // 3 -> 2 level — because 12 and 6 are even. So the real block no longer
    // covers this rule on its own.
    assert_eq!(
        short_levels(12, 2),
        1,
        "12 at fan-in 2 must still exercise the leftover rule at least once"
    );
    assert_eq!(
        short_levels(8, 2),
        0,
        "8 at fan-in 2 is a perfect binary tree — recorded so the row below is \
         understood as necessary rather than redundant"
    );

    // ⇒ A SYNTHETIC row carries the coverage the guest bump took away. 19 is
    // kept as the shape, not because any block has 19 epochs now, but because
    // it is the smallest one that makes three separate levels carry a leftover,
    // which is what distinguishes wrapping from carrying.
    assert_eq!(
        short_levels(19, 2),
        3,
        "the synthetic shape must exercise the leftover rule on three levels"
    );
    assert!(
        short_levels(12, 3) >= 2,
        "fan-in 3 at the shipped epoch count must leave a short node on at \
         least two levels"
    );
    assert_eq!(
        short_levels(12, 4),
        1,
        "fan-in 4 fills level 1 (4 / 4 / 4); its short node is the root's level (3)"
    );
    assert_eq!(
        short_levels(12, 5),
        2,
        "pure WHIR's default leaves a short node at level 1 (5 / 5 / 2) and at the root's level (3)"
    );
}

// ======================= the tree's in-order pool =======================

pub(super) use super::tree_run::in_index_order;

/// ★★★ THE ORDERING GATE, with the completion order FORCED to the reverse of
/// the index order.
///
/// Index `j` sleeps for `(n - j)` ticks, so index 0 finishes LAST and index
/// `n-1` first. A drain in completion order returns the reverse; a drain by
/// index returns what the serial loop would have. Without the forcing this test
/// would pass on a scheduler that happens to finish in order, which is the
/// failure mode the permit's own first test had.
///
/// This is the assertion behind "scheduling is invisible to the bytes": a
/// level's children, layouts and labels are three parallel vectors and the
/// level above takes contiguous subslices of each, so a permuted drain builds a
/// different tree whose label pins still verify one subtree at a time.
#[test]
fn in_index_order_returns_index_order_when_completion_is_reversed() {
    const N: usize = 8;
    let finished: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let out = in_index_order(N, 4, |j| {
        std::thread::sleep(std::time::Duration::from_millis(20 * (N - j) as u64));
        finished.lock().expect("the order log").push(j);
        j * 10
    });
    assert_eq!(
        out,
        (0..N).map(|j| j * 10).collect::<Vec<_>>(),
        "results must arrive in INDEX order"
    );
    let finished = finished.into_inner().expect("the order log");
    assert_ne!(
        finished,
        (0..N).collect::<Vec<_>>(),
        "the test did not force a reordering, so it proved nothing: {finished:?}"
    );
}

/// One worker, and the same answer — the control arm runs `task` inline and
/// must not be a different computation from the parallel one.
#[test]
fn in_index_order_with_one_worker_matches_the_parallel_result() {
    let serial = in_index_order(6, 1, |j| j * j);
    let parallel = in_index_order(6, 3, |j| j * j);
    assert_eq!(serial, parallel);
}

/// ⚠ A worker panic must arrive with its MESSAGE, not as "a scoped thread
/// panicked". Eleven anonymous failures in one box suite run is what the
/// payload capture costs to omit.
#[test]
fn in_index_order_re_raises_a_worker_panic_with_its_message() {
    let caught = std::panic::catch_unwind(|| {
        in_index_order(4, 2, |j| {
            assert_ne!(j, 2, "NODE 2 SAYS SO");
            j
        })
    });
    let payload = caught.expect_err("a panicking task must panic the caller");
    let msg = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("<not a string>");
    assert!(
        msg.contains("NODE 2 SAYS SO"),
        "the worker's own message must survive, got: {msg}"
    );
}
