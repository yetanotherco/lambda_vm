//! The emitted verifier of a continuation's cross-epoch GLOBAL memory proof.
//!
//! The global proof is per-table: one sub-proof per epoch's L2G re-commit plus
//! one GLOBAL_MEMORY table per touched page, behind one statement, closing the
//! GlobalMemory bus at ZERO. This module harvests a real fixture bundle's
//! global proof ([`real_global`]), emits its verifier
//! ([`global_verifier_program`]) and fills the arenas that program declares
//! ([`global_arena_words`]).
//!
//! It is the per-table verification path at `WrapHash::production()` against a
//! real per-table `MultiProof`, differentialled against the harvest's own
//! production challenges through the published shared pair and tampered through
//! a flipped L2G main root. Every emission primitive it drives — the spine's
//! `fork_table`, the per-table verification legs, the LogUp closure — is the
//! same machinery an aggregator over per-table wrap proofs needs, which is why
//! the leg is worth gating on its own rather than only inside a larger program.

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::{Ext, Felt, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::epoch::RootCells;
use super::executor::execute;
use super::instr::ArenaId;
use super::transcript_replay::TranscriptReplay;
use super::word::{LfmWord, base_word, ext_word};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

/// The cross-epoch global memory proof, production-accepted, harvested for
/// emission: per-table shapes and challenges (the per-table machinery's own
/// harvest), the Phase-A prep constants (page genesis commitments — AIR-set
/// constants at emit time), and the statement bytes (every field an
/// emit-time constant of the block).
pub(super) struct RealGlobal {
    /// ⚠ ONE ENTRY PER HOST `append_bytes` CALL, not one flat run.
    /// `absorb_continuation_global_statement` makes a separate call for the
    /// tag, the ELF digest, the epoch count, the private-page count, the FRI
    /// byte, the page-base count and each page base. A byte transcript
    /// concatenates and cannot tell one long field from that sequence; an
    /// ALGEBRAIC one length-prefixes every call and can, so a flattened
    /// statement is a different chain.
    pub(super) statement_appends: Vec<Vec<u8>>,
    pub(super) tables: Vec<super::epoch_tests::HostTable>,
    pub(super) legs: Vec<super::epoch_verify_tests::TableLegs>,
    pub(super) num_l2g: usize,
    pub(super) z_alpha: (FEE, FEE),
}

/// Harvest the bundle's global proof. Panics loudly on a proof production
/// rejects. Mirrors `verify_global`'s AIR reconstruction exactly (the
/// no-supplied-roots arm: data-page genesis recomputed from the ELF).
pub(super) fn real_global(
    elf_bytes: &[u8],
    bundle: &crate::continuation::ContinuationProof,
    opts: &crate::ProofOptions,
) -> RealGlobal {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use executor::elf::Elf;
    use stark::verifier::IsStarkVerifier;

    let elf = Elf::load(elf_bytes).expect("the ELF must load");
    let num_epochs = bundle.num_epochs();
    let npriv = bundle.num_private_pages();
    let page_bases: Vec<u64> = {
        let mut b: Vec<u64> = bundle.touched_pages().to_vec();
        b.sort_unstable();
        b.dedup();
        b
    };
    let l2g_airs: Vec<_> = (0..num_epochs)
        .map(|i| {
            crate::continuation::l2g_global_air(
                opts,
                crate::tables::local_to_global::epoch_label(i as u64),
            )
        })
        .collect();
    let gm_configs = crate::continuation::global_memory_configs(&page_bases, &elf, npriv);
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|config| crate::continuation::global_memory_air(opts, config, None))
        .collect();
    let mut refs: Vec<
        &dyn stark::traits::AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    > = l2g_airs
        .iter()
        .map(|a| a as &dyn stark::traits::AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>)
        .collect();
    for air in &gm_airs {
        refs.push(air);
    }

    // The statement, byte for byte — `absorb_continuation_global_statement`'s
    // encoding over emit-time constants, pinned by the harness differential
    // (the seed below absorbs through the production function; the leg's
    // emitted challenges must then match the harvested ones, which fails if
    // this local encoding ever drifts).
    let mut statement_appends: Vec<Vec<u8>> = vec![
        crate::statement::CONTINUATION_GLOBAL_TAG.to_vec(),
        crate::statement::elf_digest(elf_bytes).to_vec(),
        (num_epochs as u64).to_le_bytes().to_vec(),
        (npriv as u64).to_le_bytes().to_vec(),
        vec![opts.fri_final_poly_log_degree],
        (page_bases.len() as u64).to_le_bytes().to_vec(),
    ];
    for base in &page_bases {
        statement_appends.push(u64::to_le_bytes(*base).to_vec());
    }

    let seed = || {
        let mut t = crate::hash_pin::block_transcript(&[]);
        crate::statement::absorb_continuation_global_statement(
            &mut t,
            elf_bytes,
            num_epochs,
            npriv,
            opts.fri_final_poly_log_degree,
            &page_bases,
        );
        t
    };
    let view = bundle.global_proof_view();
    assert_eq!(refs.len(), view.len(), "one AIR per global sub-proof");
    assert!(
        crate::hash_pin::BlockVerifier::<Gl, Ext3, ()>::multi_verify_views(
            &refs,
            view,
            &mut seed(),
            &FEE::zero()
        ),
        "production's verifier must accept the global proof"
    );

    // Phase A + the shared LogUp pair, transcribed as the epoch harvest does.
    let mut transcript = seed();
    for (idx, air) in refs.iter().enumerate() {
        let v = view.get(idx);
        if air.is_preprocessed() {
            transcript.append_bytes(&air.precomputed_commitment());
        }
        transcript.append_bytes(v.lde_trace_main_merkle_root());
    }
    let lookup: Vec<FEE> = (0..stark::lookup::LOGUP_NUM_CHALLENGES)
        .map(|_| transcript.sample_field_element())
        .collect();
    let z_alpha = (lookup[0], lookup[1]);

    let num_tables = refs.len();
    let tables: Vec<super::epoch_tests::HostTable> = refs
        .iter()
        .enumerate()
        .map(|(idx, air)| {
            let v = view.get(idx);
            let mut fork = transcript.clone();
            if num_tables > 1 {
                fork.append_bytes(&(idx as u64).to_le_bytes());
            }
            if let Some(root) = v.lde_trace_aux_merkle_root() {
                fork.append_bytes(root);
            }
            if let Some(c) = v.bus_table_contribution() {
                fork.append_field_element(&c);
            }
            super::epoch_tests::host_table_forked(*air, v, idx, num_tables, &mut fork, &lookup)
        })
        .collect();
    let legs = refs
        .iter()
        .enumerate()
        .map(|(idx, air)| super::epoch_verify_tests::build_table_legs(*air, view.get(idx), &lookup))
        .collect();

    RealGlobal {
        statement_appends,
        tables,
        legs,
        num_l2g: num_epochs,
        z_alpha,
    }
}

/// Per-table arena set of the global-verifier program, in declaration order.
struct GlobalTableArenas {
    aux_root: Option<ArenaId>,
    contribution: Option<ArenaId>,
    composition_root: ArenaId,
    ood_current: ArenaId,
    ood_next: ArenaId,
    parts: ArenaId,
    fri_roots: ArenaId,
    fri_coeffs: ArenaId,
    nonce: Option<ArenaId>,
    legs: super::epoch_verify::TableQueryArenas,
}

/// The emitted verifier of the global proof — the per-table program's own
/// structure (statement, Phase A, one fork per table, full verification
/// legs, the LogUp closure) with the global statement as one constant run,
/// every preprocessed root an AIR-set constant, and the bus target ZERO
/// (`verify_global`'s own expected balance). PUBLISHES: the shared pair,
/// then each epoch's L2G re-commit main root (eight halves each, epoch
/// order) — the byte-compare material the aggregator binds against the five
/// wraps' published carved roots.
pub(super) fn global_verifier_program(g: &RealGlobal) -> LfmProgram {
    use super::epoch::{TableAbsorbs, fork_table};
    use super::statement_replay::{PhaseAPreprocessed, PhaseATable, replay_phase_a};

    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let n = g.tables.len();

    // ---- arenas, declaration order = absorb order ----
    let a_main_roots = b.declare_arena(super::edsl::digest_words(&b) * n as u32);
    let per_table: Vec<GlobalTableArenas> = g
        .tables
        .iter()
        .zip(&g.legs)
        .map(|(h, leg)| GlobalTableArenas {
            aux_root: h
                .shape
                .has_aux_root
                .then(|| b.declare_arena(super::edsl::digest_words(&b))),
            contribution: h.shape.has_contribution.then(|| b.declare_arena(1)),
            composition_root: b.declare_arena(super::edsl::digest_words(&b)),
            ood_current: b
                .declare_arena((h.shape.ood_current_dims.0 * h.shape.ood_current_dims.1) as u32),
            ood_next: b.declare_arena((h.shape.ood_next_dims.0 * h.shape.ood_next_dims.1) as u32),
            parts: b.declare_arena(h.shape.num_parts as u32),
            fri_roots: b
                .declare_arena(super::edsl::digest_words(&b) * h.shape.fri.num_committed() as u32),
            fri_coeffs: b.declare_arena(h.shape.fri.num_terminal_coeffs() as u32),
            nonce: (h.shape.grinding_factor > 0).then(|| b.declare_arena(1)),
            legs: super::epoch_verify::declare_table_arenas(&mut b, &leg.verify),
        })
        .collect();

    // ---- the statement: ONE APPEND PER HOST CALL, see `statement_appends` ----
    let mut t = TranscriptReplay::new(&[]);
    for append in &g.statement_appends {
        t.append_const_bytes(append);
    }

    // ---- Phase A: prep constants, hinted main roots ----
    let main_cells: Vec<RootCells> = (0..n)
        .map(|i| {
            RootCells::hint(
                &mut b,
                a_main_roots,
                super::proof_arena::words_per_root() as u32 * i as u32,
            )
        })
        .collect();
    // ⚠ `byte_halves`, not `lanes_flat`: Phase A absorbs a root through
    // `append_halves_misaligned`, whose byte length is `4 · halves.len()`, and
    // the host absorbs the root's THIRTY-TWO bytes in one `append_bytes`. On an
    // algebraic arm `lanes_flat` is four FULL FELTS, so that call would declare
    // sixteen bytes where the host declared thirty-two — a different length
    // ⚠ The DIGEST's felts, not the root's bytes. `replay_phase_a` absorbs
    // through `absorb_root_felts`, which declares the host's 32 bytes on both
    // arms and packs the algebraic arm's four felts into the one digest cell
    // they already are — so the root absorb CANCELS there rather than paying a
    // byte regrouping. `byte_halves` is for `program_id`, which is deliberately
    // keccak-over-bytes; handing it here would regroup felts the host never
    // serialised.
    let main_halves: Vec<Vec<Felt>> = main_cells.iter().map(RootCells::lanes_flat).collect();
    let prep_cells: Vec<Option<RootCells>> = g
        .tables
        .iter()
        .map(|h| {
            h.precomputed_root
                .as_ref()
                .map(|c| RootCells::constant(&mut b, c))
        })
        .collect();
    let phase_a: Vec<PhaseATable> = g
        .tables
        .iter()
        .enumerate()
        .map(|(i, h)| PhaseATable {
            preprocessed_root: h
                .precomputed_root
                .as_ref()
                .map(PhaseAPreprocessed::Constant),
            main_root: &main_halves[i][..],
        })
        .collect();
    let (z, alpha) = replay_phase_a(&mut t, &mut b, &phase_a);
    b.public(z.as_cell());
    b.public(alpha.as_cell());
    // The aggregator's byte-compare material: each epoch's L2G re-commit
    // root, the very cells Phase A absorbed.
    for cells in main_cells.iter().take(g.num_l2g) {
        for half in cells.lanes_flat() {
            b.public(half.as_cell());
        }
    }

    // ---- one fork per table, with the full verification legs ----
    let mut contributions: Vec<Ext> = Vec::new();
    for (i, h) in g.tables.iter().enumerate() {
        let a = &per_table[i];
        let aux = a.aux_root.map(|id| RootCells::hint(&mut b, id, 0));
        let contribution = a.contribution.map(|id| b.hint_word(id, 0).as_ext());
        let composition = RootCells::hint(&mut b, a.composition_root, 0);
        let ood_current: Vec<Ext> = (0..(h.shape.ood_current_dims.0 * h.shape.ood_current_dims.1)
            as u32)
            .map(|k| b.hint_word(a.ood_current, k).as_ext())
            .collect();
        let ood_next: Vec<Ext> = (0..(h.shape.ood_next_dims.0 * h.shape.ood_next_dims.1) as u32)
            .map(|k| b.hint_word(a.ood_next, k).as_ext())
            .collect();
        let parts: Vec<Ext> = (0..h.shape.num_parts as u32)
            .map(|k| b.hint_word(a.parts, k).as_ext())
            .collect();
        let fri_roots: Vec<RootCells> = (0..h.shape.fri.num_committed())
            .map(|k| {
                RootCells::hint(
                    &mut b,
                    a.fri_roots,
                    super::proof_arena::words_per_root() as u32 * k as u32,
                )
            })
            .collect();
        let fri_coeffs: Vec<Ext> = (0..h.shape.fri.num_terminal_coeffs() as u32)
            .map(|k| b.hint_word(a.fri_coeffs, k).as_ext())
            .collect();
        let nonce = a.nonce.map(|id| b.hint_felt(id, 0));
        if let Some(c) = contribution {
            contributions.push(c);
        }
        let mut fork = fork_table(&t, h.shape.index, h.shape.num_tables);
        let absorbs = TableAbsorbs {
            aux_root: aux.as_ref(),
            contribution,
            composition_root: &composition,
            ood_current: &ood_current,
            ood_next: &ood_next,
            parts: &parts,
            fri_roots: &fri_roots,
            fri_coeffs: &fri_coeffs,
            nonce,
        };
        let ch = super::epoch::emit_table_challenges(&mut b, &mut fork, &h.shape, &absorbs);
        let leg = &g.legs[i];
        super::epoch_verify::emit_table_verification(
            &mut b,
            &leg.verify,
            &leg.analysis,
            &ch,
            &absorbs,
            &super::epoch_verify::TableInputs {
                precomputed_root: prep_cells[i].as_ref(),
                main_root: &main_cells[i],
                rap_challenges: &[z, alpha],
            },
            &a.legs,
        );
    }

    // ---- the closure: the global bus balances to ZERO ----
    let shape = super::logup::LogUpShape {
        num_contributing_tables: contributions.len(),
        num_output_bytes: 0,
    };
    let target = b.ext_const(&FEE::zero());
    super::logup::emit_bus_closure(&mut b, &shape, &contributions, target);

    compile(b.finish())
}

/// The global program's arenas, in its declaration order.
pub(super) fn global_arena_words(g: &RealGlobal) -> Vec<Vec<LfmWord>> {
    let mut arenas: Vec<Vec<LfmWord>> = Vec::new();
    arenas.push(super::proof_arena::commitments_to_arena(
        &g.tables.iter().map(|h| h.main_root).collect::<Vec<_>>(),
    ));
    for (h, leg) in g.tables.iter().zip(&g.legs) {
        if let Some(root) = &h.aux_root {
            arenas.push(super::proof_arena::commitments_to_arena(&[*root]));
        }
        if let Some(c) = &h.contribution {
            arenas.push(vec![ext_word(c)]);
        }
        arenas.push(super::proof_arena::commitments_to_arena(&[
            h.composition_root
        ]));
        arenas.push(h.ood_current.iter().map(ext_word).collect());
        arenas.push(h.ood_next.iter().map(ext_word).collect());
        arenas.push(h.parts.iter().map(ext_word).collect());
        arenas.push(super::proof_arena::commitments_to_arena(&h.fri_roots));
        arenas.push(h.fri_coeffs.iter().map(ext_word).collect());
        if let Some(nonce) = h.nonce {
            arenas.push(vec![base_word(FE::from(nonce))]);
        }
        arenas.push(leg.opening_arena());
        arenas.push(leg.fri_arena());
    }
    arenas
}

/// ★ THE GLOBAL LEG RUNS: the emitted verifier of a REAL fixture bundle's
/// cross-epoch global proof — per-table verification of the L2G re-commits
/// and one GLOBAL_MEMORY table per touched page behind one constant-run
/// statement, closing the GlobalMemory bus at ZERO — and publishes each
/// epoch's L2G re-commit root. Differentialled against the harvest's own
/// production challenges via the published pair; tampered via a flipped
/// L2G main root (Phase A absorbs it, so the walk cannot reach it).
#[test]
fn the_global_verifier_leg_runs_and_rejects_tampers() {
    let elf_bytes = super::proof_fixture::read_inner_elf();
    let inner = super::proof_fixture::fixture_options();
    let bundle = crate::continuation::prove_continuation(
        &elf_bytes,
        &[],
        super::proof_fixture::FIXTURE_EPOCH_LOG2,
        &inner,
    )
    .expect("the fixture continuation must prove");
    let g = real_global(&elf_bytes, &bundle, &inner);
    let program = global_verifier_program(&g);
    let arenas = global_arena_words(&g);
    let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .expect("the global leg must execute");

    let pub_ext = |i: usize| super::word::word_as_ext(&exec.public_words[i].1).expect("an ext");
    assert_eq!(pub_ext(0), g.z_alpha.0, "the global z");
    assert_eq!(pub_ext(1), g.z_alpha.1, "the global alpha");
    // The published L2G re-commit roots equal the harvested main roots.
    //
    // ⚠ Compared through `proof_arena::commitment_lanes`, NOT by re-spelling
    // the byte rendering here. The program publishes `RootCells::lanes_flat`,
    // which is `u32` halves on a byte hash and FULL FELTS on an algebraic one;
    // `commitment_lanes` is the flattened `commitment_words` the arena was
    // written from, so the two agree by construction on either arm instead of
    // this test carrying a second copy of one arm's layout.
    let l2g_lanes = super::proof_arena::lanes_per_root();
    for k in 0..g.num_l2g {
        let want = super::proof_arena::commitment_lanes(&g.tables[k].main_root);
        assert_eq!(want.len(), l2g_lanes, "a root's published lane count");
        for (h, want) in want.into_iter().enumerate() {
            let got = super::word::word_as_base(&exec.public_words[2 + l2g_lanes * k + h].1)
                .expect("a root lane");
            assert_eq!(got, want, "L2G root {k} lane {h}");
        }
    }
    println!(
        "★ global leg: {} tables ({} L2G + {} pages), {} instructions, {} published words",
        g.tables.len(),
        g.num_l2g,
        g.tables.len() - g.num_l2g,
        program.instrs.len(),
        exec.public_words.len()
    );

    // Tamper: flip one byte of one L2G main root in the arena — Phase A then
    // absorbs a root the walks cannot authenticate against.
    let mut tampered = global_arena_words(&g);
    tampered[0][0][0] += FE::one();
    assert!(
        execute(&program, &tampered, &crate::hash_pin::BLOCK_HASHER).is_err(),
        "a flipped L2G re-commit root must make the global leg unprovable"
    );
}

/// ★ THE AGGREGATION PUBLISH PROFILE drops diagnostics and NOTHING else.
///
/// # Why this gate exists
///
/// A wrap's published words are the only thing an aggregation node can read
/// about it, and a node pays per word: eight hinted halves, four canonicity
/// guards, four recombinations, thirty-six bytes of statement absorb and one
/// extension-field inverse in the `LfmPublic` balance. At the production posture
/// the diagnostic set is 10,507 words against a binding set of ≈80, so the
/// profile is what decides whether a fan-in-2 leaf hints ~21,000
/// canonicity-guarded words before it has verified anything.
///
/// A saving of that size is worth exactly as much as the proof that it saves
/// only what has no consumer. So this asserts the containment directly:
/// `Aggregation`'s words are `Diagnostic`'s with the per-sub-proof runs cut out,
/// value for value — the shared pair, the attestation id and the block-binding
/// schema are identical cells, and the bus total still ends the list.
///
/// The arithmetic is asserted alongside, from the epoch's own shapes, so a
/// diagnostic added to the per-sub-proof block in future fails here naming the
/// count rather than silently widening what every node above pays for.
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
    // ---- the tail: the bus total, which both profiles still publish last.
    assert_eq!(
        diag[diag.len() - 1],
        agg[agg.len() - 1],
        "the closure's total ends the list under either profile"
    );
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

// ===================== the node's cost instrument =========================

/// A root of zeroes, standing in for a child's `program_id` where only the
/// LENGTH of the constant matters — the statement absorbs 32 bytes whatever
/// they are.
const ZERO_ROOT: stark::config::Commitment = [0u8; 32];

/// The leg's per-published-word part, and NOTHING else: hint the words under
/// the canonicity guard, absorb them as the statement, squeeze the pair, and
/// recompute the `LfmPublic` balance.
///
/// Phase A runs over ZERO tables, so the squeeze that forces the statement's
/// segment to pack and hash is present and the per-sub-proof verification is
/// not. Everything that is not a function of `count` is therefore a fixed
/// overhead common to every point below, and cancels in the marginals.
fn publics_only_program(count: usize) -> LfmProgram {
    use super::per_table_aggregator::{emit_lfm_statement, emit_public_balance, hint_public_words};
    use super::statement_replay::replay_phase_a;

    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(8 * count as u32);
    let words = hint_public_words(&mut b, arena, count);
    let mut t = TranscriptReplay::new(&[]);
    emit_lfm_statement(&mut t, &ZERO_ROOT, &words, 8);
    let (z, alpha) = replay_phase_a(&mut t, &mut b, &[]);
    let target = emit_public_balance(&mut b, &words, z, alpha);
    b.public(target.as_cell());
    compile(b.finish())
}

/// The binding legs over `children` children, and nothing else — the words are
/// hinted (the leg would have hinted them anyway) and the delta against the
/// same program without the bindings is the bindings' whole cost.
fn bindings_only_program(children: usize, with_bindings: bool) -> (LfmProgram, usize) {
    use super::per_table_aggregator::{
        LegCells, SchemaLayout, emit_chain_bindings, hint_public_words,
    };
    use super::statement_replay::replay_phase_a;

    // A block-final epoch's output length; the schema's only variable term.
    const OUT_HALVES: usize = 8;
    let layout = SchemaLayout::new(OUT_HALVES);
    let words = layout.total();

    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let mut legs = Vec::with_capacity(children);
    for _ in 0..children {
        let arena = b.declare_arena(8 * words as u32);
        let publics = hint_public_words(&mut b, arena, words);
        let mut t = TranscriptReplay::new(&[]);
        t.append_const_bytes(&ZERO_ROOT[..]);
        let (z, alpha) = replay_phase_a(&mut t, &mut b, &[]);
        legs.push(LegCells {
            publics,
            z_alpha: (z, alpha),
        });
    }
    if with_bindings {
        let layouts: Vec<SchemaLayout> = (0..children)
            .map(|_| SchemaLayout::new(OUT_HALVES))
            .collect();
        let labels: Vec<u64> = (0..children as u64).collect();
        emit_chain_bindings(&mut b, &legs, &layouts, &labels);
    }
    (compile(b.finish()), words)
}

/// ★ THE NODE'S COST, in the three units that decide the fan-in.
///
/// # What this measures and why in this order
///
/// Lane C narrowed the fan-in question to the binding legs, and did so under the
/// BATCHED assumption that a child publishes ~285 words. Under the per-table
/// format the diagnostic wrap publishes 10,507, and the leg pays LINEARLY per
/// published word — eight hints, four canonicity guards, four recombinations,
/// thirty-six bytes of statement absorb and one extension-field inverse. So the
/// term that actually grew is the per-word one, and it is measured FIRST.
///
/// The three quantities, all emission-only:
///
/// (a) the per-published-word marginal, at several counts so LINEARITY is
///     checked rather than assumed — the statement's sponge absorbs in
///     rate-sized blocks, so the hash term is a step function whose average is
///     linear, and a two-point fit would hide that;
/// (b) the binding legs, as the delta between a node with and without them;
/// (c) F(1) and F(2), which follow from (a) rather than needing a second
///     emission: this leg is `per_table_census_tests::tenant_node_program` plus
///     the statement, the balance and the bindings. The balance is pure field
///     arithmetic and hashes NOTHING, so in COMPRESSIONS the only term this leg
///     adds to C's is the statement's — which is exactly (a)'s hash column.
///
/// ⚠ C's F(1) = 2,886 / F(2) = 5,771 are COMPRESSIONS (`wrap_tests::hash_ops`
/// over a glue delta), not cells. Quoting them against a cell figure would be
/// comparing two different measurements that happen to be numbers.
#[test]
#[ignore = "emission instrument: run explicitly, prints the node cost model"]
fn the_node_cost_model_is_measured() {
    use super::per_table_aggregator::SchemaLayout;

    let hash = super::edsl::WrapHash::production();
    let census = |p: &LfmProgram| -> (usize, usize, u64) {
        let (main, aux) =
            super::airs::lfm_cell_counts_with_hasher(p, crate::hash_pin::BLOCK_HASHER);
        (
            p.instrs.len(),
            super::wrap_tests::hash_ops(p, hash),
            main + 3 * aux,
        )
    };

    // ---- (a) the per-published-word marginal.
    const OUT_HALVES: usize = 8;
    let aggregation_words = SchemaLayout::new(OUT_HALVES).total();
    // The measured diagnostic count at q=110, plus the schema this branch adds.
    let diagnostic_words = 10_507 + SchemaLayout::new(OUT_HALVES).schema_words();
    let points = [0, 64, 128, 256, 512, aggregation_words, diagnostic_words];
    println!(
        "\n★ (a) THE PER-PUBLISHED-WORD BILL, at {hash:?}\n   \
         {:>8}  {:>12}  {:>12}  {:>14}  {:>10}",
        "words", "instrs", "hash ops", "cells", "cells/word"
    );
    let mut base = (0usize, 0usize, 0u64);
    for (i, &count) in points.iter().enumerate() {
        let (instrs, ops, cells) = census(&publics_only_program(count));
        if i == 0 {
            base = (instrs, ops, cells);
        }
        let per_word = if count > 0 {
            (cells - base.2) as f64 / count as f64
        } else {
            0.0
        };
        println!("   {count:>8}  {instrs:>12}  {ops:>12}  {cells:>14}  {per_word:>10.2}");
    }
    // Linearity, asserted rather than eyeballed: the marginal between the two
    // largest sampled points must agree with the marginal between the two
    // smallest to within the sponge's rate-sized step.
    let cells_at = |n: usize| census(&publics_only_program(n)).2;
    let (c64, c512) = (cells_at(64), cells_at(512));
    let low = (c64 - base.2) as f64 / 64.0;
    let high = (c512 - c64) as f64 / (512.0 - 64.0);
    let drift = (high - low).abs() / low;
    println!(
        "   marginal 0->64 {low:.2} cells/word, 64->512 {high:.2} cells/word \
         ({:+.1}%)",
        100.0 * (high - low) / low
    );
    assert!(
        drift < 0.25,
        "the per-word bill must be linear to within the sponge's step, got \
         {low:.2} then {high:.2} cells/word"
    );

    // ---- (b) the binding legs.
    println!(
        "\n★ (b) THE BINDING LEGS (schema = {aggregation_words} words/child)\n   \
         {:>8}  {:>12}  {:>12}  {:>14}",
        "children", "Δinstrs", "Δhash ops", "Δcells"
    );
    for children in [2usize, 3] {
        let (with, _) = bindings_only_program(children, true);
        let (without, _) = bindings_only_program(children, false);
        let (wi, wo_ops, wc) = census(&with);
        let (bi, bo_ops, bc) = census(&without);
        println!(
            "   {children:>8}  {:>12}  {:>12}  {:>14}",
            wi - bi,
            wo_ops as i64 - bo_ops as i64,
            wc - bc
        );
    }

    // ---- (c) what a leaf node costs, composed.
    println!(
        "\n★ (c) LEAF NODE = fan-in × (C's F + the statement) + the bindings.\n   \
         C measured F(1) = 2,886 and F(2) = 5,771 COMPRESSIONS at production \
         heights; the columns above are what this leg adds on top, per child.\n   \
         A child publishing {aggregation_words} words instead of {diagnostic_words} \
         is the whole lever."
    );
}
