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
        .map(|(h, leg)| {
            1 + leg.verify.num_queries + 3 + h.zetas.len() + h.shape.num_queries
        })
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
