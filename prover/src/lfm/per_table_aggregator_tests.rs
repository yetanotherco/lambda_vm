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
    let layout = SchemaLayout::wrap(OUT_HALVES);
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
            .map(|_| SchemaLayout::wrap(OUT_HALVES))
            .collect();
        // One label per WRAP child, matching `SchemaLayout::wrap`'s two label words.
        let labels: Vec<[u64; 1]> = (0..children as u64).map(|k| [k]).collect();
        let label_refs: Vec<&[u64]> = labels.iter().map(|l| &l[..]).collect();
        emit_chain_bindings(&mut b, &legs, &layouts, &label_refs);
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
    let aggregation_words = SchemaLayout::wrap(OUT_HALVES).total();
    // ⚠ A SAMPLE POINT, not an assumption. 10,507 is the MEASURED diagnostic
    // count of the q=110 wrap; the diagnostic set is dominated by per-query terms
    // (one FRI terminal and one index per query per sub-proof), so this point
    // moves with the query count.
    //
    // ✓ The posture is settled and q STAYS AT 110 — the security re-tune holds it
    // by moving `security_bits` 128 -> 120 at blowup 4 rather than by taking the
    // count to 119. So this point is current, not provisional.
    //
    // Nothing is built against it either way: what this instrument produces is
    // the per-word COEFFICIENT, a property of the emitter that holds at any
    // count. The point is here so the table brackets the real range.
    let diagnostic_words = 10_507 + SchemaLayout::wrap(OUT_HALVES).schema_words();
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

// ======================= the child harvest and the node ===================

/// One child LFM proof, production-accepted, harvested for emission.
///
/// The per-table sibling of `epoch_tests::RealEpoch`, over an LFM machine's
/// proof rather than the VM's. ★ Nothing in the harvest is LFM-specific:
/// `host_table_forked` and `build_table_legs` take `(&dyn AIR,
/// StarkProofView)`, so the same two functions read a wrap proof, a node proof
/// and a VM epoch proof alike. That is what makes one node emitter serve every
/// level.
pub(super) struct RealChild {
    pub(super) artifacts: super::registry::LfmArtifacts,
    pub(super) opts: crate::ProofOptions,
    pub(super) public_words: Vec<(u32, LfmWord)>,
    pub(super) tables: Vec<super::epoch_tests::HostTable>,
    pub(super) legs: Vec<super::epoch_verify_tests::TableLegs>,
    /// The child's OWN shared LogUp pair, recovered host-side by
    /// `verify_against_chunked`'s own Phase A replay — the oracle the node's leg
    /// must reproduce in-machine. Consumed by
    /// [`the_leaf_node_verifies_and_binds_two_wraps`] through
    /// `NodePublishSet::Diagnostic`.
    pub(super) z_alpha: (FEE, FEE),
}

/// Harvest a child from a proof PRODUCTION ACCEPTS. Panics loudly otherwise —
/// nothing downstream may read a proof the verifier would reject.
pub(super) fn real_child(
    artifacts: super::registry::LfmArtifacts,
    opts: crate::ProofOptions,
    proved: &super::proof::LfmProof,
) -> RealChild {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use stark::proof::view::MultiProofView;

    assert!(
        super::proof::verify_against_artifacts(
            &artifacts,
            &proved.proof,
            &proved.public_words,
            &opts
        ),
        "the harness only reads children production accepts"
    );

    let airs = super::airs::LfmAirs::new_chunked(
        &artifacts.roots,
        &artifacts.blake3_chunk_roots,
        &opts,
        artifacts.keccak_rnd_chunks,
        artifacts.hasher,
        artifacts.chip_set,
    );
    let refs = airs.air_refs();
    let view = MultiProofView::Owned(&proved.proof);
    assert_eq!(refs.len(), view.len(), "one AIR per sub-proof");

    // The seed IS `verify_against_chunked`'s: the LFM statement over the claimed
    // words, and nothing before it.
    let seed = || {
        let mut t = crate::hash_pin::block_transcript(&[]);
        super::statement::absorb_lfm_statement(
            &mut t,
            &artifacts.program_id,
            &proved.public_words,
            opts.fri_final_poly_log_degree,
        );
        t
    };

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

    RealChild {
        artifacts,
        opts,
        public_words: proved.public_words.clone(),
        tables,
        legs,
        z_alpha: (lookup[0], lookup[1]),
    }
}

/// The child's shape, as the node's emitter reads it.
pub(super) fn child_shape(c: &RealChild) -> super::per_table_aggregator::ChildShape<'_> {
    super::per_table_aggregator::ChildShape {
        program_id: &c.artifacts.program_id,
        num_public_words: c.public_words.len(),
        fri_final_poly_log_degree: c.opts.fri_final_poly_log_degree,
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
    let mut arenas: Vec<Vec<LfmWord>> = Vec::new();
    // The published words, eight halves each — the statement's own layout.
    let mut publics = Vec::with_capacity(8 * c.public_words.len());
    for (_, word) in &c.public_words {
        for lane in word {
            let v: u64 = lane.canonical();
            publics.push(base_word(FE::from(v & 0xFFFF_FFFF)));
            publics.push(base_word(FE::from(v >> 32)));
        }
    }
    arenas.push(publics);
    arenas.push(super::proof_arena::commitments_to_arena(
        &c.tables.iter().map(|h| h.main_root).collect::<Vec<_>>(),
    ));
    for (h, leg) in c.tables.iter().zip(&c.legs) {
        if let Some(root) = &h.aux_root {
            arenas.push(super::proof_arena::commitments_to_arena(&[*root]));
        }
        if let Some(l) = &h.contribution {
            arenas.push(vec![ext_word(l)]);
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
    use super::proof::lfm_prove;
    use super::registry::build_artifacts_with_hasher;
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
    for k in 0..FAN_IN {
        let e =
            super::epoch_tests::real_epoch_from_continuation(&inner, &elf_bytes, &bundle, k, None)
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
            build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
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
    let artifacts =
        build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
    let t = Instant::now();
    let proved =
        lfm_prove(&program, &artifacts, &arenas, &wrap_opts).expect("★ THE LEAF NODE MUST PROVE");
    let prove_secs = t.elapsed().as_secs_f64();
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

/// Prove one aggregation node and hand it back as a CHILD of the next level.
///
/// The whole composition argument in one function: a node's proof is a plain
/// per-table `MultiProof`, so `real_child` reads it exactly as it reads a wrap's,
/// and the layout that describes it is `SchemaLayout::node`. Nothing about the
/// level appears here — which is what "the same emitter serves every level"
/// means operationally.
fn prove_node_as_child(
    children: &[RealChild],
    layouts: &[super::per_table_aggregator::SchemaLayout],
    labels: &[&[u64]],
    label_range: (u64, u64),
    out_halves: usize,
    opts: &crate::ProofOptions,
) -> (RealChild, super::per_table_aggregator::SchemaLayout) {
    use super::per_table_aggregator::{NodePublishSet, SchemaLayout};

    let program = node_program(
        children,
        layouts,
        labels,
        label_range,
        NodePublishSet::Aggregation,
    );
    let arenas: Vec<Vec<LfmWord>> = children.iter().flat_map(child_arena_words).collect();
    let artifacts =
        super::registry::build_artifacts_with_hasher(&program, opts, crate::hash_pin::BLOCK_HASHER);
    let proved = super::proof::lfm_prove(&program, &artifacts, &arenas, opts)
        .expect("an aggregation node must prove");
    let layout = SchemaLayout::node(out_halves);
    layout.assert_covers(proved.public_words.len());
    (real_child(artifacts, opts.clone(), &proved), layout)
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
    use super::proof::lfm_prove;
    use super::registry::build_artifacts_with_hasher;
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
    let bundle = crate::continuation::prove_continuation(&elf_bytes, &[], epoch_log2, &inner)
        .expect("the fixture continuation must prove");
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
    for leaf in 0..FAN_IN {
        let mut wraps = Vec::with_capacity(FAN_IN);
        let mut wrap_layouts = Vec::with_capacity(FAN_IN);
        let mut wrap_labels: Vec<[u64; 1]> = Vec::with_capacity(FAN_IN);
        let mut out_halves = 0usize;
        for i in 0..FAN_IN {
            let k = leaf * FAN_IN + i;
            let e = super::epoch_tests::real_epoch_from_continuation(
                &inner, &elf_bytes, &bundle, k, None,
            )
            .expect("every epoch must reconstruct from proofs alone");
            out_halves = e.statement.public_output_len.div_ceil(4);
            let program =
                super::epoch_tests::epoch_program_publishing(&e, true, Publishes::Aggregation);
            let arenas = super::epoch_tests::epoch_arena_words(&e, true);
            let artifacts =
                build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
            let proved = lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
                .expect("the epoch wrap must prove");
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
        let (child, layout) =
            prove_node_as_child(&wraps, &wrap_layouts, &refs, range, out_halves, &wrap_opts);
        println!(
            "   leaf {leaf}: {} published words, {} sub-proofs",
            child.public_words.len(),
            child.tables.len()
        );
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
    let (inner_node, inner_layout) =
        prove_node_as_child(&leaves, &leaf_layouts, &refs, range, out_halves, &wrap_opts);
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
