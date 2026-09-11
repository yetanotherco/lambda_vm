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
    // The whole table set, closing against zero — the standalone wrap. A SLICE of
    // it closes against its own published partial instead; see
    // [`global_slice_program`].
    global_slice_program(
        g,
        &super::global_split::SlicePartition::even(g.tables.len(), 1),
        0,
    )
}

/// One SLICE of the global wrap: the same statement and the same Phase A, but
/// full verification legs for `partition.slice(i)` ONLY, publishing that range's
/// PARTIAL bus sum instead of closing against zero.
///
/// ⛔ WHY A SLICE IS NOT A SUB-PROOF. The global proof is one `MultiProof` whose
/// bus balances over ALL its tables, and [`super::epoch::fork_table`] uses
/// `num_tables` as a domain separator — so a slice must replay the FULL Phase A
/// and fork each of its tables at its TRUE index within the TRUE `num_tables`.
/// Only the WALKS are divided, which is exactly the expensive half: `LFM_HASH` is
/// `queries x Merkle depth` per table, and depth is what a slice stops paying for
/// tables it does not verify.
///
/// ⇒ At `k = 1` this IS the standalone wrap, target zero, which is why
/// [`global_verifier_program`] is defined as this and the existing gate covers
/// both paths rather than only one.
pub(super) fn global_slice_program(
    g: &RealGlobal,
    partition: &super::global_split::SlicePartition,
    slice: usize,
) -> LfmProgram {
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
    let (slice_lo, slice_hi) = partition.slice(slice);
    for (i, h) in g.tables.iter().enumerate() {
        // ⚠ The loop still WALKS the full table list, because Phase A and the
        // fork's domain separator are defined over all of it — only the
        // verification legs are restricted.
        if i < slice_lo || i >= slice_hi {
            continue;
        }
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
    // ★ THE ONE PLACE A SLICE DIFFERS FROM THE WHOLE.
    //
    // Over EVERY table the bus balances to zero, and that is asserted here. Over a
    // PROPER SLICE it balances to a partial only the parent can check, so the
    // partial is summed and PUBLISHED — and the parent then pins the partition,
    // asserts every slice agreed on `(z, alpha)`, sums the partials and asserts
    // zero.
    //
    // ⛔ THE SLICE ARM DELIBERATELY HAS NO LOCAL ASSERT. Closing the slice against
    // its own sum would emit `assert_eq(total, total)` — a check that cannot fail,
    // which is worse than no check because it produces evidence. What BINDS the
    // partial is that the published word IS this sum by construction: the slice's
    // own proof makes `sum over its tables == P_i`, so a prover cannot choose
    // `P_i` freely, and the only real assert belongs to the parent.
    if partition.k() == 1 {
        let shape = super::logup::LogUpShape {
            num_contributing_tables: contributions.len(),
            num_output_bytes: 0,
        };
        let target = b.ext_const(&FEE::zero());
        super::logup::emit_bus_closure(&mut b, &shape, &contributions, target);
    } else {
        let mut total = match contributions.first() {
            Some(first) => *first,
            None => b.ext_const(&FEE::zero()),
        };
        for c in contributions.iter().skip(1) {
            total = b.eadd(total, *c);
        }
        b.public(total.as_cell());
    }

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
/// ★★★ THE GATE ON THE WHOLE MECHANISM: `k` real global SLICES verify, agree,
/// and their partial bus sums add to ZERO in the PARENT.
///
/// Everything the split rests on, end to end and at fixture scale:
///
/// - each slice replays the FULL Phase A and verifies only `partition.slice(i)`,
///   at each table's TRUE index within the TRUE `num_tables`, and publishes a
///   PARTIAL bus sum instead of closing against zero;
/// - the parent verifies the `k` slice proofs, pins the partition through the
///   `program_id`s it embeds, asserts one shared `(z, alpha)` and one set of L2G
///   roots, sums the partials and asserts ZERO;
/// - and the parent republishes the prefix in the packing the root reads.
///
/// ⛔ THE PARTIALS ARE ASSERTED NON-ZERO, and that is not decoration. If a slice's
/// own partial were zero the parent's sum-to-zero would pass without composing
/// anything, and this gate would be reporting a check that cannot fail under the
/// failure mode it exists for. A half of the table set summing to zero is a
/// coincidence of negligible probability; if it ever happens, the gate is wrong
/// about what it proves and must say so rather than go green.
///
/// ⚠ The HOST-SIDE agreement and sum checks below come FIRST on purpose. If the
/// slices genuinely disagreed, the parent's prove would fail and the failure
/// would read like a bug in the parent; asserting the claim host-side separates
/// "the machine is wrong" from "the claim is false".
#[test]
#[ignore = "box tier: proves a fixture continuation, k global slices and their parent"]
fn the_global_slices_verify_and_sum_to_zero() {
    use super::proof::lfm_prove;
    use super::registry::build_artifacts_with_hasher;
    use std::time::Instant;

    const K: usize = 2;

    let elf_bytes = super::proof_fixture::read_inner_elf();
    let inner = super::proof_fixture::fixture_options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let bundle = crate::continuation::prove_continuation(
        &elf_bytes,
        &[],
        super::proof_fixture::FIXTURE_EPOCH_LOG2,
        &inner,
    )
    .expect("the fixture continuation must prove");
    let g = real_global(&elf_bytes, &bundle, &inner);
    assert!(
        g.tables.len() >= K,
        "the fixture's global proof has {} tables and cannot be split {K} ways",
        g.tables.len()
    );
    let partition = super::global_split::SlicePartition::even(g.tables.len(), K);
    // ⛔ ONE LAYOUT PER SHAPE, read off the SAME partition the emitter branches on.
    let publishes = super::block_root::GlobalPublishes::of(
        &partition,
        super::block_root::GlobalLayout {
            num_epochs: g.num_l2g,
            lanes_per_root: super::proof_arena::lanes_per_root(),
        },
    );
    let slice_layout = publishes
        .as_slice()
        .expect("k > 1 is the SLICE shape, and the layout follows the partition");
    let arenas = global_arena_words(&g);

    let mut slices: Vec<RealChild> = Vec::with_capacity(K);
    let mut partials: Vec<FEE> = Vec::with_capacity(K);
    for i in 0..K {
        let (lo, hi) = partition.slice(i);
        let program = global_slice_program(&g, &partition, i);
        let artifacts =
            build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
        let t = Instant::now();
        let proved = lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
            .unwrap_or_else(|e| panic!("global slice {i} (tables {lo}..{hi}) must prove: {e:?}"));
        assert_eq!(
            proved.public_words.len(),
            slice_layout.total(),
            "slice {i}: {}",
            publishes.describe()
        );
        println!(
            "   ★ slice {i} (tables {lo}..{hi}): proved in {:.1}s, {} published words",
            t.elapsed().as_secs_f64(),
            proved.public_words.len(),
        );
        let partial =
            super::word::word_as_ext(&proved.public_words[slice_layout.partial_sum_word()].1)
                .expect("the partial bus sum is an extension word");
        assert_ne!(
            partial,
            FEE::zero(),
            "slice {i}'s OWN partial is zero, so the parent's sum-to-zero would \
             hold without composing anything and this gate would prove nothing \
             about the split"
        );
        partials.push(partial);
        // `real_child` verifies before harvesting, so nothing below reads a proof
        // production would reject.
        slices.push(real_child(artifacts, wrap_opts.clone(), &proved));
    }

    // ---- the claim, host-side, before any parent is emitted.
    let sum = partials.iter().fold(FEE::zero(), |acc, p| acc + *p);
    assert_eq!(
        sum,
        FEE::zero(),
        "the {K} partials do not add to zero: {partials:?}. The slices do not \
         tile the table set, or they were verified under different challenges"
    );
    let pub_word = |i: usize, w: usize| slices[i].public_words[w].1;
    for i in 1..K {
        for w in [
            slice_layout.shared.z_word(),
            slice_layout.shared.alpha_word(),
        ] {
            assert_eq!(
                pub_word(i, w),
                pub_word(0, w),
                "slice {i} published a different word {w} of the shared pair"
            );
        }
        for epoch in 0..g.num_l2g {
            for lane in 0..slice_layout.shared.lanes_per_root {
                let w = slice_layout.shared.l2g_word(epoch, lane);
                assert_eq!(
                    pub_word(i, w),
                    pub_word(0, w),
                    "slice {i} published a different L2G root at epoch {epoch} \
                     lane {lane}"
                );
            }
        }
    }
    // And the pair is production's own, not merely self-consistent.
    assert_eq!(
        super::word::word_as_ext(&pub_word(0, slice_layout.shared.z_word())).expect("z is an ext"),
        g.z_alpha.0,
        "the slices' z is not the challenge production derived"
    );
    assert_eq!(
        super::word::word_as_ext(&pub_word(0, slice_layout.shared.alpha_word()))
            .expect("alpha is an ext"),
        g.z_alpha.1,
        "the slices' alpha is not the challenge production derived"
    );

    // ---- the parent.
    let program = global_parent_program(&slices, &partition, slice_layout);
    let arenas: Vec<Vec<LfmWord>> = slices.iter().flat_map(child_arena_words).collect();
    let artifacts =
        build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
    let t = Instant::now();
    let proved = lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
        .unwrap_or_else(|e| panic!("★ THE GLOBAL PARENT MUST PROVE: {e:?}"));
    let prove_secs = t.elapsed().as_secs_f64();
    let t = Instant::now();
    assert!(
        super::proof::verify_against_artifacts(
            &artifacts,
            &proved.proof,
            &proved.public_words,
            &wrap_opts
        ),
        "the GLOBAL PARENT's proof must verify"
    );
    assert_eq!(
        proved.public_words.len(),
        slice_layout.shared.total(),
        "the parent republishes the SHARED prefix and NOTHING for the sum"
    );
    // ⛔ THE PACKING, ON REAL WORDS: every republished word equals the slices'
    // own, at the index the root's L2G compare reads it from.
    for w in [
        slice_layout.shared.z_word(),
        slice_layout.shared.alpha_word(),
    ] {
        assert_eq!(
            proved.public_words[w].1,
            pub_word(0, w),
            "republished word {w}"
        );
    }
    for epoch in 0..g.num_l2g {
        for lane in 0..slice_layout.shared.lanes_per_root {
            let w = slice_layout.shared.l2g_word(epoch, lane);
            assert_eq!(
                proved.public_words[w].1,
                pub_word(0, w),
                "the parent republished epoch {epoch} lane {lane} at index {w} \
                 carrying another lane's value; the root would refold the wrong \
                 roots and the length assert would not notice"
            );
        }
    }
    println!(
        "★★★ {K} GLOBAL SLICES VERIFIED AND SUMMED TO ZERO\n   parent: prove \
         {prove_secs:.1}s · verify {:.2}s · {} instructions · {} published words",
        t.elapsed().as_secs_f64(),
        program.instrs.len(),
        proved.public_words.len(),
    );
}

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

/// One level's proofs, their layouts and their label runs — kept as three
/// parallel vectors because `node_program` and `prove_node_program_as_child` take
/// `&[RealChild]` and `&[SchemaLayout]`, so a node's child group is a subslice of
/// each rather than a clone of every child's harvest.
pub(super) type TreeLevel = (
    Vec<RealChild>,
    Vec<super::per_table_aggregator::SchemaLayout>,
    Vec<Vec<u64>>,
);

/// The PARENT of `k` global SLICES, emitted over the slice proofs it folds.
///
/// ⛔ `partition` must be the SAME object the slices were emitted against, not a
/// second one built from the same `(num_tables, k)`. It is the single source for
/// `k` and for every bound, and the parent's partition pin is precisely that the
/// `program_id`s it embeds as constants are the ones those slices compiled to —
/// a slice with different bounds is a DIFFERENT PROGRAM and its proof is rejected
/// on identity.
pub(super) fn global_parent_program(
    slices: &[RealChild],
    partition: &super::global_split::SlicePartition,
    layout: &super::block_root::SliceLayout,
) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let shapes: Vec<_> = slices.iter().map(child_shape).collect();
    super::global_parent::emit_global_parent(
        &mut b,
        &super::global_parent::ParentInputs {
            slices: &shapes,
            partition,
            layout,
        },
    );
    compile(b.finish())
}

/// Emit a block-artifact ROOT program, for sizing or for proving.
///
/// ⛔ `publishes` IS A PARAMETER, not `RootPublishSet::default()` read here. The
/// caller that emits the artifact is the caller that must assert its width, and
/// a default read in one place against an expectation spelled in the other is
/// two copies of the artifact's shape — free to drift the day the default moves.
#[allow(clippy::too_many_arguments)]
pub(super) fn root_program(
    interior: &[RealChild],
    interior_layouts: &[super::per_table_aggregator::SchemaLayout],
    labels: &[&[u64]],
    label_range: (u64, u64),
    global: &RealChild,
    global_child_layout: &super::block_root::GlobalLayout,
    fold_shape: &super::block_root::FoldShape,
    publishes: super::block_root::RootPublishSet,
) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let shapes: Vec<_> = interior.iter().map(child_shape).collect();
    let g = child_shape(global);
    super::block_root::emit_block_root(
        &mut b,
        &super::block_root::RootInputs {
            interior: &shapes,
            interior_layouts,
            labels,
            label_range,
            global: &g,
            global_child_layout,
            fold_shape,
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
    //
    // ⚠ THREE marks, not one. The mark above is taken BEFORE
    // `build_artifacts_with_hasher`, which is not a bookkeeping call: it runs
    // `lde_columns` + `commit_lde_columns` over every chip group and builds the
    // prep round — a full commitment pass over the whole program. So a single
    // "before the prove" mark brackets the artifact build and the prove TOGETHER
    // and cannot say which of them costs what. These split it.
    let t = Instant::now();
    let artifacts =
        build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
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
    type CachedProof = (
        stark::proof::stark::MultiProof<GoldilocksField, GoldilocksExtension, ()>,
        Vec<(u32, LfmWord)>,
    );
    let Some(p) = path.filter(|_| mode != CacheMode::Off) else {
        println!("   {label}: PROVED in-process, NOT cached");
        return prove();
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
            let bytes =
                rkyv::to_bytes::<rkyv::rancor::Error>(&bundle).expect("the bundle must serialize");
            if let Some(dir) = p.parent() {
                std::fs::create_dir_all(dir).expect("the cache directory must exist");
            }
            std::fs::write(&p, &bytes).expect("the bundle must persist");
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

    let t = Instant::now();
    let arenas: Vec<Vec<LfmWord>> = children.iter().flat_map(child_arena_words).collect();
    let t_arenas = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let artifacts =
        super::registry::build_artifacts_with_hasher(program, opts, crate::hash_pin::BLOCK_HASHER);
    let t_artifacts = t.elapsed().as_secs_f64();
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
    let child = real_child(artifacts, opts.clone(), &proved);
    // ★ THE SPLIT, because "60 s per node" attributes nothing. Which half a
    // second cache layer would have to hold — the artifacts or the harvest —
    // is a different build depending on this line, and guessing it is how a
    // campaign builds the wrong cache.
    println!(
        "   {label} TIMING: arenas {t_arenas:.1}s · build_artifacts {t_artifacts:.1}s \
         · prove {t_prove:.1}s · harvest {:.1}s",
        t.elapsed().as_secs_f64()
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

/// ★★★ THE PRODUCTION-SCALE LEAF NODE — the run that answers whether a tree fits.
///
/// Everything measured so far is FIXTURE scale, where a node's children are
/// wraps of 2^3-cycle epochs. This drives the same construction from a REAL
/// block, so the wrap proofs the node verifies have production-depth Merkle
/// trees. It is a MEASUREMENT, not a gate: the tamper arms and the `(z, α)`
/// differential are covered at fixture scale by
/// [`the_leaf_node_verifies_and_binds_two_wraps`] and are not repeated here,
/// because what changes with scale is cost and nothing else.
///
/// ```text
/// LFM_CENSUS_ELF=/path/to/ethrex.elf \
/// LFM_CENSUS_INPUT=/path/to/block.bin \
/// LFM_CENSUS_EPOCH_LOG2=22 LAMBDA_VM_MAX_ROWS_LOG2=22 \
/// cargo test --release -p lambda-vm-prover --lib \
///   lfm::per_table_aggregator_tests::the_production_leaf_node_measures -- \
///   --ignored --exact --nocapture
/// ```
///
/// ⚠ Per-phase RSS marks, not one figure. `peak_rss_gib` is `VmHWM`, a PROCESS
/// high-water mark that only rises, so a single number spans the base prove, the
/// wrap proves and the node alike — the trap that made a 40.6 GiB reading look
/// like a node's cost when the node's own share was a different number. Every
/// phase boundary is marked, and the node's own artifacts+prove is the delta
/// from the last mark before it.
#[test]
#[ignore = "box tier, production scale: needs LFM_CENSUS_ELF and LFM_CENSUS_INPUT"]
fn the_production_leaf_node_measures() {
    use super::epoch_tests::{EpochInputs, Publishes};
    use super::per_table_aggregator::{FAN_IN, SchemaLayout};
    use super::proof::lfm_prove;
    use super::registry::build_artifacts_with_hasher;
    use std::time::Instant;

    // ⛔ THE DEVICE, ASSERTED IN-PROCESS. `cfg!` rather than `#[cfg]` so the body
    // below still compiles — and is still linted — on the non-cuda passes.
    //
    // A GPU box ran this for 34 minutes on the CPU because a wrapper script
    // omitted `--features cuda`, and every line of output was perfectly legible.
    // ⚠ The host-memory figure is not merely irrelevant on the CPU path, it is
    // BIASED IN THE DIRECTION THAT MATTERS: the LDE and commit buffers live in
    // host RAM instead of the card's 32 GiB, so a CPU peak OVERSTATES the host
    // number the 120.6 GiB feasibility question turns on. Reading a red off it
    // would be a false negative on the tree.
    if !cfg!(feature = "cuda") {
        panic!(
            "the production node measurement requires `--features cuda`. Without it \
             this proves on the CPU and answers a different question — and its host \
             peak is biased HIGH, so a red result would be an artefact of the build \
             rather than a fact about the tree"
        );
    }
    for var in ["LFM_CENSUS_ELF", "LFM_CENSUS_INPUT"] {
        assert!(
            std::env::var(var).is_ok(),
            "{var} must name a file: this measures the PRODUCTION node, and \
             silently falling back to the fixture would report a fixture number \
             under a production name"
        );
    }

    // ★ FAN-IN IS AN INPUT, because it is the decision this run exists to make.
    //
    // The tree's shape at 19 epochs is 5 levels / 21 nodes at fan-in 2 and
    // 3 levels / 11 nodes at fan-in 3, so the choice is worth a factor of two in
    // total tree work — and the old argument against fan-in 3 was a VRAM argument
    // that turned out to be about concurrency, not size. What remains is a HOST
    // argument (`L_children` and the node's own `W` both grow with a third leg)
    // and a CARD argument (a third leg's rows may cross a padding step), and both
    // are measurements. `FAN_IN` stays the default so the const remains the one
    // place the tree's shape comes from.
    let fan_in: usize = match std::env::var("LFM_CENSUS_FAN_IN") {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|e| panic!("LFM_CENSUS_FAN_IN must be an integer: {e}")),
        Err(_) => FAN_IN,
    };
    assert!(
        (2..=4).contains(&fan_in),
        "LFM_CENSUS_FAN_IN must be in 2..=4, got {fan_in}: one child is not an \
         aggregation, and nothing above four has been costed on either the host \
         or the card"
    );

    let inputs = EpochInputs::from_env();
    let inner = crate::recursion::Preset::Blowup4.options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    println!(
        "★ PRODUCTION LEAF NODE: FAN-IN {fan_in} · guest {}, {} input bytes, \
         2^{} cycles/epoch, inner blowup {} / {} q, wrap blowup {} / {} q",
        inputs.label,
        inputs.private_input.len(),
        inputs.epoch_log2,
        inner.blowup_factor,
        inner.fri_number_of_queries,
        wrap_opts.blowup_factor,
        wrap_opts.fri_number_of_queries,
    );

    // ---- the base layer: a real chained bundle, so the register seam is real.
    //
    // ★ CACHED, and the cache is not a convenience — it is what separates the
    // two `L`s. Proving the base in-process leaves its residue live when the node
    // runs, so the node's measured carry-in is `L_children + L_base_residue`; a
    // run that LOADS the bundle carries in `L_children` alone, which is what any
    // separately-written tree-builder would carry. The deserialise path is
    // production's own (`bin/cli/src/main.rs:877-886`).
    //
    // ⇒ The pair of numbers is the measurement: the difference between a PROVED
    // base and a LOADED one IS the base residue, and therefore how much of any
    // high peak belongs to this harness rather than to the tree.
    //
    // ⛔ THE MODE IS NAMED BY THE CALLER, NOT READ OFF THE DISK. Cache-if-present
    // made this harness choose its own experiment: a relaunch with `A_BUNDLE`
    // still exported loaded a bundle an earlier run had saved, the base "proved"
    // in 1.4 s, `prove_continuation` never ran, and the run answered the CONTROL
    // having been launched as the TEST. Every line was legible and the pass real.
    // ⇒ `prove` with the file present is a refusal, not a silent load; `load`
    // without it is a refusal, not a silent 20-minute prove.
    let bundle_path = std::env::var("A_BUNDLE").ok();
    let mode = CacheMode::from_env(bundle_path.is_some());
    let t = Instant::now();
    let bundle = cached_bundle(mode, bundle_path.map(std::path::PathBuf::from), || {
        crate::continuation::prove_continuation(
            &inputs.elf_bytes,
            &inputs.private_input,
            inputs.epoch_log2,
            &inner,
        )
        .expect("the block must prove")
    });
    assert!(
        bundle.num_epochs() >= fan_in,
        "a fan-in-{fan_in} leaf needs {fan_in} epochs, the block has {}",
        bundle.num_epochs()
    );
    println!(
        "   base: {} epochs in {:.1}s",
        bundle.num_epochs(),
        t.elapsed().as_secs_f64(),
    );
    // ★ `L` DECOMPOSED, because only part of it scales with fan-in.
    //
    // The live figure here is the BUNDLE's residency, and that is the same at
    // every fan-in; each child then adds its own term on top. Predicting a
    // fan-in-3 carry-in by scaling the whole of `L` therefore over-predicts, by
    // exactly this number's worth. Measuring the two apart is what makes the
    // next fan-in's carry-in a derivation rather than a guess.
    mark("AFTER the base, BEFORE any child (this live figure is L_bundle)");

    // ---- the children.
    let mut children = Vec::with_capacity(fan_in);
    let mut layouts = Vec::with_capacity(fan_in);
    let mut labels: Vec<[u64; 1]> = Vec::with_capacity(fan_in);
    let mut out_halves = 0usize;
    for k in 0..fan_in {
        let t = Instant::now();
        let e = super::epoch_tests::real_epoch_from_continuation(
            &inner,
            &inputs.elf_bytes,
            &bundle,
            k,
            None,
        )
        .expect("every epoch must reconstruct from proofs alone");
        out_halves = e.statement.public_output_len.div_ceil(4);
        let shapes: Vec<&super::epoch::TableChallengeShape> =
            e.tables.iter().map(|h| &h.shape).collect();
        assert_samplable(&format!("inner epoch {k}"), &shapes);
        let program =
            super::epoch_tests::epoch_program_publishing(&e, true, Publishes::Aggregation);
        let arenas = super::epoch_tests::epoch_arena_words(&e, true);
        let artifacts =
            build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
        let proved = lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
            .expect("the epoch wrap must prove");
        let layout = SchemaLayout::wrap(out_halves);
        layout.assert_covers(proved.public_words.len());
        layouts.push(layout);
        labels.push([crate::tables::local_to_global::epoch_label(k as u64)]);
        let child = real_child(artifacts, wrap_opts.clone(), &proved);
        println!(
            "   wrap {k}: {:.1}s, {} published words, {} sub-proofs",
            t.elapsed().as_secs_f64(),
            child.public_words.len(),
            child.tables.len(),
        );
        children.push(child);
        // Live, not only the high-water: the per-child residency term is the
        // difference between consecutive live marks, and a high-water cannot
        // show a difference a later phase has already exceeded.
        mark(&format!("AFTER wrap {k} (L_bundle + {} children)", k + 1));
    }

    // ---- the node.
    for (k, c) in children.iter().enumerate() {
        let shapes: Vec<&super::epoch::TableChallengeShape> =
            c.tables.iter().map(|h| &h.shape).collect();
        assert_samplable(&format!("child {k} (a wrap proof)"), &shapes);
    }
    // ★ THE PER-LEG WORK, ITEMISED — the only way to say what a third leg adds.
    //
    // A leg's cost is set by its child's sub-proof GEOMETRY, not by the child's
    // published words: per sub-proof it forks one Phase A, walks `num_queries`
    // Merkle paths of `log2_trace + log2_blowup` levels apiece, and closes a bus.
    // The census ratio alone cannot say which of those grew, so a fan-in change
    // that moved one term would be indistinguishable from one that moved another.
    println!("   child 0 sub-proof geometry — LDE = 2^(trace+blowup), the walk depth:");
    for (i, h) in children[0].tables.iter().enumerate() {
        let sh = &h.shape;
        println!(
            "     {i:>3}: 2^{}+2^{} lde, {} queries, {} parts, aux {}, contrib {}",
            sh.log2_trace_length,
            sh.log2_blowup,
            sh.num_queries,
            sh.num_parts,
            sh.has_aux_root,
            sh.has_contribution,
        );
    }

    let label_refs: Vec<&[u64]> = labels.iter().map(|l| &l[..]).collect();
    let range = (labels[0][0], labels[fan_in - 1][0]);
    let t = Instant::now();
    let program = node_program(
        &children,
        &layouts,
        &label_refs,
        range,
        super::per_table_aggregator::NodePublishSet::Aggregation,
    );
    let arenas: Vec<Vec<LfmWord>> = children.iter().flat_map(child_arena_words).collect();
    let (main, aux) =
        super::airs::lfm_cell_counts_with_hasher(&program, crate::hash_pin::BLOCK_HASHER);
    let cells = main + 3 * aux;
    const EMPTY_MACHINE_CELLS: u64 = 26_482_828;
    println!(
        "\n★ NODE CENSUS: {cells} cells ({} instructions), floor {EMPTY_MACHINE_CELLS} \
         ({:.1}%), verification work {}\n   emitted in {:.1}s",
        program.instrs.len(),
        100.0 * EMPTY_MACHINE_CELLS as f64 / cells as f64,
        cells.saturating_sub(EMPTY_MACHINE_CELLS),
        t.elapsed().as_secs_f64(),
    );

    // ★★ THE PADDING STEP — the term a census RATIO cannot show.
    //
    // `rows` is `real_rows.next_power_of_two()`, so `headroom` is each chip's
    // distance to its next DOUBLING and `cliff_cost` is what crossing it adds.
    // Raising fan-in multiplies the workload by `(n+1)/n`, and every `at_risk`
    // chip with less headroom than that doubles its LDE, its snapshot and its
    // tree at once. ⇒ In the CARD's terms growth is a STEP function of fan-in,
    // not the smooth ratio the census reports, and this panel is the only thing
    // that says which chips are standing near an edge. The wrap paid five
    // simultaneous doublings once for want of exactly this reading (#903).
    let panel = super::airs::lfm_chip_census_with_hasher(&program, crate::hash_pin::BLOCK_HASHER);
    println!("   chip panel — rows real/committed, headroom to the next doubling:");
    for c in &panel {
        println!(
            "     {:<14} {:>10}/{:>10}  headroom {:>5.1}%  {}  cliff +{} cells",
            c.name,
            c.real_rows,
            c.rows,
            100.0 * c.headroom(),
            if c.at_risk() { "AT RISK" } else { "fixed  " },
            c.cliff_cost(),
        );
    }
    let step = (fan_in + 1) as f64 / fan_in as f64;
    let stepping: Vec<&str> = panel
        .iter()
        .filter(|c| c.at_risk() && c.real_rows as f64 * step > c.rows as f64)
        .map(|c| c.name)
        .collect();
    let exposed: u64 = panel
        .iter()
        .filter(|c| c.at_risk() && c.real_rows as f64 * step > c.rows as f64)
        .map(|c| c.cliff_cost())
        .sum();
    println!(
        "   ⇒ fan-in {} would multiply the workload by {step:.3}×. Chips that would \
         STEP: {stepping:?}, adding {exposed} cells ON TOP OF the ratio",
        fan_in + 1,
    );
    // ★ THE mark the whole prediction turns on: `live` here is `L`, what the
    // node carries in. The node's own working set is what it adds to THAT, not
    // to the high-water mark an earlier phase may already have set.
    mark("BEFORE build_artifacts (this live figure IS L)");

    let t = Instant::now();
    let artifacts =
        build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
    println!("   build_artifacts: {:.1}s", t.elapsed().as_secs_f64());
    mark("after build_artifacts");
    // Counters reset HERE, not at the top: the base prove and the wraps have
    // their own device traffic, and what must be proven is that THE NODE reached
    // the card — a non-zero total from an earlier phase would satisfy a weaker
    // check while the node itself ran on the host.
    #[cfg(feature = "cuda")]
    stark::gpu_lde::reset_all_gpu_call_counters();
    let t = Instant::now();
    let proved = lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
        .expect("★ THE PRODUCTION LEAF NODE MUST PROVE");
    let prove_secs = t.elapsed().as_secs_f64();
    println!("   lfm_prove: {prove_secs:.1}s");
    mark("after lfm_prove");
    #[cfg(feature = "cuda")]
    {
        use stark::gpu_lde as g;
        let calls = [
            ("lde", g::gpu_lde_calls()),
            ("leaf_hash", g::gpu_leaf_hash_calls()),
            ("merkle_tree", g::gpu_merkle_tree_calls()),
            ("composition", g::gpu_composition_calls()),
            ("fri", g::gpu_fri_calls()),
        ];
        let total: u64 = calls.iter().map(|(_, n)| *n).sum();
        println!("   GPU dispatches during the NODE prove: {calls:?} (total {total})");
        assert!(
            total > 0,
            "the node prove reached the device ZERO times — it ran on the host \
             even though cuda is compiled in, so the host peak above is not the \
             production figure"
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
        "the production leaf node's proof must verify"
    );
    let node_layout = SchemaLayout::node(out_halves);
    node_layout.assert_covers(proved.public_words.len());
    println!(
        "\n★★★ PRODUCTION LEAF NODE PROVED AND VERIFIED\n   prove {prove_secs:.1}s\n   \
         verify {:.2}s\n   {} published words, {} sub-proofs\n   \
         PROCESS peak RSS {:?} GiB (spans the base and the wraps too — read the \
         per-phase marks above for the node's own share)",
        t.elapsed().as_secs_f64(),
        proved.public_words.len(),
        proved.proof.proofs.len(),
        super::wrap_tests::peak_rss_gib(),
    );
}

/// ★ THE TREE'S SHAPE COMES FROM THE EPOCH COUNT, not a constant.
///
/// Pure arithmetic — no proving, so it runs on every suite. It exists because
/// the shape was twice planned against a number that was not the block's, and
/// once in each direction. Block 25368371 is **39,631,559** cycles: 10 epochs at
/// 2^22, **19 at 2^21**. The retired **74,819,518** — the July prebuilt guest,
/// ~47% dearer at identical work — gives 18 and 36 instead, i.e. a tree a whole
/// level too deep. A tree built to a constant answers whichever question that
/// constant came from.
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
    // block 25368371's **39,631,559** cycles, re-measured 2026-09-07.
    //
    // ⚠ NOT 74,819,518. That figure is the JULY prebuilt guest; thin-LTO and the
    // accelerators made it ~47% cheaper at identical work — same block, same
    // keccak 10,478, same ECSM 116. The stale figure still sits in a table in
    // `real_block_benchmark_selection.md`, with the correction in an addendum
    // BELOW it saying in terms that every epochs-per-block number derived from
    // 74.8M is stale. Reading the table row and stopping gives 18/36 epochs and
    // a tree two levels too deep.
    for (epochs, fan_in, levels, nodes) in [
        (10usize, 2usize, 4usize, 11usize), // 2^22: 10 -> 5 -> 3 -> 2 -> 1
        (10, 3, 3, 7),                      // 2^22: 10 -> 4 -> 2 -> 1
        (19, 2, 5, 21),                     // 2^21: 19 -> 10 -> 5 -> 3 -> 2 -> 1
        (19, 3, 3, 11),                     // 2^21: 19 -> 7 -> 3 -> 1
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

    // ★ The leftover rule is EXERCISED, not merely permitted. 19 at fan-in 2
    // leaves one over at three levels (19, 5 and 3); a shape that never produced
    // a short node would pass every assertion above while testing nothing about
    // wrap-versus-carry.
    let short: usize = tree_shape(19, 2)
        .iter()
        .filter(|l| l.arities.iter().any(|a| *a < 2))
        .count();
    assert_eq!(short, 3, "19 at fan-in 2 must exercise the leftover rule");
}

// ======================= the production tree driver =======================

/// A 100 Hz `VmRSS` sampler with an ARGMAX TIMESTAMP.
///
/// ⛔ WHY NOT `VmHWM`. A high-water only rises, so it cannot see a peak BELOW
/// itself and cannot say WHEN its own peak happened. That is what struck the
/// 37.169 GiB "the tree does not grow per level": three reads of one monotone
/// counter, in a process that had already proved a base and four wraps, with no
/// mark before the first level — a whole-process bound that priced no level in
/// it, level 1 included, and being a within-run comparison did not rescue it.
///
/// ⇒ A sampled max carries an argmax `t`, which buys two things a high-water
/// cannot: a level's peak is the max inside ITS OWN window rather than the
/// process's, and the host peak can be checked for SIMULTANEITY against an
/// external device trace. Non-simultaneous maxima must not be summed.
struct HostSampler {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<(f64, f64)>>,
}

impl HostSampler {
    fn start() -> Self {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            let (mut peak, mut at) = (0.0f64, unix_now());
            while !flag.load(Ordering::Relaxed) {
                if let (Some(rss), _) = rss_marks()
                    && rss > peak
                {
                    peak = rss;
                    at = unix_now();
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            (peak, at)
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    /// `(peak GiB, argmax UNIX seconds)` over this sampler's window.
    fn stop(mut self) -> (f64, f64) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.handle
            .take()
            .expect("a sampler is stopped once")
            .join()
            .expect("the sampler thread must not panic")
    }
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// The cgroup memory ceiling, trying **v2 then v1**, or `None` with the reason.
///
/// ⚠ A sampler hard-coded to one layout reads NOTHING on the other box and
/// reports no error, so a percentage-of-ceiling silently becomes a percentage of
/// zero — or of a default nobody chose. Both paths are tried and a miss is
/// LOUD; the caller must not print a percentage without a ceiling.
fn cgroup_limit_gib() -> Result<f64, String> {
    const PATHS: [&str; 2] = [
        "/sys/fs/cgroup/memory.max",                   // v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes", // v1
    ];
    let mut tried = Vec::new();
    for p in PATHS {
        match std::fs::read_to_string(p) {
            Ok(s) if s.trim() == "max" => tried.push(format!("{p}=max (unlimited)")),
            Ok(s) => match s.trim().parse::<u64>() {
                Ok(b) => return Ok(b as f64 / (1024.0 * 1024.0 * 1024.0)),
                Err(e) => tried.push(format!("{p} unparsable: {e}")),
            },
            Err(e) => tried.push(format!("{p}: {e}")),
        }
    }
    Err(tried.join("; "))
}

/// Print the census and the chip padding panel for one node program.
///
/// The panel is a FORWARD instrument, not a diagnostic: printed from the working
/// fan-in-2 configuration it named `LFM_HASH`, and `LFM_HASH` is the table that
/// stepped 2^20 → 2^21 and put fan-in 3 over the card at 25.95 GiB of ~26.2
/// usable. Print it at EVERY level.
fn census_and_panel(program: &LfmProgram, label: &str, fan_in: usize) -> (u64, usize) {
    const EMPTY_MACHINE_CELLS: u64 = 26_482_828;
    let (main, aux) =
        super::airs::lfm_cell_counts_with_hasher(program, crate::hash_pin::BLOCK_HASHER);
    let cells = main + 3 * aux;
    println!(
        "   ★ CENSUS {label}: {cells} cells ({} instructions), floor {:.1}%",
        program.instrs.len(),
        100.0 * EMPTY_MACHINE_CELLS as f64 / cells as f64,
    );
    let panel = super::airs::lfm_chip_census_with_hasher(program, crate::hash_pin::BLOCK_HASHER);
    let step = (fan_in + 1) as f64 / fan_in as f64;
    for c in &panel {
        println!(
            "     {:<14} {:>10}/{:>10}  headroom {:>5.1}%  {}  cliff +{} cells",
            c.name,
            c.real_rows,
            c.rows,
            100.0 * c.headroom(),
            if c.at_risk() { "AT RISK" } else { "fixed  " },
            c.cliff_cost(),
        );
    }
    let stepping: Vec<&str> = panel
        .iter()
        .filter(|c| c.at_risk() && c.real_rows as f64 * step > c.rows as f64)
        .map(|c| c.name)
        .collect();
    println!("     ⇒ at {step:.3}× the workload these would STEP: {stepping:?}");
    (cells, program.instrs.len())
}

/// The child index range each node of a level consumes, in order.
///
/// Extracted from the driver rather than written inline because it is the one
/// piece of the tree that is pure arithmetic and can therefore be WRONG for
/// free: an off-by-one here mis-groups children and the driver only notices at
/// its consumption assert, which is twenty-odd minutes of wraps into a box run.
/// [`the_level_groups_tile_every_child`] pins it in milliseconds.
fn level_groups(arities: &[usize]) -> Vec<std::ops::Range<usize>> {
    let mut cursor = 0usize;
    arities
        .iter()
        .map(|a| {
            let r = cursor..cursor + a;
            cursor += a;
            r
        })
        .collect()
}

/// ★ The driver's grouping tiles its children exactly, at every level and every
/// shape — no gap, no overlap, nothing left over.
///
/// Pure arithmetic, so it runs on every suite. The driver asserts the same
/// invariant at runtime, but discovering it there costs a base prove and 19 wrap
/// proves first.
#[test]
fn the_level_groups_tile_every_child() {
    use super::per_table_aggregator::tree_shape;

    for epochs in [1usize, 2, 3, 5, 10, 19, 36, 64, 97] {
        for fan_in in [2usize, 3, 4] {
            let mut n = epochs;
            for (li, level) in tree_shape(epochs, fan_in).iter().enumerate() {
                let groups = level_groups(&level.arities);
                assert_eq!(
                    groups.len(),
                    level.arities.len(),
                    "{epochs}@{fan_in} level {li}: one group per node"
                );
                let mut expect = 0usize;
                for g in &groups {
                    assert_eq!(g.start, expect, "{epochs}@{fan_in} level {li}: no gap");
                    assert!(!g.is_empty(), "{epochs}@{fan_in} level {li}: no empty node");
                    expect = g.end;
                }
                assert_eq!(
                    expect, n,
                    "{epochs}@{fan_in} level {li}: the groups must consume every \
                     one of the {n} children and no more"
                );
                n = level.arities.len();
            }
            assert_eq!(n, 1, "{epochs}@{fan_in}: the interior must close to one");
        }
    }
}

/// ★★★ THE PRODUCTION TREE — every level, from 19 epoch wraps to one proof.
///
/// # What this is, and what it is NOT
///
/// It composes the tree's **INTERIOR**: levels 1..k of `emit_node`, closing to a
/// single level-k node. ⛔ It is **not** the block-artifact ROOT, which takes
/// `fan_in + 1` children (the global wrap is the extra), performs the L2G
/// compare and the attestation join, and publishes a schema variable-length in
/// the block's page count — see `tree_shape`'s own doc. Closing the interior
/// binds the epoch proofs to each other; it does not by itself make an artifact
/// about the BLOCK. The two are not the same finish line and this one must not
/// be reported as the other.
///
/// # One arm per process
///
/// `LFM_TREE_LEVELS` names which levels THIS process proves — `all`, `N`, or
/// `lo-hi`, where level 0 is the epoch wraps. Levels below `lo` are LOADED from
/// `A_CACHE_DIR`; levels in `[lo, hi]` are PROVED; nothing above `hi` runs. The
/// base follows `lo`: proved when `lo == 0`, loaded otherwise.
///
/// ⛔ The RANGE is the experiment's name, so `A_BUNDLE_MODE` is REFUSED here
/// rather than ignored — a caller who exports a mode that has no effect believes
/// they set something. Naming the mode is what stops a harness picking its own
/// experiment off the filesystem.
///
/// ⚠ Only the PROVES are skippable, not the harvest chain: a level-k node's
/// program is a function of its children's shapes, so an arm at level k still
/// re-emits and re-harvests everything below it. That cost is measured and
/// printed rather than assumed — if it is minutes, per-level arms are not cheap
/// and the answer is a second cache layer, which is a build.
///
/// ```text
/// LFM_CENSUS_ELF=… LFM_CENSUS_INPUT=… LFM_CENSUS_EPOCH_LOG2=21 \
/// LAMBDA_VM_MAX_ROWS_LOG2=21 LAMBDA_VM_MEMPOOL_RELEASE_MB=0 \
/// A_CACHE_DIR=/root/a_tree LFM_TREE_LEVELS=all \
/// cargo test --release -p lambda-vm-prover --features cuda --lib \
///   lfm::per_table_aggregator_tests::the_production_tree_composes_to_a_root -- \
///   --ignored --exact --nocapture
/// ```
///
/// # ★★★ The block-artifact ROOT
///
/// `LFM_TREE_PROVE_ROOT=1` proves the root over a tree that already exists: it
/// LOADS every interior level, exactly as `LFM_TREE_SIZE_ROOT=1` does, and the
/// only thing it proves is the root. It needs `A_CACHE_DIR`, refuses
/// `LFM_TREE_LEVELS` (which would be set and silently ignored), and refuses
/// `LFM_TREE_SIZE_ROOT` (a different experiment: that one emits BOTH options and
/// proves neither).
///
/// `LFM_TREE_ROOT_OPTION=A|B` is a NAMED INPUT with no default — `A` takes the
/// top interior level's nodes, `B` takes the single node above them, and both
/// take the global child. The sizing arm decides it. Under `A` the run stops one
/// level short, because that level is the one the root replaces, so
/// `node-<top>-0.rkyv` is neither proved nor loaded.
///
/// `LFM_TREE_ROOT_MODE=prove|load`, in the parent's style: unset it proves and
/// saves wherever a cache directory exists, and `CacheMode::Prove` still refuses
/// to overwrite `block-root.rkyv`.
///
/// ⚠ The global child must be NAMED too, or its stages pick their own modes: a
/// root run over a `k = 2` cache wants `LFM_TREE_GLOBAL_K=2` with the slices
/// loading (the level-0 default) and `LFM_TREE_PARENT_MODE=load`, or the parent
/// stage refuses to overwrite the `global-parent.rkyv` it is meant to consume.
///
/// ```text
/// … A_CACHE_DIR=/root/a_tree LFM_TREE_PROVE_ROOT=1 LFM_TREE_ROOT_OPTION=B \
/// LFM_TREE_GLOBAL_K=2 LFM_TREE_PARENT_MODE=load \
/// cargo test --release -p lambda-vm-prover --features cuda --lib \
///   lfm::per_table_aggregator_tests::the_production_tree_composes_to_a_root -- \
///   --ignored --exact --nocapture
/// ```
#[test]
#[ignore = "box tier, production scale: composes the whole interior tree"]
fn the_production_tree_composes_to_a_root() {
    use super::epoch_tests::{EpochInputs, Publishes};
    use super::per_table_aggregator::{FAN_IN, SchemaLayout, tree_node_count, tree_shape};
    use super::proof::lfm_prove;
    use super::registry::build_artifacts_with_hasher;
    use std::time::Instant;

    // ⛔ THE DEVICE, ASSERTED IN-PROCESS — see the leaf measurement's own note.
    // A CPU run completes, reads legibly, and biases every host figure the wrong
    // way, so a red would be an artefact of the build rather than a fact.
    if !cfg!(feature = "cuda") {
        panic!(
            "the production tree requires `--features cuda`. Without it this \
             proves on the CPU and answers a different question"
        );
    }
    for var in ["LFM_CENSUS_ELF", "LFM_CENSUS_INPUT"] {
        assert!(
            std::env::var(var).is_ok(),
            "{var} must name a file: this composes the PRODUCTION tree, and a \
             silent fixture fallback would report a fixture number under a \
             production name"
        );
    }
    assert!(
        std::env::var("A_BUNDLE_MODE").is_err(),
        "A_BUNDLE_MODE is set, and this test does NOT consult it — the level \
         range names the experiment. Unset it and use LFM_TREE_LEVELS (`all`, \
         `N`, or `lo-hi`); a mode that silently has no effect is worse than none"
    );

    let fan_in: usize = match std::env::var("LFM_CENSUS_FAN_IN") {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|e| panic!("LFM_CENSUS_FAN_IN must be an integer: {e}")),
        Err(_) => FAN_IN,
    };
    assert!(
        (2..=4).contains(&fan_in),
        "LFM_CENSUS_FAN_IN must be in 2..=4, got {fan_in}"
    );

    let spec = std::env::var("LFM_TREE_LEVELS").unwrap_or_else(|_| "all".to_string());
    // ★★ SIZING MODE: load every level, prove NOTHING, emit both root options
    // and print their censuses and chip panels.
    //
    // ⛔ WHY IT EXISTS. Whether the root's `LFM_HASH` crosses 2^20 -> 2^21 decides
    // whether the root is a ~500M-cell node that fits or a ~900M-cell one within
    // 0.5% of the fan-in-3 node that OOM'd at 97.4% of the card. And it CANNOT be
    // settled by scaling a rate: every level of the measured tree carries the
    // SAME sub-proof count (22), so those four points contain no information
    // about the per-sub-proof coefficient the root's 41 needs. The only place
    // sub-proof count varies at all is the wrap -> node transition, and that is
    // confounded with a change of child kind.
    // ⇒ Emit both and read the panels. `census_and_panel` is a pure function of a
    // compiled program, so this costs a cache load and seconds of emission.
    let size_root = std::env::var("LFM_TREE_SIZE_ROOT").is_ok();
    // ★★★ THE BLOCK-ARTIFACT ROOT — the last proof of the campaign, and the one
    // stage that answers *what does this artifact claim about block N?*
    //
    // ⛔ IT IS NAMED, NOT INFERRED FROM `hi == top`. Every launch line that has
    // ever built this tree ends with the interior closed, and making those runs
    // start emitting a root would change what an unset knob does — and would do
    // it at the END of an hour of proving, where a wrong option is discovered
    // after the work it invalidates.
    // ⇒ `LFM_TREE_PROVE_ROOT=1` names the experiment, and like `LFM_TREE_SIZE_
    // ROOT` it LOADS every interior level: the root is proved from a tree that
    // already exists, and proving one here would be a different and much longer
    // experiment than the one asked for.
    let prove_root = std::env::var("LFM_TREE_PROVE_ROOT").is_ok();
    assert!(
        !(size_root && prove_root),
        "LFM_TREE_SIZE_ROOT and LFM_TREE_PROVE_ROOT are two different \
         experiments — one emits BOTH root options and proves neither, the other \
         proves the ONE option it was given. Name one"
    );
    assert!(
        !prove_root || std::env::var("LFM_TREE_LEVELS").is_err(),
        "LFM_TREE_PROVE_ROOT loads every interior level, so LFM_TREE_LEVELS has \
         no effect here. Unset it: a knob that is set and silently ignored is the \
         failure A_BUNDLE_MODE's own refusal exists for — the caller believes \
         they named an experiment and did not"
    );
    // ⛔ THE OPTION IS AN INPUT AND CARRIES NO DEFAULT. The sizing arm decides
    // it; it changes the root's child count and therefore its sub-proof count,
    // and a default would silently become the answer to a question a measurement
    // was supposed to settle. `RootOption::parse` refuses everything but `A` and
    // `B`, the empty string included.
    let root_option: Option<super::block_root::RootOption> = match (
        prove_root,
        std::env::var("LFM_TREE_ROOT_OPTION").ok().as_deref(),
    ) {
        (false, None) => None,
        (false, Some(v)) => panic!(
            "LFM_TREE_ROOT_OPTION=`{v}` is set but LFM_TREE_PROVE_ROOT is not, so \
             this run emits no root and the option has no effect. Set \
             LFM_TREE_PROVE_ROOT=1 to prove one, or unset the option"
        ),
        (true, None) => panic!(
            "LFM_TREE_PROVE_ROOT is set and LFM_TREE_ROOT_OPTION is NOT. The root \
             takes either the top interior level's nodes (`A`) or the single node \
             above them (`B`) plus the global child, and the two are different \
             programs with different sub-proof counts. The sizing arm decides \
             which; this driver must not guess, and must not carry a default that \
             silently becomes the answer"
        ),
        (true, Some(v)) => Some(super::block_root::RootOption::parse(v).unwrap_or_else(|e| {
            panic!("LFM_TREE_ROOT_OPTION: {e}")
        })),
    };
    let (lo, hi_req): (usize, Option<usize>) = if size_root || prove_root {
        // `lo` above every level means no stage proves.
        (usize::MAX, None)
    } else {
        match spec.as_str() {
            "all" => (0, None),
            s => match s.split_once('-') {
                Some((a, b)) => (
                    a.parse().expect("LFM_TREE_LEVELS lo must be an integer"),
                    Some(b.parse().expect("LFM_TREE_LEVELS hi must be an integer")),
                ),
                None => {
                    let n = s
                        .parse()
                        .expect("LFM_TREE_LEVELS must be `all`, `N` or `lo-hi`");
                    (n, Some(n))
                }
            },
        }
    };
    let cache_dir = std::env::var("A_CACHE_DIR").ok();
    assert!(
        !prove_root || cache_dir.is_some(),
        "LFM_TREE_PROVE_ROOT needs A_CACHE_DIR: it proves the root over a tree, a \
         global child and a base that have already been proved, and there is \
         nowhere to load them from"
    );
    assert!(
        !size_root || cache_dir.is_some(),
        "LFM_TREE_SIZE_ROOT needs A_CACHE_DIR: it sizes the root from a tree that \
         has already been proved, and proving one here would be a different and \
         much longer experiment than the one asked for"
    );
    assert!(
        lo == 0 || cache_dir.is_some(),
        "LFM_TREE_LEVELS starts at {lo}, so levels below it must be LOADED — but \
         A_CACHE_DIR is unset. Proving them instead would silently make this a \
         different (and much longer) experiment"
    );
    // ⛔ THE ROOT NEEDS ITS OWN MODE, for the parent's reason: the launch line
    // that produces a root LOADS everything under it and must PROVE the root, so
    // a shared mode would send it to load a `block-root.rkyv` that has never
    // existed and the refusal would name the wrong stage. Unset, it proves and
    // saves wherever a cache directory exists — the only experiment a run with no
    // root on disk can be running — and `CacheMode::Prove` still REFUSES to
    // overwrite, so a second run over a populated cache is a refusal rather than
    // a silent re-prove or a silent load.
    let root_mode = match std::env::var("LFM_TREE_ROOT_MODE").ok().as_deref() {
        None if cache_dir.is_some() => CacheMode::Prove,
        None => CacheMode::Off,
        Some("prove") => CacheMode::Prove,
        Some("load") => CacheMode::Load,
        Some(other) => panic!("LFM_TREE_ROOT_MODE must be `prove` or `load`, got `{other}`"),
    };
    // Levels below `lo` load; levels in the range prove, and save when a cache
    // directory exists. `CacheMode::Prove` refuses an existing file, so a re-run
    // over a populated directory is a refusal rather than an overwrite.
    let stage_mode = |level: usize| -> CacheMode {
        if level < lo {
            CacheMode::Load
        } else if cache_dir.is_some() {
            CacheMode::Prove
        } else {
            CacheMode::Off
        }
    };

    let inputs = EpochInputs::from_env();
    let inner = crate::recursion::Preset::Blowup4.options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let ceiling = cgroup_limit_gib();
    println!(
        "★★★ PRODUCTION TREE (INTERIOR ONLY — not the block-artifact root)\n   \
         guest {}, {} input bytes, 2^{} cycles/epoch, fan-in {fan_in}\n   \
         inner blowup {} / {} q · wrap blowup {} / {} q\n   \
         levels: prove {lo}..={} · cache {}\n   cgroup ceiling: {}",
        inputs.label,
        inputs.private_input.len(),
        inputs.epoch_log2,
        inner.blowup_factor,
        inner.fri_number_of_queries,
        wrap_opts.blowup_factor,
        wrap_opts.fri_number_of_queries,
        match hi_req {
            Some(h) => h.to_string(),
            None => "top".to_string(),
        },
        cache_dir.as_deref().unwrap_or("<none>"),
        match &ceiling {
            Ok(g) => format!("{g:.2} GiB"),
            Err(why) => format!("UNKNOWN — {why}"),
        },
    );

    let whole_run = HostSampler::start();
    let t_all = Instant::now();

    // ---- the base. It is needed by EVERY arm: a level-k node's program is a
    // function of its children's shapes, so even a top-level arm re-derives the
    // whole chain from the epochs. Only the PROVES are skippable.
    let t = Instant::now();
    let bundle = cached_bundle(
        if lo == 0 {
            stage_mode(0)
        } else {
            CacheMode::Load
        },
        stage_path(cache_dir.as_deref(), "bundle"),
        || {
            crate::continuation::prove_continuation(
                &inputs.elf_bytes,
                &inputs.private_input,
                inputs.epoch_log2,
                &inner,
            )
            .expect("the block must prove")
        },
    );
    println!(
        "   base: {} epochs in {:.1}s",
        bundle.num_epochs(),
        t.elapsed().as_secs_f64()
    );
    mark("AFTER the base (this live figure is L_bundle)");

    let shape = tree_shape(bundle.num_epochs(), fan_in);
    let top = shape.len();
    let hi = hi_req.unwrap_or(top).min(top);
    assert!(
        lo <= hi || size_root || prove_root,
        "LFM_TREE_LEVELS {lo}-{hi} is empty; the tree has {top} node levels"
    );
    // ★ OPTION A STOPS ONE LEVEL SHORT, because the level it would walk is the
    // level the root REPLACES. So a run under A never loads, harvests or
    // verifies `node-{top}-0.rkyv` — it is not a child of anything — and after
    // the loop `children` IS the root's interior children, with no second
    // capture and no clobbering. Under B the loop closes the tree as always and
    // `children` is the single node the root sits above.
    let hi = match root_option {
        Some(o) if o.replaces_top() => top.saturating_sub(1),
        _ => hi,
    };
    println!(
        "   ★ SHAPE from {} epochs at fan-in {fan_in}: {top} levels, {} nodes",
        bundle.num_epochs(),
        tree_node_count(&shape),
    );
    for (i, level) in shape.iter().enumerate() {
        let short = level.arities.iter().filter(|a| **a < fan_in).count();
        println!(
            "     level {}: {} nodes ({short} short)",
            i + 1,
            level.arities.len()
        );
    }

    // ---- level 0: one wrap per epoch.
    let t_level = Instant::now();
    // ★ THREE PARALLEL VECTORS, not a vector of structs, because `node_program`
    // and `prove_node_as_child` take `&[RealChild]` and `&[SchemaLayout]` — a
    // contiguous slice of each is exactly what a node's child group is, and
    // keeping them parallel means a group is a subslice rather than a clone of
    // every child's harvest. The label run is what distinguishes the levels and
    // nothing else does: a wrap carries ONE epoch label, a node carries the
    // FIRST and LAST of its subtree, and a parent's range runs from the first
    // child's first to the last child's last — which is why contiguity across
    // siblings falls out of the pins rather than needing a check of its own.
    let mut children: Vec<RealChild> = Vec::with_capacity(bundle.num_epochs());
    let mut layouts: Vec<SchemaLayout> = Vec::with_capacity(bundle.num_epochs());
    let mut labels: Vec<Vec<u64>> = Vec::with_capacity(bundle.num_epochs());
    for k in 0..bundle.num_epochs() {
        let e = super::epoch_tests::real_epoch_from_continuation(
            &inner,
            &inputs.elf_bytes,
            &bundle,
            k,
            None,
        )
        .expect("every epoch must reconstruct from proofs alone");
        let out_halves = e.statement.public_output_len.div_ceil(4);
        if k == 0 {
            // ★ FREE, AND IT SIZES THE BLOCK-ARTIFACT ROOT. The attestation fold
            // is already emitted inside every wrap at this page count, and the
            // fold's hashed length is linear in it — so this one number is the
            // main cost driver of the root that does not yet exist.
            let shape = super::programs::ProgramIdShape {
                num_pages: e.num_pages(),
            };
            println!(
                "   ★ BLOCK FACTS: {} touched pages ⇒ attestation fold hashes {} \
                 bytes; epoch public output {} halves",
                shape.num_pages,
                shape.byte_len(),
                out_halves,
            );
        }
        let shapes: Vec<&super::epoch::TableChallengeShape> =
            e.tables.iter().map(|h| &h.shape).collect();
        assert_samplable(&format!("inner epoch {k}"), &shapes);
        let program =
            super::epoch_tests::epoch_program_publishing(&e, true, Publishes::Aggregation);
        let arenas = super::epoch_tests::epoch_arena_words(&e, true);
        let artifacts =
            build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
        let proved = cached_stage(
            stage_mode(0),
            stage_path(cache_dir.as_deref(), &format!("wrap-{k}")),
            &format!("wrap {k}"),
            || {
                lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
                    .expect("the epoch wrap must prove")
            },
        );
        let layout = SchemaLayout::wrap(out_halves);
        layout.assert_covers(proved.public_words.len());
        children.push(real_child(artifacts, wrap_opts.clone(), &proved));
        layouts.push(layout);
        labels.push(vec![crate::tables::local_to_global::epoch_label(k as u64)]);
    }
    println!(
        "   level 0: {} wraps in {:.1}s",
        children.len(),
        t_level.elapsed().as_secs_f64()
    );

    // ---- level 0, the OTHER child: the GLOBAL WRAP.
    //
    // ★ The root takes `fan_in + 1` children and this is the extra one. It is
    // not an epoch wrap and never appears at an interior level, so a cache that
    // holds `bundle` + N epoch wraps + every node is complete for the INTERIOR
    // and missing exactly the child that makes a root a root.
    //
    // ✓ Everything it needs is already in the bundle: `ContinuationProof` carries
    // `global`, `num_private_input_pages` and `touched_page_bases`
    // (`continuation.rs:581-591`), and `real_global` harvests straight from it —
    // so no re-prove of the base is ever required to produce this.
    // ⚠ But the global wrap PROOF is new work: `the_global_verifier_leg_runs_and_
    // rejects_tampers` only EXECUTES this program, it has never proved it.
    let t = Instant::now();
    let g = real_global(&inputs.elf_bytes, &bundle, &inner);

    // ★★ THE GO/NO-GO ON SLICING, and it proves NOTHING so it cannot abort.
    //
    // The unsliced global wrap aborts at `LFM_HASH` 2^21 x 329 = 26.22 GiB — over
    // the 16000 budget AND over the 80% default of 25.12, which is why no budget
    // change alone could clear it. Halving the WALKS should halve that to
    // 13.11 GiB, under 15.625. ⇒ Whether it does is one emission away, and the
    // whole partial-bus-sum story rests on it.
    //
    // ⛔ FALSIFIER, as registered: `LFM_HASH` still at 2^21 at k = 2 means the
    // walks are NOT what dominates and the mechanism is wrong. Report the miss;
    // do not repair the estimate.
    if std::env::var("LFM_TREE_SIZE_GLOBAL").is_ok() {
        println!(
            "\n★★★ SIZING THE GLOBAL WRAP — emitted, never proved. {} tables \
             ({} L2G), {} epochs.",
            g.tables.len(),
            g.num_l2g,
            bundle.num_epochs(),
        );
        for k in [1usize, 2] {
            let partition = super::global_split::SlicePartition::even(g.tables.len(), k);
            for slice in 0..k {
                let (lo, hi) = partition.slice(slice);
                let t = Instant::now();
                let program = global_slice_program(&g, &partition, slice);
                let label = format!("global k={k} slice {slice} (tables {lo}..{hi})");
                println!("\n── {label}: emitted in {:.1}s", t.elapsed().as_secs_f64());
                census_and_panel(&program, &label, fan_in);
            }
        }
        println!(
            "\n⇒ COMPARE `LFM_HASH`'s COMMITTED HEIGHT at k=1 against k=2. 2^21 -> \
             2^20 confirms the mechanism and clears the budget (26.22 -> 13.11 GiB \
             against 15.625). Still 2^21 REFUTES it."
        );

        // ★★ PROVE ONE SLICE, STANDALONE — the measurement the emit cannot make.
        //
        // The emit says the R1 barrier should be 12.98 GiB against 15.625. Whether
        // that PLUS the slice's own incremental PLUS the ~5 GiB floor clears a
        // 32 GiB card is a different question, and this campaign has been wrong
        // about exactly that arithmetic in both directions today.
        //
        // ✓ Nothing about it needs the parent: a slice is a program, `lfm_prove`
        // takes a program, and the arenas are unchanged — `per_table` declares for
        // ALL tables and only the LEGS are restricted, so `global_arena_words`
        // still matches declaration order exactly. The unread declarations cost
        // hint words, not chip rows, which is why the census halved.
        if let Ok(which) = std::env::var("LFM_TREE_PROVE_SLICE") {
            let slice: usize = which
                .parse()
                .expect("LFM_TREE_PROVE_SLICE must be an integer");
            let partition = super::global_split::SlicePartition::even(g.tables.len(), 2);
            assert!(
                slice < partition.k(),
                "slice {slice} does not exist at k={}",
                partition.k()
            );
            let (lo, hi) = partition.slice(slice);
            let program = global_slice_program(&g, &partition, slice);
            let arenas = global_arena_words(&g);
            let t = Instant::now();
            let artifacts =
                build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
            println!(
                "\n★ PROVING global slice {slice} (tables {lo}..{hi}): \
                 build_artifacts {:.1}s",
                t.elapsed().as_secs_f64()
            );
            let sampler = HostSampler::start();
            #[cfg(feature = "cuda")]
            stark::gpu_lde::reset_all_gpu_call_counters();
            let t = Instant::now();
            let proved = lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
                .expect("★ THE GLOBAL SLICE MUST PROVE");
            let prove_secs = t.elapsed().as_secs_f64();
            let (peak, at) = sampler.stop();
            #[cfg(feature = "cuda")]
            {
                let total: u64 = stark::gpu_lde::gpu_lde_calls()
                    + stark::gpu_lde::gpu_merkle_tree_calls()
                    + stark::gpu_lde::gpu_fri_calls();
                assert!(
                    total > 0,
                    "the slice prove reached the device ZERO times — it ran on the \
                     host even though cuda is compiled in, so the peak is not the \
                     production figure"
                );
                println!("   GPU dispatches during the SLICE prove: {total}");
            }
            let t = Instant::now();
            assert!(
                super::proof::verify_against_artifacts(
                    &artifacts,
                    &proved.proof,
                    &proved.public_words,
                    &wrap_opts
                ),
                "the global slice's proof must verify"
            );
            println!(
                "\n★★★ GLOBAL SLICE {slice} PROVED AND VERIFIED\n   prove \
                 {prove_secs:.1}s · verify {:.2}s · {} published words\n   host \
                 peak {peak:.3} GiB at t={at:.1}",
                t.elapsed().as_secs_f64(),
                proved.public_words.len(),
            );
        }
        return;
    }

    // ⛔ `k` IS A STAGE INPUT, AND ITS DEFAULT IS THE STAGE THAT RAN BEFORE IT.
    //
    // The unsliced wrap ABORTS on the device: `LFM_HASH` at 2^21 x 329 cols wants
    // 26.22 GiB of a 32 GiB card, which no budget change clears. At k = 2
    // `LFM_HASH` lands at 2^20 and one slice PROVED AND VERIFIED at 18,663 MiB
    // (57.2% of the card) — measured by the `LFM_TREE_PROVE_SLICE` arm above, not
    // projected.
    //
    // ⛔ BUT THE PARENT THAT FOLDS k SLICES DOES NOT EXIST. So k > 1 proves and
    // CACHES the slices and then REFUSES to continue, rather than feeding a slice
    // into a path built for the k = 1 wrap; see the refusal after the loop.
    // ⇒ Unset means k = 1, which is this stage exactly as it was: one program
    // with the bus closed against zero, one cache entry named `global-wrap`, one
    // child handed onward.
    let global_k: usize = match std::env::var("LFM_TREE_GLOBAL_K") {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|e| panic!("LFM_TREE_GLOBAL_K must be an integer: {e}")),
        Err(_) => 1,
    };
    // ⛔ AND NO SECOND BOUND CHECK HERE. `SlicePartition::even` refuses a `k`
    // outside `1..=num_tables` and refuses any partition that does not tile — it
    // IS the single source for every bound, and a `k` this stage validated
    // separately would be a second copy of the rule the constructor enforces.
    let partition = super::global_split::SlicePartition::even(g.tables.len(), global_k);
    // ⛔ THE GLOBAL WRAP NEEDS ITS OWN MODE, and it is NOT cache-if-present.
    //
    // Every other level-0 stage shares one mode, which is right: the bundle and
    // the 19 wraps are produced together. The global wrap is not — it was added
    // to the driver after a cache had already been built, so a populated cache
    // can legitimately hold every wrap and lack this one. Loading the wraps while
    // proving this stage is therefore a real, nameable experiment, and the only
    // alternatives were both wrong: `all` refuses at the bundle (Prove mode, file
    // exists) and a load-everything arm refuses here.
    // ⇒ `LFM_TREE_GLOBAL_MODE=prove|load` NAMES it. Unset, it follows level 0, so
    // a fresh `all` proves it and a sizing arm loads it, exactly as before. What
    // it must never become is "prove it if it happens to be missing" — that is
    // the harness picking its own experiment off the disk.
    let global_mode = match std::env::var("LFM_TREE_GLOBAL_MODE").ok().as_deref() {
        None => stage_mode(0),
        Some("prove") => CacheMode::Prove,
        Some("load") => CacheMode::Load,
        Some(other) => panic!("LFM_TREE_GLOBAL_MODE must be `prove` or `load`, got `{other}`"),
    };
    // ⛔ THE PARENT NEEDS ITS OWN MODE, for the global stage's reason and a
    // sharper one: the launch line that produces a parent LOADS the k slices an
    // earlier run cached (`LFM_TREE_GLOBAL_MODE=load`) and must PROVE the parent.
    // Sharing one mode would send it to load a `global-parent.rkyv` that has
    // never existed, and the refusal would name the wrong stage.
    // ⇒ `LFM_TREE_PARENT_MODE=prove|load` NAMES it. Unset, it proves and saves
    // wherever a cache directory exists — the only experiment a run with no
    // parent on disk can be running — and `CacheMode::Prove` still REFUSES to
    // overwrite, so a second run over a populated cache is a refusal rather than
    // a silent re-prove or a silent load.
    let parent_mode = match std::env::var("LFM_TREE_PARENT_MODE").ok().as_deref() {
        None if cache_dir.is_some() => CacheMode::Prove,
        None => CacheMode::Off,
        Some("prove") => CacheMode::Prove,
        Some("load") => CacheMode::Load,
        Some(other) => panic!("LFM_TREE_PARENT_MODE must be `prove` or `load`, got `{other}`"),
    };
    // ★ ONE ARENA SET SERVES EVERY SLICE. `global_slice_program` declares arenas
    // for ALL tables and restricts only the verification LEGS, so declaration
    // order is identical at every `k` and for every slice. The unread
    // declarations cost hint words, not chip rows — which is why the census halves
    // while the arenas do not change at all.
    let arenas = global_arena_words(&g);
    // ⛔ ONE LAYOUT PER SHAPE, SELECTED BY THE PARTITION THE EMITTER WAS HANDED —
    // NOT one assert widened until both shapes pass. A slice publishes a partial
    // bus sum the k = 1 set does not contain, and every index the root's L2G
    // compare reads is shifted by a wrong layout, so a wrong layout is silent and
    // downstream. See `block_root::GlobalPublishes`, which reads the shape off the
    // same `partition.k()` the emitter branches on.
    let g_publishes = super::block_root::GlobalPublishes::of(
        &partition,
        super::block_root::GlobalLayout {
            num_epochs: g.num_l2g,
            lanes_per_root: super::proof_arena::lanes_per_root(),
        },
    );
    let mut global_stages: Vec<(super::registry::LfmArtifacts, super::proof::LfmProof)> =
        Vec::with_capacity(partition.k());
    let mut slice_report: Vec<String> = Vec::with_capacity(partition.k());
    for i in 0..partition.k() {
        let (lo_t, hi_t) = partition.slice(i);
        // ⚠ AT k = 1 THIS *IS* `global_verifier_program`, which is defined as
        // exactly this call — so the default path emits the program it always
        // emitted rather than a second spelling of it.
        let program = global_slice_program(&g, &partition, i);
        let artifacts =
            build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
        // ⛔ ONE CACHE NAME PER SHAPE, for the same reason there is one layout per
        // shape. The k = 1 wrap closes its bus against zero and a slice does not,
        // so they are DIFFERENT PROGRAMS with different `program_id`s: a slice
        // loaded from `global-wrap`, or a wrap loaded from `global-wrap-0`, would
        // be a proof of a statement nobody asked for.
        // ⇒ k = 1 KEEPS THE LEGACY NAME, so every cache already on disk still
        // resolves — and it must, because `LFM_TREE_GLOBAL_MODE=load` REFUSES a
        // missing entry rather than quietly proving one.
        let stage = if partition.k() == 1 {
            "global-wrap".to_string()
        } else {
            format!("global-wrap-{i}")
        };
        // ⛔ Its own message. A root that cannot find its GLOBAL child is a
        // different failure from one that cannot find a node child, and a generic
        // cache miss would report the reader's hypothesis rather than what
        // happened.
        let label = if partition.k() == 1 {
            "the GLOBAL wrap (the root's extra child)".to_string()
        } else {
            format!(
                "the GLOBAL slice {i} of {} (tables {lo_t}..{hi_t})",
                partition.k()
            )
        };
        #[cfg(feature = "cuda")]
        stark::gpu_lde::reset_all_gpu_call_counters();
        let sampler = HostSampler::start();
        let t_stage = Instant::now();
        let proved = cached_stage(
            global_mode,
            stage_path(cache_dir.as_deref(), &stage),
            &label,
            || {
                lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
                    .unwrap_or_else(|e| panic!("★ {label} MUST PROVE: {e:?}"))
            },
        );
        let stage_secs = t_stage.elapsed().as_secs_f64();
        let (peak, at) = sampler.stop();
        assert_eq!(
            proved.public_words.len(),
            g_publishes.total(),
            "{label}: ITS LAYOUT DOES NOT DESCRIBE IT. It published {} words, the \
             layout says {} — {}. Every index the root's L2G compare reads is \
             shifted by this, so it must abort here rather than compare the wrong \
             words",
            proved.public_words.len(),
            g_publishes.total(),
            g_publishes.describe(),
        );
        // ⛔ EVERY GLOBAL STAGE IS VERIFIED **HERE**, AND NOT BY A CALL FURTHER
        // DOWN. At k = 1 `real_child` also verifies — its doc is right that
        // nothing downstream may read a proof the verifier would reject — so this
        // is one extra verify on the default path, and it is worth its seconds:
        //
        // - at k > 1 there IS no `real_child` call (the stage refuses below), so
        //   without this the slices would be CACHED and REPORTED unverified;
        // - under `LFM_TREE_GLOBAL_MODE=load` the proof came off the disk through
        //   `rkyv` and has never been checked in this process at all;
        // - and an invariant that lives in a call this stage merely happens to
        //   make is one somebody can delete without noticing. This one is the
        //   stage's own.
        let t_verify = Instant::now();
        assert!(
            super::proof::verify_against_artifacts(
                &artifacts,
                &proved.proof,
                &proved.public_words,
                &wrap_opts
            ),
            "{label}: ITS PROOF DOES NOT VERIFY. Nothing may be cached, reported \
             or handed onward from a proof production would reject"
        );
        let verify_secs = t_verify.elapsed().as_secs_f64();
        // ⚠ ONLY WHERE THIS PROCESS ACTUALLY PROVED. Under `load` no kernel runs
        // and a zero count is the correct observation, so a blanket assert here
        // would fire on a legitimate load arm.
        #[cfg(feature = "cuda")]
        {
            if global_mode != CacheMode::Load {
                let calls = stark::gpu_lde::gpu_lde_calls()
                    + stark::gpu_lde::gpu_merkle_tree_calls()
                    + stark::gpu_lde::gpu_fri_calls();
                assert!(
                    calls > 0,
                    "{label} reached the device ZERO times — it proved on the HOST \
                     with cuda compiled in, so its peak is not a production figure"
                );
                println!("     GPU dispatches during {label}: {calls}");
            }
        }
        println!(
            "     {label}: stage {stage_secs:.1}s · verify {verify_secs:.1}s · {} \
             published words{} · host peak {peak:.3} GiB at t={at:.1}",
            proved.public_words.len(),
            match g_publishes.partial_sum_word() {
                Some(w) => format!(" (partial bus sum at index {w})"),
                None => String::new(),
            },
        );
        slice_report.push(format!(
            "   slice {i} of {k}: tables {lo_t}..{hi_t} · stage {stage_secs:.1}s · \
             verify {verify_secs:.1}s · {} published words · host peak {peak:.3} \
             GiB at t={at:.1} · cache entry {stage}.rkyv",
            proved.public_words.len(),
            k = partition.k(),
        ));
        global_stages.push((artifacts, proved));
    }

    // ---- THE PARENT of the k slices.
    //
    // ★ AT k > 1 NO SLICE IS THE ROOT'S GLOBAL CHILD. A slice publishes a PARTIAL
    // bus sum the k = 1 layout does not contain, and every index the root's L2G
    // compare reads would be shifted by that one word. What IS the global child
    // is the PARENT: it verifies the k slice proofs, pins the partition through
    // the `program_id`s it embeds as constants, asserts every slice published the
    // same `(z, alpha)` and the same L2G roots, sums the partials, asserts ZERO,
    // and republishes the prefix.
    //
    // ⇒ Its published set is `2 + epochs x lanes` = `GlobalLayout::total()`,
    // exactly what the unsliced wrap published — a COINCIDENCE of arithmetic, not
    // a design (see `block_root::RootInputs`'s `global_child_layout`) — and that
    // is what lets everything below this point take ONE global child with no
    // branch of its own at either k.
    let (artifacts, proved) = if partition.k() > 1 {
        println!(
            "\n★★★ {} GLOBAL SLICES PROVED AND VERIFIED — folding them into the PARENT",
            partition.k()
        );
        for line in &slice_report {
            println!("{line}");
        }
        println!(
            "   cache directory: {}",
            cache_dir
                .as_deref()
                .unwrap_or("<none — NOTHING IS SAVED, this run's proofs die with the process>")
        );
        // ⛔ THE SLICE LAYOUT COMES OFF THE SAME `partition` THE EMITTER BRANCHED
        // ON, through the same `GlobalPublishes` the slices were asserted against.
        // A `k` this stage re-derived would be a second copy of the rule the
        // constructor enforces, and a layout picked independently of the program
        // it describes is exactly the silent-and-downstream failure
        // `GlobalPublishes` exists to prevent.
        let slice_layout = g_publishes
            .as_slice()
            .expect("k > 1 IS the slice shape, chosen by this same partition");
        let t_harvest = Instant::now();
        let slices: Vec<RealChild> = global_stages
            .into_iter()
            .map(|(a, p)| real_child(a, wrap_opts.clone(), &p))
            .collect();
        let harvest_secs = t_harvest.elapsed().as_secs_f64();
        let t_emit = Instant::now();
        // ⛔ THE SAME `partition`, not a second one built from the same numbers.
        let program = global_parent_program(&slices, &partition, slice_layout);
        println!(
            "   the GLOBAL PARENT: harvest {harvest_secs:.1}s · emitted in {:.1}s",
            t_emit.elapsed().as_secs_f64()
        );
        // ★ THE PANEL, AT THE PARENT TOO. The pre-registration says the parent's
        // census should be SMALL — k legs over slice proofs plus the sum — and
        // that a parent bigger than a level-1 node means the parent has become
        // the problem. That is read off `LFM_HASH`'s committed height here, not
        // inferred from a ratio.
        census_and_panel(&program, "the GLOBAL PARENT", fan_in);
        let arenas: Vec<Vec<LfmWord>> = slices.iter().flat_map(child_arena_words).collect();
        let artifacts =
            build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
        #[cfg(feature = "cuda")]
        stark::gpu_lde::reset_all_gpu_call_counters();
        let sampler = HostSampler::start();
        let t_stage = Instant::now();
        let proved = cached_stage(
            parent_mode,
            stage_path(cache_dir.as_deref(), "global-parent"),
            "the GLOBAL PARENT",
            || {
                lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
                    .unwrap_or_else(|e| panic!("★ THE GLOBAL PARENT MUST PROVE: {e:?}"))
            },
        );
        let stage_secs = t_stage.elapsed().as_secs_f64();
        let (peak, at) = sampler.stop();
        // ⛔ AGAINST THE PARENT'S OWN SHAPE — the SHARED prefix, with NO partial.
        // The parent asserts zero rather than publishing a sum, so it publishes
        // one word FEWER than each of its children. Checked against the slice
        // layout this would be off by exactly that word, and every L2G index the
        // root reads would shift by it: the wrong-layout failure is silent, and
        // it lands downstream.
        assert_eq!(
            proved.public_words.len(),
            slice_layout.shared.total(),
            "the GLOBAL PARENT published {} words, but it republishes the SHARED \
             prefix and NOTHING for the sum — z, alpha, then {} L2G roots x {} \
             lanes = {} words. Every index the root's L2G compare reads is \
             shifted by this, so it must abort here rather than compare the \
             wrong words",
            proved.public_words.len(),
            slice_layout.shared.num_epochs,
            slice_layout.shared.lanes_per_root,
            slice_layout.shared.total(),
        );
        // ⛔ VERIFIED HERE, for the reasons the slice loop gives: under
        // `LFM_TREE_PARENT_MODE=load` the proof came off the disk through `rkyv`
        // and has never been checked in this process, and an invariant living in
        // a call this stage merely happens to make is one somebody can delete
        // without noticing.
        let t_verify = Instant::now();
        assert!(
            super::proof::verify_against_artifacts(
                &artifacts,
                &proved.proof,
                &proved.public_words,
                &wrap_opts
            ),
            "the GLOBAL PARENT's proof DOES NOT VERIFY. Nothing may be cached, \
             reported or handed onward from a proof production would reject"
        );
        let verify_secs = t_verify.elapsed().as_secs_f64();
        // ⚠ ONLY WHERE THIS PROCESS ACTUALLY PROVED — under `load` no kernel runs
        // and zero is the correct observation.
        #[cfg(feature = "cuda")]
        {
            if parent_mode != CacheMode::Load {
                let calls = stark::gpu_lde::gpu_lde_calls()
                    + stark::gpu_lde::gpu_merkle_tree_calls()
                    + stark::gpu_lde::gpu_fri_calls();
                assert!(
                    calls > 0,
                    "the GLOBAL PARENT reached the device ZERO times — it proved \
                     on the HOST with cuda compiled in, so its peak is not a \
                     production figure"
                );
                println!("     GPU dispatches during the GLOBAL PARENT: {calls}");
            }
        }
        println!(
            "\n★★★ THE GLOBAL PARENT PROVED AND VERIFIED over {} slices\n   stage \
             {stage_secs:.1}s · verify {verify_secs:.1}s · {} published words \
             (the shared prefix, NO partial)\n   host peak {peak:.3} GiB at \
             t={at:.1}{}\n   cache entry global-parent.rkyv",
            partition.k(),
            proved.public_words.len(),
            match &ceiling {
                Ok(c) => format!(" ({:.1}% of {c:.2})", 100.0 * peak / c),
                Err(_) => String::new(),
            },
        );
        (artifacts, proved)
    } else {
        // k = 1: the loop ran once and its one stage IS the wrap the rest of the
        // tree expects, target zero and all.
        global_stages
            .pop()
            .expect("k = 1 ran the loop once and pushed its wrap")
    };
    println!(
        "   ★ THE ROOT'S GLOBAL CHILD is {}: {:.1}s to here, {} published words \
         ({} L2G roots), {} global sub-proofs, {} touched pages in the bundle",
        if partition.k() == 1 {
            "the UNSLICED global wrap".to_string()
        } else {
            format!("the PARENT of {} slices", partition.k())
        },
        t.elapsed().as_secs_f64(),
        proved.public_words.len(),
        g.num_l2g,
        g.tables.len(),
        bundle.touched_pages().len(),
    );
    let global_child = real_child(artifacts, wrap_opts.clone(), &proved);
    mark("AFTER the global child");

    // ⛔ A NAMED STOP, AND NOT A REFUSAL.
    //
    // The global child is not an interior level: the wrap (or, at k > 1, the
    // PARENT of k slices) finishes the extra child the root takes, and a run
    // whose whole job is to produce one has nothing to say about levels 1..n.
    // Without this it walks straight into them — and against a populated cache
    // EVERY interior stage is `Prove`, which refuses to overwrite. The run would
    // then end on `refusing to overwrite node-1-0.rkyv`: a true message about the
    // wrong thing, arriving after the stage it was launched for had already
    // succeeded, and reading like that stage failed.
    //
    // ⇒ `LFM_TREE_STOP_AFTER_GLOBAL=1` NAMES that experiment. Unset, nothing
    // changes and the tree composes exactly as before.
    if std::env::var("LFM_TREE_STOP_AFTER_GLOBAL").is_ok() {
        let (run_peak, run_at) = whole_run.stop();
        println!(
            "\n★★★ STOPPING AFTER THE GLOBAL CHILD, AS ASKED — NOT a failure and \
             NOT a refusal. No interior level ran and none was meant to.\n   \
             harvested: {} sub-proofs, {} published words\n   WHOLE RUN: host \
             peak {run_peak:.3} GiB at t={run_at:.1}, {:.1}s total",
            global_child.tables.len(),
            global_child.public_words.len(),
            t_all.elapsed().as_secs_f64(),
        );
        match &ceiling {
            Ok(c) => println!(
                "           = {:.1}% of the {c:.2} GiB cgroup ceiling",
                100.0 * run_peak / c
            ),
            Err(why) => println!("           ⚠ NO ceiling read, so NO percentage: {why}"),
        }
        return;
    }

    // ---- levels 1..=hi.
    let mut report: Vec<(usize, usize, u64, usize, f64, f64, f64)> = Vec::new();
    let mut penultimate: Option<TreeLevel> = None;
    for (li, level) in shape.iter().enumerate().take(hi) {
        let level_no = li + 1;
        let t_level = Instant::now();
        let (mut next, mut next_layouts, mut next_labels) = (
            Vec::with_capacity(level.arities.len()),
            Vec::with_capacity(level.arities.len()),
            Vec::with_capacity(level.arities.len()),
        );
        let groups = level_groups(&level.arities);
        for (j, g) in groups.iter().enumerate() {
            let arity = &g.len();
            let kids = &children[g.clone()];
            let kid_layouts = &layouts[g.clone()];
            let kid_labels = &labels[g.clone()];
            let label = format!("L{level_no}N{j} (arity {arity})");

            let label_refs: Vec<&[u64]> = kid_labels.iter().map(|l| &l[..]).collect();
            let range = (
                kid_labels[0][0],
                *kid_labels[arity - 1].last().expect("a label run"),
            );
            let out_halves = kid_layouts[arity - 1].out_halves;

            let t_emit = Instant::now();
            let program = node_program(
                kids,
                kid_layouts,
                &label_refs,
                range,
                super::per_table_aggregator::NodePublishSet::Aggregation,
            );
            println!(
                "   {label}: emitted in {:.1}s",
                t_emit.elapsed().as_secs_f64()
            );
            let (cells, instrs) = census_and_panel(&program, &label, fan_in);

            let sampler = HostSampler::start();
            let t_node = Instant::now();
            // ⛔ The program the census was taken on, NOT a second emission of
            // the same thing. See `prove_node_program_as_child`.
            let (child, layout) = prove_node_program_as_child(
                &label,
                &program,
                kids,
                out_halves,
                &wrap_opts,
                stage_mode(level_no),
                stage_path(cache_dir.as_deref(), &format!("node-{level_no}-{j}")),
            );
            let wall = t_node.elapsed().as_secs_f64();
            let (peak, at) = sampler.stop();
            println!(
                "   {label}: host peak {peak:.3} GiB at t={at:.1}{}, wall {wall:.1}s",
                match &ceiling {
                    Ok(g) => format!(" ({:.1}% of {g:.2})", 100.0 * peak / g),
                    Err(_) => String::new(),
                },
            );
            report.push((level_no, *arity, cells, instrs, peak, at, wall));
            next.push(child);
            next_layouts.push(layout);
            next_labels.push(vec![range.0, range.1]);
        }
        assert_eq!(
            groups.last().map(|g| g.end).unwrap_or(0),
            children.len(),
            "every child must be consumed"
        );
        // ★ The level below goes, and the trough is the point: a tree-builder at
        // level k holds level k-1 and nothing under it. Only a live sample can
        // show a release; a high-water cannot.
        // ★ Keep the level BELOW the top: those are root option A's interior
        // children (a root REPLACING the top level), while `children` after the
        // loop holds option B's single child (a root sitting ABOVE it).
        if size_root && level_no + 1 == top {
            penultimate = Some((
                std::mem::take(&mut children),
                std::mem::take(&mut layouts),
                std::mem::take(&mut labels),
            ));
        }
        children = next;
        layouts = next_layouts;
        labels = next_labels;
        mark(&format!("AFTER level {level_no}, its children released"));
        println!(
            "   level {level_no}: {} nodes in {:.1}s",
            children.len(),
            t_level.elapsed().as_secs_f64()
        );
    }

    // ---- SIZING: emit both root options, prove neither.
    if size_root {
        let g_layout = super::block_root::GlobalLayout {
            num_epochs: g.num_l2g,
            lanes_per_root: super::proof_arena::lanes_per_root(),
        };
        println!(
            "\n★★★ SIZING THE ROOT — emitted, never proved. Global child: {} \
             published words, {} sub-proofs ({} L2G).",
            global_child.public_words.len(),
            global_child.tables.len(),
            g.num_l2g,
        );
        let (pen, pen_layouts, pen_labels) = penultimate
            .take()
            .expect("size mode captures the level below the top");
        let block_range = (
            crate::tables::local_to_global::epoch_label(0),
            crate::tables::local_to_global::epoch_label(bundle.num_epochs() as u64 - 1),
        );
        for (name, replaces_top, kids, kid_layouts, kid_labels) in [
            (
                "A: root REPLACES the top level",
                true,
                &pen,
                &pen_layouts,
                &pen_labels,
            ),
            (
                "B: root sits ABOVE it (the level-top scaffold is kept)",
                false,
                &children,
                &layouts,
                &labels,
            ),
        ] {
            let refs: Vec<&[u64]> = kid_labels.iter().map(|l| &l[..]).collect();
            let shape =
                super::block_root::FoldShape::for_root(bundle.num_epochs(), fan_in, replaces_top);
            let t = Instant::now();
            let program = root_program(
                kids,
                kid_layouts,
                &refs,
                block_range,
                &global_child,
                &g_layout,
                &shape,
                super::block_root::RootPublishSet::AssertOnly,
            );
            let sub_proofs: usize =
                kids.iter().map(|c| c.tables.len()).sum::<usize>() + global_child.tables.len();
            println!(
                "\n── {name}: {} interior children + the global wrap = {} sub-proofs \
                 (emitted in {:.1}s)",
                kids.len(),
                sub_proofs,
                t.elapsed().as_secs_f64(),
            );
            census_and_panel(&program, name, fan_in);
        }
        println!(
            "\n⇒ READ `LFM_HASH`'s COMMITTED HEIGHT IN EACH PANEL. That single step \
             is 170,393,600 cells — 32% of a level-1 node — and it is what decides \
             whether the root resembles a proven-to-fit node or the fan-in-3 node \
             that aborted at 97.4% of the card."
        );
    }

    println!(
        "\n★★★ TREE COMPOSED — {} proof(s) at level {hi}",
        children.len()
    );
    if hi == top {
        assert_eq!(
            children.len(),
            1,
            "the interior must close to exactly one proof"
        );
    }
    println!("\nlevel arity        cells  instructions  host GiB   argmax t    wall s");
    for (l, a, cells, instrs, peak, at, wall) in &report {
        println!("{l:>5} {a:>5} {cells:>12} {instrs:>13} {peak:>9.3} {at:>10.1} {wall:>9.1}");
    }

    // ---- ★★★ THE BLOCK-ARTIFACT ROOT.
    //
    // The one program whose output IS the block artifact: it verifies the top
    // interior children and the global child, binds them into one execution,
    // performs the L2G compare a split tree defers to their common ancestor, and
    // publishes the block-level claim and nothing else.
    if let Some(option) = root_option {
        let g_layout = super::block_root::GlobalLayout {
            num_epochs: g.num_l2g,
            lanes_per_root: super::proof_arena::lanes_per_root(),
        };
        // ⛔ ONE OPTION DECIDES BOTH the children this run harvested and the fold
        // shape the compare refolds with. They must agree: `emit_l2g_compare`
        // refolds the global child's FLAT root list TREE-SHAPED, grouping exactly
        // as the interior did, and a shape built for the other option compares a
        // fold of the wrong depth. That failure is a `DivByZero` from the guest —
        // an `assert_eq` failing, `addr` being the diff cell, NOT a divisor — and
        // it lands on COMPLETENESS: honest prover, correct code, wrong answer.
        let shape = option.fold_shape(bundle.num_epochs(), fan_in);
        let block_range = (
            crate::tables::local_to_global::epoch_label(0),
            crate::tables::local_to_global::epoch_label(bundle.num_epochs() as u64 - 1),
        );
        let refs: Vec<&[u64]> = labels.iter().map(|l| &l[..]).collect();
        // ⛔ `AssertOnly`, NAMED HERE AND HANDED TO BOTH the emitter and the width
        // assert. The interior's L2G digest is tree-shaped, so its VALUE depends
        // on fan-in and depth; publishing it would satisfy the letter of *the
        // schema may not depend on the proving strategy* while breaking its
        // purpose — two honest provers at different postures would emit DIFFERENT
        // ARTIFACT BYTES for the same block. Nothing external could consume it
        // either: the global roots live inside this same proof, so the compare
        // binds in-machine and a published digest would have no reader.
        let publishes = super::block_root::RootPublishSet::AssertOnly;
        let t_emit = Instant::now();
        let program = root_program(
            &children,
            &layouts,
            &refs,
            block_range,
            &global_child,
            &g_layout,
            &shape,
            publishes,
        );
        let sub_proofs: usize =
            children.iter().map(|c| c.tables.len()).sum::<usize>() + global_child.tables.len();
        println!(
            "\n★★★ THE BLOCK-ARTIFACT ROOT — option {}\n   {} interior \
             children + the global child = {sub_proofs} sub-proofs ({} from the \
             global child alone) · emitted in {:.1}s",
            option.describe(),
            children.len(),
            global_child.tables.len(),
            t_emit.elapsed().as_secs_f64(),
        );
        // ★ THE PANEL BEFORE THE PROVE, as every other stage takes it. The
        // pre-registration hangs on `LFM_HASH`'s COMMITTED HEIGHT, which is read
        // off this panel and not inferred from a ratio.
        census_and_panel(&program, "the BLOCK-ARTIFACT ROOT", fan_in);
        // ⚠ DECLARATION ORDER IS ABSORB ORDER, and the global child goes LAST —
        // `emit_block_root` declares every interior child's arenas before the
        // global child's, so the arenas are a plain concatenation in that order.
        let arenas: Vec<Vec<LfmWord>> = children
            .iter()
            .chain(std::iter::once(&global_child))
            .flat_map(child_arena_words)
            .collect();
        let artifacts =
            build_artifacts_with_hasher(&program, &wrap_opts, crate::hash_pin::BLOCK_HASHER);
        #[cfg(feature = "cuda")]
        stark::gpu_lde::reset_all_gpu_call_counters();
        let sampler = HostSampler::start();
        let t_stage = Instant::now();
        let proved = cached_stage(
            root_mode,
            stage_path(cache_dir.as_deref(), "block-root"),
            "the BLOCK-ARTIFACT ROOT",
            || {
                lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
                    .unwrap_or_else(|e| panic!("★ THE BLOCK-ARTIFACT ROOT MUST PROVE: {e:?}"))
            },
        );
        let stage_secs = t_stage.elapsed().as_secs_f64();
        let (peak, at) = sampler.stop();

        // ⛔ THE ARTIFACT'S WIDTH, AND NOTHING WIDER. `root_schema_words` takes
        // no epoch count and no arity, so a mismatch means the artifact acquired
        // a dependence on HOW WE PROVED IT. ⛔ Do not widen this assert: an
        // artifact whose shape moves with the tree is not the thing this campaign
        // set out to produce, and a tolerance here would hide exactly that.
        let out_halves = layouts.last().expect("nonempty").out_halves;
        let num_reg = layouts[0].num_reg;
        let want = super::block_root::root_schema_words(num_reg, out_halves, publishes);
        assert_eq!(
            proved.public_words.len(),
            want,
            "★ THE ARTIFACT IS THE WRONG WIDTH: the root published {} words and \
             root_schema_words({num_reg} registers, {out_halves} output halves, \
             AssertOnly) is {want}. That signature takes NO epoch count and NO \
             arity, so this is the artifact acquiring a dependence on the proving \
             strategy",
            proved.public_words.len(),
        );
        // ⛔ VERIFIED HERE, and by this stage rather than by a call it happens to
        // make. A root that PROVES and does not VERIFY is the failure that reads
        // as success, and under `LFM_TREE_ROOT_MODE=load` the proof came off the
        // disk through `rkyv` and has never been checked in this process at all.
        let t_verify = Instant::now();
        assert!(
            super::proof::verify_against_artifacts(
                &artifacts,
                &proved.proof,
                &proved.public_words,
                &wrap_opts
            ),
            "★ THE BLOCK-ARTIFACT ROOT DOES NOT VERIFY. Nothing may be reported \
             or claimed from a proof production would reject"
        );
        let verify_secs = t_verify.elapsed().as_secs_f64();
        // ⚠ ONLY WHERE THIS PROCESS ACTUALLY PROVED — under `load` no kernel runs
        // and zero is the correct observation.
        #[cfg(feature = "cuda")]
        {
            if root_mode != CacheMode::Load {
                let calls = stark::gpu_lde::gpu_lde_calls()
                    + stark::gpu_lde::gpu_merkle_tree_calls()
                    + stark::gpu_lde::gpu_fri_calls();
                assert!(
                    calls > 0,
                    "the BLOCK-ARTIFACT ROOT reached the device ZERO times — it \
                     proved on the HOST with cuda compiled in, so its peak is not \
                     a production figure and the run does not show the GPU \
                     accelerating anything"
                );
                println!("     GPU dispatches during the BLOCK-ARTIFACT ROOT: {calls}");
            }
        }
        println!(
            "\n★★★ THE BLOCK IS COMPRESSED — the block-artifact ROOT PROVED AND \
             VERIFIED\n   option {}\n   stage {stage_secs:.1}s · verify \
             {verify_secs:.1}s · {} published words (= root_schema_words({num_reg}, \
             {out_halves}, AssertOnly), and NOTHING L2G-shaped)\n   host peak \
             {peak:.3} GiB at t={at:.1}{}\n   cache entry block-root.rkyv\n   ⚠ the \
             DEVICE peak is the prover's own VRAM accounting above, not a harness \
             sample — this harness counts dispatches, it does not size the card",
            option.describe(),
            proved.public_words.len(),
            match &ceiling {
                Ok(c) => format!(" ({:.1}% of {c:.2})", 100.0 * peak / c),
                Err(_) => String::new(),
            },
        );
        // ⛔ THE STANDING CAVEAT, PRINTED WITH THE CLAIM AND NOT LEFT TO A DOC.
        println!(
            "   ⛔ CAVEAT, unchanged by this proof and by design: the attestation \
             is NOT self-enforcing. The guest uses supplied roots verbatim, and \
             the binding happens OUTSIDE — `recursion::check_attestation` \
             recomputes the id from an ELF the consumer trusts, host-side. \
             \"One proof for this block\" terminates there"
        );
        // ★ THE TWO-POSTURE BYTE-IDENTITY CHECK, REFUSED BY NAME.
        //
        // `root_schema_words`' signature pins the artifact's WIDTH against the
        // proving strategy. Nothing pins its VALUE except proving one block at
        // TWO postures and comparing bytes, and this run has exactly one. ⇒ The
        // refusal is the result. It is not a skip, and it is emphatically not a
        // pass: a one-posture "identical" is a check that cannot fail, which is
        // worse than no check because it produces evidence.
        let runs = vec![super::block_root::ArtifactUnderPosture {
            posture: format!(
                "{} epochs at 2^{}, fan-in {fan_in}, root option {}",
                bundle.num_epochs(),
                inputs.epoch_log2,
                if option.replaces_top() { "A" } else { "B" },
            ),
            words: proved.public_words.clone(),
        }];
        match super::block_root::why_posture_identity_cannot_run(&runs) {
            Some(why) => println!("\n   ⚠ {why}"),
            None => super::block_root::assert_artifact_is_posture_independent(&runs),
        }
    }

    let (run_peak, run_at) = whole_run.stop();
    println!(
        "\nWHOLE RUN: host peak {run_peak:.3} GiB at t={run_at:.1}, {:.1}s total",
        t_all.elapsed().as_secs_f64()
    );
    match &ceiling {
        Ok(g) => println!(
            "           = {:.1}% of the {g:.2} GiB cgroup ceiling",
            100.0 * run_peak / g
        ),
        Err(why) => println!("           ⚠ NO ceiling read, so NO percentage: {why}"),
    }
}
