//! Gates for the assembled epoch verify, and the production-shape census.
//!
//! The census is here rather than in a log because it is an EVALUATION of the
//! gated forms at the block's own shape, not an estimate: every term comes from
//! a form that has its own F1 against an emitted program, and the shape comes
//! from a box run whose file is named below. What it cannot evaluate is the
//! per-table half — that needs the real AIRs, which need the guest ELF — so
//! that one term is an input, quoted with its measurement's convention.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::transcript_hash::RpxTranscriptHash;
use multilinear::constraint_argument::FactorKind;
use stark::traits::AIR;

use crate::multilinear_continuation::epoch_groups;
use crate::multilinear_prove::{chain_config, stacks};
use crate::tables::types::{FE, FEE, GoldilocksExtension as FEE3, GoldilocksField};

use super::algebraic_commit::commitment_to_digest;
use super::builder::{Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_chain::{ChainShape, chain_shape_rows};
use super::whir_chain_tests::const_rows;
use super::whir_epoch::{
    closure_rows, emit_epoch_closure, emit_roots_block, epoch_group_costs, expected_rows,
    group_columns, preprocessed_targets, roots_block_constants, roots_block_cost,
};
use super::whir_transcript::{SpongeEntry, WhirTranscript};
use super::word::{LfmWord, ext_word, word_as_ext};

/// Epoch 0 of block 25368371 at `epoch_size_log2 = 21`, `(width, num_vars)` per
/// table in sub-proof order.
///
/// ⚠ A FIXTURE READ FROM A BOX RUN, not a laptop measurement: the file is
/// `thoughts/shared/whir-recursion/handoffs/sh1-epoch-shapes-2026-09-18.log`,
/// `whir_epoch_shapes` at a3b128db on FAST, guest ethrex
/// `8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a`
/// (3,948,504 B), input `ethrex_mainnet_25368371`. The widths are the AIRs' and
/// the heights are the proof's, which is the pair `verify_epoch_bookend`
/// (`multilinear_continuation.rs:1331`) builds its layouts from.
///
/// The three numbers the same log states about the SHAPE these produce — seven
/// polynomials in group 0, one in the bookend, 112 queries — are asserted below
/// before anything is counted, so a reconstruction that does not reproduce the
/// box's own layout fails loudly instead of censusing a different epoch.
fn epoch_zero_shapes() -> Vec<(usize, usize)> {
    vec![
        (21, 20),  // BITWISE
        (6, 20),   // DECODE
        (19, 2),   // COMMIT
        (511, 2),  // KECCAK
        (1480, 2), // KECCAK_RND
        (10, 5),   // KECCAK_RC
        (667, 2),  // ECSM
        (521, 2),  // ECDAS
        (41, 2),   // HINT
        (5, 7),    // REGISTER
        (38, 19),  // CPU[0]
        (38, 19),  // CPU[1]
        (38, 19),  // CPU[2]
        (38, 19),  // CPU[3]
        (17, 19),  // LT[0]
        (17, 19),  // LT[1]
        (29, 18),  // SHIFT[0]
        (49, 17),  // MEMW[0]
        (29, 19),  // MEMW_A[0]
        (29, 16),  // MEMW_A[1]
        (18, 19),  // LOAD[0]
        (26, 11),  // MUL[0]
        (34, 2),   // DVRM[0]
        (14, 17),  // BRANCH[0]
        (10, 20),  // MEMW_R[0]
        (10, 20),  // MEMW_R[1]
        (10, 20),  // MEMW_R[2]
        (10, 20),  // MEMW_R[3]
        (10, 18),  // MEMW_R[4]
        (12, 13),  // EQ[0]
        (26, 17),  // BYTEWISE[0]
        (16, 19),  // STORE[0]
        (38, 10),  // CPU32[0]
        (9, 21),   // the l2g bookend, committed alone in the last group
    ]
}

/// The per-table half, from `bs2-table-costs-2026-09-19.log`: 34 tables'
/// `table_verify_cost` with the sponge threaded, at the same shape and the same
/// guest.
///
/// ⚠ CONST-INCLUSIVE, and measured with EACH TABLE ITS OWN PROGRAM, so its
/// constants are 34 pools. In one epoch program there is one pool and they
/// collide; the census says so rather than adding the two conventions.
const BS2_PER_TABLE_ROWS: usize = 319_047;
const BS2_PER_TABLE_PERMS: usize = 19_394;

/// The DECODE group, whole, from V1f item 3's production cross-check: the
/// wrapper and its one chain at `n_stack` 23 under the posture
/// `ChainConfig::with_security(2, 4, 25, 128, uniform(20))`. CONST-FREE.
const DECODE_GROUP_ROWS: usize = 161_783;
const DECODE_GROUP_PERMS: usize = 19_205;

/// BITWISE's eleven preprocessed columns in closed form
/// (`preprocessed::bitwise_preprocessed_rows`), plus KECCAK_RC's nine at five
/// variables and REGISTER's three at seven
/// (`preprocessed::const_mle_rows`), both evaluated below rather than quoted.
fn preprocessed_rows() -> usize {
    let bitwise = super::preprocessed::bitwise_preprocessed_rows();
    let keccak_rc = super::preprocessed::const_mle_rows(
        &crate::tables::keccak_rc::preprocessed_columns()
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>(),
        5,
    );
    // ⚠ NOT an all-zero register file. `const_mle_rows` charges one row per
    // NONZERO entry, so zeroed INIT and FINI columns would cost ONE interned
    // constant each instead of a fold — and the census would report a number no
    // real epoch can reach. A real `R_i` has most of its 127 word-addresses
    // written, so this uses a full file and the term is an upper end of the
    // band rather than a floor. The BAND, not a point: an epoch whose register
    // file is half zero pays about half of those two columns.
    let init: Vec<u32> = (0..crate::tables::register::NUM_REGISTER_ADDRESSES)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761) | 1)
        .collect();
    let register_columns = crate::tables::register::preprocessed_columns_with_fini(&init, &init);
    let register = super::preprocessed::const_mle_rows(
        &register_columns
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>(),
        7,
    );
    bitwise + keccak_rc + register
}

/// ★ THE PRODUCTION-SHAPE RECOUNT: every gated form evaluated at epoch 0 of the
/// block, in BOTH conventions, against arm B.
///
/// Not an emitted count and it does not claim to be — the block's own prove is
/// the box's. What it IS: the same closed forms whose F1s are pinned against
/// emitted programs at gated shapes, evaluated at the shape a box run measured.
/// Arm B is the sizing note's 12,301,266 instructions for one epoch wrap.
#[test]
fn the_production_epoch_recount() {
    let shapes = epoch_zero_shapes();
    assert_eq!(shapes.len(), 34, "epoch 0 is 34 tables (sh1)");
    let sizes = epoch_groups(shapes.len());
    assert_eq!(sizes, vec![33, 1], "the bookend is committed alone");
    let config = chain_config(&shapes);
    let (layouts, _domains) = stacks(&shapes, &sizes, &config).expect("the epoch's stacks build");

    // ⚠ ASSERTED BEFORE ANYTHING IS COUNTED. These four are sh1's own printed
    // numbers for this epoch; a reconstruction that misses any of them is
    // censusing a different shape, and would do it silently.
    assert_eq!(layouts.len(), 2, "two commitment groups");
    assert_eq!(layouts[0].n_stack(), 25, "sh1: group0 n_stack 25");
    assert_eq!(layouts[0].num_polys(), 7, "sh1: group0 polys 7");
    assert_eq!(layouts[1].n_stack(), 25, "sh1: bookend n_stack 25");
    assert_eq!(layouts[1].num_polys(), 1, "sh1: bookend polys 1");
    assert_eq!(config.num_queries, 112, "sh1: Q 112");
    let rounds = ChainShape::new(&config, 25).rounds();
    let chains = layouts.iter().map(|l| l.num_polys()).sum::<usize>();
    assert_eq!(chains, 8, "sh1: chains 8");
    assert_eq!(chains * rounds, 56, "sh1: rounds 56");

    let costs = epoch_group_costs(&shapes, &sizes, &layouts, &config, SpongeEntry::fresh());
    let shape = ChainShape::new(&config, 25);
    let chain_shape = chain_shape_rows(&shape);
    let groups: usize = costs.iter().map(|c| c.operations()).sum();
    let group_perms: usize = costs.iter().map(|c| c.perms()).sum();
    let chain_share = chains * chain_shape;

    let published = 0usize; // epoch 0 of this block publishes nothing
    let closure = closure_rows(shapes.len(), published);
    let preprocessed = preprocessed_rows();

    let const_free = BS2_PER_TABLE_ROWS + groups + DECODE_GROUP_ROWS + preprocessed + closure;
    let perms = BS2_PER_TABLE_PERMS + group_perms + DECODE_GROUP_PERMS;

    let (columns, group_of) = group_columns(&shapes[..33]);
    // ★ How many (table, polynomial) pairs the layout actually makes, which is
    // what `weight_at` pays an `eq` for — NOT the table count. A table whose
    // columns straddle a polynomial boundary is claimed at its one point in
    // each of them and pays in each.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (column, placement) in layouts[0].placements().iter().enumerate() {
        let pair = (group_of[column], placement.poly);
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    }
    println!("== V1g epoch-0 recount, forms evaluated at sh1's shape ==");
    println!(
        "  group 0's layout makes {} (table, polynomial) pairs over 33 tables \
         and {} polynomials — {} tables straddle a boundary",
        pairs.len(),
        layouts[0].num_polys(),
        pairs.len() - 33
    );
    println!(
        "  group 0: C {} polys {} | operations {} perms {}",
        columns.len(),
        layouts[0].num_polys(),
        costs[0].operations(),
        costs[0].perms()
    );
    println!(
        "  bookend: C 9 polys {} | operations {} perms {}",
        layouts[1].num_polys(),
        costs[1].operations(),
        costs[1].perms()
    );
    println!(
        "  of the groups' {groups} operations, {chain_share} are the chains' \
         own shape rows ({chains} x {chain_shape}) and {} are the wrappers'",
        groups - chain_share
    );
    println!("  per-table half (bs2, CONST-INCLUSIVE):  {BS2_PER_TABLE_ROWS} rows");
    println!("  the two groups (CONST-FREE):            {groups} rows");
    println!("  the DECODE group (CONST-FREE, item 3):  {DECODE_GROUP_ROWS} rows");
    println!("  the preprocessed seam:                  {preprocessed} rows");
    println!("  the closure:                            {closure} rows");
    println!("  the epoch statement:                    0 operations");
    println!("  ---------------------------------------------------");
    println!("  PER EPOCH: {const_free} rows | {perms} permutations");
    println!(
        "  arm B 12,301,266 / {const_free} = {:.2}x",
        12_301_266.0 / const_free as f64
    );
    println!(
        "  ⚠ CONVENTIONS: the per-table term counts 34 separate pools and the \
         rest count none; the assembled program has ONE pool, so the const-free \
         total is below this and the const-inclusive total is that plus one pool."
    );

    // The judgement this census exists to settle, asserted rather than left in
    // the printout: arm B is an over-estimate, and by a factor no rounding of
    // the terms above can close.
    assert!(
        const_free * 3 < 12_301_266,
        "the recount must stay far below arm B; it is {const_free}"
    );
    // ★ The wrapper's own share is NOT the fixture-scale "few hundred a group":
    // `weight_at` charges one prefix indicator per column per stack position,
    // and this epoch is 3,837 columns most of which sit at two variables in a
    // 25-variable stack. Asserted so a change that makes it small again is
    // read as a change and not as agreement.
    assert!(
        groups - chain_share > 50_000,
        "the wrappers' own share at the production shape is {}",
        groups - chain_share
    );
    // ★ And the reason it exceeded a hand bound that assumed ONE eq group per
    // table: tables straddle polynomial boundaries. Asserted so the straddling
    // is a measured property of this layout rather than an explanation offered
    // after the fact.
    assert!(
        pairs.len() > 33,
        "group 0's columns were expected to straddle polynomials; the layout \
         makes {} pairs over 33 tables",
        pairs.len()
    );
}

/// The two conventions, side by side on one shape, so the difference between
/// them is a number rather than a caveat.
#[test]
fn the_closure_costs_what_publishing_adds() {
    let silent = closure_rows(34, 0);
    let publishing = closure_rows(34, 8);
    println!(
        "closure: {silent} rows when the epoch publishes nothing, {publishing} \
         when it publishes 8 bytes ({} a byte)",
        (publishing - silent) as f64 / 8.0
    );
    assert_eq!(
        publishing - silent,
        2 + 8 * 4 - 1,
        "a published byte costs two MulAdds, an inverse and an Add, with the \
         alpha square and `z - bus_id` once each and the first sum needing no Add"
    );
    // ⛔ The half instance 51 is about: an epoch that publishes nothing
    // discharges the commit bus WITHOUT reading z or alpha, so a verifier bug
    // in that term is invisible on every silent epoch.
    assert_eq!(
        silent,
        34 + 33 + 2,
        "a silent epoch's closure is the divisions, the sum and the assert — \
         the expected value is the literal zero and reads no challenge"
    );
}

// =============================================================================
// The roots block
// =============================================================================

type HostTranscript = DefaultTranscript<FEE3, RpxTranscriptHash>;

/// A root, as the verifier absorbs it and as the machine holds it.
fn a_root(seed: u8) -> ([u8; 32], super::word::LfmWord) {
    let mut bytes = [0u8; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = seed.wrapping_mul(31).wrapping_add(i as u8).wrapping_mul(7);
    }
    // ⚠ The top limb is masked so every eight-byte group is a canonical field
    // element: an absorbed root that is not is a different question, and it is
    // the transcript's, not this leg's.
    bytes[7] = 0;
    bytes[15] = 0;
    bytes[23] = 0;
    bytes[31] = 0;
    let word = commitment_to_digest(&bytes);
    (bytes, word)
}

/// The host's three challenges after the roots block, through its own function.
fn host_roots_block(carried: &[[u8; 32]], derived: &[[u8; 32]]) -> (FEE, FEE, FEE) {
    let mut transcript = HostTranscript::new(&[]);
    stark::multilinear_table::absorb_roots_and_challenge::<FEE3, _>(
        &mut transcript,
        carried,
        derived,
    )
}

/// The machine's three, and what the block cost while drawing them.
fn machine_roots_block(
    carried: &[super::word::LfmWord],
    derived: &[super::word::LfmWord],
    leg: bool,
) -> (Vec<FEE>, usize, usize) {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(carried.len() as u32);
    let cells: Vec<_> = (0..carried.len())
        .map(|i| b.hint_word(arena, i as u32))
        .collect();
    if !leg {
        // The hints cancel out of the subtraction; three publishes stand in for
        // the three challenges so the publish count cancels too.
        for i in 0..3 {
            b.public(cells[i % cells.len()]);
        }
        let program = compile(b.finish());
        return (Vec::new(), program.instrs.len(), const_rows(&program));
    }
    let mut transcript = WhirTranscript::new();
    let (z, alpha, beta) = emit_roots_block(&mut b, &mut transcript, &cells, derived);
    for wire in [z, alpha, beta] {
        b.public(wire.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("the roots block must be admissible");
    let rows = program.instrs.len();
    let consts = const_rows(&program);
    let arena_words: Vec<_> = carried.to_vec();
    let exec = execute(&program, &[arena_words], &crate::hash_pin::BLOCK_HASHER)
        .expect("the roots block executes");
    let drawn: Vec<FEE> = exec
        .public_words
        .iter()
        .map(|(_, word)| word_as_ext(word).expect("a published challenge"))
        .collect();
    (drawn, rows, consts)
}

/// The `LFM_CONST` words the roots block interns, by value — the ones a program
/// without the leg does not already have.
fn machine_roots_block_constants(
    carried: &[super::word::LfmWord],
    derived: &[super::word::LfmWord],
) -> Vec<super::word::LfmWord> {
    let words_of = |leg: bool| -> Vec<super::word::LfmWord> {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let arena = b.declare_arena(carried.len() as u32);
        let cells: Vec<_> = (0..carried.len())
            .map(|i| b.hint_word(arena, i as u32))
            .collect();
        if leg {
            let mut transcript = WhirTranscript::new();
            let (z, _, _) = emit_roots_block(&mut b, &mut transcript, &cells, derived);
            b.public(z.as_cell());
        } else {
            b.public(cells[0]);
        }
        compile(b.finish())
            .instrs
            .iter()
            .filter_map(|instr| match instr {
                super::instr::Instr::Const { value, .. } => Some(*value),
                _ => None,
            })
            .collect()
    };
    let without = words_of(false);
    let mut out: Vec<super::word::LfmWord> = Vec::new();
    for word in words_of(true) {
        if !without.contains(&word) && !out.contains(&word) {
            out.push(word);
        }
    }
    out
}

/// The shapes: epoch 0's eight carried roots and one derived, the two other
/// polynomial counts the block's epochs reach, and a block with none derived.
fn roots_shapes() -> Vec<(usize, usize)> {
    vec![(8, 1), (9, 1), (12, 1), (8, 0), (1, 1)]
}

/// ★ THE GATE: the machine draws the three challenges the HOST draws, from the
/// host's own `absorb_roots_and_challenge`.
///
/// A transcript's state is not observable and its next draw is, so the three
/// challenges ARE the comparison — and all three are compared, because the
/// third is the one a reader drops.
#[test]
fn the_roots_block_draws_what_the_host_draws() {
    for (carried, derived) in roots_shapes() {
        let carried_pairs: Vec<_> = (0..carried).map(|i| a_root(i as u8)).collect();
        let derived_pairs: Vec<_> = (0..derived).map(|i| a_root(200 + i as u8)).collect();
        let carried_bytes: Vec<[u8; 32]> = carried_pairs.iter().map(|(b, _)| *b).collect();
        let derived_bytes: Vec<[u8; 32]> = derived_pairs.iter().map(|(b, _)| *b).collect();
        let carried_words: Vec<_> = carried_pairs.iter().map(|(_, w)| *w).collect();
        let derived_words: Vec<_> = derived_pairs.iter().map(|(_, w)| *w).collect();

        let (z, alpha, beta) = host_roots_block(&carried_bytes, &derived_bytes);
        let (drawn, _, _) = machine_roots_block(&carried_words, &derived_words, true);
        assert_eq!(drawn.len(), 3, "the block draws three challenges");
        assert_eq!(
            drawn[0], z,
            "{carried} carried + {derived} derived: z must be the host's"
        );
        assert_eq!(
            drawn[1], alpha,
            "{carried} carried + {derived} derived: alpha must be the host's"
        );
        assert_eq!(
            drawn[2], beta,
            "{carried} carried + {derived} derived: beta must be the host's"
        );
    }
}

/// ★ F1. The emitted count equals the closed form, and the pool is NAMED.
#[test]
fn the_roots_block_emits_its_closed_form() {
    for (carried, derived) in roots_shapes() {
        let carried_words: Vec<_> = (0..carried).map(|i| a_root(i as u8).1).collect();
        let derived_words: Vec<_> = (0..derived).map(|i| a_root(200 + i as u8).1).collect();
        let (_, with_rows, with_consts) = machine_roots_block(&carried_words, &derived_words, true);
        let (_, without_rows, without_consts) =
            machine_roots_block(&carried_words, &derived_words, false);
        let measured = (with_rows - without_rows) - (with_consts - without_consts);
        let (predicted, schedule) = roots_block_cost(carried, derived, SpongeEntry::fresh());
        let predicted = predicted + schedule.rows();
        println!(
            "roots block {carried} carried + {derived} derived: {measured} rows emitted, \
             {predicted} predicted ({} leg + {} sponge, {} permutations)",
            predicted - schedule.rows(),
            schedule.rows(),
            schedule.perms(),
        );
        assert_eq!(
            measured, predicted,
            "{carried} carried + {derived} derived: the emitted operation count \
             must equal the closed form"
        );
        let named = roots_block_constants(&derived_words, &schedule);
        // ★ instance 63: the test ASKS which constants the program interns that
        // the form does not name, and prints them, so an unnamed one is a term
        // to identify rather than a number to add.
        let interned = machine_roots_block_constants(&carried_words, &derived_words);
        let unnamed: Vec<_> = interned
            .iter()
            .filter(|word| !named.contains(word))
            .collect();
        assert!(
            unnamed.is_empty(),
            "{carried} carried + {derived} derived: the block interns \
             {unnamed:?} which the form does not name"
        );
        assert_eq!(
            interned.len(),
            named.len(),
            "{carried} carried + {derived} derived: the interned pool and the \
             named pool must be the same set"
        );
    }
}

/// ⛔ The derived root is PROGRAM TEXT, and this is what says so.
///
/// A hinted root is a value the prover chooses; an interned one is not. The
/// distinction is invisible in the challenge (an honest prover hints the same
/// bytes) and visible in two places at once: the block interns one constant per
/// derived root, and its arena is the CARRIED roots alone. Both are asserted,
/// because either on its own is satisfied by a program that hints the root and
/// happens to intern something else.
#[test]
fn the_derived_root_is_program_text_and_not_an_arena_word() {
    let carried_words: Vec<_> = (0..8).map(|i| a_root(i as u8).1).collect();
    let derived_words: Vec<_> = (0..1).map(|i| a_root(200 + i as u8).1).collect();
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(carried_words.len() as u32);
    let cells: Vec<_> = (0..carried_words.len())
        .map(|i| b.hint_word(arena, i as u32))
        .collect();
    let mut transcript = WhirTranscript::new();
    let (z, _, _) = emit_roots_block(&mut b, &mut transcript, &cells, &derived_words);
    b.public(z.as_cell());
    let program = compile(b.finish());

    // The arena holds the carried roots and nothing else — a program whose
    // arena had nine words for eight carried roots would be taking the derived
    // one from the prover.
    assert_eq!(
        program.arena_schema.lens,
        vec![carried_words.len() as u32],
        "the block's arena is the carried roots alone"
    );
    // And the derived root appears in the program TEXT.
    let consts: Vec<_> = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            super::instr::Instr::Const { value, .. } => Some(*value),
            _ => None,
        })
        .collect();
    assert!(
        consts.contains(&derived_words[0]),
        "the derived DECODE root must be interned as a program constant"
    );
}

// =============================================================================
// The closure
// =============================================================================

/// Bus outputs that BALANCE to a given total: `n - 1` arbitrary shares and a
/// last one that closes the sum, all as `(p, q)` with `q` non-zero.
fn balancing_outputs(n: usize, total: FEE, seed: u64) -> Vec<(FEE, FEE)> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        FEE::from(state >> 3)
    };
    let mut out: Vec<(FEE, FEE)> = Vec::with_capacity(n);
    let mut running = FEE::zero();
    for _ in 0..n - 1 {
        let q = next() + FEE::one();
        let p = next();
        running += (&p / &q).expect("a non-zero denominator");
        out.push((p, q));
    }
    // The last share closes the sum exactly, which is what makes the honest arm
    // honest rather than approximately so.
    let q = FEE::from(7u64);
    let p = (total - running) * &q;
    out.push((p, q));
    out
}

/// The machine's closure over one epoch's outputs: the two values it published
/// and what the program cost.
fn machine_closure(
    outputs: &[(FEE, FEE)],
    published: &[u8],
    start_index: u64,
    z: FEE,
    alpha: FEE,
    leg: bool,
) -> (Option<Vec<FEE>>, usize, usize) {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let words = 2 * outputs.len() + 2;
    let arena = b.declare_arena(words as u32);
    let wires: Vec<_> = (0..words)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let pairs: Vec<(_, _)> = (0..outputs.len())
        .map(|i| (wires[2 * i], wires[2 * i + 1]))
        .collect();
    let z_wire = wires[words - 2];
    let alpha_wire = wires[words - 1];
    if !leg {
        for wire in wires.iter().take(2) {
            b.public(wire.as_cell());
        }
        let program = compile(b.finish());
        return (None, program.instrs.len(), const_rows(&program));
    }
    let verdict = emit_epoch_closure(&mut b, &pairs, published, start_index, z_wire, alpha_wire);
    b.public(verdict.balance.as_cell());
    b.public(verdict.expected.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the closure must be admissible");
    let rows = program.instrs.len();
    let consts = const_rows(&program);
    let mut arena_words: Vec<_> = Vec::with_capacity(words);
    for (p, q) in outputs {
        arena_words.push(ext_word(p));
        arena_words.push(ext_word(q));
    }
    arena_words.push(ext_word(&z));
    arena_words.push(ext_word(&alpha));
    let drawn = execute(&program, &[arena_words], &crate::hash_pin::BLOCK_HASHER)
        .ok()
        .map(|exec| {
            exec.public_words
                .iter()
                .map(|(_, word)| word_as_ext(word).expect("a published value"))
                .collect()
        });
    (drawn, rows, consts)
}

/// A silent epoch and a publishing one, with the carried commit index the
/// publishing one continues from.
fn closure_fixtures() -> Vec<(&'static str, Vec<u8>, u64)> {
    vec![
        ("silent", Vec::new(), 0),
        (
            "publishing 8 bytes at index 0",
            vec![1, 2, 3, 250, 0, 9, 9, 7],
            0,
        ),
        (
            "publishing 8 bytes continuing a prior epoch",
            vec![1, 2, 3, 250, 0, 9, 9, 7],
            4_096,
        ),
    ]
}

/// ★ THE GATE: the machine's expected value is the host's own
/// `compute_commit_bus_offset`, and the honest balance executes against it.
#[test]
fn the_closure_is_the_hosts_commit_bus_offset() {
    let z = FEE::from(0x5eed_1234u64);
    let alpha = FEE::from(0xbeef_5678u64);
    for (name, published, start_index) in closure_fixtures() {
        let host = crate::compute_commit_bus_offset(&published, start_index, &z, &alpha)
            .expect("the host's offset exists at these challenges");
        let outputs = balancing_outputs(34, host, 0xC105_u64);
        let (drawn, _, _) = machine_closure(&outputs, &published, start_index, z, alpha, true);
        let drawn = drawn
            .unwrap_or_else(|| panic!("{name}: the machine must execute a balance that closes"));
        assert_eq!(
            drawn[1], host,
            "{name}: the expected value must be the host's commit bus offset"
        );
        assert_eq!(
            drawn[0], drawn[1],
            "{name}: the honest balance equals the expected value"
        );
    }
}

/// ★ THE REFUSAL: a balance that does not close has no satisfying assignment.
///
/// The host answers `BusImbalance`; the machine's `assert_eq_ext` is a
/// difference and a division by zero, so the two are the same event rather than
/// two behaviours that have to agree.
#[test]
fn a_balance_that_does_not_close_is_refused() {
    let z = FEE::from(0x5eed_1234u64);
    let alpha = FEE::from(0xbeef_5678u64);
    let published = vec![1u8, 2, 3, 250, 0, 9, 9, 7];
    let host = crate::compute_commit_bus_offset(&published, 0, &z, &alpha).expect("an offset");
    let mut outputs = balancing_outputs(34, host, 0xC105_u64);
    // The control first: the honest arm executes, so a refusal below is the
    // tamper and not a leg that refuses everything.
    assert!(
        machine_closure(&outputs, &published, 0, z, alpha, true)
            .0
            .is_some(),
        "the honest balance must execute"
    );
    outputs[7].0 += FEE::one();
    assert!(
        machine_closure(&outputs, &published, 0, z, alpha, true)
            .0
            .is_none(),
        "a tampered bus output must leave the machine with no satisfying assignment"
    );
}

/// ★ F1 for the closure, at both a silent and a publishing epoch.
#[test]
fn the_closure_emits_its_closed_form() {
    let z = FEE::from(0x5eed_1234u64);
    let alpha = FEE::from(0xbeef_5678u64);
    for (name, published, start_index) in closure_fixtures() {
        let host = crate::compute_commit_bus_offset(&published, start_index, &z, &alpha)
            .expect("an offset");
        let outputs = balancing_outputs(34, host, 0xC105_u64);
        let (_, with_rows, with_consts) =
            machine_closure(&outputs, &published, start_index, z, alpha, true);
        let (_, without_rows, without_consts) =
            machine_closure(&outputs, &published, start_index, z, alpha, false);
        let measured = (with_rows - without_rows) - (with_consts - without_consts);
        let predicted = closure_rows(outputs.len(), published.len());
        println!(
            "closure, {name}: {measured} rows emitted, {predicted} predicted \
             ({} tables, {} published; expected half {})",
            outputs.len(),
            published.len(),
            expected_rows(published.len()),
        );
        assert_eq!(
            measured, predicted,
            "{name}: the emitted operation count must equal the closed form"
        );
    }
}

// =============================================================================
// The per-table walk: the preprocessed claim's address
// =============================================================================

/// ★ THE GATE ON THE INDIRECTION: a preprocessed column's claimed value is
/// addressed through the LAYOUT, never by the column's own index.
///
/// The host reads `slot(slot_of, column)` to a factor, takes that factor's
/// unshifted source, and indexes `column_values[source.column]`
/// (`multilinear_table.rs:1060-1080`). The two agree whenever the layout happens
/// to be the identity, which is exactly why this fixture is built NOT to be: its
/// slot map and its factor list both permute, so a leg that indexed by the
/// column would return `[0, 1, 2]` where the host returns `[2, 0, 1]`.
///
/// ⚠ THE FIXTURE IS THE TEST. A permutation drawn from a real table would make
/// this pass or fail for reasons outside the function; built here, the expected
/// answer is a property of the two slices and nothing else.
#[test]
fn the_preprocessed_targets_follow_the_layout_not_the_column_index() {
    // Column 0 rides factor 2, column 1 factor 0, column 2 factor 1; and the
    // factors read columns 2, 0, 1 in turn. Composing the two is the answer.
    let slot_of = [2usize, 0, 1];
    let kinds = [
        FactorKind::direct(2),
        FactorKind::direct(0),
        FactorKind::direct(1),
    ];
    let identity: Vec<usize> = (0..3).collect();
    let targets = preprocessed_targets(&slot_of, &kinds, 3);
    assert_eq!(
        targets,
        vec![1usize, 2, 0],
        "column c rides slot_of[c], whose factor reads kinds[slot].source().column"
    );
    // ⛔ ANTI-VACUITY, AND IT IS THE COMPOSITION THAT HAS TO MOVE. The first
    // version of this fixture permuted both slices and they composed back to
    // the identity, so `preprocessed_targets` returning the column index passed
    // it — a check that could not fail, caught by running the mutation that
    // drops the indirection and watching this test stay green. Asserting the
    // two inputs are not the identity is not enough; the ANSWER must differ
    // from the column index.
    assert_ne!(
        targets, identity,
        "a fixture whose slot map and factor list compose to the identity gates nothing: a leg \
         that ignored both slices would return exactly this"
    );
    assert_ne!(
        slot_of.to_vec(),
        identity,
        "a fixture whose slot map is the identity gates nothing"
    );
    let direct: Vec<usize> = kinds
        .iter()
        .filter_map(FactorKind::source)
        .map(|s| s.column)
        .collect();
    assert_ne!(
        direct, identity,
        "a fixture whose factor list is the identity gates nothing"
    );
}

/// ★ THE TWO REFUSALS the host makes, at emit time — each with the control that
/// says the fixture is otherwise good.
#[test]
fn a_preprocessed_column_with_no_slot_is_refused() {
    let kinds = [FactorKind::direct(0)];
    // The control: one column, one slot, and it resolves.
    assert_eq!(preprocessed_targets(&[0usize], &kinds, 1), vec![0usize]);
    // Two columns over a one-entry slot map: the second has no factor at all.
    let panicked = std::panic::catch_unwind(|| preprocessed_targets(&[0usize], &kinds, 2));
    assert!(
        panicked.is_err(),
        "a preprocessed column with no factor slot must be refused at emit time"
    );
}

#[test]
fn a_preprocessed_column_read_at_an_offset_is_refused() {
    // The control: the same layout with the factor unshifted resolves.
    assert_eq!(
        preprocessed_targets(&[0usize], &[FactorKind::direct(3)], 1),
        vec![3usize]
    );
    let shifted = [FactorKind::shifted(3, 1)];
    let panicked = std::panic::catch_unwind(|| preprocessed_targets(&[0usize], &shifted, 1));
    assert!(
        panicked.is_err(),
        "a preprocessed column whose factor is read at an offset must be refused"
    );
    // And a PUBLIC factor is refused for the same reason: it has no column.
    let public = [FactorKind::Public];
    let panicked = std::panic::catch_unwind(|| preprocessed_targets(&[0usize], &public, 1));
    assert!(
        panicked.is_err(),
        "a preprocessed column riding a public factor has no claimed value"
    );
}

// =============================================================================
// The two walks, on a real multi-table proof
// =============================================================================

/// The AIR type the three example tables share.
type WalkAir = stark::lookup::AirWithBuses<
    GoldilocksField,
    FEE3,
    stark::lookup::NullBoundaryConstraintBuilder,
    (),
    stark::constraints::builder::EmptyConstraints,
>;

/// The repo's own three-table example — CPU sends, ADD and MUL receive — proved
/// through `multi_prove` and committed the way an EPOCH commits: everything
/// together, the last table alone, which is `epoch_groups`' `[n − 1, 1]`.
///
/// ⚠ WHAT THIS FIXTURE IS AND IS NOT, so nobody later takes it for more. It is a
/// real two-group, three-table argument over a bus that BALANCES ACROSS THE
/// TABLES, which is exactly what the walks are about: the order, the sponge
/// threaded table to table, and the group walk's three counters. It carries NO
/// preprocessed columns and no prepared opening — the three preprocessed routes
/// are gated in `preprocessed_tests`, the claim's address is gated above on the
/// slices alone, and the real AIRs meet the walks on the driver's fixture in
/// item 5c.
fn walk_airs(options: &stark::proof::options::ProofOptions) -> (WalkAir, WalkAir, WalkAir) {
    (
        stark::examples::multi_table_lookup::new_cpu_air_with_lookup(options),
        stark::examples::multi_table_lookup::new_add_air_with_lookup(options),
        stark::examples::multi_table_lookup::new_mul_air_with_lookup(options),
    )
}

fn base_column(values: &[u64]) -> Vec<FE> {
    values.iter().map(|v| FE::from(*v)).collect()
}

/// The example's own balancing trace: the CPU table sends each row to ADD or
/// MUL, and the two receive exactly what was sent.
fn walk_columns() -> [Vec<Vec<FE>>; 3] {
    [
        vec![
            base_column(&[1, 0, 1, 0, 1, 1, 0, 0]),
            base_column(&[0, 1, 0, 1, 0, 0, 1, 1]),
            base_column(&[1, 2, 3, 4, 5, 6, 7, 8]),
            base_column(&[10, 20, 30, 40, 50, 60, 70, 80]),
            base_column(&[11, 40, 33, 160, 55, 66, 490, 640]),
        ],
        vec![
            base_column(&[1, 3, 5, 6]),
            base_column(&[10, 30, 50, 60]),
            base_column(&[11, 33, 55, 66]),
            base_column(&[1, 1, 1, 1]),
        ],
        vec![
            base_column(&[2, 4, 7, 8]),
            base_column(&[20, 40, 70, 80]),
            base_column(&[40, 160, 490, 640]),
            base_column(&[1, 1, 1, 1]),
        ],
    ]
}

/// The chain parameters the fixture argues at — small, because the walks'
/// claims are about order and slicing and not about the chains.
fn walk_config() -> multilinear::whir_chain::ChainConfig {
    multilinear::whir_chain::ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: multilinear::whir_chain::GrindBits::default(),
    }
}

/// How the fixture's tables are grouped: the epoch's own split.
const WALK_SIZES: [usize; 2] = [2, 1];

/// One of the fixture's tables, committed at the height its trace implies.
fn walk_table<'a>(
    air: &'a WalkAir,
    columns: &[Vec<FE>],
) -> stark::multilinear_table::CommittedTable<'a, GoldilocksField, FEE3> {
    let num_vars = columns[0].len().trailing_zeros() as usize;
    let owned = columns.to_vec();
    stark::multilinear_table::CommittedTable::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        columns.len(),
        num_vars,
        stark::multilinear_air::Uniforms::default(),
        |col| owned[col as usize].clone(),
    )
    .expect("the example table commits")
}

/// The host's own per-table pieces of `multi_verify`, taken by calling the host's
/// own functions in the host's own order.
struct HostWalk {
    z: FEE,
    alpha: FEE,
    beta: FEE,
    outputs: Vec<(FEE, FEE)>,
    points: Vec<Vec<FEE>>,
    values: Vec<FEE>,
}

/// ★ The reference the walk is gated against, and it is the HOST's.
///
/// This is not a second implementation of `multi_verify`: it CALLS
/// `absorb_roots_and_challenge` and `multilinear_table::verify`, in the order
/// and on the transcript `multi_verify` uses, and keeps what they return.
/// Comparing the machine against a copy of the machine would be no evidence
/// about either (instance 68), and comparing it against a re-derivation of the
/// host's arithmetic would only say the two derivations agree.
fn host_walk(
    proof: &stark::multilinear_table::MultiProof<GoldilocksField, FEE3>,
    statements: &[stark::multilinear_table::TableStatement<'_, GoldilocksField, FEE3>],
) -> HostWalk {
    let mut transcript = DefaultTranscript::<FEE3, RpxTranscriptHash>::new(&[]);
    let (z, alpha, beta) =
        stark::multilinear_table::absorb_roots_and_challenge(&mut transcript, &proof.roots, &[]);
    let mut walk = HostWalk {
        z,
        alpha,
        beta,
        outputs: Vec::new(),
        points: Vec::new(),
        values: Vec::new(),
    };
    for (table, statement) in proof.tables.iter().zip(statements) {
        let (output, reduced) = stark::multilinear_table::verify(
            table,
            *statement,
            &z,
            &alpha,
            &beta,
            &mut transcript,
            0,
        )
        .expect("the host verifies its own table");
        walk.outputs.push(output);
        walk.points.push(reduced.point.clone());
        walk.values.extend(reduced.column_values.iter().cloned());
    }
    walk
}

/// Every word the walk program hints, in the order it hints them: the carried
/// roots first, then each table's proof, table by table.
///
/// ⚠ ONE ORDER, WRITTEN ONCE. The program below and this function are the two
/// halves that have to agree, and nothing but execution catches a disagreement:
/// a misaligned arena hands the machine somebody else's field element and the
/// argument stops satisfying its own refusals.
fn walk_arena(
    proof: &stark::multilinear_table::MultiProof<GoldilocksField, FEE3>,
    layouts: &[multilinear::stacking::StackedLayout],
    config: &multilinear::whir_chain::ChainConfig,
) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = proof.roots.iter().map(commitment_to_digest).collect();
    for table in &proof.tables {
        words.push(ext_word(&table.bus_output.0));
        words.push(ext_word(&table.bus_output.1));
        for layer in &table.gkr.layers {
            for round in &layer.sumcheck.rounds {
                words.extend(round.evaluations.iter().map(ext_word));
            }
            for value in [&layer.p_lo, &layer.p_hi, &layer.q_lo, &layer.q_hi] {
                words.push(ext_word(value));
            }
        }
        for round in &table.constraint.sumcheck.rounds {
            words.extend(round.evaluations.iter().map(ext_word));
        }
        words.extend(table.constraint.factor_values.iter().map(ext_word));
        for round in &table.constraint.reduce.sumcheck.rounds {
            words.extend(round.evaluations.iter().map(ext_word));
        }
        words.extend(table.constraint.reduce.column_values.iter().map(ext_word));
    }
    for (group, opening) in proof.columns.iter().enumerate() {
        let shape = ChainShape::new(config, layouts[group].n_stack());
        for chain in &opening.polys {
            words.push(ext_word(&chain.final_value));
            super::whir_chain::push_round_words(&mut words, &shape, chain);
        }
    }
    words
}

/// The walk program: the roots block, the shared alpha ladder, and the
/// per-table walk, publishing every verdict the host's own pieces can be
/// compared against.
///
/// ★ THE ALPHA LADDER IS EPOCH-LEVEL AND SHARED. `emit_interaction` reads
/// `alpha_powers[i + 1]` (`whir_bus.rs:247`) and panics if the ladder is short,
/// so ONE ladder of the longest table's length serves every table — and
/// `table_verify_cost` does not charge it, which is why the assembled form has
/// to name it separately.
/// How much of the assembly a program carries, so the F1 can MEASURE each term
/// as a controlled delta instead of subtracting one total from another.
///
/// ★ A subtraction attributes a gap to whatever the arithmetic is written
/// against; a delta between two programs that differ by ONE stage attributes it
/// to that stage. The 3-row gap this instrument was built to chase is exactly
/// the case where the two answers differ.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WalkStage {
    /// The roots block and the shared alpha ladder, and nothing else.
    Spine,
    /// Those, then the per-table walk.
    Tables,
    /// Those, then the commitment groups.
    Groups,
}

fn walk_program(
    proof: &stark::multilinear_table::MultiProof<GoldilocksField, FEE3>,
    shapes: &[super::whir_table::TableShape<'_>],
    slots: &[&[usize]],
    layouts: &[multilinear::stacking::StackedLayout],
    domains: &[multilinear::whir::Domain<GoldilocksField>],
    config: &multilinear::whir_chain::ChainConfig,
    sizes: &[usize],
    stage: WalkStage,
) -> LfmProgram {
    let words = walk_arena(proof, layouts, config);
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(words.len() as u32);
    let mut at = 0u32;
    let carried: Vec<_> = proof
        .roots
        .iter()
        .map(|_| {
            let cell = b.hint_word(arena, at);
            at += 1;
            cell
        })
        .collect();

    let mut transcript = WhirTranscript::new();
    let (z, alpha, beta) = emit_roots_block(&mut b, &mut transcript, &carried, &[]);
    let ladder = super::whir_poly::emit_challenge_powers(&mut b, alpha, walk_alpha_powers(shapes));
    if stage == WalkStage::Spine {
        b.public(z.as_cell());
        b.public(alpha.as_cell());
        b.public(beta.as_cell());
        for wire in &ladder {
            b.public(wire.as_cell());
        }
        let program = compile(b.finish());
        validate(&program).expect("the spine must be admissible");
        return program;
    }

    // Each table's proof wires, in `walk_arena`'s order.
    let mut gkr_store: Vec<Vec<super::whir_gkr::GkrLayerWires>> = Vec::new();
    let mut sumcheck_store: Vec<Vec<Vec<Ext>>> = Vec::new();
    let mut factor_store: Vec<Vec<Ext>> = Vec::new();
    let mut reduce_store: Vec<Vec<Vec<Ext>>> = Vec::new();
    let mut column_store: Vec<Vec<Ext>> = Vec::new();
    let mut outputs: Vec<(Ext, Ext)> = Vec::new();
    for table in &proof.tables {
        let mut take = |b: &mut LfmBuilder, count: usize| -> Vec<Ext> {
            (0..count)
                .map(|_| {
                    let wire = b.hint_word(arena, at).as_ext();
                    at += 1;
                    wire
                })
                .collect()
        };
        let output = take(&mut b, 2);
        outputs.push((output[0], output[1]));
        let mut layers = Vec::with_capacity(table.gkr.layers.len());
        for layer in &table.gkr.layers {
            let sumcheck: Vec<Vec<Ext>> = layer
                .sumcheck
                .rounds
                .iter()
                .map(|round| take(&mut b, round.evaluations.len()))
                .collect();
            let halves = take(&mut b, 4);
            layers.push(super::whir_gkr::GkrLayerWires {
                sumcheck,
                p_lo: halves[0],
                p_hi: halves[1],
                q_lo: halves[2],
                q_hi: halves[3],
            });
        }
        gkr_store.push(layers);
        sumcheck_store.push(
            table
                .constraint
                .sumcheck
                .rounds
                .iter()
                .map(|round| take(&mut b, round.evaluations.len()))
                .collect(),
        );
        factor_store.push(take(&mut b, table.constraint.factor_values.len()));
        reduce_store.push(
            table
                .constraint
                .reduce
                .sumcheck
                .rounds
                .iter()
                .map(|round| take(&mut b, round.evaluations.len()))
                .collect(),
        );
        column_store.push(take(&mut b, table.constraint.reduce.column_values.len()));
    }

    let wires: Vec<super::whir_table::TableProofWires<'_>> = (0..proof.tables.len())
        .map(|i| super::whir_table::TableProofWires {
            bus_output: outputs[i],
            gkr: &gkr_store[i],
            sumcheck: &sumcheck_store[i],
            factor_values: &factor_store[i],
            reduce: super::whir_reduce::ReduceWires {
                sumcheck: &reduce_store[i],
                column_values: &column_store[i],
            },
        })
        .collect();
    let plans: Vec<super::whir_epoch::PreprocessedPlan<'_>> = (0..proof.tables.len())
        .map(|_| super::whir_epoch::PreprocessedPlan {
            settled: 0,
            route: super::whir_epoch::PreprocessedRoute::None,
        })
        .collect();

    let walk = super::whir_epoch::emit_table_walk(
        &mut b,
        &mut transcript,
        &wires,
        shapes,
        &plans,
        slots,
        z,
        &ladder,
        beta,
    );

    // ★ THE GROUPS. Each polynomial's ROOT is the cell the roots block already
    // hinted — the same wire, not a second copy. A program that hinted it twice
    // would let a prover absorb one root into the transcript and open the chain
    // against another, and the honest arena would look identical; the schema
    // assertion below is what makes that unspellable.
    let mut storages: Vec<Vec<super::whir_chain::RoundStorage>> = Vec::new();
    let mut finals: Vec<Vec<Ext>> = Vec::new();
    let mut chain_shapes: Vec<ChainShape> = Vec::new();
    for (group, opening) in proof.columns.iter().enumerate() {
        let shape = ChainShape::new(config, layouts[group].n_stack());
        let mut group_storage = Vec::new();
        let mut group_finals = Vec::new();
        for _ in &opening.polys {
            group_finals.push(b.hint_word(arena, at).as_ext());
            at += 1;
            group_storage.push(super::whir_chain::RoundStorage::hint(
                &mut b, arena, at, &shape,
            ));
            at += super::whir_chain::RoundStorage::words(&shape);
        }
        storages.push(group_storage);
        finals.push(group_finals);
        chain_shapes.push(shape);
    }
    let openings: Vec<Vec<_>> = storages
        .iter()
        .map(|group| {
            group
                .iter()
                .map(super::whir_chain::RoundStorage::openings)
                .collect::<Vec<_>>()
        })
        .collect();
    let chain_wires: Vec<Vec<Vec<super::whir_chain::ChainRoundWires<'_>>>> = storages
        .iter()
        .zip(&openings)
        .map(|(group, group_openings)| {
            group
                .iter()
                .zip(group_openings)
                .map(|(chain, (current, next))| chain.wires(current, next))
                .collect()
        })
        .collect();
    let mut root_at = 0usize;
    let mut polys: Vec<Vec<super::whir_stacked::StackedPolyWires<'_>>> = Vec::new();
    for (group, opening) in proof.columns.iter().enumerate() {
        polys.push(
            (0..opening.polys.len())
                .map(|poly| super::whir_stacked::StackedPolyWires {
                    rounds: &chain_wires[group][poly],
                    root: carried[root_at + poly],
                    final_value: finals[group][poly],
                })
                .collect(),
        );
        root_at += layouts[group].num_polys();
    }
    let groups: Vec<super::whir_epoch::GroupWires<'_>> = (0..proof.columns.len())
        .map(|group| super::whir_epoch::GroupWires {
            layout: &layouts[group],
            polys: &polys[group],
            shape: &chain_shapes[group],
            domain: &domains[group],
        })
        .collect();
    let gammas = if stage == WalkStage::Groups {
        super::whir_epoch::emit_group_walk(&mut b, &mut transcript, &groups, sizes, &walk)
    } else {
        Vec::new()
    };

    b.public(z.as_cell());
    b.public(alpha.as_cell());
    b.public(beta.as_cell());
    for gamma in &gammas {
        b.public(gamma.as_cell());
    }
    for (p, q) in &walk.outputs {
        b.public(p.as_cell());
        b.public(q.as_cell());
    }
    for point in &walk.points {
        for wire in point {
            b.public(wire.as_cell());
        }
    }
    for value in &walk.values {
        b.public(value.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("the walk must be admissible");
    program
}

/// The shared ladder's length: the longest bus in the walk.
fn walk_alpha_powers(shapes: &[super::whir_table::TableShape<'_>]) -> usize {
    shapes
        .iter()
        .map(|shape| super::whir_bus::alpha_powers_read(shape.bus))
        .max()
        .unwrap_or(1)
}

/// ★★ THE VALUE GATE ON THE PER-TABLE WALK: three tables, one transcript, and
/// the machine's verdict is the host's verdict table by table.
///
/// The machine derives every challenge itself, so executing an honest proof at
/// all is already the challenge-stream comparison — a wrong challenge anywhere
/// leaves a refusal with no satisfying assignment. What the published values add
/// is that the walk THREADED the sponge: table 1's challenges are a function of
/// where table 0 left it, so a walk that entered each table fresh would still
/// execute table 0 and diverge from table 1 onward.
#[test]
fn the_table_walk_computes_what_the_host_computes() {
    let options = stark::proof::options::ProofOptions::default_test_options();
    let (cpu, add, mul) = walk_airs(&options);
    let columns = walk_columns();
    let config = walk_config();

    let committed = stark::multilinear_table::CommittedTables::<
        _,
        _,
        multilinear::whir_hash::RpxWhir,
    >::commit_grouped(
        vec![
            walk_table(&cpu, &columns[0]),
            walk_table(&add, &columns[1]),
            walk_table(&mul, &columns[2]),
        ],
        &WALK_SIZES,
        &config,
    )
    .expect("the three tables commit in two groups");

    let mut prover = DefaultTranscript::<FEE3, RpxTranscriptHash>::new(&[]);
    let proof = stark::multilinear_table::multi_prove(&committed, &config, &mut prover, None)
        .expect("the fixture proves");
    let statements: Vec<_> = committed.tables().iter().map(|t| t.statement()).collect();

    // ⚠ THE FIXTURE MUST BE A PROOF THE HOST ACCEPTS, or every reference below
    // refers to nothing. Two groups and three tables, checked by name.
    assert_eq!(
        proof.columns.len(),
        WALK_SIZES.len(),
        "two commitment groups"
    );
    assert_eq!(proof.tables.len(), 3, "three tables");
    let layouts: Vec<_> = committed
        .groups()
        .iter()
        .map(|g| g.layout().clone())
        .collect();
    let domains: Vec<_> = committed
        .groups()
        .iter()
        .map(|g| g.domain().clone())
        .collect();
    let mut verifier = DefaultTranscript::<FEE3, RpxTranscriptHash>::new(&[]);
    stark::multilinear_table::multi_verify::<_, _, _, multilinear::whir_hash::RpxWhir>(
        &proof,
        &statements,
        &layouts,
        &domains,
        committed.sizes(),
        &FEE::zero(),
        &config,
        &mut verifier,
        None,
    )
    .expect("the fixture must be a proof the host accepts");

    let host = host_walk(&proof, &statements);

    // The shapes, rebuilt from the statements the host itself verified against.
    let buses: Vec<Vec<stark::multilinear_logup::InteractionShape<FEE3>>> = statements
        .iter()
        .map(|statement| {
            let slots = statement.slot_of.to_vec();
            stark::multilinear_logup::interaction_shapes(
                statement.interactions,
                statement.slot_of.len(),
                |column| {
                    slots
                        .get(column)
                        .copied()
                        .ok_or(multilinear::Error::UnknownPolynomial {
                            index: column,
                            len: slots.len(),
                        })
                },
            )
            .expect("the bus probes")
        })
        .collect();
    let shapes: Vec<super::whir_table::TableShape<'_>> = statements
        .iter()
        .zip(&buses)
        .map(|(statement, bus)| super::whir_table::TableShape {
            ir: statement.shape,
            bus,
            kinds: statement.kinds,
            num_columns: statement.slot_of.len(),
            num_vars: statement.num_vars,
        })
        .collect();
    let slots: Vec<&[usize]> = statements.iter().map(|s| s.slot_of).collect();

    let program = walk_program(
        &proof,
        &shapes,
        &slots,
        &layouts,
        &domains,
        &config,
        &WALK_SIZES,
        WalkStage::Groups,
    );
    let arena = walk_arena(&proof, &layouts, &config);
    let exec = execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER)
        .expect("the machine must execute the host's own proof");

    let published: Vec<FEE> = exec
        .public_words
        .iter()
        .map(|(_, word)| word_as_ext(word).expect("a published extension element"))
        .collect();
    let mut at = 0usize;
    let mut next = |count: usize| -> Vec<FEE> {
        let slice = published[at..at + count].to_vec();
        at += count;
        slice
    };

    let challenges = next(3);
    assert_eq!(challenges[0], host.z, "z must be the host's");
    assert_eq!(challenges[1], host.alpha, "alpha must be the host's");
    assert_eq!(challenges[2], host.beta, "beta must be the host's");

    // ★ THE GROUP WALK'S OWN GATE: each group's batching challenge is the one
    // the HOST verifier sampled at that point in the stream. The offsets are
    // derived from the structure — three challenges, then every table's draws,
    // then per group its gamma and its chains' — and the derivation is checked
    // against the recorder's total, so an offset that drifts fails here rather
    // than silently comparing the wrong pair.
    let gammas = next(proof.columns.len());
    let recorded = recorded_draws(
        &proof,
        &statements,
        &layouts,
        &domains,
        committed.sizes(),
        &config,
    );
    let mut draw_at = 3 + recorded.table_draws;
    for (group, gamma) in gammas.iter().enumerate() {
        assert_eq!(
            *gamma, recorded.sampled[draw_at],
            "group {group}: the batching challenge must be the one the host sampled"
        );
        draw_at += 1 + recorded.chain_draws[group];
    }
    assert_eq!(
        draw_at,
        recorded.sampled.len(),
        "the derived offsets must account for every extension element the host drew"
    );

    for (index, (p, q)) in host.outputs.iter().enumerate() {
        let got = next(2);
        assert_eq!(
            &got[0], p,
            "table {index}: the bus numerator must be the host's"
        );
        assert_eq!(
            &got[1], q,
            "table {index}: the bus denominator must be the host's"
        );
    }
    for (index, point) in host.points.iter().enumerate() {
        let got = next(point.len());
        assert_eq!(
            &got, point,
            "table {index}: the reduced point must be the host's — this is the half that fails \
             when the sponge is not threaded from the previous table"
        );
    }
    let got = next(host.values.len());
    assert_eq!(
        got, host.values,
        "every table's claimed column values must be the host's"
    );
    assert_eq!(at, published.len(), "every published word accounted for");
}

/// ★ F1 FOR THE PER-TABLE WALK: the emitted count is the closed form's, and the
/// convention is named.
///
/// ⚠ THE CONVENTION. `table_walk_cost` is CONST-FREE in exactly the sense every
/// other form in `whir_epoch` is: it reports operations and carries its interned
/// words as a SET, because one epoch program has one pool and the tables'
/// constants collide in it. So the measurement subtracts the program's constants
/// and compares against `operations()`, and the pool is asserted separately as a
/// count of distinct words.
///
/// ⚠ AND THE TWO TERMS THE WALK'S FORM DOES NOT OWN, both subtracted here with
/// their own reasons: the ROOTS BLOCK, whose form is `roots_block_cost`, and the
/// shared ALPHA LADDER, which `table_verify_cost` does not charge because the
/// ladder is EPOCH-level — one ladder serves every table, so a per-table form
/// that charged it would charge it once per table.
#[test]
fn the_table_walk_emits_its_closed_form() {
    let options = stark::proof::options::ProofOptions::default_test_options();
    let (cpu, add, mul) = walk_airs(&options);
    let columns = walk_columns();
    let config = walk_config();

    let committed = stark::multilinear_table::CommittedTables::<
        _,
        _,
        multilinear::whir_hash::RpxWhir,
    >::commit_grouped(
        vec![
            walk_table(&cpu, &columns[0]),
            walk_table(&add, &columns[1]),
            walk_table(&mul, &columns[2]),
        ],
        &WALK_SIZES,
        &config,
    )
    .expect("the three tables commit in two groups");
    let mut prover = DefaultTranscript::<FEE3, RpxTranscriptHash>::new(&[]);
    let proof = stark::multilinear_table::multi_prove(&committed, &config, &mut prover, None)
        .expect("the fixture proves");
    let statements: Vec<_> = committed.tables().iter().map(|t| t.statement()).collect();

    let buses: Vec<Vec<stark::multilinear_logup::InteractionShape<FEE3>>> = statements
        .iter()
        .map(|statement| {
            let slots = statement.slot_of.to_vec();
            stark::multilinear_logup::interaction_shapes(
                statement.interactions,
                statement.slot_of.len(),
                |column| {
                    slots
                        .get(column)
                        .copied()
                        .ok_or(multilinear::Error::UnknownPolynomial {
                            index: column,
                            len: slots.len(),
                        })
                },
            )
            .expect("the bus probes")
        })
        .collect();
    let shapes: Vec<super::whir_table::TableShape<'_>> = statements
        .iter()
        .zip(&buses)
        .map(|(statement, bus)| super::whir_table::TableShape {
            ir: statement.shape,
            bus,
            kinds: statement.kinds,
            num_columns: statement.slot_of.len(),
            num_vars: statement.num_vars,
        })
        .collect();
    let slots: Vec<&[usize]> = statements.iter().map(|s| s.slot_of).collect();
    let plans: Vec<super::whir_epoch::PreprocessedPlan<'_>> = (0..shapes.len())
        .map(|_| super::whir_epoch::PreprocessedPlan {
            settled: 0,
            route: super::whir_epoch::PreprocessedRoute::None,
        })
        .collect();

    let layouts: Vec<_> = committed
        .groups()
        .iter()
        .map(|g| g.layout().clone())
        .collect();
    let domains: Vec<_> = committed
        .groups()
        .iter()
        .map(|g| g.domain().clone())
        .collect();
    let words = walk_arena(&proof, &layouts, &config);

    // The predictions, each from its own landed form.
    let (roots_leg, roots_schedule) = roots_block_cost(proof.roots.len(), 0, SpongeEntry::fresh());
    let roots_ops = roots_leg + roots_schedule.rows();
    let ladder = super::whir_poly::challenge_powers_rows(walk_alpha_powers(&shapes));
    let plans: Vec<super::whir_epoch::PreprocessedPlan<'_>> = (0..shapes.len())
        .map(|_| super::whir_epoch::PreprocessedPlan {
            settled: 0,
            route: super::whir_epoch::PreprocessedRoute::None,
        })
        .collect();
    let walk = super::whir_epoch::table_walk_cost(&shapes, &plans, roots_schedule.entry());
    let table_shapes: Vec<(usize, usize)> = shapes
        .iter()
        .map(|shape| (shape.num_columns, shape.num_vars))
        .collect();
    let group_costs = epoch_group_costs(&table_shapes, &WALK_SIZES, &layouts, &config, walk.entry);
    let group_ops: usize = group_costs.iter().map(|cost| cost.operations()).sum();
    let group_perms: usize = group_costs.iter().map(|cost| cost.perms()).sum();
    let predicted_perms = roots_schedule.perms() + walk.perms + group_perms;

    let stages = [WalkStage::Spine, WalkStage::Tables, WalkStage::Groups];
    let mut measured = Vec::new();
    let mut perms = Vec::new();
    for stage in stages {
        let program = walk_program(
            &proof,
            &shapes,
            &slots,
            &layouts,
            &domains,
            &config,
            &WALK_SIZES,
            stage,
        );
        let hints = super::whir_chain_tests::hint_rows(&program);
        // ⚠ The spine stops before the tables, so it hints the ROOTS alone; the
        // two full stages hint every word the filler writes. Each stage's own
        // hints are subtracted from its own count, so the deltas below stay
        // honest either way — what this asserts is that no stage hints a word
        // TWICE, which is what refuses a second copy of a commitment root: the
        // chains open against the very cells the roots block absorbed, and a
        // program that hinted them again would hint more words than the filler
        // writes while looking identical to every value gate.
        let expected_hints = if stage == WalkStage::Spine {
            proof.roots.len()
        } else {
            words.len()
        };
        assert_eq!(
            hints, expected_hints,
            "every word this stage reads is hinted exactly once"
        );
        assert_eq!(
            program.arena_schema.lens,
            vec![words.len() as u32],
            "one arena, of exactly the words the filler writes"
        );
        let consts = const_rows(&program);
        measured.push(program.instrs.len() - consts - hints - program.public_len as usize);
        perms.push(super::whir_chain_tests::perm_rows(&program));
    }

    let spine = measured[0];
    let table_walk = measured[1] - measured[0];
    let group_walk = measured[2] - measured[1];
    println!(
        "the two walks, by stage: spine {spine} emitted / {} predicted ({roots_ops} roots block \
         + {ladder} alpha ladder); table walk {table_walk} / {}; groups {group_walk} / \
         {group_ops}; permutations {} / {predicted_perms}",
        roots_ops + ladder,
        walk.operations(),
        perms[2],
    );

    // ★ EACH TERM AGAINST ITS OWN STAGE, so a gap names the leg it belongs to
    // rather than landing on whichever term the arithmetic was written against.
    assert_eq!(
        spine,
        roots_ops + ladder,
        "the spine: the roots block and the shared alpha ladder"
    );
    assert_eq!(
        table_walk,
        walk.operations(),
        "the per-table walk, as the delta between the two programs that differ by it"
    );
    assert_eq!(
        group_walk, group_ops,
        "the commitment groups, as the delta between the two programs that differ by them"
    );
    assert_eq!(
        perms[2], predicted_perms,
        "permutations: the roots block's sponge, every table's, and every chain's"
    );
}

/// What the HOST drew over a whole `multi_verify`, and how those draws split.
struct RecordedDraws {
    /// Every extension element the host sampled, in order.
    sampled: Vec<FEE>,
    /// Draws the TABLE phase made, after the roots block's three.
    table_draws: usize,
    /// Draws each group's chains made, after that group's batching challenge.
    chain_draws: Vec<usize>,
}

/// ★ The host's draw stream, recorded — and split by a derivation the total
/// then checks.
///
/// `table_draws` is MEASURED, by recording a second run that stops where the
/// table phase does; `chain_draws` is DERIVED from the chain structure, the same
/// expression `whir_stacked_tests` asserts against its own recorder. Neither is
/// read off the full stream by eye, and the caller asserts that the two together
/// account for every element — so an offset that drifts fails instead of
/// comparing the wrong pair (instance 68's shape, applied to an index).
fn recorded_draws(
    proof: &stark::multilinear_table::MultiProof<GoldilocksField, FEE3>,
    statements: &[stark::multilinear_table::TableStatement<'_, GoldilocksField, FEE3>],
    layouts: &[multilinear::stacking::StackedLayout],
    domains: &[multilinear::whir::Domain<GoldilocksField>],
    sizes: &[usize],
    config: &multilinear::whir_chain::ChainConfig,
) -> RecordedDraws {
    // The table phase alone, on its own recorder.
    let mut tables_only = super::whir_chain_tests::Recording::new();
    let (z, alpha, beta) =
        stark::multilinear_table::absorb_roots_and_challenge(&mut tables_only, &proof.roots, &[]);
    for (table, statement) in proof.tables.iter().zip(statements) {
        stark::multilinear_table::verify(table, *statement, &z, &alpha, &beta, &mut tables_only, 0)
            .expect("the host verifies its own table");
    }
    let table_draws = tables_only.sampled.len() - 3;

    // The whole verify, so the groups' own draws are in the same stream.
    let mut whole = super::whir_chain_tests::Recording::new();
    stark::multilinear_table::multi_verify::<_, _, _, multilinear::whir_hash::RpxWhir>(
        proof,
        statements,
        layouts,
        domains,
        sizes,
        &FEE::zero(),
        config,
        &mut whole,
        None,
    )
    .expect("the host accepts its own proof");

    let chain_draws = layouts
        .iter()
        .map(|layout| {
            let shape = ChainShape::new(config, layout.n_stack());
            layout.num_polys() * (shape.num_vars + 2 * (shape.rounds() - 1))
        })
        .collect();

    RecordedDraws {
        sampled: whole.sampled,
        table_draws,
        chain_draws,
    }
}
