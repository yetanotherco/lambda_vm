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

use crate::multilinear_continuation::epoch_groups;
use crate::multilinear_prove::{chain_config, stacks};
use crate::tables::types::{FEE, GoldilocksExtension as FEE3};

use super::algebraic_commit::commitment_to_digest;
use super::builder::LfmBuilder;
use super::compiler::compile;
use super::executor::execute;
use super::validator::validate;
use super::whir_chain::{ChainShape, chain_shape_rows};
use super::whir_chain_tests::const_rows;
use super::whir_epoch::{
    closure_rows, emit_epoch_closure, emit_roots_block, epoch_group_costs, expected_rows,
    group_columns, roots_block_constants, roots_block_cost,
};
use super::whir_transcript::{SpongeEntry, WhirTranscript};
use super::word::{ext_word, word_as_ext};

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
