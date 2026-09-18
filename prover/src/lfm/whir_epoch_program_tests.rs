//! Gates for the assembled epoch verify, and the production-shape census.
//!
//! The census is here rather than in a log because it is an EVALUATION of the
//! gated forms at the block's own shape, not an estimate: every term comes from
//! a form that has its own F1 against an emitted program, and the shape comes
//! from a box run whose file is named below. What it cannot evaluate is the
//! per-table half — that needs the real AIRs, which need the guest ELF — so
//! that one term is an input, quoted with its measurement's convention.

use crate::multilinear_continuation::epoch_groups;
use crate::multilinear_prove::{chain_config, stacks};

use super::whir_chain::{ChainShape, chain_shape_rows};
use super::whir_epoch::{closure_rows, epoch_group_costs, group_columns};
use super::whir_transcript::SpongeEntry;

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
        1 + 8 * 4 - 1,
        "a published byte costs two MulAdds, an inverse and an Add, with the \
         alpha square once and the first sum needing no Add"
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
