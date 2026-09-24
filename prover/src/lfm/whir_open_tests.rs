//! Gates for the WHIR query opening.
//!
//! The commitment under test is a real one — `CodewordCommitment` over a real
//! codeword, opened at a real index — so the leaf the machine recomputes is the
//! leaf a committer actually hashed, and the path is a path the tree actually
//! produced. `from_codeword_on_host` rather than `from_codeword`, so no test
//! here asks for a card.

use multilinear::whir_commit::{CodewordCommitment, CosetOpening, verify_opening};
use multilinear::whir_hash::RpxWhir;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::commitment_to_digest;
use super::builder::{Ext, Felt, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapDigest;
use super::executor::execute;
use super::validator::validate;
use super::whir_open::{
    BlockValues, emit_verify_opening, verify_opening_perms, verify_opening_rows,
};
use super::word::{LfmWord, ext_word};

type F = GoldilocksField;
type E = GoldilocksExtension;

fn base(v: u64) -> FE {
    FE::from(v.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 3)
}

fn ext(v: u64) -> FEE {
    FEE::new([base(v), base(v ^ 0x5151), base(v.wrapping_add(7))])
}

/// A base word `(v, 0, 0, 0)` — what a hinted base opening value must be, and
/// what [`the_leaf_pins_a_hinted_base_value_to_its_low_lane`] shows the bus
/// forces it to be.
fn base_word(v: FE) -> LfmWord {
    [v, FE::zero(), FE::zero(), FE::zero()]
}

/// One shape: the codeword's log size, the fold, and which field the block is
/// held in. `(log_domain, log_folding)` gives a tree of `log_domain −
/// log_folding` levels and a block of `2^log_folding` values.
struct Shape {
    log_domain: usize,
    log_folding: usize,
    name: &'static str,
}

/// The shapes a chain actually reaches, small enough to commit on a laptop: a
/// block of two (the `S mod k` tail round's width), of four, and the production
/// sixteen; and a one-level tree, where the walk's loop runs once.
const SHAPES: &[Shape] = &[
    Shape {
        log_domain: 6,
        log_folding: 1,
        name: "block 2, 5 levels",
    },
    Shape {
        log_domain: 6,
        log_folding: 2,
        name: "block 4, 4 levels",
    },
    Shape {
        log_domain: 8,
        log_folding: 4,
        name: "block 16, 4 levels",
    },
    Shape {
        log_domain: 5,
        log_folding: 4,
        name: "block 16, 1 level",
    },
];

/// The arena, in the order [`opening_program`] hints it: the block's values,
/// then one word per sibling, then the root, then the index.
fn opening_arena(
    values: &[LfmWord],
    siblings: &[LfmWord],
    root: LfmWord,
    index: usize,
) -> Vec<LfmWord> {
    let mut words = values.to_vec();
    words.extend_from_slice(siblings);
    words.push(root);
    words.push([FE::from(index as u64), FE::zero(), FE::zero(), FE::zero()]);
    words
}

/// Hints in that order, decomposes the index in ONE `BitDec` — which is what
/// `WhirTranscript::sample_u64_pow2` emits, so the assembled verifier hands
/// these bits over rather than rebuilding them — and publishes the root so the
/// program has an output.
fn opening_program(block: usize, is_ext: bool, depth: usize) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena((block + depth + 2) as u32);

    // ⚠ Only the field this round holds is hinted. Building both vectors and
    // choosing between them afterwards hints the block TWICE, which F1 caught
    // as `Hint 11` where the shape says nine.
    let mut base_values: Vec<Felt> = Vec::new();
    let mut ext_values: Vec<Ext> = Vec::new();
    if is_ext {
        ext_values = (0..block)
            .map(|i| b.hint_word(arena, i as u32).as_ext())
            .collect();
    } else {
        base_values = (0..block).map(|i| b.hint_felt(arena, i as u32)).collect();
    }

    let siblings: Vec<WrapDigest> = (0..depth)
        .map(|i| WrapDigest::from_cell(b.hint_word(arena, (block + i) as u32)))
        .collect();
    let root = b.hint_word(arena, (block + depth) as u32);
    let root_lanes = b.unpack(root);
    let index = b.hint_felt(arena, (block + depth + 1) as u32);
    let bits = b.bit_dec(index, depth);

    let values = if is_ext {
        BlockValues::Ext(&ext_values)
    } else {
        BlockValues::Base(&base_values)
    };
    emit_verify_opening(&mut b, values, &bits, &siblings, &root_lanes);
    b.public(root);

    let program = compile(b.finish());
    validate(&program).expect("the opening leg must be admissible");
    program
}

/// Rows the plumbing above contributes and the leg does not: the hints, the
/// root's one `Unpack`, the one `BitDec`, and the public. Counted by
/// construction rather than by differencing against a second program.
fn opening_plumbing(block: usize, depth: usize) -> usize {
    (block + depth + 2) + 1 + 1 + 1
}

fn const_rows(program: &LfmProgram) -> usize {
    program
        .instrs
        .iter()
        .filter(|instr| matches!(instr, super::instr::Instr::Const { .. }))
        .count()
}

/// The instruction histogram, by variant. F1 twice named a term I had written
/// down before measuring; this is what names them.
fn histogram(program: &LfmProgram) -> String {
    use super::instr::Instr;
    let mut counts = [0usize; 11];
    for instr in &program.instrs {
        let i = match instr {
            Instr::Const { .. } => 0,
            Instr::BaseAlu { .. } => 1,
            Instr::ExtAlu { .. } => 2,
            Instr::Select { .. } => 3,
            Instr::BitDec { .. } => 4,
            Instr::Hash { .. } => 5,
            Instr::Hint { .. } => 6,
            Instr::Pack { .. } => 7,
            Instr::Unpack { .. } => 8,
            Instr::Public { .. } => 9,
            // No byte hash is reachable from this leg under the algebraic pin;
            // counted so a change that reached one would show here.
            Instr::KeccakF(_) | Instr::Blake3(_) => 10,
        };
        counts[i] += 1;
    }
    let names = [
        "Const", "BaseAlu", "ExtAlu", "Select", "BitDec", "Hash", "Hint", "Pack", "Unpack",
        "Public", "ByteHash",
    ];
    names
        .iter()
        .zip(counts)
        .filter(|(_, c)| *c > 0)
        .map(|(n, c)| format!("{n} {c}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `LFM_HASH` invocations: one per sponge permutation, whether it is a leaf's
/// duplex block or a Merkle parent's compress. Both lower to `Instr::Hash` and
/// differ only in `HashMode`, so counting the variant counts permutations.
fn perm_rows(program: &LfmProgram) -> usize {
    program
        .instrs
        .iter()
        .filter(|instr| matches!(instr, super::instr::Instr::Hash { .. }))
        .count()
}

/// Commits a base-field codeword and opens one block.
fn base_commitment(shape: &Shape, seed: u64) -> (CodewordCommitment<F, RpxWhir>, Vec<FE>) {
    let codeword: Vec<FE> = (0..1usize << shape.log_domain)
        .map(|i| base(seed + i as u64))
        .collect();
    let commitment = CodewordCommitment::<F, RpxWhir>::from_codeword_on_host(
        codeword.clone(),
        shape.log_folding,
    )
    .expect("the codeword commits");
    (commitment, codeword)
}

/// The same over the cubic extension — every block but round 0's current one.
fn ext_commitment(shape: &Shape, seed: u64) -> CodewordCommitment<E, RpxWhir> {
    let codeword: Vec<FEE> = (0..1usize << shape.log_domain)
        .map(|i| ext(seed + i as u64))
        .collect();
    CodewordCommitment::<E, RpxWhir>::from_codeword_on_host(codeword, shape.log_folding)
        .expect("the codeword commits")
}

fn siblings_of<V: math::field::traits::IsField>(opening: &CosetOpening<V>) -> Vec<LfmWord> {
    opening
        .proof
        .merkle_path
        .iter()
        .map(commitment_to_digest)
        .collect()
}

/// ★ F1 for the opening: rows and PERMUTATIONS, both against their closed
/// forms, at every shape and in both fields.
///
/// The permutation count is pinned separately and on purpose. It is the term
/// the chain's cost is dominated by, it is a function of the block's felts and
/// the tree's depth alone, and pinning it apart from the rows means a form that
/// moved work between hashing and arithmetic at constant total would fail one
/// of the two rather than neither.
#[test]
fn the_opening_emits_its_closed_form() {
    for shape in SHAPES {
        let block = 1usize << shape.log_folding;
        let depth = shape.log_domain - shape.log_folding;
        for is_ext in [false, true] {
            let felts = if is_ext { 3 * block } else { block };
            let unpacks = if is_ext { block } else { 0 };
            let program = opening_program(block, is_ext, depth);
            let constants = const_rows(&program);
            let measured = program.instrs.len() - opening_plumbing(block, depth) - constants;
            let predicted = verify_opening_rows(felts, unpacks, depth);
            let perms = perm_rows(&program);
            let predicted_perms = verify_opening_perms(felts, depth);
            println!(
                "{} {}: {measured:>3} rows emitted, {predicted:>3} predicted; \
                 {perms:>2} permutations, {predicted_perms:>2} predicted \
                 ({constants} interned constants)",
                shape.name,
                if is_ext { "ext " } else { "base" },
            );
            println!("    [{}]", histogram(&program));
            assert_eq!(
                measured,
                predicted,
                "{} {}: the emitted row count must equal the closed form",
                shape.name,
                if is_ext { "ext" } else { "base" }
            );
            assert_eq!(
                perms,
                predicted_perms,
                "{} {}: the permutation count must equal the closed form",
                shape.name,
                if is_ext { "ext" } else { "base" }
            );
        }
    }
}

/// ★ The leg accepts exactly what `whir_commit::verify_opening` accepts.
///
/// The host runs first and must return `true`, so the arm is about an opening
/// that genuinely authenticates; then the machine must EXECUTE it. Nothing here
/// compares the emitter against the emitter: the leaf the machine recomputes
/// comes out of `prover::lfm::rpo`, and the leaf the tree hashed came out of
/// `crypto::hash::rpx` — two implementations that have to agree.
#[test]
fn the_opening_accepts_what_the_host_accepts() {
    for shape in SHAPES {
        let block = 1usize << shape.log_folding;
        let depth = shape.log_domain - shape.log_folding;
        let num_leaves = 1usize << depth;

        let (commitment, _) = base_commitment(shape, 0x31);
        let program = opening_program(block, false, depth);
        for index in [0usize, 1, num_leaves / 2, num_leaves - 1] {
            let opening = commitment.open(index).expect("the block opens");
            assert!(
                verify_opening::<F, RpxWhir>(&commitment.root(), depth, index, &opening),
                "{}: the host must accept its own opening at {index}",
                shape.name
            );
            let values: Vec<LfmWord> = opening.values.iter().copied().map(base_word).collect();
            let arena = opening_arena(
                &values,
                &siblings_of(&opening),
                commitment_to_digest(&commitment.root()),
                index,
            );
            execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER).unwrap_or_else(|e| {
                panic!(
                    "{} base at {index}: the machine refused a valid opening: {e:?}",
                    shape.name
                )
            });
        }

        let commitment = ext_commitment(shape, 0x77);
        let program = opening_program(block, true, depth);
        for index in [0usize, 1, num_leaves - 1] {
            let opening = commitment.open(index).expect("the block opens");
            assert!(
                verify_opening::<E, RpxWhir>(&commitment.root(), depth, index, &opening),
                "{}: the host must accept its own opening at {index}",
                shape.name
            );
            let values: Vec<LfmWord> = opening.values.iter().map(ext_word).collect();
            let arena = opening_arena(
                &values,
                &siblings_of(&opening),
                commitment_to_digest(&commitment.root()),
                index,
            );
            execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER).unwrap_or_else(|e| {
                panic!(
                    "{} ext at {index}: the machine refused a valid opening: {e:?}",
                    shape.name
                )
            });
        }
    }
}

/// ★ The tamper arm, in the three halves this lane's legs use: the untouched
/// opening EXECUTES (or the refusals below are refusals of nothing), the HOST
/// rejects each forgery, and the machine refuses it.
///
/// Four sites, each a different part of the argument: a value (the leaf moves),
/// a sibling (the path moves), the root (what the walk is compared against),
/// and the INDEX (the same leaf against the same root at the wrong position —
/// the one a walk that ignored its bits would still accept).
#[test]
fn a_tampered_opening_cannot_execute() {
    let shape = &SHAPES[2];
    let block = 1usize << shape.log_folding;
    let depth = shape.log_domain - shape.log_folding;
    let num_leaves = 1usize << depth;
    let index = 3usize;

    let commitment = ext_commitment(shape, 0xB1);
    let root = commitment.root();
    let opening = commitment.open(index).expect("the block opens");
    let program = opening_program(block, true, depth);

    let honest_values: Vec<LfmWord> = opening.values.iter().map(ext_word).collect();
    let honest_siblings = siblings_of(&opening);
    let honest_root = commitment_to_digest(&root);

    assert!(
        verify_opening::<E, RpxWhir>(&root, depth, index, &opening),
        "the control opening must authenticate"
    );
    assert!(
        execute(
            &program,
            &[opening_arena(
                &honest_values,
                &honest_siblings,
                honest_root,
                index
            )],
            &crate::hash_pin::BLOCK_HASHER
        )
        .is_ok(),
        "the untouched opening must execute, or the arm below proves nothing"
    );

    // A value: the leaf hashes to something else.
    let mut forged = opening.clone();
    forged.values[block / 2] += FEE::one();
    assert!(
        !verify_opening::<E, RpxWhir>(&root, depth, index, &forged),
        "the host must reject a corrupted value"
    );
    let values: Vec<LfmWord> = forged.values.iter().map(ext_word).collect();
    assert!(
        execute(
            &program,
            &[opening_arena(&values, &honest_siblings, honest_root, index)],
            &crate::hash_pin::BLOCK_HASHER
        )
        .is_err(),
        "the machine must refuse a corrupted value"
    );

    // A sibling: the path walks somewhere else.
    let mut forged = opening.clone();
    forged.proof.merkle_path[0][0] ^= 1;
    assert!(
        !verify_opening::<E, RpxWhir>(&root, depth, index, &forged),
        "the host must reject a corrupted sibling"
    );
    assert!(
        execute(
            &program,
            &[opening_arena(
                &honest_values,
                &siblings_of(&forged),
                honest_root,
                index
            )],
            &crate::hash_pin::BLOCK_HASHER
        )
        .is_err(),
        "the machine must refuse a corrupted sibling"
    );

    // The root: the walk arrives, at the wrong place.
    let mut wrong_root = root;
    wrong_root[0] ^= 1;
    assert!(
        !verify_opening::<E, RpxWhir>(&wrong_root, depth, index, &opening),
        "the host must reject a wrong root"
    );
    assert!(
        execute(
            &program,
            &[opening_arena(
                &honest_values,
                &honest_siblings,
                commitment_to_digest(&wrong_root),
                index
            )],
            &crate::hash_pin::BLOCK_HASHER
        )
        .is_err(),
        "the machine must refuse a wrong root"
    );

    // The INDEX: the right leaf and the right path, at the wrong position. A
    // walk that took the same order at every level would pass everything above
    // and fail only here.
    let elsewhere = (index + 1) % num_leaves;
    assert!(
        !verify_opening::<E, RpxWhir>(&root, depth, elsewhere, &opening),
        "the host must reject an opening claimed at the wrong index"
    );
    assert!(
        execute(
            &program,
            &[opening_arena(
                &honest_values,
                &honest_siblings,
                honest_root,
                elsewhere
            )],
            &crate::hash_pin::BLOCK_HASHER
        )
        .is_err(),
        "the machine must refuse an opening claimed at the wrong index"
    );
}

/// ★ The bus pins a hinted base value's upper lanes, which is what makes it
/// safe for the fold to lift one.
///
/// A hint constrains nothing — `hint_felt` is `hint_word` retyped — so lanes
/// 1–3 of a round-0 opening value are free as far as the arena is concerned,
/// and `Felt::as_ext` would carry them into the fold as coordinates. The leaf's
/// `Pack` is what forbids it: `Pack` receives each lane as a `base_token`
/// `(addr, v, 0, 0, 0)`, and the memory bus is a multiset, so the arena's
/// word-token send can only balance when those lanes are zero.
///
/// This is a property of the MACHINE, not of the emitter, so it is checked the
/// only way it can be: the same program, the same valid opening, one arena word
/// with a nonzero upper lane. The control is the same word with a zero one.
///
/// ⚠ What this actually watches is the EXECUTOR's half — `Pack` reads its lanes
/// through `read_base` and refuses a word with a nonzero upper lane
/// (`executor.rs:871`), which is why the variant is named here rather than
/// settling for `is_err`: a refusal for some other reason would be a refusal
/// this test could not tell apart from the one it exists for. The half that
/// carries the SOUNDNESS is the AIR's, and no test reaches it — it is the
/// `base_token` on `Pack`'s receive, read at `chips.rs:1934-1937`.
#[test]
fn the_leaf_pins_a_hinted_base_value_to_its_low_lane() {
    let shape = &SHAPES[1];
    let block = 1usize << shape.log_folding;
    let depth = shape.log_domain - shape.log_folding;
    let index = 2usize;

    let (commitment, _) = base_commitment(shape, 0xC3);
    let opening = commitment.open(index).expect("the block opens");
    let program = opening_program(block, false, depth);
    let siblings = siblings_of(&opening);
    let root = commitment_to_digest(&commitment.root());

    let honest: Vec<LfmWord> = opening.values.iter().copied().map(base_word).collect();
    assert!(
        execute(
            &program,
            &[opening_arena(&honest, &siblings, root, index)],
            &crate::hash_pin::BLOCK_HASHER
        )
        .is_ok(),
        "the control must execute, or the refusal below is a refusal of nothing"
    );

    let mut smuggled = honest.clone();
    smuggled[0][1] = FE::one();
    let refusal = execute(
        &program,
        &[opening_arena(&smuggled, &siblings, root, index)],
        &crate::hash_pin::BLOCK_HASHER,
    )
    .expect_err(
        "a base opening value with a nonzero lane 1 must have no execution: the leaf's \
         Pack is what pins it, and the fold lifts these values through Felt::as_ext",
    );
    println!("smuggled lane 1 refused with: {refusal:?}");
    assert!(
        matches!(refusal, super::executor::LfmExecError::NotBaseWord(_)),
        "the refusal must be the base-word check at the leaf's Pack, not some other \
         failure that happens to also stop the program: got {refusal:?}"
    );
}
