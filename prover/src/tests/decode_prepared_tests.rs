//! ★★ DECODE's out-of-band commitment: what it is a function of, and where it
//! is allowed to sit.
//!
//! ```text
//! cargo test -p lambda-vm-prover --lib decode_prepared
//! ```
//!
//! # WHERE IT IS WIRED, kept current rather than left as a plan
//!
//! ⛔ This section said "NOT YET WIRED — every `multi_prove` and `multi_verify`
//! call site still passes `None`" for one commit longer than it was true. It is
//! wired: `prove_epoch` and `verify_epoch_bookend` both build the commitment
//! through `decode_prepared_for` and pass it down, so a proof carries the
//! prepared opening and the per-epoch MLE evaluation of the five columns is no
//! longer paid. What this file pins is the derivation and the index rule, which
//! the wiring depends on; a statement about what calls it belongs beside the
//! callers, and is written here only because the branch has twice been bitten by
//! a feature that was not wired at all rather than wired wrongly.
//!
//! # No ELF on disk
//!
//! These drive the commitment from an instruction map built here, through
//! `decode::preprocessed_columns` — the same function the ELF path feeds. A
//! build product missing from a worktree would otherwise turn these into a
//! silent skip, which is a pass for the wrong reason.

use executor::vm::instruction::decoding::{ArithOp, Instruction};
use executor::vm::memory::U64HashMap;
use multilinear::whir_chain::{ChainConfig, GrindBits};
use multilinear::whir_hash::{KeccakWhir, RpxWhir, WhirHash};
use stark::traits::AIR;

use crate::multilinear_continuation::{DecodePrepared, decode_prepared_from_columns};
use crate::tables::decode::preprocessed_columns;
use crate::test_utils::{E, F};

fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
        format: multilinear::whir_chain::ChainFormat::DEFAULT,
    }
}

/// A program: `n` distinct instructions at consecutive word addresses.
fn program(n: u64, imm: i32) -> U64HashMap<Instruction> {
    let mut instrs: U64HashMap<Instruction> = U64HashMap::default();
    for i in 0..n {
        instrs.insert(
            0x1000 + 4 * i,
            Instruction::ArithImm {
                dst: 3,
                src: 1,
                imm: imm + i as i32,
                op: ArithOp::Add,
            },
        );
    }
    instrs
}

fn prepared<H: WhirHash>(instrs: &U64HashMap<Instruction>, digest: u8) -> DecodePrepared<H> {
    decode_prepared_from_columns::<H>([digest; 32], preprocessed_columns(instrs), &config())
        .expect("prepared commitment")
}

/// ★ The commitment is a function of the INSTRUCTION TABLE and of nothing else.
///
/// Built twice from one program, in one process, the roots must agree. This is
/// not a formality on this branch: before the pc sort landed (`a850dd29`) the
/// DECODE row order was a function of a hash map's iteration order as well as of
/// the program, and a root pinned per ELF sha would have been pinned to a value
/// that moved between two runs of one binary. The instrument that measures a
/// commitment has to reproduce within one tree before it can be quoted across
/// two — which is the mistake this branch already made once, with a digest that
/// agreed by luck.
#[test]
fn the_decode_commitment_is_a_function_of_the_program_alone() {
    let instrs = program(200, 7);
    let a = prepared::<KeccakWhir>(&instrs, 1);
    let b = prepared::<KeccakWhir>(&instrs, 1);
    let c = prepared::<KeccakWhir>(&instrs, 1);
    assert_eq!(a.roots, b.roots, "two derivations of one program disagree");
    assert_eq!(
        b.roots, c.roots,
        "three derivations of one program disagree"
    );
    assert!(!a.roots.is_empty(), "the group committed to nothing");
}

/// ★ ...and it MOVES when the program does.
///
/// The sensitivity half. Without it the equality above is satisfied by a
/// derivation that returns a constant, which is a root that binds no program.
/// Both a changed instruction and a changed length must move it.
#[test]
fn a_different_program_commits_to_a_different_root() {
    let same_length = (
        prepared::<KeccakWhir>(&program(200, 7), 1).roots,
        prepared::<KeccakWhir>(&program(200, 8), 1).roots,
    );
    assert_ne!(
        same_length.0, same_length.1,
        "one instruction's immediate changed and the root did not: the \
         commitment is not binding the instruction table"
    );

    let different_length = prepared::<KeccakWhir>(&program(201, 7), 1).roots;
    assert_ne!(
        same_length.0, different_length,
        "the program grew by an instruction and the root did not move"
    );
}

/// ★ The digest travels WITH the roots, and is the caller's, not the columns'.
///
/// The pair is what the field machine pins as program text, so a pair that could
/// be assembled from two different ELFs is the defect to prevent. This says the
/// two halves are carried together; that they describe one ELF is
/// [`crate::multilinear_continuation::decode_prepared_for`]'s one-call
/// structure, not something a test can observe from the outside.
#[test]
fn the_digest_is_carried_beside_the_roots() {
    let instrs = program(64, 3);
    let a = prepared::<KeccakWhir>(&instrs, 0xAA);
    let b = prepared::<KeccakWhir>(&instrs, 0xBB);
    assert_eq!(a.elf_digest, [0xAAu8; 32]);
    assert_ne!(a.elf_digest, b.elf_digest);
    assert_eq!(
        a.roots, b.roots,
        "the roots are the program's; only the digest is the caller's"
    );
}

/// The two configurations commit the same columns to DIFFERENT roots, which is
/// the seam's whole claim restated for this commitment: a hash swap changes the
/// digest, not the shape.
#[test]
fn the_two_hashes_commit_the_same_columns_to_different_roots() {
    let instrs = program(128, 5);
    let keccak = prepared::<KeccakWhir>(&instrs, 1);
    let rpx = prepared::<RpxWhir>(&instrs, 1);
    assert_ne!(keccak.roots, rpx.roots);
    assert_eq!(
        keccak.roots.len(),
        rpx.roots.len(),
        "the two hashes disagree on how many polynomials the group has"
    );
    assert_eq!(
        keccak.settled_at(0).len(),
        rpx.settled_at(0).len(),
        "the two hashes disagree on how many columns the group covers"
    );
}

/// ★ THE SHAPE, asserted where it is a fact about the program.
///
/// The five columns of `rows` stack into one polynomial of
/// `ceil_log2(5 * rows)` variables, and its codeword is `n_stack + log_blowup`
/// — the residency budget the wiring has to reserve. One power of two either way
/// doubles it, so it is derived from the column count and the row count rather
/// than quoted.
///
/// ⚠ This is why (e-host) is not a runtime assertion in `multi_verify`: the
/// verifier DERIVES this layout from the ELF and never reads a shape from a
/// proof, so an assertion there would compare a caller's numbers against the
/// same caller's numbers. The shape is a fact about the program, and here is
/// where it can fail.
#[test]
fn the_group_shape_is_the_one_the_program_implies() {
    // 1,024 instructions plus the CPU padding entry rounds to 2,048 rows.
    let instrs = program(1_024, 7);
    let columns = preprocessed_columns(&instrs);
    assert_eq!(columns.len(), 5, "DECODE's preprocessed column count");
    let rows = columns[0].len();
    assert_eq!(rows, 2_048);

    let p = prepared::<KeccakWhir>(&instrs, 1);
    let settled = p.settled_at(0);
    let check = p.check(&settled);
    assert_eq!(check.at.len(), 5);
    assert_eq!(
        check.layout.num_polys(),
        1,
        "the five columns must stack into ONE polynomial, or the group carries \
         more than one root and everything downstream indexes wrongly"
    );
    let cells = 5 * rows;
    assert_eq!(
        check.layout.n_stack(),
        cells.next_power_of_two().trailing_zeros() as usize,
        "n_stack is not ceil_log2(5 * rows)"
    );
    assert_eq!(p.roots.len(), check.layout.num_polys());
}

/// ★ DECODE's index is found by NAME, and "exactly one" is the assertion.
///
/// The opening binds the pinned columns to one table's reduced point. A prover
/// who could aim them at a different table's point would be settling them
/// against challenges they were never bound to, so the index is never inferred
/// from a position — and a table set with two DECODEs, or none, is a layout
/// nobody meant to build.
#[test]
fn decode_is_found_by_name_and_only_once() {
    use crate::multilinear_continuation::decode_table_index;

    let opts = stark::proof::options::ProofOptions::default_test_options();
    let decode = crate::test_utils::create_decode_air(&opts);
    let eq = crate::test_utils::create_eq_air(&opts);

    type Air<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

    let one: Vec<Air<'_>> = vec![&eq, &decode, &eq];
    assert_eq!(decode_table_index(&one).expect("one DECODE"), 1);

    let none: Vec<Air<'_>> = vec![&eq, &eq];
    assert!(
        decode_table_index(&none).is_err(),
        "a table set with no DECODE was accepted"
    );

    let two: Vec<Air<'_>> = vec![&decode, &eq, &decode];
    assert!(
        decode_table_index(&two).is_err(),
        "a table set with two DECODEs was accepted, so the index it returns is \
         a choice rather than a fact"
    );
}
