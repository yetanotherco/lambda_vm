//! Laptop gates for the lean BLAKE3 tables inside a W-LFM proof
//! ([`super::b3l_tables`]): a program's `B3l` compressions proved on the real
//! `LfmMem` bus, and one named refusal per way such a proof could fail to bind
//! its compressions.

use crate::tables::types::FE;

use super::b3l_tables::{B3lShape, test_drops};
use super::blake3_lean::{HeadConstraints, RoundConstraints, head, round};
use super::builder::{B3Block, B3Cv, Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::hash::HasherKind;
use super::layout::b3l as l;
use super::wlfm::{
    WlfmBuild, WlfmProof, WlfmRate, build_wlfm, execute_and_fill_unmerged, prove_wlfm,
    prove_wlfm_traces, test_format, verify_wlfm,
};
use super::word::LfmWord;

const HASHER: HasherKind = HasherKind::Poseidon1W16;

/// The four words of `u32` lanes a 64-byte block packs into.
fn lane_words(b: &mut LfmBuilder, block: &[u8; 64]) -> [Cell; 4] {
    core::array::from_fn(|w| {
        let lanes: LfmWord = core::array::from_fn(|k| {
            let i = 16 * w + 4 * k;
            FE::from(u64::from(u32::from_le_bytes([
                block[i],
                block[i + 1],
                block[i + 2],
                block[i + 3],
            ])))
        });
        b.digest_const(lanes).as_cell()
    })
}

/// A program that hashes as the in-guest verifier will: a three-block
/// `Blake3Std` chain over lane blocks, a joint leaf of three felt blocks from
/// its midstate, a parent over the two digests with its reversed digest, and
/// every digest published. `flags_last` is the parent's flags: a forged twin
/// changes it and keeps every table's height.
fn program(flags_last: u8) -> LfmProgram {
    use crypto::hash::blake3::chain::{block_flags, block_len_of, num_blocks};
    let mut b = LfmBuilder::new();
    // 1. Blake3Std over 150 bytes: three blocks, the last partial.
    let msg: Vec<u8> = (0..150u32).map(|i| (i * 7 + 3) as u8).collect();
    let n = num_blocks(msg.len());
    let mut cv = B3Cv::Iv;
    let mut chain = None;
    for i in 0..n {
        let mut block = [0u8; 64];
        let end = ((i + 1) * 64).min(msg.len());
        block[..end - i * 64].copy_from_slice(&msg[i * 64..end]);
        let words = lane_words(&mut b, &block);
        let d = b.b3_compress(
            cv,
            B3Block::Lanes(words),
            block_len_of(i, msg.len()) as u8,
            block_flags(i, n) as u8,
        );
        cv = B3Cv::Words(d);
        chain = Some(d);
    }
    let chain = chain.expect("three blocks");
    // 2. A joint leaf of three polynomials' blocks of eight felts.
    let mid = crypto::merkle_tree::backends::blake3_joint::joint_leaf_midstate(3, 8, 8);
    let mid_words: [Cell; 2] = core::array::from_fn(|w| {
        let lanes: LfmWord = core::array::from_fn(|k| FE::from(u64::from(mid[4 * w + k])));
        b.digest_const(lanes).as_cell()
    });
    let len = crypto::merkle_tree::backends::blake3_joint::JOINT_LEAF_PREFIX_LEN + 3 * 8 * 8;
    let total = num_blocks(len);
    let mut cv = B3Cv::Words(mid_words);
    let mut leaf = None;
    for i in 1..total {
        let felts: [Cell; 2] = core::array::from_fn(|w| {
            let v: LfmWord =
                core::array::from_fn(|k| FE::from((i as u64) << 40 | (w as u64) << 8 | k as u64));
            b.digest_const(v).as_cell()
        });
        let d = b.b3_compress(
            cv,
            B3Block::Felts(felts),
            block_len_of(i, len) as u8,
            block_flags(i, total) as u8,
        );
        cv = B3Cv::Words(d);
        leaf = Some(d);
    }
    let leaf = leaf.expect("three blocks");
    // 3. The parent of the two, with its reversed digest.
    let (parent, rev) = b.b3_compress_rev(
        B3Cv::Iv,
        B3Block::Lanes([chain[0], chain[1], leaf[0], leaf[1]]),
        64,
        flags_last,
    );
    for c in chain.iter().chain(&leaf).chain(&parent).chain(&rev) {
        b.public(*c);
    }
    compile(b.finish())
}

fn honest() -> LfmProgram {
    program(crypto::hash::blake3::chain::FLAGS_ONE_BLOCK as u8)
}

fn build(program: &LfmProgram) -> WlfmBuild {
    build_wlfm(program, HASHER, WlfmRate::Two).unwrap_or_else(|e| panic!("the W-LFM build: {e}"))
}

fn prove(program: &LfmProgram, build: &WlfmBuild) -> WlfmProof {
    prove_wlfm(program, build, &[]).unwrap_or_else(|e| panic!("the W-LFM prove: {e}"))
}

/// ★ THE ANCHOR: a program of lean compressions proves and verifies through
/// W-LFM, its lean tables after the LFM chips', and its published digests
/// are standard BLAKE3's.
#[test]
fn lean_compressions_round_trip_through_wlfm() {
    let _fast = test_format::set(Some(2));
    let program = honest();
    super::validator::validate(&program).expect("admissible");
    let build = build(&program);
    let a = &build.artifacts;
    assert_eq!(a.b3l, B3lShape::of(7), "seven compressions");
    let airs = a.airs();
    let refs = airs.air_refs();
    let lfm = airs.lfm.air_refs().len();
    assert_eq!(refs.len(), lfm + a.b3l.num_tables());
    assert_eq!(refs[lfm].name(), super::b3l_tables::ROUND_NAME);
    assert_eq!(
        refs.last().map(|r| r.name()),
        Some(super::b3l_tables::TABLE_NAME)
    );
    let proof = prove(&program, &build);
    verify_wlfm(a, &proof, &proof.public_words)
        .unwrap_or_else(|e| panic!("the honest lean proof must verify: {e}"));
    // The published parent is blake3 over the two digests' bytes.
    let digest = |words: &[(u32, LfmWord)]| -> Vec<u8> {
        words
            .iter()
            .flat_map(|(_, w)| w.iter())
            .flat_map(|v| super::b3l::lane_u32(v).expect("a u32 lane").to_le_bytes())
            .collect()
    };
    let w = &proof.public_words;
    let (chain, leaf, parent, rev) = (
        digest(&w[0..2]),
        digest(&w[2..4]),
        digest(&w[4..6]),
        digest(&w[6..8]),
    );
    let mut both = chain.clone();
    both.extend_from_slice(&leaf);
    assert_eq!(parent, blake3::hash(&both).as_bytes().to_vec());
    let mut back = parent.clone();
    back.reverse();
    assert_eq!(rev, back);
    let msg: Vec<u8> = (0..150u32).map(|i| (i * 7 + 3) as u8).collect();
    assert_eq!(chain, blake3::hash(&msg).as_bytes().to_vec());
}

/// A program with no compression keeps today's AIR set: no lean table, not
/// even the byte table, and the same tables as before.
#[test]
fn a_program_with_no_compression_has_no_lean_table() {
    let _fast = test_format::set(Some(2));
    let program = super::wlfm_tests::honest();
    let build = super::wlfm_tests::build(&program, WlfmRate::Two);
    let a = &build.artifacts;
    assert!(a.b3l.is_empty());
    let airs = a.airs();
    assert_eq!(airs.air_refs().len(), airs.lfm.air_refs().len());
    assert_eq!(a.table_num_vars.len(), airs.lfm.air_refs().len());
}

/// A tampered published digest is refused: the head's digest sends no longer
/// balance the claimed words' reads.
#[test]
fn a_tampered_digest_word_is_refused() {
    let _fast = test_format::set(Some(2));
    let program = honest();
    let build = build(&program);
    let proof = prove(&program, &build);
    let mut claimed = proof.public_words.clone();
    claimed[4].1[0] += FE::one();
    assert!(verify_wlfm(&build.artifacts, &proof, &claimed).is_err());
}

/// ★ A forged instruction column: the same program with the parent's flags
/// changed, proved against the honest build with the prover's prefix check
/// off (the forger's prover). Its round rows chain, its digests balance — it
/// is a valid proof of ANOTHER program — and the head's prepared opening is
/// what refuses it.
#[test]
fn a_forged_head_instruction_column_is_refused_by_the_prepared_opening() {
    let _fast = test_format::set(Some(2));
    let honest = honest();
    let forged = program(crypto::hash::blake3::chain::FLAGS_ONE_BLOCK as u8 ^ 0x10);
    let build = build(&honest);
    let filled =
        execute_and_fill_unmerged(&forged, &[], HASHER).expect("the forged program executes");
    let proof = prove_wlfm_traces(&build, filled, false).expect("the forger's prover");
    let err = verify_wlfm(&build.artifacts, &proof, &proof.public_words)
        .expect_err("a forged flags column must be refused");
    assert!(!err.is_empty());
}

/// A round row's carry witness changed after the fill: the add3 identity
/// refuses it; the same proof under the dropped add3 constraint is not
/// refused by the constraint batch.
#[test]
fn a_corrupted_round_witness_is_refused() {
    let _fast = test_format::set(Some(2));
    let program = honest();
    let build = build(&program);
    let corrupt = || {
        let mut filled =
            execute_and_fill_unmerged(&program, &[], HASHER).expect("the program executes");
        let t = &mut filled.b3l_mut().expect("lean tables").rounds[0];
        // Round 0 of compression 0, G 0's a' low byte, one up: its add3 sum is
        // then off by one.
        let col = round::g_base(0) + round::A1;
        let v = *t.main_table.get(0, col);
        t.main_table.set(0, col, v + FE::one());
        filled
    };
    let proof = prove_wlfm_traces(&build, corrupt(), true).expect("the prover does not check");
    assert!(verify_wlfm(&build.artifacts, &proof, &proof.public_words).is_err());
}

/// ★ MUTATION: the head's felt canonicity is load-bearing. A felt block whose
/// bytes encode `v + p` (the same field element, another byte string, so
/// another digest) is refused; with the canonicity triples dropped on both
/// sides it is accepted.
#[test]
fn a_non_canonical_felt_encoding_needs_the_canonicity_check() {
    let _fast = test_format::set(Some(2));
    // One felt compression whose digest nobody reads: the forgery moves only
    // the head's bytes and the rows after them.
    let mut b = LfmBuilder::new();
    let v = 5u64;
    let w0 = b.digest_const([FE::from(v); 4]).as_cell();
    let w1 = b.digest_const([FE::from(v + 1); 4]).as_cell();
    b.b3_compress(B3Cv::Iv, B3Block::Felts([w0, w1]), 64, 11);
    let program = compile(b.finish());
    let forged_fill = || {
        let mut exec = execute(&program, &[], &HASHER).expect("executes");
        // Felt 0 is `v`; write it as the eight big-endian bytes of `v + p`.
        let p = 0xFFFF_FFFF_0000_0001u64;
        exec.records.b3l[0].block[..8].copy_from_slice(&(v + p).to_be_bytes());
        super::proof::lfm_fill_executed_under(&program, exec, HASHER, 0.0, &Vec::new())
    };
    let run = |drop: bool| {
        let _drops = test_drops::set(
            RoundConstraints::default(),
            HeadConstraints {
                drop_canonicity: drop,
            },
        );
        let build = build(&program);
        let proof =
            prove_wlfm_traces(&build, forged_fill(), true).expect("the prover does not check");
        verify_wlfm(&build.artifacts, &proof, &proof.public_words)
    };
    assert!(run(false).is_err(), "the canonicity check refuses v + p");
    run(true).unwrap_or_else(|e| panic!("with the check dropped the forgery is accepted: {e}"));
}

/// The head's instruction group in the trace is the program's, column for
/// column, and the round rows carry `(1, k + 1, r)`.
#[test]
fn the_lean_prefixes_are_the_programs() {
    let program = honest();
    let mut filled = execute_and_fill_unmerged(&program, &[], HASHER).expect("executes");
    let t = filled.b3l_mut().expect("lean tables");
    let g = &program.groups.b3l;
    for k in 0..g.real_rows {
        for c in 0..l::PREP_WIDTH {
            assert_eq!(
                t.heads[0].main_table.get(k, c),
                g.at(k, c),
                "head row {k} column {c}"
            );
        }
    }
    for i in 0..l::ROUNDS * g.real_rows {
        let (chunk, r) = t.round_at(i).expect("a round row");
        let at = |c| *chunk.main_table.get(r, c);
        assert_eq!(at(round::MU), FE::one());
        assert_eq!(at(round::TAG), FE::from(l::tag_for_row(i / l::ROUNDS)));
        assert_eq!(at(round::RND), FE::from((i % l::ROUNDS) as u64));
    }
    let _ = head::NUM_COLUMNS;
}
