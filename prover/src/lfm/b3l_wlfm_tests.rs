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
    let _wide = super::b3l_tables::test_hold::set(super::b3l_tables::B3lHold::Wide);
    let corrupt = || {
        let mut filled =
            execute_and_fill_unmerged(&program, &[], HASHER).expect("the program executes");
        let t = filled.b3l_mut().expect("lean tables").rounds[0]
            .wide_mut()
            .expect("held wide");
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
            assert_eq!(t.heads[0].get(k, c), *g.at(k, c), "head row {k} column {c}");
        }
    }
    for i in 0..l::ROUNDS * g.real_rows {
        let (chunk, r) = t.round_at(i).expect("a round row");
        let at = |c| chunk.get(r, c);
        assert_eq!(at(round::MU), FE::one());
        assert_eq!(at(round::TAG), FE::from(l::tag_for_row(i / l::ROUNDS)));
        assert_eq!(at(round::RND), FE::from((i % l::ROUNDS) as u64));
    }
    let _ = head::NUM_COLUMNS;
}

// =============================================================================
// The BLAKE3 family
// =============================================================================

fn build_under(program: &LfmProgram, family: super::wlfm::WlfmFamily) -> WlfmBuild {
    super::wlfm::build_wlfm_under(
        program,
        HASHER,
        WlfmRate::Two,
        super::wlfm::WlfmLevers::with_family(family),
        false,
    )
    .unwrap_or_else(|e| panic!("the {family:?} W-LFM build: {e}"))
}

/// ★ A program of lean compressions proves and verifies under the BLAKE3
/// family — commitments, transcript and grind all 7-round BLAKE3 — and is
/// another identity than the same program under Poseidon1.
#[test]
fn lean_compressions_round_trip_under_blake3() {
    use super::wlfm::WlfmFamily;
    let _fast = test_format::set(Some(2));
    let program = honest();
    let b3 = build_under(&program, WlfmFamily::B3);
    assert_eq!(b3.artifacts.family, WlfmFamily::B3);
    let proof = prove(&program, &b3);
    verify_wlfm(&b3.artifacts, &proof, &proof.public_words)
        .unwrap_or_else(|e| panic!("the BLAKE3 W-LFM proof must verify: {e}"));
    let p1 = build_under(&program, WlfmFamily::P1);
    assert_ne!(b3.artifacts.program_id, p1.artifacts.program_id);
}

/// A proof made under one family is refused against the other's artifacts,
/// both ways: the family is the verifier's constant.
#[test]
fn a_proof_under_one_family_is_refused_under_the_other() {
    use super::wlfm::WlfmFamily;
    let _fast = test_format::set(Some(2));
    let program = honest();
    let b3 = build_under(&program, WlfmFamily::B3);
    let p1 = build_under(&program, WlfmFamily::P1);
    let b3_proof = prove(&program, &b3);
    let p1_proof = prove(&program, &p1);
    assert!(verify_wlfm(&p1.artifacts, &b3_proof, &b3_proof.public_words).is_err());
    assert!(verify_wlfm(&b3.artifacts, &p1_proof, &p1_proof.public_words).is_err());
    // Under B3's artifacts with only the family relabelled P1 (identity
    // unchanged), the B3 proof is still refused: the transcript and the
    // commitments are the family's, not the label's.
    let mut relabelled = b3.artifacts.clone();
    relabelled.family = WlfmFamily::P1;
    assert!(verify_wlfm(&relabelled, &b3_proof, &b3_proof.public_words).is_err());
}

/// The default family is today's: a build under `WlfmLevers::DEFAULT` and one
/// under `with_family(P1)` are the same artifacts, with today's statement tag.
#[test]
fn the_default_family_is_todays_poseidon1() {
    use super::wlfm::{WLFM_STATEMENT_TAG, WlfmFamily, WlfmLevers};
    let program = super::wlfm_tests::honest();
    let a = super::wlfm_tests::build(&program, WlfmRate::Two).artifacts;
    let b = build_under(&program, WlfmFamily::P1).artifacts;
    assert_eq!(WlfmLevers::DEFAULT.family, WlfmFamily::P1);
    assert_eq!(a.family, WlfmFamily::P1);
    assert_eq!(a.program_id, b.program_id);
    assert_eq!(WlfmFamily::P1.statement_tag(), WLFM_STATEMENT_TAG);
    assert_eq!(
        WlfmFamily::P1.hash_name(),
        <super::wlfm::WlfmHash as multilinear::whir_hash::WhirHash>::NAME
    );
}

/// The in-guest W-leg walks Poseidon1 children only: a BLAKE3 child is
/// refused at emit time, never walked under the wrong hash.
#[test]
fn the_w_leg_refuses_a_blake3_child() {
    let program = honest();
    let b3 = build_under(&program, super::wlfm::WlfmFamily::B3);
    let child = super::wlfm_leg::WlfmChild {
        artifacts: b3.artifacts,
        num_public_words: 8,
    };
    let err = super::wlfm_leg::wlfm_leg_program(&child).expect_err("refused");
    assert!(err.contains("Poseidon1"), "{err}");
}

/// The W-leg walks a Poseidon1 child that carries lean tables: the leg is
/// generic over the child's tables, the lean ones included. Executed, not
/// proved: the leg's program accepts the honest child's proof.
#[test]
fn the_w_leg_executes_over_a_poseidon1_child_with_lean_tables() {
    let _fast = test_format::set(Some(2));
    let program = honest();
    let build = build(&program);
    let proof = prove(&program, &build);
    let child = super::wlfm_leg::WlfmChild {
        artifacts: build.artifacts.clone(),
        num_public_words: proof.public_words.len(),
    };
    let leg = super::wlfm_leg::wlfm_leg_program(&child).expect("the leg emits");
    let arena = super::wlfm_leg::wlfm_leg_arena(&child, &proof).expect("the leg's arena");
    let exec = execute(&leg, &[arena], &HASHER).expect("the leg accepts the honest child");
    assert!(!exec.public_words.is_empty());
}

/// The holding and the streaming are the prover's: a lean program proved with
/// its tables held wide on the forks, narrow on the forks, and narrow streamed
/// under the base's production options, is the same proof byte for byte.
#[test]
fn wide_narrow_and_streamed_prove_the_same_bytes() {
    use super::b3l_tables::{B3lHold, test_hold};
    use super::wlfm::{WlfmProving, prove_wlfm_as};
    // Grinding off: an ungrinded proof has no nonce search to vary.
    let _fast = test_format::set(Some(2));
    let program = honest();
    let build = build(&program);
    let bytes = |hold: B3lHold, proving: WlfmProving| {
        let _hold = test_hold::set(hold);
        let proof = prove_wlfm_as(&program, &build, &[], proving).expect("proves");
        verify_wlfm(&build.artifacts, &proof, &proof.public_words).expect("verifies");
        rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("serializes")
            .to_vec()
    };
    let wide = bytes(B3lHold::Wide, WlfmProving::Held);
    assert_eq!(wide, bytes(B3lHold::Narrow, WlfmProving::Held), "narrow");
    assert_eq!(
        wide,
        bytes(B3lHold::Narrow, WlfmProving::Streamed),
        "streamed"
    );
}

/// The production fill (straight into narrow columns, blocks in parallel) is
/// the wide fill column for column, over a program whose rounds straddle
/// chunks and blocks, with lane and felt blocks, IV and chained rows.
#[test]
fn the_direct_narrow_fill_is_the_wide_fill() {
    use super::b3l_tables::{B3lHold, build_b3l_traces_held};
    let mut b = LfmBuilder::new();
    let mut cv = B3Cv::Iv;
    for i in 0..100u64 {
        let w: [Cell; 4] = core::array::from_fn(|j| {
            let lanes: LfmWord = core::array::from_fn(|k| {
                FE::from((i * 31 + j as u64 * 7 + k as u64) & 0xFFFF_FFFF)
            });
            b.digest_const(lanes).as_cell()
        });
        let block = if i % 3 == 0 {
            B3Block::Felts([w[0], w[1]])
        } else {
            B3Block::Lanes(w)
        };
        let d = if i % 5 == 0 {
            b.b3_compress_rev(cv, block, 64, (i % 16) as u8).0
        } else {
            b.b3_compress(cv, block, (i % 65) as u8, (i % 16) as u8)
        };
        cv = if i % 7 == 0 { B3Cv::Iv } else { B3Cv::Words(d) };
        if i == 99 {
            b.public(d[0]);
        }
    }
    let program = compile(b.finish());
    super::validator::validate(&program).expect("admissible");
    let exec = execute(&program, &[], &HASHER).expect("executes");
    let wide = build_b3l_traces_held(&program, &exec.records.b3l, B3lHold::Wide).expect("tables");
    let narrow =
        build_b3l_traces_held(&program, &exec.records.b3l, B3lHold::Narrow).expect("tables");
    assert!(wide.rounds.len() > 1, "several round chunks");
    let pairs = wide
        .rounds
        .iter()
        .zip(&narrow.rounds)
        .chain(wide.heads.iter().zip(&narrow.heads))
        .chain(core::iter::once((&wide.table, &narrow.table)));
    for (t, (w, n)) in pairs.enumerate() {
        assert!(
            matches!(n, super::b3l_tables::HeldTable::Narrow(_)),
            "table {t} narrow"
        );
        assert_eq!(
            (w.rows(), w.cols()),
            (n.rows(), n.cols()),
            "table {t} shape"
        );
        for c in 0..w.cols() {
            assert_eq!(w.column(c), n.column(c), "table {t} column {c}");
        }
    }
}
