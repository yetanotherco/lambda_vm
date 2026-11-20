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

pub(super) fn honest() -> LfmProgram {
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

/// The in-guest W-leg walks a BLAKE3 child under BLAKE3 only: a Poseidon1
/// builder refuses it at emit time, never walking it under the wrong hash, and
/// the leg's own builder (BLAKE3, lean rows) emits it.
#[test]
fn the_w_leg_walks_a_blake3_child_under_blake3_only() {
    let program = honest();
    let b3 = build_under(&program, super::wlfm::WlfmFamily::B3);
    let child = super::wlfm_leg::WlfmChild {
        artifacts: b3.artifacts,
        num_public_words: 8,
    };
    let mut b = super::wlfm_leg::wlfm_leg_builder(super::wlfm::WlfmFamily::P1);
    let arena = b.declare_arena(0);
    let Err(err) = super::wlfm_leg::emit_wlfm_leg(&mut b, &child, arena) else {
        panic!("a Poseidon1 builder walked a BLAKE3 child");
    };
    assert!(err.contains("B3"), "{err}");
    super::wlfm_leg::wlfm_leg_program(&child).expect("the BLAKE3 leg emits");
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
    // The round chunks as their records: on a card build the card fills them.
    assert_eq!(wide, bytes(B3lHold::Records, WlfmProving::Held), "records");
    assert_eq!(
        wide,
        bytes(B3lHold::Records, WlfmProving::Streamed),
        "records, streamed"
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
    let records =
        build_b3l_traces_held(&program, &exec.records.b3l, B3lHold::Records).expect("tables");
    assert!(
        records.rounds.iter().all(|t| t.is_records()),
        "round chunks as records"
    );
    for (t, (n, r)) in narrow.rounds.iter().zip(&records.rounds).enumerate() {
        assert_eq!(
            (n.rows(), n.cols()),
            (r.rows(), r.cols()),
            "round chunk {t} shape"
        );
        for c in 0..n.cols() {
            assert_eq!(n.column(c), r.column(c), "round chunk {t} column {c}");
        }
    }
    for (t, (n, r)) in narrow.heads.iter().zip(&records.heads).enumerate() {
        for c in 0..n.cols() {
            assert_eq!(n.column(c), r.column(c), "head chunk {t} column {c}");
        }
    }
    for c in 0..narrow.table.cols() {
        assert_eq!(
            narrow.table.column(c),
            records.table.column(c),
            "byte table column {c}"
        );
    }
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

// =============================================================================
// Cut chips
// =============================================================================

/// A program of `n` published constants and a sum over them; `last` is the
/// last constant, which a forged twin changes (every table keeps its height).
fn constants(n: u64, last: u64) -> LfmProgram {
    let mut b = LfmBuilder::new();
    let mut acc = b.felt_const(FE::from(1u64));
    for i in 0..n {
        let v = if i + 1 == n { last } else { 1_000 + i };
        let c = b.felt_const(FE::from(v));
        acc = b.add(acc, c);
    }
    b.public(acc.as_cell());
    compile(b.finish())
}

/// ★ A chip cut into chunk tables (the cap forced to 2^3 rows) proves and
/// verifies, is another identity than the uncut program, and the W-leg walks
/// it in-guest.
#[test]
fn a_cut_chip_round_trips_and_is_another_identity() {
    let _fast = test_format::set(Some(2));
    let program = constants(40, 77);
    let uncut = build(&program);
    assert!(uncut.artifacts.lfm_chunks.iter().all(|c| c.len() == 1));
    let _cut = test_format::set_chunk_log(3);
    let cut = build(&program);
    let a = &cut.artifacts;
    assert!(
        a.lfm_chunks.iter().any(|c| c.len() > 1),
        "some chip is cut at 2^3 rows: {:?}",
        a.lfm_chunks
    );
    assert_eq!(a.table_num_vars.len(), a.airs().air_refs().len());
    assert_ne!(a.program_id, uncut.artifacts.program_id);
    let proof = prove(&program, &cut);
    verify_wlfm(a, &proof, &proof.public_words).expect("the cut proof verifies");
    let child = super::wlfm_leg::WlfmChild {
        artifacts: a.clone(),
        num_public_words: proof.public_words.len(),
    };
    let leg = super::wlfm_leg::wlfm_leg_program(&child).expect("the leg emits");
    let arena = super::wlfm_leg::wlfm_leg_arena(&child, &proof).expect("the leg's arena");
    execute(&leg, &[arena], &HASHER).expect("the leg accepts the cut child");
}

/// A forged constant in a cut chip's LAST chunk, proved against the honest
/// build by the forger's prover, is refused by that chunk's prepared opening.
#[test]
fn a_forged_column_in_a_later_chunk_is_refused() {
    let _fast = test_format::set(Some(2));
    let _cut = test_format::set_chunk_log(3);
    let honest = constants(40, 77);
    let forged = constants(40, 78);
    let build = build(&honest);
    let filled = execute_and_fill_unmerged(&forged, &[], HASHER).expect("executes");
    let proof = prove_wlfm_traces(&build, filled, false).expect("the forger's prover");
    assert!(verify_wlfm(&build.artifacts, &proof, &proof.public_words).is_err());
    // The honest one, for contrast.
    let proof = prove(&honest, &build);
    verify_wlfm(&build.artifacts, &proof, &proof.public_words).expect("honest verifies");
}

/// The cut rule: a chip that fits is one table as it is; above the cap, full
/// chunks and the remainder's own power of two.
#[test]
fn the_chunk_rule_keeps_what_fits_and_cuts_the_rest() {
    use super::wlfm::chip_chunk_rows;
    assert_eq!(chip_chunk_rows(5, 8, 3), vec![8]);
    assert_eq!(chip_chunk_rows(8, 8, 3), vec![8]);
    assert_eq!(chip_chunk_rows(9, 16, 3), vec![8, 4]);
    assert_eq!(chip_chunk_rows(27, 32, 3), vec![8, 8, 8, 4]);
    assert_eq!(chip_chunk_rows(32, 32, 3), vec![8, 8, 8, 8]);
    assert_eq!(chip_chunk_rows(1 << 21, 1 << 21, 22), vec![1 << 21]);
    assert_eq!(chip_chunk_rows(27_000_000, 1 << 25, 22).len(), 7);
}

// =============================================================================
// Chips packed narrow
// =============================================================================

/// The chips packed narrow before the prove ([`super::wlfm::NARROW_CHIPS_ENV`])
/// are the prover's holding: off by default, and on, a program of lean and
/// chip tables, and one with a chip cut into chunks, prove the same bytes as
/// with the chips wide, held and streamed; a forged constant in a cut chip's
/// last chunk is still refused.
#[test]
fn chips_packed_narrow_prove_the_same_bytes() {
    use super::wlfm::{WlfmProving, narrow_chips, prove_wlfm_as, take_wlfm_stamps};
    assert!(!narrow_chips(), "the chips are wide by default");
    let _fast = test_format::set(Some(2));
    let bytes = |program: &LfmProgram, build: &WlfmBuild, on: bool, proving: WlfmProving| {
        let _narrow = test_format::set_narrow_chips(on);
        let _ = take_wlfm_stamps();
        let proof = prove_wlfm_as(program, build, &[], proving).expect("proves");
        let stamps = take_wlfm_stamps().expect("stamped");
        assert_eq!(
            stamps.narrow_tables > 0,
            on,
            "chips packed narrow iff on ({} tables)",
            stamps.narrow_tables
        );
        verify_wlfm(&build.artifacts, &proof, &proof.public_words).expect("verifies");
        rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("serializes")
            .to_vec()
    };
    let program = honest();
    let built = build(&program);
    let wide = bytes(&program, &built, false, WlfmProving::Held);
    assert_eq!(
        wide,
        bytes(&program, &built, true, WlfmProving::Held),
        "narrow"
    );
    assert_eq!(
        wide,
        bytes(&program, &built, true, WlfmProving::Streamed),
        "narrow, streamed"
    );
    let _cut = test_format::set_chunk_log(3);
    let program = constants(40, 77);
    let built = build(&program);
    assert!(built.artifacts.lfm_chunks.iter().any(|c| c.len() > 1));
    let wide = bytes(&program, &built, false, WlfmProving::Streamed);
    assert_eq!(
        wide,
        bytes(&program, &built, true, WlfmProving::Streamed),
        "a cut chip narrow"
    );
    let _narrow = test_format::set_narrow_chips(true);
    let filled = execute_and_fill_unmerged(&constants(40, 78), &[], HASHER).expect("executes");
    let forged = prove_wlfm_traces(&built, filled, false).expect("the forger's prover");
    assert!(verify_wlfm(&built.artifacts, &forged, &forged.public_words).is_err());
}

/// The chips' narrow pack is `NarrowColumns`' own of the same rows, the rows
/// past the trace zero, over words of every width.
#[test]
fn the_chips_narrow_pack_is_the_row_major_pack() {
    use super::wlfm::pack_rows_narrow;
    let width = 5;
    let real = 9_000; // past two of the pack's row blocks
    let rows = 1 << 14;
    let words: Vec<u64> = (0..real * width)
        .map(|i| {
            let (r, c) = ((i / width) as u64, i % width);
            match c {
                0 => r & 0xff,
                1 => r * 7,
                2 => r << 12,
                3 => 0xFFFF_FFFF_0000_0000 - r, // p − 1 − r: canonical, eight bytes
                _ => 0,
            }
        })
        .collect();
    let cells: Vec<FE> = words.iter().map(|&w| FE::from(w)).collect();
    let packed = pack_rows_narrow(&cells, width, rows).expect("packs");
    let mut padded: Vec<u64> = cells
        .iter()
        .map(|f| {
            use math::field::traits::IsPrimeField;
            crate::tables::types::GoldilocksField::canonical(f.value())
        })
        .collect();
    padded.resize(rows * width, 0);
    let expected =
        multilinear::narrow::NarrowColumns::pack_row_major(&padded, width).expect("packs");
    assert_eq!(packed.widths(), &[1, 2, 4, 8, 1]);
    assert_eq!(packed, expected);
    assert!(
        pack_rows_narrow(&cells[..7], width, rows).is_err(),
        "part of a row"
    );
    assert!(
        pack_rows_narrow(&cells, width, 8).is_err(),
        "more rows than the table"
    );
}

/// The W-leg's arena over `child`'s `proof` in pieces (its front, then each
/// group's words, as a child's prove has them group by group) against the
/// arena `wlfm_leg_arena` writes from the whole proof; the marked program
/// against the leg's program, its marks against where the pieces end. Returns
/// the child's group count.
fn pieces_are_the_arena(
    child: &super::wlfm_leg::WlfmChild,
    proof: &WlfmProof,
    what: &str,
) -> usize {
    use super::wlfm_leg::{
        wlfm_leg_arena, wlfm_leg_front, wlfm_leg_group, wlfm_leg_program, wlfm_leg_program_marked,
    };
    let whole = wlfm_leg_arena(child, proof).expect("the leg's arena");
    let mut pieces = vec![
        wlfm_leg_front(
            child,
            &proof.public_words,
            &proof.proof.roots,
            &proof.commit_ood,
        )
        .expect("the front"),
    ];
    let groups = child.artifacts.sizes.len();
    for g in 0..groups {
        let prepared = child
            .artifacts
            .stacks
            .iter()
            .position(|s| s.group == g)
            .map(|k| &proof.prepared[k]);
        pieces.push(
            wlfm_leg_group(
                child,
                g,
                &proof.argues[g],
                &proof.proof.columns[g],
                prepared,
            )
            .unwrap_or_else(|e| panic!("{what}: group {g}: {e}")),
        );
    }
    let joined: Vec<_> = pieces.iter().flatten().copied().collect();
    assert!(joined == whole, "{what}: the pieces are the arena");
    let (marked, marks) = wlfm_leg_program_marked(child).expect("the marked leg");
    let plain = wlfm_leg_program(child).expect("the leg");
    assert_eq!(
        marked.instrs.len(),
        plain.instrs.len(),
        "{what}: the same program"
    );
    assert_eq!(
        marked.arena_schema.lens, plain.arena_schema.lens,
        "{what}: the same arena"
    );
    let mut at = 0u32;
    let ends: Vec<u32> = pieces
        .iter()
        .map(|p| {
            at += p.len() as u32;
            at
        })
        .collect();
    assert_eq!(marks, ends, "{what}: the marks are where the pieces end");
    if groups > 1 {
        // Another group's parts write other words, or are refused.
        let swapped = wlfm_leg_group(child, 0, &proof.argues[1], &proof.proof.columns[1], None);
        assert!(
            swapped.map_or(true, |w| w != pieces[1]),
            "{what}: a group's words are its own"
        );
    }
    groups
}

/// ★ The W-leg's arena in pieces (`wlfm_leg_front`, then `wlfm_leg_group` for
/// each group, as a child's prove has them group by group) is the arena
/// `wlfm_leg_arena` writes from the whole proof, word for word: the lean
/// fixture under both families and the field-chip fixture, one of them with
/// several groups; the marked program is the leg's program, its marks where
/// the pieces end.
#[test]
fn the_legs_arena_in_pieces_is_its_arena_under_both_families() {
    use super::wlfm::WlfmFamily;
    use super::wlfm_leg::WlfmChild;
    let _fast = test_format::set(Some(2));
    // One stacked polynomial a group: every fixture spans several groups.
    let _groups = test_format::set_group_polys(1);
    let child_of = |build: &WlfmBuild, proof: &WlfmProof| WlfmChild {
        artifacts: build.artifacts.clone(),
        num_public_words: proof.public_words.len(),
    };
    let mut most = 0;
    let lean = honest();
    for family in [WlfmFamily::P1, WlfmFamily::B3] {
        let build = build_under(&lean, family);
        let proof = prove(&lean, &build);
        most = most.max(pieces_are_the_arena(
            &child_of(&build, &proof),
            &proof,
            &format!("lean {family:?}"),
        ));
    }
    let field = super::wlfm_tests::honest();
    let build = super::wlfm_tests::build(&field, WlfmRate::Two);
    let proof = super::wlfm_tests::prove(&field, &build);
    most = most.max(pieces_are_the_arena(
        &child_of(&build, &proof),
        &proof,
        "field chips",
    ));
    // The level-shaped probe program: lean compressions over hinted leaves and
    // paths, and extension multiply-adds, under both families.
    let hasher = HasherKind::Poseidon1W16;
    let (level, arenas) = super::levels_probe::level_program(4, 256);
    for family in [WlfmFamily::P1, WlfmFamily::B3] {
        let build = super::wlfm::build_wlfm_under(
            &level,
            hasher,
            WlfmRate::Two,
            super::wlfm::WlfmLevers::with_family(family),
            false,
        )
        .expect("builds");
        let filled =
            super::wlfm::execute_and_fill_unmerged(&level, &arenas, hasher).expect("executes");
        let proof = super::wlfm::prove_wlfm_traces_as(
            &build,
            filled,
            true,
            super::wlfm::WlfmProving::Streamed,
        )
        .expect("proves");
        most = most.max(pieces_are_the_arena(
            &child_of(&build, &proof),
            &proof,
            &format!("level probe {family:?}"),
        ));
    }
    assert!(most >= 2, "some fixture has several groups (most {most})");
}

/// ★ A W-LFM prove observed group by group (`prove_wlfm_traces_observed`)
/// hands out, as it opens each group, the parts from which the next level's
/// arena is written piece by piece: the front at the first group, each group's
/// words at its own; together they are the arena `wlfm_leg_arena` writes from
/// the finished proof, under both families, held and streamed. The observer
/// reads: the proof is the unobserved prove's, byte for byte.
#[test]
fn an_observed_prove_hands_out_the_next_levels_arena_group_by_group() {
    use super::wlfm::{WlfmFamily, WlfmProving};
    use super::wlfm_leg::{WlfmChild, wlfm_leg_arena, wlfm_leg_front, wlfm_leg_group};
    let _fast = test_format::set(Some(2));
    let _groups = test_format::set_group_polys(1);
    let hasher = HasherKind::Poseidon1W16;
    let (level, arenas) = super::levels_probe::level_program(4, 256);
    for family in [WlfmFamily::P1, WlfmFamily::B3] {
        let build = super::wlfm::build_wlfm_under(
            &level,
            hasher,
            WlfmRate::Two,
            super::wlfm::WlfmLevers::with_family(family),
            false,
        )
        .expect("builds");
        for proving in [WlfmProving::Streamed, WlfmProving::Held] {
            let filled = || {
                super::wlfm::execute_and_fill_unmerged(&level, &arenas, hasher).expect("executes")
            };
            let first = filled();
            let public = first.public_words().to_vec();
            let child = WlfmChild {
                artifacts: build.artifacts.clone(),
                num_public_words: public.len(),
            };
            let groups = child.artifacts.sizes.len();
            assert!(groups >= 2, "{family:?}: several groups");
            let front: std::sync::Mutex<Option<Vec<super::word::LfmWord>>> =
                std::sync::Mutex::new(None);
            let pieces: std::sync::Mutex<Vec<Option<Vec<super::word::LfmWord>>>> =
                std::sync::Mutex::new(vec![None; groups]);
            let observed =
                super::wlfm::prove_wlfm_traces_observed(&build, first, true, proving, &|opened| {
                    let mut f = front.lock().unwrap();
                    if f.is_none() {
                        *f = Some(
                            wlfm_leg_front(&child, &public, opened.roots, opened.commit_ood)
                                .expect("the front"),
                        );
                    }
                    let words = wlfm_leg_group(
                        &child,
                        opened.group,
                        opened.argue.expect("a batched argue"),
                        opened.opening,
                        opened.prepared,
                    )
                    .expect("the group's words");
                    let mut p = pieces.lock().unwrap();
                    assert!(p[opened.group].is_none(), "a group opened twice");
                    p[opened.group] = Some(words);
                })
                .expect("proves");
            let plain =
                super::wlfm::prove_wlfm_traces_as(&build, filled(), true, proving).expect("proves");
            let bytes = |p: &WlfmProof| rkyv::to_bytes::<rkyv::rancor::Error>(p).expect("bytes");
            assert!(
                bytes(&observed)[..] == bytes(&plain)[..],
                "{family:?} {proving:?}: the observer moves no byte"
            );
            let mut joined = front.into_inner().unwrap().expect("a front");
            for (g, piece) in pieces.into_inner().unwrap().into_iter().enumerate() {
                joined.extend(piece.unwrap_or_else(|| panic!("group {g} was never observed")));
            }
            assert!(
                joined == wlfm_leg_arena(&child, &observed).expect("the arena"),
                "{family:?} {proving:?}: the observed pieces are the next level's arena"
            );
        }
    }
}

/// ★ The next level streamed beside the one below's prove is the level it is
/// without the stream: the level-shaped probe proved under both families, its
/// W-leg executed over the arena's pieces as the observed prove hands them out
/// (`levels::stream_level_over_pieces` on a thread of its own, as the levels
/// driver runs it under `LAMBDA_VM_LEVELS_STREAM_NEXT`), and over the arena
/// written from the finished proof: one proof, byte for byte. A piece of the
/// wrong length, or pieces that stop short, refuse.
#[test]
fn the_next_level_streamed_beside_the_prove_is_the_same_level() {
    use super::levels::stream_level_over_pieces;
    use super::wlfm::{WlfmFamily, WlfmProving};
    use super::wlfm_leg::{
        WlfmChild, wlfm_leg_arena, wlfm_leg_front, wlfm_leg_group, wlfm_leg_program_marked,
    };
    use crate::hash_pin::{BlockFamily, ProofFamily};
    let _fast = test_format::set(Some(2));
    let _groups = test_format::set_group_polys(1);
    let hasher = HasherKind::Poseidon1W16;
    let (level, arenas) = super::levels_probe::level_program(4, 256);
    for family in [WlfmFamily::P1, WlfmFamily::B3] {
        let levers = super::wlfm::WlfmLevers::with_family(family);
        let build = super::wlfm::build_wlfm_under(&level, hasher, WlfmRate::Two, levers, false)
            .expect("builds");
        let filled =
            super::wlfm::execute_and_fill_unmerged(&level, &arenas, hasher).expect("executes");
        let public = filled.public_words().to_vec();
        let child = WlfmChild {
            artifacts: build.artifacts.clone(),
            num_public_words: public.len(),
        };
        let (leg, marks) = wlfm_leg_program_marked(&child).expect("the marked leg");
        assert!(marks.len() >= 3, "{family:?}: a front and several groups");
        let leg_hasher = leg.hasher(<BlockFamily as ProofFamily>::SOCKET);
        let (tx, rx) = std::sync::mpsc::channel();
        let t0 = std::time::Instant::now();
        let (proof, streamed) = std::thread::scope(|scope| {
            let next = scope.spawn(|| stream_level_over_pieces(&leg, &marks, leg_hasher, rx, t0));
            let front_sent = std::sync::atomic::AtomicBool::new(false);
            let proof = super::wlfm::prove_wlfm_traces_observed(
                &build,
                filled,
                true,
                WlfmProving::Streamed,
                &|opened| {
                    if !front_sent.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        let front =
                            wlfm_leg_front(&child, &public, opened.roots, opened.commit_ood)
                                .expect("the front");
                        tx.send((0, front)).expect("the stream listens");
                    }
                    let words = wlfm_leg_group(
                        &child,
                        opened.group,
                        opened.argue.expect("a batched argue"),
                        opened.opening,
                        opened.prepared,
                    )
                    .expect("the group's words");
                    tx.send((opened.group + 1, words))
                        .expect("the stream listens");
                },
            )
            .expect("proves");
            drop(tx);
            (proof, next.join().expect("the stream ran"))
        });
        let (streamed, times) = streamed.expect("the next level streams");
        assert_eq!(times.pieces, marks.len(), "{family:?}: every piece landed");
        let arena = wlfm_leg_arena(&child, &proof).expect("the arena");
        let plain = execute_and_fill_unmerged(&leg, std::slice::from_ref(&arena), leg_hasher)
            .expect("the leg executes");
        assert!(
            streamed.public_words() == plain.public_words(),
            "{family:?}: the streamed level publishes the same words"
        );
        let leg_build =
            super::wlfm::build_wlfm_under(&leg, leg_hasher, WlfmRate::Two, levers, false)
                .expect("the leg builds");
        let bytes = |p: &WlfmProof| rkyv::to_bytes::<rkyv::rancor::Error>(p).expect("bytes");
        let prove = |f| {
            super::wlfm::prove_wlfm_traces_as(&leg_build, f, true, WlfmProving::Streamed)
                .expect("the leg proves")
        };
        assert!(
            bytes(&prove(streamed))[..] == bytes(&prove(plain))[..],
            "{family:?}: the streamed level is the same proof, byte for byte"
        );
        // The refusals: a piece cut short, and pieces that stop before the last.
        let piece = |i: usize| {
            arena[marks.get(i.wrapping_sub(1)).map_or(0, |&m| m as usize)..marks[i] as usize]
                .to_vec()
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let mut short = piece(1);
        short.pop();
        tx.send((0, piece(0))).expect("sends");
        tx.send((1, short)).expect("sends");
        drop(tx);
        let err = stream_level_over_pieces(&leg, &marks, leg_hasher, rx, t0)
            .err()
            .expect("a short piece refuses");
        assert!(err.contains("words"), "{family:?}: {err}");
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send((0, piece(0))).expect("sends");
        drop(tx);
        let err = stream_level_over_pieces(&leg, &marks, leg_hasher, rx, t0)
            .err()
            .expect("stopped pieces refuse");
        assert!(err.contains("stopped"), "{family:?}: {err}");
    }
}

/// The head's block-byte pair lookups are what tie a felt block's hashed bytes
/// to the felt the memory bus reads. The same 64 columns feed the
/// little-endian message words the rounds hash and the big-endian felts the
/// bus reads; only the 32 pair lookups make them bytes. A head row whose
/// "bytes" are `b0 + 256·δ`, `b1 − δ` keeps message word 0 (so the digest)
/// and moves felt 0 by `δ·(2^64 − 2^48)`, to any value: here memory holds
/// `v_mem` while the compression runs over `v_hash`'s bytes, felt 0's
/// canonicity witnesses set over the moved bytes. The pair lookup finds no
/// row for the moved bytes, so the verifier refuses it on the bus balance;
/// with the pair lookups and their counts both dropped the same proof
/// verifies (run by hand, not kept as a switch).
#[test]
fn a_felt_block_hashed_off_its_felt_is_refused_by_the_byte_pair_lookup() {
    let _fast = test_format::set(Some(2));
    let _wide = super::b3l_tables::test_hold::set(super::b3l_tables::B3lHold::Wide);
    let v_mem = 0x0123_4567_89AB_CDEFu64;
    let v_hash = 5u64;
    let mut b = LfmBuilder::new();
    let w0 = b
        .digest_const([
            FE::from(v_mem),
            FE::from(7u64),
            FE::from(8u64),
            FE::from(9u64),
        ])
        .as_cell();
    let w1 = b.digest_const([FE::from(10u64); 4]).as_cell();
    b.b3_compress(B3Cv::Iv, B3Block::Felts([w0, w1]), 64, 11);
    let program = compile(b.finish());
    let mut exec = execute(&program, &[], &HASHER).expect("executes");
    assert_eq!(exec.records.b3l[0].block[..8], v_mem.to_be_bytes());
    exec.records.b3l[0].block[..8].copy_from_slice(&v_hash.to_be_bytes());
    let mut filled =
        super::proof::lfm_fill_executed_under(&program, exec, HASHER, 0.0, &Vec::new());
    let t = filled.b3l_mut().expect("lean tables").heads[0]
        .wide_mut()
        .expect("held wide");
    // δ·(2^64 − 2^48) = v_mem − v_hash: the felt the bus reads moves to v_mem,
    // message word 0 stays v_hash's.
    let two64 = FE::from(1u64 << 32) * FE::from(1u64 << 32);
    let delta = (FE::from(v_mem) - FE::from(v_hash))
        * (two64 - FE::from(1u64 << 48)).inv().expect("nonzero");
    let b0 = *t.main_table.get(0, head::MB);
    let b1 = *t.main_table.get(0, head::MB + 1);
    t.main_table.set(0, head::MB, b0 + FE::from(256u64) * delta);
    t.main_table.set(0, head::MB + 1, b1 - delta);
    // Felt 0's canonicity witnesses over the moved "bytes": its high half is
    // not 2^32 − 1, so Z = 0 and GINV = 1 / (2^32 − 1 − hi).
    let hi = (0..4).fold(FE::zero(), |acc, k| {
        acc + *t.main_table.get(0, head::MB + k) * FE::from(1u64 << (8 * (3 - k)))
    });
    let g = FE::from(0xFFFF_FFFFu64) - hi;
    t.main_table.set(0, head::CZ, FE::zero());
    t.main_table.set(
        0,
        head::CG,
        g.inv().expect("the moved high half is not 2^32 - 1"),
    );
    let build = build(&program);
    let proof = prove_wlfm_traces(&build, filled, true).expect("the prover does not check");
    let verdict = verify_wlfm(&build.artifacts, &proof, &proof.public_words);
    println!("FELT BLOCK HASHED OFF ITS FELT: verifier says {verdict:?}");
    assert_eq!(
        verdict,
        Err("BusImbalance".to_string()),
        "the byte-pair lookup refuses a felt block hashed off its felt, on the bus balance"
    );
}
