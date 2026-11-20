//! A probe, test-only: one W-LFM program at a recursion level's size, made of
//! what a BLAKE3 level-1 program hashes — joint leaves of 24 felt blocks from
//! a midstate and 19-parent Merkle paths — with extension-field work beside
//! them, proved by the level driver's prover ([`WlfmProving::Streamed`])
//! under either family. It prices the host seams a level pays (emit, the
//! prepared build, execute, fill) and the card's prove a compression, before
//! the in-guest BLAKE3 verifier exists to emit a real level.
//!
//! Words (the card test): `RTP_LOGN` (compressions, log2, comma list; 16 by
//! default), `RTP_FAMILY` (`b3` or `p1`, comma list; `b3`), `RTP_XALU` (field
//! rows a compression; 15, about the 200 non-hash cells a compression a
//! verified group spends), `RTP_RUNS` (timed runs after a warm-up; 1),
//! `RTP_PROVE` (0: execute and fill only, no build or prove — the host seams
//! alone, for a host without a card; 1 by default), `RTP_PREP_HOST` (1: the
//! program's prepared stacks committed on the host, as a verifier commits
//! them, so the card holds none of them through the prove; 0 by default: on
//! the card).
//!
//! Each run prints its fill's split (the fixed chips' traces, the lean
//! tables') and, when it proves, the prove's split (host preparation, phase A,
//! phase B) and the card ledger's budget and promise at its start; the last
//! run also prints one `RTP GROUP` line a group (its stamps: uploads, commit,
//! argue, codewords, openings, and the ledger's promises in GiB).

use std::time::Instant;

use crate::tables::types::FE;

use super::builder::{B3Block, B3Cv, Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::hash::HasherKind;
use super::wlfm::{
    WlfmFamily, WlfmLevers, WlfmProving, WlfmRate, build_wlfm_under, execute_and_fill_unmerged,
    prove_wlfm_traces_as, verify_wlfm,
};
use super::word::LfmWord;

const HASHER: HasherKind = HasherKind::Poseidon1W16;
/// Compressions a query: a joint leaf of three 64-value blocks (24 felt
/// blocks from the midstate) and a path of 19 parents.
const LEAF_BLOCKS: usize = 24;
const PATH: usize = 19;
const PER_QUERY: usize = LEAF_BLOCKS + PATH;

/// A deterministic word: four felts, or four `u32` lanes.
fn word(seed: u64, lanes_u32: bool) -> LfmWord {
    core::array::from_fn(|k| {
        let x = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((k as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        if lanes_u32 {
            FE::from(x >> 32)
        } else {
            FE::from(x % 0xFFFF_FFFF_0000_0001)
        }
    })
}

/// The program: `queries` queries, each a joint leaf (its values hinted, two
/// felt words a block) and a path (each sibling hinted as two lane words),
/// and `xalu` extension multiply-adds a compression over hinted extension
/// words. Returns the program and its arena.
pub(crate) fn level_program(queries: usize, xalu: usize) -> (LfmProgram, Vec<Vec<LfmWord>>) {
    let mut b = LfmBuilder::new();
    let felt_words = queries * LEAF_BLOCKS * 2;
    let sib_words = queries * PATH * 2;
    let ext_words = 4;
    let arena = b.declare_arena((felt_words + sib_words + ext_words) as u32);
    let mut at = 0u32;
    let mut words: Vec<LfmWord> = Vec::with_capacity(felt_words + sib_words + ext_words);
    let mut hint = |b: &mut LfmBuilder, w: LfmWord, words: &mut Vec<LfmWord>| -> Cell {
        words.push(w);
        let c = b.hint_word(arena, at);
        at += 1;
        c
    };
    let mid = crypto::merkle_tree::backends::blake3_joint::joint_leaf_midstate(3, 64, 8);
    let mid_words: [Cell; 2] = core::array::from_fn(|w| {
        let lanes: LfmWord = core::array::from_fn(|k| FE::from(u64::from(mid[4 * w + k])));
        b.digest_const(lanes).as_cell()
    });
    let exts: Vec<super::builder::Ext> = (0..ext_words)
        .map(|i| {
            let mut w = word(9_000 + i as u64, false);
            w[3] = FE::zero();
            hint(&mut b, w, &mut words).as_ext()
        })
        .collect();
    let mut acc = exts[0];
    let mut digests = Vec::with_capacity(queries);
    for q in 0..queries {
        let mut cv = B3Cv::Words(mid_words);
        let mut cur = None;
        for blk in 0..LEAF_BLOCKS {
            let f: [Cell; 2] = core::array::from_fn(|w| {
                let seed = ((q * LEAF_BLOCKS + blk) * 2 + w) as u64;
                hint(&mut b, word(seed, false), &mut words)
            });
            let flags = if blk + 1 == LEAF_BLOCKS { 2 | 8 } else { 0 };
            let d = b.b3_compress(cv, B3Block::Felts(f), 64, flags);
            cv = B3Cv::Words(d);
            cur = Some(d);
            for _ in 0..xalu {
                acc = b.emul_add(acc, exts[1], exts[2]);
            }
        }
        let mut node = cur.expect("a leaf");
        for lvl in 0..PATH {
            let s: [Cell; 2] = core::array::from_fn(|w| {
                let seed = 1_000_000 + ((q * PATH + lvl) * 2 + w) as u64;
                hint(&mut b, word(seed, true), &mut words)
            });
            let block = if (q >> (lvl % 8)) & 1 == 0 {
                [node[0], node[1], s[0], s[1]]
            } else {
                [s[0], s[1], node[0], node[1]]
            };
            node = b.b3_compress(B3Cv::Iv, B3Block::Lanes(block), 64, 1 | 2 | 8);
            for _ in 0..xalu {
                acc = b.emul_add(acc, exts[1], exts[3]);
            }
        }
        digests.push(node);
    }
    for d in digests.iter().take(4) {
        b.public(d[0]);
    }
    b.public(acc.as_cell());
    (compile(b.finish()), vec![words])
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// The lean level probe at laptop size: the program proves and verifies under
/// both families through the streamed prover.
#[test]
fn the_level_probe_proves_at_a_small_size() {
    let _fast = super::wlfm::test_format::set(Some(2));
    let (program, arenas) = level_program(2, 1);
    assert_eq!(program.groups.b3l.real_rows, 2 * PER_QUERY);
    for family in [WlfmFamily::P1, WlfmFamily::B3] {
        let build = build_wlfm_under(
            &program,
            HASHER,
            WlfmRate::Two,
            WlfmLevers::with_family(family),
            false,
        )
        .expect("builds");
        let filled = execute_and_fill_unmerged(&program, &arenas, HASHER).expect("executes");
        let proof =
            prove_wlfm_traces_as(&build, filled, true, WlfmProving::Streamed).expect("proves");
        verify_wlfm(&build.artifacts, &proof, &proof.public_words).expect("verifies");
    }
}

/// ★ The card probe: emit, build, execute, fill, prove (streamed) and verify
/// one level-sized program per size and family, and print one `RTP RESULT`
/// line a run.
#[test]
#[ignore = "a level-sized W-LFM prove; box tier (card)"]
fn rtp_a_level_sized_lean_program_on_the_card() {
    let xalu: usize = env_list("RTP_XALU", "15")[0].parse().expect("RTP_XALU");
    let runs: usize = env_list("RTP_RUNS", "1")[0].parse().expect("RTP_RUNS");
    let prove_it = env_list("RTP_PROVE", "1")[0] != "0";
    let prep_host = env_list("RTP_PREP_HOST", "0")[0] == "1";
    for logn in env_list("RTP_LOGN", "16") {
        let logn: u32 = logn.parse().expect("RTP_LOGN");
        let queries = (1usize << logn).div_ceil(PER_QUERY);
        let t = Instant::now();
        let (program, arenas) = level_program(queries, xalu);
        let emit = t.elapsed().as_secs_f64();
        let n = program.groups.b3l.real_rows;
        for family in env_list("RTP_FAMILY", "b3") {
            let family = match family.as_str() {
                "b3" => WlfmFamily::B3,
                "p1" => WlfmFamily::P1,
                other => panic!("RTP_FAMILY {other}: b3 or p1"),
            };
            if !prove_it {
                for run in 0..=runs {
                    let t = Instant::now();
                    let execution = super::executor::execute(&program, &arenas, &HASHER)
                        .expect("the program executes");
                    let execute = t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    let filled = super::proof::lfm_fill_executed_under(
                        &program,
                        execution,
                        HASHER,
                        execute,
                        &Vec::new(),
                    );
                    let fill = t.elapsed().as_secs_f64();
                    drop(filled);
                    let (chips, lean) = super::proof::take_fill_split().unwrap_or_default();
                    println!(
                        "RTP HOST logn {logn} · run {run} · compressions {n} · instructions {} · \
                         execute {execute:.3}s · fill {fill:.3}s (chips {chips:.3}s · lean \
                         {lean:.3}s) · us a compression: execute {:.3} · fill {:.3} (chips {:.3} \
                         · lean {:.3})",
                        program.instrs.len(),
                        1e6 * execute / n as f64,
                        1e6 * fill / n as f64,
                        1e6 * chips / n as f64,
                        1e6 * lean / n as f64,
                    );
                }
                break;
            }
            let t = Instant::now();
            let build = build_wlfm_under(
                &program,
                HASHER,
                WlfmRate::Two,
                WlfmLevers::with_family(family),
                prep_host,
            )
            .expect("builds");
            let built = t.elapsed().as_secs_f64();
            let a = &build.artifacts;
            let airs = a.airs();
            let shapes = a.shapes(&airs);
            let cells: usize = shapes.iter().map(|&(w, v)| w << v).sum();
            let tables: Vec<String> = airs
                .air_refs()
                .iter()
                .zip(&shapes)
                .filter(|(_, (_, v))| *v > 10)
                .map(|(air, &(w, v))| format!("{} {w}x2^{v}", air.name()))
                .collect();
            println!(
                "RTP TABLES logn {logn} · family {family:?} · prepared on the {} · {} · groups {:?}",
                if prep_host { "host" } else { "card" },
                tables.join(" · "),
                a.sizes
            );
            for run in 0..=runs {
                let t = Instant::now();
                let execution = super::executor::execute(&program, &arenas, &HASHER)
                    .expect("the program executes");
                let execute = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let filled = super::proof::lfm_fill_executed_under(
                    &program,
                    execution,
                    HASHER,
                    execute,
                    &Vec::new(),
                );
                let fill = t.elapsed().as_secs_f64();
                let (chips, lean) = super::proof::take_fill_split().unwrap_or_default();
                let t = Instant::now();
                let proof = prove_wlfm_traces_as(&build, filled, true, WlfmProving::Streamed)
                    .expect("proves");
                let prove = t.elapsed().as_secs_f64();
                let st = super::wlfm::take_wlfm_stamps().unwrap_or_default();
                let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                println!(
                    "RTP SPLIT logn {logn} · family {family:?} · run {run} · fill chips \
                     {chips:.3}s · fill lean {lean:.3}s · prove prep {:.3}s · phase A {:.3}s · \
                     phase B {:.3}s · card budget {:.2} GiB · promised at the start {:.2} GiB · \
                     prep parts: merge {:.3}s airs {:.3}s pairs {:.3}s check {:.3}s layouts \
                     {:.3}s tables {:.3}s",
                    st.prep,
                    st.phase_a,
                    st.phase_b,
                    gib(st.card_budget),
                    gib(st.card_reserved_at_start),
                    st.prep_parts[0],
                    st.prep_parts[1],
                    st.prep_parts[2],
                    st.prep_parts[3],
                    st.prep_parts[4],
                    st.prep_parts[5],
                );
                if run == runs {
                    for (g, s) in st.groups.iter().enumerate() {
                        println!(
                            "RTP GROUP logn {logn} · family {family:?} · group {g} · tables {} · \
                             polys {} · cells {} · up A {:.3}s (paid {:.3}) · commit {:.3}s · \
                             pack {:.3}s · ood {:.3}s · up B {:.3}s · argue {:.3}s · encode \
                             {:.3}s · open {:.3}s (paid {:.3}) · turn {:.3}-{:.3}s · overlapped \
                             {} · GiB promised: commit {:.2}/{:.2} · argue {:.2}/{:.2} · open \
                             {:.2}",
                            s.tables,
                            s.polys,
                            s.cells,
                            s.upload_a,
                            s.upload_paid,
                            s.commit,
                            s.pack,
                            s.ood,
                            s.upload_b,
                            s.argue,
                            s.encode,
                            s.open,
                            s.open_paid,
                            s.start_b,
                            s.end_b,
                            s.overlapped,
                            gib(s.commit_base),
                            gib(s.commit_reserved),
                            gib(s.argue_base),
                            gib(s.argue_reserved),
                            gib(s.open_reserved),
                        );
                    }
                }
                let t = Instant::now();
                verify_wlfm(a, &proof, &proof.public_words).expect("verifies");
                let verify = t.elapsed().as_secs_f64();
                println!(
                    "RTP RESULT logn {logn} · family {family:?} · run {run}{} · compressions {n} · \
                     instructions {} · groups {} · committed cells {cells} · emit {emit:.3}s · \
                     build {built:.3}s · execute {execute:.3}s · fill {fill:.3}s · prove \
                     {prove:.3}s · verify {verify:.3}s · prove us a compression {:.3} · fill us a \
                     compression {:.3} · execute us a compression {:.3}",
                    if run == 0 { " (warm-up)" } else { "" },
                    program.instrs.len(),
                    a.sizes.len(),
                    1e6 * prove / n as f64,
                    1e6 * fill / n as f64,
                    1e6 * execute / n as f64,
                );
            }
        }
    }
    println!("RTP DONE");
}
