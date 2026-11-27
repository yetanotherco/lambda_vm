//! Laptop gates for the LFM chips held as their records
//! ([`super::chip_records`], `LAMBDA_VM_WLFM_CHIP_RECORDS`): the knob's words,
//! the records' host fill against the host's chip traces column for column,
//! proofs byte-equal across the three holdings (wide, narrow, records), and a
//! forged value or prefix word refused. On a card build the same proofs take
//! the card's fill (`chip_records_card_tests` compares it with the host's).

use math::field::traits::IsPrimeField;
use multilinear::narrow::NarrowColumns;

use crate::tables::types::{FE, FEE, GoldilocksField};

use super::builder::{Cell, LfmBuilder};
use super::chip_records::{ChipKind, LfmChipValues, bitdec_row};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::layout;
use super::wlfm::{
    CHIP_RECORDS_ENV, WlfmBuild, WlfmProving, WlfmRate, build_wlfm, chip_records,
    execute_and_fill_unmerged, prove_wlfm_as, prove_wlfm_traces, take_wlfm_stamps, test_format,
    verify_wlfm,
};
use super::wlfm_knobs::knob_word_on;
use super::wlfm_tests::{HASHER, arenas};

const P_MINUS_ONE: u64 = 0xFFFF_FFFF_0000_0000;

/// A program touching every chip the records hold, and `LFM_HASH`: `n`
/// constants summed (`last` the last, which a forged twin changes), bit
/// decompositions of `p − 1` (top 32 bits all ones: `Z = 1`), of zero and of
/// the sum, the extension ALU, two selects over hinted words, an unpack and a
/// pack, a width-16 hash, and four published words.
pub(super) fn every_chip(n: u64, last: u64) -> LfmProgram {
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(4);
    let h: Vec<Cell> = (0..4).map(|i| b.hint_word(arena, i)).collect();
    let mut acc = b.felt_const(FE::from(1u64));
    for i in 0..n {
        let v = if i + 1 == n { last } else { 1_000 + i };
        let c = b.felt_const(FE::from(v));
        acc = b.add(acc, c);
    }
    let minus_one = b.felt_const(FE::from(P_MINUS_ONE));
    let zero = b.felt_const(FE::from(0u64));
    let top = b.bit_dec(minus_one, 64);
    let _ = b.bit_dec(zero, 1);
    let bits = b.bit_dec(acc, 8);
    let [hi, lo] = b.bit_dec_be_halves(acc);
    let e1 = b.ext_const(&FEE::new([FE::from(1u64), FE::from(2u64), FE::from(3u64)]));
    let e2 = b.ext_const(&FEE::new([FE::from(4u64), FE::from(5u64), FE::from(6u64)]));
    let p = b.emul(e1, e2);
    let pb = b.emul_base(p, acc);
    let (l, r) = b.select(bits[0], h[0], h[1]);
    let (l2, _) = b.select(top[63], l, h[2]);
    let lanes = b.unpack(l2);
    let w = b.pack_word([lanes[1], lanes[0], hi, lo]);
    let hashed = b.hash16([w, r, h[2], h[3]]);
    b.public(hashed[0]);
    b.public(pb.as_cell());
    b.public(acc.as_cell());
    b.public(w);
    compile(b.finish())
}

fn build(program: &LfmProgram) -> WlfmBuild {
    build_wlfm(program, HASHER, WlfmRate::Two).unwrap_or_else(|e| panic!("the W-LFM build: {e}"))
}

/// A word canonical.
fn canonical(f: &FE) -> u64 {
    GoldilocksField::canonical(f.value())
}

// =============================================================================
// The knob
// =============================================================================

/// The chips are filled on the host by default, and the knob turns on for
/// `1` alone.
#[test]
fn the_chip_records_are_off_by_default_and_on_for_one_alone() {
    assert_eq!(CHIP_RECORDS_ENV, "LAMBDA_VM_WLFM_CHIP_RECORDS");
    assert!(
        !chip_records(),
        "the chips are filled on the host by default"
    );
    assert!(knob_word_on(Some("1")));
    assert!(knob_word_on(Some(" 1 ")));
    for word in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some("on"),
        Some("11"),
        Some("yes"),
    ] {
        assert!(!knob_word_on(word), "{word:?} turns the knob on");
    }
    let program = every_chip(5, 9);
    let filled = execute_and_fill_unmerged(&program, &arenas(), HASHER).expect("executes");
    assert!(filled.chips().is_none(), "off: no chip held as records");
    let _on = test_format::set_chip_records(true);
    let filled = execute_and_fill_unmerged(&program, &arenas(), HASHER).expect("executes");
    assert!(filled.chips().is_some(), "on: the chips held as records");
}

#[cfg(feature = "cuda")]
#[test]
fn the_levels_values_knob_pins_every_record_vector() {
    const KNOB: &str = "LAMBDA_VM_LEVELS_VALUES_PINNED";
    const CHILD: &str = "LAMBDA_VM_LEVELS_VALUES_TEST_CHILD";
    const TEST: &str = "lfm::chip_records_tests::the_levels_values_knob_pins_every_record_vector";

    // The lock and both knobs are process-wide. Run the real executor and
    // fold with each setting and with the prefix knob unset.
    if std::env::var_os(CHILD).is_none() {
        for word in [None, Some("1"), Some(" 1 ")] {
            let mut cmd = std::process::Command::new(std::env::current_exe().expect("test binary"));
            cmd.args(["--exact", TEST, "--nocapture"])
                .env(CHILD, "1")
                .env_remove("LAMBDA_VM_LEVELS_POOL_CLASS")
                .env_remove(KNOB);
            if let Some(word) = word {
                cmd.env(KNOB, word);
            }
            let output = cmd.output().expect("run the test in a child process");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success() && stdout.contains("test result: ok. 1 passed;"),
                "{KNOB}={word:?}:\n{stdout}\n{stderr}"
            );
        }
        return;
    }

    math_cuda::pinned::set_lock(|_, _| true);
    assert_eq!(math_cuda::pinned::set_pool_gib_for_tests(1), 1);
    let pool = math_cuda::pinned::lock_pool().expect("lock the stand-in pool");
    assert_eq!((pool.pieces, pool.refused), (1, 0));
    let word = std::env::var(KNOB).ok();
    let pinned = word.as_deref() == Some("1");
    let program = every_chip(37, 77);
    let mut exec = execute(&program, &arenas(), &HASHER).expect("executes");
    let values = LfmChipValues::take(&mut exec.records).upload_values();
    let mut vectors = values.split(", ");
    for kind in [
        "Balu", "Xalu", "Select", "Bitdec", "Lanes", "Hint", "Public",
    ] {
        let vector = vectors.next().expect("a record vector");
        let (bytes, status) = vector
            .strip_prefix(&format!("{kind} "))
            .and_then(|v| v.split_once(" B pinned "))
            .expect("the record vector's size and pool status");
        assert!(bytes.parse::<usize>().expect("bytes") > 0, "{vector}");
        assert_eq!(status, if pinned { "true" } else { "false" }, "{vector}");
    }
    assert!(vectors.next().is_none(), "{values}");
}

/// A fill that holds its chips as records is refused by the STARK prove:
/// their traces are empty there.
#[test]
fn the_stark_prove_refuses_chips_held_as_records() {
    let program = every_chip(5, 9);
    let _on = test_format::set_chip_records(true);
    let filled = execute_and_fill_unmerged(&program, &arenas(), HASHER).expect("executes");
    let options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .map(crate::security_posture::stark_options)
        .expect("blowup 2");
    let artifacts = super::registry::build_artifacts_with_hasher(&program, &options, HASHER);
    let err = filled
        .prove(
            &artifacts,
            &options,
            stark::residency_mode::ResidencyMode::Retain,
        )
        .err()
        .expect("refused");
    assert!(format!("{err:?}").contains("records"), "{err:?}");
}

// =============================================================================
// The host fill is the chip traces
// =============================================================================

/// ★ PIN: every chip table held as records, filled on the host
/// ([`super::chip_records::LfmChipRecords::host_columns`]), is the host's
/// `chip_trace` table column for column (canonical words), chunk by chunk:
/// cut chips (the cap at 2^3 rows, so the sum's chip is five tables and its
/// last has padding rows), padding rows, and `LFM_BITDEC`'s `Z = 1` row
/// (`p − 1`), a zero, and nonzero `GINV`s.
#[test]
fn chip_records_host_fill_is_chip_trace() {
    let _cut = test_format::set_chunk_log(3);
    let program = every_chip(37, 77);
    let mut built = build(&program);
    built.hold_chip_prefixes().expect("the prefixes pack");
    let a = &built.artifacts;
    let exec = execute(&program, &arenas(), &HASHER).expect("executes");
    let mut traces = super::trace::build_traces_with_hasher(&program, &exec.records, HASHER);
    let mut exec = exec;
    let values = LfmChipValues::take(&mut exec.records);
    let airs = a.airs();
    let mut t = 0usize;
    let mut seen = Vec::new();
    let (mut cut_chips, mut padding_rows, mut z_rows, mut ginv_rows) = (0, 0, 0, 0);
    for (i, (air, trace, _)) in airs
        .lfm
        .air_trace_pairs(&mut traces)
        .into_iter()
        .enumerate()
    {
        let chunks = &a.lfm_chunks[i];
        let Some(kind) = ChipKind::of_air(air.name()) else {
            t += chunks.len();
            continue;
        };
        seen.push(kind);
        let (width, height) = (trace.main_table.width, trace.main_table.height);
        assert_eq!(width, kind.num_columns(), "{kind:?}: the trace's width");
        assert_eq!(width, kind.air_columns(), "{kind:?}: the AIR's width");
        cut_chips += usize::from(chunks.len() > 2);
        let mut start = 0usize;
        for &rows in chunks {
            let prefix = built.chip_prefixes()[t].clone().expect("a held prefix");
            let records = values
                .table(kind, prefix, start, rows)
                .expect("held")
                .expect("the records fit");
            let (_, _, real) = records.span();
            padding_rows += rows - real;
            let host = records.host_columns();
            assert_eq!(host.len(), width, "{kind:?} table {t}: columns");
            for (c, column) in host.iter().enumerate() {
                let expected: Vec<u64> = (0..rows)
                    .map(|r| {
                        let row = start + r;
                        if row < height {
                            canonical(trace.main_table.get(row, c))
                        } else {
                            0
                        }
                    })
                    .collect();
                assert_eq!(column, &expected, "{kind:?} table {t} column {c}");
            }
            if kind == ChipKind::Bitdec {
                let z = layout::bitdec::PREP_WIDTH + 64;
                z_rows += host[z][..real].iter().filter(|&&v| v == 1).count();
                ginv_rows += host[z + 1][..real].iter().filter(|&&v| v != 0).count();
            }
            // The packed host fill is the same columns.
            let packed =
                multilinear::records::TableRecords::host_fill(&records).expect("the host fills it");
            for (c, column) in host.iter().enumerate() {
                assert_eq!(
                    &packed.column(c),
                    column,
                    "{kind:?} table {t} packed column {c}"
                );
            }
            start += rows;
            t += 1;
        }
    }
    assert_eq!(
        seen,
        ChipKind::ALL.to_vec(),
        "every chip held, in the set's order"
    );
    assert!(cut_chips >= 1, "a chip cut into three tables or more");
    assert!(padding_rows > 0, "padding rows");
    assert_eq!(z_rows, 1, "one decomposition of p − 1");
    assert!(ginv_rows >= 2, "decompositions with GINV");
}

/// The bit decomposition's value columns at its edges: `p − 1` (`Z = 1`, no
/// inverse), zero (`GINV` the inverse of `2^32 − 1`), and a word whose top is
/// `2^32 − 2` (`GINV = 1`); every `GINV` times its `g` is one.
#[test]
fn the_bit_decomposition_rows_at_their_edges() {
    let row = bitdec_row(P_MINUS_ONE);
    assert_eq!(row[64], 1);
    assert_eq!(row[65], 0);
    assert_eq!(&row[..32], &[0; 32]);
    assert_eq!(&row[32..64], &[1; 32]);
    let zero = bitdec_row(0);
    assert_eq!(&zero[..64], &[0; 64]);
    assert_eq!(zero[64], 0);
    assert_eq!(FE::from(zero[65]) * FE::from(0xFFFF_FFFFu64), FE::one());
    let one = bitdec_row(0xFFFF_FFFE_0000_0005);
    assert_eq!((one[64], one[65]), (0, 1));
    // A non-canonical word is decomposed as its canonical one.
    assert_eq!(bitdec_row(P_MINUS_ONE + 1 + 7), bitdec_row(7));
}

// =============================================================================
// Proofs
// =============================================================================

/// The proof's bytes over `arenas` with the chips `hold` (0 wide, 1 narrow,
/// 2 records, 3 records with no prefix held by the build), `proving` as given;
/// the proof verifies, and the stamps say the chips were held so.
fn proof_bytes(
    program: &LfmProgram,
    built: &WlfmBuild,
    arenas: &[Vec<super::word::LfmWord>],
    hold: u8,
    proving: WlfmProving,
) -> Vec<u8> {
    let _narrow = test_format::set_narrow_chips(hold == 1);
    let _records = test_format::set_chip_records(hold >= 2);
    let _ = take_wlfm_stamps();
    let proof = prove_wlfm_as(program, built, arenas, proving).expect("proves");
    let stamps = take_wlfm_stamps().expect("stamped");
    assert_eq!(
        stamps.records_tables > 0,
        hold >= 2,
        "chips held as records iff on"
    );
    assert_eq!(
        stamps.records_prefix_pack > 0.0,
        hold == 3,
        "the prove packs the prefixes only when the build holds none"
    );
    assert_eq!(stamps.narrow_tables > 0, hold == 1, "chips narrow iff on");
    verify_wlfm(&built.artifacts, &proof, &proof.public_words).expect("verifies");
    rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .expect("serializes")
        .to_vec()
}

/// A byte pin's case: its name, the program, its arenas, and the chip cut.
type Case = (
    &'static str,
    LfmProgram,
    Vec<Vec<super::word::LfmWord>>,
    Option<u32>,
);

/// ★ PIN: the chips held as their records are the prover's holding: a program
/// of every chip, one of lean compressions and chips, and one with chips cut
/// into chunks, prove the same bytes with the chips wide, narrow and as
/// records (with the build's prefixes, and without them), held and streamed.
#[test]
fn chips_as_records_prove_the_same_bytes() {
    let _fast = test_format::set(Some(2));
    let cases: Vec<Case> = vec![
        ("every chip", every_chip(5, 9), arenas(), None),
        (
            "lean and chips",
            super::b3l_wlfm_tests::honest(),
            Vec::new(),
            None,
        ),
        ("cut chips", every_chip(37, 77), arenas(), Some(3)),
    ];
    for (what, program, arenas, cut) in cases {
        let _cut = cut.map(test_format::set_chunk_log);
        let bare = build(&program);
        let mut held = build(&program);
        held.hold_chip_prefixes().expect("the prefixes pack");
        assert!(held.chip_prefix_secs() > 0.0);
        let wide = proof_bytes(&program, &bare, &arenas, 0, WlfmProving::Held);
        for (hold, built, proving, name) in [
            (1, &bare, WlfmProving::Held, "narrow"),
            (2, &held, WlfmProving::Held, "records"),
            (2, &held, WlfmProving::Streamed, "records, streamed"),
            (
                3,
                &bare,
                WlfmProving::Held,
                "records, prefixes packed by the prove",
            ),
            (
                3,
                &bare,
                WlfmProving::Streamed,
                "records, streamed, prefixes by the prove",
            ),
        ] {
            assert_eq!(
                wide,
                proof_bytes(&program, built, &arenas, hold, proving),
                "{what}: {name}"
            );
        }
    }
}

// =============================================================================
// Refusals
// =============================================================================

/// A forged value word in one record (a base ALU row's `out`), held as
/// records and proved by a prover that does not check: refused.
#[test]
fn a_forged_value_word_in_the_records_is_refused() {
    let _fast = test_format::set(Some(2));
    let program = every_chip(5, 9);
    let built = build(&program);
    let forged = |by: u64| {
        let mut exec = execute(&program, &arenas(), &HASHER).expect("executes");
        let out = &mut exec.records.balu[2].out;
        *out += FE::from(by);
        super::proof::lfm_fill_executed_holding(&program, exec, HASHER, 0.0, &Vec::new(), true)
    };
    let proof = prove_wlfm_traces(&built, forged(0), true).expect("the honest prove");
    verify_wlfm(&built.artifacts, &proof, &proof.public_words).expect("honest verifies");
    let proof = prove_wlfm_traces(&built, forged(1), false).expect("the forger's prover");
    assert!(
        verify_wlfm(&built.artifacts, &proof, &proof.public_words).is_err(),
        "a forged out word must be refused"
    );
}

/// `built`'s table `t` with its held prefix's word at `(row, col)` changed
/// (kept within the column's width).
fn forge_prefix(built: WlfmBuild, t: usize, row: usize, col: usize) -> WlfmBuild {
    let mut prefix: NarrowColumns = built.chip_prefixes()[t]
        .as_deref()
        .expect("a held prefix")
        .clone();
    let word = prefix.get(row, col);
    assert!(prefix.set(row, col, word ^ 1), "the forged word fits");
    built.with_chip_prefix(t, prefix)
}

/// The tables of the chip `name`, as positions.
fn tables_of(built: &WlfmBuild, name: &str) -> Vec<usize> {
    let airs = built.artifacts.airs();
    airs.air_refs()
        .iter()
        .enumerate()
        .filter(|(_, a)| a.name() == name)
        .map(|(t, _)| t)
        .collect()
}

/// ★ A forged prefix word (an address of a base ALU row) in the prefix a
/// chip's records carry: the records skip the prover's prefix check, so the
/// prepared opening is what refuses it; the same build unforged verifies.
#[test]
fn a_forged_prefix_word_is_refused_by_the_prepared_opening() {
    let _fast = test_format::set(Some(2));
    let _on = test_format::set_chip_records(true);
    let program = every_chip(5, 9);
    let mut built = build(&program);
    built.hold_chip_prefixes().expect("the prefixes pack");
    let balu = tables_of(&built, "LFM_BALU");
    assert_eq!(balu.len(), 1);
    let honest = prove_wlfm_as(&program, &built, &arenas(), WlfmProving::Held).expect("proves");
    verify_wlfm(&built.artifacts, &honest, &honest.public_words).expect("honest verifies");
    let forged = forge_prefix(built, balu[0], 1, layout::balu::A_ADDR);
    let proof = prove_wlfm_as(&program, &forged, &arenas(), WlfmProving::Held)
        .expect("the prover does not check a records table's prefix");
    assert!(
        verify_wlfm(&forged.artifacts, &proof, &proof.public_words).is_err(),
        "a forged prefix word must be refused"
    );
}

/// The same forgery in a cut chip's last chunk (its padding rows past the
/// real ones included in the table): refused by that chunk's opening.
#[test]
fn a_forged_prefix_word_in_a_later_chunk_is_refused() {
    let _fast = test_format::set(Some(2));
    let _cut = test_format::set_chunk_log(3);
    let _on = test_format::set_chip_records(true);
    let program = every_chip(37, 77);
    let mut built = build(&program);
    built.hold_chip_prefixes().expect("the prefixes pack");
    let balu = tables_of(&built, "LFM_BALU");
    assert!(
        balu.len() >= 3,
        "the sum's chip is cut: {} tables",
        balu.len()
    );
    let honest = prove_wlfm_as(&program, &built, &arenas(), WlfmProving::Streamed).expect("proves");
    verify_wlfm(&built.artifacts, &honest, &honest.public_words).expect("honest verifies");
    let last = *balu.last().expect("a last chunk");
    let forged = forge_prefix(built, last, 0, layout::balu::OUT_ADDR);
    let proof = prove_wlfm_as(&program, &forged, &arenas(), WlfmProving::Streamed)
        .expect("the prover does not check a records table's prefix");
    assert!(
        verify_wlfm(&forged.artifacts, &proof, &proof.public_words).is_err(),
        "a forged prefix word in a later chunk must be refused"
    );
}

/// A forged program (another constant) proved against the honest build with
/// its chips held as records, by a prover that does not check: refused, as
/// with the chips' traces.
#[test]
fn a_forged_program_held_as_records_is_refused() {
    let _fast = test_format::set(Some(2));
    let _on = test_format::set_chip_records(true);
    let built = build(&every_chip(5, 9));
    let filled = execute_and_fill_unmerged(&every_chip(5, 9), &arenas(), HASHER)
        .expect("the honest program executes");
    let honest = prove_wlfm_traces(&built, filled, false).expect("the honest prove");
    verify_wlfm(&built.artifacts, &honest, &honest.public_words).expect("honest verifies");
    let filled = execute_and_fill_unmerged(&every_chip(5, 10), &arenas(), HASHER)
        .expect("the forged program executes");
    assert!(filled.chips().is_some());
    let proof = prove_wlfm_traces(&built, filled, false).expect("the forger's prover");
    assert!(verify_wlfm(&built.artifacts, &proof, &proof.public_words).is_err());
}
