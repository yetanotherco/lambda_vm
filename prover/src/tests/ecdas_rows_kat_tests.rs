//! The ECDAS halves rows' words, pinned: a change to how the fill is compiled
//! (its writer inlined or not) leaves every word of the table as it was.

use crate::tables::ecdas::{self, CompactEcdasOp, EcdasForm, EcdasOperation};
use crate::tables::gpack::TraceForm;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use stark::trace::TraceTable;

/// A step a row, every field distinct from its neighbours' (not a valid
/// doubling or addition: the words are what is pinned).
fn step(i: usize) -> ::ecsm::EcdasStep {
    let bytes =
        |f: usize| -> [u8; 32] { std::array::from_fn(|b| ((i * 31 + f * 7 + b) % 251) as u8) };
    let wide =
        |f: usize| -> [u8; 33] { std::array::from_fn(|b| ((i * 17 + f * 5 + b) % 241) as u8) };
    let carries = |k: usize| -> [i64; 64] {
        std::array::from_fn(|j| ((i + k) as i64 * 37 + j as i64 * 11) % 4000 - 2000)
    };
    ::ecsm::EcdasStep {
        x_a: bytes(1),
        y_a: bytes(2),
        x_g: bytes(3),
        y_g: bytes(4),
        round: (i % 256) as u8,
        op: (i % 2) as u8,
        next_op: ((i + 1) % 2) as u8,
        lambda: bytes(5),
        x_r: bytes(6),
        y_r: bytes(7),
        q0: wide(8),
        q1: wide(9),
        q2: wide(10),
        c0: carries(0),
        c1: carries(1),
        c2: carries(2),
    }
}

/// The main trace's words, row-major, packed or wide.
fn words(table: &TraceTable<GoldilocksField, GoldilocksExtension>) -> Vec<u64> {
    match table.narrow_main() {
        Some(packed) => {
            let columns: Vec<Vec<u64>> = (0..packed.cols()).map(|c| packed.column(c)).collect();
            (0..packed.rows())
                .flat_map(|r| columns.iter().map(move |column| column[r]))
                .collect()
        }
        None => {
            let main = &table.main_table;
            (0..main.height)
                .flat_map(|r| (0..main.width).map(move |c| *main.get(r, c).value()))
                .collect()
        }
    }
}

/// FNV-1a over the words' little-endian bytes.
fn digest(words: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// 300 steps padded to 512 rows, packed and wide: the same words, whose digest
/// is the one the build with the writer out of line by the code-unit split (2f4941f37) wrote.
#[test]
fn the_ecdas_halves_rows_words_are_pinned() {
    let rows: Vec<CompactEcdasOp> = (0..300)
        .map(|i| {
            CompactEcdasOp::from_op(&EcdasOperation {
                timestamp: 4 * i as u64 + 3,
                step: step(i),
            })
        })
        .collect();
    // The fill without the relation check (the steps are synthetic); the
    // checked fill is the same row closure with its check on.
    let narrow = ecdas::generate_ecdas_rows_in(&rows, 512, TraceForm::Narrow, EcdasForm::Halves);
    let wide = ecdas::generate_ecdas_rows_in(&rows, 512, TraceForm::Wide, EcdasForm::Halves);
    let (n, w) = (words(&narrow), words(&wide));
    assert_eq!(n, w, "packed and wide hold the same words");
    let got = digest(&n);
    println!("ECDAS HALVES KAT: {} words · digest {got:#018x}", n.len());
    assert_eq!(got, PINNED, "the ECDAS halves rows' words moved");
}

const PINNED: u64 = 0x70db_4aa3_81be_de2e;
