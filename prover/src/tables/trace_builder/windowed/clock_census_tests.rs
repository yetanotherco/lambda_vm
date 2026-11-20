//! ★ The clock's facts MEMW_A's limbs rest on, on a real block
//! (`BLOCK_WHIR_ELF`, `BLOCK_WHIR_INPUT`): walked as the block's production
//! build walks it, in the limbs' form, every MEMW_R, MEMW_A and MEMW access
//! has a timestamp and an old timestamp of high word 0 (the CPU's high word
//! is the constant 0, its rows bounded by the block verifier), and no aligned
//! access is refused by the limbs' predicate (so the limbs' build routes
//! exactly as LT's). Every window but the run's last (the finish's, which a
//! `Walker` does not take: its HALT finalizations are at 2^32 − 1, high word
//! 0, and the small-block proofs cover them).
//!
//! Prints `CLOCK CENSUS` lines: rows per table, the highest timestamp, the
//! high words seen, and the comparisons LT no longer takes.

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;

use super::WindowedTraceBuilder;
use crate::tables::memw_aligned::MemwAlignedOrder;

#[test]
#[ignore = "a real block's walk (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT); box tier"]
fn a_real_blocks_accesses_are_on_high_word_zero_and_fit_the_limbs() {
    let elf =
        std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).expect("the ELF");
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).expect("input");
    let program = Elf::load(&elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    let t = std::time::Instant::now();
    let mut builder = WindowedTraceBuilder::new(&program, &input, &options.max_rows)
        .and_then(|b| b.memw_aligned_order(MemwAlignedOrder::Limbs))
        .expect("the builder");
    let mut executor = Executor::new(&program, input.clone()).expect("the executor starts");
    let (mut walker, _) = builder.split();
    // (rows, timestamps whose high word is not 0) per MEMW_R, MEMW_A, MEMW.
    let mut seen = [[0u64; 2]; 3];
    let (mut windows, mut cycles, mut highest, mut refused) = (0usize, 0usize, 0u64, 0u64);
    let mut ahead: Option<Vec<Log>> = None;
    while let Some(logs) = executor.resume_with_limit(window).expect("the run") {
        // The run's last window is the finish's.
        let Some(previous) = ahead.replace(logs.to_vec()) else {
            continue;
        };
        cycles += previous.len();
        let w = walker.walk(&previous).expect("a window");
        let memw = &w.walk.memw;
        assert!(
            memw.words.is_none(),
            "the walk writes register rows, not words"
        );
        let mut note = |table: usize, timestamps: [u64; 2]| {
            seen[table][0] += 1;
            for ts in timestamps {
                highest = highest.max(ts);
                seen[table][1] += u64::from(ts >> 32 != 0);
            }
        };
        for row in &memw.register_rows {
            let c = row.card_record();
            // The old timestamp's high word is the row's own (MEMW_R's rule).
            note(0, [c.timestamp, c.timestamp]);
        }
        for row in &memw.aligned {
            assert!(row.limbs(), "a MEMW_A row in the limbs' form");
            note(1, [row.timestamp(), row.old_timestamp()]);
        }
        for op in &memw.general {
            let width = usize::from(op.width.max(1));
            for &old in &op.old_timestamp[..width] {
                note(2, [op.timestamp, old]);
            }
            // An aligned access the limbs' predicate sent to MEMW.
            refused += u64::from(super::super::is_aligned_op(op));
        }
        windows += 1;
    }
    let [memw_r, memw_a, memw] = seen;
    println!(
        "CLOCK CENSUS: {windows} windows · {cycles} cycles · MEMW_R rows {} · MEMW_A rows {} \
         (LT comparisons the limbs remove) · MEMW accesses {} · highest timestamp {highest:#x} · \
         high words not 0: MEMW_R {} MEMW_A {} MEMW {} · aligned accesses the limbs refused \
         {refused} · {:.1}s",
        memw_r[0],
        memw_a[0],
        memw[0],
        memw_r[1],
        memw_a[1],
        memw[1],
        t.elapsed().as_secs_f64()
    );
    assert!(windows > 0, "a block of one window walks nothing here");
    assert_eq!(
        (memw_r[1], memw_a[1], memw[1]),
        (0, 0, 0),
        "every access on high word 0"
    );
    assert_eq!(
        refused, 0,
        "no aligned access refused by the limbs' predicate"
    );
    assert!(memw_a[0] > 0, "the block makes MEMW_A rows");
    println!("CLOCK CENSUS VERDICT PASS");
}
