//! The WHIR commit's encoding takes the column engine in production, the
//! process's LDE pipeline (`math_cuda::lde_cm`).
//!
//! ```text
//! cargo test --release -p math-cuda --test whir_encode_setting -- --nocapture
//! ```
//!
//! The parity tests force the pipeline per call (`with_engine`), so none of
//! them reads what production takes; this one does. It is its own binary with
//! one test, so the process-wide engine counter moves for this commit alone.

use math_cuda::DeviceHash;
use math_cuda::lde_cm::{codewords_encoded, spread_supports};

const P: u64 = 0xFFFF_FFFF_0000_0001;

#[test]
fn the_whir_encoding_takes_the_column_engine() {
    // A 2^14 codeword at blowup 4: a shape the engine takes.
    const LOG_EVALS: usize = 12;
    const LOG_BLOWUP: usize = 2;
    assert!(spread_supports(
        (LOG_EVALS + LOG_BLOWUP) as u32,
        LOG_BLOWUP as u32
    ));
    let evals: Vec<u64> = (0..1u64 << LOG_EVALS)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) % P)
        .collect();

    let before = codewords_encoded();
    math_cuda::whir::commit_codeword_to_host(&evals, LOG_BLOWUP, 4, DeviceHash::Rpx256)
        .expect("the device commit");
    let through_engine = codewords_encoded() - before;
    println!(
        "whir_encode_setting: the encoding took the {} \
         ({through_engine} codeword through the engine)",
        if through_engine == 0 {
            "legacy path"
        } else {
            "column engine"
        },
    );
    assert_eq!(
        through_engine, 1,
        "the WHIR encoding did not take the column engine"
    );
}
