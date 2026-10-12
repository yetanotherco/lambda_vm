//! The WHIR commit's encoding follows the process's LDE setting: the column
//! engine by default, the legacy `lift_spread` + tiled NTT under
//! `LAMBDA_VM_LDE_LEGACY=1` (`math_cuda::lde_cm::LEGACY_ENV`), the switch that
//! sends every device LDE back.
//!
//! ```text
//! cargo test --release -p math-cuda --test whir_encode_setting -- --nocapture
//! LAMBDA_VM_LDE_LEGACY=1 cargo test --release -p math-cuda --test whir_encode_setting -- --nocapture
//! ```
//!
//! The parity tests force the pipeline per call (`with_engine`), so none of
//! them reads the setting; this one does. It is its own binary with one test,
//! so the process-wide engine counter moves for this commit alone, and the
//! once-per-process `[gpu] LDE: …` banner is printed by this commit's decision
//! (visible under `--nocapture`).

use math_cuda::DeviceHash;
use math_cuda::lde_cm::{LEGACY_ENV, codewords_encoded, spread_supports};

const P: u64 = 0xFFFF_FFFF_0000_0001;

#[test]
fn the_whir_encoding_follows_the_process_setting() {
    // A 2^14 codeword at blowup 4: a shape the engine takes.
    const LOG_EVALS: usize = 12;
    const LOG_BLOWUP: usize = 2;
    assert!(spread_supports(
        (LOG_EVALS + LOG_BLOWUP) as u32,
        LOG_BLOWUP as u32
    ));
    let setting = std::env::var(LEGACY_ENV).unwrap_or_else(|_| "<unset>".into());
    let legacy = setting == "1";
    let evals: Vec<u64> = (0..1u64 << LOG_EVALS)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) % P)
        .collect();

    let before = codewords_encoded();
    math_cuda::whir::commit_codeword_to_host(&evals, LOG_BLOWUP, 4, DeviceHash::Rpx256)
        .expect("the device commit");
    let through_engine = codewords_encoded() - before;
    println!(
        "whir_encode_setting: {LEGACY_ENV}={setting}: the encoding took the {} \
         ({through_engine} codeword through the engine)",
        if through_engine == 0 {
            "legacy path"
        } else {
            "column engine"
        },
    );
    assert_eq!(
        through_engine,
        u64::from(!legacy),
        "{LEGACY_ENV}={setting}: the WHIR encoding did not follow the process setting"
    );
}
