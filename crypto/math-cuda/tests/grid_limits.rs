//! CUDA caps grid.y at 65,535. The NTT's and the Möbius transform's 2-D tile
//! launches put a tile's high index in grid.y, and at the WHIR base's largest
//! stacks that index needs more: the NTT's first 5-level tile launches
//! `n >> 13` blocks in y, 65,536 at a 2^29 codeword (stack 27, blowup 4), and
//! the launch failed — every such commit fell back to the host
//! (`thoughts/zf/gap/fix/STRUCT.md` §9.2). The launches now move the excess
//! into grid.z (`ntt::split_grid_y`); the butterflies are the same, so the
//! output is too.
//!
//! The cheap tests force the split at every tile (`with_grid_y_cap(1, ..)`) on
//! small shapes and compare against the unsplit launch — the index math the
//! big shapes rely on. The `#[ignore]`d ones commit at the real sizes, where
//! the unsplit launch cannot run, and compare the legacy encoding against the
//! column engine's (`LAMBDA_VM_GAP_K1B`), which uses grid.x alone. Run them
//! alone: several GiB of device and host memory each.
//!
//! ```text
//! cargo test --release -p math-cuda --test grid_limits
//! cargo test --release -p math-cuda --test grid_limits -- --ignored --test-threads=1 --nocapture
//! ```

use math_cuda::DeviceHash;
use math_cuda::lde_cm::with_k1b;
use math_cuda::ntt::with_grid_y_cap;

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// Deterministic canonical field elements (splitmix64), fast enough for 2^29.
fn felts(seed: u64, len: usize) -> Vec<u64> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) % P
        })
        .collect()
}

fn canon(v: u64) -> u64 {
    if v >= P { v - P } else { v }
}

/// A position-sensitive digest of a codeword's canonical values, so two large
/// codewords compare without both being held on the host.
fn digest(values: &[u64]) -> u64 {
    values.iter().fold(0x243F_6A88_85A3_08D3u64, |acc, &v| {
        acc.rotate_left(7).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ canon(v)
    })
}

#[test]
fn ntt_tiles_split_into_grid_z_match_the_single_y_launch() {
    for log_n in [13u32, 14, 16, 18] {
        let x = felts(0x6E77 + u64::from(log_n), 1 << log_n);
        for forward in [true, false] {
            let run = || {
                if forward {
                    math_cuda::ntt::forward(&x)
                } else {
                    math_cuda::ntt::inverse(&x)
                }
            };
            let whole = run().expect("unsplit launch");
            let split = with_grid_y_cap(1, run).expect("every tile split into z");
            assert_eq!(
                whole, split,
                "2^{log_n} forward={forward}: the grid.z launch changed the transform"
            );
        }
    }
}

#[test]
fn whir_commit_tiles_split_into_grid_z_match_the_single_y_launch() {
    // The Möbius tiles start at 2^13 evaluations and the NTT tiles at a 2^13
    // codeword; spacing 1 and 4 both.
    for (log_evals, log_blowup) in [(13usize, 1usize), (14, 2), (16, 0), (17, 2)] {
        let evals = felts(0x3141 + log_evals as u64, 1 << log_evals);
        for k1b in [false, true] {
            let commit = || {
                with_k1b(k1b, || {
                    math_cuda::whir::commit_codeword_to_host(
                        &evals,
                        log_blowup,
                        2,
                        DeviceHash::Rpx256,
                    )
                })
            };
            let whole = commit().expect("unsplit commit");
            let split = with_grid_y_cap(1, commit).expect("split commit");
            assert_eq!(
                whole.0.iter().map(|&v| canon(v)).collect::<Vec<_>>(),
                split.0.iter().map(|&v| canon(v)).collect::<Vec<_>>(),
                "2^{log_evals} at 2^{log_blowup} (K1B {k1b}): codeword"
            );
            assert_eq!(whole.1, split.1, "2^{log_evals} (K1B {k1b}): Merkle nodes");
        }
    }
}

/// One commit's root and codeword digest, under the legacy encoding or the
/// column engine's.
fn commit_digest(evals: &[u64], log_blowup: usize, k1b: bool) -> ([u8; 32], u64) {
    let t = std::time::Instant::now();
    let (values, nodes) = with_k1b(k1b, || {
        math_cuda::whir::commit_codeword_to_host(evals, log_blowup, 4, DeviceHash::Rpx256)
    })
    .unwrap_or_else(|e| panic!("commit (K1B {k1b}) failed: {e:?}"));
    let root: [u8; 32] = nodes[..32].try_into().unwrap();
    let d = digest(&values);
    println!(
        "grid_limits: 2^{} evaluations at 2^{log_blowup}, K1B {k1b}: committed in {:.2} s, root0 {:016x}",
        evals.len().trailing_zeros(),
        t.elapsed().as_secs_f64(),
        u64::from_le_bytes(root[..8].try_into().unwrap())
    );
    (root, d)
}

/// The WHIR base commits at stacks 27 and 28 (blowup 4): codewords of 2^29 and
/// 2^30, whose first NTT tile needs 65,536 and 131,072 blocks in y.
#[test]
#[ignore = "4–8 GiB codewords: run alone on the box"]
fn whir_commit_at_stacks_27_and_28() {
    for log_evals in [27usize, 28] {
        let evals = felts(0x2728 + log_evals as u64, 1 << log_evals);
        let legacy = commit_digest(&evals, 2, false);
        let engine = commit_digest(&evals, 2, true);
        assert_eq!(
            legacy, engine,
            "2^{log_evals}: legacy and engine encodings disagree"
        );
    }
}

/// 2^29 evaluations at spacing 1: the Möbius tile's first launch needs 65,536
/// blocks in y too, and so does the NTT's on the 2^29 codeword.
#[test]
#[ignore = "a 4 GiB codeword and 4 GiB of coefficients: run alone on the box"]
fn whir_commit_with_the_mobius_past_the_limit() {
    let evals = felts(0x2929, 1 << 29);
    let legacy = commit_digest(&evals, 0, false);
    let engine = commit_digest(&evals, 0, true);
    assert_eq!(
        legacy, engine,
        "2^29 at spacing 1: legacy and engine encodings disagree"
    );
}
