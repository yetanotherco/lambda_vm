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
//! the unsplit launch cannot run: stack 27 against the host pipeline those
//! commits fell back to, and stacks 27 and 28 against the column engine's
//! encoding (`LAMBDA_VM_GAP_K1B`), which uses grid.x alone. Run them alone:
//! several GiB of device and host memory each.
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

/// Stack 27 against the host pipeline its commits fell back to (`multilinear`'s
/// encode and tree, as `tests/whir_commit.rs` uses at small sizes), at the
/// production shape: blowup 4, first6 leaves, RPX. The codeword is 2^29, past
/// what the unsplit launch could run, so equal codewords and roots are the
/// pipeline's bytes at that size — what the fallback produced, now from the
/// device. `commit_codeword_to_host` has no host fallback of its own: it runs
/// the kernels or errors.
#[test]
#[ignore = "a 4 GiB codeword on each side and a host encode: run alone on the box"]
fn the_stack_27_commit_matches_the_host_pipeline() {
    use math::field::element::FieldElement;
    use math::field::goldilocks::GoldilocksField as F;
    use multilinear::mle::Mle;
    use multilinear::whir::{self, Domain};
    use multilinear::whir_commit::CodewordCommitment;
    use multilinear::whir_hash::RpxWhir;

    const LOG_EVALS: usize = 27;
    const LOG_BLOWUP: usize = 2;
    const LOG_FOLDING: usize = 6;
    let evals = felts(0x2727_0006, 1 << LOG_EVALS);

    let t = std::time::Instant::now();
    let (device, nodes) = math_cuda::whir::commit_codeword_to_host(
        &evals,
        LOG_BLOWUP,
        LOG_FOLDING,
        DeviceHash::Rpx256,
    )
    .unwrap_or_else(|e| panic!("the stack-27 device commit failed: {e:?}"));
    println!(
        "grid_limits: stack 27 device commit in {:.2} s",
        t.elapsed().as_secs_f64()
    );

    let t = std::time::Instant::now();
    let f = Mle::new(evals.into_iter().map(FieldElement::<F>::from_raw).collect())
        .expect("power of two");
    let domain = Domain::<F>::new(LOG_EVALS + LOG_BLOWUP).expect("domain");
    let host_codeword =
        whir::encode::<F, F>(&whir::lift_coefficients(&f), &domain).expect("host encode");
    drop(f);
    assert_eq!(device.len(), host_codeword.len());
    if let Some(i) = device
        .iter()
        .zip(&host_codeword)
        .position(|(&d, h)| canon(d) != canon(*h.value()))
    {
        panic!("stack 27: codeword position {i} differs from the host encode");
    }
    drop(device);
    let host = CodewordCommitment::<F, RpxWhir>::from_codeword(host_codeword, LOG_FOLDING)
        .expect("host commit");
    println!(
        "grid_limits: stack 27 host encode and tree in {:.2} s",
        t.elapsed().as_secs_f64()
    );
    let root: [u8; 32] = nodes[..32].try_into().unwrap();
    assert_eq!(
        root,
        host.root(),
        "stack 27: the device root is not the host's"
    );
    println!(
        "grid_limits: stack 27 codeword and root equal to the host's, root0 {:016x}",
        u64::from_le_bytes(root[..8].try_into().unwrap())
    );
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

/// The control: with the split pushed back past the limit (65,536 blocks in
/// y, the old geometry), the same 2^29 transform must be refused by the device
/// — the failure the fix removes, reproduced in the same binary, so the passing
/// tests above are not passing for some other reason.
#[test]
#[ignore = "a 4 GiB transform: run alone on the box"]
fn the_unsplit_launch_is_refused_at_65536_blocks() {
    let x = felts(0x6553_6000, 1 << 29);
    // Not `expect_err`: its message would print the 2^29 values on success.
    match with_grid_y_cap(1 << 16, || math_cuda::ntt::forward(&x)) {
        Ok(v) => panic!("a 65,536-block grid.y launched ({} values back)", v.len()),
        Err(err) => println!("grid_limits: unsplit 2^29 forward NTT refused as expected: {err:?}"),
    }
    let split = math_cuda::ntt::forward(&x).expect("the split launch runs");
    assert_eq!(split.len(), x.len());
}
