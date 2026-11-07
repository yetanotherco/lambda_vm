//! The GPU nonce search must produce nonces the host predicate accepts. There
//! is nothing to compare against the CPU search itself — any nonce satisfying
//! `is_valid_nonce` is as good as any other, and the CPU's `find_any` does not
//! even agree with itself between runs — so what is pinned here is validity,
//! plus the search completeness that minimality stands in for.
//!
//! Runs on the merge-queue GPU box via `make test-math-cuda`
//! (`cargo test -p math-cuda --release`) — `device::backend()` inside
//! `generate_nonce_gpu` requires a real GPU, like the other tests here.
//!
//! Uses real grinding factors (>= the min-factor gate). The end-to-end prover
//! suite only exercises `grinding_factor: 1`, where `limit = 1 << 63` lets a
//! broken kernel return an accepted nonce ~half the time; these factors make a
//! wrong kernel fail deterministically.
//!
//! The lanes come from `stark::grinding::inner_hash_lanes`, the same call the
//! prover makes — building them here instead would leave the production
//! conversion untested.

use stark::grinding::{inner_hash_lanes, is_valid_nonce};

/// The kernel's outer hash is keccak, so the lanes and the validity pin are
/// keccak's grinding digest — the same type the dispatch in
/// `generate_nonce_maybe_gpu` keys the device search on.
type Keccak = stark::config::GrindingDigest<stark::config::KeccakStarkHash>;

/// At a moderate factor the kernel returns a valid nonce, and it is the
/// smallest one (the exhaustive CPU scan below it is cheap at factor 14).
///
/// Minimality is not a contract — any valid nonce would do — but it is a cheap
/// probe of search completeness: a stride or bounds bug that skipped part of
/// the range would still return a *valid* nonce, just not the first one, and
/// plain validity checking would miss that. Deterministic despite the grid
/// being parallel, because `atomicMin` is an order-independent reduction. If a
/// future kernel drops minimality deliberately, relax this to validity rather
/// than treating the red as a defect.
#[test]
fn gpu_grind_returns_smallest_valid_nonce() {
    let seed = [14u8; 32];
    let factor = 14u8;
    let nonce =
        math_cuda::grinding::generate_nonce_gpu(&inner_hash_lanes::<Keccak>(&seed, factor), factor)
            .expect("GPU grind (needs a GPU)");
    assert!(
        is_valid_nonce::<Keccak>(&seed, nonce, factor),
        "GPU nonce {nonce} fails is_valid_nonce (factor {factor})"
    );
    assert!(
        (0..nonce).all(|n| !is_valid_nonce::<Keccak>(&seed, n, factor)),
        "GPU nonce {nonce} is not the smallest valid nonce (factor {factor})"
    );
}

/// At the production factor the kernel returns a valid nonce (validity only —
/// scanning 0..nonce would be ~2^20 hashes).
#[test]
fn gpu_grind_valid_at_production_factor() {
    let seed = [20u8; 32];
    let factor = 20u8;
    let nonce =
        math_cuda::grinding::generate_nonce_gpu(&inner_hash_lanes::<Keccak>(&seed, factor), factor)
            .expect("GPU grind (needs a GPU)");
    assert!(
        is_valid_nonce::<Keccak>(&seed, nonce, factor),
        "GPU nonce {nonce} fails is_valid_nonce (factor {factor})"
    );
}

/// Below the min-factor gate the GPU path declines (→ CPU search), so the tiny
/// factors every non-GPU-benchmark test uses never pay a launch.
#[test]
fn gpu_grind_declines_below_min_factor() {
    let seed = [1u8; 32];
    assert!(
        math_cuda::grinding::generate_nonce_gpu(&inner_hash_lanes::<Keccak>(&seed, 1), 1).is_none(),
        "GPU grind should decline factor 1"
    );
}

/// ★ THE BLOCK SIZE AND THE GRID CANNOT MOVE THE ANSWER, EXECUTED.
///
/// `search` scans contiguous blocks from zero and returns the first hitting
/// block's minimum, so the nonce is a function of the inner hash and the
/// grinding factor alone. That is what makes a sweep of either knob move no
/// proof byte — and it is a property that can fail: a stride or bounds defect
/// would return a different valid nonce at a different stride, and only an
/// equality across settings can see it.
///
/// Through `generate_nonce_gpu_at` rather than the environment because the
/// knobs cache in a `OnceLock`: one process cannot read two settings through
/// `LAMBDA_VM_GRIND_*`, and two processes would be two device contexts.
#[test]
fn the_nonce_is_the_same_at_every_scan_factor_and_grid() {
    let seed = [20u8; 32];
    let factor = 20u8;
    let lanes = inner_hash_lanes::<Keccak>(&seed, factor);

    let record = math_cuda::grinding::Knobs::DEFAULT;
    let expected = math_cuda::grinding::generate_nonce_gpu_at(&lanes, factor, record)
        .expect("GPU grind at the record posture (needs a GPU)");
    assert!(
        is_valid_nonce::<Keccak>(&seed, expected, factor),
        "the record posture's nonce {expected} fails is_valid_nonce"
    );

    // Both knobs, both directions, including the pair the ruling names.
    for knobs in [
        math_cuda::grinding::Knobs {
            scan: 1,
            grid: 1024,
        },
        math_cuda::grinding::Knobs {
            scan: 2,
            grid: 1024,
        },
        math_cuda::grinding::Knobs {
            scan: 8,
            grid: 4096,
        },
        math_cuda::grinding::Knobs { scan: 8, grid: 256 },
        math_cuda::grinding::Knobs {
            scan: 1,
            grid: 4096,
        },
    ] {
        let nonce = math_cuda::grinding::generate_nonce_gpu_at(&lanes, factor, knobs)
            .expect("GPU grind (needs a GPU)");
        assert!(
            is_valid_nonce::<Keccak>(&seed, nonce, factor),
            "nonce {nonce} from {knobs:?} fails is_valid_nonce"
        );
        assert_eq!(
            nonce, expected,
            "{knobs:?} returned {nonce}, the record posture returned {expected} — \
             the launch geometry moved the answer, so the search is not scanning \
             contiguously from zero"
        );
    }
}

/// And it is still the SMALLEST at every setting, not merely the same one.
///
/// Factor 14 so the exhaustive host scan below the answer stays cheap. Equality
/// across settings (the test above) would be satisfied by a search that
/// consistently skipped the same range; minimality is what rules that out.
#[test]
fn the_nonce_is_still_the_smallest_at_a_narrow_grid() {
    let seed = [14u8; 32];
    let factor = 14u8;
    let lanes = inner_hash_lanes::<Keccak>(&seed, factor);
    for knobs in [
        math_cuda::grinding::Knobs { scan: 1, grid: 256 },
        math_cuda::grinding::Knobs {
            scan: 8,
            grid: 4096,
        },
    ] {
        let nonce = math_cuda::grinding::generate_nonce_gpu_at(&lanes, factor, knobs)
            .expect("GPU grind (needs a GPU)");
        assert!(
            (0..nonce).all(|n| !is_valid_nonce::<Keccak>(&seed, n, factor)),
            "nonce {nonce} from {knobs:?} is not the smallest valid nonce"
        );
    }
}
