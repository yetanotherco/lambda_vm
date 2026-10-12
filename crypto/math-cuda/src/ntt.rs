//! Forward and inverse NTT over Goldilocks base field. Matches the algebraic
//! contract of `math::polynomial::Polynomial::evaluate_fft` /
//! `interpolate_fft`:
//!   input  = n elements in natural order
//!   output = n elements in natural order.
//!
//! Parity is checked by `tests/ntt.rs` against the CPU implementation.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsFFTField, IsField};

use crate::Result;
use crate::device::backend;

/// Host-side twiddle table: `[ω^0, ω^1, ..., ω^{n/2-1}]` where ω is the
/// primitive n-th root of unity. Exposed for `device::Backend::cached_twiddles`
/// and for direct use in tests / benches.
pub fn twiddles_forward(log_n: u64) -> Vec<u64> {
    // Smallest meaningful NTT is size 2 (log_n = 1); size-1 has nothing to
    // twiddle. The shift `1 << (log_n - 1)` underflows for log_n = 0.
    assert!(log_n >= 1, "twiddles_forward: log_n must be >= 1");
    let omega = *GoldilocksField::get_primitive_root_of_unity(log_n)
        .expect("primitive root")
        .value();
    powers_of(omega, 1usize << (log_n - 1))
}

/// Inverse twiddle table: `[ω^{-i}]` for i in [0, n/2).
pub fn twiddles_inverse(log_n: u64) -> Vec<u64> {
    assert!(log_n >= 1, "twiddles_inverse: log_n must be >= 1");
    let omega = GoldilocksField::get_primitive_root_of_unity(log_n).expect("primitive root");
    let omega_inv = FieldElement::<GoldilocksField>::inv(&omega).expect("inverse");
    powers_of(*omega_inv.value(), 1usize << (log_n - 1))
}

fn powers_of(base: u64, count: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(count);
    let mut w = 1u64;
    for _ in 0..count {
        out.push(w);
        w = GoldilocksField::mul(&w, &base);
    }
    out
}

/// Forward NTT on a slice of `n = 2^log_n` Goldilocks coefficients. Takes
/// natural-order input and returns natural-order evaluations.
pub fn forward(coeffs: &[u64]) -> Result<Vec<u64>> {
    ntt_inplace(coeffs, /*forward=*/ true)
}

/// Inverse NTT on a slice of `n = 2^log_n` Goldilocks evaluations. Takes
/// natural-order evaluations and returns natural-order coefficients. Includes
/// the 1/n scaling.
pub fn inverse(evals: &[u64]) -> Result<Vec<u64>> {
    ntt_inplace(evals, /*forward=*/ false)
}

fn ntt_inplace(input: &[u64], forward: bool) -> Result<Vec<u64>> {
    let n = input.len();
    // Empty / size-1 has no work to do. `is_power_of_two()` returns false for
    // 0, so this branch must come before the assert to avoid panicking on
    // empty input.
    if n <= 1 {
        return Ok(input.to_vec());
    }
    assert!(n.is_power_of_two(), "ntt length must be a power of two");
    assert!(
        n <= u32::MAX as usize,
        "ntt length {n} exceeds u32 range — kernel grid would silently truncate",
    );
    let log_n = n.trailing_zeros() as u64;

    let be = backend()?;
    let stream = be.next_stream();

    let mut x_dev = stream.clone_htod(input)?;
    let tw_dev = if forward {
        be.fwd_twiddles_for(log_n)?
    } else {
        be.inv_twiddles_for(log_n)?
    };

    let n_u64 = n as u64;

    // 1. Bit-reverse: natural → bit-reversed.
    unsafe {
        stream
            .launch_builder(&be.bit_reverse_permute)
            .arg(&mut x_dev)
            .arg(&n_u64)
            .arg(&log_n)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    // 2. DIT butterfly levels. For log_n >= 8 we fuse 8 levels per kernel via
    // the shmem kernel; for very small sizes (< 256 elements) we stick with
    // the per-level kernel because the shmem block dimensions assume n ≥ 256.
    run_ntt_body(stream.as_ref(), &mut x_dev, tw_dev.as_ref(), n_u64, log_n)?;

    // 3. For iNTT, multiply by 1/n.
    if !forward {
        let n_fe = FieldElement::<GoldilocksField>::from(n as u64);
        let inv_n = *n_fe.inv().expect("n is non-zero").value();
        unsafe {
            stream
                .launch_builder(&be.scalar_mul)
                .arg(&mut x_dev)
                .arg(&inv_n)
                .arg(&n_u64)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    let out = stream.clone_dtoh(&x_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Largest grid.y a tile launch uses before it moves the excess into grid.z.
///
/// CUDA caps grid.y (and grid.z) at 65,535 on every compute capability, and a
/// tile grid is a power of two, so 2^15 is the largest y that fits. A tile's
/// y count is `n >> (level + k)`: the NTT's first 5-level tile needs
/// `n >> 13`, which reaches 65,536 at a 2^29 codeword — the WHIR base commit at
/// stack 27, blowup 4 — and the launch failed there, every such commit falling
/// back to the host (`thoughts/zf/gap/fix/STRUCT.md` §9.2). Every launch at or
/// below the cap is the one it always was.
const GRID_Y_CAP: u64 = 1 << 15;

thread_local! {
    /// Per-thread override of [`GRID_Y_CAP`]. See [`with_grid_y_cap`].
    static GRID_Y_CAP_OVERRIDE: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// `blocks` tile blocks (a power of two) as `(grid.y, grid.z)`: all of them in
/// y up to the cap, the rest as multiples of it in z. The tile kernels read
/// their block's high index as `blockIdx.z * gridDim.y + blockIdx.y`.
pub(crate) fn split_grid_y(blocks: u64) -> (u32, u32) {
    let cap = GRID_Y_CAP_OVERRIDE
        .with(std::cell::Cell::get)
        .unwrap_or(GRID_Y_CAP);
    assert!(
        blocks.is_power_of_two(),
        "a tile grid is a power of two, got {blocks}"
    );
    let y = blocks.min(cap);
    let z = blocks / y;
    assert!(
        z <= 65_535,
        "a tile grid of {blocks} blocks exceeds grid.y x grid.z"
    );
    (y as u32, z as u32)
}

/// Run `f` with the tile grids split at `cap` blocks in y instead of 2^15 on
/// the calling thread (the launches happen there), so a test can drive the
/// grid.z path at a size that fits a test. `cap` must be a power of two.
#[doc(hidden)]
pub fn with_grid_y_cap<R>(cap: u64, f: impl FnOnce() -> R) -> R {
    assert!(cap.is_power_of_two(), "the y cap is a power of two");
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            GRID_Y_CAP_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(GRID_Y_CAP_OVERRIDE.with(|c| c.replace(Some(cap))));
    f()
}

/// Run the butterfly body of a bit-reversed-input DIT NTT. Split out so the
/// LDE orchestrator can reuse it on the same device buffer.
/// Columns a fused tile spans — one warp, which is what keeps it coalesced.
/// Mirrors `NTT_TILE_COLS` in `kernels/ntt.cu`.
const TILE_COLS: u32 = 32;

/// Levels a fused tile takes at once. `32 × 32` threads is a full block and
/// `32 × 33 × 8` bytes of shared memory, which is room to spare.
const TILE_LEVELS: u64 = 5;

pub(crate) fn run_ntt_body(
    stream: &cudarc::driver::CudaStream,
    x_dev: &mut cudarc::driver::CudaSlice<u64>,
    tw_dev: &cudarc::driver::CudaSlice<u64>,
    n: u64,
    log_n: u64,
) -> Result<()> {
    let be = backend()?;
    // Levels 0..min(log_n, 8): one shmem-fused launch. Loads are fully
    // coalesced (base_step=0 → `row = tid`) and 8 butterfly rounds stay on
    // chip. This is the big DRAM-bandwidth win.
    let fused = core::cmp::min(log_n, 8);
    if fused >= 8 {
        let grid_x = (n / 256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let base_step = 0u64;
        unsafe {
            stream
                .launch_builder(&be.ntt_dit_8_levels)
                .arg(&mut *x_dev)
                .arg(tw_dev)
                .arg(&n)
                .arg(&log_n)
                .arg(&base_step)
                .launch(cfg)?;
        }
    } else {
        // Sub-256-element NTT. Use per-level.
        let half_cfg = LaunchConfig::for_num_elems((n / 2) as u32);
        for level in 0..fused {
            unsafe {
                stream
                    .launch_builder(&be.ntt_dit_level)
                    .arg(&mut *x_dev)
                    .arg(tw_dev)
                    .arg(&n)
                    .arg(&log_n)
                    .arg(&level)
                    .launch(half_cfg)?;
            }
        }
    }

    // Levels 8..log_n, five at a time through a 2D tile: `threadIdx.x` walks
    // the contiguous low bits and `threadIdx.y` the stride the butterflies
    // move, so both the load and the store stay coalesced. The older
    // fused-with-row-remap path gathered along the stride instead and lost
    // more bandwidth than it saved in passes — that is why the per-level
    // fallback is still here for what does not tile.
    let half_cfg = LaunchConfig::for_num_elems((n / 2) as u32);
    let mut level = fused;
    while level < log_n {
        let k = core::cmp::min(TILE_LEVELS, log_n - level);
        // A tile needs a warp of contiguous indices below it, and is only
        // worth its idle half when it fuses more than a level.
        if k >= 2 && (1u64 << level) >= TILE_COLS as u64 {
            let rows = 1u32 << k;
            let (grid_y, grid_z) = split_grid_y(n >> (level + k));
            let cfg = LaunchConfig {
                grid_dim: (((1u64 << level) / TILE_COLS as u64) as u32, grid_y, grid_z),
                block_dim: (TILE_COLS, rows, 1),
                shared_mem_bytes: rows * (TILE_COLS + 1) * 8,
            };
            let k_u32 = k as u32;
            unsafe {
                stream
                    .launch_builder(&be.ntt_dit_tile)
                    .arg(&mut *x_dev)
                    .arg(tw_dev)
                    .arg(&n)
                    .arg(&log_n)
                    .arg(&level)
                    .arg(&k_u32)
                    .launch(cfg)?;
            }
            level += k;
            continue;
        }
        unsafe {
            stream
                .launch_builder(&be.ntt_dit_level)
                .arg(&mut *x_dev)
                .arg(tw_dev)
                .arg(&n)
                .arg(&log_n)
                .arg(&level)
                .launch(half_cfg)?;
        }
        level += 1;
    }
    Ok(())
}

/// Pointwise multiply: `x[i] *= w[i]`.
pub fn pointwise_mul(x: &[u64], w: &[u64]) -> Result<Vec<u64>> {
    assert_eq!(x.len(), w.len());
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let be = backend()?;
    let stream = be.next_stream();

    let mut x_dev = stream.clone_htod(x)?;
    let w_dev = stream.clone_htod(w)?;

    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.pointwise_mul)
            .arg(&mut x_dev)
            .arg(&w_dev)
            .arg(&n_u64)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    let out = stream.clone_dtoh(&x_dev)?;
    stream.synchronize()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_grids_split_into_z_only_past_the_cap() {
        // Up to 2^30 blocks fit (2^15 in y times 2^15 in z); the largest tile
        // grid a Goldilocks transform can ask for is 2^32 >> 13 = 2^19.
        for log in 0..=30u32 {
            let blocks = 1u64 << log;
            let (y, z) = split_grid_y(blocks);
            assert_eq!(
                u64::from(y) * u64::from(z),
                blocks,
                "2^{log}: every block launched"
            );
            assert!(
                y <= 65_535 && z <= 65_535,
                "2^{log}: ({y}, {z}) past the device limits"
            );
            if blocks <= GRID_Y_CAP {
                assert_eq!(
                    (y, z),
                    (blocks as u32, 1),
                    "2^{log}: the launch it always was"
                );
            }
        }
        // The WHIR base codewords at stacks 27 and 28 (blowup 4): the first
        // 5-level tile's n >> 13.
        assert_eq!(split_grid_y((1 << 29) >> 13), (1 << 15, 2));
        assert_eq!(split_grid_y((1 << 30) >> 13), (1 << 15, 4));
        // A test cap moves the split down, and restores on return.
        assert_eq!(with_grid_y_cap(2, || split_grid_y(64)), (2, 32));
        assert_eq!(split_grid_y(64), (64, 1));
    }

    #[test]
    #[should_panic(expected = "exceeds grid.y x grid.z")]
    fn a_tile_grid_past_both_dimensions_is_refused_not_truncated() {
        split_grid_y(1 << 31);
    }
}
