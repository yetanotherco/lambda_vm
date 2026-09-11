//! Parity for the degree-1 (num_parts==1) composition-parts de-interleave
//! kernel (`comp_h_to_slabs_ext3`).
//!
//! On the prove path a table with `num_parts == 1` has `H` itself as its single
//! composition part, already on the LDE coset. The device path keeps it resident
//! by de-interleaving the interleaved ext3 evals `H` (`h[row*3 + k]`) into the
//! 3-slab layout every downstream consumer (R2 commit, R3 OOD, R4 DEEP, openings)
//! reads (`buf[(0*3 + k) * lde_size + row]`). It is a pure transpose — no
//! arithmetic — so raw u64 equality must hold bit-for-bit.
//!
//! Requires a visible GPU (like the other math-cuda GPU parity tests).

use math_cuda::constraint_interp::{comp_h_from_host_interleaved, comp_h_to_slabs};
use math_cuda::device::backend;

fn check(num_rows: usize, seed: u64) {
    // Deterministic interleaved ext3 `H` (raw, possibly non-canonical limbs —
    // the stronger test, and exactly what a real resident `H` carries).
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    let interleaved: Vec<u64> = (0..num_rows * 3).map(|_| next()).collect();

    let h = comp_h_from_host_interleaved(&interleaved, num_rows).expect("upload H");
    let handle = comp_h_to_slabs(&h).expect("de-interleave H into slabs");
    assert_eq!(handle.m, 1, "num_rows={num_rows}: single part");
    assert_eq!(handle.lde_size, num_rows, "num_rows={num_rows}: lde_size");
    assert_eq!(
        handle.buf.len(),
        3 * num_rows,
        "num_rows={num_rows}: slab buffer"
    );

    let be = backend().expect("cuda backend");
    let stream = be.next_stream();
    handle
        .wait_ready_on(stream.as_ref())
        .expect("wait on de-interleave");
    let slab = stream
        .clone_dtoh(handle.buf.as_ref())
        .expect("download slabs");
    stream.synchronize().expect("sync download");

    for row in 0..num_rows {
        for k in 0..3 {
            let got = slab[k * num_rows + row];
            let want = interleaved[row * 3 + k];
            assert_eq!(
                got, want,
                "num_rows={num_rows} row={row} comp={k}: slab {got:#018x} vs interleaved {want:#018x}"
            );
        }
    }
}

#[test]
fn comp_h_to_slabs_parity() {
    for log in 1..=14 {
        check(1usize << log, 0x00C0_FFEE_0000_0000 ^ log as u64);
    }
}
