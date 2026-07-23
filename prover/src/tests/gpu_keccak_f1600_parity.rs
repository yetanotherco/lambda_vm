//! Phase 6 (precompiles): the batched on-GPU Keccak-f[1600] permutation
//! (`math_cuda::precompile::gpu_keccak_f1600_batch`) must be bit-identical to the executor's
//! reference `keccak_f1600` applied per state. This is the first precompile computation moved to
//! device — the building block for a device-side KeccakPermute ecall (its memory effects depend on
//! the permutation output). Self-contained (no ELF needed); requires a GPU.
//!
//! `cargo test -p lambda-vm-prover --release --features cuda --lib gpu_keccak_f1600 -- --ignored --nocapture`

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_keccak_f1600_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping: no CUDA backend");
        return;
    }
    let n = 4096usize;
    // Deterministic, well-mixed input states (no RNG): a splitmix-style fill per lane.
    let mut flat = Vec::with_capacity(n * 25);
    for i in 0..n {
        for j in 0..25 {
            let seed = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((j as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
                .wrapping_add(0xABCD_1234_5678_9EF0);
            flat.push(seed);
        }
    }

    let gpu = math_cuda::precompile::gpu_keccak_f1600_batch(&flat).expect("gpu keccak batch");
    assert_eq!(gpu.len(), flat.len());

    for i in 0..n {
        let mut st = [0u64; 25];
        st.copy_from_slice(&flat[i * 25..i * 25 + 25]);
        executor::vm::instruction::execution::keccak_f1600(&mut st);
        assert_eq!(
            &gpu[i * 25..i * 25 + 25],
            &st[..],
            "keccak-f1600 state {i} mismatch (GPU vs CPU)"
        );
    }
    println!("gpu_keccak_f1600 OK: {n} Keccak-f[1600] permutations bit-identical to CPU reference");
}
