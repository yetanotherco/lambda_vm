#[cfg(target_arch = "riscv64")]
use core::arch::asm;

/// Memory-mapped private input region start address.
/// Layout: 4-byte LE length prefix at this address, data at +4.
/// The host pre-loads the input; the guest reads directly (no ecall).
/// Must match `executor::vm::memory::PRIVATE_INPUT_START_INDEX`.
#[cfg(target_arch = "riscv64")]
pub const PRIVATE_INPUT_START: usize = 0xFF000000;

/// Maximum private-input length the guest will read, in bytes (512 MiB).
/// The host caps stored input at this size in `Memory::store_private_inputs`,
/// so an honest length prefix is always `<=` this bound; a larger value can only
/// come from a malformed or forged prefix. The reader clamps to this cap so a
/// bogus length can never make the guest fabricate an arbitrarily long slice.
/// Must match `executor::vm::memory::MAX_PRIVATE_INPUT_SIZE`.
#[cfg(target_arch = "riscv64")]
const MAX_PRIVATE_INPUT_SIZE: usize = 512 * 1024 * 1024;

#[cfg(target_arch = "riscv64")]
pub enum SyscallNumbers {
    Print = 1,
    Panic = 2,
    Commit = 64,
    Halt = 93,
}

/// Syscall number for KeccakPermute (u64::MAX - 1).
#[cfg(target_arch = "riscv64")]
const KECCAK_SYSCALL_NUMBER: usize = usize::MAX - 1;

/// Syscall number for the ECSM secp256k1 scalar-multiply accelerator (-11 as usize).
#[cfg(target_arch = "riscv64")]
const ECSM_SYSCALL_NUMBER: usize = usize::MAX - 10;

// Field-native hash/transcript measurement ecalls (EXPERIMENT 1). These are
// TRUSTED, execute-only stubs: the executor computes the correct value host-side
// and returns it in one cycle. They drive no chip, so a program that emits them
// must never be proven (the same unbalanced-Ecall-bus caveat as `Print`); they
// exist only to measure the optimistic cycle ceiling of a field-native
// hash/transcript accelerator. Guarded by the `sim-hash-ecalls` cfg so a normal
// guest build is byte-identical (the wrappers below are simply not compiled).
#[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
const SIM_ABSORB_FELTS_SYSCALL_NUMBER: usize = usize::MAX - 2;
#[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
const SIM_ABSORB_BYTES_SYSCALL_NUMBER: usize = usize::MAX - 3;
#[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
const SIM_TRANSCRIPT_SAMPLE_SYSCALL_NUMBER: usize = usize::MAX - 4;
#[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
const SIM_HASH_PAIR_SYSCALL_NUMBER: usize = usize::MAX - 5;
#[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
const SIM_HASH_FELTS_SYSCALL_NUMBER: usize = usize::MAX - 6;

/// Syscall number for the Goldilocks inverse HINT (u64::MAX - 7). EXPERIMENT 5.
#[cfg(all(target_arch = "riscv64", feature = "sim-inv-hint"))]
const INV_GOLDILOCKS_HINT_SYSCALL_NUMBER: usize = usize::MAX - 7;

/// Syscall number for the `REDUCED_OPENING_ROW` measurement stub (Level A).
///
/// MEASUREMENT-ONLY: has no chip table, so NEVER prove a build that emits it —
/// the unmatched Ecall on the LogUp bus would fail verification (same caveat as
/// `Print`). Only used behind the `sim-ro-ecalls` feature of the recursion
/// verifier guest.
#[cfg(target_arch = "riscv64")]
const REDUCED_OPENING_ROW_SYSCALL_NUMBER: usize = usize::MAX - 20;

/// Syscall number for the `REDUCED_OPENING_QUERY` measurement stub (Level B).
/// Same MEASUREMENT-ONLY caveat as [`REDUCED_OPENING_ROW_SYSCALL_NUMBER`].
#[cfg(target_arch = "riscv64")]
const REDUCED_OPENING_QUERY_SYSCALL_NUMBER: usize = usize::MAX - 21;

/// No-op. The `Print` ecall (a7=1) has no receiver on the Ecall bus, so emitting
/// it makes the LogUp bus unbalance and the proof fail to verify. Printing isn't
/// needed in provable programs, so `print_string` does nothing on every target.
/// Keep it as a no-op rather than deleting call sites: that way no guest path can
/// ever reintroduce an unmatched Print ecall. (See `SyscallNumbers::Print`.)
pub fn print_string(_s: &str) {}

/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_write(_fildes: i32, buf: *const u8, size: usize) -> isize {
    print_string("sys_write called\n");
    let content = unsafe { core::slice::from_raw_parts(buf, size) };
    print_string(&("SYS_WRITE: ".to_owned() + str::from_utf8(content).unwrap_or("<invalid utf8>"))); // Does the print of the sys write
    size.try_into().unwrap_or(-1)
}

#[cfg(target_arch = "riscv64")]
/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_panic(msg_ptr: *const u8, len: usize) {
    print_string("Sys panic called\n");
    unsafe {
        asm!(
            "ecall",
            in("a0") msg_ptr,
            in("a1") len,
            in("a7") SyscallNumbers::Panic as usize,
        )
    }
}

#[cfg(target_arch = "riscv64")]
pub fn commit(slice: &[u8]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") 1usize,
            in("a1") slice.as_ptr(),
            in("a2") slice.len(),
            in("a7") SyscallNumbers::Commit as usize,
        )
    }
}

/// Read private input bytes from the memory-mapped region at
/// `PRIVATE_INPUT_START = 0xFF000000`.
///
/// The host pre-loads the input before execution; this function reads the
/// 4-byte LE length prefix and then copies the data bytes into a new `Vec`.
/// No ecall is performed — it's a plain memory read (ZisK-style).
#[cfg(target_arch = "riscv64")]
pub fn get_private_input() -> Vec<u8> {
    // Copy the borrowed private-input bytes into an owned `Vec`. The raw-pointer
    // read (length prefix + data slice) and its single `unsafe` block live in
    // `get_private_input_slice`, so the memory layout is defined in one place.
    get_private_input_slice().to_vec()
}

#[cfg(not(target_arch = "riscv64"))]
pub fn get_private_input() -> Vec<u8> {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

/// Borrow the private input bytes in place from the memory-mapped region —
/// no copy, no allocation. Same layout as [`get_private_input`]; the returned
/// slice starts at `PRIVATE_INPUT_START + 4` (a 4-aligned address) and lives
/// for the whole execution (the host never remaps the region).
#[cfg(target_arch = "riscv64")]
pub fn get_private_input_slice() -> &'static [u8] {
    // SAFETY: The host pre-loads private input at PRIVATE_INPUT_START before
    // execution and never remaps it afterward, so the returned slice is valid
    // for the `'static` lifetime of the guest's single-threaded execution
    // region, which stays mapped and unmodified for the whole execution.
    let len_ptr = PRIVATE_INPUT_START as *const u32;
    // Clamp the prover-written length prefix to `MAX_PRIVATE_INPUT_SIZE`. An
    // honest prefix (written by the host, which caps stored input at this size)
    // is always within bound, so clamping never changes behavior for real
    // inputs — it only bounds the slice length when a malformed or forged prefix
    // claims more, keeping the read deterministic.
    let len = (unsafe { core::ptr::read_volatile(len_ptr) } as usize).min(MAX_PRIVATE_INPUT_SIZE);
    let data_ptr = (PRIVATE_INPUT_START + 4) as *const u8;
    unsafe { core::slice::from_raw_parts(data_ptr, len) }
}

#[cfg(not(target_arch = "riscv64"))]
pub fn get_private_input_slice() -> &'static [u8] {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[cfg(target_arch = "riscv64")]
pub fn sys_halt() -> ! {
    // NOTE: no print_string here — the Print ecall is unmatched on the Ecall bus
    // and would cause a verification failure.
    unsafe {
        asm!(
            "ecall",
            in("a0") 0usize, // exit_code = 0 (enforced by HALT read on x10)
            in("a7") SyscallNumbers::Halt as usize,
        );
    }
    unreachable!()
}

#[cfg(not(target_arch = "riscv64"))]
pub fn sys_halt() -> ! {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[cfg(target_arch = "riscv64")]
/// Apply the Keccak-f[1600] permutation to a 25-element u64 state in-place.
pub fn keccak_permute(state: &mut [u64; 25]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") state.as_mut_ptr(),
            in("a7") KECCAK_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// Apply the Keccak-f[1600] permutation to a 25-element u64 state in-place.
pub fn keccak_permute(_state: &mut [u64; 25]) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[cfg(target_arch = "riscv64")]
/// Compute `xR = (k·G)_x` on secp256k1 via the ECSM accelerator. All values are 32-byte
/// little-endian. Requires `0 < k < N` and a canonical valid `xG` curve coordinate.
/// `xG` and `k` must not overlap; `xR` may alias either input.
pub fn ecsm_mul(xr: &mut [u8; 32], xg: &[u8; 32], k: &[u8; 32]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") xr.as_mut_ptr(), // x10 = address to write xR
            in("a1") xg.as_ptr(),     // x11 = address of xG
            in("a2") k.as_ptr(),      // x12 = address of k
            in("a7") ECSM_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// Compute `xR = (k·G)_x` on secp256k1 via the ECSM accelerator (32-byte little-endian values).
pub fn ecsm_mul(_xr: &mut [u8; 32], _xg: &[u8; 32], _k: &[u8; 32]) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

/// Goldilocks inverse HINT (EXPERIMENT 5). Asks the executor for `x^-1`: it
/// overwrites `x` in place with the inverse and returns it. The returned value
/// is UNTRUSTED — the caller MUST verify `x * result == 1` (one field multiply)
/// and reject on mismatch. Because only the true inverse passes that check,
/// hinting is SOUND: a wrong hint can only make an honest proof reject, never
/// make a false one accept. `x` must be a nonzero canonical Goldilocks element.
///
/// The ecall drives no chip on this branch (the value is checked in-circuit), so
/// a build that emits it is execute-only — never prove it (the Ecall-bus caveat
/// shared with `Print`).
#[cfg(all(target_arch = "riscv64", feature = "sim-inv-hint"))]
pub fn inv_goldilocks_hint(x: u64) -> u64 {
    let mut slot = x;
    unsafe {
        asm!(
            "ecall",
            in("a0") &mut slot as *mut u64,
            in("a7") INV_GOLDILOCKS_HINT_SYSCALL_NUMBER,
        )
    }
    slot
}

// =============================================================================
// Field-native hash/transcript measurement ecalls (EXPERIMENT 1)
//
// Raw ecall wrappers for the five trusted stubs. The host reproduces the exact
// byte semantics of the corresponding guest software path (see the executor's
// `sim_hash` module) and returns in one cycle. 4-argument calls pass the fourth
// operand in a3 (x13); the executor reads a0/a1/a2/a3 accordingly. All pointers
// are into guest memory. Only compiled under the `sim-hash-ecalls` feature (the
// `crypto` swap sites forward it), so a normal build never references them.
// =============================================================================

/// `ABSORB_FELTS(state_ptr, elems_ptr, count, kind)`: absorb `count` field
/// elements (each `kind` Goldilocks limbs, read as raw doublewords) into the
/// sponge at `state_ptr`, using the canonical `stream_bytes` serialization.
#[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
pub fn sim_absorb_felts(state_ptr: *mut u8, elems_ptr: *const u8, count: usize, kind: usize) {
    unsafe {
        asm!(
            "ecall",
            in("a0") state_ptr,
            in("a1") elems_ptr,
            in("a2") count,
            in("a3") kind,
            in("a7") SIM_ABSORB_FELTS_SYSCALL_NUMBER,
        )
    }
}

/// `ABSORB_BYTES(state_ptr, bytes_ptr, len)`: absorb `len` raw bytes into the
/// sponge at `state_ptr`.
#[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
pub fn sim_absorb_bytes(state_ptr: *mut u8, bytes_ptr: *const u8, len: usize) {
    unsafe {
        asm!(
            "ecall",
            in("a0") state_ptr,
            in("a1") bytes_ptr,
            in("a2") len,
            in("a7") SIM_ABSORB_BYTES_SYSCALL_NUMBER,
        )
    }
}

/// `TRANSCRIPT_SAMPLE(state_ptr, out32_ptr)`: run the whole transcript
/// `sample()` on the sponge at `state_ptr` and write the 32-byte result to
/// `out32_ptr` (finalize-reset + reverse + re-absorb, in one call).
#[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
pub fn sim_transcript_sample(state_ptr: *mut u8, out32_ptr: *mut u8) {
    unsafe {
        asm!(
            "ecall",
            in("a0") state_ptr,
            in("a1") out32_ptr,
            in("a7") SIM_TRANSCRIPT_SAMPLE_SYSCALL_NUMBER,
        )
    }
}

/// `HASH_PAIR(l_ptr, r_ptr, out_ptr)`: Keccak-256 of the two concatenated
/// 32-byte nodes (the fixed Merkle-parent shape); writes 32 bytes to `out_ptr`.
#[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
pub fn sim_hash_pair(l_ptr: *const u8, r_ptr: *const u8, out_ptr: *mut u8) {
    unsafe {
        asm!(
            "ecall",
            in("a0") l_ptr,
            in("a1") r_ptr,
            in("a2") out_ptr,
            in("a7") SIM_HASH_PAIR_SYSCALL_NUMBER,
        )
    }
}

/// `HASH_FELTS(a_ptr, a_count, b_ptr, b_count, kind, out_ptr)`: one-shot leaf
/// hash of the concatenation `a ‖ b` of two field-element slices (each `kind`
/// limbs); writes the 32-byte digest to `out_ptr`. A single-slice leaf passes
/// `b_count = 0`. The two-slice form matches the verifier's `evaluations ‖
/// evaluations_sym` leaf shape. Uses a0..a5.
#[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
#[allow(clippy::too_many_arguments)]
pub fn sim_hash_felts(
    a_ptr: *const u8,
    a_count: usize,
    b_ptr: *const u8,
    b_count: usize,
    kind: usize,
    out_ptr: *mut u8,
) {
    unsafe {
        asm!(
            "ecall",
            in("a0") a_ptr,
            in("a1") a_count,
            in("a2") b_ptr,
            in("a3") b_count,
            in("a4") kind,
            in("a5") out_ptr,
            in("a7") SIM_HASH_FELTS_SYSCALL_NUMBER,
        )
    }
}

// =============================================================================
// DEEP reduced-opening measurement stubs (Level A / Level B).
//
// MEASUREMENT-ONLY. Each computes the CORRECT reduced-opening value host-side
// in one VM cycle (trusted passthrough) so the guest still accepts the proof,
// letting us measure the cycle ceiling of a fused reduced-opening chip. They
// have NO chip table — NEVER prove a build that emits them (LogUp bus would
// unbalance, like `Print`). See `math::sim_ro` for the input struct layouts and
// `others/accelerator_noop_sim_spec.md` (Experiment 2).
// =============================================================================

/// Level A — `REDUCED_OPENING_ROW`. `input_ptr` points to a
/// `math::sim_ro::ReducedOpeningRowInput`; the host writes
/// `(base_row_sum, base_row_sum_sym)` (2 extension elements = 6 u64) at
/// `out_ptr` for the given `row_idx`.
#[cfg(target_arch = "riscv64")]
pub fn reduced_opening_row(input_ptr: usize, row_idx: usize, out_ptr: usize) {
    unsafe {
        asm!(
            "ecall",
            in("a0") input_ptr,
            in("a1") row_idx,
            in("a2") out_ptr,
            in("a7") REDUCED_OPENING_ROW_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// Level A reduced-opening measurement stub (riscv64 guest only).
pub fn reduced_opening_row(_input_ptr: usize, _row_idx: usize, _out_ptr: usize) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

/// Level B — `REDUCED_OPENING_QUERY`. `input_ptr` points to a
/// `math::sim_ro::ReducedOpeningQueryInput`; the host writes
/// `(deep_eval, deep_eval_sym)` (2 extension elements = 6 u64) at `out_ptr`.
#[cfg(target_arch = "riscv64")]
pub fn reduced_opening_query(input_ptr: usize, out_ptr: usize) {
    unsafe {
        asm!(
            "ecall",
            in("a0") input_ptr,
            in("a1") out_ptr,
            in("a7") REDUCED_OPENING_QUERY_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// Level B reduced-opening measurement stub (riscv64 guest only).
pub fn reduced_opening_query(_input_ptr: usize, _out_ptr: usize) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}
// =============================================================================
// Stub implementations for unsupported std functions
// These functions are required by Rust's std zkvm module but are not supported
// in Lambda VM. They will panic at runtime if called.
// =============================================================================

/// # Safety
///
/// This function is not supported in Lambda VM.
/// It will panic if called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_read(_fd: u32, _buf: *mut u8, _nbytes: usize) -> usize {
    panic!("sys_read is not supported: io::Read for Stdin is not implemented in Lambda VM");
}

/// # Safety
///
/// This function is not supported in Lambda VM.
/// It will panic if called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_argc() -> usize {
    panic!("sys_argc is not supported: command-line arguments are not available in Lambda VM");
}

/// # Safety
///
/// This function is not supported in Lambda VM.
/// It will panic if called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_argv(_buf: *mut u32, _buf_nwords: usize, _arg_idx: usize) -> usize {
    panic!("sys_argv is not supported: command-line arguments are not available in Lambda VM");
}
