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

/// Syscall number for Keccak-256 of a buffer (u64::MAX - 4). Must match
/// `executor::...::execution::KECCAK256_SYSCALL_NUMBER`.
#[cfg(target_arch = "riscv64")]
const KECCAK256_SYSCALL_NUMBER: usize = usize::MAX - 4;

/// Syscall number for the BLAKE3 6-round compression accelerator (u64::MAX - 2).
#[cfg(target_arch = "riscv64")]
const BLAKE3_SYSCALL_NUMBER: usize = usize::MAX - 2;

/// Syscall number for the BLAKE3 6-round chained-absorb accelerator
/// (u64::MAX - 3).
#[cfg(target_arch = "riscv64")]
const BLAKE3_ABSORB_SYSCALL_NUMBER: usize = usize::MAX - 3;

/// Syscall number for the ECSM secp256k1 scalar-multiply accelerator (-11 as usize).
#[cfg(target_arch = "riscv64")]
const ECSM_SYSCALL_NUMBER: usize = usize::MAX - 10;

/// Syscall number for the ECDSA public-key recovery accelerator (-12 as usize). Must match
/// the executor's `ECREC_SYSCALL_NUMBER`.
#[cfg(target_arch = "riscv64")]
const ECREC_SYSCALL_NUMBER: usize = usize::MAX - 11;

/// Copy-accelerator syscall number, serving `memcpy` and `memmove` (-30 as usize).
/// Must match the executor.
#[cfg(all(target_arch = "riscv64", not(feature = "fast-memcpy")))]
pub(crate) const DMA_MEMCPY_SYSCALL_NUMBER: usize = usize::MAX - 29;
/// Maximum bytes sent in one DMA ecall. Larger `memcpy` calls are split by the
/// strong assembly stub so continuation table height remains bounded by cycles.
#[cfg(all(target_arch = "riscv64", not(feature = "fast-memcpy")))]
pub(crate) const DMA_MEMCPY_MAX_BYTES: usize = 256;

/// `memset` syscall number (-31 as usize). Must match the executor.
#[cfg(all(target_arch = "riscv64", not(feature = "fast-memcpy")))]
pub(crate) const DMA_MEMSET_SYSCALL_NUMBER: usize = usize::MAX - 30;

/// Syscall number for the non-constraining Hint ecall (-32 as usize).
/// Must match `executor::...::execution::HINT_SYSCALL_NUMBER` (u64::MAX - 31).
#[cfg(target_arch = "riscv64")]
const HINT_SYSCALL_NUMBER: usize = usize::MAX - 31;

/// Hint selectors passed in `a0` (must match the executor's `HINT_*`).
pub const HINT_FIELD_INV: usize = 0;
pub const HINT_SCALAR_INV: usize = 1;
pub const HINT_FIELD_SQRT: usize = 2;

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
            // `a0` carries the file descriptor in and the byte count out, so it must
            // be declared clobbered: with a plain `in` the compiler is entitled to
            // assume it still holds 1 afterwards and reuse it, which would read a
            // value the COPY chip proves is `count`.
            inout("a0") 1usize => _,
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

/// Whether [`keccak256`] accepts these addresses — a sufficient form of the
/// executor's `keccak256_addresses_ok`, cheap enough to run before every call:
/// every input block's 136-byte window, and the 32-byte output, stay below one
/// `2^32` boundary of the low address word. (`136·(⌊len/136⌋ + 1) ≤ len + 136`,
/// so this refuses at most a few calls the VM would take; a caller falls back to
/// the sponge over [`keccak_permute`] for those.)
#[inline(always)]
pub fn keccak256_fits(input: &[u8], out: &[u8; 32]) -> bool {
    const TOP: u64 = 1 << 32;
    let ptr_lo = (input.as_ptr() as u64) & (TOP - 1);
    let out_lo = (out.as_ptr() as u64) & (TOP - 1);
    // `len < 2^32` first, so the sum below cannot overflow.
    (input.len() as u64) < TOP && ptr_lo + input.len() as u64 + 136 <= TOP && out_lo + 32 <= TOP
}

#[cfg(target_arch = "riscv64")]
/// Keccak-256 of `input` into `out`, in one VM call (the KECCAK table's sponge
/// mode). The caller checks [`keccak256_fits`] first: the VM refuses operands
/// whose blocks cross a `2^32` address-word boundary.
pub fn keccak256(input: &[u8], out: &mut [u8; 32]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") input.as_ptr(),
            in("a1") input.len(),
            in("a2") out.as_mut_ptr(),
            in("a7") KECCAK256_SYSCALL_NUMBER,
            options(nostack),
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// Keccak-256 of `input` into `out`, in one VM call.
pub fn keccak256(_input: &[u8], _out: &mut [u8; 32]) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[cfg(target_arch = "riscv64")]
/// BLAKE3 **6-round** compression via the accelerator (internal variant — NOT
/// standard 7-round BLAKE3; see `thoughts/blake3/blake3-chip/DESIGN.md`).
///
/// `state` is the 176-byte region as 22 dwords: `h[8 words] | m[16 words] |
/// t | (block_len, flags) | out[16 words]`, all words little-endian, two per
/// dword. The accelerator reads dwords 0..14 and writes `out` to dwords 14..22.
/// Using `[u64; 22]` guarantees the 8-byte alignment the ecall requires.
pub fn blake3_compress_6round(state: &mut [u64; 22]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") state.as_mut_ptr(),
            in("a7") BLAKE3_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// BLAKE3 6-round compression via the accelerator (internal variant).
pub fn blake3_compress_6round(_state: &mut [u64; 22]) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

/// Dwords in the BLAKE3 absorb control region: `cv_in[8 words] | cv_out[8]`.
/// Mirrors `BLAKE3_ABSORB_CTRL_DWORDS` in `executor::vm::instruction::execution`.
pub const BLAKE3_ABSORB_CTRL_DWORDS: usize = 8;

#[cfg(target_arch = "riscv64")]
/// Fold whole 64-byte blocks into a BLAKE3 **6-round** chaining value with a
/// single ecall (internal variant — see [`blake3_compress_6round`]).
///
/// `ctrl` is the 64-byte control region as 8 dwords: dwords `0..4` carry the
/// incoming chaining value `cv_in` and the accelerator writes the outgoing one
/// into dwords `4..8`. `blocks` is the message, read in place — no copy, which
/// is the point of this call. `first_flags` is the flag word of the run's first
/// block; every later block carries none.
///
/// The caller must satisfy what the executor checks, or the ecall faults:
/// `blocks.len()` is a nonzero multiple of 64 and at most `2^16` blocks,
/// `blocks` is 8-byte aligned, and `blocks` does not overlap `ctrl`. Using
/// `[u64; 8]` for `ctrl` guarantees its own alignment.
pub fn blake3_absorb(ctrl: &mut [u64; BLAKE3_ABSORB_CTRL_DWORDS], blocks: &[u8], first_flags: u32) {
    unsafe {
        asm!(
            "ecall",
            in("a0") ctrl.as_mut_ptr(),
            in("a1") blocks.as_ptr(),
            in("a2") blocks.len() / 64,
            in("a3") first_flags as usize,
            in("a7") BLAKE3_ABSORB_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// Fold whole 64-byte blocks into a BLAKE3 6-round chaining value in one ecall.
pub fn blake3_absorb(
    _ctrl: &mut [u64; BLAKE3_ABSORB_CTRL_DWORDS],
    _blocks: &[u8],
    _first_flags: u32,
) {
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

/// Whether [`ecrecover`] accepts these operands: the executor's own conditions
/// (its ECREC arm), so a caller that checks this first never stops the VM. Each
/// operand's bytes stay below one `2^32` boundary of its low address word (the
/// 72-byte output, the 64-byte signature, the 32-byte hash), and the signature and
/// the hash do not overlap. The parity is the caller's `recid & 1`, always in range.
#[inline(always)]
pub fn ecrecover_fits(out: &[u8; 72], sig: &[u8; 64], msg: &[u8; 32]) -> bool {
    ecrecover_addresses_ok(
        out.as_ptr() as u64,
        sig.as_ptr() as u64,
        msg.as_ptr() as u64,
    )
}

/// [`ecrecover_fits`] on raw addresses.
#[inline(always)]
pub fn ecrecover_addresses_ok(out: u64, sig: u64, msg: u64) -> bool {
    const TOP: u64 = 1 << 32;
    let below_boundary = |addr: u64, last: u64| (addr & (TOP - 1)) + last < TOP;
    below_boundary(out, 71)
        && below_boundary(sig, 63)
        && below_boundary(msg, 31)
        && !(sig < msg.wrapping_add(32) && msg < sig.wrapping_add(64))
}

#[cfg(target_arch = "riscv64")]
/// Recover the secp256k1 public key of an ECDSA signature via the ECREC accelerator.
/// `sig` is `r ‖ s` and `msg` the message hash, each value 32-byte big-endian (as
/// Ethereum carries them); `parity` is the parity of `R`'s y (the recovery id's low
/// bit, 0 or 1). Writes `out = Q.x ‖ Q.y ‖ status`: the key's coordinates, 32-byte
/// big-endian each, and a little-endian status doubleword, 0 when the call computed the
/// key and 1 when it bailed (the key bytes are then zero and the caller must recover
/// the key another way). `sig` and `msg` must not overlap; `out` may alias either.
pub fn ecrecover(out: &mut [u8; 72], sig: &[u8; 64], msg: &[u8; 32], parity: u64) {
    unsafe {
        asm!(
            "ecall",
            in("a0") out.as_mut_ptr(), // x10 = address to write the key and status
            in("a1") sig.as_ptr(),     // x11 = address of r ‖ s
            in("a2") msg.as_ptr(),     // x12 = address of the message hash
            in("a3") parity,           // x13 = parity of R's y
            in("a7") ECREC_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// Recover the secp256k1 public key of an ECDSA signature via the ECREC accelerator.
pub fn ecrecover(_out: &mut [u8; 72], _sig: &[u8; 64], _msg: &[u8; 32], _parity: u64) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

/// Ask the host for a non-constraining hint (modular inverse/sqrt).
/// `hint_id` selects the operation ([`HINT_FIELD_INV`]/[`HINT_SCALAR_INV`]/
/// [`HINT_FIELD_SQRT`]); `input`/`out` are 32-byte **big-endian** field/scalar
/// elements — k256's own serialization, so consumers pass `to_bytes()` straight
/// through. Note this differs from [`ecsm_mul`], which is little-endian.
/// The result is UNTRUSTED — the caller MUST verify it in-guest (e.g. `x·inv == 1`)
/// AND recompute in software on failure, since this ecall adds no correctness
/// constraint and the prover chooses the returned bytes.
#[cfg(target_arch = "riscv64")]
pub fn hint(hint_id: usize, out: &mut [u8; 32], input: &[u8; 32]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") hint_id,           // x10 = hint selector
            in("a1") input.as_ptr(),    // x11 = input address (32-byte BE)
            in("a2") out.as_mut_ptr(),  // x12 = output address (32-byte BE)
            in("a7") HINT_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
pub fn hint(_hint_id: usize, _out: &mut [u8; 32], _input: &[u8; 32]) {
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

#[cfg(test)]
mod ecrecover_fits_tests {
    use super::ecrecover_addresses_ok as ok;

    #[test]
    fn disjoint_operands_below_the_boundary_fit() {
        assert!(ok(0x1000, 0x2000, 0x3000));
        // The hash right after the signature, and right before it.
        assert!(ok(0x1000, 0x2000, 0x2040));
        assert!(ok(0x1000, 0x2000, 0x1fe0));
        // The output may alias either input.
        assert!(ok(0x2000, 0x2000, 0x3000));
        // High address words are free.
        assert!(ok(0xffff_ffff_0000_1000, 0x7_0000_2000, 0x7_0000_3000));
    }

    #[test]
    fn overlapping_signature_and_hash_do_not_fit() {
        assert!(!ok(0x1000, 0x2000, 0x2000));
        assert!(!ok(0x1000, 0x2000, 0x203f));
        assert!(!ok(0x1000, 0x2000, 0x1fe1));
    }

    #[test]
    fn an_operand_across_a_low_word_boundary_does_not_fit() {
        const TOP: u64 = 1 << 32;
        // The last byte exactly at 2^32 − 1 fits; one further does not.
        assert!(ok(TOP - 72, 0x2000, 0x3000));
        assert!(!ok(TOP - 71, 0x2000, 0x3000));
        assert!(ok(0x1000, TOP - 64, 0x3000));
        assert!(!ok(0x1000, TOP - 63, 0x3000));
        assert!(ok(0x1000, 0x2000, TOP - 32));
        assert!(!ok(0x1000, 0x2000, TOP - 31));
        // The same at a nonzero high word.
        assert!(!ok(0x1000, 0x2000, 5 * TOP - 31));
    }
}
