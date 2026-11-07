//! Tests for the Keccak-f[1600] permutation and the Keccak syscall.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{ExecutionError, KECCAK_SYSCALL_NUMBER, keccak_f1600};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;

#[test]
fn test_keccak_f1600_zero_input() {
    let mut state = [0u64; 25];
    keccak_f1600(&mut state);

    let expected: [u64; 25] = [
        0xF1258F7940E1DDE7,
        0x84D5CCF933C0478A,
        0xD598261EA65AA9EE,
        0xBD1547306F80494D,
        0x8B284E056253D057,
        0xFF97A42D7F8E6FD4,
        0x90FEE5A0A44647C4,
        0x8C5BDA0CD6192E76,
        0xAD30A6F71B19059C,
        0x30935AB7D08FFC64,
        0xEB5AA93F2317D635,
        0xA9A6E6260D712103,
        0x81A57C16DBCF555F,
        0x43B831CD0347C826,
        0x01F22F1A11A5569F,
        0x05E5635A21D9AE61,
        0x64BEFEF28CC970F2,
        0x613670957BC46611,
        0xB87C5A554FD00ECB,
        0x8C3EE88A1CCF32C8,
        0x940C7922AE3A2614,
        0x1841F924A2C509E4,
        0x16F53526E70465C2,
        0x75F644E97F30A13B,
        0xEAF1FF7B5CECA249,
    ];

    assert_eq!(state, expected, "keccak-f[1600] on zero input mismatch");
}

#[test]
fn test_keccak_f1600_nonzero_input() {
    let mut state = [0u64; 25];
    state[0] = 1;
    let original = state;
    keccak_f1600(&mut state);
    assert_ne!(state, original);
    assert!(state.iter().any(|&x| x != 0));
}

#[test]
fn test_keccak_syscall_rejects_unaligned_state_addr() {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();

    registers.write(17, KECCAK_SYSCALL_NUMBER).unwrap();
    registers.write(10, 0x1001).unwrap();

    let err = Instruction::EcallEbreak
        .run(&mut pc, &mut registers, &mut memory)
        .unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::UnalignedKeccakStateAddress(0x1001)
    ));
}

#[test]
fn test_keccak_syscall_rejects_overflowing_state_range() {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();

    registers.write(17, KECCAK_SYSCALL_NUMBER).unwrap();
    registers.write(10, u64::MAX - 191).unwrap();

    let err = Instruction::EcallEbreak
        .run(&mut pc, &mut registers, &mut memory)
        .unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::KeccakStateAddressOverflow(addr) if addr == u64::MAX - 191
    ));
}

// =============================================================================
// Keccak-256 of a guest buffer (KECCAK256_SYSCALL_NUMBER)
// =============================================================================

mod keccak256_call {
    use crate::vm::instruction::decoding::Instruction;
    use crate::vm::instruction::execution::{
        ExecutionError, KECCAK256_RATE_BYTES, KECCAK256_SYSCALL_NUMBER, keccak_f1600,
        keccak256_addresses_ok, keccak256_blocks, keccak256_digest, keccak256_padded_block,
        keccak256_permutations,
    };
    use crate::vm::memory::Memory;
    use crate::vm::registers::Registers;
    use tiny_keccak::{Hasher, Keccak};

    fn reference(input: &[u8]) -> [u8; 32] {
        let mut hasher = Keccak::v256();
        hasher.update(input);
        let mut out = [0u8; 32];
        hasher.finalize(&mut out);
        out
    }

    /// Deterministic, non-repeating bytes.
    fn bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    /// Runs the ecall over `input` placed at `ptr`, digest at `out`; returns the
    /// digest read back and the log's `(src2_val, dst_val)`.
    fn run_call(input: &[u8], ptr: u64, out: u64) -> Result<([u8; 32], u64, u64), ExecutionError> {
        let mut pc = 0;
        let mut registers = Registers::default();
        let mut memory = Memory::default();
        for (i, b) in input.iter().enumerate() {
            memory.store_byte(ptr + i as u64, *b);
        }
        registers.write(17, KECCAK256_SYSCALL_NUMBER).unwrap();
        registers.write(10, ptr).unwrap();
        registers.write(11, input.len() as u64).unwrap();
        registers.write(12, out).unwrap();
        let log = Instruction::EcallEbreak.run(&mut pc, &mut registers, &mut memory)?;
        assert_eq!(pc, 4, "the call falls through to the next instruction");
        assert_eq!(registers.read(10).unwrap(), ptr, "x10 unchanged");
        assert_eq!(
            registers.read(11).unwrap(),
            input.len() as u64,
            "x11 unchanged"
        );
        assert_eq!(registers.read(12).unwrap(), out, "x12 unchanged");
        let digest: [u8; 32] = memory.load_bytes(out, 32).unwrap().try_into().unwrap();
        Ok((digest, log.src2_val, log.dst_val))
    }

    /// The lengths the padding cases turn on, plus multi-block and multi-KB.
    const LENGTHS: &[usize] = &[
        0, 1, 7, 8, 9, 31, 32, 33, 64, 127, 128, 134, 135, 136, 137, 143, 144, 200, 270, 271, 272,
        273, 407, 408, 409, 1000, 4096, 10_000,
    ];

    #[test]
    fn digest_matches_the_reference_at_every_padding_case() {
        for &len in LENGTHS {
            let input = bytes(len, len as u64 + 1);
            assert_eq!(keccak256_digest(&input), reference(&input), "len {len}");
        }
        // Every length to three blocks and a bit.
        for len in 0..450 {
            let input = bytes(len, 7);
            assert_eq!(keccak256_digest(&input), reference(&input), "len {len}");
        }
    }

    #[test]
    fn the_call_writes_the_reference_digest_at_any_alignment() {
        for &len in LENGTHS {
            for ptr_off in 0..8u64 {
                for out_off in [0u64, 3, 5] {
                    let input = bytes(len, 31 * len as u64 + ptr_off);
                    let ptr = 0x10_000 + ptr_off;
                    let out = 0x40_000 + out_off;
                    let (digest, blocks, logged_ptr) = run_call(&input, ptr, out).unwrap();
                    assert_eq!(
                        digest,
                        reference(&input),
                        "len {len} ptr+{ptr_off} out+{out_off}"
                    );
                    assert_eq!(blocks, len as u64 / 136 + 1, "logged permutation count");
                    assert_eq!(logged_ptr, ptr);
                }
            }
        }
    }

    #[test]
    fn the_digest_may_overwrite_its_own_input() {
        for len in [0usize, 5, 32, 135, 136, 300] {
            let input = bytes(len, 99);
            let ptr = 0x2_0003u64;
            // Output starting inside the input (or right at it).
            let (digest, _, _) = run_call(&input, ptr, ptr + (len as u64) / 2).unwrap();
            assert_eq!(
                digest,
                reference(&input),
                "len {len}: hash of the bytes before the call"
            );
        }
    }

    #[test]
    fn permutations_chain_the_padded_blocks() {
        for &len in LENGTHS {
            let input = bytes(len, 3 * len as u64);
            let perms = keccak256_permutations(&input);
            assert_eq!(perms.len() as u64, keccak256_blocks(len as u64));
            let mut state = [0u64; 25];
            for (j, (start, end)) in perms.iter().enumerate() {
                let block = keccak256_padded_block(&input, j);
                let mut want = state;
                for (lane, chunk) in want.iter_mut().zip(block.chunks_exact(8)) {
                    *lane ^= u64::from_le_bytes(chunk.try_into().unwrap());
                }
                assert_eq!(*start, want, "len {len} block {j} input");
                keccak_f1600(&mut want);
                assert_eq!(*end, want, "len {len} block {j} output");
                state = *end;
            }
        }
    }

    #[test]
    fn padded_last_block_cases() {
        let rate = KECCAK256_RATE_BYTES as usize;
        // len % 136 == 0: the last block is padding alone.
        let b = keccak256_padded_block(&[0xAA; 136], 1);
        assert_eq!(b[0], 0x01);
        assert_eq!(b[rate - 1], 0x80);
        assert!(b[1..rate - 1].iter().all(|&x| x == 0));
        // len % 136 == 135: 0x01 and 0x80 meet in byte 135.
        let b = keccak256_padded_block(&[0xAA; 135], 0);
        assert!(b[..135].iter().all(|&x| x == 0xAA));
        assert_eq!(b[135], 0x81);
        // A full block of a longer input is unpadded.
        let b = keccak256_padded_block(&[0x55; 300], 1);
        assert!(b.iter().all(|&x| x == 0x55));
    }

    #[test]
    fn address_conditions_match_the_chip() {
        let top = 1u64 << 32;
        // The last block's window ends exactly at the boundary: accepted.
        assert!(keccak256_addresses_ok(top - 136, 0, 0));
        assert!(keccak256_addresses_ok(top - 272, 136, 0));
        assert!(keccak256_addresses_ok(top - 272, 271, 0));
        // One byte further: refused, even though the input itself ends earlier.
        assert!(!keccak256_addresses_ok(top - 135, 0, 0));
        assert!(!keccak256_addresses_ok(top - 271, 136, 0));
        // Only the low word matters.
        assert!(keccak256_addresses_ok(5 * top + 0x1000, 100, 7 * top));
        // Lengths of 2^32 or more never fit.
        assert!(!keccak256_addresses_ok(0, top, 0));
        assert!(!keccak256_addresses_ok(0, u64::MAX, 0));
        // Output: the 32 bytes stay below the boundary.
        assert!(keccak256_addresses_ok(0, 0, top - 32));
        assert!(!keccak256_addresses_ok(0, 0, top - 31));
    }

    #[test]
    fn the_call_refuses_operands_across_a_word_boundary() {
        let top = 1u64 << 32;
        for (ptr, len, out) in [
            (top - 135, 0u64, 0x1000u64),
            (top - 100, 99, 0x1000),
            (0x1000, 10, top - 31),
            (0x1000, top, 0x2000),
            (0x1000, u64::MAX, 0x2000),
        ] {
            let mut pc = 0;
            let mut registers = Registers::default();
            let mut memory = Memory::default();
            registers.write(17, KECCAK256_SYSCALL_NUMBER).unwrap();
            registers.write(10, ptr).unwrap();
            registers.write(11, len).unwrap();
            registers.write(12, out).unwrap();
            let err = Instruction::EcallEbreak
                .run(&mut pc, &mut registers, &mut memory)
                .unwrap_err();
            assert!(
                matches!(err, ExecutionError::Keccak256AddressOverflow { .. }),
                "ptr {ptr:#x} len {len} out {out:#x}: {err:?}"
            );
        }
    }
}
