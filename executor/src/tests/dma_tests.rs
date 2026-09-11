use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{
    DMA_MEMCPY_MAX_BYTES, DMA_MEMCPY_SYSCALL_NUMBER, ExecutionError, dma_memcpy_data_rows,
    dma_memcpy_trace_rows,
};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;
use proptest::prelude::*;

fn run_dma(memory: &mut Memory, dst: u64, src: u64, count: u64) -> Result<(), ExecutionError> {
    let mut registers = Registers::default();
    let mut pc = 0;
    registers.write(17, DMA_MEMCPY_SYSCALL_NUMBER)?;
    registers.write(10, dst)?;
    registers.write(11, src)?;
    registers.write(12, count)?;
    Instruction::EcallEbreak.run(&mut pc, &mut registers, memory)?;
    Ok(())
}

#[test]
fn dma_memcpy_copies_unaligned_body_and_tail() {
    let mut memory = Memory::default();
    let input: Vec<u8> = (0..27).map(|i| (i * 7 + 3) as u8).collect();
    for (i, &byte) in input.iter().enumerate() {
        memory.store_byte(0x1003 + i as u64, byte);
    }

    run_dma(&mut memory, 0x2005, 0x1003, input.len() as u64).unwrap();
    assert_eq!(
        memory.load_bytes(0x2005, input.len() as u64).unwrap(),
        input
    );
}

#[test]
fn dma_memcpy_has_snapshot_semantics_for_overlap() {
    let mut memory = Memory::default();
    let input: Vec<u8> = (0..32).map(|i| i as u8).collect();
    for (i, &byte) in input.iter().enumerate() {
        memory.store_byte(0x3000 + i as u64, byte);
    }

    run_dma(&mut memory, 0x3004, 0x3000, 24).unwrap();
    assert_eq!(
        memory.load_bytes(0x3004, 24).unwrap(),
        input[..24],
        "overlap must read the complete source snapshot before writing"
    );
}

#[test]
fn dma_memcpy_rejects_wrapping_ranges() {
    let mut memory = Memory::default();
    assert!(run_dma(&mut memory, 0x1000, u64::MAX - 3, 8).is_err());
    assert!(run_dma(&mut memory, u64::MAX - 3, 0x1000, 8).is_err());
}

#[test]
fn dma_memcpy_rejects_oversized_direct_ecall() {
    let mut memory = Memory::default();
    assert!(matches!(
        run_dma(
            &mut memory,
            0x2000,
            0x1000,
            DMA_MEMCPY_MAX_BYTES + 1
        ),
        Err(ExecutionError::DmaMemcpyChunkTooLarge(n))
            if n == DMA_MEMCPY_MAX_BYTES + 1
    ));
}

/// The row helpers are what the trace builder sizes the DMA trace with and what
/// the CLI reports as the accelerator's cost, so pin them to the chunking rule
/// the trace builder actually walks rather than to the closed form itself.
#[test]
fn dma_row_helpers_match_the_chunk_loop() {
    for count in 0..=DMA_MEMCPY_MAX_BYTES {
        let mut chunks = 0u64;
        let mut remaining = count;
        while remaining != 0 {
            remaining -= if remaining >= 8 { 8 } else { 1 };
            chunks += 1;
        }

        assert_eq!(dma_memcpy_data_rows(count), chunks, "count {count}");
        assert_eq!(
            dma_memcpy_trace_rows(count),
            chunks + 1,
            "count {count}: the terminal row is always emitted"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Differentially compare the DMA snapshot semantics against a byte-vector
    /// oracle. The generated ranges cover unaligned copies, both overlap
    /// directions, zero/small/tail lengths, full chunks, and page crossings.
    #[test]
    fn dma_memcpy_matches_snapshot_oracle(
        src_offset in 0usize..768,
        dst_offset in 0usize..768,
        count in 0usize..=DMA_MEMCPY_MAX_BYTES as usize,
        seed in any::<u64>(),
    ) {
        const BASE: u64 = 0x0F00;
        const REGION: usize = 1024;

        let mut initial = vec![0u8; REGION];
        let mut state = seed;
        for byte in &mut initial {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }

        let mut expected = initial.clone();
        let snapshot = expected[src_offset..src_offset + count].to_vec();
        expected[dst_offset..dst_offset + count].copy_from_slice(&snapshot);

        let mut memory = Memory::default();
        for (i, &byte) in initial.iter().enumerate() {
            memory.store_byte(BASE + i as u64, byte);
        }
        run_dma(
            &mut memory,
            BASE + dst_offset as u64,
            BASE + src_offset as u64,
            count as u64,
        )
        .unwrap();

        let actual = memory.load_bytes(BASE, REGION as u64).unwrap();
        prop_assert_eq!(actual, expected);
    }
}
