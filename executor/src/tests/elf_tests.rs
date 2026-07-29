use crate::elf::*;
use crate::vm::memory::{MAX_PRIVATE_INPUT_SIZE, PRIVATE_INPUT_START_INDEX};

/// Build a minimal valid RISC-V ET_EXEC ELF with a single PT_LOAD segment at
/// `p_vaddr` of `p_memsz` bytes (all BSS: `p_filesz = 0`). Enough for `Elf::load`,
/// which only parses the executable header + program headers.
fn minimal_elf_with_segment(p_vaddr: u64, p_memsz: u64) -> Vec<u8> {
    let mut buf = vec![0u8; EXECUTABLE_HEADER_SIZE + PROGRAM_HEADER_SIZE];
    // e_ident
    buf[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
    buf[4] = ELF_64_BIT;
    buf[5] = ELF_LITTLE_ENDIAN;
    buf[6] = ELF_CURRENT_VERSION;
    buf[16..18].copy_from_slice(&ET_EXEC.to_le_bytes());
    buf[18..20].copy_from_slice(&EM_RISCV.to_le_bytes());
    buf[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
    buf[24..32].copy_from_slice(&0x10000u64.to_le_bytes()); // e_entry (word-aligned)
    buf[32..40].copy_from_slice(&(EXECUTABLE_HEADER_SIZE as u64).to_le_bytes()); // e_phoff
    buf[52..54].copy_from_slice(&(EXECUTABLE_HEADER_SIZE as u16).to_le_bytes()); // e_ehsize
    buf[54..56].copy_from_slice(&(PROGRAM_HEADER_SIZE as u16).to_le_bytes()); // e_phentsize
    buf[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum
    // single program header
    let ph = EXECUTABLE_HEADER_SIZE;
    buf[ph..ph + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
    buf[ph + 4..ph + 8].copy_from_slice(&PF_X.to_le_bytes());
    buf[ph + 16..ph + 24].copy_from_slice(&p_vaddr.to_le_bytes()); // p_vaddr
    buf[ph + 40..ph + 48].copy_from_slice(&p_memsz.to_le_bytes()); // p_memsz
    buf
}

#[test]
fn rejects_segment_inside_private_input_region() {
    // An ELF data segment placed inside the reserved region would get a prover-chosen,
    // ELF-unbound genesis (private-input pages are non-preprocessed) — must be rejected.
    let elf = minimal_elf_with_segment(PRIVATE_INPUT_START_INDEX, 4);
    assert!(matches!(
        Elf::load(&elf),
        Err(ElfError::SegmentInPrivateInputRegion)
    ));
}

#[test]
fn rejects_segment_with_overflowing_vaddr_span() {
    // p_vaddr + p_memsz overflows u64 → rejected explicitly as AddrTooLarge (not
    // saturated to u64::MAX and mis-reported, and no panic/wrap).
    let elf = minimal_elf_with_segment(0xFFFF_FFFF_FFFF_F000, 0x2000);
    assert!(matches!(Elf::load(&elf), Err(ElfError::AddrTooLarge)));
}

#[test]
fn rejects_segment_straddling_region_start() {
    // Ends 4 bytes into the region → overlaps → rejected.
    let elf = minimal_elf_with_segment(PRIVATE_INPUT_START_INDEX - 4, 8);
    assert!(matches!(
        Elf::load(&elf),
        Err(ElfError::SegmentInPrivateInputRegion)
    ));
}

#[test]
fn accepts_segment_below_region() {
    assert!(Elf::load(&minimal_elf_with_segment(0x10000, 4)).is_ok());
}

#[test]
fn accepts_segment_ending_exactly_at_region_start() {
    // seg_end == PRIVATE_INPUT_START_INDEX (exclusive) → no overlap → accepted.
    let elf = minimal_elf_with_segment(PRIVATE_INPUT_START_INDEX - 4, 4);
    assert!(Elf::load(&elf).is_ok());
}

#[test]
fn rejects_segment_at_max_size_boundary() {
    // The `[base, base+MAX)` byte cap ends here, but an honest max-size input (plus its
    // 4-byte length prefix) spills onto this page, so the verifier can classify it private.
    // It must therefore be rejected too — the reservation covers the full classifiable
    // span, not just `[base, base+MAX)`.
    let boundary = PRIVATE_INPUT_START_INDEX + MAX_PRIVATE_INPUT_SIZE;
    assert!(matches!(
        Elf::load(&minimal_elf_with_segment(boundary, 4)),
        Err(ElfError::SegmentInPrivateInputRegion)
    ));
}

#[test]
fn rejects_segment_far_above_region() {
    // Any segment reaching at/above the private-input base is rejected — nothing
    // legitimate loads that high (ELF is low; stack/private input are runtime).
    let high = PRIVATE_INPUT_START_INDEX + MAX_PRIVATE_INPUT_SIZE + (16 << 20);
    assert!(matches!(
        Elf::load(&minimal_elf_with_segment(high, 4)),
        Err(ElfError::SegmentInPrivateInputRegion)
    ));
}
