use std::collections::BTreeMap;

use elf::{
    ElfBytes,
    abi::{EM_RISCV, ET_EXEC, PT_LOAD},
    endian::LittleEndian,
    file::Class,
};

pub struct Elf {
    pub entry_point: u32,

    pub image: BTreeMap<u32, u32>,
}
pub(crate) const WORD_SIZE: u32 = 4;
pub const MAX_MEMORY_SIZE: u32 = u32::MAX;
pub const MAX_SEGMENTS: usize = 256;

#[derive(Debug, thiserror::Error)]
pub enum ElfError {
    #[error(transparent)]
    Parse(#[from] elf::ParseError),
    #[error("Not a 32-bit ELF")]
    Not32Bit,
    #[error("Not a RISC-V ELF")]
    NotRiscV,
    #[error("ELF is not executable")]
    NotExecutable,
    #[error("Entrypoint is invalid")]
    InvalidEntryPoint,
    #[error("ELF has no segments")]
    NoSegments,
    #[error("ELF has too many segments")]
    TooManySegments,
    #[error("Segment file size is too large")]
    FileSizeTooLarge,
    #[error("Segment memory size is too large")]
    MemSizeTooLarge,
    #[error("Segment virtual address is too large")]
    VAddrTooLarge,
    #[error("Segment virtual address is unaligned")]
    UnalignedVAddr,
    #[error("Segment offset is too large")]
    OffsetTooLarge,
    #[error("Segment address is too large")]
    AddrTooLarge,
    #[error("Segment offset is invalid")]
    InvalidOffset,
}

impl Elf {
    pub fn load(input: &[u8]) -> Result<Elf, ElfError> {
        let mut image: BTreeMap<u32, u32> = BTreeMap::new();
        let elf = ElfBytes::<LittleEndian>::minimal_parse(input)?;
        if elf.ehdr.class != Class::ELF32 {
            return Err(ElfError::Not32Bit);
        }
        if elf.ehdr.e_machine != EM_RISCV {
            return Err(ElfError::NotRiscV);
        }
        if elf.ehdr.e_type != ET_EXEC {
            return Err(ElfError::NotExecutable);
        }
        let entry_point: u32 = elf
            .ehdr
            .e_entry
            .try_into()
            .map_err(|_| ElfError::InvalidEntryPoint)?;
        if !entry_point.is_multiple_of(WORD_SIZE) {
            return Err(ElfError::InvalidEntryPoint);
        }
        let segments = elf.segments().ok_or(ElfError::NoSegments)?;
        if segments.len() > MAX_SEGMENTS {
            return Err(ElfError::TooManySegments);
        }
        for segment in segments.iter().filter(|segment| segment.p_type == PT_LOAD) {
            let file_size: u32 = segment
                .p_filesz
                .try_into()
                .map_err(|_| ElfError::FileSizeTooLarge)?;
            let mem_size: u32 = segment
                .p_memsz
                .try_into()
                .map_err(|_| ElfError::MemSizeTooLarge)?;
            let vaddr: u32 = segment
                .p_vaddr
                .try_into()
                .map_err(|_| ElfError::VAddrTooLarge)?;
            if !vaddr.is_multiple_of(WORD_SIZE) {
                return Err(ElfError::UnalignedVAddr);
            }
            let offset: u32 = segment
                .p_offset
                .try_into()
                .map_err(|_| ElfError::OffsetTooLarge)?;
            for i in (0..mem_size).step_by(WORD_SIZE as usize) {
                let addr = vaddr.checked_add(i).ok_or(ElfError::AddrTooLarge)?;
                if i >= file_size {
                    image.insert(addr, 0);
                } else {
                    let mut word = 0;
                    let len = (file_size - i).min(WORD_SIZE);
                    for j in 0..len {
                        let offset = (offset + i + j) as usize;
                        let byte = input.get(offset).ok_or(ElfError::InvalidOffset)?;
                        word |= (*byte as u32) << (j * 8);
                    }
                    image.insert(addr, word);
                }
            }
        }
        Ok(Self { entry_point, image })
    }
}
