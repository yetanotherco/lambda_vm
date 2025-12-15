use std::collections::BTreeMap;

use elf::{
    ElfBytes,
    abi::{EM_RISCV, ET_EXEC, PT_LOAD},
    endian::LittleEndian,
    file::Class,
};

#[derive(Debug)]
pub struct ExecutableHeader {
    e_ident: [u8;16],	/* Magic number and other info */
    e_type: u16,		/* Object file type */
    e_machine: u16,        /* Architecture */
    e_version: u32,		/* Object file version */
    e_entry: u32,		/* Entry point virtual address */
    e_phoff: u32,		/* Program header table file offset */
    e_shoff: u32,		/* Section header table file offset */
    e_flags: u32,		/* Processor-specific flags */
    e_ehsize: u16,		/* ELF header size in bytes */
    e_phentsize: u16,		/* Program header table entry size */
    e_phnum: u16,		/* Program header table entry count */
    e_shentsize: u16,		/* Section header table entry size */
    e_shnum: u16,		/* Section header table entry count */
    e_shstrndx: u16,		/* Section header string table index */
}

#[derive(Debug)]
pub struct ElfProgram {
    pub ehdr: ExecutableHeader,
}

impl ElfProgram {
    pub fn parse(input: &[u8]) -> Result<Self, ElfError> {
        let ehdr = ExecutableHeader::parse(&input)?;
        Ok(Self { ehdr })
    }
}

impl ExecutableHeader {
    pub fn parse(input: &[u8]) -> Result<Self, ElfError> {
        if input.len() < 52 {
            panic!("Input too short to be a valid ELF header");
        }
        let e_ident: [u8; 16]  = input[0..16].try_into().unwrap();
        if e_ident[0] != 0x7F
            || e_ident[1] != b'E'
            || e_ident[2] != b'L'
            || e_ident[3] != b'F'
        {
            panic!("Invalid ELF magic number");
        }
        if e_ident[4] != 1 {
            panic!("Not a 32-bit ELF");
        }
        if e_ident[5] != 1 {
            panic!("Not a little-endian ELF");
        }
        if e_ident[6] != 1 {
            panic!("Invalid ELF Version");
        }
        let e_type = u16::from_le_bytes(input[16..18].try_into().unwrap());
        let e_machine = u16::from_le_bytes(input[18..20].try_into().unwrap());
        let e_version = u32::from_le_bytes(input[20..24].try_into().unwrap());
        let e_entry = u32::from_le_bytes(input[24..28].try_into().unwrap());
        let e_phoff = u32::from_le_bytes(input[28..32].try_into().unwrap());
        let e_shoff = u32::from_le_bytes(input[32..36].try_into().unwrap());
        let e_flags = u32::from_le_bytes(input[36..40].try_into().unwrap());
        let e_ehsize = u16::from_le_bytes(input[40..42].try_into().unwrap());
        let e_phentsize = u16::from_le_bytes(input[42..44].try_into().unwrap());
        let e_phnum = u16::from_le_bytes(input[44..46].try_into().unwrap());
        let e_shentsize = u16::from_le_bytes(input[46..48].try_into().unwrap());
        let e_shnum = u16::from_le_bytes(input[48..50].try_into().unwrap());
        let e_shstrndx = u16::from_le_bytes(input[50..52].try_into().unwrap());
        Ok(Self {
            e_ident,
            e_type,
            e_machine,
            e_version,
            e_entry,
            e_phoff,
            e_shoff,
            e_flags,
            e_ehsize,
            e_phentsize,
            e_phnum,
            e_shentsize,
            e_shnum,
            e_shstrndx,
        })
    }
}

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
        let elf_program = ElfProgram::parse(input)?;
        println!("ELF Program Header: {:?}", elf_program.ehdr);
        println!("Elfbytes Header: {:?}", elf.ehdr);
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
