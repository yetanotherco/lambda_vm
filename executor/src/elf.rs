use std::collections::BTreeMap;

const EI_NIDENT: usize = 16;
const EM_RISCV: u16 = 243;
const ET_EXEC: u16 = 2;
const PT_LOAD: u32 = 1;
const MAX_PROGRAM_HEADERS: usize = 256;
const EXECUTABLE_HEADER_SIZE: usize = 52;
const PROGRAM_HEADER_SIZE: usize = 32;
const ELF_32_BIT: u8 = 1;
const ELF_LITTLE_ENDIAN: u8 = 1;
const ELF_CURRENT_VERSION: u8 = 1;

#[derive(Debug)]
pub struct ExecutableHeader {
    /// Magic number and other info
    _e_ident: [u8; EI_NIDENT],
    /// Object file type
    e_type: u16,
    /// Architecture
    e_machine: u16,
    /// Object file version
    _e_version: u32,
    /// Entry point virtual address
    e_entry: u32,
    /// Program header table file offset
    e_phoff: u32,
    /// Section header table file offset
    _e_shoff: u32,
    /// Processor-specific flags
    _e_flags: u32,
    /// ELF header size in bytes
    _e_ehsize: u16,
    /// Program header table entry size
    e_phentsize: u16,
    /// Program header table entry count
    e_phnum: u16,
    /// Section header table entry size
    _e_shentsize: u16,
    /// Section header table entry count
    _e_shnum: u16,
    /// Section header string table index
    _e_shstrndx: u16,
}

#[derive(Debug)]
pub struct ProgramHeader {
    /// Segment type
    p_type: u32,
    /// Segment file offset
    p_offset: u32,
    /// Segment virtual address
    p_vaddr: u32,
    /// Segment physical address
    _p_paddr: u32,
    /// Segment size in file
    p_filesz: u32,
    /// Segment size in memory
    p_memsz: u32,
    /// Segment flags
    _p_flags: u32,
    /// Segment alignment
    _p_align: u32,
}

#[derive(Debug)]
pub struct ElfProgram {
    pub ehdr: ExecutableHeader,
    pub phdrs: Vec<ProgramHeader>,
}

impl ElfProgram {
    pub fn parse(input: &[u8]) -> Result<Self, ElfError> {
        let ehdr = ExecutableHeader::parse(input)?;
        let phdrs = Self::parse_phdrs(input, &ehdr)?;
        Ok(Self { ehdr, phdrs })
    }

    fn parse_phdrs(input: &[u8], ehdr: &ExecutableHeader) -> Result<Vec<ProgramHeader>, ElfError> {
        let mut phdrs = Vec::new();
        let phoff = ehdr.e_phoff as usize;
        let phentsize = ehdr.e_phentsize as usize;
        let phnum = ehdr.e_phnum as usize;
        for i in 0..phnum {
            let offset = phoff
                .checked_add(i.checked_mul(phentsize).ok_or(ElfError::InvalidProgram)?)
                .ok_or(ElfError::InvalidProgram)?;
            let phdr = ProgramHeader::parse(
                &input[offset
                    ..offset
                        .checked_add(phentsize)
                        .ok_or(ElfError::InvalidProgram)?],
            )?;
            phdrs.push(phdr);
        }
        Ok(phdrs)
    }
}

impl ExecutableHeader {
    pub fn parse(input: &[u8]) -> Result<Self, ElfError> {
        if input.len() < EXECUTABLE_HEADER_SIZE {
            return Err(ElfError::ExecutableHeaderSize);
        }
        let e_ident: [u8; EI_NIDENT] = input[0..EI_NIDENT]
            .try_into()
            .map_err(|_| ElfError::Casting)?;
        if e_ident[0] != 0x7F || e_ident[1] != b'E' || e_ident[2] != b'L' || e_ident[3] != b'F' {
            return Err(ElfError::InvalidELFMagicNumber);
        }
        if e_ident[4] != ELF_32_BIT {
            return Err(ElfError::Not32Bit);
        }
        if e_ident[5] != ELF_LITTLE_ENDIAN {
            return Err(ElfError::NotLittleEndian);
        }
        if e_ident[6] != ELF_CURRENT_VERSION {
            return Err(ElfError::InvalidElfVersion);
        }
        let e_type = u16::from_le_bytes(
            input[EI_NIDENT..18]
                .try_into()
                .map_err(|_| ElfError::Casting)?,
        );
        let e_machine =
            u16::from_le_bytes(input[18..20].try_into().map_err(|_| ElfError::Casting)?);
        let e_version =
            u32::from_le_bytes(input[20..24].try_into().map_err(|_| ElfError::Casting)?);
        let e_entry = u32::from_le_bytes(input[24..28].try_into().map_err(|_| ElfError::Casting)?);
        let e_phoff = u32::from_le_bytes(input[28..32].try_into().map_err(|_| ElfError::Casting)?);
        let e_shoff = u32::from_le_bytes(input[32..36].try_into().map_err(|_| ElfError::Casting)?);
        let e_flags = u32::from_le_bytes(input[36..40].try_into().map_err(|_| ElfError::Casting)?);
        let e_ehsize = u16::from_le_bytes(input[40..42].try_into().map_err(|_| ElfError::Casting)?);
        let e_phentsize =
            u16::from_le_bytes(input[42..44].try_into().map_err(|_| ElfError::Casting)?);
        let e_phnum = u16::from_le_bytes(input[44..46].try_into().map_err(|_| ElfError::Casting)?);
        let e_shentsize =
            u16::from_le_bytes(input[46..48].try_into().map_err(|_| ElfError::Casting)?);
        let e_shnum = u16::from_le_bytes(input[48..50].try_into().map_err(|_| ElfError::Casting)?);
        let e_shstrndx =
            u16::from_le_bytes(input[50..52].try_into().map_err(|_| ElfError::Casting)?);
        Ok(Self {
            _e_ident: e_ident,
            e_type,
            e_machine,
            _e_version: e_version,
            e_entry,
            e_phoff,
            _e_shoff: e_shoff,
            _e_flags: e_flags,
            _e_ehsize: e_ehsize,
            e_phentsize,
            e_phnum,
            _e_shentsize: e_shentsize,
            _e_shnum: e_shnum,
            _e_shstrndx: e_shstrndx,
        })
    }
}

impl ProgramHeader {
    pub fn parse(input: &[u8]) -> Result<Self, ElfError> {
        if input.len() < PROGRAM_HEADER_SIZE {
            return Err(ElfError::ProgramHeaderSize);
        }
        let p_type = u32::from_le_bytes(input[0..4].try_into().map_err(|_| ElfError::Casting)?);
        let p_offset = u32::from_le_bytes(input[4..8].try_into().map_err(|_| ElfError::Casting)?);
        let p_vaddr = u32::from_le_bytes(input[8..12].try_into().map_err(|_| ElfError::Casting)?);
        let p_paddr = u32::from_le_bytes(input[12..16].try_into().map_err(|_| ElfError::Casting)?);
        let p_filesz = u32::from_le_bytes(input[16..20].try_into().map_err(|_| ElfError::Casting)?);
        let p_memsz = u32::from_le_bytes(input[20..24].try_into().map_err(|_| ElfError::Casting)?);
        let p_flags = u32::from_le_bytes(input[24..28].try_into().map_err(|_| ElfError::Casting)?);
        let p_align = u32::from_le_bytes(input[28..32].try_into().map_err(|_| ElfError::Casting)?);
        Ok(Self {
            p_type,
            p_offset,
            p_vaddr,
            _p_paddr: p_paddr,
            p_filesz,
            p_memsz,
            _p_flags: p_flags,
            _p_align: p_align,
        })
    }
}
pub struct Elf {
    pub entry_point: u32,

    pub image: BTreeMap<u32, u32>,
}

pub(crate) const WORD_SIZE: u32 = 4;

#[derive(Debug, thiserror::Error)]
pub enum ElfError {
    #[error("Not a 32-bit ELF")]
    Not32Bit,
    #[error("Not a RISC-V ELF")]
    NotRiscV,
    #[error("ELF is not executable")]
    NotExecutable,
    #[error("Entrypoint is invalid")]
    InvalidEntryPoint,
    #[error("ELF has too many program headers")]
    TooManyProgramHeaders,
    #[error("Program Header virtual address is unaligned")]
    UnalignedVAddr,
    #[error("Program Header address is too large")]
    AddrTooLarge,
    #[error("Program Header offset is invalid")]
    InvalidOffset,
    #[error("Executable Header size is invalid")]
    ExecutableHeaderSize,
    #[error("Invalid ELF magic number")]
    InvalidELFMagicNumber,
    #[error("ELF is not little endian")]
    NotLittleEndian,
    #[error("Invalid ELF version")]
    InvalidElfVersion,
    #[error("Failed to cast")]
    Casting,
    #[error("Program Header size is invalid")]
    ProgramHeaderSize,
    #[error("Invalid program")]
    InvalidProgram,
}

impl Elf {
    pub fn load(input: &[u8]) -> Result<Elf, ElfError> {
        let mut image: BTreeMap<u32, u32> = BTreeMap::new();
        let elf_program = ElfProgram::parse(input)?;
        if elf_program.ehdr.e_machine != EM_RISCV {
            return Err(ElfError::NotRiscV);
        }
        if elf_program.ehdr.e_type != ET_EXEC {
            return Err(ElfError::NotExecutable);
        }
        let entry_point: u32 = elf_program.ehdr.e_entry;
        if !entry_point.is_multiple_of(WORD_SIZE) {
            return Err(ElfError::InvalidEntryPoint);
        }
        let phdrs = elf_program.phdrs;
        if phdrs.len() > MAX_PROGRAM_HEADERS {
            return Err(ElfError::TooManyProgramHeaders);
        }
        for program_header in phdrs
            .iter()
            .filter(|program_header| program_header.p_type == PT_LOAD)
        {
            if !program_header.p_vaddr.is_multiple_of(WORD_SIZE) {
                return Err(ElfError::UnalignedVAddr);
            }
            for i in (0..program_header.p_memsz).step_by(WORD_SIZE as usize) {
                let addr = program_header
                    .p_vaddr
                    .checked_add(i)
                    .ok_or(ElfError::AddrTooLarge)?;
                if i >= program_header.p_filesz {
                    image.insert(addr, 0);
                } else {
                    let mut word = 0;
                    let len = (program_header
                        .p_filesz
                        .checked_sub(i)
                        .ok_or(ElfError::InvalidProgram)?)
                    .min(WORD_SIZE);
                    for j in 0..len {
                        let offset = (program_header
                            .p_offset
                            .checked_add(i)
                            .ok_or(ElfError::InvalidProgram)?
                            .checked_add(j)
                            .ok_or(ElfError::InvalidProgram)?)
                            as usize;
                        let byte = input.get(offset).ok_or(ElfError::InvalidOffset)?;
                        word |= (*byte as u32)
                            .checked_shl(j.checked_mul(8).ok_or(ElfError::InvalidProgram)?)
                            .ok_or(ElfError::InvalidProgram)?;
                    }
                    image.insert(addr, word);
                }
            }
        }
        Ok(Self { entry_point, image })
    }
}
