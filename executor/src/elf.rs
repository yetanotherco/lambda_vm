const EI_NIDENT: usize = 16;
const EM_RISCV: u16 = 243;
const ET_EXEC: u16 = 2;
const PT_LOAD: u32 = 1;
const PF_X: u32 = 0x1;
const MAX_PROGRAM_HEADERS: usize = 256;
const EXECUTABLE_HEADER_SIZE: usize = 64; // 64-bit ELF header is 64 bytes
const PROGRAM_HEADER_SIZE: usize = 56; // 64-bit program header is 56 bytes
const ELF_64_BIT: u8 = 2;
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
    /// Entry point virtual address (64-bit)
    e_entry: u64,
    /// Program header table file offset (64-bit)
    e_phoff: u64,
    /// Section header table file offset (64-bit)
    _e_shoff: u64,
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
    /// Segment flags (moved in 64-bit format)
    p_flags: u32,
    /// Segment file offset (64-bit)
    p_offset: u64,
    /// Segment virtual address (64-bit)
    p_vaddr: u64,
    /// Segment physical address (64-bit)
    _p_paddr: u64,
    /// Segment size in file (64-bit)
    p_filesz: u64,
    /// Segment size in memory (64-bit)
    p_memsz: u64,
    /// Segment alignment (64-bit)
    _p_align: u64,
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
        if e_ident[4] != ELF_64_BIT {
            return Err(ElfError::Not64Bit);
        }
        if e_ident[5] != ELF_LITTLE_ENDIAN {
            return Err(ElfError::NotLittleEndian);
        }
        if e_ident[6] != ELF_CURRENT_VERSION {
            return Err(ElfError::InvalidElfVersion);
        }
        // 64-bit ELF header layout:
        // 0-15: e_ident
        // 16-17: e_type
        // 18-19: e_machine
        // 20-23: e_version
        // 24-31: e_entry (8 bytes)
        // 32-39: e_phoff (8 bytes)
        // 40-47: e_shoff (8 bytes)
        // 48-51: e_flags
        // 52-53: e_ehsize
        // 54-55: e_phentsize
        // 56-57: e_phnum
        // 58-59: e_shentsize
        // 60-61: e_shnum
        // 62-63: e_shstrndx
        let e_type = u16::from_le_bytes(input[16..18].try_into().map_err(|_| ElfError::Casting)?);
        let e_machine =
            u16::from_le_bytes(input[18..20].try_into().map_err(|_| ElfError::Casting)?);
        let e_version =
            u32::from_le_bytes(input[20..24].try_into().map_err(|_| ElfError::Casting)?);
        let e_entry = u64::from_le_bytes(input[24..32].try_into().map_err(|_| ElfError::Casting)?);
        let e_phoff = u64::from_le_bytes(input[32..40].try_into().map_err(|_| ElfError::Casting)?);
        let e_shoff = u64::from_le_bytes(input[40..48].try_into().map_err(|_| ElfError::Casting)?);
        let e_flags = u32::from_le_bytes(input[48..52].try_into().map_err(|_| ElfError::Casting)?);
        let e_ehsize = u16::from_le_bytes(input[52..54].try_into().map_err(|_| ElfError::Casting)?);
        let e_phentsize =
            u16::from_le_bytes(input[54..56].try_into().map_err(|_| ElfError::Casting)?);
        let e_phnum = u16::from_le_bytes(input[56..58].try_into().map_err(|_| ElfError::Casting)?);
        let e_shentsize =
            u16::from_le_bytes(input[58..60].try_into().map_err(|_| ElfError::Casting)?);
        let e_shnum = u16::from_le_bytes(input[60..62].try_into().map_err(|_| ElfError::Casting)?);
        let e_shstrndx =
            u16::from_le_bytes(input[62..64].try_into().map_err(|_| ElfError::Casting)?);
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
        // 64-bit program header layout:
        // 0-3: p_type
        // 4-7: p_flags
        // 8-15: p_offset (8 bytes)
        // 16-23: p_vaddr (8 bytes)
        // 24-31: p_paddr (8 bytes)
        // 32-39: p_filesz (8 bytes)
        // 40-47: p_memsz (8 bytes)
        // 48-55: p_align (8 bytes)
        let p_type = u32::from_le_bytes(input[0..4].try_into().map_err(|_| ElfError::Casting)?);
        let p_flags = u32::from_le_bytes(input[4..8].try_into().map_err(|_| ElfError::Casting)?);
        let p_offset = u64::from_le_bytes(input[8..16].try_into().map_err(|_| ElfError::Casting)?);
        let p_vaddr = u64::from_le_bytes(input[16..24].try_into().map_err(|_| ElfError::Casting)?);
        let p_paddr = u64::from_le_bytes(input[24..32].try_into().map_err(|_| ElfError::Casting)?);
        let p_filesz = u64::from_le_bytes(input[32..40].try_into().map_err(|_| ElfError::Casting)?);
        let p_memsz = u64::from_le_bytes(input[40..48].try_into().map_err(|_| ElfError::Casting)?);
        let p_align = u64::from_le_bytes(input[48..56].try_into().map_err(|_| ElfError::Casting)?);
        Ok(Self {
            p_type,
            p_flags,
            p_offset,
            p_vaddr,
            _p_paddr: p_paddr,
            p_filesz,
            p_memsz,
            _p_align: p_align,
        })
    }
}

/// A contiguous segment of instructions loaded from a PT_LOAD segment
#[derive(Debug, Clone)]
pub struct Segment {
    /// Base virtual address for this segment
    pub base_addr: u64,
    /// The 32-bit instruction words in this segment
    pub values: Vec<u32>,
    /// Whether this segment is executable (has PF_X flag)
    pub is_executable: bool,
}

pub struct Elf {
    pub entry_point: u64,
    /// Segments loaded from PT_LOAD headers, sorted by base_addr
    pub data: Vec<Segment>,
}

pub(crate) const WORD_SIZE: u64 = 4;

#[derive(Debug, thiserror::Error)]
pub enum ElfError {
    #[error("Not a 64-bit ELF")]
    Not64Bit,
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
        let elf_program = ElfProgram::parse(input)?;
        if elf_program.ehdr.e_machine != EM_RISCV {
            return Err(ElfError::NotRiscV);
        }
        if elf_program.ehdr.e_type != ET_EXEC {
            return Err(ElfError::NotExecutable);
        }
        let entry_point: u64 = elf_program.ehdr.e_entry;
        if !entry_point.is_multiple_of(WORD_SIZE) {
            return Err(ElfError::InvalidEntryPoint);
        }
        let phdrs = elf_program.phdrs;
        if phdrs.len() > MAX_PROGRAM_HEADERS {
            return Err(ElfError::TooManyProgramHeaders);
        }
        // Build segments from PT_LOAD headers
        let mut segments: Vec<Segment> = Vec::new();
        for program_header in phdrs
            .iter()
            .filter(|program_header| program_header.p_type == PT_LOAD)
        {
            if !program_header.p_vaddr.is_multiple_of(WORD_SIZE) {
                return Err(ElfError::UnalignedVAddr);
            }
            let mut values = Vec::new();
            for i in (0..program_header.p_memsz).step_by(WORD_SIZE as usize) {
                let word = if i < program_header.p_filesz {
                    // Read initialized data from file
                    let remaining = program_header.p_filesz - i;
                    let len = remaining.min(WORD_SIZE);
                    let mut word = 0u32;
                    for j in 0..len {
                        let offset = (program_header.p_offset + i + j) as usize;
                        let byte = input.get(offset).ok_or(ElfError::InvalidOffset)?;
                        word |= (*byte as u32)
                            .checked_shl((j as u32).checked_mul(8).ok_or(ElfError::InvalidProgram)?)
                            .ok_or(ElfError::InvalidProgram)?;
                    }
                    word
                } else {
                    // BSS section - zero-initialized
                    0
                };
                values.push(word);
            }
            if !values.is_empty() {
                segments.push(Segment {
                    base_addr: program_header.p_vaddr,
                    values,
                    is_executable: (program_header.p_flags & PF_X) != 0,
                });
            }
        }
        // Sort segments by base address for efficient binary search
        segments.sort_by_key(|s| s.base_addr);
        Ok(Self {
            entry_point,
            data: segments,
        })
    }
}
