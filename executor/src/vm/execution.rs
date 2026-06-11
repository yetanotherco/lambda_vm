use std::{cmp::Ordering, fmt::Debug};

use crate::{
    elf::Elf,
    vm::{
        instruction::{
            decoding::{DecodedInstruction, Instruction, InstructionError, decode_segment_words},
            decompress::{decompress, instr_len},
            execution::ExecutionError,
        },
        logs::Log,
        memory::{Memory, MemoryError, U64HashMap},
        registers::Registers,
    },
};

pub struct ReturnValues {
    pub memory_values: Vec<u8>,
    pub register_values: (i64, i64),
}

/// Result of program execution including logs and predecoded instructions
pub struct ExecutionResult {
    pub return_values: ReturnValues,
    pub logs: Vec<Log>,
    /// Predecoded instructions map (pc -> decoded instruction + byte width).
    /// Use this to look up instructions by their PC from the logs.
    pub instructions: U64HashMap<DecodedInstruction>,
}

/// Size of each log chunk - balances memory usage vs callback overhead
const CHUNK_SIZE: usize = 100_000;

/// Executor state for chunked execution
pub struct Executor {
    memory: Memory,
    registers: Registers,
    pc: u64,
    pub instructions: InstructionCache,
    logs: Vec<Log>,
}

impl Executor {
    pub fn new(program: &Elf, private_inputs: Vec<u8>) -> Result<Self, ExecutorError> {
        let mut memory = Memory::default();
        memory.store_private_inputs(private_inputs)?;
        let instructions = InstructionCache::new(&program.data)?;
        load_program(&program.data, &mut memory)?;

        Ok(Self {
            memory,
            registers: Registers::default(),
            pc: program.entry_point,
            instructions,
            logs: Vec::with_capacity(CHUNK_SIZE),
        })
    }

    /// Resume execution and return next logs. Returns None when program is finished.
    pub fn resume(&mut self) -> Result<Option<&[Log]>, ExecutorError> {
        if self.pc == 0 {
            return Ok(None);
        }

        self.logs.clear();

        while self.pc != 0 && self.logs.len() < CHUNK_SIZE {
            // Instructions must be at least 2-byte aligned. With the RV64C "C"
            // extension a compressed instruction can start on any 2-byte boundary
            // (so `pc % 4 == 2` is legal); only an odd `pc` is truly misaligned.
            if !self.pc.is_multiple_of(2) {
                return Err(ExecutorError::InstructionAddressMisaligned(self.pc));
            }
            let decoded = match self.instructions.get(self.pc) {
                Some(&decoded) => decoded,
                None => {
                    // Not predecoded (e.g. a jump outside the known segments): fetch
                    // a halfword, and only read the second halfword if it is a 4-byte
                    // instruction. Reading per-halfword avoids over-reading past the
                    // end of a region that ends in a compressed instruction.
                    let lo = self.memory.load_half(self.pc)?;
                    if instr_len(lo) == 2 {
                        DecodedInstruction {
                            instr: decompress(lo)?,
                            len: 2,
                        }
                    } else {
                        let hi = self.memory.load_half(self.pc + 2)?;
                        let word = ((hi as u32) << 16) | (lo as u32);
                        DecodedInstruction {
                            instr: Instruction::parse(word)?,
                            len: 4,
                        }
                    }
                }
            };
            let log = decoded.instr.run(
                &mut self.pc,
                &mut self.registers,
                &mut self.memory,
                decoded.len,
            )?;
            self.logs.push(log);
        }

        if self.logs.is_empty() {
            Ok(None)
        } else {
            Ok(Some(&self.logs))
        }
    }

    fn get_return_values(&self) -> Result<ReturnValues, ExecutorError> {
        let memory_return_value = self.memory.read_return_value()?;
        let registers_return_values = self.registers.read_return_values();

        Ok(ReturnValues {
            memory_values: memory_return_value,
            register_values: (
                registers_return_values.0 as i64,
                registers_return_values.1 as i64,
            ),
        })
    }

    /// Get return values after execution is complete (call after resume() returns None)
    pub fn finish(self) -> Result<ReturnValues, ExecutorError> {
        self.get_return_values()
    }

    /// Run to completion and return all logs (consumes executor)
    pub fn run(mut self) -> Result<ExecutionResult, ExecutorError> {
        let mut logs = Vec::with_capacity(CHUNK_SIZE);

        while let Some(chunk) = self.resume()? {
            logs.extend_from_slice(chunk);
        }

        Ok(ExecutionResult {
            return_values: self.get_return_values()?,
            logs,
            instructions: self.instructions.into_instruction_map(),
        })
    }
}

fn load_program(segments: &[crate::elf::Segment], memory: &mut Memory) -> Result<(), MemoryError> {
    for segment in segments {
        for (i, inst) in segment.values.iter().enumerate() {
            let addr = segment.base_addr + (i as u64 * 4);
            memory.store_word(addr, *inst)?;
        }
    }
    Ok(())
}

pub struct InstructionSegment {
    base_addr: u64,
    /// Exclusive end address (`base_addr + byte length`).
    end_addr: u64,
    /// Decoded instructions indexed by 2-byte slot: slot `i` covers the halfword at
    /// `base_addr + 2*i`. A slot is `Some` at an instruction start and `None` for
    /// the second half of a 4-byte instruction (or a non-instruction tail).
    entries: Vec<Option<DecodedInstruction>>,
}

pub struct InstructionCache {
    segments: Vec<InstructionSegment>,
}

impl InstructionCache {
    /// Creates an InstructionCache from a hashmap of address -> decoded instruction.
    /// Used for testing where we don't have real ELF segments.
    pub fn from_map(map: &U64HashMap<DecodedInstruction>) -> Self {
        if map.is_empty() {
            return Self {
                segments: Vec::new(),
            };
        }

        let mut sorted: Vec<_> = map.iter().collect();
        sorted.sort_by_key(|(addr, _)| **addr);

        let mut segments = Vec::new();
        let mut base_addr = *sorted[0].0;
        let mut entries: Vec<Option<DecodedInstruction>> = Vec::new();
        let mut next_addr = base_addr;

        for (&addr, &decoded) in sorted {
            if addr != next_addr {
                // Gap between instructions: close the current segment.
                segments.push(InstructionSegment {
                    base_addr,
                    end_addr: next_addr,
                    entries: std::mem::take(&mut entries),
                });
                base_addr = addr;
                next_addr = addr;
            }
            entries.push(Some(decoded));
            // The second halfword slot of a 4-byte instruction holds no start.
            if decoded.len == 4 {
                entries.push(None);
            }
            next_addr += decoded.len as u64;
        }

        segments.push(InstructionSegment {
            base_addr,
            end_addr: next_addr,
            entries,
        });

        Self { segments }
    }

    pub fn new(segments: &[crate::elf::Segment]) -> Result<Self, InstructionError> {
        let mut result = Vec::new();
        for seg in segments.iter().filter(|s| s.is_executable) {
            // Two 2-byte slots per 4-byte memory word.
            let num_slots = seg.values.len() * 2;
            let mut entries: Vec<Option<DecodedInstruction>> = vec![None; num_slots];
            for (byte_offset, decoded) in decode_segment_words(&seg.values)? {
                entries[(byte_offset / 2) as usize] = Some(decoded);
            }
            result.push(InstructionSegment {
                base_addr: seg.base_addr,
                end_addr: seg.base_addr + (num_slots as u64) * 2,
                entries,
            });
        }
        Ok(Self { segments: result })
    }

    pub fn get(&self, pc: u64) -> Option<&DecodedInstruction> {
        // Fast path: most programs have a single executable segment
        let segment = if self.segments.len() == 1 {
            let seg = &self.segments[0];
            if pc < seg.base_addr || pc >= seg.end_addr {
                return None;
            }
            seg
        } else {
            // Use binary search to find the segment containing pc
            let idx = self
                .segments
                .binary_search_by(|seg| {
                    if pc < seg.base_addr {
                        Ordering::Greater
                    } else if pc >= seg.end_addr {
                        Ordering::Less
                    } else {
                        Ordering::Equal
                    }
                })
                .ok()?;
            &self.segments[idx]
        };

        let byte_offset = pc - segment.base_addr;
        if !byte_offset.is_multiple_of(2) {
            return None;
        }
        segment.entries.get((byte_offset / 2) as usize)?.as_ref()
    }

    pub fn instruction_count(&self) -> usize {
        self.segments
            .iter()
            .map(|s| s.entries.iter().filter(|e| e.is_some()).count())
            .sum()
    }

    pub fn into_instruction_map(self) -> U64HashMap<DecodedInstruction> {
        let mut map = U64HashMap::default();
        for segment in self.segments {
            let base_addr = segment.base_addr;
            for (i, slot) in segment.entries.into_iter().enumerate() {
                if let Some(decoded) = slot {
                    map.insert(base_addr + (i as u64) * 2, decoded);
                }
            }
        }
        map
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ExecutorError {
    #[error("Failed to decode instruction: {0}")]
    Instruction(#[from] InstructionError),
    #[error("Failed to execute instruction: {0}")]
    ExecutionError(#[from] ExecutionError),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("Instruction address misaligned: {0:#018x}")]
    InstructionAddressMisaligned(u64),
}
