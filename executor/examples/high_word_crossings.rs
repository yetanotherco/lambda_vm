//! Lists the memory accesses of a run whose aligned 8-byte words lie in two
//! different 4 GiB ranges (their addresses' high 32-bit words differ), or come
//! within a few words of such a boundary: the CPU's loads and stores that
//! cross from one range's last word into the next range's first, and the
//! ECALLs whose memory ranges reach across or up to a boundary.
//!
//! ```text
//! cargo run --release -p executor --example high_word_crossings -- <guest.elf> <input.bin|->
//! ```
//!
//! It runs the fast engine under the word history window by window and keeps
//! nothing between windows. Each finding is one `CROSSING` line; the run ends
//! with a `CROSSINGS:` summary.
use executor::elf::Elf;
use executor::vm::engine::{Engine, EngineKind};
use executor::vm::history::{ECALL, NO_ACCESS, WORD_BYTES};
use executor::vm::instruction::execution::{
    BLAKE3_ABSORB_SYSCALL_NUMBER, BLAKE3_SYSCALL_NUMBER, DMA_MEMCPY_SYSCALL_NUMBER,
    DMA_MEMSET_SYSCALL_NUMBER, ECREC_SYSCALL_NUMBER, ECSM_SYSCALL_NUMBER, HINT_SYSCALL_NUMBER,
    KECCAK_SYSCALL_NUMBER, KECCAK256_SYSCALL_NUMBER,
};
use executor::vm::memory::PRIVATE_INPUT_START_INDEX;
use std::collections::BTreeMap;

/// How close to a boundary an ECALL's range counts as reaching it, in bytes:
/// a COPY_W source key can start one word before the range, and a keccak
/// block spans 136 bytes.
const NEAR: u64 = 256;

fn syscall_name(number: u64) -> String {
    match number {
        64 => "commit".into(),
        93 => "halt".into(),
        KECCAK_SYSCALL_NUMBER => "keccak-permute".into(),
        KECCAK256_SYSCALL_NUMBER => "keccak256".into(),
        BLAKE3_SYSCALL_NUMBER => "blake3".into(),
        BLAKE3_ABSORB_SYSCALL_NUMBER => "blake3-absorb".into(),
        ECSM_SYSCALL_NUMBER => "ecsm".into(),
        ECREC_SYSCALL_NUMBER => "ecrec".into(),
        DMA_MEMCPY_SYSCALL_NUMBER => "memcpy".into(),
        DMA_MEMSET_SYSCALL_NUMBER => "memset".into(),
        HINT_SYSCALL_NUMBER => "hint".into(),
        other => format!("{other:#x}"),
    }
}

/// The region of the guest's memory map an address lies in.
fn region(address: u64, input_len: u64) -> &'static str {
    let input_end = PRIVATE_INPUT_START_INDEX + 4 + input_len;
    if (PRIVATE_INPUT_START_INDEX..PRIVATE_INPUT_START_INDEX + 4).contains(&address) {
        "input length prefix"
    } else if (PRIVATE_INPUT_START_INDEX..input_end).contains(&address) {
        "private input"
    } else if address >= 0u64.wrapping_sub(1 << 28) {
        "stack"
    } else if address < 0xC000_0000 {
        "code, data or heap"
    } else {
        "other"
    }
}

/// Whether `[start, end)` (wrapping past 2^64 is a crossing) holds two
/// different high words.
fn spans(start: u64, end: u64) -> bool {
    end <= start || (start >> 32) != ((end - 1) >> 32)
}

/// The load's or store's width and kind from the instruction at `pc`.
fn access_of(program: &Elf, pc: u64) -> (u8, &'static str) {
    let word = program
        .data
        .iter()
        .find_map(|s| {
            let off = pc.checked_sub(s.base_addr)?;
            s.values.get((off / 4) as usize).copied()
        })
        .unwrap_or(0);
    let funct3 = (word >> 12) & 7;
    let kind = match word & 0x7f {
        0x03 => "load",
        0x23 => "store",
        _ => "?",
    };
    (1 << (funct3 & 3), kind)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: high_word_crossings <elf> <input|->");
        std::process::exit(2);
    }
    let elf = std::fs::read(&args[1]).expect("the ELF");
    let program = Elf::load(&elf).expect("an ELF the loader takes");
    let input = if args[2] == "-" {
        Vec::new()
    } else {
        std::fs::read(&args[2]).expect("the input")
    };
    let input_len = input.len() as u64;
    println!(
        "INPUT: {input_len} bytes at {:#x}..{:#x}",
        PRIVATE_INPUT_START_INDEX + 4,
        PRIVATE_INPUT_START_INDEX + 4 + input_len
    );
    let mut engine = Engine::new(EngineKind::Fast, &program, input).expect("the run starts");
    engine.enable_word_history().expect("the word history");
    let window = 1usize << 20;
    let (mut cycles, mut crossing_accesses, mut boundary_accesses) = (0u64, 0u64, 0u64);
    let mut ecalls: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut shown = 0usize;
    // How far the ECALLs' ranges reach into the private input: a check that
    // the boundary search sees them.
    let (mut input_ranges, mut highest_low) = (0u64, 0u64);
    while let Some((logs, history)) = engine.resume_word_history(window).expect("the run") {
        for (i, (log, extra)) in logs.iter().zip(&history.extras).enumerate() {
            let cycle = history.first + i as u64;
            let side = extra.side;
            if side == NO_ACCESS {
                continue;
            }
            if side & ECALL != 0 {
                let group = history.groups[(side & !ECALL) as usize];
                let name = syscall_name(log.src1_val);
                let entry = ecalls.entry(name.clone()).or_default();
                entry.0 += 1;
                let ranges = &history.word_ranges
                    [group.first as usize..(group.first + group.count) as usize];
                for r in ranges {
                    let end = r.addr.wrapping_add(WORD_BYTES * u64::from(r.len));
                    if region(r.addr, input_len) == "private input" {
                        input_ranges += 1;
                    }
                    if end <= 1 << 33 {
                        highest_low = highest_low.max(end);
                    }
                }
                let reaches = ranges.iter().any(|r| {
                    let end = r.addr.wrapping_add(WORD_BYTES * u64::from(r.len));
                    spans(r.addr.wrapping_sub(NEAR), end.wrapping_add(NEAR))
                });
                if reaches {
                    entry.1 += 1;
                    if shown < 400 {
                        shown += 1;
                        let regs = history
                            .reg_groups
                            .get((side & !ECALL) as usize)
                            .map(|g| {
                                history.reg_cells[g.first as usize..(g.first + g.count) as usize]
                                    .iter()
                                    .map(|c| format!("x{}={:#x}", c.reg, c.value))
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            })
                            .unwrap_or_default();
                        let words = ranges
                            .iter()
                            .map(|r| {
                                let end = r.addr.wrapping_add(WORD_BYTES * u64::from(r.len));
                                format!(
                                    "[{:#x}..{end:#x}) {} words{} ({})",
                                    r.addr,
                                    r.len,
                                    if spans(r.addr, end) { " SPANS" } else { "" },
                                    region(r.addr, input_len)
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        println!(
                            "CROSSING ecall {name} cycle {cycle} pc {:#x} {regs} words {words}",
                            log.current_pc
                        );
                    }
                }
                continue;
            }
            let access = history.word_accesses[side as usize];
            if access.next == NO_ACCESS {
                continue;
            }
            crossing_accesses += 1;
            let (width, kind) = access_of(&program, log.current_pc);
            let first_word = access.addr & !(WORD_BYTES - 1);
            if spans(first_word, first_word.wrapping_add(2 * WORD_BYTES)) {
                boundary_accesses += 1;
                if shown < 400 {
                    shown += 1;
                    println!(
                        "CROSSING {kind} cycle {cycle} pc {:#x} address {:#x} width {width} ({})",
                        log.current_pc,
                        access.addr,
                        region(access.addr, input_len)
                    );
                }
            }
        }
        cycles += logs.len() as u64;
    }
    let calls = ecalls
        .iter()
        .map(|(name, (n, near))| format!("{name} {n} ({near} near a boundary)"))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "CROSSINGS: cycles {cycles} · two-word accesses {crossing_accesses} · across a 4 GiB \
         boundary {boundary_accesses} · ecalls: {calls} · ecall ranges in the private input \
         {input_ranges} · highest ecall word end below 2^33 {highest_low:#x}"
    );
}
