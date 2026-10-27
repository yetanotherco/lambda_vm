use executor::{elf::Elf, vm::execution::Executor};
use executor::vm::instruction::execution::{DMA_MEMCPY_SYSCALL_NUMBER, DMA_MEMSET_SYSCALL_NUMBER, KECCAK_SYSCALL_NUMBER, ECSM_SYSCALL_NUMBER};
use std::{collections::BTreeMap, io::Write, path::Path};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let elf = Elf::load(&std::fs::read(&args[1]).unwrap()).unwrap();
    let ecalls: std::collections::BTreeSet<u64> = elf.data.iter()
        .filter(|segment| segment.is_executable)
        .flat_map(|segment| segment.values.iter().enumerate().filter_map(move |(i, &word)|
            (word == 0x00000073).then_some(segment.base_addr + i as u64 * 4)))
        .collect();
    for file in &args[2..] {
        let input = std::fs::read(file).unwrap();
        let expected = if input.is_empty() {
            None
        } else {
            let decoded = rkyv::from_bytes::<
                ethrex_guest_program::l1::ProgramInput,
                rkyv::rancor::Error,
            >(&input)
            .unwrap();
            Some(
                ethrex_guest_program::l1::execution_program(
                    decoded,
                    std::sync::Arc::new(ethrex_crypto::NativeCrypto),
                )
                .unwrap()
                .encode(),
            )
        };
        let mut vm = Executor::new(&elf, input).unwrap();
        let mut cycles = 0u64;
        let (mut copy_calls, mut copy_bytes, mut copy_rows) = (0u64, 0u64, 0u64);
        let (mut keccak_calls, mut ecsm_calls) = (0u64, 0u64);
        let mut pcs = BTreeMap::<u64, u64>::new();
        let profile = std::env::var("PROFILE").ok();
        while let Some(logs) = vm.resume().unwrap() {
            cycles += logs.len() as u64;
            for log in logs {
                if log.src1_val == KECCAK_SYSCALL_NUMBER && ecalls.contains(&log.current_pc) { keccak_calls += 1; }
                if log.src1_val == ECSM_SYSCALL_NUMBER && ecalls.contains(&log.current_pc) { ecsm_calls += 1; }
                if (log.src1_val == DMA_MEMCPY_SYSCALL_NUMBER || log.src1_val == DMA_MEMSET_SYSCALL_NUMBER)
                    && ecalls.contains(&log.current_pc) {
                    copy_calls += 1;
                    copy_bytes += log.dst_val;
                    copy_rows += log.src2_val;
                }
            }
            if profile.is_some() {
                for log in logs {
                    *pcs.entry(log.current_pc).or_default() += 1;
                }
            }
        }
        let output = vm.finish().unwrap().memory_values;
        if let Some(expected) = expected {
            assert_eq!(output, expected, "native output mismatch: {file}");
        }
        let hex: String = output.iter().map(|b| format!("{b:02x}")).collect();
        eprintln!("{}\tcopy_calls={}\tcopy_bytes={}\tcopy_rows={}\tkeccak_calls={}\tecsm_calls={}", Path::new(file).file_name().unwrap().to_str().unwrap(), copy_calls, copy_bytes, copy_rows, keccak_calls, ecsm_calls);
        println!(
            "{}\t{}\t{}",
            Path::new(file).file_name().unwrap().to_str().unwrap(),
            cycles,
            hex
        );
        if let Some(path) = profile {
            let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
            for (pc, count) in pcs {
                writeln!(out, "{pc}\t{count}").unwrap();
            }
        }
    }
}
