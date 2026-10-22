use executor::{
    elf::{Elf, SymbolTable},
    vm::execution::{Executor, ReturnValues},
};
use std::ops::Range;

// NOTE: These tests require 64-bit RISC-V ELF files (RV64IM).
// The test binaries need to be recompiled for 64-bit architecture.
fn run_program_without_expect(
    elf_path: &str,
    private_inputs: Vec<u8>,
) -> Result<ReturnValues, executor::vm::execution::ExecutorError> {
    println!("Testing {}", elf_path);
    let elf_data = std::fs::read(elf_path).unwrap();
    let program = Elf::load(&elf_data).unwrap();
    println!("Program entry: 0x{:016x}", program.entry_point);
    let mut executor = Executor::new(&program, private_inputs)?;
    while let Some(_logs) = executor.resume()? {}
    executor.finish()
}

fn run_program_and_check_public_output(
    elf_path: &str,
    expected_output: Vec<u8>,
    private_inputs: Vec<u8>,
) {
    let result =
        run_program_without_expect(elf_path, private_inputs).expect("Failed to run program");

    assert_eq!(result.memory_values, expected_output);
}

fn run_program_and_check_output(elf_path: &str, expected_output: i64, private_inputs: Vec<u8>) {
    let result =
        run_program_without_expect(elf_path, private_inputs).expect("Failed to run program");

    assert!(result.register_values.0 == expected_output);
}

#[test]
fn test_basic_rust() {
    run_program_and_check_output("./program_artifacts/rust/basic_rust.elf", 0, vec![]);
}

#[test]
fn test_add() {
    run_program_and_check_output("./program_artifacts/rust/add.elf", 3, vec![]);
}

#[test]
fn test_if() {
    run_program_and_check_output("./program_artifacts/rust/if.elf", 10, vec![]);
}

#[test]
fn test_fibonacci() {
    run_program_and_check_output("./program_artifacts/rust/fibonacci.elf", 55, vec![]);
}

#[test]
fn test_fibonacci_iterative() {
    run_program_and_check_output(
        "./program_artifacts/rust/fibonacci_iterative.elf",
        1597,
        vec![],
    );
}

#[test]
fn test_byte() {
    run_program_and_check_output("./program_artifacts/rust/byte.elf", 0xDE, vec![]);
}

#[test]
fn test_byte_signed() {
    run_program_and_check_output("./program_artifacts/rust/byte_signed.elf", -8, vec![]);
}

#[test]
fn test_half() {
    run_program_and_check_output("./program_artifacts/rust/half.elf", 0xDEAD, vec![]);
}

#[test]
fn test_half_signed() {
    run_program_and_check_output(
        "./program_artifacts/rust/half_signed.elf",
        0xBEEF - 0xDEAD,
        vec![],
    );
}

#[test]
fn test_rlp() {
    run_program_and_check_output("./program_artifacts/rust/rlp.elf", 65536, vec![]);
}

#[test]
fn test_allocator() {
    run_program_and_check_public_output(
        "./program_artifacts/rust/allocator.elf",
        b"Hello World".to_vec(),
        vec![],
    );
}

#[test]
fn test_ethereum_types() {
    run_program_and_check_output("./program_artifacts/rust/ethereum_types.elf", 1, vec![]);
}

#[test]
fn test_vector() {
    run_program_and_check_public_output(
        "./program_artifacts/rust/vector.elf",
        [1, 2, 3, 4, 5].to_vec(),
        vec![],
    );
}

#[test]
fn test_hashmap() {
    run_program_and_check_output("./program_artifacts/rust/hashmap.elf", 3, vec![]);
}

#[test]
fn test_asm() {
    run_program_and_check_output("./program_artifacts/rust/asm.elf", 42, vec![]);
}

#[test]
fn test_print() {
    run_program_and_check_output("./program_artifacts/rust/print.elf", 1, vec![]);
}

#[test]
fn test_stdout() {
    run_program_and_check_output("./program_artifacts/rust/stdout.elf", 1, vec![]);
}

#[test]
fn test_panic() {
    let result = run_program_without_expect("./program_artifacts/rust/panic.elf", vec![]);
    assert!(result.is_err());
    if let Err(executor::vm::execution::ExecutorError::ExecutionError(
        executor::vm::instruction::execution::ExecutionError::Panic(msg),
    )) = result
    {
        assert_eq!(msg, "This is a panic test");
    } else {
        panic!("Expected panic error");
    }
}

#[test]
fn test_commit() {
    run_program_and_check_public_output(
        "./program_artifacts/rust/commit.elf",
        vec![4, 5, 6, 7],
        vec![4, 5, 6, 7],
    );
}

const ETHREX_ELF: &str = "./program_artifacts/rust/ethrex.elf";
const MEM_SWEEP_ELF: &str = "./program_artifacts/rust/mem_sweep.elf";

/// How the runtime's `memcpy` (syscalls' `fast-memcpy` feature) starts: `mv t0, a0`
/// and `li t1, 32`. compiler_builtins' `memcpy` does not.
const FAST_MEMCPY_HEAD: [u32; 2] = [0x0005_0293, 0x0200_0313];

/// The most instructions `test_mem_sweep` may spend inside the library's `memmove`.
/// Built with compiler_builtins' `mem-unaligned` setting (the sweep guest's
/// `.cargo/config.toml`) it spends 4,026,828; without it 5,975,778,
/// reassembling every misaligned word with shifts (both measured at the Makefile's
/// pinned nightly). Only compiler_builtins reads that setting's name, with no
/// stability guarantee, so a toolchain bump can silently drop it: this bound is
/// what notices.
const SWEEP_MEMMOVE_MAX: u64 = 5_000_000;

/// The address range and the instruction words of the function that starts at `pc`.
fn function_at(program: &Elf, symbols: &SymbolTable, pc: u64) -> Option<(Range<u64>, Vec<u32>)> {
    let f = symbols.lookup(pc).filter(|f| f.address == pc)?;
    let segment = program.data.iter().find(|s| {
        s.is_executable && pc >= s.base_addr && pc < s.base_addr + 4 * s.values.len() as u64
    })?;
    let first = ((pc - segment.base_addr) / 4) as usize;
    let words = segment.values[first..first + (f.size / 4) as usize].to_vec();
    Some((pc..pc + f.size, words))
}

/// The address range and the instruction words of the function `name` in an ELF.
fn function_in(elf_bytes: &[u8], name: &str) -> (Range<u64>, Vec<u32>) {
    let program = Elf::load(elf_bytes).unwrap();
    let symbols = SymbolTable::parse(elf_bytes);
    for segment in program.data.iter().filter(|s| s.is_executable) {
        for i in 0..segment.values.len() as u64 {
            let pc = segment.base_addr + 4 * i;
            if symbols
                .lookup(pc)
                .is_some_and(|f| f.address == pc && f.name == name)
            {
                return function_at(&program, &symbols, pc).unwrap();
            }
        }
    }
    panic!("no function `{name}` in the ELF");
}

/// The address range and the instruction words of what `name` executes: the
/// function itself or, when it is a two-instruction tail jump (`auipc rd, hi` then
/// `jalr x0, lo(rd)`, which is how these guests' `memmove` enters compiler_builtins'
/// implementation), the function it jumps to. The words are compared across ELFs,
/// so they must not depend on where the code sits.
fn body_of(elf_bytes: &[u8], name: &str) -> (Range<u64>, Vec<u32>) {
    let (range, words) = function_in(elf_bytes, name);
    let &[auipc, jalr] = words.as_slice() else {
        return (range, words);
    };
    let rd = (auipc >> 7) & 31;
    let is_auipc = auipc & 0x7f == 0x17;
    let is_jump = jalr & 0x7f == 0x67 && (jalr >> 7) & 31 == 0 && (jalr >> 15) & 31 == rd;
    if !(is_auipc && is_jump) {
        return (range, words);
    }
    let hi = i64::from((auipc & 0xffff_f000) as i32);
    let lo = i64::from(jalr as i32 >> 20);
    let target = (range.start as i64 + hi + lo) as u64;
    let program = Elf::load(elf_bytes).unwrap();
    let symbols = SymbolTable::parse(elf_bytes);
    function_at(&program, &symbols, target)
        .unwrap_or_else(|| panic!("`{name}` jumps to {target:#x}, where no function starts"))
}

/// In-VM sweep of the memory routines the guest links (`memcpy` is the runtime's
/// own, from syscalls' `fast-memcpy`; `memmove`, `memset` and `memcmp` are the
/// library's, built with the ethrex guest's flags). The guest compares every result
/// with values recomputed from the index and commits `OK` plus its case counts, or
/// `FAIL` plus the first failing case. Counts, in order: memcpy at every length
/// 0..=1100 and offsets mod 8; memcpy at lengths 0..=160 and offsets mod 16; memcpy
/// large lengths, lengths up to a page and the compiler's own copies; memcpy onto
/// itself and between touching regions; memmove overlapping both ways; memset and
/// memcmp.
#[test]
fn test_mem_sweep() {
    let elf = std::fs::read(MEM_SWEEP_ELF).unwrap();
    let (_, memcpy) = function_in(&elf, "memcpy");
    assert_eq!(
        memcpy[..2],
        FAST_MEMCPY_HEAD,
        "the sweep guest does not link the runtime's memcpy"
    );
    let (memmove, _) = body_of(&elf, "memmove");
    let program = Elf::load(&elf).unwrap();
    let mut executor = Executor::new(&program, vec![]).expect("Failed to run program");
    let mut in_memmove = 0u64;
    while let Some(logs) = executor.resume().expect("Failed to run program") {
        in_memmove += logs
            .iter()
            .filter(|log| memmove.contains(&log.current_pc))
            .count() as u64;
    }
    let out = executor
        .finish()
        .expect("Failed to run program")
        .memory_values;
    if !out.starts_with(b"OK") {
        assert!(
            out.starts_with(b"FAIL") && out.len() == 15,
            "unexpected output {out:?}"
        );
        panic!(
            "mem_sweep failed: stage {} length {} source alignment {} destination alignment {} first wrong byte at destination + {}",
            out[4],
            u32::from_le_bytes(out[5..9].try_into().unwrap()),
            out[9],
            out[10],
            i32::from_le_bytes(out[11..15].try_into().unwrap()),
        );
    }
    let counts: Vec<u32> = out[2..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // 1101 lengths x 64; 161 lengths x 256; 6 x 16 + 2 x 4 + 428 x 4 + 16; 301 x 8 x 3;
    // 201 x 8 x 19; 301 x 8 x 2 memset + 64 + 64 lengths x 64 x 7 memcmp.
    assert_eq!(counts, [70_464, 41_216, 1_832, 7_224, 30_552, 33_552]);
    assert!(
        in_memmove <= SWEEP_MEMMOVE_MAX,
        "the sweep spent {in_memmove} instructions inside memmove (at most \
         {SWEEP_MEMMOVE_MAX}): compiler_builtins' `mem-unaligned` setting is not in effect"
    );
}

/// The guest every block proof is about links the very `memcpy` and `memmove` that
/// `test_mem_sweep` exercises: the same instruction words in both ELFs.
#[test]
fn test_ethrex_links_the_swept_memory_routines() {
    let ethrex = std::fs::read(ETHREX_ELF).unwrap();
    let sweep = std::fs::read(MEM_SWEEP_ELF).unwrap();
    let (_, memcpy) = function_in(&ethrex, "memcpy");
    assert_eq!(
        memcpy[..2],
        FAST_MEMCPY_HEAD,
        "the ethrex guest does not link the runtime's memcpy"
    );
    for name in ["memcpy", "memmove"] {
        assert_eq!(
            body_of(&ethrex, name).1,
            body_of(&sweep, name).1,
            "`{name}` differs between the ethrex guest and the sweep guest"
        );
    }
}

#[test]
fn test_ef_io_demo_concatenates_writes() {
    // Demo guest reads its private input via EF `read_input`, then emits it
    // back as the public output via TWO `write_output` calls (split in halves).
    // The COMMIT AIR concatenates the two calls; the executor's
    // `commit_public_output` appends in the same order.
    let input: Vec<u8> = b"hello world!".to_vec();
    run_program_and_check_public_output(
        "./program_artifacts/rust/ef_io_demo.elf",
        input.clone(),
        input,
    );
}

#[test]
fn test_commit_sum() {
    run_program_and_check_public_output(
        "./program_artifacts/rust/commit_sum.elf",
        10u8.to_le_bytes().to_vec(),
        vec![3, 7],
    );
}

#[test]
fn test_serde() {
    #[derive(serde::Serialize)]
    struct MyData {
        val: i32,
        values: Vec<u8>,
    }
    let my_data = MyData {
        val: 42,
        values: vec![1, 2, 3, 4, 5],
    };
    let serialized = serde_json::to_vec(&my_data).unwrap();
    run_program_and_check_public_output("./program_artifacts/rust/serde.elf", serialized, vec![]);
}

#[test]
fn test_random() {
    run_program_and_check_public_output("./program_artifacts/rust/random.elf", vec![116], vec![]);
}

#[test]
fn test_memory() {
    let size = 100000u32;
    let output = vec![1; size as usize];
    run_program_and_check_public_output(
        "./program_artifacts/rust/memory.elf",
        output[(size - 1000) as usize..].to_vec(),
        size.to_be_bytes().to_vec(),
    );
}

#[test]
fn test_keccak() {
    use tiny_keccak::Hasher;
    let input_a = b"hello world";
    let input_b = b"!";
    let mut output = [0u8; 32];
    let mut hasher = tiny_keccak::Keccak::v256();
    hasher.update(input_a);
    hasher.update(input_b);
    hasher.finalize(&mut output);
    run_program_and_check_public_output(
        "./program_artifacts/rust/keccak.elf",
        output.to_vec(),
        vec![],
    );
}

#[test]
fn test_keccak_precompile() {
    use tiny_keccak::Hasher;

    fn keccak256(input: &[u8]) -> [u8; 32] {
        let mut output = [0u8; 32];
        let mut hasher = tiny_keccak::Keccak::v256();
        hasher.update(input);
        hasher.finalize(&mut output);
        output
    }

    // Known-answer vectors for the `keccak_permute`-ecall-backed sponge
    // (`lambda_vm_syscalls::keccak::keccak256`), computed with the trusted
    // `tiny_keccak` crate. Covers empty input, one rate block (136 bytes)
    // minus one byte, exactly one rate block, and a multi-block input —
    // the sponge's padding edge cases. Inputs must match
    // `executor/programs/rust/keccak_precompile/src/main.rs` exactly.
    const RATE_BYTES: usize = 136;
    let multi_block_input: Vec<u8> = (0..2 * RATE_BYTES + 17).map(|i| i as u8).collect();

    let expected: Vec<u8> = [
        keccak256(b""),
        keccak256(b"abc"),
        keccak256(&[0x5a; RATE_BYTES - 1]),
        keccak256(&[0x3c; RATE_BYTES]),
        keccak256(&multi_block_input),
    ]
    .into_iter()
    .flatten()
    .collect();

    run_program_and_check_public_output(
        "./program_artifacts/rust/keccak_precompile.elf",
        expected,
        vec![],
    );
}

#[test]
fn test_keccak_transcript_pattern() {
    use tiny_keccak::Hasher;

    fn keccak256(chunks: &[&[u8]]) -> [u8; 32] {
        let mut output = [0u8; 32];
        let mut hasher = tiny_keccak::Keccak::v256();
        for chunk in chunks {
            hasher.update(chunk);
        }
        hasher.finalize(&mut output);
        output
    }

    // Reference values for `executor/programs/rust/keccak_transcript_pattern`,
    // which drives `PlatformKeccak256` the same way `DefaultTranscript::sample`
    // does: several small, non-rate-aligned `update()`s per round, interleaved
    // with `finalize_reset()`, each round reseeded with the reversed prior
    // digest. Covers the cross-call sponge buffering that a one-shot hash of a
    // whole slice can't reach.
    let digest1 = keccak256(&[&[0xaa; 5], &[0xbb; 40], &[0xcc; 17], &[0xdd; 100]]);
    let mut reversed1 = digest1;
    reversed1.reverse();
    let digest2 = keccak256(&[&reversed1, &[0xee; 3], &[0xff; 130]]);
    let mut reversed2 = digest2;
    reversed2.reverse();
    let digest3 = keccak256(&[&reversed2]);

    let expected: Vec<u8> = [digest1, digest2, digest3].into_iter().flatten().collect();

    run_program_and_check_public_output(
        "./program_artifacts/rust/keccak_transcript_pattern.elf",
        expected,
        vec![],
    );
}

#[test]
fn test_stdin_read_panics() {
    let result = run_program_without_expect("./program_artifacts/rust/stdin_read.elf", vec![]);
    assert!(result.is_err());
    if let Err(executor::vm::execution::ExecutorError::ExecutionError(
        executor::vm::instruction::execution::ExecutionError::Panic(msg),
    )) = result
    {
        assert!(
            msg.contains("sys_read is not supported"),
            "Expected sys_read panic, got: {}",
            msg
        );
    } else {
        panic!("Expected panic error for stdin_read");
    }
}

#[test]
fn test_args_panics() {
    let result = run_program_without_expect("./program_artifacts/rust/args_test.elf", vec![]);
    assert!(result.is_err());
    if let Err(executor::vm::execution::ExecutorError::ExecutionError(
        executor::vm::instruction::execution::ExecutionError::Panic(msg),
    )) = result
    {
        assert!(
            msg.contains("sys_argc is not supported"),
            "Expected sys_argc panic, got: {}",
            msg
        );
    } else {
        panic!("Expected panic error for args_test");
    }
}

#[ignore = "Ignored until the vm is fast enough to run this test"]
#[test]
fn test_ckzg() {
    run_program_and_check_public_output("./program_artifacts/rust/ckzg.elf", vec![1, 1], vec![]);
}
