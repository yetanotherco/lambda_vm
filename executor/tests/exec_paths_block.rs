//! A real block run on the executor's fast access paths against the general
//! path, log by log: the two must write the same logs and the same public
//! output. Ignored: it needs a guest ELF and a block's private input,
//! `EXEC_PATHS_ELF` and `EXEC_PATHS_INPUT`, e.g.
//!
//! ```text
//! EXEC_PATHS_ELF=ethrex.elf EXEC_PATHS_INPUT=ethrex_mainnet_25475471_8140dd6d.bin \
//!   cargo test --release -p executor --test exec_paths_block -- --ignored --nocapture
//! ```
use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::memory::ExecPaths;

#[test]
#[ignore = "needs EXEC_PATHS_ELF and EXEC_PATHS_INPUT"]
fn a_block_writes_the_same_logs_on_every_path() {
    let elf = std::fs::read(std::env::var("EXEC_PATHS_ELF").expect("EXEC_PATHS_ELF")).unwrap();
    let input =
        std::fs::read(std::env::var("EXEC_PATHS_INPUT").expect("EXEC_PATHS_INPUT")).unwrap();
    let program = Elf::load(&elf).expect("the ELF");
    let window = 1 << 20;
    let mut reference =
        Executor::new_with_paths(&program, input.clone(), ExecPaths::GENERAL).unwrap();
    let mut fast = Executor::new_with_paths(&program, input, ExecPaths::FAST).unwrap();
    let (mut cycles, mut mismatched) = (0usize, 0usize);
    // The fast run hands its logs out as the block prover takes them.
    while let Some(want) = reference
        .resume_with_limit(window)
        .expect("the reference run")
    {
        let got = fast
            .resume_owned(window)
            .expect("the fast run")
            .expect("the fast run ended early");
        assert_eq!(got.len(), want.len(), "window at cycle {cycles}");
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            let (a, b) = (
                [a.current_pc, a.next_pc, a.src1_val, a.src2_val, a.dst_val],
                [b.current_pc, b.next_pc, b.src1_val, b.src2_val, b.dst_val],
            );
            if a != b {
                mismatched += 1;
                if mismatched <= 10 {
                    eprintln!("cycle {}: fast {a:x?} general {b:x?}", cycles + i);
                }
            }
        }
        cycles += want.len();
    }
    assert!(
        fast.resume_owned(window).unwrap().is_none(),
        "the fast run went on"
    );
    let (reference, fast) = (reference.finish().unwrap(), fast.finish().unwrap());
    assert_eq!(fast.memory_values, reference.memory_values, "public output");
    assert_eq!(fast.register_values, reference.register_values, "a0, a1");
    eprintln!("EXEC PATHS: {cycles} cycles, {mismatched} logs differ, public output equal");
    assert_eq!(mismatched, 0);
}
