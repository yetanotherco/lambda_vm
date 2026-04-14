use sp1_sdk::blocking::{ProveRequest, Prover, ProverClient};
use sp1_sdk::{include_elf, ProvingKey, SP1Stdin};
use std::time::Instant;

const FIB_ELF: sp1_sdk::Elf = include_elf!("fibonacci-program");

fn main() {
    sp1_sdk::utils::setup_logger();

    let n: u64 = std::env::args()
        .nth(1)
        .expect("Usage: fibonacci-script <n>")
        .parse()
        .expect("n must be a u64");

    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();
    stdin.write(&n);

    // Cycle count — executed *before* the timer starts, matching Lambda's
    // pre-pass for symmetry. This costs extra wall-clock but does not inflate
    // the measured proving time.
    let (_, report) = client
        .execute(FIB_ELF.clone(), stdin.clone())
        .run()
        .unwrap();
    println!("Cycles: {}", report.total_instruction_count());

    // Timed window: end-to-end single-shot proving, including `setup`
    // (verifying-key derivation) and the `core` proof itself. No recursion /
    // compression, no verification.
    let start = Instant::now();
    let pk = client.setup(FIB_ELF.clone()).expect("setup failed");
    let proof = client
        .prove(&pk, stdin)
        .core()
        .run()
        .expect("prove failed");
    let elapsed = start.elapsed();

    println!("Proving time: {:.3}s", elapsed.as_secs_f64());

    // Verify (outside the timer, same as Lambda).
    client
        .verify(&proof, pk.verifying_key(), None)
        .expect("verify failed");

    println!("Proof verified successfully");
}
