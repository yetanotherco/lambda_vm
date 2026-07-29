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

    // SP1 splits a core proof by trace area / AIR height, not by cycle count, so the shard
    // count changes regime within a cycle sweep — and that regime change is what shapes its
    // cost curve (per-shard fixed cost amortizing, then parallelism across shards).
    if let sp1_sdk::SP1Proof::Core(shards) = &proof.proof {
        println!("Shards: {}", shards.len());
    }

    // Count main-trace field elements from the proof shards.
    // round 0 = preprocessed, round 1 = main trace.
    let total_elements: usize = match &proof.proof {
        sp1_sdk::SP1Proof::Core(shards) => shards
            .iter()
            .map(|shard| {
                shard
                    .evaluation_proof
                    .row_counts_and_column_counts
                    .get(1)
                    .map(|round| round.iter().map(|&(r, c)| r * c).sum::<usize>())
                    .unwrap_or(0)
            })
            .sum(),
        _ => 0,
    };
    println!("Elements: {}", total_elements);


    // Verify (outside the timer, same as Lambda).
    client
        .verify(&proof, pk.verifying_key(), None)
        .expect("verify failed");

    println!("Proof verified successfully");
}
