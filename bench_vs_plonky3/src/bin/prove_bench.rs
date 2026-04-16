//! Minimal wall-clock benchmark harness for Lambda STARK vs Plonky3.
//!
//! Builds the same Fibonacci AIR as `instruments_breakdown` (but without any
//! instrumentation) and prints a single line `Proving time: X.XXXs` to
//! stdout, suitable for parsing by `bench_vs_plonky3/run.sh`.
//!
//! Usage:
//!   prove_bench --prover {lambda|p3} [--log-rows K] [--num-sequences N]
//!               [--blowup B] [--queries Q] [--grinding G]
//!
//! Defaults match production (`GoldilocksCubicProofOptions::with_blowup(2)`):
//!   log-rows=19, num-sequences=16, blowup=2, queries=219, grinding=0.

use std::process::ExitCode;
use std::time::Instant;

use bench_vs_plonky3::{lambda_fibonacci_pair, plonky3_config, plonky3_fibonacci};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProverKind {
    Lambda,
    P3,
}

struct Args {
    prover: ProverKind,
    log_rows: u32,
    num_sequences: usize,
    blowup: u8,
    queries: usize,
    grinding: u8,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            prover: ProverKind::Lambda,
            log_rows: 19,
            num_sequences: 16,
            blowup: 2,
            queries: 219,
            grinding: 0,
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage: prove_bench --prover {{lambda|p3}} \
         [--log-rows K] [--num-sequences N] \
         [--blowup B] [--queries Q] [--grinding G]"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut prover_set = false;
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--prover" => {
                let v = iter.next().ok_or("--prover needs a value")?;
                args.prover = match v.as_str() {
                    "lambda" => ProverKind::Lambda,
                    "p3" => ProverKind::P3,
                    other => return Err(format!("unknown prover: {other}")),
                };
                prover_set = true;
            }
            "--log-rows" => {
                let v = iter.next().ok_or("--log-rows needs a value")?;
                args.log_rows = v.parse().map_err(|_| "--log-rows: invalid u32")?;
            }
            "--num-sequences" => {
                let v = iter.next().ok_or("--num-sequences needs a value")?;
                args.num_sequences = v.parse().map_err(|_| "--num-sequences: invalid usize")?;
            }
            "--blowup" => {
                let v = iter.next().ok_or("--blowup needs a value")?;
                args.blowup = v.parse().map_err(|_| "--blowup: invalid u8")?;
            }
            "--queries" => {
                let v = iter.next().ok_or("--queries needs a value")?;
                args.queries = v.parse().map_err(|_| "--queries: invalid usize")?;
            }
            "--grinding" => {
                let v = iter.next().ok_or("--grinding needs a value")?;
                args.grinding = v.parse().map_err(|_| "--grinding: invalid u8")?;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    if !prover_set {
        return Err("--prover is required".into());
    }
    if args.log_rows < 2 || args.log_rows > 30 {
        return Err("--log-rows must be in [2, 30]".into());
    }
    if args.num_sequences == 0 {
        return Err("--num-sequences must be > 0".into());
    }
    Ok(args)
}

fn proof_options(args: &Args) -> ProofOptions {
    ProofOptions {
        blowup_factor: args.blowup,
        fri_number_of_queries: args.queries,
        coset_offset: 3,
        grinding_factor: args.grinding,
    }
}

fn run_lambda(args: &Args) -> std::time::Duration {
    let rows = 1usize << args.log_rows;
    let options = proof_options(args);

    let initial_values: Vec<(FE, FE)> = (0..args.num_sequences)
        .map(|i| (FE::from((i + 1) as u64), FE::from((i + 2) as u64)))
        .collect();

    let mut trace = lambda_fibonacci_pair::compute_trace::<F, E>(&initial_values, rows);
    let pub_inputs = lambda_fibonacci_pair::create_public_inputs(initial_values);
    let air = lambda_fibonacci_pair::FibonacciPairMultiColAIR::<F, E>::with_num_sequences(
        &options,
        args.num_sequences,
    );

    let start = Instant::now();
    let _proof = Prover::<F, E, _>::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("lambda prove failed");
    start.elapsed()
}

fn run_p3(args: &Args) -> std::time::Duration {
    let rows = 1usize << args.log_rows;
    let config = plonky3_config::matched_params_config();
    let air = plonky3_fibonacci::P3FibonacciAir {
        num_sequences: args.num_sequences,
    };
    let trace = plonky3_fibonacci::generate_fibonacci_trace(args.num_sequences, rows);
    let pis = plonky3_fibonacci::public_values(args.num_sequences);

    let start = Instant::now();
    let _proof = p3_uni_stark::prove(&config, &air, trace, &pis);
    start.elapsed()
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    let elapsed = match args.prover {
        ProverKind::Lambda => run_lambda(&args),
        ProverKind::P3 => run_p3(&args),
    };

    println!("Proving time: {:.3}s", elapsed.as_secs_f64());
    ExitCode::SUCCESS
}
