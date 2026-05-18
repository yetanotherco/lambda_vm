use std::process::ExitCode;
use std::time::Instant;

use bench_vs_plonky3::{lambda_fibonacci_pair, plonky3_config, plonky3_fibonacci};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::verifier::{IsStarkVerifier, Verifier};

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
    audit_only: bool,
}

struct BenchMetrics {
    setup_s: f64,
    prove_s: f64,
    verify_s: f64,
    proof_size_bytes: usize,
    peak_rss_kb: Option<u64>,
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
            audit_only: false,
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage: prove_bench --prover {{lambda|p3}} [--log-rows K] [--num-sequences N] \
         [--blowup B] [--queries Q] [--grinding G] [--audit-only]"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut prover_set = false;
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--prover" => {
                let value = iter.next().ok_or("--prover needs a value")?;
                args.prover = match value.as_str() {
                    "lambda" => ProverKind::Lambda,
                    "p3" => ProverKind::P3,
                    other => return Err(format!("unknown prover: {other}")),
                };
                prover_set = true;
            }
            "--log-rows" => {
                let value = iter.next().ok_or("--log-rows needs a value")?;
                args.log_rows = value.parse().map_err(|_| "--log-rows: invalid u32")?;
            }
            "--num-sequences" => {
                let value = iter.next().ok_or("--num-sequences needs a value")?;
                args.num_sequences = value
                    .parse()
                    .map_err(|_| "--num-sequences: invalid usize")?;
            }
            "--blowup" => {
                let value = iter.next().ok_or("--blowup needs a value")?;
                args.blowup = value.parse().map_err(|_| "--blowup: invalid u8")?;
            }
            "--queries" => {
                let value = iter.next().ok_or("--queries needs a value")?;
                args.queries = value.parse().map_err(|_| "--queries: invalid usize")?;
            }
            "--grinding" => {
                let value = iter.next().ok_or("--grinding needs a value")?;
                args.grinding = value.parse().map_err(|_| "--grinding: invalid u8")?;
            }
            "--audit-only" => args.audit_only = true,
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
    if !args.blowup.is_power_of_two() {
        return Err("--blowup must be a power of two".into());
    }
    if args.queries == 0 {
        return Err("--queries must be > 0".into());
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

fn rows(args: &Args) -> usize {
    1usize << args.log_rows
}

fn main_cols(args: &Args) -> usize {
    2 * args.num_sequences
}

fn print_audit(args: &Args) {
    let rows = rows(args);
    let cols = main_cols(args);
    let trace_cells = rows * cols;
    match args.prover {
        ProverKind::Lambda => {
            println!(
                "AUDIT\tprover=lambda\tworkload=fib_pair\trows={rows}\tmain_cols={cols}\ttrace_cells={trace_cells}\tpublic_values={}\ttransition_constraints={}\tbase_transition_constraints={}\tboundary_constraints={}\taux_cols=0\tcomposition_chunks=1\tblowup={}\tqueries={}\tgrinding={}\ttrace_generation_timed=false\tverify_in_ratio=false",
                2 * args.num_sequences,
                2 * args.num_sequences,
                2 * args.num_sequences,
                2 * args.num_sequences,
                args.blowup,
                args.queries,
                args.grinding,
            );
        }
        ProverKind::P3 => {
            println!(
                "AUDIT\tprover=p3\tworkload=fib_pair\trows={rows}\tmain_cols={cols}\ttrace_cells={trace_cells}\tpublic_values={}\tair_constraints={}\tfirst_row_constraints={}\ttransition_constraints={}\tboundary_constraints=0\tquotient_chunks=1\tval_packing_width={}\thash_lanes={}\tblowup={}\tqueries={}\tgrinding={}\ttrace_generation_timed=false\tverify_in_ratio=false",
                2 * args.num_sequences,
                4 * args.num_sequences,
                2 * args.num_sequences,
                2 * args.num_sequences,
                plonky3_config::val_packing_width(),
                plonky3_config::hash_lanes(),
                args.blowup,
                args.queries,
                args.grinding,
            );
        }
    }
}

fn peak_rss_kb() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    let maxrss = unsafe { usage.assume_init().ru_maxrss };
    #[cfg(target_os = "macos")]
    {
        Some((maxrss as u64).div_ceil(1024))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Some(maxrss as u64)
    }
}

fn run_lambda(args: &Args) -> BenchMetrics {
    let setup_start = Instant::now();
    let rows = rows(args);
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
    let setup_s = setup_start.elapsed().as_secs_f64();

    let start = Instant::now();
    let proof = Prover::<F, E, _>::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("lambda prove failed");
    let prove_s = start.elapsed().as_secs_f64();

    let proof_size_bytes = serde_cbor::to_vec(&proof)
        .expect("lambda proof serialization failed")
        .len();

    let start = Instant::now();
    let verified = Verifier::<F, E, _>::verify(&proof, &air, &mut DefaultTranscript::<E>::new(&[]));
    let verify_s = start.elapsed().as_secs_f64();
    assert!(verified, "lambda verify failed");

    BenchMetrics {
        setup_s,
        prove_s,
        verify_s,
        proof_size_bytes,
        peak_rss_kb: peak_rss_kb(),
    }
}

fn run_p3(args: &Args) -> BenchMetrics {
    let setup_start = Instant::now();
    let rows = rows(args);
    let config = plonky3_config::params_config(args.blowup, args.queries, args.grinding);
    let air = plonky3_fibonacci::P3FibonacciAir {
        num_sequences: args.num_sequences,
    };
    let trace = plonky3_fibonacci::generate_fibonacci_trace(args.num_sequences, rows);
    let pis = plonky3_fibonacci::public_values(args.num_sequences);
    let setup_s = setup_start.elapsed().as_secs_f64();

    let start = Instant::now();
    let proof = p3_uni_stark::prove(&config, &air, trace, &pis);
    let prove_s = start.elapsed().as_secs_f64();

    let proof_size_bytes = serde_cbor::to_vec(&proof)
        .expect("p3 proof serialization failed")
        .len();

    let start = Instant::now();
    p3_uni_stark::verify(&config, &air, &proof, &pis).expect("p3 verify failed");
    let verify_s = start.elapsed().as_secs_f64();

    BenchMetrics {
        setup_s,
        prove_s,
        verify_s,
        proof_size_bytes,
        peak_rss_kb: peak_rss_kb(),
    }
}

fn print_metrics(args: &Args, metrics: &BenchMetrics) {
    let prover = match args.prover {
        ProverKind::Lambda => "lambda",
        ProverKind::P3 => "p3",
    };
    let rows = rows(args);
    let cols = main_cols(args);
    let cells_per_sec = (rows * cols) as f64 / metrics.prove_s;
    println!(
        "METRICS\tworkload=fib_pair\tprover={prover}\tlog_rows={}\trows={rows}\tnum_sequences={}\tmain_cols={cols}\tblowup={}\tfri_queries={}\tgrinding={}\tsetup_s={:.6}\tprove_s={:.6}\tverify_s={:.6}\tproof_size_bytes={}\tpeak_rss_kb={}\trows_per_sec={:.3}\tcells_per_sec={:.3}",
        args.log_rows,
        args.num_sequences,
        args.blowup,
        args.queries,
        args.grinding,
        metrics.setup_s,
        metrics.prove_s,
        metrics.verify_s,
        metrics.proof_size_bytes,
        metrics
            .peak_rss_kb
            .map(|v| v.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        rows as f64 / metrics.prove_s,
        cells_per_sec,
    );
}

fn real_main() -> Result<(), String> {
    let args = parse_args()?;
    print_audit(&args);
    if args.audit_only {
        return Ok(());
    }

    let metrics = match args.prover {
        ProverKind::Lambda => run_lambda(&args),
        ProverKind::P3 => run_p3(&args),
    };

    println!(
        "Proving time: {:.6}s, verify: {:.6}s, setup excluded from ratio: {:.6}s",
        metrics.prove_s, metrics.verify_s, metrics.setup_s
    );
    print_metrics(&args, &metrics);
    Ok(())
}

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            print_usage();
            eprintln!("error: {err}");
            ExitCode::from(2)
        }
    }
}
