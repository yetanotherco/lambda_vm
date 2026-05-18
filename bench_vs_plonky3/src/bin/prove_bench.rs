//! Minimal wall-clock benchmark harness for Lambda STARK vs Plonky3.
//!
//! Builds the same Fibonacci AIR as `instruments_breakdown` (but without any
//! instrumentation) and prints human-readable timings plus one tab-separated
//! `METRICS` line, suitable for parsing by `bench_vs_plonky3/run.sh`.
//!
//! Usage:
//!   prove_bench --prover {lambda|p3} [--log-rows K] [--num-sequences N]
//!               [--blowup B] [--queries Q] [--grinding G] [--breakdown]
//!
//! Defaults match production (`GoldilocksCubicProofOptions::with_blowup(2)`):
//!   log-rows=19, num-sequences=16, blowup=2, queries=219, grinding=0.

use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use bench_vs_plonky3::{
    lambda_fibonacci_pair, lambda_quadratic_pair, plonky3_config, plonky3_fibonacci,
};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::verifier::{IsStarkVerifier, Verifier};
use tracing_subscriber::layer::SubscriberExt;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProverKind {
    Lambda,
    LambdaChunks,
    P3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Workload {
    /// Fibonacci-pair (d_max=1) — degenerate chunks case, exercises the
    /// `num_chunks=1` short-circuit. Both Lambda and P3 implementations exist.
    FibPair,
    /// Quadratic-pair (d_max=2, num_chunks=2) — first non-degenerate chunks
    /// case. Lambda only; no P3 implementation. Used to validate chunks payoff
    /// vs single-H on a workload where single-H pays `decompose_and_extend_d2`.
    QuadraticPair,
}

struct Args {
    prover: ProverKind,
    workload: Workload,
    log_rows: u32,
    num_sequences: usize,
    blowup: u8,
    queries: usize,
    grinding: u8,
    breakdown: bool,
}

struct BenchMetrics {
    prove_s: f64,
    verify_s: f64,
    proof_size_bytes: usize,
    peak_rss_kb: Option<u64>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            prover: ProverKind::Lambda,
            workload: Workload::FibPair,
            log_rows: 19,
            num_sequences: 16,
            blowup: 2,
            queries: 219,
            grinding: 0,
            breakdown: false,
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage: prove_bench --prover {{lambda|lambda-chunks|p3}} \
         [--workload {{fib_pair|quadratic_pair}}] \
         [--log-rows K] [--num-sequences N] \
         [--blowup B] [--queries Q] [--grinding G] [--breakdown]"
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
                    "lambda-chunks" => ProverKind::LambdaChunks,
                    "p3" => ProverKind::P3,
                    other => return Err(format!("unknown prover: {other}")),
                };
                prover_set = true;
            }
            "--workload" => {
                let v = iter.next().ok_or("--workload needs a value")?;
                args.workload = match v.as_str() {
                    "fib_pair" => Workload::FibPair,
                    "quadratic_pair" => Workload::QuadraticPair,
                    other => return Err(format!("unknown workload: {other}")),
                };
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
            "--breakdown" => {
                args.breakdown = true;
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
    if !args.blowup.is_power_of_two() {
        return Err("--blowup must be a power of two".into());
    }
    if args.queries == 0 {
        return Err("--queries must be > 0".into());
    }
    if args.prover == ProverKind::P3 && args.workload != Workload::FibPair {
        return Err(
            "--prover p3 only supports --workload fib_pair (no P3 quadratic_pair AIR yet)".into(),
        );
    }
    Ok(args)
}

fn proof_options(args: &Args) -> ProofOptions {
    ProofOptions {
        blowup_factor: args.blowup,
        fri_number_of_queries: args.queries,
        coset_offset: 3,
        grinding_factor: args.grinding,
        fri_last_layer_degree_bound: 0,
        fri_folding_factor: 2,
    }
}

fn ms(seconds: f64) -> f64 {
    seconds * 1000.0
}

fn print_breakdown(
    prover: &str,
    log_rows: u32,
    rows: usize,
    phase: &str,
    elapsed_ms: f64,
    extra: &str,
) {
    println!(
        "BREAKDOWN\tprover={prover}\tlog_rows={log_rows}\trows={rows}\tphase={phase}\tms={elapsed_ms:.3}{extra}"
    );
}

#[cfg(feature = "instruments")]
fn emit_lambda_breakdown(args: &Args, rows: usize, total_ms: f64) {
    print_breakdown("lambda", args.log_rows, rows, "prove_total", total_ms, "");

    if let Some(timing) = stark::instruments::take() {
        print_breakdown(
            "lambda",
            args.log_rows,
            rows,
            "prepass",
            ms(timing.prepass.as_secs_f64()),
            "",
        );
        print_breakdown(
            "lambda",
            args.log_rows,
            rows,
            "main_commits",
            ms(timing.main_commits.as_secs_f64()),
            "",
        );
        print_breakdown(
            "lambda",
            args.log_rows,
            rows,
            "aux_build",
            ms(timing.aux_build.as_secs_f64()),
            "",
        );
        print_breakdown(
            "lambda",
            args.log_rows,
            rows,
            "aux_commit",
            ms(timing.aux_commit.as_secs_f64()),
            "",
        );
        print_breakdown(
            "lambda",
            args.log_rows,
            rows,
            "rounds_2_4",
            ms(timing.rounds_2_4.as_secs_f64()),
            "",
        );

        let r1 = timing.round1_sub;
        print_breakdown(
            "lambda",
            args.log_rows,
            rows,
            "r1_main_lde",
            ms(r1.main_lde.as_secs_f64()),
            "",
        );
        print_breakdown(
            "lambda",
            args.log_rows,
            rows,
            "r1_main_merkle",
            ms(r1.main_merkle.as_secs_f64()),
            "",
        );
        print_breakdown(
            "lambda",
            args.log_rows,
            rows,
            "r1_aux_lde",
            ms(r1.aux_lde.as_secs_f64()),
            "",
        );
        print_breakdown(
            "lambda",
            args.log_rows,
            rows,
            "r1_aux_merkle",
            ms(r1.aux_merkle.as_secs_f64()),
            "",
        );

        for (name, table_rows, dur, sub) in timing.table_timings {
            let extra = format!("\ttable={name}\ttable_rows={table_rows}");
            print_breakdown(
                "lambda",
                args.log_rows,
                rows,
                "table_total",
                ms(dur.as_secs_f64()),
                &extra,
            );
            print_breakdown(
                "lambda",
                args.log_rows,
                rows,
                "r2_constraints",
                ms(sub.constraints.as_secs_f64()),
                &extra,
            );
            print_breakdown(
                "lambda",
                args.log_rows,
                rows,
                "r2_comp_decompose",
                ms(sub.comp_decompose.as_secs_f64()),
                &extra,
            );
            print_breakdown(
                "lambda",
                args.log_rows,
                rows,
                "r2_comp_commit",
                ms(sub.comp_commit.as_secs_f64()),
                &extra,
            );
            print_breakdown(
                "lambda",
                args.log_rows,
                rows,
                "r3_ood",
                ms(sub.ood.as_secs_f64()),
                &extra,
            );
            print_breakdown(
                "lambda",
                args.log_rows,
                rows,
                "r4_deep_comp",
                ms(sub.deep_comp.as_secs_f64()),
                &extra,
            );
            print_breakdown(
                "lambda",
                args.log_rows,
                rows,
                "r4_deep_extend",
                ms(sub.deep_extend.as_secs_f64()),
                &extra,
            );
            print_breakdown(
                "lambda",
                args.log_rows,
                rows,
                "r4_fri_commit",
                ms(sub.fri_commit.as_secs_f64()),
                &extra,
            );
            print_breakdown(
                "lambda",
                args.log_rows,
                rows,
                "r4_queries",
                ms(sub.queries.as_secs_f64()),
                &extra,
            );
        }
    }
}

#[cfg(not(feature = "instruments"))]
fn emit_lambda_breakdown(args: &Args, rows: usize, total_ms: f64) {
    print_breakdown("lambda", args.log_rows, rows, "prove_total", total_ms, "");
    eprintln!("warning: Lambda phase breakdown requires building with --features instruments");
}

struct SpanState {
    name: String,
    active_since: Option<Instant>,
    accumulated: std::time::Duration,
}

struct P3TimingLayer {
    spans: Mutex<std::collections::HashMap<u64, SpanState>>,
    results: Arc<Mutex<Vec<(String, f64)>>>,
}

impl<S: tracing::Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>>
    tracing_subscriber::Layer<S> for P3TimingLayer
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        self.spans.lock().unwrap().insert(
            id.into_u64(),
            SpanState {
                name: attrs.metadata().name().to_string(),
                active_since: None,
                accumulated: std::time::Duration::ZERO,
            },
        );
    }

    fn on_enter(&self, id: &tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(entry) = self.spans.lock().unwrap().get_mut(&id.into_u64())
            && entry.active_since.is_none()
        {
            entry.active_since = Some(Instant::now());
        }
    }

    fn on_exit(&self, id: &tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(entry) = self.spans.lock().unwrap().get_mut(&id.into_u64())
            && let Some(start) = entry.active_since.take()
        {
            entry.accumulated += start.elapsed();
        }
    }

    fn on_close(&self, id: tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(entry) = self.spans.lock().unwrap().remove(&id.into_u64()) {
            let mut total = entry.accumulated;
            if let Some(start) = entry.active_since {
                total += start.elapsed();
            }
            self.results
                .lock()
                .unwrap()
                .push((entry.name, ms(total.as_secs_f64())));
        }
    }
}

type P3SpanResults = Arc<Mutex<Vec<(String, f64)>>>;

fn p3_span_subscriber() -> (impl tracing::Subscriber + Send + Sync, P3SpanResults) {
    let results = Arc::new(Mutex::new(Vec::new()));
    let layer = P3TimingLayer {
        spans: Mutex::new(std::collections::HashMap::new()),
        results: Arc::clone(&results),
    };
    let filter = tracing_subscriber::filter::LevelFilter::DEBUG;
    (
        tracing_subscriber::registry().with(filter).with(layer),
        results,
    )
}

fn peak_rss_kb() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes `usage` when it returns 0.
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
    match args.workload {
        Workload::FibPair => run_lambda_fib_pair(args),
        Workload::QuadraticPair => run_lambda_quadratic_pair(args),
    }
}

fn run_lambda_fib_pair(args: &Args) -> BenchMetrics {
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
    let prove_s = start.elapsed().as_secs_f64();
    if args.breakdown {
        emit_lambda_breakdown(args, rows, ms(prove_s));
    }

    let proof_size_bytes = serde_cbor::to_vec(&_proof)
        .expect("lambda proof serialization failed")
        .len();

    let start = Instant::now();
    let verified =
        Verifier::<F, E, _>::verify(&_proof, &air, &mut DefaultTranscript::<E>::new(&[]));
    let verify_s = start.elapsed().as_secs_f64();
    assert!(verified, "lambda verify failed");

    BenchMetrics {
        prove_s,
        verify_s,
        proof_size_bytes,
        peak_rss_kb: peak_rss_kb(),
    }
}

fn run_lambda_quadratic_pair(args: &Args) -> BenchMetrics {
    let rows = 1usize << args.log_rows;
    let options = proof_options(args);

    let initial_values: Vec<(FE, FE)> = (0..args.num_sequences)
        .map(|i| (FE::from((i + 1) as u64), FE::from((i + 3) as u64)))
        .collect();

    let mut trace = lambda_quadratic_pair::compute_trace::<F, E>(&initial_values, rows);
    let pub_inputs = lambda_quadratic_pair::create_public_inputs(initial_values);
    let air = lambda_quadratic_pair::QuadraticPairMultiColAIR::<F, E>::with_num_sequences(
        &options,
        args.num_sequences,
    );

    let start = Instant::now();
    let proof = Prover::<F, E, _>::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("lambda prove failed (quadratic_pair single-H)");
    let prove_s = start.elapsed().as_secs_f64();
    if args.breakdown {
        emit_lambda_breakdown(args, rows, ms(prove_s));
    }

    let proof_size_bytes = serde_cbor::to_vec(&proof)
        .expect("lambda proof serialization failed")
        .len();

    let start = Instant::now();
    let verified =
        Verifier::<F, E, _>::verify(&proof, &air, &mut DefaultTranscript::<E>::new(&[]));
    let verify_s = start.elapsed().as_secs_f64();
    assert!(verified, "lambda verify failed (quadratic_pair single-H)");

    BenchMetrics {
        prove_s,
        verify_s,
        proof_size_bytes,
        peak_rss_kb: peak_rss_kb(),
    }
}

/// Phase 5.1 chunks-protocol Lambda runner.
///
/// Mirrors [`run_lambda`] but invokes `Prover::prove_chunks` and
/// `Verifier::verify_chunks`. For fib_pair (d_max=1) this exercises the
/// degenerate `num_chunks=1` short-circuit; for quadratic_pair (d_max=2)
/// it exercises `num_chunks=2` — the first non-degenerate chunks case
/// where the multi-chunk recompose actually runs.
fn run_lambda_chunks(args: &Args) -> BenchMetrics {
    match args.workload {
        Workload::FibPair => run_lambda_chunks_fib_pair(args),
        Workload::QuadraticPair => run_lambda_chunks_quadratic_pair(args),
    }
}

fn run_lambda_chunks_fib_pair(args: &Args) -> BenchMetrics {
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
    let proof = Prover::<F, E, _>::prove_chunks(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("lambda chunks prove failed (fib_pair)");
    let prove_s = start.elapsed().as_secs_f64();
    if args.breakdown {
        print_breakdown(
            "lambda-chunks",
            args.log_rows,
            rows,
            "prove_total",
            ms(prove_s),
            "",
        );
    }

    let proof_size_bytes = serde_cbor::to_vec(&proof)
        .expect("lambda chunks proof serialization failed")
        .len();

    let start = Instant::now();
    let verified =
        Verifier::<F, E, _>::verify_chunks(&proof, &air, &mut DefaultTranscript::<E>::new(&[]));
    let verify_s = start.elapsed().as_secs_f64();
    assert!(verified, "lambda chunks verify failed (fib_pair)");

    BenchMetrics {
        prove_s,
        verify_s,
        proof_size_bytes,
        peak_rss_kb: peak_rss_kb(),
    }
}

fn run_lambda_chunks_quadratic_pair(args: &Args) -> BenchMetrics {
    let rows = 1usize << args.log_rows;
    let options = proof_options(args);

    let initial_values: Vec<(FE, FE)> = (0..args.num_sequences)
        .map(|i| (FE::from((i + 1) as u64), FE::from((i + 3) as u64)))
        .collect();

    let mut trace = lambda_quadratic_pair::compute_trace::<F, E>(&initial_values, rows);
    let pub_inputs = lambda_quadratic_pair::create_public_inputs(initial_values);
    let air = lambda_quadratic_pair::QuadraticPairMultiColAIR::<F, E>::with_num_sequences(
        &options,
        args.num_sequences,
    );

    let start = Instant::now();
    let proof = Prover::<F, E, _>::prove_chunks(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("lambda chunks prove failed (quadratic_pair, num_chunks=2)");
    let prove_s = start.elapsed().as_secs_f64();
    if args.breakdown {
        print_breakdown(
            "lambda-chunks",
            args.log_rows,
            rows,
            "prove_total",
            ms(prove_s),
            "",
        );
    }

    let proof_size_bytes = serde_cbor::to_vec(&proof)
        .expect("lambda chunks proof serialization failed")
        .len();

    let start = Instant::now();
    let verified =
        Verifier::<F, E, _>::verify_chunks(&proof, &air, &mut DefaultTranscript::<E>::new(&[]));
    let verify_s = start.elapsed().as_secs_f64();
    assert!(verified, "lambda chunks verify failed (quadratic_pair)");

    BenchMetrics {
        prove_s,
        verify_s,
        proof_size_bytes,
        peak_rss_kb: peak_rss_kb(),
    }
}

fn run_p3(args: &Args) -> BenchMetrics {
    let rows = 1usize << args.log_rows;
    let config = plonky3_config::params_config(args.blowup, args.queries, args.grinding);
    let air = plonky3_fibonacci::P3FibonacciAir {
        num_sequences: args.num_sequences,
    };
    let trace = plonky3_fibonacci::generate_fibonacci_trace(args.num_sequences, rows);
    let pis = plonky3_fibonacci::public_values(args.num_sequences);

    let (prove_s, _proof, span_results) = if args.breakdown {
        let (subscriber, results) = p3_span_subscriber();
        let start = Instant::now();
        let proof = {
            let _guard = tracing::subscriber::set_default(subscriber);
            p3_uni_stark::prove(&config, &air, trace, &pis)
        };
        (start.elapsed().as_secs_f64(), proof, Some(results))
    } else {
        let start = Instant::now();
        let proof = p3_uni_stark::prove(&config, &air, trace, &pis);
        (start.elapsed().as_secs_f64(), proof, None)
    };

    if args.breakdown {
        print_breakdown("p3", args.log_rows, rows, "prove_total", ms(prove_s), "");
        if let Some(results) = span_results {
            let mut span_data = results.lock().unwrap().clone();
            span_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (name, elapsed_ms) in span_data {
                if elapsed_ms >= 0.1 {
                    let extra = format!("\tspan={name}");
                    print_breakdown("p3", args.log_rows, rows, "span", elapsed_ms, &extra);
                }
            }
        }
    }

    let proof_size_bytes = serde_cbor::to_vec(&_proof)
        .expect("p3 proof serialization failed")
        .len();

    let start = Instant::now();
    p3_uni_stark::verify(&config, &air, &_proof, &pis).expect("p3 verify failed");
    let verify_s = start.elapsed().as_secs_f64();

    BenchMetrics {
        prove_s,
        verify_s,
        proof_size_bytes,
        peak_rss_kb: peak_rss_kb(),
    }
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

    let metrics = match args.prover {
        ProverKind::Lambda => run_lambda(&args),
        ProverKind::LambdaChunks => run_lambda_chunks(&args),
        ProverKind::P3 => run_p3(&args),
    };

    let prover_name = match args.prover {
        ProverKind::Lambda => "lambda",
        ProverKind::LambdaChunks => "lambda-chunks",
        ProverKind::P3 => "p3",
    };
    let workload_name = match args.workload {
        Workload::FibPair => "fib_pair",
        Workload::QuadraticPair => "quadratic_pair",
    };
    let rows = 1usize << args.log_rows;
    let main_cols = 2 * args.num_sequences;
    let aux_cols = 0usize;
    let cells = rows * main_cols;
    let rows_per_sec = rows as f64 / metrics.prove_s;
    let cells_per_sec = cells as f64 / metrics.prove_s;
    let peak_rss_kb = metrics
        .peak_rss_kb
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());

    println!("Proving time: {:.6}s", metrics.prove_s);
    println!("Verification time: {:.6}s", metrics.verify_s);
    println!("Proof size: {} bytes", metrics.proof_size_bytes);
    println!("Peak RSS: {peak_rss_kb} KB");
    println!(
        "METRICS\tworkload={workload_name}\tprover={prover_name}\tlog_rows={}\trows={rows}\t\
         num_sequences={}\tmain_cols={main_cols}\taux_cols={aux_cols}\ttables=1\t\
         logup=false\tblowup={}\tfri_queries={}\tgrinding={}\tprove_s={:.6}\t\
         verify_s={:.6}\tproof_size_bytes={}\tpeak_rss_kb={peak_rss_kb}\t\
         rows_per_sec={:.3}\tcells_per_sec={:.3}",
        args.log_rows,
        args.num_sequences,
        args.blowup,
        args.queries,
        args.grinding,
        metrics.prove_s,
        metrics.verify_s,
        metrics.proof_size_bytes,
        rows_per_sec,
        cells_per_sec,
    );
    ExitCode::SUCCESS
}
