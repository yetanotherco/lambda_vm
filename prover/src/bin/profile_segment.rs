//! Memory profiler for segmented proving.
//!
//! Usage:
//!   cargo run --bin profile_segment --release -- <elf_name> <segment_size>
//!
//! Example:
//!   cargo run --bin profile_segment --release -- loop_32768 64
//!   cargo run --bin profile_segment --release -- loop_32768 0   # no segmentation
//!
//! With dhat profiling (generates dhat-heap.json):
//!   cargo run --bin profile_segment --release --features dhat-heap -- loop_32768 64

use std::env;
use std::time::Instant;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::traits::AIR;

use lambda_vm_prover::segment::split_into_segments;
use lambda_vm_prover::tables::lt::generate_lt_trace;
use lambda_vm_prover::tables::trace_builder::Traces;
use lambda_vm_prover::test_utils::{
    E, F, collect_bitwise_ops_from_logs, collect_bitwise_ops_from_lt, collect_lt_lookups_from_logs,
    create_bitwise_air, create_cpu_air, create_lt_air, generate_minimal_bitwise_trace, run_asm_elf,
};

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <elf_name> <segment_size>", args[0]);
        eprintln!("Example: {} loop_32768 64", args[0]);
        eprintln!("         {} loop_32768 0   # no segmentation", args[0]);
        std::process::exit(1);
    }

    let elf_name = &args[1];
    let segment_size: usize = args[2].parse().expect("segment_size must be a number");

    if segment_size == 0 {
        run_without_segmentation(elf_name);
    } else {
        run_with_segmentation(elf_name, segment_size);
    }

    #[cfg(feature = "dhat-heap")]
    {
        println!("\ndhat profiling data saved to dhat-heap.json");
        println!("View with: dhat-viewer (https://nnethercote.github.io/dh_view/dh_view.html)");
    }
}

fn run_without_segmentation(elf_name: &str) {
    println!("=== Non-Segmented Proving Profiler ===");
    println!("Program: {}", elf_name);

    // Load and execute ELF
    let start = Instant::now();
    let (_elf, logs, instructions) = run_asm_elf(elf_name);
    let execution_time = start.elapsed();
    println!(
        "\nExecution: {} instructions in {:?}",
        logs.len(),
        execution_time
    );

    // Create AIRs
    let proof_options = ProofOptions::default_test_options();
    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);
    let lt_air = create_lt_air(&proof_options);

    // Generate traces for entire program
    println!("\n=== Proving (no segmentation) ===");
    let prove_start = Instant::now();

    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone())
        .expect("Failed to generate traces")
        .cpu;

    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_ops_from_logs(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_ops_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Traces: CPU {} rows, Bitwise {} rows, LT {} rows",
        cpu_trace.main_table.height, bitwise_trace.main_table.height, lt_trace.main_table.height
    );

    // Prove
    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&bitwise_air, &mut bitwise_trace, &()),
        (&lt_air, &mut lt_trace, &()),
    ];

    let _proof = Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("Prover failed");

    let total_prove_time = prove_start.elapsed();
    println!("\n=== Summary ===");
    println!("Total proving time: {:?}", total_prove_time);
}

fn run_with_segmentation(elf_name: &str, segment_size: usize) {
    println!("=== Segmented Proving Profiler ===");
    println!("Program: {}", elf_name);
    println!("Segment size: {} rows", segment_size);

    // Load and execute ELF
    let start = Instant::now();
    let (_elf, logs, instructions) = run_asm_elf(elf_name);
    let execution_time = start.elapsed();
    println!(
        "\nExecution: {} instructions in {:?}",
        logs.len(),
        execution_time
    );

    // Split into segments
    let segments = split_into_segments(&logs, segment_size).expect("Failed to split into segments");
    let num_segments = segments.len();
    println!("Segments: {} x {} rows each", num_segments, segment_size);

    // Create AIRs (reused for all segments)
    let proof_options = ProofOptions::default_test_options();
    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);
    let lt_air = create_lt_air(&proof_options);

    // Prove each segment
    println!("\n=== Proving {} segments ===", num_segments);
    let prove_start = Instant::now();

    for (i, segment_logs) in segments.iter().enumerate() {
        let seg_start = Instant::now();

        // Generate traces for this segment
        let mut cpu_trace = Traces::from_logs(segment_logs, instructions.clone())
            .expect("Failed to generate traces")
            .cpu;

        let lt_lookups = collect_lt_lookups_from_logs(segment_logs, &instructions);
        let mut lt_trace = generate_lt_trace(&lt_lookups);
        let mut bitwise_lookups = collect_bitwise_ops_from_logs(segment_logs, &instructions);
        bitwise_lookups.extend(collect_bitwise_ops_from_lt(&lt_lookups));
        let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

        // Prove segment
        let air_trace_pairs: Vec<(
            &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            _,
            _,
        )> = vec![
            (&cpu_air, &mut cpu_trace, &()),
            (&bitwise_air, &mut bitwise_trace, &()),
            (&lt_air, &mut lt_trace, &()),
        ];

        let _proof = Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
            .expect("Prover failed");

        let seg_time = seg_start.elapsed();
        println!(
            "  Segment {}/{}: {} rows, {:?}",
            i + 1,
            num_segments,
            segment_logs.len(),
            seg_time
        );
    }

    let total_prove_time = prove_start.elapsed();
    println!("\n=== Summary ===");
    println!("Total proving time: {:?}", total_prove_time);
    println!(
        "Average per segment: {:?}",
        total_prove_time / num_segments as u32
    );
}
