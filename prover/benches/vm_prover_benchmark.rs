//! Realistic VM prover benchmark with lookups.
//!
//! This benchmark exercises the full proving pipeline with real VM tables:
//! - CPU table (2^15 rows = 32768 instructions)
//! - Bitwise table (receives AND/OR/XOR/MSB lookups)
//! - LT table (receives less-than comparisons)
//!
//! Uses an actual ELF program that runs through the executor, generating
//! real execution traces with lookups.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use executor::vm::execution::run_program;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use math::field::element::FieldElement;
use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use prover::constraints::cpu::create_all_cpu_constraints;
use prover::tables::bitwise::{
    BitwiseLookup, bus_interactions as bitwise_bus_interactions, cols as bitwise_cols,
};
use prover::tables::cpu::{
    CpuOperation, bus_interactions as cpu_bus_interactions, cols as cpu_cols,
};
use prover::tables::lt::{
    LtOperation, bus_interactions as lt_bus_interactions, cols as lt_cols, generate_lt_trace,
};
use prover::tables::trace_builder::Traces;

type F = prover::tables::types::GoldilocksField;
type E = prover::tables::types::GoldilocksExtension;
type FE = FieldElement<F>;

/// Configuration for a benchmark case
struct BenchConfig {
    name: &'static str,
    elf_name: &'static str,
    expected_steps: usize,
}

impl BenchConfig {
    const fn new(name: &'static str, elf_name: &'static str, expected_steps: usize) -> Self {
        Self {
            name,
            elf_name,
            expected_steps,
        }
    }
}

/// Benchmark configurations
const CONFIGS: &[BenchConfig] = &[
    BenchConfig::new("vm_32k", "bench_32k", 32768), // 2^15 CPU rows.
];

/// Creates proof options suitable for benchmarking
fn benchmark_proof_options() -> ProofOptions {
    ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 30,
        coset_offset: 3,
        grinding_factor: 0,
    }
}

/// Helper to run an ELF from the program_artifacts directory
fn run_asm_elf(name: &str) -> (Vec<Log>, U64HashMap<Instruction>) {
    let path = format!(
        "{}/executor/program_artifacts/asm/{}.elf",
        env!("CARGO_MANIFEST_DIR").replace("/prover", ""),
        name
    );
    let elf_data = std::fs::read(&path).expect(&format!("Failed to read ELF: {}", path));
    let program = Elf::load(&elf_data).expect("Failed to load ELF");
    let result =
        run_program(program.image, program.entry_point, vec![]).expect("Failed to run program");
    (result.logs, result.instructions)
}

/// Collect bitwise lookups from logs for minimal table generation.
fn collect_bitwise_lookups_from_logs(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Vec<(BitwiseLookup, u8, u8, u8)> {
    logs.iter()
        .enumerate()
        .flat_map(|(i, log)| {
            let instruction = *instructions.get(&log.current_pc).unwrap();
            let op = CpuOperation::from_log(log, (i as u64) * 4, instruction);
            op.collect_bitwise_lookups()
        })
        .collect()
}

/// Collect LT lookups from executor logs.
fn collect_lt_lookups_from_logs(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Vec<LtOperation> {
    use executor::vm::instruction::decoding::{ArithOp, Comparison};

    let mut lookups = Vec::new();

    for log in logs {
        let instruction = *instructions.get(&log.current_pc).unwrap();

        let is_slt = matches!(
            &instruction,
            Instruction::Arith {
                op: ArithOp::SetLessThan,
                ..
            } | Instruction::Arith {
                op: ArithOp::SetLessThanU,
                ..
            } | Instruction::ArithImm {
                op: ArithOp::SetLessThan,
                ..
            } | Instruction::ArithImm {
                op: ArithOp::SetLessThanU,
                ..
            } | Instruction::ArithW {
                op: ArithOp::SetLessThan,
                ..
            } | Instruction::ArithW {
                op: ArithOp::SetLessThanU,
                ..
            } | Instruction::ArithImmW {
                op: ArithOp::SetLessThan,
                ..
            } | Instruction::ArithImmW {
                op: ArithOp::SetLessThanU,
                ..
            }
        );

        let is_blt = matches!(
            &instruction,
            Instruction::Branch {
                cond: Comparison::LessThan,
                ..
            } | Instruction::Branch {
                cond: Comparison::LessThanUnsigned,
                ..
            } | Instruction::Branch {
                cond: Comparison::GreaterOrEqual,
                ..
            } | Instruction::Branch {
                cond: Comparison::GreaterOrEqualUnsigned,
                ..
            }
        );

        if is_slt || is_blt {
            let signed = matches!(
                &instruction,
                Instruction::Arith {
                    op: ArithOp::SetLessThan,
                    ..
                } | Instruction::ArithImm {
                    op: ArithOp::SetLessThan,
                    ..
                } | Instruction::ArithW {
                    op: ArithOp::SetLessThan,
                    ..
                } | Instruction::ArithImmW {
                    op: ArithOp::SetLessThan,
                    ..
                } | Instruction::Branch {
                    cond: Comparison::LessThan,
                    ..
                } | Instruction::Branch {
                    cond: Comparison::GreaterOrEqual,
                    ..
                }
            );

            let arg1 = log.src1_val;
            let arg2 = match &instruction {
                Instruction::ArithImm { imm, .. } | Instruction::ArithImmW { imm, .. } => {
                    *imm as i64 as u64
                }
                _ => log.src2_val,
            };

            lookups.push(LtOperation::new(arg1, arg2, signed));
        }
    }

    lookups
}

/// Collect bitwise lookups from LT operations (MSB16 and IS_HALFWORD).
fn collect_bitwise_lookups_from_lt(lt_ops: &[LtOperation]) -> Vec<(BitwiseLookup, u8, u8, u8)> {
    let mut lookups = Vec::new();

    for op in lt_ops {
        // MSB16 for lhs_msb (bits 48-63 of lhs)
        let lhs_2 = ((op.lhs >> 48) & 0xFFFF) as u16;
        let x = (lhs_2 & 0xFF) as u8;
        let y = ((lhs_2 >> 8) & 0xFF) as u8;
        lookups.push((BitwiseLookup::Msb16, x, y, 0));

        // MSB16 for rhs_msb (bits 48-63 of rhs)
        let rhs_2 = ((op.rhs >> 48) & 0xFFFF) as u16;
        let x = (rhs_2 & 0xFF) as u8;
        let y = ((rhs_2 >> 8) & 0xFF) as u8;
        lookups.push((BitwiseLookup::Msb16, x, y, 0));

        // IS_HALFWORD for lhs_sub_rhs (4 halfwords)
        let lhs_sub_rhs = op.lhs.wrapping_sub(op.rhs);
        for shift in [0, 16, 32, 48] {
            let half = ((lhs_sub_rhs >> shift) & 0xFFFF) as u16;
            lookups.push((
                BitwiseLookup::IsHalf,
                (half & 0xFF) as u8,
                ((half >> 8) & 0xFF) as u8,
                0,
            ));
        }
    }

    lookups
}

/// Generate a minimal bitwise trace containing only the rows needed for the given lookups.
///
/// This is much faster than the full 2^20 row table for benchmarking.
fn generate_minimal_bitwise_trace(lookups: &[(BitwiseLookup, u8, u8, u8)]) -> TraceTable<F, E> {
    // Collect unique (x, y, z) tuples and count multiplicities per lookup type
    let mut row_data: HashMap<(u8, u8, u8), [u64; 11]> = HashMap::new();

    for (lookup_type, x, y, z) in lookups {
        let key = (*x, *y, *z);
        let mu_idx = match lookup_type {
            BitwiseLookup::AndByte => 0,
            BitwiseLookup::OrByte => 1,
            BitwiseLookup::XorByte => 2,
            BitwiseLookup::Msb8 => 3,
            BitwiseLookup::Msb16 => 4,
            BitwiseLookup::Zero => 5,
            BitwiseLookup::IsByte => 6,
            BitwiseLookup::IsHalf => 7,
            BitwiseLookup::IsB20 => 8,
            BitwiseLookup::Hwsl => 9,
            BitwiseLookup::Hwslc => 10,
        };
        row_data.entry(key).or_insert([0; 11])[mu_idx] += 1;
    }

    // Need at least 4 rows for FRI, pad to power of 2
    let unique_rows: Vec<_> = row_data.keys().cloned().collect();
    let num_rows = unique_rows.len().max(4).next_power_of_two();

    let mut data = vec![FE::zero(); num_rows * bitwise_cols::NUM_COLUMNS];

    for (row_idx, (x, y, z)) in unique_rows.iter().enumerate() {
        let base = row_idx * bitwise_cols::NUM_COLUMNS;
        let x = *x as u32;
        let y = *y as u32;
        let z = *z as u32;

        // Input columns
        data[base + bitwise_cols::X] = FE::from(x as u64);
        data[base + bitwise_cols::Y] = FE::from(y as u64);
        data[base + bitwise_cols::Z] = FE::from(z as u64);

        // Bitwise operation results
        data[base + bitwise_cols::AND] = FE::from((x & y) as u64);
        data[base + bitwise_cols::OR] = FE::from((x | y) as u64);
        data[base + bitwise_cols::XOR] = FE::from((x ^ y) as u64);

        // MSB extractions
        let msb8 = (x >> 7) & 1;
        let halfword = x + y * 256;
        let msb16 = (halfword >> 15) & 1;
        data[base + bitwise_cols::MSB8] = FE::from(msb8 as u64);
        data[base + bitwise_cols::MSB16] = FE::from(msb16 as u64);

        // Zero check
        let is_zero = if x == 0 && y == 0 { 1u64 } else { 0u64 };
        data[base + bitwise_cols::ZERO] = FE::from(is_zero);

        // Shift operations
        let sll = if z == 0 {
            halfword
        } else {
            (halfword << z) & 0xFFFF
        };
        let sllc = if z == 0 { 0 } else { halfword >> (16 - z) };
        data[base + bitwise_cols::SLL] = FE::from(sll as u64);
        data[base + bitwise_cols::SLLC] = FE::from(sllc as u64);

        // Multiplicity columns
        let mus = &row_data[&(x as u8, y as u8, z as u8)];
        data[base + bitwise_cols::MU_AND] = FE::from(mus[0]);
        data[base + bitwise_cols::MU_OR] = FE::from(mus[1]);
        data[base + bitwise_cols::MU_XOR] = FE::from(mus[2]);
        data[base + bitwise_cols::MU_MSB8] = FE::from(mus[3]);
        data[base + bitwise_cols::MU_MSB16] = FE::from(mus[4]);
        data[base + bitwise_cols::MU_ZERO] = FE::from(mus[5]);
        data[base + bitwise_cols::MU_IS_BYTE] = FE::from(mus[6]);
        data[base + bitwise_cols::MU_IS_HALF] = FE::from(mus[7]);
        data[base + bitwise_cols::MU_IS_B20] = FE::from(mus[8]);
        data[base + bitwise_cols::MU_HWSL] = FE::from(mus[9]);
        data[base + bitwise_cols::MU_HWSLC] = FE::from(mus[10]);
    }

    TraceTable::new_main(data, bitwise_cols::NUM_COLUMNS, 1)
}

// =============================================================================
// AIR creation helpers
// =============================================================================

fn create_cpu_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    // Get all CPU constraints
    let (is_bit, add, other, _) = create_all_cpu_constraints();

    // All CPU constraints
    let mut transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = Vec::new();
    for c in is_bit {
        transition_constraints.push(Box::new(c));
    }
    for c in add {
        transition_constraints.push(Box::new(c));
    }
    for c in other {
        transition_constraints.push(c);
    }

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: cpu_bus_interactions(),
    };

    AirWithBuses::new(
        cpu_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

fn create_bitwise_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: bitwise_bus_interactions(),
    };

    AirWithBuses::new(
        bitwise_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

fn create_lt_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: lt_bus_interactions(),
    };

    AirWithBuses::new(
        lt_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

/// Generate all traces from ELF execution.
fn generate_traces_from_elf(
    elf_name: &str,
    expected_steps: usize,
) -> (
    TraceTable<F, E>,
    TraceTable<F, E>,
    TraceTable<F, E>,
    usize, // bitwise lookups count
    usize, // lt ops count
) {
    // Run the ELF program
    let (logs, instructions) = run_asm_elf(elf_name);
    assert_eq!(
        logs.len(),
        expected_steps,
        "{}.elf should have {} steps, got {}",
        elf_name,
        expected_steps,
        logs.len()
    );

    // Generate CPU trace using Traces::from_logs
    let traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let cpu_trace = traces.cpu;

    // Collect LT operations and generate LT trace
    let lt_ops = collect_lt_lookups_from_logs(&logs, &instructions);
    let lt_trace = generate_lt_trace(&lt_ops);

    // Collect all bitwise lookups (from CPU + from LT table)
    let mut bitwise_lookups = collect_bitwise_lookups_from_logs(&logs, &instructions);
    let lt_bitwise_lookups = collect_bitwise_lookups_from_lt(&lt_ops);
    bitwise_lookups.extend(lt_bitwise_lookups);

    // Generate minimal bitwise trace for speed
    let bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    let bitwise_count = bitwise_lookups.len();
    let lt_count = lt_ops.len();

    (cpu_trace, bitwise_trace, lt_trace, bitwise_count, lt_count)
}

/// Benchmark proving for a configuration
fn bench_prove(c: &mut Criterion, group_name: &str, config: &BenchConfig) {
    let proof_options = benchmark_proof_options();

    c.bench_with_input(
        BenchmarkId::new(format!("{}/prove", group_name), config.name),
        config,
        |b, config| {
            b.iter_with_setup(
                || {
                    let (cpu_trace, bitwise_trace, lt_trace, _, _) =
                        generate_traces_from_elf(config.elf_name, config.expected_steps);
                    let cpu_air = create_cpu_air(&proof_options);
                    let bitwise_air = create_bitwise_air(&proof_options);
                    let lt_air = create_lt_air(&proof_options);
                    (
                        cpu_trace,
                        bitwise_trace,
                        lt_trace,
                        cpu_air,
                        bitwise_air,
                        lt_air,
                    )
                },
                |(mut cpu_trace, mut bitwise_trace, mut lt_trace, cpu_air, bitwise_air, lt_air)| {
                    let air_trace_pairs: Vec<(
                        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
                        _,
                        _,
                    )> = vec![
                        (&cpu_air, &mut cpu_trace, &()),
                        (&bitwise_air, &mut bitwise_trace, &()),
                        (&lt_air, &mut lt_trace, &()),
                    ];

                    Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
                        .unwrap()
                },
            )
        },
    );
}

/// Benchmark verification for a configuration
fn bench_verify(c: &mut Criterion, group_name: &str, config: &BenchConfig) {
    let proof_options = benchmark_proof_options();

    // Pre-generate the proof
    let (mut cpu_trace, mut bitwise_trace, mut lt_trace, bitwise_count, lt_count) =
        generate_traces_from_elf(config.elf_name, config.expected_steps);

    println!(
        "{}: CPU {} rows, Bitwise {} rows, LT {} rows, {} bitwise lookups, {} lt ops",
        config.name,
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        lt_trace.main_table.height,
        bitwise_count,
        lt_count,
    );

    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);
    let lt_air = create_lt_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&bitwise_air, &mut bitwise_trace, &()),
        (&lt_air, &mut lt_trace, &()),
    ];

    let proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    c.bench_with_input(
        BenchmarkId::new(format!("{}/verify", group_name), config.name),
        &(proof, cpu_air, bitwise_air, lt_air),
        |b, (proof, cpu_air, bitwise_air, lt_air)| {
            b.iter(|| {
                let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
                    vec![cpu_air, bitwise_air, lt_air];
                Verifier::multi_verify(&airs, proof, &mut DefaultTranscript::<E>::new(&[]))
            })
        },
    );
}

/// VM prover benchmarks
fn vm_prover_benchmarks(c: &mut Criterion) {
    for config in CONFIGS {
        bench_prove(c, "vm_prover", config);
        bench_verify(c, "vm_prover", config);
    }
}

criterion_group! {
    name = vm_prover;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(60));
    targets = vm_prover_benchmarks
}

criterion_main!(vm_prover);
