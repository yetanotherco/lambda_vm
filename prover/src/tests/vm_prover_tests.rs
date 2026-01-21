//! VM Prover integration tests using multi_prove.
//!
//! These tests verify the full prover pipeline:
//! - Run ELF through executor
//! - Generate traces for CPU and Bitwise tables
//! - Use multi_prove/multi_verify with bus interactions
//!
//! Wired buses:
//! - CPU sends AND_BYTE, OR_BYTE, XOR_BYTE to Bitwise (×8 each)
//! - CPU sends MSB16 to Bitwise (for rv1_sign_bit, arg2_sign_bit when word_instr=1)
//! - CPU sends MSB8 to Bitwise (for res_sign_bit when word_instr=1)
//! - CPU sends ZERO to Bitwise (for is_equal when BEQ=1)
//!
//! TODO: LT bus (needs LT table integration)

use crypto::fiat_shamir::default_transcript::DefaultTranscript;

use executor::{elf::Elf, vm::execution::run_program};

use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables64::bitwise::{
    bus_interactions as bitwise_bus_interactions, cols as bitwise_cols, generate_bitwise_trace,
    update_multiplicities, BitwiseLookup,
};
use crate::tables64::cpu::{
    bus_interactions as cpu_bus_interactions, collect_bitwise_lookups_from_logs,
    generate_cpu_trace_from_logs,
};
use crate::tables64::types::{GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Helper to run an ELF from the program_artifacts directory
fn run_asm_elf(name: &str) -> Vec<executor::vm::logs::Log> {
    let path = format!(
        "{}/executor/program_artifacts/asm/{}.elf",
        env!("CARGO_MANIFEST_DIR").replace("/prover", ""),
        name
    );
    let elf_data = std::fs::read(&path).expect("Failed to read ELF");
    let program = Elf::load(&elf_data).expect("Failed to load ELF");
    let (_results, logs) =
        run_program(program.image, program.entry_point, vec![]).expect("Failed to run program");
    logs
}

// =============================================================================
// AIR creation helpers
// =============================================================================

fn create_cpu_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, stark::lookup::NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: cpu_bus_interactions(),
    };

    AirWithBuses::new(
        crate::tables64::cpu::cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

fn create_bitwise_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, stark::lookup::NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: bitwise_bus_interactions(),
    };

    AirWithBuses::new(
        crate::tables64::bitwise::cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

// =============================================================================
// Prover test helpers
// =============================================================================

/// Run multi_prove and multi_verify for CPU + Bitwise tables.
///
/// Returns true if verification succeeds.
fn prove_and_verify_vm(
    cpu_trace: &mut TraceTable<F, E>,
    bitwise_trace: &mut TraceTable<F, E>,
) -> bool {
    let proof_options = ProofOptions::default_test_options();

    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, cpu_trace, &()),
        (&bitwise_air, bitwise_trace, &()),
    ];

    let multi_proof =
        match Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])) {
            Ok(proof) => proof,
            Err(e) => {
                eprintln!("Prover error: {:?}", e);
                return false;
            }
        };

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &bitwise_air];

    Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]))
}

// =============================================================================
// Integration tests
// =============================================================================

/// Test CPU table alone (no bus interactions) to verify basic prove/verify works.
#[test]
fn test_cpu_only_no_bus() {
    let logs = run_asm_elf("lui");
    assert_eq!(logs.len(), 2);

    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);
    println!("CPU trace: {} rows x {} cols", cpu_trace.main_table.height, cpu_trace.main_table.width);

    let proof_options = ProofOptions::default_test_options();

    // Create AIR with NO bus interactions
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![], // NO bus interactions
    };
    let cpu_air: AirWithBuses<F, E, stark::lookup::NullBoundaryConstraintBuilder, ()> =
        AirWithBuses::new(
            crate::tables64::cpu::cols::NUM_COLUMNS,
            auxiliary_trace_build_data,
            &proof_options,
            1,
            transition_constraints,
        );

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&cpu_air, &mut cpu_trace, &())];

    let multi_proof = Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("Prover failed");

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![&cpu_air];
    assert!(
        Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "CPU-only verification failed"
    );
}

#[test]
#[ignore] // Slow: run with `make test_all`
fn test_vm_prover_lui() {
    // LUI program: 2 steps (power of 2)
    // Only uses ADD (for LUI), no bitwise operations (AND=0, OR=0, XOR=0)
    let logs = run_asm_elf("lui");
    assert_eq!(logs.len(), 2, "lui.elf should have 2 steps");

    // Generate CPU trace
    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);
    println!(
        "CPU trace: {} rows x {} cols",
        cpu_trace.main_table.height, cpu_trace.main_table.width
    );

    // Generate Bitwise trace and update multiplicities based on CPU lookups
    let mut bitwise_trace = generate_bitwise_trace();
    let bitwise_lookups = collect_bitwise_lookups_from_logs(&logs);
    update_multiplicities(&mut bitwise_trace, &bitwise_lookups);
    println!(
        "Bitwise trace: {} rows x {} cols, {} lookups",
        bitwise_trace.main_table.height, bitwise_trace.main_table.width, bitwise_lookups.len()
    );

    // Run prover and verifier
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for lui program"
    );
}

#[test]
#[ignore] // Slow: run with `make test_all`
fn test_vm_prover_beq() {
    // BEQ program: uses branch instruction, sends ZERO lookups
    let logs = run_asm_elf("beq");
    println!("beq.elf has {} steps", logs.len());

    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);

    // Generate Bitwise trace and update multiplicities
    let mut bitwise_trace = generate_bitwise_trace();
    let bitwise_lookups = collect_bitwise_lookups_from_logs(&logs);
    update_multiplicities(&mut bitwise_trace, &bitwise_lookups);
    println!("BEQ test: {} bitwise lookups", bitwise_lookups.len());

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for beq program"
    );
}

#[test]
#[ignore] // Slow: run with `make test_all`
fn test_vm_prover_add_64bit() {
    // 64-bit addition: 6 steps (padded to 8)
    let logs = run_asm_elf("add_64bit");
    println!("add_64bit.elf has {} steps", logs.len());

    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);

    // Generate Bitwise trace and update multiplicities
    let mut bitwise_trace = generate_bitwise_trace();
    let bitwise_lookups = collect_bitwise_lookups_from_logs(&logs);
    update_multiplicities(&mut bitwise_trace, &bitwise_lookups);

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for add_64bit program"
    );
}

#[test]
#[ignore] // Slow: run with `make test_all`
fn test_vm_prover_subw() {
    // SUBW: 32-bit word subtraction, sends MSB16/MSB8 lookups (word_instr=1)
    let logs = run_asm_elf("subw");
    println!("subw.elf has {} steps", logs.len());

    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);

    // Generate Bitwise trace and update multiplicities
    let mut bitwise_trace = generate_bitwise_trace();
    let bitwise_lookups = collect_bitwise_lookups_from_logs(&logs);
    update_multiplicities(&mut bitwise_trace, &bitwise_lookups);
    println!("SUBW test: {} bitwise lookups", bitwise_lookups.len());

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for subw program"
    );
}

// =============================================================================
// Fast tests using minimal (dummy) bitwise table
// =============================================================================
//
// These tests use a minimal bitwise table that only contains rows for the
// actual lookups. This is ~1000x faster than the full 2^20 row table.
//
// **WARNING: The minimal table is NOT production-safe!**
// The verifier expects the full deterministic 2^20 row public table.
// A minimal table would require the prover to reveal all values,
// making the proof size unacceptably large.

use std::collections::HashMap;

/// Generates a minimal bitwise trace containing only the rows needed for the given lookups.
///
/// **WARNING: FOR TESTING ONLY - NOT PRODUCTION SAFE!**
fn generate_minimal_bitwise_trace(
    lookups: &[(BitwiseLookup, u8, u8, u8)],
) -> TraceTable<F, E> {
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

    type FE = math::field::element::FieldElement<F>;
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

#[test]
fn test_vm_prover_lui_fast() {
    let logs = run_asm_elf("lui");
    assert_eq!(logs.len(), 2, "lui.elf should have 2 steps");

    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);
    let bitwise_lookups = collect_bitwise_lookups_from_logs(&logs);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast LUI: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for lui program (fast)"
    );
}

#[test]
fn test_vm_prover_beq_fast() {
    let logs = run_asm_elf("beq");
    println!("beq.elf has {} steps", logs.len());

    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);
    let bitwise_lookups = collect_bitwise_lookups_from_logs(&logs);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast BEQ: Bitwise {} rows (minimal), {} lookups",
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for beq program (fast)"
    );
}

#[test]
fn test_vm_prover_add_64bit_fast() {
    let logs = run_asm_elf("add_64bit");
    println!("add_64bit.elf has {} steps", logs.len());

    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);
    let bitwise_lookups = collect_bitwise_lookups_from_logs(&logs);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast ADD64: Bitwise {} rows (minimal), {} lookups",
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for add_64bit program (fast)"
    );
}

#[test]
fn test_vm_prover_subw_fast() {
    let logs = run_asm_elf("subw");
    println!("subw.elf has {} steps", logs.len());

    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);
    let bitwise_lookups = collect_bitwise_lookups_from_logs(&logs);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast SUBW: Bitwise {} rows (minimal), {} lookups",
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for subw program (fast)"
    );
}
