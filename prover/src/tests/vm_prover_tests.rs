//! VM Prover integration tests using multi_prove.
//!
//! These tests verify the full prover pipeline:
//! - Run ELF through executor
//! - Generate traces for CPU and Bitwise tables
//! - Use multi_prove/multi_verify with bus interactions
//!
//! Currently wired buses:
//! - CPU sends AND_BYTE, OR_BYTE, XOR_BYTE to Bitwise (×8 each)
//!
//! TODO: LT bus (needs CPU sender with DWordHHW packing)

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
    bus_interactions as bitwise_bus_interactions, generate_bitwise_trace,
};
use crate::tables64::cpu::{bus_interactions as cpu_bus_interactions, generate_cpu_trace_from_logs};
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

    // Generate Bitwise trace (the full precomputed table)
    let mut bitwise_trace = generate_bitwise_trace();
    println!(
        "Bitwise trace: {} rows x {} cols",
        bitwise_trace.main_table.height, bitwise_trace.main_table.width
    );

    // Run prover and verifier
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for lui program"
    );
}

#[test]
fn test_vm_prover_beq() {
    // BEQ program: uses branch instruction, no bitwise ops
    let logs = run_asm_elf("beq");
    println!("beq.elf has {} steps", logs.len());

    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);
    let mut bitwise_trace = generate_bitwise_trace();

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for beq program"
    );
}

#[test]
fn test_vm_prover_add_64bit() {
    // 64-bit addition: 6 steps (padded to 8)
    let logs = run_asm_elf("add_64bit");
    println!("add_64bit.elf has {} steps", logs.len());

    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);
    let mut bitwise_trace = generate_bitwise_trace();

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for add_64bit program"
    );
}

#[test]
fn test_vm_prover_subw() {
    // SUBW: 32-bit word subtraction
    let logs = run_asm_elf("subw");
    println!("subw.elf has {} steps", logs.len());

    let mut cpu_trace = generate_cpu_trace_from_logs(&logs);
    let mut bitwise_trace = generate_bitwise_trace();

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for subw program"
    );
}
