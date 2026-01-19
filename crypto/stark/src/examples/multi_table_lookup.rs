use crate::{
    constraints::transition::TransitionConstraint,
    lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, NullBoundaryConstraintBuilder,
        Packing,
    },
    proof::options::ProofOptions,
    trace::TraceTable,
};
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};
use std::collections::HashMap;

type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;

pub fn new_cpu_air_with_lookup(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with ADD table (CPU sends to ADD bus)
            BusInteraction::sender(Some(0), Packing::Direct.columns(&[2, 3, 4])),
            // Interaction with MUL table (CPU sends to MUL bus)
            BusInteraction::sender(Some(1), Packing::Direct.columns(&[2, 3, 4])),
        ],
    };

    AirWithBuses::new(
        5, // CPU has 5 main columns: add_flag, mul_flag, a, b, c
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

pub fn new_mul_air_with_lookup(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with CPU table (MUL table receives from MUL bus)
            BusInteraction::receiver(Some(3), Packing::Direct.columns(&[0, 1, 2])),
        ],
    };

    AirWithBuses::new(
        4, // MUL has 4 main columns: a, b, c, multiplicity
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

pub fn new_add_air_with_lookup(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with CPU table (ADD table receives from ADD bus)
            BusInteraction::receiver(Some(3), Packing::Direct.columns(&[0, 1, 2])),
        ],
    };

    AirWithBuses::new(
        4, // ADD has 4 main columns: a, b, c, multiplicity
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

/// Operation type for CPU trace
#[derive(Debug, Clone, PartialEq, Eq)]
enum Operation {
    Add(FE, FE, FE), // (a, b, c) where c = a + b
    Mul(FE, FE, FE), // (a, b, c) where c = a * b
}

/// Generates random CPU, ADD, and MUL traces for multi-table lookup
///
/// # Arguments
/// * `trace_length` - Desired length of the CPU trace (will be padded to power of 2)
/// * `seed` - Random seed for reproducibility (optional)
///
/// # Returns
/// A tuple of (cpu_trace, add_trace, mul_trace)
///
/// # Rules
/// - CPU trace has 5 columns: add_flag, mul_flag, a, b, c
/// - If add_flag=1: c = a + b (mod p)
/// - If mul_flag=1: c = a * b (mod p)
/// - ADD/MUL traces collect unique operations with their multiplicities
pub fn generate_random_traces(
    trace_length: usize,
    seed: Option<u64>,
) -> (TraceTable<F, E>, TraceTable<F, E>, TraceTable<F, E>) {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    // Initialize RNG
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };

    // Ensure trace_length is at least 4 and a power of 2
    let trace_length = trace_length.max(4).next_power_of_two();

    // Generate random operations
    let mut operations = Vec::with_capacity(trace_length);

    for _ in 0..trace_length {
        // Randomly choose operation type (50% add, 50% mul)
        let is_add = rng.gen_bool(0.5);

        let a = FE::from(rng.gen_range(0..2013265921));
        let b = FE::from(rng.gen_range(0..2013265921));

        let op = if is_add {
            let c = &a + &b;
            Operation::Add(a, b, c)
        } else {
            let c = &a * &b;
            Operation::Mul(a, b, c)
        };

        operations.push(op);
    }

    // Build CPU trace columns
    let mut add_flag_col = Vec::with_capacity(trace_length);
    let mut mul_flag_col = Vec::with_capacity(trace_length);
    let mut a_col = Vec::with_capacity(trace_length);
    let mut b_col = Vec::with_capacity(trace_length);
    let mut c_col = Vec::with_capacity(trace_length);

    // Count multiplicities for ADD and MUL operations
    // Use representative values as keys since FieldElement doesn't implement Hash
    let mut add_ops: HashMap<(u32, u32, u32), u32> = HashMap::new();
    let mut mul_ops: HashMap<(u32, u32, u32), u32> = HashMap::new();

    for op in &operations {
        match op {
            Operation::Add(a, b, c) => {
                add_flag_col.push(FE::one());
                mul_flag_col.push(FE::zero());
                a_col.push(a.clone());
                b_col.push(b.clone());
                c_col.push(c.clone());
                let key = (
                    a.representative().limbs[0] as u32,
                    b.representative().limbs[0] as u32,
                    c.representative().limbs[0] as u32,
                );
                *add_ops.entry(key).or_insert(0) += 1;
            }
            Operation::Mul(a, b, c) => {
                add_flag_col.push(FE::zero());
                mul_flag_col.push(FE::one());
                a_col.push(a.clone());
                b_col.push(b.clone());
                c_col.push(c.clone());
                let key = (
                    a.representative().limbs[0] as u32,
                    b.representative().limbs[0] as u32,
                    c.representative().limbs[0] as u32,
                );
                *mul_ops.entry(key).or_insert(0) += 1;
            }
        }
    }

    let cpu_trace =
        TraceTable::from_columns_main(vec![add_flag_col, mul_flag_col, a_col, b_col, c_col], 1);

    // Build ADD trace
    let add_trace_length = add_ops.len().max(4).next_power_of_two();
    let mut add_a_col = Vec::with_capacity(add_trace_length);
    let mut add_b_col = Vec::with_capacity(add_trace_length);
    let mut add_c_col = Vec::with_capacity(add_trace_length);
    let mut add_mult_col = Vec::with_capacity(add_trace_length);

    for ((a_rep, b_rep, c_rep), mult) in add_ops.iter() {
        add_a_col.push(FE::from(*a_rep as u64));
        add_b_col.push(FE::from(*b_rep as u64));
        add_c_col.push(FE::from(*c_rep as u64));
        add_mult_col.push(FE::from(*mult as u64));
    }

    // Pad ADD trace to power of 2
    while add_a_col.len() < add_trace_length {
        add_a_col.push(FE::zero());
        add_b_col.push(FE::zero());
        add_c_col.push(FE::zero());
        add_mult_col.push(FE::zero());
    }

    let add_trace =
        TraceTable::from_columns_main(vec![add_a_col, add_b_col, add_c_col, add_mult_col], 1);

    // Build MUL trace
    let mul_trace_length = mul_ops.len().max(4).next_power_of_two();
    let mut mul_a_col = Vec::with_capacity(mul_trace_length);
    let mut mul_b_col = Vec::with_capacity(mul_trace_length);
    let mut mul_c_col = Vec::with_capacity(mul_trace_length);
    let mut mul_mult_col = Vec::with_capacity(mul_trace_length);

    for ((a_rep, b_rep, c_rep), mult) in mul_ops.iter() {
        mul_a_col.push(FE::from(*a_rep as u64));
        mul_b_col.push(FE::from(*b_rep as u64));
        mul_c_col.push(FE::from(*c_rep as u64));
        mul_mult_col.push(FE::from(*mult as u64));
    }

    // Pad MUL trace to power of 2
    while mul_a_col.len() < mul_trace_length {
        mul_a_col.push(FE::zero());
        mul_b_col.push(FE::zero());
        mul_c_col.push(FE::zero());
        mul_mult_col.push(FE::zero());
    }

    let mul_trace =
        TraceTable::from_columns_main(vec![mul_a_col, mul_b_col, mul_c_col, mul_mult_col], 1);

    (cpu_trace, add_trace, mul_trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_random_traces() {
        // Generate traces with a fixed seed for reproducibility
        let (cpu_trace, add_trace, mul_trace) = generate_random_traces(16, Some(3));

        // Verify CPU trace has correct shape
        assert_eq!(cpu_trace.num_rows(), 16);
        assert_eq!(cpu_trace.num_cols(), 5);

        // Verify ADD and MUL traces are valid (at least 4 rows, power of 2)
        assert!(add_trace.num_rows() >= 4);
        assert!(mul_trace.num_rows() >= 4);
        assert!(add_trace.num_rows().is_power_of_two());
        assert!(mul_trace.num_rows().is_power_of_two());
        assert_eq!(add_trace.num_cols(), 4);
        assert_eq!(mul_trace.num_cols(), 4);

        println!("CPU trace length: {}", cpu_trace.num_rows());
        println!("ADD trace length: {}", add_trace.num_rows());
        println!("MUL trace length: {}", mul_trace.num_rows());
    }

    #[test]
    fn test_generate_random_traces_power_of_two() {
        // Test that non-power-of-2 lengths are rounded up
        let (cpu_trace, _, _) = generate_random_traces(100, Some(123));

        // 100 should be rounded up to 128
        assert_eq!(cpu_trace.num_rows(), 128);
        assert!(cpu_trace.num_rows().is_power_of_two());
    }

    #[test]
    fn test_random_traces_proof_verification() {
        use crate::proof::options::ProofOptions;
        use crate::prover::{IsStarkProver, Prover};
        use crate::traits::AIR;
        use crate::verifier::{IsStarkVerifier, Verifier};
        use crypto::fiat_shamir::default_transcript::DefaultTranscript;

        // Generate random traces
        let (mut cpu_trace, mut add_trace, mut mul_trace) = generate_random_traces(16, Some(999));

        let proof_options = ProofOptions::default_test_options();
        let cpu_air = new_cpu_air_with_lookup(&proof_options);
        let add_air = new_add_air_with_lookup(&proof_options);
        let mul_air = new_mul_air_with_lookup(&proof_options);

        let air_trace_pairs: Vec<(
            &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            _,
            _,
        )> = vec![
            (&cpu_air, &mut cpu_trace, &()),
            (&add_air, &mut add_trace, &()),
            (&mul_air, &mut mul_trace, &()),
        ];

        let multi_proof =
            Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
                .expect("Failed to generate proof");

        let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
            vec![&cpu_air, &add_air, &mul_air];

        assert!(
            Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]),),
            "Proof verification failed for random traces"
        );
    }
}
