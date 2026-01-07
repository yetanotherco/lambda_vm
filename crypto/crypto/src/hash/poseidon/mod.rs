use alloc::{borrow::ToOwned, vec::Vec};
use math::field::element::FieldElement as FE;

pub mod parameters;
pub mod starknet;

use parameters::PermutationParameters;

mod private {
    use super::*;

    pub trait Sealed {}

    impl<P: PermutationParameters> Sealed for P {}
}

pub trait Poseidon: PermutationParameters + self::private::Sealed {
    fn hades_permutation(state: &mut [FE<Self::F>]);
    fn full_round(state: &mut [FE<Self::F>], rindex: usize);
    fn partial_round(state: &mut [FE<Self::F>], index: usize);
    fn hash(x: &FE<Self::F>, y: &FE<Self::F>) -> FE<Self::F>;
    fn hash_single(x: &FE<Self::F>) -> FE<Self::F>;
    fn hash_many(inputs: &[FE<Self::F>]) -> FE<Self::F>;
}

impl<P: PermutationParameters> Poseidon for P {
    fn hades_permutation(state: &mut [FE<Self::F>]) {
        let mut index = 0;
        for _ in 0..P::N_FULL_ROUNDS / 2 {
            Self::full_round(state, index);
            index += P::N_ROUND_CONSTANTS_COLS;
        }
        for _ in 0..P::N_PARTIAL_ROUNDS {
            Self::partial_round(state, index);
            index += 1;
        }
        for _ in 0..P::N_FULL_ROUNDS / 2 {
            Self::full_round(state, index);
            index += P::N_ROUND_CONSTANTS_COLS;
        }
    }

    #[inline]
    fn full_round(state: &mut [FE<Self::F>], index: usize) {
        for (i, value) in state.iter_mut().enumerate() {
            *value = &(*value) + &P::ROUND_CONSTANTS[index + i];
            *value = &(*value).square() * &*value;
        }
        Self::mix(state);
    }

    #[inline]
    fn partial_round(state: &mut [FE<Self::F>], index: usize) {
        state[2] = &state[2] + &P::ROUND_CONSTANTS[index];
        state[2] = &state[2].square() * &state[2];
        Self::mix(state);
    }

    fn hash(x: &FE<Self::F>, y: &FE<Self::F>) -> FE<Self::F> {
        let mut state: Vec<FE<Self::F>> = vec![x.clone(), y.clone(), FE::from(2)];
        Self::hades_permutation(&mut state);
        let x = &state[0];
        x.clone()
    }

    fn hash_single(x: &FE<Self::F>) -> FE<Self::F> {
        let mut state: Vec<FE<Self::F>> = vec![x.clone(), FE::zero(), FE::from(1)];
        Self::hades_permutation(&mut state);
        let x = &state[0];
        x.clone()
    }

    fn hash_many(inputs: &[FE<Self::F>]) -> FE<Self::F> {
        let r = P::RATE; // chunk size
        let m = P::STATE_SIZE; // state size

        // Pad input with 1 followed by 0's (if necessary).
        let mut values = inputs.to_owned();
        values.push(FE::from(1));
        values.resize(values.len().div_ceil(r) * r, FE::zero());

        assert!(values.len() % r == 0);
        let mut state: Vec<FE<Self::F>> = vec![FE::zero(); m];

        // Process each block
        for block in values.chunks(r) {
            let mut block_state: Vec<FE<Self::F>> =
                state[0..r].iter().zip(block).map(|(s, b)| s + b).collect();
            block_state.extend_from_slice(&state[r..]);

            Self::hades_permutation(&mut block_state);
            state = block_state;
        }

        state[0].clone()
    }
}

