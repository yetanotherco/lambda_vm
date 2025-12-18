use lambdaworks_math::field::fields::fft_friendly::{
    babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
};
use stark_platinum_prover::{fri::FieldElement, table::TableView};

pub(crate) fn compute_element_from_two_limbs_starting_at(
    step: &TableView<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>,
    index: usize,
) -> FieldElement<Babybear31PrimeField> {
    let two_fifty_six = FieldElement::<Babybear31PrimeField>::from(256);
    step.get_main_evaluation_element(0, index)
        + two_fifty_six * step.get_main_evaluation_element(0, index + 1)
}

pub(crate) fn compute_element_from_four_limbs_starting_at(
    step: &TableView<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>,
    index: usize,
) -> FieldElement<Babybear31PrimeField> {
    let two_fifty_six = FieldElement::<Babybear31PrimeField>::from(256);
    step.get_main_evaluation_element(0, index)
        + two_fifty_six * step.get_main_evaluation_element(0, index + 1)
        + two_fifty_six * two_fifty_six * step.get_main_evaluation_element(0, index + 2)
        + two_fifty_six
            * two_fifty_six
            * two_fifty_six
            * step.get_main_evaluation_element(0, index + 3)
}

pub(crate) fn compute_element_from_two_limbs_starting_at_extension(
    step: &TableView<Degree4BabyBearU32ExtensionField, Degree4BabyBearU32ExtensionField>,
    index: usize,
) -> FieldElement<Degree4BabyBearU32ExtensionField> {
    let two_fifty_six = FieldElement::<Babybear31PrimeField>::from(256);
    step.get_main_evaluation_element(0, index)
        + two_fifty_six * step.get_main_evaluation_element(0, index + 1)
}

pub(crate) fn compute_element_from_four_limbs_starting_at_extension(
    step: &TableView<Degree4BabyBearU32ExtensionField, Degree4BabyBearU32ExtensionField>,
    index: usize,
) -> FieldElement<Degree4BabyBearU32ExtensionField> {
    let two_fifty_six = FieldElement::<Babybear31PrimeField>::from(256);
    step.get_main_evaluation_element(0, index)
        + two_fifty_six * step.get_main_evaluation_element(0, index + 1)
        + two_fifty_six * two_fifty_six * step.get_main_evaluation_element(0, index + 2)
        + two_fifty_six
            * two_fifty_six
            * two_fifty_six
            * step.get_main_evaluation_element(0, index + 3)
}
