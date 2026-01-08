use math::field::{
    fields::fft_friendly::babybear_u32::Babybear31PrimeField,
    traits::{IsField, IsSubFieldOf},
};
use stark::{fri::FieldElement, table::TableView};

lazy_static::lazy_static! {
    pub static ref TWO_FIFTY_SIX: FieldElement<Babybear31PrimeField> =
        FieldElement::<Babybear31PrimeField>::from(256);
}

pub(crate) fn compute_element_from_two_limbs_starting_at<F, E>(
    step: &TableView<F, E>,
    index: usize,
) -> FieldElement<F>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let two_fifty_six = FieldElement::<F>::from(256);

    step.get_main_evaluation_element(0, index)
        + two_fifty_six * step.get_main_evaluation_element(0, index + 1)
}
