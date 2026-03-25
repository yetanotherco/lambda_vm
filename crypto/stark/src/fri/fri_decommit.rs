use crypto::merkle_tree::proof::Proof;
use math::field::element::FieldElement;
use math::field::traits::IsField;

use crate::config::Commitment;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct FriDecommitment<F: IsField> {
    pub layers_auth_paths: Vec<Proof<Commitment>>,
    /// For each FRI layer, the sibling evaluations at the queried position.
    /// Vec<Vec<FE>> supports higher-arity FRI where arity-k folds k values per layer.
    /// For arity-2 (current default) each inner Vec has exactly one element.
    pub layers_evaluations_sym: Vec<Vec<FieldElement<F>>>,
}
