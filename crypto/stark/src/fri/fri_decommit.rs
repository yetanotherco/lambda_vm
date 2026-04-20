use crypto::merkle_tree::proof::Proof;
use math::field::element::FieldElement;
use math::field::traits::IsField;

use crate::config::Commitment;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct FriDecommitment<F: IsField> {
    pub layers_auth_paths: Vec<Proof<Commitment>>,
    /// For arity-4 FRI: the 3 sibling evaluations per layer at positions
    /// {index^1, index^2, index^3} within the 4-element orbit.
    pub layers_evaluations_siblings: Vec<[FieldElement<F>; 3]>,
}
