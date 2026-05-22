use crypto::merkle_tree::proof::Proof;
use math::field::element::FieldElement;
use math::field::traits::IsField;

use crate::config::Commitment;

/// Per-query FRI decommitment. Each committed layer folds by 4, so a query
/// reveals one 4-element fold orbit (the quad leaf) and one authentication path
/// for that leaf.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct FriDecommitment<F: IsField> {
    pub layers_auth_paths: Vec<Proof<Commitment>>,
    pub layers_evaluations: Vec<[FieldElement<F>; 4]>,
}
