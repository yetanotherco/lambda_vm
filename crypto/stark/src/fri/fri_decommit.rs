use crypto::merkle_tree::proof::Proof;
use math::field::element::FieldElement;
use math::field::traits::IsField;

use crate::config::Commitment;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct FriDecommitment<F: IsField> {
    pub layers_auth_paths: Vec<Proof<Commitment>>,
    /// Per-layer sibling evaluations: `layers_evaluations_sym[i]` holds
    /// `folding_factor - 1` evaluations needed to reconstruct the full leaf.
    /// For binary folding (folding_factor=2) each inner Vec has length 1.
    pub layers_evaluations_sym: Vec<Vec<FieldElement<F>>>,
}
