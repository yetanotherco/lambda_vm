use alloc::vec::Vec;
use crypto::merkle_tree::proof::Proof;
use math::field::element::FieldElement;
use math::field::traits::IsField;

use crate::config::Commitment;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct FriDecommitment<F: IsField> {
    pub layers_auth_paths: Vec<Proof<Commitment>>,
    pub layers_evaluations_sym: Vec<FieldElement<F>>,
}
