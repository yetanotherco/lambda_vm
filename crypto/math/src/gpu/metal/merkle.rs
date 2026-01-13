//! Metal GPU-accelerated Merkle tree implementation using Poseidon2 hash.
//!
//! This module provides GPU-accelerated Merkle tree construction for Goldilocks field
//! elements using the Poseidon2 hash function.

use super::device::MetalState;
use super::errors::MetalError;
use super::shaders::MerklePipelines;
use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks_native::GoldilocksField;
use alloc::string::ToString;
use alloc::vec::Vec;

type Fp = FieldElement<GoldilocksField>;

/// Metal-accelerated Merkle tree builder using Poseidon2 hash.
pub struct MetalMerkleTree {
    state: MetalState,
    pipelines: MerklePipelines,
}

impl MetalMerkleTree {
    /// Create a new Metal Merkle tree builder.
    pub fn new() -> Result<Self, MetalError> {
        let state = MetalState::new()?;
        let pipelines = MerklePipelines::new(&state.device, &state.library)?;

        Ok(Self { state, pipelines })
    }

    /// Build a Merkle tree from field elements and return the root.
    /// Returns (root, all_nodes) where all_nodes is the complete tree in level order.
    pub fn build(&self, leaves: &[Fp]) -> Result<(Fp, Vec<Fp>), MetalError> {
        if leaves.is_empty() {
            return Err(MetalError::InvalidInput("Empty leaves".to_string()));
        }

        // Pad to power of two
        let n = leaves.len().next_power_of_two();
        let mut padded_leaves: Vec<u64> = leaves.iter().map(|x| *x.value()).collect();
        while padded_leaves.len() < n {
            padded_leaves.push(*padded_leaves.last().unwrap());
        }

        // Hash leaves on GPU
        let hashed_leaves = self.hash_leaves_gpu(&padded_leaves)?;

        // Build tree levels
        let mut all_nodes = Vec::with_capacity(2 * n - 1);
        let mut current_level = hashed_leaves;

        // Store leaves first (will be at end of final array)
        let leaves_start = n - 1;
        all_nodes.resize(leaves_start, 0u64);
        all_nodes.extend(&current_level);

        // Build each level from bottom to top
        while current_level.len() > 1 {
            let next_level = self.build_level_gpu(&current_level)?;
            let level_start = current_level.len() / 2 - 1;

            // Insert this level's nodes
            for (i, &node) in next_level.iter().enumerate() {
                if level_start + i < all_nodes.len() {
                    all_nodes[level_start + i] = node;
                }
            }

            current_level = next_level;
        }

        // Root is at index 0
        let root = current_level[0];
        all_nodes[0] = root;

        // Convert to field elements
        let nodes_fp: Vec<Fp> = all_nodes.iter().map(|&x| Fp::from(x)).collect();
        Ok((Fp::from(root), nodes_fp))
    }

    /// Hash leaves on GPU (single element per leaf).
    fn hash_leaves_gpu(&self, leaves: &[u64]) -> Result<Vec<u64>, MetalError> {
        let n = leaves.len();

        // Create input buffer
        let input_buffer = self.state.create_buffer_with_data(leaves)?;

        // Create output buffer
        let output_buffer = self.state.create_buffer(n)?;

        // Create command buffer and encoder
        let command_buffer = self.state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.hash_leaves);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        let n_u32 = n as u32;
        encoder.set_bytes(2, core::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);

        // Dispatch threads
        let threadgroup_size = self.state.recommended_threadgroup_size().min(n);
        let num_threadgroups = (n + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize::new(num_threadgroups as u64, 1, 1),
            metal::MTLSize::new(threadgroup_size as u64, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read output
        let mut output = vec![0u64; n];
        self.state.read_buffer(&output_buffer, &mut output)?;

        Ok(output)
    }

    /// Build one level of the tree on GPU.
    fn build_level_gpu(&self, prev_level: &[u64]) -> Result<Vec<u64>, MetalError> {
        let num_pairs = prev_level.len() / 2;

        if num_pairs == 0 {
            return Ok(prev_level.to_vec());
        }

        // Create buffers
        let input_buffer = self.state.create_buffer_with_data(prev_level)?;
        let output_buffer = self.state.create_buffer(num_pairs)?;

        // Create command buffer and encoder
        let command_buffer = self.state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.build_level);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        let num_pairs_u32 = num_pairs as u32;
        encoder.set_bytes(
            2,
            core::mem::size_of::<u32>() as u64,
            &num_pairs_u32 as *const u32 as *const _,
        );

        // Dispatch threads
        let threadgroup_size = self.state.recommended_threadgroup_size().min(num_pairs);
        let num_threadgroups = (num_pairs + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize::new(num_threadgroups as u64, 1, 1),
            metal::MTLSize::new(threadgroup_size as u64, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read output
        let mut output = vec![0u64; num_pairs];
        self.state.read_buffer(&output_buffer, &mut output)?;

        Ok(output)
    }

    /// Build Merkle tree and return only the root (more efficient).
    pub fn build_root_only(&self, leaves: &[Fp]) -> Result<Fp, MetalError> {
        if leaves.is_empty() {
            return Err(MetalError::InvalidInput("Empty leaves".to_string()));
        }

        // Pad to power of two
        let n = leaves.len().next_power_of_two();
        let mut padded_leaves: Vec<u64> = leaves.iter().map(|x| *x.value()).collect();
        while padded_leaves.len() < n {
            padded_leaves.push(*padded_leaves.last().unwrap());
        }

        // Hash leaves on GPU
        let mut current_level = self.hash_leaves_gpu(&padded_leaves)?;

        // Build tree levels until we reach the root
        while current_level.len() > 1 {
            current_level = self.build_level_gpu(&current_level)?;
        }

        Ok(Fp::from(current_level[0]))
    }

    /// Get the device name for logging.
    pub fn device_name(&self) -> &str {
        self.state.device_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_merkle_tree_small() {
        if let Ok(merkle) = MetalMerkleTree::new() {
            println!("Testing Metal Merkle tree on: {}", merkle.device_name());

            // Test with small input
            let leaves: Vec<Fp> = (1..=8).map(|i| Fp::from(i as u64)).collect();
            let result = merkle.build_root_only(&leaves);

            assert!(result.is_ok(), "Failed to build Merkle tree: {:?}", result.err());
            let root = result.unwrap();
            println!("Root: {:?}", root.value());

            // Verify determinism
            let root2 = merkle.build_root_only(&leaves).unwrap();
            assert_eq!(root, root2, "Merkle tree should be deterministic");
        } else {
            println!("Metal not available, skipping test");
        }
    }

    #[test]
    fn test_metal_merkle_tree_power_of_two() {
        if let Ok(merkle) = MetalMerkleTree::new() {
            for log_n in 2..=10 {
                let n = 1 << log_n;
                let leaves: Vec<Fp> = (0..n).map(|i| Fp::from(i as u64)).collect();
                let result = merkle.build_root_only(&leaves);

                assert!(
                    result.is_ok(),
                    "Failed for size 2^{}: {:?}",
                    log_n,
                    result.err()
                );
            }
        }
    }

    #[test]
    fn test_metal_merkle_non_power_of_two() {
        if let Ok(merkle) = MetalMerkleTree::new() {
            // Test non-power-of-two sizes
            for n in [3, 5, 7, 9, 15, 17, 100] {
                let leaves: Vec<Fp> = (0..n).map(|i| Fp::from(i as u64)).collect();
                let result = merkle.build_root_only(&leaves);

                assert!(result.is_ok(), "Failed for size {}: {:?}", n, result.err());
            }
        }
    }
}
