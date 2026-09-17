//! The encoder that turns a block into the schema-prefixed SSZ stateless input, and the
//! check that the guest accepts what came out.
//!
//! Three copies of this used to exist — `ethrex-fixtures` for the synthetic blocks it
//! builds from a funded genesis, its `real_block` binary for a real mainnet block rebuilt
//! as an Amsterdam one, and `ethrex-block-converter` for an ethrex-replay cache — which
//! meant they had to move together on every ethrex rev bump with nothing enforcing it.
//! They differ in where the block comes FROM, not in how it is encoded, so only the
//! sourcing stays in each.

use ethrex_common::types::Block;
use ethrex_common::types::block_access_list::BlockAccessList;
use ethrex_common::types::block_execution_witness::RpcExecutionWitness;
use ethrex_common::types::stateless_ssz::{
    Bytes20, ExecutionPayload, ExecutionRequests, LogsBloom, NewPayloadRequest,
    STATELESS_INPUT_SCHEMA_ID, SszExecutionWitness, SszPublicKeys, SszStatelessInput,
};
use ethrex_guest_program::crypto::NativeCrypto;
use ethrex_guest_program::l1::run_stateless_guest;
use libssz::SszEncode;
use libssz_types::{ProgressiveList, SszList, SszVector};

fn empty_execution_requests() -> ExecutionRequests {
    ExecutionRequests {
        deposits: ProgressiveList::new(),
        withdrawals: ProgressiveList::new(),
        consolidations: ProgressiveList::new(),
        builder_deposits: ProgressiveList::new(),
        builder_exits: ProgressiveList::new(),
    }
}

pub fn build_stateless_input(
    block: &Block,
    witness: &RpcExecutionWitness,
    block_access_list: Option<&BlockAccessList>,
    chain_id: u64,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let header = &block.header;
    let bal = block_access_list.ok_or("Amsterdam fixture has no block access list")?;
    let block_access_list_hash = header
        .block_access_list_hash
        .ok_or("Amsterdam fixture header has no block_access_list_hash")?;
    if bal.compute_hash(&NativeCrypto) != block_access_list_hash {
        return Err("fixture BAL does not match block_access_list_hash".into());
    }
    let slot_number = header
        .slot_number
        .ok_or("Amsterdam fixture header has no slot_number")?;

    let transactions = block
        .body
        .transactions
        .iter()
        .map(|tx| tx.encode_canonical_to_vec().into())
        .collect::<Vec<ProgressiveList<u8>>>()
        .into();
    let public_keys = block
        .body
        .transactions
        .iter()
        .enumerate()
        .map(|(i, tx)| {
            let key = tx
                .public_key(&NativeCrypto)
                .map_err(|e| format!("failed to recover public key for transaction {i}: {e}"))?
                .ok_or_else(|| format!("transaction {i} has no recoverable signature"))?;
            SszVector::try_from(key.to_vec())
                .map_err(|e| format!("public key for transaction {i} is invalid: {e:?}"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let withdrawals = block
        .body
        .withdrawals
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(
            |withdrawal| ethrex_common::types::stateless_ssz::Withdrawal {
                index: withdrawal.index,
                validator_index: withdrawal.validator_index,
                address: Bytes20(withdrawal.address.0),
                amount: withdrawal.amount,
            },
        )
        .collect::<Vec<_>>()
        .into();
    let logs_bloom: LogsBloom = SszVector::try_from(header.logs_bloom.0.to_vec())?;
    let extra_data = SszList::try_from(header.extra_data.to_vec())?;
    let base_fee = header.base_fee_per_gas.ok_or("fixture has no base fee")?;
    let mut base_fee_bytes = [0u8; 32];
    base_fee_bytes[..8].copy_from_slice(&base_fee.to_le_bytes());

    let execution_payload = ExecutionPayload {
        parent_hash: header.parent_hash.0,
        fee_recipient: Bytes20(header.coinbase.0),
        state_root: header.state_root.0,
        receipts_root: header.receipts_root.0,
        logs_bloom,
        prev_randao: header.prev_randao.0,
        block_number: header.number,
        gas_limit: header.gas_limit,
        gas_used: header.gas_used,
        timestamp: header.timestamp,
        extra_data,
        base_fee_per_gas: base_fee_bytes,
        block_hash: header.compute_block_hash(&NativeCrypto).0,
        transactions,
        withdrawals,
        blob_gas_used: header.blob_gas_used.unwrap_or_default(),
        excess_blob_gas: header.excess_blob_gas.unwrap_or_default(),
        block_access_list: ethrex_rlp::encode::RLPEncode::encode_to_vec(bal).into(),
        slot_number,
    };
    let ssz_witness = SszExecutionWitness {
        state: ProgressiveList::from(
            witness
                .state
                .iter()
                .map(|bytes| SszList::try_from(bytes.to_vec()))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        codes: ProgressiveList::from(
            witness
                .codes
                .iter()
                .map(|bytes| SszList::try_from(bytes.to_vec()))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        headers: SszList::try_from(
            witness
                .headers
                .iter()
                .map(|bytes| SszList::try_from(bytes.to_vec()))
                .collect::<Result<Vec<_>, _>>()?,
        )?,
    };
    let input = SszStatelessInput {
        new_payload_request: NewPayloadRequest {
            execution_payload,
            versioned_hashes: block
                .body
                .transactions
                .iter()
                .flat_map(|tx| tx.blob_versioned_hashes())
                .map(|hash| hash.0)
                .collect::<Vec<_>>()
                .into(),
            parent_beacon_block_root: header.parent_beacon_block_root.unwrap_or_default().0,
            execution_requests: empty_execution_requests(),
        },
        witness: ssz_witness,
        chain_id,
        public_keys: SszPublicKeys::from(public_keys),
    };
    let mut bytes = STATELESS_INPUT_SCHEMA_ID.to_be_bytes().to_vec();
    input.ssz_append(&mut bytes);
    Ok(bytes)
}

/// The guest's public output: the fields of ethrex's `SszStatelessValidationResult` in
/// declaration order — `new_payload_request_root(32) || successful_validation(1) ||
/// chain_id(8) || schema_id(2)`.
///
/// Those first 32 bytes are the payload request's `hash_tree_root`, not a state root. The
/// post-state root is checked inside `validate_stateless_execution`, and what the guest
/// publishes about it is the flag at [`VALIDATION_FLAG`] — so the flag is the whole verdict,
/// and there is no root in the output worth comparing against the block.
pub const GUEST_OUTPUT_LEN: usize = 43;
/// Offset of `successful_validation` within that output.
pub const VALIDATION_FLAG: usize = 32;

/// Run an encoded input through the stateless guest on the host, and report whether the
/// block validated.
///
/// This is the check every producer owes its output before writing it: a guest that cannot
/// decode the input does not fail, it commits an all-zero result and exits cleanly, so an
/// unvalidated fixture benchmarks nothing while looking like a 99% improvement.
pub fn validate_natively(bytes: &[u8]) -> Result<(), String> {
    let output = run_stateless_guest(bytes, std::sync::Arc::new(NativeCrypto));
    if output.len() != GUEST_OUTPUT_LEN {
        return Err(format!(
            "guest returned {} bytes, expected {GUEST_OUTPUT_LEN}",
            output.len()
        ));
    }
    if output[VALIDATION_FLAG] == 0 {
        return Err("the guest rejected the block (successful_validation = 0)".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ethrex_common::types::stateless_ssz::SszStatelessValidationResult;

    /// [`GUEST_OUTPUT_LEN`] and [`VALIDATION_FLAG`] are offsets into a struct this crate
    /// does not own, and the only thing checking them is `validate_natively`'s length
    /// comparison -- which a reordering that kept 43 bytes would pass while the flag
    /// moved somewhere else. Encoding the upstream struct pins both, and it is the same
    /// encoding the guest publishes: `run_stateless_guest` ends in
    /// `SszStatelessValidationResult { .. }.ssz_append(&mut out)`.
    ///
    /// No CI job runs this crate's tests -- it is a path dependency with no lockfile of
    /// its own, and building it standalone would be a second cold build of the ethrex
    /// host tree. What covers the same property on every PR is `native_output` in
    /// tooling/ethrex-tests, which asserts both constants against real fixtures
    /// (`make test-ethrex-offline`). This test is what turns that failure into a
    /// localized one: there, a reordering shows up as a committed fixture being rejected.
    #[test]
    fn output_constants_match_the_upstream_result_layout() {
        let mut accepted = Vec::new();
        SszStatelessValidationResult {
            successful_validation: true,
            ..Default::default()
        }
        .ssz_append(&mut accepted);

        assert_eq!(
            accepted.len(),
            GUEST_OUTPUT_LEN,
            "the stateless result's encoded length moved"
        );
        assert_eq!(
            accepted[VALIDATION_FLAG], 1,
            "successful_validation is no longer at VALIDATION_FLAG"
        );
        // Every other field is zero here, so that byte is the flag rather than merely
        // agreeing with it -- a field reordered into this offset would fail here too.
        assert_eq!(
            accepted.iter().filter(|byte| **byte != 0).count(),
            1,
            "another field is non-zero, so the offset no longer identifies the flag"
        );

        // The rejection path returns the Default, which is what makes an input the guest
        // cannot decode a clean exit instead of an abort: same length, flag clear.
        let mut rejected = Vec::new();
        SszStatelessValidationResult::default().ssz_append(&mut rejected);
        assert_eq!(rejected.len(), GUEST_OUTPUT_LEN);
        assert_eq!(rejected[VALIDATION_FLAG], 0);
    }
}
