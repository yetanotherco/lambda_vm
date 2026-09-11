//! Shared SSZ input construction for this crate's two generators.
//!
//! `build_stateless_input` is the encoder both binaries need — `ethrex-fixtures` for the
//! synthetic blocks it builds from a funded genesis, and `real_block` for a real mainnet
//! block rebuilt as an Amsterdam one. It used to be copied into each, which meant the
//! copies had to be moved together on every ethrex rev bump with nothing enforcing it.

use ethrex_common::types::Block;
use ethrex_common::types::block_access_list::BlockAccessList;
use ethrex_common::types::block_execution_witness::RpcExecutionWitness;
use ethrex_common::types::stateless_ssz::{
    Bytes20, ExecutionPayload, ExecutionRequests, LogsBloom, NewPayloadRequest,
    STATELESS_INPUT_SCHEMA_ID, SszExecutionWitness, SszPublicKeys, SszStatelessInput,
};
use ethrex_guest_program::crypto::NativeCrypto;
use libssz::SszEncode;
use libssz_types::{ProgressiveList, SszList, SszVector};

pub fn empty_execution_requests() -> ExecutionRequests {
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
