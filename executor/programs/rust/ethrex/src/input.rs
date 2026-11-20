use ethrex_guest_program::l1::ProgramInput;
use rkyv::rancor::Error;

use crate::shared_table::OffsetSharedValidator;

// The same checked rkyv access as from_bytes. Validation tracks every shared
// pointer (one per trie node), in a table keyed by offset into the input with the
// same answers as rkyv's own (`shared_table`). The table is sized at a bounded
// estimate for larger witnesses so it rarely grows; an underestimate still grows.
// The deserializer's pool is not reserved: the trie decodes its nodes, the input's
// only shared pointers, without it.
pub fn decode_input(bytes: &[u8]) -> Result<ProgramInput, Error> {
    use rkyv::de::Pool;
    use rkyv::validation::{Validator, archive::ArchiveValidator};
    let capacity = if bytes.len() >= 64 * 1024 {
        (bytes.len() / 512).min(65536)
    } else {
        0
    };
    let archived = {
        let mut validator = Validator::new(
            ArchiveValidator::new(bytes),
            OffsetSharedValidator::new(bytes, capacity),
        );
        rkyv::api::access_with_context::<rkyv::Archived<ProgramInput>, _, Error>(
            bytes,
            &mut validator,
        )?
    };
    rkyv::api::deserialize_using::<ProgramInput, _, Error>(archived, &mut Pool::new())
}
