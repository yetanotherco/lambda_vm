use ethrex_guest_program::l1::ProgramInput;
use rkyv::rancor::Error;

// The same checked rkyv access as from_bytes. Validation tracks every shared
// pointer, so its table is reserved at a bounded estimate for larger witnesses to
// avoid repeated rehashes; an underestimate still grows and small inputs keep the
// default. The deserializer's pool is not reserved: the trie decodes its nodes,
// the input's only shared pointers, without it.
pub fn decode_input(bytes: &[u8]) -> Result<ProgramInput, Error> {
    use rkyv::de::Pool;
    use rkyv::validation::{Validator, archive::ArchiveValidator, shared::SharedValidator};
    let capacity = if bytes.len() >= 64 * 1024 {
        (bytes.len() / 512).min(65536)
    } else {
        0
    };
    let archived = {
        let mut validator = Validator::new(
            ArchiveValidator::new(bytes),
            SharedValidator::with_capacity(capacity),
        );
        rkyv::api::access_with_context::<rkyv::Archived<ProgramInput>, _, Error>(
            bytes,
            &mut validator,
        )?
    };
    rkyv::api::deserialize_using::<ProgramInput, _, Error>(archived, &mut Pool::new())
}
