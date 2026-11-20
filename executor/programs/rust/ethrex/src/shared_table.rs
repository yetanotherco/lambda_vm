//! A shared-pointer validator for rkyv's checked access, keyed by offset into the input.
//!
//! rkyv validates every `ArchivedRc` through a [`SharedContext`]: the first visit of an
//! address starts it (and checks the pointee), a later visit finds it finished, a visit
//! while it is still being checked is a cycle, and the same address claimed as two
//! types is an error. rkyv's [`SharedValidator`] keeps that state in a general hash map
//! keyed by address with the full `TypeId` beside it; for the guest's input, one entry
//! per trie node, those map operations were most of the cost of validating a node.
//!
//! [`OffsetSharedValidator`] keeps the same state, with the same answers, in an
//! open-addressing table of one word a slot: the address's offset into the input, a
//! small index into the list of types seen, and the finished bit. Addresses outside the
//! input (which rkyv's bounds check then rejects) go to an inner [`SharedValidator`],
//! so every address has exactly one place where its state can live.

use core::any::TypeId;
use core::fmt;

use rkyv::rancor::Source;
use rkyv::validation::SharedContext;
use rkyv::validation::shared::{SharedValidator, ValidationState};

/// See the module documentation.
pub struct OffsetSharedValidator {
    base: usize,
    len: usize,
    /// 0 = empty; otherwise `(offset + 1) << 32 | type_index << 1 | finished`.
    slots: Vec<u64>,
    /// `slots.len() - 1`; `slots.len()` is a power of two.
    mask: usize,
    /// `64 - log2(slots.len())`.
    shift: u32,
    used: usize,
    /// Every type seen, by index; `last` caches the most recent lookup.
    types: Vec<TypeId>,
    last: Option<(TypeId, u64)>,
    outside: SharedValidator,
}

impl OffsetSharedValidator {
    /// A validator for the input `bytes`, its table sized for about `capacity` shared
    /// pointers (it grows past that).
    pub fn new(bytes: &[u8], capacity: usize) -> Self {
        // Offsets are stored in 32 bits: an input of 4 GiB or more keeps every address
        // in the inner validator.
        let len = if bytes.len() < u32::MAX as usize {
            bytes.len()
        } else {
            0
        };
        let slots = (capacity.max(8) * 2).next_power_of_two();
        Self {
            base: bytes.as_ptr() as usize,
            len,
            slots: vec![0; slots],
            mask: slots - 1,
            shift: 64 - slots.trailing_zeros(),
            used: 0,
            types: Vec::new(),
            last: None,
            outside: SharedValidator::new(),
        }
    }

    /// The offset of `address` into the input, if it is inside it.
    #[inline(always)]
    fn offset(&self, address: usize) -> Option<u64> {
        let offset = address.wrapping_sub(self.base);
        (offset < self.len).then_some(offset as u64)
    }

    /// The index of `type_id`, adding it if new.
    #[inline(always)]
    fn type_index(&mut self, type_id: TypeId) -> u64 {
        if let Some((last, index)) = self.last
            && last == type_id
        {
            return index;
        }
        let index = match self.types.iter().position(|t| *t == type_id) {
            Some(i) => i as u64,
            None => {
                self.types.push(type_id);
                (self.types.len() - 1) as u64
            }
        };
        self.last = Some((type_id, index));
        index
    }

    #[inline(always)]
    fn home(&self, key: u64) -> usize {
        // Fibonacci hashing of the offset; the table index is the product's top bits.
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    /// The slot holding `key` (offset + 1), or the empty slot where it would go.
    #[inline(always)]
    fn find(&self, key: u64) -> usize {
        let mut i = self.home(key);
        loop {
            let slot = self.slots[i];
            if slot == 0 || slot >> 32 == key {
                return i;
            }
            i = (i + 1) & self.mask;
        }
    }

    #[cold]
    fn grow(&mut self) {
        let old = core::mem::take(&mut self.slots);
        let slots = old.len() * 2;
        self.slots = vec![0; slots];
        self.mask = slots - 1;
        self.shift = 64 - slots.trailing_zeros();
        for slot in old {
            if slot != 0 {
                let i = self.find(slot >> 32);
                self.slots[i] = slot;
            }
        }
    }

    fn type_of(&self, slot: u64) -> TypeId {
        self.types[((slot & 0xFFFF_FFFF) >> 1) as usize]
    }
}

impl<E: Source> SharedContext<E> for OffsetSharedValidator {
    fn start_shared(&mut self, address: usize, type_id: TypeId) -> Result<ValidationState, E> {
        let Some(offset) = self.offset(address) else {
            return self.outside.start_shared(address, type_id);
        };
        let key = offset + 1;
        let index = self.type_index(type_id);
        let i = self.find(key);
        let slot = self.slots[i];
        if slot == 0 {
            self.slots[i] = key << 32 | index << 1;
            self.used += 1;
            if self.used * 2 > self.slots.len() {
                self.grow();
            }
            return Ok(ValidationState::Started);
        }
        if (slot & 0xFFFF_FFFF) >> 1 != index {
            return Err(E::new(TypeMismatch {
                previous: self.type_of(slot),
                current: type_id,
            }));
        }
        if slot & 1 == 0 {
            Ok(ValidationState::Pending)
        } else {
            Ok(ValidationState::Finished)
        }
    }

    fn finish_shared(&mut self, address: usize, type_id: TypeId) -> Result<(), E> {
        let Some(offset) = self.offset(address) else {
            return self.outside.finish_shared(address, type_id);
        };
        let index = self.type_index(type_id);
        let i = self.find(offset + 1);
        let slot = self.slots[i];
        if slot == 0 {
            return Err(E::new(NotStarted));
        }
        if (slot & 0xFFFF_FFFF) >> 1 != index {
            return Err(E::new(TypeMismatch {
                previous: self.type_of(slot),
                current: type_id,
            }));
        }
        if slot & 1 == 1 {
            return Err(E::new(AlreadyFinished));
        }
        self.slots[i] = slot | 1;
        Ok(())
    }
}

// rkyv's errors for the same three conditions are private; these carry the same
// messages.

#[derive(Debug)]
struct TypeMismatch {
    previous: TypeId,
    current: TypeId,
}

impl fmt::Display for TypeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the same memory region has been claimed as two different types: {:?} and {:?}",
            self.previous, self.current,
        )
    }
}

impl core::error::Error for TypeMismatch {}

#[derive(Debug)]
struct NotStarted;

impl fmt::Display for NotStarted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shared pointer was not started validation")
    }
}

impl core::error::Error for NotStarted {}

#[derive(Debug)]
struct AlreadyFinished;

impl fmt::Display for AlreadyFinished {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shared pointer was already finished validation")
    }
}

impl core::error::Error for AlreadyFinished {}

#[cfg(test)]
mod tests {
    use super::*;

    type TestError = rkyv::rancor::Failure;

    /// What a validator answered: the state, or that it failed.
    fn start<C: SharedContext<TestError>>(c: &mut C, a: usize, t: TypeId) -> Option<u8> {
        match c.start_shared(a, t) {
            Ok(ValidationState::Started) => Some(0),
            Ok(ValidationState::Pending) => Some(1),
            Ok(ValidationState::Finished) => Some(2),
            Err(_) => None,
        }
    }

    fn finish<C: SharedContext<TestError>>(c: &mut C, a: usize, t: TypeId) -> Option<()> {
        c.finish_shared(a, t).ok()
    }

    /// Step by step, the same answers as rkyv's validator: random starts and finishes
    /// over in-input and outside addresses, three types, a table small enough to grow
    /// several times and to collide.
    #[test]
    fn answers_like_rkyvs_shared_validator() {
        let types = [
            TypeId::of::<u8>(),
            TypeId::of::<u64>(),
            TypeId::of::<[u8; 3]>(),
        ];
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for round in 0..200 {
            let input = vec![0u8; 4096];
            let base = input.as_ptr() as usize;
            let mut ours = OffsetSharedValidator::new(&input, (round % 4) as usize);
            let mut theirs = SharedValidator::new();
            let span = [8u64, 64, 600, 4096][round % 4];
            for _ in 0..2000 {
                let r = next();
                // Mostly inside the input (first and last bytes included), some just
                // outside either end, some far away.
                let address = match r % 16 {
                    0 => base.wrapping_sub(1 + (r >> 8) as usize % 64),
                    1 => base + 4096 + (r >> 8) as usize % 64,
                    2 => (r >> 4) as usize,
                    3 => base + 4095,
                    _ => base + ((r >> 8) % span) as usize % 4096,
                };
                let t = types[((r >> 40) % if r % 7 == 0 { 3 } else { 1 }) as usize];
                if (r >> 50) % 2 == 0 {
                    assert_eq!(start(&mut ours, address, t), start(&mut theirs, address, t));
                } else {
                    assert_eq!(
                        finish(&mut ours, address, t),
                        finish(&mut theirs, address, t)
                    );
                }
            }
        }
    }

    /// The four answers for one address, in order, and a type mismatch both before and
    /// after the finish.
    #[test]
    fn one_address_through_its_states() {
        let input = vec![0u8; 64];
        let a = input.as_ptr() as usize + 17;
        let (t, u) = (TypeId::of::<u8>(), TypeId::of::<u16>());
        let mut v = OffsetSharedValidator::new(&input, 0);
        assert_eq!(finish(&mut v, a, t), None, "finish before start");
        assert_eq!(start(&mut v, a, t), Some(0), "started");
        assert_eq!(start(&mut v, a, t), Some(1), "pending: a cycle");
        assert_eq!(start(&mut v, a, u), None, "other type while pending");
        assert_eq!(finish(&mut v, a, u), None, "finish as the other type");
        assert_eq!(finish(&mut v, a, t), Some(()), "finished");
        assert_eq!(finish(&mut v, a, t), None, "finished twice");
        assert_eq!(start(&mut v, a, t), Some(2), "finished: seen before");
        assert_eq!(start(&mut v, a, u), None, "other type once finished");
        assert_eq!(
            start(&mut v, a + 1, t),
            Some(0),
            "the next byte is another address"
        );
    }

    /// Checked access of the committed ethrex inputs: both validators accept them, and
    /// with bytes flipped they accept and reject the same ones.
    #[test]
    fn checked_access_of_ethrex_inputs_agrees() {
        use ethrex_guest_program::l1::ProgramInput;
        use rkyv::rancor::Error;
        use rkyv::validation::{Validator, archive::ArchiveValidator};

        fn ours(bytes: &[u8]) -> bool {
            let mut v = Validator::new(
                ArchiveValidator::new(bytes),
                OffsetSharedValidator::new(bytes, bytes.len() / 512),
            );
            rkyv::api::access_with_context::<rkyv::Archived<ProgramInput>, _, Error>(bytes, &mut v)
                .is_ok()
        }
        fn theirs(bytes: &[u8]) -> bool {
            let mut v = Validator::new(ArchiveValidator::new(bytes), SharedValidator::new());
            rkyv::api::access_with_context::<rkyv::Archived<ProgramInput>, _, Error>(bytes, &mut v)
                .is_ok()
        }

        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests");
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut flips = 0;
        let mut rejected = 0;
        for name in [
            "ethrex_10_transfers.bin",
            "ethrex_bench_4.bin",
            "ethrex_empty_block.bin",
            "ethrex_simple_tx.bin",
        ] {
            let bytes = std::fs::read(format!("{dir}/{name}")).unwrap();
            // rkyv reads the input from an aligned buffer in the guest; copy it into one.
            let mut aligned = rkyv::util::AlignedVec::<16>::new();
            aligned.extend_from_slice(&bytes);
            assert!(ours(&aligned) && theirs(&aligned), "{name} accepted");
            for _ in 0..300 {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
                let mut bad = aligned.clone();
                let i = (r >> 8) as usize % bad.len();
                bad[i] ^= 1 << (r % 8);
                let (a, b) = (ours(&bad), theirs(&bad));
                assert_eq!(a, b, "{name}: bit {} of byte {i}", r % 8);
                flips += 1;
                rejected += usize::from(!a);
            }
        }
        assert!(
            flips == 1200 && rejected > 0,
            "{rejected} of {flips} flips rejected"
        );
    }
}
