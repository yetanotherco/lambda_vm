//! Test-only capture of the verifier's FRI challenges and DEEP values, for the
//! exported test vectors (`tests/vectors/zf_fri`, the README's (d)): a vector
//! carries a proof AND the ζ, ι and DEEP values a correct verifier derives
//! from it, so the device prover and the in-guest verifier can check each stage separately.
//!
//! Compiled only for tests and the `test-utils` feature. Thread-local: the
//! host verifier is sequential on the calling thread, so [`capture`] sees
//! exactly the verification it wraps.

use core::any::Any;
use core::cell::RefCell;
use std::vec::Vec;

use math::field::element::FieldElement;
use math::field::traits::IsField;

/// What one table's verification derived: ζ (every folding challenge), ι
/// (the query pair indices) and the DEEP values p₀(υ), p₀(−υ) per query.
#[derive(Clone, Debug)]
pub struct FriCapture<E: IsField> {
    pub zetas: Vec<FieldElement<E>>,
    pub iotas: Vec<usize>,
    pub deep: Vec<FieldElement<E>>,
    pub deep_sym: Vec<FieldElement<E>>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Vec<Box<dyn Any>>>> = const { RefCell::new(None) };
}

/// Run `f` (a verification) and return its result with one record per table
/// verified, in order. Records are `FriCapture<E>` for the proof's extension
/// field; downcast with [`FriCapture::from_any`].
pub fn capture<T>(f: impl FnOnce() -> T) -> (T, Vec<Box<dyn Any>>) {
    ACTIVE.with(|a| *a.borrow_mut() = Some(Vec::new()));
    let out = f();
    let records = ACTIVE.with(|a| a.borrow_mut().take()).unwrap_or_default();
    (out, records)
}

impl<E: IsField + 'static> FriCapture<E> {
    pub fn from_any(record: &dyn Any) -> Option<&Self> {
        record.downcast_ref::<Self>()
    }
}

/// Start a table's record with its challenges (called after the replay).
pub(crate) fn record_challenges<E: IsField + 'static>(zetas: &[FieldElement<E>], iotas: &[usize]) {
    ACTIVE.with(|a| {
        if let Some(records) = a.borrow_mut().as_mut() {
            records.push(Box::new(FriCapture::<E> {
                zetas: zetas.to_vec(),
                iotas: iotas.to_vec(),
                deep: Vec::new(),
                deep_sym: Vec::new(),
            }));
        }
    });
}

/// Add the DEEP values to the current table's record.
pub(crate) fn record_deep<E: IsField + 'static>(
    deep: &[FieldElement<E>],
    deep_sym: &[FieldElement<E>],
) {
    ACTIVE.with(|a| {
        if let Some(records) = a.borrow_mut().as_mut()
            && let Some(last) = records.last_mut()
            && let Some(rec) = last.downcast_mut::<FriCapture<E>>()
        {
            rec.deep = deep.to_vec();
            rec.deep_sym = deep_sym.to_vec();
        }
    });
}
