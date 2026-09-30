//! Page-locked slots for an epoch's trace columns (D-TRACE stage 1b).
//!
//! The producer writes an epoch's columns into a slot leased here, freezes it,
//! and makes every committed column a view into it ([`Mle::shared`]). The
//! upload to the card ([`crate::gpu::upload_columns`]) then reads them where
//! they lie, as a DMA, instead of through the driver's staging copy. The slot
//! goes back to the pool when the last view of it drops, which is when the
//! epoch's proof is done.
//!
//! Without the `cuda` feature there is nothing to lend: every lease misses
//! with [`Miss::NoDevice`], and the columns are built as before.
//!
//! [`Mle::shared`]: crate::mle::Mle::shared

use std::sync::Arc;

use math::field::{element::FieldElement, goldilocks::GoldilocksField};

use crate::mle::HostColumns;

/// Why a lease was not given. The epoch then builds its columns the old way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Miss {
    /// No slot is free and not every slot exists yet: the allocation has not
    /// run, has not finished, or failed.
    NotReady,
    /// Every slot exists and is lent out.
    NoneFree,
    /// The epoch is bigger than a slot.
    TooSmall,
    /// The build has no device support.
    NoDevice,
}

impl Miss {
    /// Its name on the log lines.
    pub fn name(&self) -> &'static str {
        match self {
            Self::NotReady => "not-ready",
            Self::NoneFree => "none-free",
            Self::TooSmall => "too-small",
            Self::NoDevice => "no-device",
        }
    }
}

/// A pool's counters since it was made: slots that exist and are free now,
/// leases given, and misses by kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub slots: usize,
    pub created: usize,
    pub free: usize,
    pub leased: u64,
    pub not_ready: u64,
    pub none_free: u64,
    pub too_small: u64,
}

/// Slots of page-locked memory, lent one epoch at a time. Cloning gives
/// another handle to the same pool.
#[derive(Clone)]
pub struct Pool(#[cfg(feature = "cuda")] math_cuda::pinned_slots::SlotPool);

impl Pool {
    /// The process's pool: two slots of `LAMBDA_VM_WHIR_PINNED_SLOT_MB` MiB
    /// (3072 by default), none allocated until [`Pool::fill`].
    pub fn global() -> Self {
        #[cfg(feature = "cuda")]
        return Self(math_cuda::pinned_slots::SlotPool::global().clone());
        #[cfg(not(feature = "cuda"))]
        return Self();
    }

    /// A pool of its own: `slots` slots of `slot_bytes` bytes each.
    pub fn new(slots: usize, slot_bytes: usize) -> Self {
        #[cfg(feature = "cuda")]
        return Self(math_cuda::pinned_slots::SlotPool::new(slots, slot_bytes));
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (slots, slot_bytes);
            Self()
        }
    }

    /// Allocates the slots not yet allocated, one after another on this
    /// thread, each lendable as soon as it exists: the seconds each took, or
    /// why the allocation failed.
    pub fn fill(&self) -> Result<Vec<f64>, String> {
        #[cfg(feature = "cuda")]
        return self.0.fill().map_err(|e| format!("{e}"));
        #[cfg(not(feature = "cuda"))]
        return Err("no device support in this build".to_string());
    }

    /// A free slot for `elems` field elements, or why there is none. Never
    /// waits.
    pub fn try_lease(&self, elems: usize) -> Result<Lease, Miss> {
        #[cfg(feature = "cuda")]
        return self
            .0
            .try_lease(elems)
            .map(Lease)
            .map_err(|miss| match miss {
                math_cuda::pinned_slots::Miss::NotReady => Miss::NotReady,
                math_cuda::pinned_slots::Miss::NoneFree => Miss::NoneFree,
                math_cuda::pinned_slots::Miss::TooSmall => Miss::TooSmall,
            });
        #[cfg(not(feature = "cuda"))]
        {
            let _ = elems;
            Err(Miss::NoDevice)
        }
    }

    /// The most field elements one slot holds.
    pub fn slot_elems(&self) -> usize {
        #[cfg(feature = "cuda")]
        return self.0.block_elems();
        #[cfg(not(feature = "cuda"))]
        return 0;
    }

    pub fn stats(&self) -> Stats {
        #[cfg(feature = "cuda")]
        {
            let s = self.0.stats();
            Stats {
                slots: s.slots,
                created: s.created,
                free: s.free,
                leased: s.leased,
                not_ready: s.not_ready,
                none_free: s.none_free,
                too_small: s.too_small,
            }
        }
        #[cfg(not(feature = "cuda"))]
        Stats::default()
    }
}

/// A slot lent out for writing, exclusively. Dropped unfrozen, it goes back to
/// the pool.
#[cfg(feature = "cuda")]
pub struct Lease(math_cuda::pinned_slots::SlotLease);

/// A slot a build without device support never lends. Never constructed.
#[cfg(not(feature = "cuda"))]
pub struct Lease(std::convert::Infallible);

impl Lease {
    /// The slot's place in the pool's allocation order.
    pub fn slot(&self) -> usize {
        #[cfg(feature = "cuda")]
        return self.0.slot();
        #[cfg(not(feature = "cuda"))]
        match self.0 {}
    }

    /// The field elements leased, as many as asked for, to write. They hold
    /// what the slot last held.
    pub fn values_mut(&mut self) -> &mut [FieldElement<GoldilocksField>] {
        #[cfg(feature = "cuda")]
        {
            let raw = self.0.as_mut_slice();
            // SAFETY: `FieldElement<GoldilocksField>` is `repr(transparent)` over
            // its raw `u64` (the layout `gpu::upload_columns` relies on), and
            // every `u64` is a raw value it may hold; the borrow is the lease's.
            unsafe {
                std::slice::from_raw_parts_mut(
                    raw.as_mut_ptr().cast::<FieldElement<GoldilocksField>>(),
                    raw.len(),
                )
            }
        }
        #[cfg(not(feature = "cuda"))]
        match self.0 {}
    }

    /// Ends the writing: the slot, read-only from now on, as the backing its
    /// columns' views share. It goes back to the pool when the last view does.
    pub fn freeze(self) -> Arc<dyn HostColumns<GoldilocksField>> {
        #[cfg(feature = "cuda")]
        return Arc::new(PinnedColumns(self.0.freeze()));
        #[cfg(not(feature = "cuda"))]
        match self.0 {}
    }
}

/// A frozen slot as the columns' backing.
#[cfg(feature = "cuda")]
struct PinnedColumns(math_cuda::pinned_slots::FrozenSlot);

#[cfg(feature = "cuda")]
impl HostColumns<GoldilocksField> for PinnedColumns {
    fn slice(&self) -> &[FieldElement<GoldilocksField>] {
        let raw = self.0.as_slice();
        // SAFETY: as in `Lease::values_mut`; the frozen slot is only read.
        unsafe {
            std::slice::from_raw_parts(
                raw.as_ptr().cast::<FieldElement<GoldilocksField>>(),
                raw.len(),
            )
        }
    }

    fn is_pinned(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool nobody filled lends nothing and says why: not ready with device
    /// support, no device without. Asking never allocates.
    #[test]
    fn an_unfilled_pool_misses() {
        let pool = Pool::new(2, 1 << 20);
        let expected = if cfg!(feature = "cuda") {
            Miss::NotReady
        } else {
            Miss::NoDevice
        };
        assert_eq!(pool.try_lease(16).err(), Some(expected));
        assert_eq!(pool.stats().created, 0, "a lease allocated a slot");
    }

    #[test]
    fn misses_have_their_log_names() {
        let names: Vec<&str> = [
            Miss::NotReady,
            Miss::NoneFree,
            Miss::TooSmall,
            Miss::NoDevice,
        ]
        .iter()
        .map(Miss::name)
        .collect();
        assert_eq!(names, ["not-ready", "none-free", "too-small", "no-device"]);
    }
}
