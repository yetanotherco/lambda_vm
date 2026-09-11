//! The opening's two factors, resident across the groups of rounds.
//!
//! A chained WHIR does not run one sumcheck: it runs a group of rounds, folds
//! its codeword, commits the successor, answers an out-of-domain point and
//! folds the next group's weight into the one it carries. The factors — the
//! message and the weight, each the width of a stacked polynomial — are the
//! same two tables throughout, so they stay here and the host drives them.
//!
//! The message arrives in the base field, which is where it is committed, and
//! is lifted on the way in: the sumcheck runs in one field.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
use crate::sumcheck::SumcheckSession;

/// The weight and the message, on device, with the cube they currently span.
pub struct OpeningSession {
    stream: Arc<CudaStream>,
    weight: Arc<CudaSlice<u64>>,
    message: Arc<CudaSlice<u64>>,
    len: usize,
}

impl OpeningSession {
    /// `weight` is interleaved ext3; `message` is the base-field polynomial
    /// being opened, lifted here.
    pub fn new(weight: &[u64], message: &[u64]) -> Result<Self> {
        assert!(weight.len().is_multiple_of(3), "three u64 per ext3 element");
        let len = weight.len() / 3;
        let be = backend()?;
        let stream = be.next_stream();
        let weight_dev = crate::device::htod_or_trim(&stream, weight)?;
        Self::with_weight(stream, weight_dev, len, message)
    }

    /// The same with the weight built here: each share is a column's subcube
    /// offset, the point it is claimed at, and the scale it carries, and its
    /// cells are `scale·eq(point, ·)`. The gaps stay zero.
    ///
    /// A stacked polynomial's weight is as wide as the polynomial — writing it
    /// on the host and sending it costs more than the rounds that read it.
    pub fn from_shares(
        shares: &[(usize, Vec<u64>, [u64; 3])],
        len: usize,
        message: &[u64],
    ) -> Result<Self> {
        let be = backend()?;
        let stream = be.next_stream();
        // Zeroed because the gaps between the shares' subcubes are part of the
        // weight and nothing writes them.
        let mut weight = crate::device::alloc_zeros_or_trim::<u64>(&stream, len * 3)?;
        crate::sumcheck::eq_expand_shares_ext3(&stream, &mut weight, shares)?;
        Self::with_weight(stream, weight, len, message)
    }

    fn with_weight(
        stream: Arc<CudaStream>,
        weight_dev: CudaSlice<u64>,
        len: usize,
        message: &[u64],
    ) -> Result<Self> {
        assert_eq!(message.len(), len, "the message spans the weight's cube");
        assert!(len.is_power_of_two(), "the cube is a power of two");

        let be = backend()?;
        let base = crate::device::htod_or_trim(&stream, message)?;
        // SAFETY: the kernel writes every element it is sized for.
        let mut lifted = unsafe { stream.alloc::<u64>(len * 3) }?;
        let count = len as u64;
        unsafe {
            stream
                .launch_builder(&be.mle_lift_base_ext3)
                .arg(&base)
                .arg(&count)
                .arg(&mut lifted)
                .launch(LaunchConfig::for_num_elems(len as u32))?;
        }

        Ok(Self {
            stream,
            weight: Arc::new(weight_dev),
            message: Arc::new(lifted),
            len,
        })
    }

    /// Cube indices the factors still span.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn num_vars(&self) -> usize {
        self.len.trailing_zeros() as usize
    }

    /// A sumcheck over `[weight, message]` as they stand.
    ///
    /// The rounds fold them in place, so the caller tells this session how far
    /// they got with [`bound`](Self::bound).
    pub fn sumcheck(
        &self,
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
    ) -> Result<SumcheckSession> {
        let addresses = {
            let (weight, _weight_guard) = self.weight.device_ptr(&self.stream);
            let (message, _message_guard) = self.message.device_ptr(&self.stream);
            [weight, message]
        };
        SumcheckSession::from_device(
            self.stream.clone(),
            &addresses,
            self.len,
            vec![self.weight.clone(), self.message.clone()],
            nodes,
            consts,
            num_slots,
            root_slot,
        )
    }

    /// Records that `rounds` variables have been bound by a sumcheck over
    /// these factors.
    pub fn bound(&mut self, rounds: usize) {
        self.len >>= rounds;
    }

    /// The message's value at `point`, which the out-of-domain answer needs.
    ///
    /// Folds a copy: the message is the next group's factor and the rounds
    /// have not bound this point.
    pub fn evaluate_message(&self, point: &[u64]) -> Result<[u64; 3]> {
        assert_eq!(point.len(), self.num_vars() * 3, "the point spans the cube");
        let mut copy = self.stream.alloc_zeros::<u64>(self.len * 3)?;
        {
            let source = self.message.slice(0..self.len * 3);
            self.stream.memcpy_dtod(&source, &mut copy)?;
        }
        crate::sumcheck::evaluate_resident_ext3(&self.stream, &mut copy, self.len, point)
    }

    /// `weight += scale · eq(point, ·)`, the weight the next group carries.
    pub fn add_scaled_eq(&self, point: &[u64], scale: &[u64]) -> Result<()> {
        assert_eq!(point.len(), self.num_vars() * 3, "the point spans the cube");
        assert_eq!(scale.len(), 3, "an ext3 scale");
        let be = backend()?;
        let eq = crate::sumcheck::eq_table_ext3(&self.stream, point, self.len)?;
        let scale_dev = crate::device::htod_or_trim(&self.stream, scale)?;
        let count = self.len as u64;
        // The kernel writes through a shared reference, the way the tree's
        // layers are folded: what orders these is the stream, and the weight
        // has one owner.
        unsafe {
            self.stream
                .launch_builder(&be.add_scaled_ext3)
                .arg(self.weight.as_ref())
                .arg(&eq)
                .arg(&count)
                .arg(&scale_dev)
                .launch(LaunchConfig::for_num_elems(self.len as u32))?;
        }
        Ok(())
    }
}
