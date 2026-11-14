//! A compact wire encoding of a W-LFM proof: no archive headers, no padding.
//!
//! The rkyv archive a proof file is written with carries an 8-byte relative
//! pointer and length for every vector and pads to alignment. A C1 proof is
//! mostly short vectors (a block of `2^k` values, a Merkle path, a sumcheck
//! round), so that framing is a few percent of its bytes. This encoding writes
//! every length as a LEB128 varint and every value at its width: a base field
//! element as its eight canonical little-endian bytes, an extension element as
//! its three, a digest as its 32 bytes.
//!
//! ★ AN ENCODING, NOT A FORMAT. The decoded proof is the same value the prover
//! made, and the verifier checks that value; nothing here reaches a
//! transcript. The decoder is strict so one proof has exactly one encoding:
//! a field element at or above the modulus, an overlong varint, an unknown tag,
//! a length that cannot fit in what is left, or a trailing byte is refused.

use math::field::element::FieldElement;
use multilinear::claim_reduce::ReduceProof;
use multilinear::constraint_argument::ConstraintCore;
use multilinear::gkr::{GkrProof, LayerProof};
use multilinear::gkr_lockstep::{LockstepLayer, LockstepProof, TreeHalves};
use multilinear::stacked_eval::StackedProof;
use multilinear::sumcheck::{RoundProof as SumcheckRound, SumcheckProof};
use multilinear::whir_chain::{ChainProof, ChainRound, RoundNonces, RoundOpenings};
use multilinear::whir_commit::{Commitment, CosetOpening};
use multilinear::whir_round::{GroupedRoundProof, RoundProof};
use stark::multilinear_table::batched::BatchedArgue;
use stark::multilinear_table::{MultiProof, TableProof};

use super::wlfm::WlfmProof;
use crate::tables::types::{FE, FEE, GoldilocksExtension as E, GoldilocksField as F};

/// The Goldilocks modulus: a base element's canonical value is below it.
const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;

/// Which encoding a compressed file's C1 proof is written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WireFormat {
    /// The rkyv archive (today's file).
    Rkyv,
    /// [`encode`]: varints and canonical values, no framing.
    Compact,
    /// [`encode_dedup`]: [`Compact`](Self::Compact) with every Merkle sibling
    /// already sent replaced by a reference to it, so the nodes many paths
    /// share travel once.
    Dedup,
}

/// The bytes being written, and under [`WireFormat::Dedup`] every Merkle
/// sibling written so far with its index.
pub struct Out {
    bytes: Vec<u8>,
    seen: Option<std::collections::BTreeMap<Commitment, u64>>,
}

impl Out {
    fn new(dedup: bool) -> Self {
        Self {
            bytes: Vec::new(),
            seen: dedup.then(std::collections::BTreeMap::new),
        }
    }

    fn push(&mut self, byte: u8) {
        self.bytes.push(byte);
    }

    fn extend_from_slice(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }
}

/// A cursor over encoded bytes, and under [`WireFormat::Dedup`] every Merkle
/// sibling read so far.
pub struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    table: Option<Vec<Commitment>>,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            at: 0,
            table: None,
        }
    }

    fn dedup(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            at: 0,
            table: Some(Vec::new()),
        }
    }

    fn left(&self) -> usize {
        self.bytes.len() - self.at
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.left() {
            return Err(format!(
                "the encoding ends at byte {} and {n} more were wanted",
                self.bytes.len()
            ));
        }
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }

    /// Refused unless every byte was read.
    pub fn finish(self) -> Result<(), String> {
        if self.left() != 0 {
            return Err(format!("{} trailing bytes", self.left()));
        }
        Ok(())
    }
}

/// A value with one compact encoding.
pub trait Wire: Sized {
    fn put(&self, out: &mut Out);
    fn get(r: &mut Reader<'_>) -> Result<Self, String>;
}

/// LEB128, at most ten bytes, the shortest form only.
fn put_varint(mut v: u64, out: &mut Out) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn get_varint(r: &mut Reader<'_>) -> Result<u64, String> {
    let mut v = 0u64;
    for i in 0..10 {
        let byte = r.take(1)?[0];
        let bits = u64::from(byte & 0x7f);
        if i == 9 && bits > 1 {
            return Err("a varint past 64 bits".into());
        }
        v |= bits << (7 * i);
        if byte & 0x80 == 0 {
            if i > 0 && byte == 0 {
                return Err("an overlong varint".into());
            }
            return Ok(v);
        }
    }
    Err("a varint past ten bytes".into())
}

impl Wire for u64 {
    fn put(&self, out: &mut Out) {
        put_varint(*self, out);
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        get_varint(r)
    }
}

impl Wire for u32 {
    fn put(&self, out: &mut Out) {
        put_varint(u64::from(*self), out);
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        u32::try_from(get_varint(r)?).map_err(|_| "a u32 out of range".to_string())
    }
}

impl Wire for Commitment {
    fn put(&self, out: &mut Out) {
        out.extend_from_slice(self);
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        let mut d = [0u8; 32];
        d.copy_from_slice(r.take(32)?);
        Ok(d)
    }
}

impl Wire for FE {
    fn put(&self, out: &mut Out) {
        use math::field::traits::IsPrimeField;
        out.extend_from_slice(&F::canonical(self.value()).to_le_bytes());
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        let mut b = [0u8; 8];
        b.copy_from_slice(r.take(8)?);
        let v = u64::from_le_bytes(b);
        if v >= GOLDILOCKS {
            return Err("a field element at or above the modulus".into());
        }
        Ok(FE::from(v))
    }
}

impl Wire for FEE {
    fn put(&self, out: &mut Out) {
        for c in self.value() {
            c.put(out);
        }
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        Ok(FieldElement::<E>::new([
            FE::get(r)?,
            FE::get(r)?,
            FE::get(r)?,
        ]))
    }
}

impl<A: Wire, B: Wire> Wire for (A, B) {
    fn put(&self, out: &mut Out) {
        self.0.put(out);
        self.1.put(out);
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        Ok((A::get(r)?, B::get(r)?))
    }
}

impl<T: Wire, const N: usize> Wire for [T; N] {
    fn put(&self, out: &mut Out) {
        for x in self {
            x.put(out);
        }
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        let items = (0..N).map(|_| T::get(r)).collect::<Result<Vec<_>, _>>()?;
        items
            .try_into()
            .map_err(|_| "an array of the wrong length".to_string())
    }
}

impl<T: Wire> Wire for Vec<T> {
    fn put(&self, out: &mut Out) {
        put_varint(self.len() as u64, out);
        for x in self {
            x.put(out);
        }
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        let n = usize::try_from(get_varint(r)?).map_err(|_| "a length out of range")?;
        // Every element takes at least one byte, so a longer claim is a lie
        // and never an allocation.
        if n > r.left() {
            return Err(format!("{n} elements in {} bytes", r.left()));
        }
        (0..n).map(|_| T::get(r)).collect()
    }
}

impl<T: Wire> Wire for Option<T> {
    fn put(&self, out: &mut Out) {
        match self {
            None => out.push(0),
            Some(x) => {
                out.push(1);
                x.put(out);
            }
        }
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        match r.take(1)?[0] {
            0 => Ok(None),
            1 => Ok(Some(T::get(r)?)),
            t => Err(format!("an option tag {t}")),
        }
    }
}

/// A struct's fields in order, each through [`Wire`].
macro_rules! wire_struct {
    ($ty:ty { $($field:ident),+ $(,)? }) => {
        impl Wire for $ty {
            fn put(&self, out: &mut Out) {
                $(self.$field.put(out);)+
            }
            fn get(r: &mut Reader<'_>) -> Result<Self, String> {
                Ok(Self { $($field: Wire::get(r)?),+ })
            }
        }
    };
}

wire_struct!(SumcheckRound<E> { evaluations });
wire_struct!(SumcheckProof<E> { rounds });
wire_struct!(CosetOpening<F> { values, proof });
wire_struct!(CosetOpening<E> { values, proof });
wire_struct!(RoundProof<F, E> { current, next });
wire_struct!(RoundProof<E, E> { current, next });
wire_struct!(GroupedRoundProof<F, E> { current, next });
wire_struct!(RoundNonces {
    folding,
    ood,
    query
});
wire_struct!(ChainRound<F, E> { sumcheck, next_root, ood_value, nonces, openings });
wire_struct!(ChainProof<F, E> { rounds, final_value });
wire_struct!(StackedProof < F, E > { polys });
wire_struct!(LayerProof<E> { sumcheck, p_lo, p_hi, q_lo, q_hi });
wire_struct!(GkrProof<E> { layers });
wire_struct!(TreeHalves<E> { p_lo, p_hi, q_lo, q_hi });
wire_struct!(LockstepLayer<E> { sumcheck, halves });
wire_struct!(LockstepProof<E> { layers });
wire_struct!(ReduceProof<E> { sumcheck, column_values });
wire_struct!(ConstraintCore<E> { sumcheck, factor_values, reduce });
wire_struct!(TableProof<E> { gkr, bus_output, constraint });
wire_struct!(BatchedArgue<E> { bus_outputs, gkr, constraint, factor_values, reduces });
wire_struct!(MultiProof<F, E> { roots, tables, columns, preprocessed });
wire_struct!(WlfmProof {
    proof,
    argues,
    prepared,
    commit_ood,
    public_words
});

/// A Merkle path. Under [`WireFormat::Dedup`] each sibling is a varint: 0 and
/// its 32 bytes for a digest not sent before, `i ≥ 1` for the `i`-th distinct
/// sibling sent so far. A sibling is referenced whenever it was sent before —
/// so the decoder refuses a "new" one it already holds, and a proof keeps one
/// encoding.
impl Wire for crypto::merkle_tree::proof::Proof<Commitment> {
    fn put(&self, out: &mut Out) {
        if out.seen.is_none() {
            self.merkle_path.put(out);
            return;
        }
        put_varint(self.merkle_path.len() as u64, out);
        for d in &self.merkle_path {
            let seen = out.seen.as_mut().expect("dedup");
            match seen.get(d) {
                Some(&i) => put_varint(i + 1, out),
                None => {
                    let i = seen.len() as u64;
                    seen.insert(*d, i);
                    out.push(0);
                    out.extend_from_slice(d);
                }
            }
        }
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        if r.table.is_none() {
            return Ok(Self {
                merkle_path: Wire::get(r)?,
            });
        }
        let n = usize::try_from(get_varint(r)?).map_err(|_| "a length out of range")?;
        if n > r.left() {
            return Err(format!("{n} siblings in {} bytes", r.left()));
        }
        let mut merkle_path = Vec::with_capacity(n);
        for _ in 0..n {
            let t = get_varint(r)?;
            let d = if t == 0 {
                let d = Commitment::get(r)?;
                let table = r.table.as_mut().expect("dedup");
                if table.contains(&d) {
                    return Err("a sibling sent twice".into());
                }
                table.push(d);
                d
            } else {
                let table = r.table.as_ref().expect("dedup");
                *usize::try_from(t - 1)
                    .ok()
                    .and_then(|i| table.get(i))
                    .ok_or_else(|| format!("a reference to sibling {t} of {}", table.len()))?
            };
            merkle_path.push(d);
        }
        Ok(Self { merkle_path })
    }
}

impl Wire for RoundOpenings<F, E> {
    fn put(&self, out: &mut Out) {
        match self {
            RoundOpenings::Base(p) => {
                out.push(0);
                p.put(out);
            }
            RoundOpenings::Extension(p) => {
                out.push(1);
                p.put(out);
            }
            RoundOpenings::BaseGrouped(p) => {
                out.push(2);
                p.put(out);
            }
            RoundOpenings::BaseJoint(p) => {
                out.push(3);
                p.put(out);
            }
        }
    }
    fn get(r: &mut Reader<'_>) -> Result<Self, String> {
        Ok(match r.take(1)?[0] {
            0 => RoundOpenings::Base(Wire::get(r)?),
            1 => RoundOpenings::Extension(Wire::get(r)?),
            2 => RoundOpenings::BaseGrouped(Wire::get(r)?),
            3 => RoundOpenings::BaseJoint(Wire::get(r)?),
            t => return Err(format!("a round-openings tag {t}")),
        })
    }
}

/// `proof`'s compact bytes.
pub fn encode(proof: &WlfmProof) -> Vec<u8> {
    let mut out = Out::new(false);
    proof.put(&mut out);
    out.bytes
}

/// The proof `bytes` encode, refused unless they are exactly one proof's
/// encoding.
pub fn decode(bytes: &[u8]) -> Result<WlfmProof, String> {
    let mut r = Reader::new(bytes);
    let proof = WlfmProof::get(&mut r)?;
    r.finish()?;
    Ok(proof)
}

/// `proof`'s compact bytes with every repeated Merkle sibling referenced
/// ([`WireFormat::Dedup`]).
pub fn encode_dedup(proof: &WlfmProof) -> Vec<u8> {
    let mut out = Out::new(true);
    proof.put(&mut out);
    out.bytes
}

/// [`decode`] for [`encode_dedup`]'s bytes.
pub fn decode_dedup(bytes: &[u8]) -> Result<WlfmProof, String> {
    let mut r = Reader::dedup(bytes);
    let proof = WlfmProof::get(&mut r)?;
    r.finish()?;
    Ok(proof)
}

/// `proof` in `wire`'s encoding (`None` for [`WireFormat::Rkyv`], which the
/// file writes itself).
pub fn encode_as(proof: &WlfmProof, wire: WireFormat) -> Option<Vec<u8>> {
    match wire {
        WireFormat::Rkyv => None,
        WireFormat::Compact => Some(encode(proof)),
        WireFormat::Dedup => Some(encode_dedup(proof)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get<T: Wire>(bytes: &[u8]) -> Result<T, String> {
        let mut r = Reader::new(bytes);
        let v = T::get(&mut r)?;
        r.finish()?;
        Ok(v)
    }

    fn put<T: Wire>(v: &T) -> Vec<u8> {
        let mut out = Out::new(false);
        v.put(&mut out);
        out.bytes
    }

    /// Varints round-trip at the edges, and an overlong, a too-long or a
    /// past-64-bit one is refused.
    #[test]
    fn varints_have_one_encoding() {
        for v in [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX] {
            assert_eq!(get::<u64>(&put(&v)), Ok(v), "{v}");
        }
        assert_eq!(put(&127u64), vec![0x7f]);
        assert_eq!(put(&128u64), vec![0x80, 0x01]);
        assert!(get::<u64>(&[0x80, 0x00]).is_err(), "overlong");
        assert!(get::<u64>(&[0xff; 11]).is_err(), "eleven bytes");
        let mut past = vec![0xff; 9];
        past.push(0x02);
        assert!(get::<u64>(&past).is_err(), "past 64 bits");
        assert!(get::<u32>(&put(&(u64::from(u32::MAX) + 1))).is_err());
    }

    /// Field elements: eight canonical bytes each; the modulus and above are
    /// refused, the largest canonical value is not.
    #[test]
    fn field_elements_are_canonical() {
        let x = FE::from(0x1234_5678_9abc_def0u64);
        assert_eq!(get::<FE>(&put(&x)), Ok(x));
        assert_eq!(put(&x).len(), 8);
        let e = FEE::new([FE::from(1u64), FE::from(2u64), FE::from(3u64)]);
        assert_eq!(get::<FEE>(&put(&e)), Ok(e));
        assert_eq!(put(&e).len(), 24);
        assert!(get::<FE>(&GOLDILOCKS.to_le_bytes()).is_err());
        assert!(get::<FE>(&u64::MAX.to_le_bytes()).is_err());
        assert!(get::<FE>(&(GOLDILOCKS - 1).to_le_bytes()).is_ok());
    }

    /// Containers: a length that cannot fit is refused before any
    /// allocation, an unknown option or openings tag is refused, and a
    /// trailing byte is refused.
    #[test]
    fn containers_refuse_lies_unknown_tags_and_trailing_bytes() {
        let v: Vec<FE> = vec![FE::from(5u64); 3];
        assert_eq!(get::<Vec<FE>>(&put(&v)), Ok(v.clone()));
        let mut huge = Out::new(false);
        put_varint(u64::MAX >> 1, &mut huge);
        assert!(get::<Vec<FE>>(&huge.bytes).is_err());
        assert!(get::<Option<FE>>(&[2]).is_err());
        assert!(get::<RoundOpenings<F, E>>(&[4]).is_err());
        let mut trailing = put(&v);
        trailing.push(0);
        assert!(get::<Vec<FE>>(&trailing).is_err());
        let truncated = &put(&v)[..10];
        assert!(get::<Vec<FE>>(truncated).is_err());
    }

    fn path(ds: &[u8]) -> crypto::merkle_tree::proof::Proof<Commitment> {
        crypto::merkle_tree::proof::Proof {
            merkle_path: ds.iter().map(|&b| [b; 32]).collect(),
        }
    }

    /// Dedup paths: a repeated sibling is one byte, the paths decode back,
    /// and a "new" sibling the table holds, a reference past the table and a
    /// zero-length lie are refused.
    #[test]
    fn dedup_paths_reference_repeats_and_refuse_lies() {
        let paths = vec![path(&[1, 2, 3]), path(&[4, 2, 3]), path(&[1, 5, 3])];
        let mut out = Out::new(true);
        paths.put(&mut out);
        let plain = put(&paths);
        assert!(
            out.bytes.len() < plain.len(),
            "{} vs {}",
            out.bytes.len(),
            plain.len()
        );
        // 1 (count) + per path 1 (len) + 5 new × 33 + 4 refs × 1.
        assert_eq!(out.bytes.len(), 1 + 3 + 5 * 33 + 4);
        let mut r = Reader::dedup(&out.bytes);
        let back = Vec::<crypto::merkle_tree::proof::Proof<Commitment>>::get(&mut r).unwrap();
        r.finish().unwrap();
        let flat = |v: &[crypto::merkle_tree::proof::Proof<Commitment>]| {
            v.iter().map(|p| p.merkle_path.clone()).collect::<Vec<_>>()
        };
        assert_eq!(flat(&back), flat(&paths));
        // A sibling sent new twice.
        let mut twice = vec![1u8, 2, 0];
        twice.extend_from_slice(&[7u8; 32]);
        twice.push(0);
        twice.extend_from_slice(&[7u8; 32]);
        let mut r = Reader::dedup(&twice);
        assert!(Vec::<crypto::merkle_tree::proof::Proof<Commitment>>::get(&mut r).is_err());
        // A reference past the table.
        let mut r = Reader::dedup(&[1, 1, 1]);
        assert!(Vec::<crypto::merkle_tree::proof::Proof<Commitment>>::get(&mut r).is_err());
    }
}
