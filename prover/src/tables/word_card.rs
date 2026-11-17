//! The word memory format's rows as the card's fills read them
//! (`multilinear::records`' word rows, `math_cuda::fill`'s `*RowC`): the same
//! bytes. Each prover row is `#[repr(C)]` with its offsets asserted beside its
//! struct; each card mirror's offsets are asserted here to be those, so a
//! slice of the prover's rows is read in place, never copied.

use super::mem_load2w::Load2WRow;
use super::mem_load8::Load8Row;
use super::mem_loadw::LoadWRow;
use super::mem_store2w::Store2WRow;
use super::mem_store8::Store8Row;
use super::mem_storew::StoreWRow;
use super::memw_w::MemwWordRow;
use multilinear::records as card;

/// `rows` as `C`, the card's mirror of `R`.
///
/// SAFETY (the callers'): `R` and `C` are `#[repr(C)]` with the same size and
/// alignment and every field `C` names at the offset `R` holds it (asserted at
/// each caller), of the same type, or a `u8` where `R` holds a `bool` or a
/// one-byte `#[repr(u8)]` enum (0 or 1). Padding is never read as a field.
unsafe fn view<R, C>(rows: &[R]) -> &[C] {
    unsafe { std::slice::from_raw_parts(rows.as_ptr().cast(), rows.len()) }
}

macro_rules! same_size {
    ($r:ty, $c:ty) => {
        assert!(std::mem::size_of::<$r>() == std::mem::size_of::<$c>());
        assert!(std::mem::align_of::<$r>() == std::mem::align_of::<$c>());
    };
}

/// MEM_LOAD8's rows as the card reads them. The prover's offsets
/// (`mem_load8.rs`): 0, 8, 16, 24 in 32 bytes.
pub(crate) fn card_load8_rows(rows: &[Load8Row]) -> &[card::Load8Row] {
    use std::mem::offset_of;
    const _: () = {
        same_size!(Load8Row, card::Load8Row);
        assert!(offset_of!(card::Load8Row, address) == 0);
        assert!(offset_of!(card::Load8Row, timestamp) == 8);
        assert!(offset_of!(card::Load8Row, step) == 16);
        assert!(offset_of!(card::Load8Row, word) == 24);
    };
    // SAFETY: as `view` asks (asserted above and beside the prover's struct).
    unsafe { view(rows) }
}

/// MEM_STORE8's rows. The prover's offsets (`mem_store8.rs`): 0, 8, 16, 24,
/// 32 in 40 bytes.
pub(crate) fn card_store8_rows(rows: &[Store8Row]) -> &[card::Store8Row] {
    use std::mem::offset_of;
    const _: () = {
        same_size!(Store8Row, card::Store8Row);
        assert!(offset_of!(card::Store8Row, address) == 0);
        assert!(offset_of!(card::Store8Row, timestamp) == 8);
        assert!(offset_of!(card::Store8Row, step) == 16);
        assert!(offset_of!(card::Store8Row, old) == 24);
        assert!(offset_of!(card::Store8Row, new) == 32);
    };
    // SAFETY: as `view` asks.
    unsafe { view(rows) }
}

/// MEM_LOADW's rows. The prover's offsets (`mem_loadw.rs`): 0, 8, 16, 20
/// (width), 21 (signed, a `bool`), 24 in 32 bytes.
pub(crate) fn card_loadw_rows(rows: &[LoadWRow]) -> &[card::LoadWRow] {
    use std::mem::offset_of;
    const _: () = {
        same_size!(LoadWRow, card::LoadWRow);
        assert!(offset_of!(card::LoadWRow, address) == 0);
        assert!(offset_of!(card::LoadWRow, timestamp) == 8);
        assert!(offset_of!(card::LoadWRow, step) == 16);
        assert!(offset_of!(card::LoadWRow, width) == 20);
        assert!(offset_of!(card::LoadWRow, signed) == 21);
        assert!(offset_of!(card::LoadWRow, word) == 24);
    };
    // SAFETY: as `view` asks; `signed` reads the prover's `bool` as 0 or 1.
    unsafe { view(rows) }
}

/// MEM_STOREW's rows. The prover's offsets (`mem_storew.rs`): 0, 8, 16, 20
/// (width), 24 (value), 32 (old) in 40 bytes.
pub(crate) fn card_storew_rows(rows: &[StoreWRow]) -> &[card::StoreWRow] {
    use std::mem::offset_of;
    const _: () = {
        same_size!(StoreWRow, card::StoreWRow);
        assert!(offset_of!(card::StoreWRow, address) == 0);
        assert!(offset_of!(card::StoreWRow, timestamp) == 8);
        assert!(offset_of!(card::StoreWRow, step) == 16);
        assert!(offset_of!(card::StoreWRow, width) == 20);
        assert!(offset_of!(card::StoreWRow, value) == 24);
        assert!(offset_of!(card::StoreWRow, old) == 32);
    };
    // SAFETY: as `view` asks.
    unsafe { view(rows) }
}

/// MEM_LOAD2W's rows. The prover's offsets (`mem_load2w.rs`): 0, 8, 16
/// (steps), 24 (words), 40 (width), 41 (signed, a `bool`) in 48 bytes.
pub(crate) fn card_load2w_rows(rows: &[Load2WRow]) -> &[card::Load2WRow] {
    use std::mem::offset_of;
    const _: () = {
        same_size!(Load2WRow, card::Load2WRow);
        assert!(offset_of!(card::Load2WRow, address) == 0);
        assert!(offset_of!(card::Load2WRow, timestamp) == 8);
        assert!(offset_of!(card::Load2WRow, steps) == 16);
        assert!(offset_of!(card::Load2WRow, words) == 24);
        assert!(offset_of!(card::Load2WRow, width) == 40);
        assert!(offset_of!(card::Load2WRow, signed) == 41);
    };
    // SAFETY: as `view` asks; `signed` reads the prover's `bool` as 0 or 1.
    unsafe { view(rows) }
}

/// MEM_STORE2W's rows. The prover's offsets (`mem_store2w.rs`): 0, 8, 16
/// (steps), 24 (value), 32 (old), 48 (width) in 56 bytes.
pub(crate) fn card_store2w_rows(rows: &[Store2WRow]) -> &[card::Store2WRow] {
    use std::mem::offset_of;
    const _: () = {
        same_size!(Store2WRow, card::Store2WRow);
        assert!(offset_of!(card::Store2WRow, address) == 0);
        assert!(offset_of!(card::Store2WRow, timestamp) == 8);
        assert!(offset_of!(card::Store2WRow, steps) == 16);
        assert!(offset_of!(card::Store2WRow, value) == 24);
        assert!(offset_of!(card::Store2WRow, old) == 32);
        assert!(offset_of!(card::Store2WRow, width) == 48);
    };
    // SAFETY: as `view` asks.
    unsafe { view(rows) }
}

/// MEMW_W's rows. The prover's offsets (`memw_w.rs`): 0, 8, 16, 20 (kind, a
/// `#[repr(u8)]` enum of 0 or 1), 24 (old), 32 (new) in 40 bytes.
pub(crate) fn card_memw_w_rows(rows: &[MemwWordRow]) -> &[card::MemwWordRow] {
    use std::mem::offset_of;
    const _: () = {
        same_size!(MemwWordRow, card::MemwWordRow);
        assert!(offset_of!(card::MemwWordRow, address) == 0);
        assert!(offset_of!(card::MemwWordRow, timestamp) == 8);
        assert!(offset_of!(card::MemwWordRow, step) == 16);
        assert!(offset_of!(card::MemwWordRow, kind) == 20);
        assert!(offset_of!(card::MemwWordRow, old) == 24);
        assert!(offset_of!(card::MemwWordRow, new) == 32);
    };
    // SAFETY: as `view` asks; `kind` reads the prover's enum as its byte.
    unsafe { view(rows) }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tables::card_fill_inputs::{col_major, splitmix};
    use crate::tables::gpack::TraceForm;
    use crate::tables::memw_w::MessageKind;

    /// A timestamp and an old one before it: the step a spread from 0 to
    /// 2^31.
    fn times(s: &mut u64) -> (u64, u64) {
        let timestamp = 4 * (splitmix(s) % (1 << 28)) + 4;
        let gap = 1 + splitmix(s) % timestamp.min(1 << 31);
        (timestamp, timestamp - gap)
    }

    /// An address whose offset in its word is `offset`; one in eight at the
    /// last word of a low 2^32 range (a crossing access there carries).
    fn address(s: &mut u64, offset: u64) -> u64 {
        let base = if splitmix(s).is_multiple_of(8) {
            (splitmix(s) % (1 << 20)) << 32 | 0xFFFF_FFF8
        } else {
            splitmix(s) & !7 & !(0xFFFF_FFFF << 32)
        };
        base | offset
    }

    fn step(timestamp: u64, old: u64) -> u32 {
        (timestamp - old - 1) as u32
    }

    /// `n` rows of each family, from `seed`: each the row its constructor
    /// makes (its own checks pass), and the card's mirror made from the same
    /// inputs, field by field.
    pub(crate) struct Rows {
        pub load8: (Vec<Load8Row>, Vec<card::Load8Row>),
        pub store8: (Vec<Store8Row>, Vec<card::Store8Row>),
        pub loadw: (Vec<LoadWRow>, Vec<card::LoadWRow>),
        pub storew: (Vec<StoreWRow>, Vec<card::StoreWRow>),
        pub load2w: (Vec<Load2WRow>, Vec<card::Load2WRow>),
        pub store2w: (Vec<Store2WRow>, Vec<card::Store2WRow>),
        pub memw_w: (Vec<MemwWordRow>, Vec<card::MemwWordRow>),
    }

    pub(crate) fn rows(seed: u64, n: usize) -> Rows {
        let mut s = seed;
        let mut r = Rows {
            load8: (Vec::new(), Vec::new()),
            store8: (Vec::new(), Vec::new()),
            loadw: (Vec::new(), Vec::new()),
            storew: (Vec::new(), Vec::new()),
            load2w: (Vec::new(), Vec::new()),
            store2w: (Vec::new(), Vec::new()),
            memw_w: (Vec::new(), Vec::new()),
        };
        while r.load8.0.len() < n {
            let address = address(&mut s, 0);
            let (timestamp, old) = times(&mut s);
            let word = splitmix(&mut s);
            if let Ok(row) = Load8Row::new(address, timestamp, old, word) {
                r.load8.0.push(row);
                r.load8.1.push(card::Load8Row {
                    address,
                    timestamp,
                    step: step(timestamp, old),
                    word,
                });
            }
        }
        while r.store8.0.len() < n {
            let address = address(&mut s, 0);
            let (timestamp, old_ts) = times(&mut s);
            let (old, new) = (splitmix(&mut s), splitmix(&mut s));
            if let Ok(row) = Store8Row::new(address, timestamp, old_ts, old, new) {
                r.store8.0.push(row);
                r.store8.1.push(card::Store8Row {
                    address,
                    timestamp,
                    step: step(timestamp, old_ts),
                    old,
                    new,
                });
            }
        }
        while r.loadw.0.len() < n {
            let width = [1u8, 2, 4][(splitmix(&mut s) % 3) as usize];
            let offset = splitmix(&mut s) % 8;
            let address = address(&mut s, offset);
            let (timestamp, old) = times(&mut s);
            let signed = splitmix(&mut s) & 1 == 1;
            let word = splitmix(&mut s);
            if let Ok(row) = LoadWRow::new(address, timestamp, old, width, signed, word) {
                r.loadw.0.push(row);
                r.loadw.1.push(card::LoadWRow {
                    address,
                    timestamp,
                    step: step(timestamp, old),
                    width,
                    signed: u8::from(signed),
                    word,
                });
            }
        }
        while r.storew.0.len() < n {
            let width = [1u8, 2, 4][(splitmix(&mut s) % 3) as usize];
            let offset = splitmix(&mut s) % 8;
            let address = address(&mut s, offset);
            let (timestamp, old_ts) = times(&mut s);
            let (value, old) = (splitmix(&mut s), splitmix(&mut s));
            if let Ok(row) = StoreWRow::new(address, timestamp, old_ts, width, value, old) {
                r.storew.0.push(row);
                r.storew.1.push(card::StoreWRow {
                    address,
                    timestamp,
                    step: step(timestamp, old_ts),
                    width,
                    value,
                    old,
                });
            }
        }
        while r.load2w.0.len() < n {
            let width = [2u8, 4, 8][(splitmix(&mut s) % 3) as usize];
            let offset = splitmix(&mut s) % 8;
            let address = address(&mut s, offset);
            let (timestamp, old0) = times(&mut s);
            let old1 = timestamp - 1 - splitmix(&mut s) % timestamp;
            let signed = splitmix(&mut s) & 1 == 1;
            let words = [splitmix(&mut s), splitmix(&mut s)];
            if let Ok(row) = Load2WRow::new(address, timestamp, [old0, old1], width, signed, words)
            {
                r.load2w.0.push(row);
                r.load2w.1.push(card::Load2WRow {
                    address,
                    timestamp,
                    steps: [step(timestamp, old0), step(timestamp, old1)],
                    words,
                    width,
                    signed: u8::from(signed),
                });
            }
        }
        while r.store2w.0.len() < n {
            let width = [2u8, 4, 8][(splitmix(&mut s) % 3) as usize];
            let offset = splitmix(&mut s) % 8;
            let address = address(&mut s, offset);
            let (timestamp, old0) = times(&mut s);
            let old1 = timestamp - 1 - splitmix(&mut s) % timestamp;
            let value = splitmix(&mut s);
            let old = [splitmix(&mut s), splitmix(&mut s)];
            if let Ok(row) = Store2WRow::new(address, timestamp, [old0, old1], width, value, old) {
                r.store2w.0.push(row);
                r.store2w.1.push(card::Store2WRow {
                    address,
                    timestamp,
                    steps: [step(timestamp, old0), step(timestamp, old1)],
                    value,
                    old,
                    width,
                });
            }
        }
        while r.memw_w.0.len() < n {
            let kind = if splitmix(&mut s) & 1 == 1 {
                MessageKind::Write
            } else {
                MessageKind::Access
            };
            let address = address(&mut s, 0);
            let (timestamp, old_ts) = times(&mut s);
            let (old, new) = (splitmix(&mut s), splitmix(&mut s));
            if let Ok(row) = MemwWordRow::new(kind, address, timestamp, old_ts, old, new) {
                r.memw_w.0.push(row);
                r.memw_w.1.push(card::MemwWordRow {
                    address,
                    timestamp,
                    step: step(timestamp, old_ts),
                    kind: kind as u8,
                    old,
                    new,
                });
            }
        }
        r
    }

    /// The card's view of a row is the row: each field the card reads is
    /// the input the prover's row was made from.
    #[test]
    fn the_cards_view_of_a_word_row_is_the_row() {
        let r = rows(7, 256);
        assert_eq!(card_load8_rows(&r.load8.0), &r.load8.1[..]);
        assert_eq!(card_store8_rows(&r.store8.0), &r.store8.1[..]);
        assert_eq!(card_loadw_rows(&r.loadw.0), &r.loadw.1[..]);
        assert_eq!(card_storew_rows(&r.storew.0), &r.storew.1[..]);
        assert_eq!(card_load2w_rows(&r.load2w.0), &r.load2w.1[..]);
        assert_eq!(card_store2w_rows(&r.store2w.0), &r.store2w.1[..]);
        assert_eq!(card_memw_w_rows(&r.memw_w.0), &r.memw_w.1[..]);
        // Every width, sign, offset and kind is drawn.
        assert!(r.loadw.1.iter().any(|w| w.signed == 1 && w.width == 4));
        assert!(r.load2w.1.iter().any(|w| w.width == 8));
        assert!(
            r.store2w
                .1
                .iter()
                .any(|w| w.address & 0xFFFF_FFF8 == 0xFFFF_FFF8)
        );
        assert!(r.memw_w.1.iter().any(|w| w.kind == 1));
        assert!(r.memw_w.1.iter().any(|w| w.kind == 0));
    }

    fn put(out: &mut Vec<u8>, words: &[u64]) {
        words
            .iter()
            .for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
    }

    /// Each mirror row as the card's struct's bytes, field by field, the
    /// padding written as zeros (never read).
    fn bytes_of(r: &Rows, family: usize, out: &mut Vec<u8>) {
        let u = |out: &mut Vec<u8>, v: u64| out.extend_from_slice(&v.to_le_bytes());
        let w = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_le_bytes());
        match family {
            0 => r.load8.1.iter().for_each(|c| {
                u(out, c.address);
                u(out, c.timestamp);
                w(out, c.step);
                w(out, 0);
                u(out, c.word);
            }),
            1 => r.store8.1.iter().for_each(|c| {
                u(out, c.address);
                u(out, c.timestamp);
                w(out, c.step);
                w(out, 0);
                u(out, c.old);
                u(out, c.new);
            }),
            2 => r.loadw.1.iter().for_each(|c| {
                u(out, c.address);
                u(out, c.timestamp);
                w(out, c.step);
                out.extend_from_slice(&[c.width, c.signed, 0, 0]);
                u(out, c.word);
            }),
            3 => r.storew.1.iter().for_each(|c| {
                u(out, c.address);
                u(out, c.timestamp);
                w(out, c.step);
                out.extend_from_slice(&[c.width, 0, 0, 0]);
                u(out, c.value);
                u(out, c.old);
            }),
            4 => r.load2w.1.iter().for_each(|c| {
                u(out, c.address);
                u(out, c.timestamp);
                w(out, c.steps[0]);
                w(out, c.steps[1]);
                u(out, c.words[0]);
                u(out, c.words[1]);
                out.extend_from_slice(&[c.width, c.signed, 0, 0, 0, 0, 0, 0]);
            }),
            5 => r.store2w.1.iter().for_each(|c| {
                u(out, c.address);
                u(out, c.timestamp);
                w(out, c.steps[0]);
                w(out, c.steps[1]);
                u(out, c.value);
                u(out, c.old[0]);
                u(out, c.old[1]);
                out.extend_from_slice(&[c.width, 0, 0, 0, 0, 0, 0, 0]);
            }),
            _ => r.memw_w.1.iter().for_each(|c| {
                u(out, c.address);
                u(out, c.timestamp);
                w(out, c.step);
                out.extend_from_slice(&[c.kind, 0, 0, 0]);
                u(out, c.old);
                u(out, c.new);
            }),
        }
    }

    /// Family `family`'s host fill of `r`'s rows, column-major.
    fn host_fill(r: &Rows, family: usize) -> Vec<u64> {
        use crate::tables::{
            mem_load2w, mem_load8, mem_loadw, mem_store2w, mem_store8, mem_storew, memw_w,
        };
        let f = TraceForm::Wide;
        col_major(&match family {
            0 => mem_load8::generate_mem_load8_trace_segments_as(&[&r.load8.0], f),
            1 => mem_store8::generate_mem_store8_trace_segments_as(&[&r.store8.0], f),
            2 => mem_loadw::generate_mem_loadw_trace_segments_as(&[&r.loadw.0], f),
            3 => mem_storew::generate_mem_storew_trace_segments_as(&[&r.storew.0], f),
            4 => mem_load2w::generate_mem_load2w_trace_segments_as(&[&r.load2w.0], f),
            5 => mem_store2w::generate_mem_store2w_trace_segments_as(&[&r.store2w.0], f),
            _ => memw_w::generate_memw_w_trace_segments_as(&[&r.memw_w.0], f),
        })
    }

    /// Writes the seven word fills' known-answer vectors to
    /// `WORD_FILL_VECTORS` for `crypto/math-cuda/tests/host_kat/
    /// word_fill_host_kat.cpp`: per family, tables of several heights (whole,
    /// padded), each its rows as the card gets them and the words the host
    /// fill writes.
    #[test]
    #[ignore = "writes the word fills' vectors to WORD_FILL_VECTORS"]
    fn write_the_word_fill_kernel_vectors() {
        let path = std::env::var("WORD_FILL_VECTORS").expect("WORD_FILL_VECTORS");
        let mut out: Vec<u8> = b"CTWRDFL1".to_vec();
        let sizes = [1usize, 3, 4, 5, 64, 1000, 4096];
        put(&mut out, &[sizes.len() as u64]);
        for family in 0..7 {
            for (k, &n) in sizes.iter().enumerate() {
                let r = rows(1000 * family as u64 + k as u64, n);
                let want = host_fill(&r, family);
                put(&mut out, &[n as u64]);
                bytes_of(&r, family, &mut out);
                put(&mut out, &[want.len() as u64]);
                put(&mut out, &want);
            }
        }
        std::fs::write(&path, &out).expect("the vectors written");
        println!(
            "WORD FILL VECTORS: 7 families × {} sizes, {} bytes → {path}",
            sizes.len(),
            out.len()
        );
    }
}
