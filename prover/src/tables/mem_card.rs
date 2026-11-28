//! The fused memory tables' rows as the card's fill reads them: MEM_LOAD's
//! and MEM_STORE's records (`mem_load::MemLoadRow`, `mem_store::MemStoreRow`)
//! read in place as `multilinear::records::MemLoadRow` / `MemStoreRow` and
//! filled by `fill_mem_load` / `fill_mem_store` (`crypto/math-cuda/kernels/
//! fill.cu`), which write the words the host fills
//! (`generate_mem_load_trace_segments_as`, `generate_mem_store_trace_segments_as`)
//! write for the same rows.

use super::mem_load::MemLoadRow;
use super::mem_store::MemStoreRow;

/// MEM_LOAD rows as the card's fill reads them: the same bytes.
pub(crate) fn card_load_rows(rows: &[MemLoadRow]) -> &[multilinear::records::MemLoadRow] {
    use multilinear::records::MemLoadRow as Card;
    use std::mem::{align_of, offset_of, size_of};
    // The prover's offsets are asserted beside its struct (`mem_load.rs`): 0,
    // 8, 16, 20, 28, 29 in 32 bytes. The card's mirror's are these.
    const _: () = {
        assert!(size_of::<MemLoadRow>() == size_of::<Card>());
        assert!(align_of::<MemLoadRow>() == align_of::<Card>());
        assert!(offset_of!(Card, address) == 0);
        assert!(offset_of!(Card, timestamp) == 8);
        assert!(offset_of!(Card, step) == 16);
        assert!(offset_of!(Card, res) == 20);
        assert!(offset_of!(Card, width) == 28);
        assert!(offset_of!(Card, signed) == 29);
    };
    // SAFETY: both are `#[repr(C)]` with the same fields at the same offsets
    // (asserted above and beside the prover's struct); the card's `signed` is
    // a `u8` and reads the prover's `bool` (0 or 1). The trailing padding is
    // never read as a field.
    unsafe { std::slice::from_raw_parts(rows.as_ptr().cast(), rows.len()) }
}

/// MEM_STORE rows as the card's fill reads them: the same bytes.
pub(crate) fn card_store_rows(rows: &[MemStoreRow]) -> &[multilinear::records::MemStoreRow] {
    use multilinear::records::MemStoreRow as Card;
    use std::mem::{align_of, offset_of, size_of};
    // The prover's offsets are asserted beside its struct (`mem_store.rs`):
    // 0, 8, 16, 20, 24, 32 in 40 bytes.
    const _: () = {
        assert!(size_of::<MemStoreRow>() == size_of::<Card>());
        assert!(align_of::<MemStoreRow>() == align_of::<Card>());
        assert!(offset_of!(Card, address) == 0);
        assert!(offset_of!(Card, timestamp) == 8);
        assert!(offset_of!(Card, step) == 16);
        assert!(offset_of!(Card, width) == 20);
        assert!(offset_of!(Card, value) == 24);
        assert!(offset_of!(Card, old) == 32);
    };
    // SAFETY: as above; the padding after `width` is never read as a field.
    unsafe { std::slice::from_raw_parts(rows.as_ptr().cast(), rows.len()) }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tables::card_fill_inputs::{col_major, splitmix};
    use crate::tables::gpack::TraceForm;

    /// `n` MEM_LOAD rows over every width, sign and a spread of steps, from
    /// `seed`; each the row `MemLoadRow::new` makes (its own checks pass).
    pub(crate) fn load_rows(seed: u64, n: usize) -> Vec<MemLoadRow> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let width = [1u8, 2, 4, 8][(splitmix(&mut s) % 4) as usize];
            let address = splitmix(&mut s) & !(u64::from(width) - 1);
            let timestamp = 4 * (splitmix(&mut s) % (1 << 28)) + 4;
            let gap = 1 + splitmix(&mut s) % timestamp.min(1 << 31);
            let signed = splitmix(&mut s) & 1 == 1;
            let raw = splitmix(&mut s).to_le_bytes();
            let mut res = raw;
            if width < 8 {
                let fill = if signed && raw[usize::from(width) - 1] >> 7 == 1 {
                    0xFF
                } else {
                    0
                };
                res[usize::from(width)..].iter_mut().for_each(|b| *b = fill);
            }
            if let Ok(row) =
                MemLoadRow::new(address, timestamp, timestamp - gap, width, signed, res)
            {
                out.push(row);
            }
        }
        out
    }

    /// `n` MEM_STORE rows likewise.
    pub(crate) fn store_rows(seed: u64, n: usize) -> Vec<MemStoreRow> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let width = [1u8, 2, 4, 8][(splitmix(&mut s) % 4) as usize];
            let address = splitmix(&mut s) & !(u64::from(width) - 1);
            let timestamp = 4 * (splitmix(&mut s) % (1 << 28)) + 4;
            let gap = 1 + splitmix(&mut s) % timestamp.min(1 << 31);
            let (value, old) = (splitmix(&mut s), splitmix(&mut s));
            if let Ok(row) =
                MemStoreRow::new(address, timestamp, timestamp - gap, width, value, old)
            {
                out.push(row);
            }
        }
        out
    }

    /// The card's view of a row is the row: each field the card reads is the
    /// prover's.
    #[test]
    fn the_cards_view_of_a_fused_row_is_the_row() {
        for r in load_rows(1, 64) {
            let c = card_load_rows(std::slice::from_ref(&r))[0];
            assert_eq!(
                (
                    c.address,
                    c.timestamp,
                    c.step,
                    c.res,
                    c.width,
                    c.signed != 0
                ),
                (
                    r.address(),
                    r.timestamp(),
                    r.step(),
                    r.res(),
                    r.width(),
                    r.signed()
                )
            );
        }
        for r in store_rows(2, 64) {
            let c = card_store_rows(std::slice::from_ref(&r))[0];
            assert_eq!(
                (c.address, c.timestamp, c.step, c.width, c.value, c.old),
                (
                    r.address(),
                    r.timestamp(),
                    r.step(),
                    r.width(),
                    r.value(),
                    r.old()
                )
            );
        }
    }

    /// MEM_LOAD rows as the card's 32-byte records, field by field, the
    /// padding written as zeros (never read).
    fn load_bytes(rows: &[MemLoadRow]) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 * rows.len());
        for r in rows {
            out.extend_from_slice(&r.address().to_le_bytes());
            out.extend_from_slice(&r.timestamp().to_le_bytes());
            out.extend_from_slice(&r.step().to_le_bytes());
            out.extend_from_slice(&r.res());
            out.extend_from_slice(&[r.width(), u8::from(r.signed()), 0, 0]);
        }
        out
    }

    /// MEM_STORE rows as the card's 40-byte records likewise.
    fn store_bytes(rows: &[MemStoreRow]) -> Vec<u8> {
        let mut out = Vec::with_capacity(40 * rows.len());
        for r in rows {
            out.extend_from_slice(&r.address().to_le_bytes());
            out.extend_from_slice(&r.timestamp().to_le_bytes());
            out.extend_from_slice(&r.step().to_le_bytes());
            out.extend_from_slice(&[r.width(), 0, 0, 0]);
            out.extend_from_slice(&r.value().to_le_bytes());
            out.extend_from_slice(&r.old().to_le_bytes());
        }
        out
    }

    fn put(out: &mut Vec<u8>, words: &[u64]) {
        words
            .iter()
            .for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
    }

    /// Writes `fill_mem_load` / `fill_mem_store`'s known-answer vectors to
    /// `MEM_FILL_VECTORS` for `crypto/math-cuda/tests/host_kat/
    /// mem_fill_host_kat.cpp`: tables of several heights (whole, padded), each
    /// its rows as the card gets them and the words the host fill writes.
    #[test]
    #[ignore = "writes the fused fills' vectors to MEM_FILL_VECTORS"]
    fn write_the_fused_fill_kernel_vectors() {
        use crate::tables::mem_load::generate_mem_load_trace_segments_as;
        use crate::tables::mem_store::generate_mem_store_trace_segments_as;
        let path = std::env::var("MEM_FILL_VECTORS").expect("MEM_FILL_VECTORS");
        let mut out: Vec<u8> = b"CTMEMFL1".to_vec();
        let sizes = [1usize, 3, 4, 5, 64, 1000, 4096];
        put(&mut out, &[sizes.len() as u64]);
        for (k, &n) in sizes.iter().enumerate() {
            let loads = load_rows(100 + k as u64, n);
            let stores = store_rows(200 + k as u64, n);
            let want_l = col_major(&generate_mem_load_trace_segments_as(
                &[&loads],
                TraceForm::Wide,
            ));
            let want_s = col_major(&generate_mem_store_trace_segments_as(
                &[&stores],
                TraceForm::Wide,
            ));
            put(&mut out, &[n as u64]);
            out.extend_from_slice(&load_bytes(&loads));
            put(&mut out, &[want_l.len() as u64]);
            put(&mut out, &want_l);
            out.extend_from_slice(&store_bytes(&stores));
            put(&mut out, &[want_s.len() as u64]);
            put(&mut out, &want_s);
        }
        std::fs::write(&path, &out).expect("the vectors written");
        println!(
            "MEM FILL VECTORS: {} sizes, {} bytes → {path}",
            sizes.len(),
            out.len()
        );
    }
}
