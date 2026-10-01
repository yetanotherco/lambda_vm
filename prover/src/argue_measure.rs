//! ⛔⛔ MEASUREMENT ONLY — `LAMBDA_VM_ARGUE_BATCHED_MEASURE=1`. NEVER A DEFAULT,
//! NEVER A PRODUCTION PATH, AND IT MUST NOT BECOME ONE.
//!
//! What it is for: timing the base with the batched argue (D-BATCH B-3's
//! base-only A/B) before any production caller can carry the batched proof
//! type. Under it, every epoch's argue runs `multi_prove_batched` in place of
//! `multi_prove`, and the epoch proof that comes back carries the roots and
//! the openings but NO argue: it is a placeholder that does not verify and
//! must never be wrapped. The global proof is argued as today.
//!
//! The guard, both halves:
//! - the knob is honoured only while [`permit_base_only`] is held, and that
//!   exists only in TEST builds (the base-only timing test takes it). A process
//!   that sets the knob without the permit is refused at its first epoch
//!   prove, so no production binary can ever argue this way;
//! - while the knob is set, every epoch verify and every level-0 wrap harvest
//!   is refused ([`refuse_consumers`]), so a placeholder cannot reach anything
//!   that would read it.
//!
//! The real wiring of the batched format into the epoch proof is D-BATCH B-4b,
//! and replaces this knob; it does not grow out of it.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::Error;

/// The knob. Set to `1` to request the measurement.
pub const ENV: &str = "LAMBDA_VM_ARGUE_BATCHED_MEASURE";

thread_local! {
    /// A test's override of the environment, on its own thread only (other
    /// tests prove epochs beside it): 0 = read the env, 1 = off, 2 = on.
    static FORCED: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// How many base-only permits are held (test builds only take one).
static PERMITS: AtomicU8 = AtomicU8::new(0);

/// Whether the measurement was asked for.
pub fn requested() -> bool {
    match FORCED.with(std::cell::Cell::get) {
        1 => false,
        2 => true,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var(ENV).is_ok_and(|v| v == "1"))
        }
    }
}

fn permitted() -> bool {
    PERMITS.load(Ordering::Relaxed) > 0
}

/// Whether this epoch prove argues batched: `Ok(false)` unless requested, and
/// a REFUSAL when requested without the base-only permit.
pub(crate) fn active() -> Result<bool, Error> {
    if !requested() {
        return Ok(false);
    }
    if !permitted() {
        return Err(Error::Prover(format!(
            "{ENV}=1 is a measurement-only knob for the base-only timing test, and this \
             process holds no base-only permit: refused before any argue"
        )));
    }
    Ok(true)
}

/// `LAMBDA_VM_ARGUE_BATCHED_MEASURE_CAP`: the bin cap (`log2` cells) the
/// measurement argues under — so a ledger run can read a bin's peak at 26 and
/// at 27 (D-BATCH's own fallback rule). Measurement only, like the knob: the
/// placeholder proofs it shapes are never verified. Unset is the format's 27.
pub const CAP_ENV: &str = "LAMBDA_VM_ARGUE_BATCHED_MEASURE_CAP";

/// The measurement's bin cap.
pub(crate) fn bin_log_cells() -> Result<u8, Error> {
    match std::env::var(CAP_ENV) {
        Err(_) => match multilinear::whir_chain::ArgueFormat::BATCHED {
            multilinear::whir_chain::ArgueFormat::Batched { bin_log_cells } => Ok(bin_log_cells),
            multilinear::whir_chain::ArgueFormat::PerTable => Ok(27),
        },
        Ok(v) => v
            .parse::<u8>()
            .ok()
            .filter(|cap| (20..=30).contains(cap))
            .ok_or_else(|| Error::Prover(format!("{CAP_ENV}={v} must be in 20..=30"))),
    }
}

/// The refusal every consumer of an epoch proof calls first: while the knob
/// is set, an epoch proof may be a placeholder with no argue.
pub(crate) fn refuse_consumers(what: &str) -> Result<(), Error> {
    if requested() {
        return Err(Error::Prover(format!(
            "{what} refused: {ENV}=1 is set, and the epoch proofs of this process carry no \
             argue (measurement only)"
        )));
    }
    Ok(())
}

/// The base-only timing test's permit: while held, a requested measurement is
/// honoured. Test builds only.
#[cfg(test)]
pub(crate) struct Permit(());

#[cfg(test)]
pub(crate) fn permit_base_only() -> Permit {
    PERMITS.fetch_add(1, Ordering::Relaxed);
    Permit(())
}

#[cfg(test)]
impl Drop for Permit {
    fn drop(&mut self) {
        PERMITS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A test's override: `Some(on)` forces, `None` gives the environment back.
#[cfg(test)]
pub(crate) fn force(on: Option<bool>) {
    FORCED.with(|f| {
        f.set(match on {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ The guard refuses: requested without the permit, the epoch prover
    /// refuses; with it, it argues batched — and every consumer of an epoch
    /// proof is refused while the knob is set, permit or not.
    #[test]
    fn the_measurement_knob_refuses_without_its_permit_and_blocks_every_consumer() {
        force(Some(false));
        assert_eq!(active().ok(), Some(false), "unset: today's argue");
        assert!(
            refuse_consumers("an epoch verify").is_ok(),
            "unset: consumers run"
        );

        force(Some(true));
        assert!(active().is_err(), "set without the permit: refused");
        assert!(
            refuse_consumers("an epoch verify").is_err(),
            "set: an epoch verify is refused"
        );
        {
            let _permit = permit_base_only();
            assert_eq!(active().ok(), Some(true), "set with the permit: batched");
            assert!(
                refuse_consumers("a level-0 wrap harvest").is_err(),
                "the permit never lets a consumer through"
            );
        }
        assert!(active().is_err(), "the permit dropped: refused again");
        force(None);
    }
}
