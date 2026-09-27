//! Temporary A/B knobs for the gap-fix campaign's prover-side fixes, one per
//! fix, default OFF.
//!
//! `LAMBDA_VM_GAP_<ID>=1` turns fix `<ID>` on. Unset, empty or `0` leaves the
//! pre-fix code path untouched, so one binary runs both arms of an A/B, and
//! none of these fixes changes a proof byte or a program identity. Each knob is
//! read once per process and prints one line the first time it reads ON, so a
//! log shows the knob reached the path it names. A knob is deleted when its
//! fix is kept (the fix becomes the only path) or dropped.
//!
//! The device-layer fixes (I1, I5) read theirs in `math_cuda::gap`; [`banner`]
//! names all five so one line states an arm's whole setting.

use std::sync::OnceLock;

/// `None`, `""` and `"0"` read as off, `"1"` as on. Anything else panics with
/// the knob's name: a typo that silently read as off would turn a B arm into a
/// second A arm.
pub fn parse(name: &str, value: Option<&str>) -> bool {
    match value {
        None | Some("") | Some("0") => false,
        Some("1") => true,
        Some(other) => panic!("{name} must be `0` or `1`, got `{other}`"),
    }
}

fn read(name: &str, what: &str) -> bool {
    let on = parse(name, std::env::var(name).ok().as_deref());
    if on {
        println!("GAP KNOB {name}=1 read: {what}");
    }
    on
}

/// Every gap knob, in fix order: I1 and I5 are read by the device layer.
pub const ALL: [&str; 5] = [
    "LAMBDA_VM_GAP_I1",
    "LAMBDA_VM_GAP_I2",
    "LAMBDA_VM_GAP_I3",
    "LAMBDA_VM_GAP_I4",
    "LAMBDA_VM_GAP_I5",
];

/// One line naming every knob's value as this process sees it, e.g.
/// `GAP KNOBS: I1=0 I2=1 I3=0 I4=0 I5=0`.
pub fn banner() -> String {
    let fields: Vec<String> = ALL
        .iter()
        .map(|name| {
            let on = parse(name, std::env::var(name).ok().as_deref());
            format!(
                "{}={}",
                name.trim_start_matches("LAMBDA_VM_GAP_"),
                u8::from(on)
            )
        })
        .collect();
    format!("GAP KNOBS: {}", fields.join(" "))
}

/// I2: the level-0 lead-in takes the base's own DECODE derivations (the
/// univariate commitment, and on WHIR the prepared opening) instead of
/// deriving them from the ELF a second time.
pub fn i2_reuse_base_decode() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        read(
            "LAMBDA_VM_GAP_I2",
            "the level-0 lead-in reuses the base's DECODE derivations",
        )
    })
}

/// I3: an artifact build holds the card permit only around its device
/// commits, not across the whole host build.
pub fn i3_narrow_artifact_permit() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        read(
            "LAMBDA_VM_GAP_I3",
            "build_artifacts holds the card permit only around its device commits",
        )
    })
}

/// I4: each base epoch's host preparation (and the global proof's) runs off
/// the prover thread, ahead of the prove that needs it.
pub fn i4_prep_off_prover_thread() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        read(
            "LAMBDA_VM_GAP_I4",
            "base epoch and global preparation run off the prover thread",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn off_unless_exactly_one() {
        assert!(!parse("K", None));
        assert!(!parse("K", Some("")));
        assert!(!parse("K", Some("0")));
        assert!(parse("K", Some("1")));
    }

    #[test]
    #[should_panic(expected = "K must be `0` or `1`, got `true`")]
    fn anything_else_is_refused() {
        parse("K", Some("true"));
    }
}
