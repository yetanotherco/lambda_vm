//! Emits a per-proof trace-shape profile as JSONL — heights, widths, constraint
//! and interaction counts, never trace values. One line per proof; a
//! continuation epoch maps onto one line.
//!
//! Four deliberate mismatches with the consumer's model, none of which affect
//! the prover-cost dimensions being measured: `count_weight` is emitted as 1;
//! bus ids are remapped to a dense `u16` range; message lengths drop our leading
//! bus-id element; and preprocessed columns are reported in `preprocessed`, with
//! `cached_mains` always empty.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use stark::lookup::BusInteraction;
use stark::traits::AIR;

/// Schema version of the emitted JSONL. Matches what the consumer expects.
const SCHEMA: &str = "v2";

/// Per-AIR column widths, split the way the consumer's keygen splits them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileTraceWidth {
    pub preprocessed: Option<usize>,
    pub cached_mains: Vec<usize>,
    pub common_main: usize,
    pub after_challenge: Vec<usize>,
}

/// Shape of one AIR within a segment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AirShapeRecord {
    pub air_name: String,
    pub air_id: usize,
    pub log_height: usize,
    pub height: usize,
    pub width: ProfileTraceWidth,
    pub num_constraints: usize,
    pub num_interactions: usize,
    pub max_constraint_degree: usize,
    pub buses: Vec<u16>,
    pub interaction_message_lens: Vec<usize>,
    pub interaction_count_weights: Vec<u32>,
}

/// One proof's worth of AIR shapes — one JSONL line.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentProfile {
    pub schema: String,
    pub segment_idx: usize,
    pub global_max_constraint_degree: usize,
    pub airs: Vec<AirShapeRecord>,
}

/// Assigns dense `u16` bus indices to our sparse `u64` bus ids.
///
/// Shared across a run so an id keeps its index across segments; interactions
/// on the same bus must land on the same index for the replayed LogUp to have
/// the same bus structure.
#[derive(Default)]
pub struct BusIndexMap {
    map: HashMap<u64, u16>,
}

impl BusIndexMap {
    pub fn new() -> Self {
        Self::default()
    }

    fn index_of(&mut self, bus_id: u64) -> u16 {
        let next = self.map.len();
        *self.map.entry(bus_id).or_insert_with(|| {
            u16::try_from(next).expect("bus count exceeds u16 — the bus index is u16")
        })
    }
}

/// The shape inputs one AIR contributes, decoupled from the [`AIR`] trait so
/// the mapping below is testable without standing up a full AIR.
pub struct AirShapeInput<'a> {
    pub name: &'a str,
    pub height: usize,
    pub common_main: usize,
    pub after_challenge_width: usize,
    pub num_precomputed: usize,
    pub num_constraints: usize,
    pub max_constraint_degree: usize,
    pub interactions: &'a [BusInteraction],
}

impl<'a> AirShapeInput<'a> {
    /// Reads every shape dimension off a live AIR.
    pub fn from_air<A>(air: &'a A, height: usize) -> Self
    where
        A: AIR + ?Sized,
    {
        let (common_main, after_challenge_width) = air.trace_layout();
        Self {
            name: air.name(),
            height,
            common_main,
            after_challenge_width,
            num_precomputed: air.num_precomputed_columns(),
            num_constraints: air.num_transition_constraints(),
            max_constraint_degree: air.max_constraint_degree(),
            interactions: air.bus_interactions(),
        }
    }
}

/// Builds the shape record for one AIR.
fn record_for(input: &AirShapeInput<'_>, air_id: usize, buses: &mut BusIndexMap) -> AirShapeRecord {
    let preprocessed = match input.num_precomputed {
        0 => None,
        n => Some(n),
    };
    let after_challenge = match input.after_challenge_width {
        0 => Vec::new(),
        n => vec![n],
    };

    let mut bus_indices = Vec::with_capacity(input.interactions.len());
    let mut message_lens = Vec::with_capacity(input.interactions.len());
    for interaction in input.interactions {
        bus_indices.push(buses.index_of(interaction.bus_id));
        // Ours counts the bus id as element 0; theirs counts message fields only.
        message_lens.push(interaction.num_bus_elements().saturating_sub(1));
    }

    AirShapeRecord {
        air_name: input.name.to_string(),
        air_id,
        log_height: input.height.max(1).trailing_zeros() as usize,
        height: input.height,
        width: ProfileTraceWidth {
            preprocessed,
            cached_mains: Vec::new(),
            common_main: input.common_main,
            after_challenge,
        },
        num_constraints: input.num_constraints,
        num_interactions: input.interactions.len(),
        max_constraint_degree: input.max_constraint_degree,
        buses: bus_indices,
        interaction_message_lens: message_lens,
        interaction_count_weights: vec![1; input.interactions.len()],
    }
}

/// Builds a segment profile from per-AIR shapes in proving order.
pub fn build_segment<'a>(
    segment_idx: usize,
    inputs: impl IntoIterator<Item = AirShapeInput<'a>>,
    buses: &mut BusIndexMap,
) -> SegmentProfile {
    let airs: Vec<AirShapeRecord> = inputs
        .into_iter()
        .enumerate()
        .map(|(air_id, input)| record_for(&input, air_id, buses))
        .collect();

    let global_max_constraint_degree = airs
        .iter()
        .map(|a| a.max_constraint_degree)
        .max()
        .unwrap_or(0);

    SegmentProfile {
        schema: SCHEMA.to_string(),
        segment_idx,
        global_max_constraint_degree,
        airs,
    }
}

/// Scales every AIR's `common_main` to estimate the width the same table would
/// need over a 31-bit field, where our 64-bit-field limbs no longer fit.
///
/// The result is an estimate for a second, pessimistic replay — it is not a
/// claim about any real BabyBear arithmetization. Emit it as a separate profile
/// and compare the two runs separately.
pub fn scale_widths(profile: &SegmentProfile, factor_for: impl Fn(&str) -> f64) -> SegmentProfile {
    let mut scaled = profile.clone();
    for air in &mut scaled.airs {
        let factor = factor_for(&air.air_name);
        air.width.common_main = ((air.width.common_main as f64) * factor).ceil() as usize;
        if let Some(pre) = air.width.preprocessed.as_mut() {
            *pre = ((*pre as f64) * factor).ceil() as usize;
        }
    }
    scaled
}

/// Appends one segment profile as a JSONL line.
pub fn append_jsonl(path: &std::path::Path, profile: &SegmentProfile) -> std::io::Result<()> {
    use std::io::Write;

    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let line = serde_json::to_string(profile)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writeln!(file, "{line}")
}

/// Environment variable holding the output path. Unset means no capture.
pub const PROFILE_PATH_ENV: &str = "LAMBDA_VM_SHAPE_PROFILE";

/// Bus indices and the segment counter, shared by every capture in a process so
/// a bus id keeps one index across segments and each proof gets its own line.
static CAPTURE_STATE: std::sync::Mutex<Option<(BusIndexMap, usize)>> = std::sync::Mutex::new(None);

/// Appends one line for this proof, if `LAMBDA_VM_SHAPE_PROFILE` is set.
///
/// A capture failure is logged and ignored: profiling must never fail a proof.
pub fn capture<'a, A>(airs: impl IntoIterator<Item = (&'a A, usize)>)
where
    A: AIR + ?Sized + 'a,
{
    let Ok(path) = std::env::var(PROFILE_PATH_ENV) else {
        return;
    };

    let inputs: Vec<(&'a A, usize)> = airs.into_iter().collect();
    let mut guard = match CAPTURE_STATE.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let state = guard.get_or_insert_with(|| (BusIndexMap::new(), 0));
    let (buses, next_idx) = state;

    let segment_idx = *next_idx;
    *next_idx += 1;

    let profile = build_segment(
        segment_idx,
        inputs
            .iter()
            .map(|(air, height)| AirShapeInput::from_air(*air, *height)),
        buses,
    );
    drop(guard);

    if let Err(e) = append_jsonl(std::path::Path::new(&path), &profile) {
        log::warn!("shape profile: could not append to {path}: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark::lookup::{BusValue, Multiplicity};

    fn interaction(bus_id: u64, num_values: usize) -> BusInteraction {
        BusInteraction::sender(
            bus_id,
            Multiplicity::One,
            (0..num_values).map(BusValue::column).collect(),
        )
    }

    fn shape<'a>(
        name: &'a str,
        height: usize,
        interactions: &'a [BusInteraction],
    ) -> AirShapeInput<'a> {
        AirShapeInput {
            name,
            height,
            common_main: 64,
            after_challenge_width: 3,
            num_precomputed: 0,
            num_constraints: 12,
            max_constraint_degree: 3,
            interactions,
        }
    }

    #[test]
    fn per_interaction_vectors_agree_with_num_interactions() {
        let ints = vec![interaction(7, 4), interaction(9, 2)];
        let mut buses = BusIndexMap::new();
        let profile = build_segment(0, [shape("CPU", 1 << 19, &ints)], &mut buses);

        let rec = &profile.airs[0];
        assert_eq!(rec.num_interactions, 2);
        assert_eq!(rec.buses.len(), rec.num_interactions);
        assert_eq!(rec.interaction_message_lens.len(), rec.num_interactions);
        assert_eq!(rec.interaction_count_weights.len(), rec.num_interactions);
    }

    #[test]
    fn message_len_drops_our_leading_bus_id_element() {
        // `num_bus_elements()` counts the bus id plus one element per value.
        let ints = vec![interaction(7, 4)];
        assert_eq!(ints[0].num_bus_elements(), 5);

        let mut buses = BusIndexMap::new();
        let profile = build_segment(0, [shape("CPU", 1 << 19, &ints)], &mut buses);

        assert_eq!(profile.airs[0].interaction_message_lens, vec![4]);
    }

    #[test]
    fn height_and_log_height_agree() {
        let ints = Vec::new();
        let mut buses = BusIndexMap::new();
        let profile = build_segment(0, [shape("MUL", 1 << 20, &ints)], &mut buses);

        let rec = &profile.airs[0];
        assert_eq!(rec.height, 1 << 20);
        assert_eq!(rec.log_height, 20);
        assert_eq!(1usize << rec.log_height, rec.height);
    }

    #[test]
    fn single_row_table_has_log_height_zero() {
        let ints = Vec::new();
        let mut buses = BusIndexMap::new();
        let profile = build_segment(0, [shape("HALT", 1, &ints)], &mut buses);

        assert_eq!(profile.airs[0].log_height, 0);
        assert_eq!(profile.airs[0].height, 1);
    }

    #[test]
    fn same_bus_id_keeps_one_index_across_airs_and_segments() {
        let a_ints = vec![interaction(7, 1), interaction(9, 1)];
        let b_ints = vec![interaction(9, 1), interaction(7, 1)];
        let mut buses = BusIndexMap::new();

        let first = build_segment(
            0,
            [
                shape("CPU", 1 << 19, &a_ints),
                shape("MEMW", 1 << 19, &b_ints),
            ],
            &mut buses,
        );
        let second = build_segment(1, [shape("MEMW", 1 << 19, &b_ints)], &mut buses);

        // Each bus id gets exactly one index, and the order within an AIR is the
        // declaration order, not the discovery order.
        assert_eq!(first.airs[0].buses, vec![0, 1]);
        assert_eq!(first.airs[1].buses, vec![1, 0]);
        assert_eq!(second.airs[0].buses, vec![1, 0]);
    }

    #[test]
    fn air_id_follows_proving_order() {
        let ints = Vec::new();
        let mut buses = BusIndexMap::new();
        let profile = build_segment(
            0,
            [
                shape("BITWISE", 1 << 20, &ints),
                shape("DECODE", 1 << 21, &ints),
                shape("CPU", 1 << 19, &ints),
            ],
            &mut buses,
        );

        let ids: Vec<usize> = profile.airs.iter().map(|a| a.air_id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(profile.airs[1].air_name, "DECODE");
    }

    #[test]
    fn widths_split_preprocessed_and_after_challenge() {
        let ints = Vec::new();
        let mut input = shape("BITWISE", 1 << 20, &ints);
        input.num_precomputed = 3;
        let mut buses = BusIndexMap::new();
        let profile = build_segment(0, [input], &mut buses);

        let w = &profile.airs[0].width;
        assert_eq!(w.preprocessed, Some(3));
        assert_eq!(w.common_main, 64);
        assert_eq!(w.after_challenge, vec![3]);
        assert!(w.cached_mains.is_empty());
    }

    #[test]
    fn no_aux_columns_emits_empty_after_challenge() {
        let ints = Vec::new();
        let mut input = shape("HALT", 1, &ints);
        input.after_challenge_width = 0;
        let mut buses = BusIndexMap::new();
        let profile = build_segment(0, [input], &mut buses);

        assert!(profile.airs[0].width.after_challenge.is_empty());
        assert_eq!(profile.airs[0].width.preprocessed, None);
    }

    #[test]
    fn global_degree_is_the_max_over_airs() {
        let ints = Vec::new();
        let mut a = shape("CPU", 1 << 19, &ints);
        a.max_constraint_degree = 3;
        let mut b = shape("MUL", 1 << 20, &ints);
        b.max_constraint_degree = 5;
        let mut buses = BusIndexMap::new();

        let profile = build_segment(0, [a, b], &mut buses);
        assert_eq!(profile.global_max_constraint_degree, 5);
    }

    #[test]
    fn width_scaling_applies_per_air_and_leaves_the_original() {
        let ints = Vec::new();
        let mut buses = BusIndexMap::new();
        let profile = build_segment(
            0,
            [shape("MUL", 1 << 20, &ints), shape("EQ", 1 << 20, &ints)],
            &mut buses,
        );

        let scaled = scale_widths(&profile, |name| if name == "MUL" { 2.0 } else { 1.0 });
        assert_eq!(scaled.airs[0].width.common_main, 128);
        assert_eq!(scaled.airs[1].width.common_main, 64);
        assert_eq!(profile.airs[0].width.common_main, 64);
    }

    #[test]
    fn width_scaling_rounds_up() {
        let ints = Vec::new();
        let mut input = shape("LT", 1 << 20, &ints);
        input.common_main = 41;
        let mut buses = BusIndexMap::new();
        let profile = build_segment(0, [input], &mut buses);

        let scaled = scale_widths(&profile, |_| 1.5);
        // 41 * 1.5 = 61.5 — a fractional column is still a whole column.
        assert_eq!(scaled.airs[0].width.common_main, 62);
    }

    #[test]
    fn serialized_line_carries_every_consumer_field() {
        let ints = vec![interaction(7, 4)];
        let mut buses = BusIndexMap::new();
        let profile = build_segment(0, [shape("CPU", 1 << 19, &ints)], &mut buses);

        let json = serde_json::to_string(&profile).unwrap();
        for key in [
            "\"schema\"",
            "\"segment_idx\"",
            "\"global_max_constraint_degree\"",
            "\"air_name\"",
            "\"air_id\"",
            "\"log_height\"",
            "\"height\"",
            "\"preprocessed\"",
            "\"cached_mains\"",
            "\"common_main\"",
            "\"after_challenge\"",
            "\"num_constraints\"",
            "\"num_interactions\"",
            "\"max_constraint_degree\"",
            "\"buses\"",
            "\"interaction_message_lens\"",
            "\"interaction_count_weights\"",
        ] {
            assert!(json.contains(key), "missing {key} in {json}");
        }
        assert!(json.contains("\"schema\":\"v2\""));
    }

    /// A line of the consumer's own captured profile must deserialize into our
    /// mirror of the schema — this is what pins the two definitions together.
    #[test]
    fn parses_a_line_from_a_real_capture() {
        let line = r#"{"schema":"v2","segment_idx":0,"global_max_constraint_degree":4,"airs":[{"air_name":"VmConnectorAir","air_id":1,"log_height":1,"height":2,"width":{"preprocessed":null,"cached_mains":[],"common_main":6,"after_challenge":[]},"num_constraints":8,"num_interactions":5,"max_constraint_degree":3,"buses":[0,0,2,3,3],"interaction_message_lens":[2,2,9,2,2],"interaction_count_weights":[1,1,1,1,1]}]}"#;

        let parsed: SegmentProfile = serde_json::from_str(line).unwrap();
        assert_eq!(parsed.schema, SCHEMA);
        assert_eq!(parsed.global_max_constraint_degree, 4);

        let air = &parsed.airs[0];
        assert_eq!(air.air_name, "VmConnectorAir");
        assert_eq!(air.num_interactions, 5);
        assert_eq!(air.buses.len(), air.num_interactions);
        assert_eq!(air.interaction_message_lens, vec![2, 2, 9, 2, 2]);
        assert_eq!(air.width.preprocessed, None);
        assert!(air.width.cached_mains.is_empty());
    }

    #[test]
    fn round_trips_through_the_consumer_shape() {
        let ints = vec![interaction(7, 4), interaction(9, 2)];
        let mut buses = BusIndexMap::new();
        let profile = build_segment(3, [shape("CPU", 1 << 19, &ints)], &mut buses);

        let json = serde_json::to_string(&profile).unwrap();
        let back: SegmentProfile = serde_json::from_str(&json).unwrap();

        assert_eq!(back.segment_idx, 3);
        assert_eq!(back.airs[0].buses, profile.airs[0].buses);
        assert_eq!(
            back.airs[0].interaction_message_lens,
            profile.airs[0].interaction_message_lens
        );
    }

    #[test]
    fn appends_one_line_per_segment() {
        let dir = std::env::temp_dir().join("lambda_vm_shape_profile_test");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("profile.jsonl");

        let ints = Vec::new();
        let mut buses = BusIndexMap::new();
        for idx in 0..3 {
            let profile = build_segment(idx, [shape("CPU", 1 << 19, &ints)], &mut buses);
            append_jsonl(&path, &profile).unwrap();
        }

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 3);
        for (idx, line) in lines.iter().enumerate() {
            let parsed: SegmentProfile = serde_json::from_str(line).unwrap();
            assert_eq!(parsed.segment_idx, idx);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
