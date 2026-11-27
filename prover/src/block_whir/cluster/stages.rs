//! Host clocks for a range attempt. Reading a clock never waits on a card.

use std::{
    fmt::{Display, Write},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Default)]
pub struct Clock(Option<AttemptClock>);

#[derive(Clone, Copy)]
pub(crate) struct Stamp {
    at: Instant,
    unix_ms: u128,
}

#[derive(Clone, Copy)]
struct AttemptClock {
    start: Instant,
    id: [u8; 16],
    worker: Option<usize>,
}

impl Clock {
    pub(crate) fn new(start: Instant, id: [u8; 16], worker: Option<usize>) -> Self {
        Self(Some(AttemptClock { start, id, worker }))
    }

    pub(crate) fn reference() -> Self {
        let now = wall();
        let id = (now.as_nanos() << 32 | u128::from(std::process::id())).to_be_bytes();
        let clock = Self::new(Instant::now(), id, None);
        clock.mark("attempt-start");
        clock
    }

    pub(crate) fn for_worker(self, id: usize) -> Self {
        Self(self.0.map(|clock| AttemptClock {
            worker: Some(id),
            ..clock
        }))
    }

    pub fn mark(self, stage: impl Display) {
        if let Some(stamp) = self.sample() {
            self.mark_at(stage, stamp);
        }
    }

    pub(crate) fn sample(self) -> Option<Stamp> {
        self.0.map(|_| Stamp {
            at: Instant::now(),
            unix_ms: wall().as_millis(),
        })
    }

    pub(crate) fn mark_at(self, stage: impl Display, stamp: Stamp) {
        self.log(stage, stamp, format_args!(""));
    }

    pub(crate) fn opened(self, group: usize, start: Instant) {
        if let Some(stamp) = self.sample() {
            self.opened_at(group, stamp, (stamp.at - start).as_secs_f64());
        }
    }

    pub(crate) fn opened_at(self, group: usize, stamp: Stamp, seconds: f64) {
        self.log(
            format_args!("base-opening-group-{group}"),
            stamp,
            format_args!(" seconds={seconds:.6}"),
        );
    }

    fn log(self, stage: impl Display, stamp: Stamp, extra: impl Display) {
        let Some(clock) = self.0 else { return };
        let Stamp { at, unix_ms } = stamp;
        let mut id = String::with_capacity(32);
        for byte in clock.id {
            write!(id, "{byte:02x}").unwrap();
        }
        let t = (at - clock.start).as_secs_f64();
        if let Some(worker) = clock.worker {
            eprintln!("CLUSTER STAGE W{worker} {id} {stage} t={t:.6} unix_ms={unix_ms}{extra}");
        } else {
            eprintln!("CLUSTER STAGE C {id} {stage} t={t:.6} unix_ms={unix_ms}{extra}");
        }
    }
}

fn wall() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}
