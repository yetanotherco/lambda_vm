//! Tracing layer that accumulates per-span wall-clock durations from
//! Plonky3's `tracing` instrumentation. Used by `prove_bench --breakdown`
//! and by the `instruments_breakdown` test.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing_subscriber::Layer;

pub type SpanResults = Arc<Mutex<Vec<(String, f64)>>>;

struct SpanState {
    name: String,
    active_since: Option<Instant>,
    accumulated: Duration,
}

pub struct P3TimingLayer {
    spans: Mutex<HashMap<u64, SpanState>>,
    results: SpanResults,
}

impl P3TimingLayer {
    pub fn new() -> (Self, SpanResults) {
        let results: SpanResults = Arc::new(Mutex::new(Vec::new()));
        let layer = Self {
            spans: Mutex::new(HashMap::new()),
            results: Arc::clone(&results),
        };
        (layer, results)
    }
}

impl<S> Layer<S> for P3TimingLayer
where
    S: tracing::Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        self.spans.lock().unwrap().insert(
            id.into_u64(),
            SpanState {
                name: attrs.metadata().name().to_string(),
                active_since: None,
                accumulated: Duration::ZERO,
            },
        );
    }

    fn on_enter(&self, id: &tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(entry) = self.spans.lock().unwrap().get_mut(&id.into_u64())
            && entry.active_since.is_none()
        {
            entry.active_since = Some(Instant::now());
        }
    }

    fn on_exit(&self, id: &tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(entry) = self.spans.lock().unwrap().get_mut(&id.into_u64())
            && let Some(start) = entry.active_since.take()
        {
            entry.accumulated += start.elapsed();
        }
    }

    fn on_close(&self, id: tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(entry) = self.spans.lock().unwrap().remove(&id.into_u64()) {
            let mut total = entry.accumulated;
            if let Some(start) = entry.active_since {
                total += start.elapsed();
            }
            self.results
                .lock()
                .unwrap()
                .push((entry.name, total.as_secs_f64() * 1000.0));
        }
    }
}
