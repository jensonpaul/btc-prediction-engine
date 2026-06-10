//! Lock-free prediction ring.
//!
//! Latest snapshot read = single atomic `ArcSwap::load` — zero lock cost.
//! All range queries acquire a read lock (many concurrent readers allowed).

use std::collections::VecDeque;
use std::sync::Arc;
use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::types::{PredictionSnapshot, TrendDirection};

pub const DEFAULT_PRED_CAPACITY: usize = 100_000;

pub struct PredictionStore {
    inner:    RwLock<VecDeque<PredictionSnapshot>>,
    latest:   ArcSwap<Option<PredictionSnapshot>>,
    capacity: usize,
}

impl PredictionStore {
    pub fn new() -> Self { Self::with_capacity(DEFAULT_PRED_CAPACITY) }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner:    RwLock::new(VecDeque::with_capacity(capacity)),
            latest:   ArcSwap::from_pointee(None),
            capacity,
        }
    }

    pub fn push(&self, snap: PredictionSnapshot) {
        {
            let mut g = self.inner.write();
            if g.len() == self.capacity { g.pop_front(); }
            g.push_back(snap.clone());
        }
        self.latest.store(Arc::new(Some(snap)));
    }

    /// Latest snapshot — **zero lock**, single atomic load.
    #[inline]
    pub fn latest(&self) -> Option<PredictionSnapshot> {
        self.latest.load().as_ref().clone()
    }

    /// Snapshot with `snapshot_at` closest to and ≤ `ts_micros`.
    pub fn at_or_before(&self, ts_micros: i64) -> Option<PredictionSnapshot> {
        let g = self.inner.read();
        let v: Vec<_> = g.iter().collect();
        match v.binary_search_by_key(&ts_micros, |s| s.snapshot_at) {
            Ok(i)  => Some(v[i].clone()),
            Err(0) => None,
            Err(i) => Some(v[i - 1].clone()),
        }
    }

    /// All snapshots in `[start, end]`.
    pub fn range(&self, start: i64, end: i64) -> Vec<PredictionSnapshot> {
        self.inner.read().iter()
            .filter(|s| s.snapshot_at >= start && s.snapshot_at <= end)
            .cloned()
            .collect()
    }

    /// Weighted-vote aggregate trend over a time range.
    pub fn aggregate_trend(&self, start: i64, end: i64) -> Option<(TrendDirection, f64)> {
        let snaps = self.range(start, end);
        if snaps.is_empty() { return None; }
        let (mut bull, mut bear, mut side) = (0.0_f64, 0.0_f64, 0.0_f64);
        for s in &snaps {
            match s.fused_direction {
                TrendDirection::Bullish  => bull += s.fused_confidence,
                TrendDirection::Bearish  => bear += s.fused_confidence,
                TrendDirection::Sideways => side += s.fused_confidence,
            }
        }
        let total = bull + bear + side;
        if total == 0.0 { return Some((TrendDirection::Sideways, 0.0)); }
        let (dir, w) = if bull >= bear && bull >= side { (TrendDirection::Bullish,  bull) }
                       else if bear >= side            { (TrendDirection::Bearish,  bear) }
                       else                            { (TrendDirection::Sideways, side) };
        Some((dir, w / total))
    }

    pub fn len(&self) -> usize { self.inner.read().len() }
    pub fn is_empty(&self) -> bool { self.inner.read().is_empty() }
}

impl Default for PredictionStore { fn default() -> Self { Self::new() } }
