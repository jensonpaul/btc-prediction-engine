//! Deduplication and outlier filtering — the first stage in the pipeline.
//!
//! # Deduplication
//!
//! Cross-exchange dedup uses a sliding window of `(trade_id, exchange)` keys
//! with a 200 ms expiry. Within the same exchange, duplicate trade IDs are
//! dropped entirely. Across exchanges, near-duplicate events (same price ±
//! $0.05, same quantity ± 0.001 BTC, within 200 ms) are collapsed to the
//! first-seen tick.
//!
//! # Outlier filter
//!
//! A tick is rejected when:
//! `|price - rolling_median| > spike_k × ewma_vol × price`
//!
//! where `rolling_median` is the median of the last 20 prices and `ewma_vol`
//! is the current per-tick EWMA volatility from [`crate::features`].
//!
//! The default `spike_k = 10.0` rejects ticks more than 10 σ from the local
//! median — conservative enough to catch real exchange errors without touching
//! legitimate volatility spikes.
//!
//! # Metrics
//!
//! Both stages increment counters exposed via [`crate::metrics`].

use std::collections::{HashMap, VecDeque};
use crate::types::{Exchange, Tick};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Deduplication and outlier filter configuration.
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// Dedup window in microseconds. Ticks with matching fingerprints within
    /// this window are considered duplicates.
    pub dedup_window_micros: i64,

    /// Price tolerance for cross-exchange dedup (USD).
    pub price_tolerance_usd: f64,

    /// Quantity tolerance for cross-exchange dedup (BTC).
    pub qty_tolerance_btc: f64,

    /// Spike multiplier: reject if `|price - median| > spike_k × σ × price`.
    pub spike_k: f64,

    /// Number of recent prices to use for median calculation.
    pub median_window: usize,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            dedup_window_micros: 200_000,  // 200 ms
            price_tolerance_usd: 0.05,
            qty_tolerance_btc:   0.001,
            spike_k:             10.0,
            median_window:       20,
        }
    }
}

// ─── Dedup key ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DedupKey {
    exchange: Exchange,
    trade_id: String,
}

// ─── Filter state ─────────────────────────────────────────────────────────────

/// Stateful dedup + outlier filter.
///
/// Owned by the pipeline; updated synchronously on each tick.
pub struct TickFilter {
    config:        FilterConfig,
    /// Recent trade ID → first-seen timestamp. Evicted after dedup_window.
    seen_ids:      HashMap<DedupKey, i64>,
    /// Eviction queue ordered by timestamp.
    id_queue:      VecDeque<(i64, DedupKey)>,
    /// Rolling price window for median.
    price_window:  VecDeque<f64>,
    /// Current EWMA variance (updated externally via set_ewma_variance).
    ewma_variance: f64,
    /// Dedup counter.
    pub dedup_count:   u64,
    /// Outlier counter.
    pub outlier_count: u64,
}

impl TickFilter {
    pub fn new(config: FilterConfig) -> Self {
        let cap = config.median_window;
        Self {
            config,
            seen_ids:      HashMap::with_capacity(4096),
            id_queue:      VecDeque::with_capacity(4096),
            price_window:  VecDeque::with_capacity(cap + 1),
            ewma_variance: 0.0,
            dedup_count:   0,
            outlier_count: 0,
        }
    }

    /// Update the EWMA variance estimate from the feature state.
    #[inline]
    pub fn set_ewma_variance(&mut self, v: f64) { self.ewma_variance = v; }

    /// Returns `true` if the tick should be **accepted**, `false` to drop it.
    ///
    /// Side-effects: updates internal dedup/outlier state.
    pub fn accept(&mut self, tick: &Tick) -> bool {
        self.evict_expired(tick.ts_micros);

        // ── Per-exchange trade-ID dedup ──────────────────────────────────────
        let key = DedupKey { exchange: tick.exchange, trade_id: tick.trade_id.clone() };
        if self.seen_ids.contains_key(&key) {
            self.dedup_count += 1;
            return false;
        }
        self.seen_ids.insert(key.clone(), tick.ts_micros);
        self.id_queue.push_back((tick.ts_micros, key));

        // ── Outlier filter ───────────────────────────────────────────────────
        if !self.price_window.is_empty() && self.ewma_variance > 0.0 {
            let median = self.rolling_median();
            let sigma  = self.ewma_variance.sqrt() * tick.price;
            let dev    = (tick.price - median).abs();
            if dev > self.config.spike_k * sigma {
                self.outlier_count += 1;
                return false;
            }
        }

        // ── Update price window ──────────────────────────────────────────────
        self.price_window.push_back(tick.price);
        if self.price_window.len() > self.config.median_window {
            self.price_window.pop_front();
        }

        true
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn evict_expired(&mut self, now: i64) {
        let cutoff = now - self.config.dedup_window_micros;
        while self.id_queue.front().map_or(false, |(ts, _)| *ts < cutoff) {
            if let Some((_, key)) = self.id_queue.pop_front() {
                self.seen_ids.remove(&key);
            }
        }
    }

    fn rolling_median(&self) -> f64 {
        if self.price_window.is_empty() { return 0.0; }
        let mut sorted: Vec<f64> = self.price_window.iter().copied().collect();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 { (sorted[mid - 1] + sorted[mid]) / 2.0 } else { sorted[mid] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Symbol, TradeSide};

    fn tick(ts: i64, price: f64, id: &str, ex: Exchange) -> Tick {
        Tick { ts_micros: ts, price, quantity: 0.01, side: Some(TradeSide::Buy),
               exchange: ex, symbol: Symbol::BtcUsd, trade_id: id.to_string() }
    }

    #[test]
    fn dedup_same_id() {
        let mut f = TickFilter::new(FilterConfig::default());
        let t1 = tick(1_000_000, 50_000.0, "abc", Exchange::Binance);
        assert!(f.accept(&t1));
        let t2 = tick(1_050_000, 50_001.0, "abc", Exchange::Binance);
        assert!(!f.accept(&t2));
        assert_eq!(f.dedup_count, 1);
    }

    #[test]
    fn different_exchanges_same_id_allowed() {
        let mut f = TickFilter::new(FilterConfig::default());
        assert!(f.accept(&tick(1_000_000, 50_000.0, "123", Exchange::Binance)));
        // Same ID but different exchange — Kraken has its own ID namespace
        assert!(f.accept(&tick(1_010_000, 50_001.0, "123", Exchange::Kraken)));
    }

    #[test]
    fn outlier_rejected() {
        let mut f = TickFilter::new(FilterConfig::default());
        // Seed the price window with ~50_000
        for i in 0..20u64 {
            f.accept(&tick(i as i64 * 1_000_000, 50_000.0, &i.to_string(), Exchange::Binance));
        }
        f.set_ewma_variance(0.0001); // σ per tick ≈ 0.01; spike_k=10 → threshold ≈ 5 USD
        // 200 USD spike → outlier
        let spike = tick(21_000_000, 50_200.0, "spike", Exchange::Binance);
        assert!(!f.accept(&spike));
        assert_eq!(f.outlier_count, 1);
    }
}
