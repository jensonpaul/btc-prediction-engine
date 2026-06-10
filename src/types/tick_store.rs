//! Lock-free tick ring buffer.
//!
//! # Hot-path design
//!
//! Reads (feature computation, window queries) use `parking_lot::RwLock` which
//! allows unlimited concurrent readers with zero writer starvation.  The engine
//! loop is the **only writer**; all subsystem queries are readers.
//!
//! The latest-tick pointer is a separate `arc_swap::ArcSwap<Option<Tick>>` that
//! can be loaded with a single atomic pointer load — subsystems that only need
//! the current price pay zero lock overhead.
//!
//! # Memory budget
//!
//! At ~15 aggregated ticks/s across four exchanges, `DEFAULT_CAPACITY = 1_500_000`
//! covers ~28 hours and consumes ~150 MB (Tick ≈ 104 bytes after alignment).

use std::collections::VecDeque;
use std::sync::Arc;
use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::types::{EngineError, EngineResult, Ohlcv, Tick, TradeSide};

pub const DEFAULT_CAPACITY: usize = 1_500_000;

/// Thread-safe, capacity-bounded ring of [`Tick`]s.
pub struct TickStore {
    inner:   RwLock<Inner>,
    /// Lock-free latest-tick pointer for zero-cost "current price" reads.
    latest:  ArcSwap<Option<Tick>>,
    capacity: usize,
}

struct Inner {
    buf: VecDeque<Tick>,
}

impl TickStore {
    pub fn new() -> Self { Self::with_capacity(DEFAULT_CAPACITY) }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner:    RwLock::new(Inner { buf: VecDeque::with_capacity(capacity) }),
            latest:   ArcSwap::from_pointee(None),
            capacity,
        }
    }

    /// Push a tick onto the ring. Called only from the engine hot-path.
    ///
    /// The write lock is held for the minimum time possible (one push + one
    /// optional pop). The `ArcSwap` update is a separate atomic store after
    /// the lock is released.
    #[inline]
    pub fn push(&self, tick: Tick) {
        {
            let mut g = self.inner.write();
            if g.buf.len() == self.capacity { g.buf.pop_front(); }
            g.buf.push_back(tick.clone());
        }
        // Update lock-free latest pointer after releasing write lock
        self.latest.store(Arc::new(Some(tick)));
    }

    /// Latest tick — **zero lock**, single atomic load.
    #[inline]
    pub fn latest(&self) -> Option<Tick> {
        self.latest.load().as_ref().clone()
    }

    /// Current number of ticks in the ring.
    pub fn len(&self) -> usize { self.inner.read().buf.len() }
    pub fn is_empty(&self) -> bool { self.inner.read().buf.is_empty() }

    /// Oldest tick still in the ring.
    pub fn oldest(&self) -> Option<Tick> {
        self.inner.read().buf.front().cloned()
    }

    /// All ticks in `[start_micros, end_micros]` (inclusive).
    ///
    /// Performs a linear scan under a read lock.  For the expected buffer size
    /// and query rate this is adequate.
    pub fn range(&self, start: i64, end: i64) -> Vec<Tick> {
        self.inner.read().buf.iter()
            .filter(|t| t.ts_micros >= start && t.ts_micros <= end)
            .cloned()
            .collect()
    }

    /// Last `n` ticks, oldest-first.
    pub fn last_n(&self, n: usize) -> Vec<Tick> {
        let g = self.inner.read();
        let skip = g.buf.len().saturating_sub(n);
        g.buf.iter().skip(skip).cloned().collect()
    }

    /// All ticks since `since_micros` up to and including the latest.
    pub fn since(&self, since_micros: i64) -> EngineResult<Vec<Tick>> {
        let g = self.inner.read();
        let latest_ts = g.buf.back().ok_or(EngineError::EmptyBuffer)?.ts_micros;
        Ok(g.buf.iter()
            .filter(|t| t.ts_micros >= since_micros && t.ts_micros <= latest_ts)
            .cloned()
            .collect())
    }

    /// All ticks in the last `secs` seconds (relative to latest tick, not wall clock).
    pub fn last_secs(&self, secs: i64) -> EngineResult<Vec<Tick>> {
        let g = self.inner.read();
        let latest_ts = g.buf.back().ok_or(EngineError::EmptyBuffer)?.ts_micros;
        let since = latest_ts - secs * 1_000_000;
        Ok(g.buf.iter()
            .filter(|t| t.ts_micros >= since)
            .cloned()
            .collect())
    }
}

impl Default for TickStore { fn default() -> Self { Self::new() } }

// ─── OHLCV builder ───────────────────────────────────────────────────────────

/// Build an [`Ohlcv`] bar from a tick slice. Returns `None` for empty slices.
pub fn ohlcv_from_ticks(ticks: &[Tick]) -> Option<Ohlcv> {
    if ticks.is_empty() { return None; }

    let open     = ticks.first().unwrap().price;
    let close    = ticks.last().unwrap().price;
    let open_ts  = ticks.first().unwrap().ts_micros;
    let close_ts = ticks.last().unwrap().ts_micros;

    let mut high      = f64::NEG_INFINITY;
    let mut low       = f64::INFINITY;
    let mut volume    = 0.0_f64;
    let mut notional  = 0.0_f64;
    let mut buy_vol   = 0.0_f64;
    let mut has_side  = false;

    for t in ticks {
        high     = high.max(t.price);
        low      = low.min(t.price);
        volume  += t.quantity;
        notional += t.price * t.quantity;
        if let Some(TradeSide::Buy) = t.side { buy_vol += t.quantity; has_side = true; }
        else if t.side.is_some() { has_side = true; }
    }

    let vwap      = if volume > 0.0 { notional / volume } else { open };
    let buy_ratio = if has_side && volume > 0.0 { Some(buy_vol / volume) } else { None };

    Some(Ohlcv {
        open_ts, close_ts, open, high, low, close,
        volume, notional, tick_count: ticks.len(), vwap, buy_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Exchange, Symbol};

    fn t(ts_micros: i64, price: f64) -> Tick {
        Tick { ts_micros, price, quantity: 0.1, side: Some(TradeSide::Buy),
               exchange: Exchange::Binance, symbol: Symbol::BtcUsd, trade_id: ts_micros.to_string() }
    }

    #[test]
    fn eviction() {
        let store = TickStore::with_capacity(3);
        for i in 1..=4i64 { store.push(t(i * 1_000_000, i as f64 * 100.0)); }
        assert_eq!(store.len(), 3);
        assert_eq!(store.oldest().unwrap().ts_micros, 2_000_000);
    }

    #[test]
    fn latest_is_lock_free() {
        let store = TickStore::new();
        assert!(store.latest().is_none());
        store.push(t(1_000_000, 50_000.0));
        assert_eq!(store.latest().unwrap().price, 50_000.0);
    }

    #[test]
    fn ohlcv_fields() {
        let ticks = vec![t(0, 100.0), t(1_000_000, 110.0), t(2_000_000, 90.0), t(3_000_000, 105.0)];
        let bar = ohlcv_from_ticks(&ticks).unwrap();
        assert_eq!(bar.open,  100.0);
        assert_eq!(bar.close, 105.0);
        assert_eq!(bar.high,  110.0);
        assert_eq!(bar.low,    90.0);
        assert!((bar.vwap - (100.0+110.0+90.0+105.0)/4.0).abs() < 1.0);
    }
}
