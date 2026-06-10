//! Subsystem query API.
//!
//! [`QueryEngine`] wraps a [`PredictionEngine`] and exposes typed, ergonomic
//! methods for all common access patterns.  No inference runs at query time —
//! everything is assembled from pre-computed stores.
//!
//! # Latency characteristics
//!
//! | Method | Cost |
//! |---|---|
//! | `latest()`, `latest_price()`, `fused()` | **Zero lock** — single atomic load |
//! | `trend(scale)`, `forecast(step)` | Zero lock — reads from ArcSwap snapshot |
//! | `window()`, `sub_windows()` | Read lock on TickStore + PredStore |
//! | `aggregate_trend()` | Read lock on PredStore, O(n) where n = snaps in range |
//!
//! # Example
//!
//! ```rust,no_run
//! use btc_prediction_engine::prelude::*;
//!
//! async fn my_subsystem(q: QueryEngine) {
//!     // Polling — cheap zero-lock reads
//!     loop {
//!         if let Some(snap) = q.latest() {
//!             println!("{:?} @ ${:.2}", snap.fused_direction, snap.price);
//!         }
//!         tokio::time::sleep(std::time::Duration::from_secs(1)).await;
//!     }
//! }
//!
//! async fn risk_check(q: QueryEngine) {
//!     // 5-min OHLCV for position sizing
//!     if let Ok(win) = q.window_5m() {
//!         if let Some(ohlcv) = win.ohlcv {
//!             let rtn = ohlcv.return_pct();
//!             println!("5m return: {:+.4}%", rtn * 100.0);
//!         }
//!     }
//! }
//! ```

use chrono::Utc;

use crate::engine::PredictionEngine;
use crate::types::{
    EngineError, EngineResult, Ohlcv, PredictionSnapshot, ShortTermForecast,
    TimeScale, TrendDirection, TrendSignal, WindowProjection,
};
use crate::types::tick_store::ohlcv_from_ticks;

/// Subsystem-facing query interface. `Clone + Send + Sync`.
#[derive(Clone)]
pub struct QueryEngine {
    engine: PredictionEngine,
}

impl QueryEngine {
    pub fn new(engine: PredictionEngine) -> Self { Self { engine } }

    // ── Zero-lock reads ───────────────────────────────────────────────────────

    /// Latest snapshot — **zero lock**.
    #[inline]
    pub fn latest(&self) -> Option<PredictionSnapshot> {
        self.engine.latest_snapshot()
    }

    /// Latest BTC/USD price — **zero lock**.
    #[inline]
    pub fn latest_price(&self) -> Option<f64> {
        self.engine.latest_price()
    }

    /// Fused direction + confidence — **zero lock**.
    #[inline]
    pub fn fused(&self) -> Option<(TrendDirection, f64)> {
        self.latest().map(|s| (s.fused_direction, s.fused_confidence))
    }

    // ── Trend queries — zero lock ──────────────────────────────────────────

    pub fn trend(&self, scale: TimeScale) -> Option<TrendSignal> {
        self.latest().map(|s| match scale {
            TimeScale::Micro  => s.micro,
            TimeScale::Short  => s.short,
            TimeScale::Medium => s.medium,
            TimeScale::Broad  => s.broad,
        })
    }

    pub fn micro_trend(&self)  -> Option<TrendSignal> { self.trend(TimeScale::Micro)  }
    pub fn short_trend(&self)  -> Option<TrendSignal> { self.trend(TimeScale::Short)  }
    pub fn medium_trend(&self) -> Option<TrendSignal> { self.trend(TimeScale::Medium) }
    pub fn broad_trend(&self)  -> Option<TrendSignal> { self.trend(TimeScale::Broad)  }

    // ── Forecast queries — zero lock ─────────────────────────────────────────

    /// Short-term forecast for `step_secs`. Returns `None` if this step size
    /// is not in [`crate::pipeline::PipelineConfig::forecast_steps`].
    pub fn forecast(&self, step_secs: u32) -> Option<ShortTermForecast> {
        self.latest()?.forecasts.remove(&step_secs)
    }

    /// 5-second forecast (6 steps = 30 s horizon).
    pub fn forecast_5s(&self)  -> Option<ShortTermForecast> { self.forecast(5)  }
    /// 30-second forecast (10 steps = 5 min horizon).
    pub fn forecast_30s(&self) -> Option<ShortTermForecast> { self.forecast(30) }

    // ── Point-in-time snapshot ────────────────────────────────────────────────

    /// Snapshot closest to and ≤ `ts_micros`. Acquires a read lock on
    /// PredictionStore.
    pub fn at(&self, ts_micros: i64) -> Option<PredictionSnapshot> {
        self.engine.pred_store().at_or_before(ts_micros)
    }

    // ── Window projection queries ─────────────────────────────────────────────

    /// Project the engine state onto `[start_micros, end_micros]`.
    ///
    /// Assembles an OHLCV bar from the tick ring and attaches the prediction
    /// snapshot as of `end_micros`. No inference is run.
    pub fn window(&self, start_micros: i64, end_micros: i64) -> EngineResult<WindowProjection> {
        if start_micros > end_micros {
            return Err(EngineError::NoDataInRange { start: start_micros, end: end_micros });
        }
        let ticks      = self.engine.tick_store().range(start_micros, end_micros);
        let tick_count = ticks.len();
        let ohlcv      = ohlcv_from_ticks(&ticks);
        let prediction = self.engine.pred_store().at_or_before(end_micros);
        Ok(WindowProjection { window_start: start_micros, window_end: end_micros, ohlcv, prediction, tick_count })
    }

    /// Last `duration_secs` seconds ending at **wall clock now**.
    pub fn window_ending_now(&self, duration_secs: i64) -> EngineResult<WindowProjection> {
        let end = Utc::now().timestamp_micros();
        self.window(end - duration_secs * 1_000_000, end)
    }

    /// Last `duration_secs` seconds ending at the **latest tick timestamp**.
    ///
    /// Preferred over `window_ending_now` for replayed or delayed streams.
    pub fn window_ending_latest(&self, duration_secs: i64) -> EngineResult<WindowProjection> {
        let end = self.engine.tick_store().latest()
            .ok_or(EngineError::EmptyBuffer)?.ts_micros;
        self.window(end - duration_secs * 1_000_000, end)
    }

    // ── Named windows ────────────────────────────────────────────────────────

    pub fn window_30s(&self) -> EngineResult<WindowProjection> { self.window_ending_latest(30)     }
    pub fn window_5m(&self)  -> EngineResult<WindowProjection> { self.window_ending_latest(300)    }
    pub fn window_1h(&self)  -> EngineResult<WindowProjection> { self.window_ending_latest(3_600)  }
    pub fn window_24h(&self) -> EngineResult<WindowProjection> { self.window_ending_latest(86_400) }

    // ── Sub-window decomposition ──────────────────────────────────────────────

    /// Decompose `[start_micros, end_micros]` into equal sub-windows of
    /// `step_secs` each.
    ///
    /// Returns one [`WindowProjection`] per sub-window, chronological order.
    /// Sub-windows with no ticks have `ohlcv: None`.
    pub fn sub_windows(
        &self,
        start_micros: i64,
        end_micros:   i64,
        step_secs:    i64,
    ) -> EngineResult<Vec<WindowProjection>> {
        if step_secs <= 0 {
            return Err(EngineError::Feature("step_secs must be > 0".into()));
        }
        let step   = step_secs * 1_000_000;
        let mut ws = Vec::new();
        let mut t  = start_micros;
        while t < end_micros {
            ws.push(self.window(t, (t + step).min(end_micros))?);
            t += step;
        }
        Ok(ws)
    }

    /// 5-min window decomposed into 5-second sub-windows (60 slices).
    pub fn sub_windows_5s_in_5m(&self) -> EngineResult<Vec<WindowProjection>> {
        let end   = self.engine.tick_store().latest().ok_or(EngineError::EmptyBuffer)?.ts_micros;
        self.sub_windows(end - 300 * 1_000_000, end, 5)
    }

    /// 5-min window decomposed into 30-second sub-windows (10 slices).
    pub fn sub_windows_30s_in_5m(&self) -> EngineResult<Vec<WindowProjection>> {
        let end   = self.engine.tick_store().latest().ok_or(EngineError::EmptyBuffer)?.ts_micros;
        self.sub_windows(end - 300 * 1_000_000, end, 30)
    }

    // ── Aggregate trend ───────────────────────────────────────────────────────

    /// Weighted-vote aggregate trend over a time range.
    pub fn aggregate_trend(&self, start: i64, end: i64) -> Option<(TrendDirection, f64)> {
        self.engine.pred_store().aggregate_trend(start, end)
    }

    /// Aggregate trend for the last `duration_secs` seconds.
    pub fn aggregate_trend_last(&self, secs: i64) -> Option<(TrendDirection, f64)> {
        let end = self.engine.tick_store().latest()?.ts_micros;
        self.aggregate_trend(end - secs * 1_000_000, end)
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    pub fn tick_count(&self)       -> usize { self.engine.tick_count() }
    pub fn prediction_count(&self) -> usize { self.engine.prediction_count() }
    pub fn connected_feeds(&self)  -> u32   { self.engine.connected_feeds() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::types::{Exchange, Symbol, TradeSide};

    async fn seeded(n: usize) -> QueryEngine {
        let (engine, _) = PredictionEngine::start(EngineConfig::default()).await;
        let base = 1_700_000_000_000_000i64;
        for i in 0..n {
            let tick = Tick {
                ts_micros: base + i as i64 * 1_000_000,
                price:     50_000.0 + i as f64 * 5.0,
                quantity:  0.01,
                side:      Some(TradeSide::Buy),
                exchange:  Exchange::Binance,
                symbol:    Symbol::BtcUsd,
                trade_id:  i.to_string(),
            };
            engine.inject_tick(tick);
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        QueryEngine::new(engine)
    }

    use crate::types::Tick;

    #[tokio::test]
    async fn latest_after_inject() {
        let q = seeded(50).await;
        assert!(q.latest().is_some());
        assert!(q.latest_price().unwrap() > 0.0);
    }

    #[tokio::test]
    async fn window_5m_covers_injected() {
        let q = seeded(100).await;
        let w = q.window_5m().unwrap();
        assert!(w.tick_count > 0);
        assert!(w.ohlcv.is_some());
    }

    #[tokio::test]
    async fn sub_windows_count() {
        let q    = seeded(120).await;
        let base = 1_700_000_000_000_000i64;
        // 120 s range, 10 s steps → 12 sub-windows
        let subs = q.sub_windows(base, base + 120 * 1_000_000, 10).unwrap();
        assert_eq!(subs.len(), 12);
    }

    #[tokio::test]
    async fn fused_returns_direction() {
        let q = seeded(50).await;
        let (dir, conf) = q.fused().unwrap();
        assert!(conf >= 0.0 && conf <= 1.0);
        let _ = dir; // just ensure it's populated
    }
}
