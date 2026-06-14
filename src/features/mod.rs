//! Incremental feature engineering — all O(1) per tick.
//!
//! # Design principles
//!
//! Every feature is expressed on a **real-time horizon**, not a tick-count
//! window.  This keeps features economically consistent across market regimes:
//! the same 30-second momentum means the same thing whether the market is
//! printing 5 ticks/s or 50 ticks/s.
//!
//! All features are **scale-normalised** where possible (percentage, z-score,
//! or ratio), so they generalise across BTC price regimes from $20 k to $100 k+.
//!
//! # Feature hierarchy (time horizons)
//!
//! ```text
//! 5 s   — microstructure / order-flow momentum
//! 30 s  — short-term signal
//! 300 s — prediction horizon (matches label window)
//! 1800 s — regime / normalisation reference
//! ```
//!
//! # Feature inventory
//!
//! | Feature              | Range        | Description                               |
//! |----------------------|--------------|-------------------------------------------|
//! | `return_5s`          | fraction     | Log-return over last 5 s                  |
//! | `return_30s`         | fraction     | Log-return over last 30 s                 |
//! | `return_300s`        | fraction     | Log-return over last 300 s                |
//! | `vol_30s`            | fraction     | Realised vol (std of log-returns), 30 s   |
//! | `vol_300s`           | fraction     | Realised vol, 300 s                       |
//! | `vol_1800s`          | fraction     | Realised vol, 1800 s (regime reference)   |
//! | `vol_ratio`          | ratio        | vol_30s / vol_1800s — expansion detector  |
//! | `ofi_5s`             | [−1, 1]      | Order-flow imbalance, 5 s                 |
//! | `ofi_30s`            | [−1, 1]      | Order-flow imbalance, 30 s                |
//! | `ofi_300s`           | [−1, 1]      | Order-flow imbalance, 300 s               |
//! | `ofi_delta_30s`      | [−1, 1]      | ofi_5s − ofi_30s (rate-of-change proxy)   |
//! | `buy_ratio_30s`      | [0, 1]       | buy_vol / total_vol over 30 s             |
//! | `buy_ratio_300s`     | [0, 1]       | buy_vol / total_vol over 300 s            |
//! | `vwap_dev_30s`       | fraction     | (price − VWAP_30s) / vol_1800s (z-score) |
//! | `vwap_dev_300s`      | fraction     | (price − VWAP_300s) / vol_1800s           |
//! | `volume_ratio`       | ratio        | volume_30s / volume_300s                  |
//! | `tick_velocity`      | ticks/s      | Rolling tick rate, 30 s window            |
//! | `spread_pct`         | fraction     | Cross-exchange (max−min)/mid              |
//! | `book_imbalance_5`   | [−1, 1]      | Top-5 bid/ask volume imbalance            |
//! | `book_imbalance_full`| [−1, 1]      | Full-depth bid/ask volume imbalance       |
//! | `book_spread_pct`    | fraction     | Best bid-ask spread / mid                 |
//! | `book_pressure`      | [−1, 1]      | Micro-price deviation from mid            |
//! | `trend_strength`     | ≥ 0          | abs(return_1800s) / vol_1800s             |
//! | `vol_regime`         | ratio        | vol_300s / vol_1800s                      |
//! | `activity_regime`    | ratio        | tick_rate_30s / tick_rate_1800s           |
//! | `zreturn_30s`        | z-score      | return_30s / vol_1800s                    |
//! | `zreturn_300s`       | z-score      | return_300s / vol_1800s                   |

use std::collections::{HashMap, VecDeque};
use crate::types::{Exchange, Tick, TradeSide};
use crate::price_fusion::FusedTick;

// ─── Time-windowed log-return tracker ────────────────────────────────────────

/// Tracks the price `horizon_micros` ago and delivers a log-return on demand.
///
/// Uses a ring buffer keyed by timestamp so the "oldest price in the window"
/// is always the entry whose timestamp is closest to `now − horizon`.
pub struct WindowedReturn {
    horizon_micros: i64,
    /// (ts_micros, log_price) — oldest first.
    buf: VecDeque<(i64, f64)>,
}

impl WindowedReturn {
    pub fn new(horizon_secs: i64) -> Self {
        Self {
            horizon_micros: horizon_secs * 1_000_000,
            buf: VecDeque::new(),
        }
    }

    pub fn update(&mut self, ts: i64, price: f64) {
        if price > 0.0 {
            self.buf.push_back((ts, price.ln()));
        }
        // Evict entries older than 2× horizon so we keep the one *just* outside
        // the window as the reference price (closest to `now - horizon`).
        let cutoff = ts - self.horizon_micros * 2;
        while self.buf.len() > 1 {
            let second_ts = self.buf[1].0;
            if second_ts <= ts - self.horizon_micros {
                // The second entry is still within or at the horizon boundary —
                // pop the first so the second becomes the new "oldest" candidate.
                self.buf.pop_front();
            } else {
                break;
            }
        }
        // Also evict anything truly ancient (> 2× horizon old).
        while self.buf.front().map_or(false, |e| e.0 < cutoff) {
            if self.buf.len() > 1 { self.buf.pop_front(); } else { break; }
        }
    }

    /// Log-return from the reference price (≈ `now − horizon`) to `log_price_now`.
    /// Returns `None` until enough history has accumulated.
    pub fn value(&self, log_price_now: f64) -> Option<f64> {
        let (oldest_ts, oldest_lp) = *self.buf.front()?;
        let (newest_ts, _)         = *self.buf.back()?;
        // Require that the oldest entry is at least 50% of the horizon old,
        // so we don't emit near-zero returns while the window is filling up.
        if newest_ts - oldest_ts < self.horizon_micros / 2 {
            return None;
        }
        Some(log_price_now - oldest_lp)
    }
}

// ─── Realised volatility (time-windowed, O(1)) ───────────────────────────────

/// Rolling realised volatility: sample std-dev of log-returns in a time window.
///
/// Maintains running `sum` and `sum_sq` so `value()` is O(1).
pub struct RealisedVol {
    window_micros: i64,
    /// (ts_micros, log_return)
    entries: VecDeque<(i64, f64)>,
    prev:    Option<(i64, f64)>,
    sum:     f64,
    sum_sq:  f64,
}

impl RealisedVol {
    pub fn new(window_secs: i64) -> Self {
        Self {
            window_micros: window_secs * 1_000_000,
            entries: VecDeque::new(),
            prev:    None,
            sum:     0.0,
            sum_sq:  0.0,
        }
    }

    #[inline]
    fn push_return(&mut self, ts: i64, r: f64) {
        self.sum    += r;
        self.sum_sq += r * r;
        self.entries.push_back((ts, r));
    }

    #[inline]
    fn evict_stale(&mut self, cutoff: i64) {
        while self.entries.front().map_or(false, |e| e.0 < cutoff) {
            let (_, r) = self.entries.pop_front().unwrap();
            self.sum    -= r;
            self.sum_sq -= r * r;
        }
    }

    pub fn update_fused(&mut self, ts: i64, price: f64) {
        if let Some((_, pp)) = self.prev {
            if pp > 0.0 && price > 0.0 {
                let r = (price / pp).ln();
                self.push_return(ts, r);
                self.evict_stale(ts - self.window_micros);
            }
        }
        self.prev = Some((ts, price));
    }

    /// Sample standard deviation of log-returns — O(1).
    #[inline]
    pub fn value(&self) -> Option<f64> {
        let n = self.entries.len();
        if n < 3 { return None; }
        let nf = n as f64;
        let variance = (self.sum_sq - self.sum * self.sum / nf) / (nf - 1.0);
        Some(variance.max(0.0).sqrt())
    }
}

// ─── Rolling VWAP (time-windowed) ────────────────────────────────────────────

/// Rolling VWAP over a fixed time window.
///
/// Stores (ts, price×volume, volume) entries and evicts stale ones on each
/// update, maintaining running `pv_sum` and `vol_sum` for O(1) reads.
pub struct RollingVwap {
    window_micros: i64,
    /// (ts_micros, pv, vol)
    entries: VecDeque<(i64, f64, f64)>,
    pv_sum:  f64,
    vol_sum: f64,
}

impl RollingVwap {
    pub fn new(window_secs: i64) -> Self {
        Self {
            window_micros: window_secs * 1_000_000,
            entries: VecDeque::new(),
            pv_sum:  0.0,
            vol_sum: 0.0,
        }
    }

    pub fn update(&mut self, ts: i64, price: f64, volume: f64) {
        let pv = price * volume;
        self.pv_sum  += pv;
        self.vol_sum += volume;
        self.entries.push_back((ts, pv, volume));

        let cutoff = ts - self.window_micros;
        while self.entries.front().map_or(false, |e| e.0 < cutoff) {
            let (_, old_pv, old_vol) = self.entries.pop_front().unwrap();
            self.pv_sum  -= old_pv;
            self.vol_sum -= old_vol;
        }
    }

    #[inline]
    pub fn vwap(&self) -> Option<f64> {
        if self.vol_sum > 0.0 { Some(self.pv_sum / self.vol_sum) } else { None }
    }

    /// VWAP deviation normalised by an external volatility estimate.
    /// Returns (price - vwap) / (vol_ref * price) — a dimensionless z-score.
    #[inline]
    pub fn deviation_z(&self, price: f64, vol_ref: f64) -> Option<f64> {
        let vwap = self.vwap()?;
        if vwap == 0.0 || vol_ref == 0.0 { return None; }
        // Deviation as a fraction of price, then z-scored.
        Some((price - vwap) / vwap / vol_ref)
    }
}

// ─── Order flow imbalance (time-windowed, O(1)) ───────────────────────────────

/// Rolling OFI in a real-time window with O(1) `value()`.
pub struct OrderFlowImbalance {
    window_micros: i64,
    /// (ts_micros, buy_qty, sell_qty)
    entries:  VecDeque<(i64, f64, f64)>,
    buy_sum:  f64,
    sell_sum: f64,
}

impl OrderFlowImbalance {
    pub fn new(window_secs: i64) -> Self {
        Self {
            window_micros: window_secs * 1_000_000,
            entries:  VecDeque::new(),
            buy_sum:  0.0,
            sell_sum: 0.0,
        }
    }

    #[inline]
    fn evict_stale(&mut self, cutoff: i64) {
        while self.entries.front().map_or(false, |e| e.0 < cutoff) {
            let (_, bq, sq) = self.entries.pop_front().unwrap();
            self.buy_sum  -= bq;
            self.sell_sum -= sq;
        }
    }

    pub fn update(&mut self, tick: &Tick) {
        let (bq, sq) = match tick.side {
            Some(TradeSide::Buy)  => (tick.quantity, 0.0),
            Some(TradeSide::Sell) => (0.0, tick.quantity),
            None                  => (0.0, 0.0),
        };
        self.buy_sum  += bq;
        self.sell_sum += sq;
        self.entries.push_back((tick.ts_micros, bq, sq));
        self.evict_stale(tick.ts_micros - self.window_micros);
    }

    pub fn update_fused(&mut self, ts: i64, buy_vol: f64, sell_vol: f64) {
        self.buy_sum  += buy_vol;
        self.sell_sum += sell_vol;
        self.entries.push_back((ts, buy_vol, sell_vol));
        self.evict_stale(ts - self.window_micros);
    }

    /// (buy − sell) / (buy + sell) ∈ [−1, 1] — O(1).
    #[inline]
    pub fn value(&self) -> f64 {
        let total = self.buy_sum + self.sell_sum;
        if total == 0.0 { 0.0 } else { (self.buy_sum - self.sell_sum) / total }
    }

    /// buy / (buy + sell) ∈ [0, 1] — O(1).
    #[inline]
    pub fn buy_ratio(&self) -> f64 {
        let total = self.buy_sum + self.sell_sum;
        if total == 0.0 { 0.5 } else { self.buy_sum / total }
    }

    /// Total volume observed in the window.
    #[inline]
    pub fn total_volume(&self) -> f64 {
        self.buy_sum + self.sell_sum
    }
}

// ─── Tick velocity ────────────────────────────────────────────────────────────

/// Rolling tick rate (events per second) in a real-time window.
pub struct TickVelocity {
    window_micros: i64,
    buf: VecDeque<i64>,
}

impl TickVelocity {
    pub fn new(window_secs: i64) -> Self {
        Self { window_micros: window_secs * 1_000_000, buf: VecDeque::new() }
    }

    pub fn update(&mut self, ts: i64) {
        self.buf.push_back(ts);
        let cutoff = ts - self.window_micros;
        while self.buf.front().map_or(false, |&t| t < cutoff) { self.buf.pop_front(); }
    }

    pub fn rate(&self) -> f64 {
        let n = self.buf.len();
        if n < 2 { return 0.0; }
        let span = (self.buf.back().unwrap() - self.buf.front().unwrap()) as f64 / 1_000_000.0;
        if span == 0.0 { 0.0 } else { (n as f64 - 1.0) / span }
    }
}

/// Maintains two tick-velocity windows to produce an activity-regime ratio.
pub struct ActivityRegime {
    fast:  TickVelocity,
    slow:  TickVelocity,
}

impl ActivityRegime {
    pub fn new() -> Self {
        Self {
            fast: TickVelocity::new(30),
            slow: TickVelocity::new(1800),
        }
    }

    pub fn update(&mut self, ts: i64) {
        self.fast.update(ts);
        self.slow.update(ts);
    }

    /// tick_rate_30s / tick_rate_1800s.  Values > 1 indicate a burst of activity.
    pub fn ratio(&self) -> f64 {
        let slow = self.slow.rate();
        if slow == 0.0 { 1.0 } else { self.fast.rate() / slow }
    }

    pub fn fast_rate(&self) -> f64 { self.fast.rate() }
}

// ─── Book imbalance tracker ───────────────────────────────────────────────────

/// Tracks the latest book imbalances (top-5 and full) per exchange and
/// fuses them into single cross-exchange estimates.
pub struct BookImbalanceTracker {
    /// (imb_top5, imb_full, received_at_micros) per exchange.
    last: HashMap<Exchange, (f64, f64, i64)>,
}

const BOOK_STALE_MICROS: i64 = 2_000_000; // 2 s

impl BookImbalanceTracker {
    pub fn new() -> Self { Self { last: HashMap::new() } }

    /// Record top-5 and full-depth imbalances from one exchange snapshot.
    pub fn update(&mut self, exchange: Exchange, imb_top5: f64, imb_full: f64, ts_micros: i64) {
        self.last.insert(exchange, (imb_top5, imb_full, ts_micros));
    }

    /// Fused top-5 imbalance across all fresh snapshots.
    pub fn fused_top5(&self, now_micros: i64) -> Option<f64> {
        let (mut total, mut count) = (0.0_f64, 0u32);
        for &(imb5, _, ts) in self.last.values() {
            if now_micros - ts <= BOOK_STALE_MICROS { total += imb5; count += 1; }
        }
        if count == 0 { None } else { Some(total / count as f64) }
    }

    /// Fused full-depth imbalance across all fresh snapshots.
    pub fn fused_full(&self, now_micros: i64) -> Option<f64> {
        let (mut total, mut count) = (0.0_f64, 0u32);
        for &(_, imb_full, ts) in self.last.values() {
            if now_micros - ts <= BOOK_STALE_MICROS { total += imb_full; count += 1; }
        }
        if count == 0 { None } else { Some(total / count as f64) }
    }
}

// ─── Book spread / pressure tracker ──────────────────────────────────────────

/// Tracks per-exchange book spread (in USD) and micro-price deviation.
///
/// Produces percentage spread and a normalised "pressure" metric that
/// captures micro-price vs. mid-price deviations — a cheap proxy for
/// short-term order-book skew.
pub struct BookPressureTracker {
    /// (spread_usd, microprice_dev, mid_price, ts_micros)
    last: HashMap<Exchange, (f64, f64, f64, i64)>,
}

impl BookPressureTracker {
    pub fn new() -> Self { Self { last: HashMap::new() } }

    /// Record a snapshot's spread and micro-price deviation.
    ///
    /// `microprice_dev = (microprice − mid) / mid` — positive when bids dominate.
    pub fn update(
        &mut self, exchange: Exchange,
        spread_usd: f64, microprice_dev: f64, mid: f64, ts: i64,
    ) {
        self.last.insert(exchange, (spread_usd, microprice_dev, mid, ts));
    }

    /// Volume-averaged percentage spread across fresh snapshots.
    pub fn spread_pct(&self, now_micros: i64) -> Option<f64> {
        let (mut num, mut count) = (0.0_f64, 0u32);
        for &(spread_usd, _, mid, ts) in self.last.values() {
            if now_micros - ts <= BOOK_STALE_MICROS && mid > 0.0 {
                num += spread_usd / mid;
                count += 1;
            }
        }
        if count == 0 { None } else { Some(num / count as f64) }
    }

    /// Average micro-price deviation across fresh snapshots.
    pub fn pressure(&self, now_micros: i64) -> Option<f64> {
        let (mut total, mut count) = (0.0_f64, 0u32);
        for &(_, mp_dev, _, ts) in self.last.values() {
            if now_micros - ts <= BOOK_STALE_MICROS { total += mp_dev; count += 1; }
        }
        if count == 0 { None } else { Some(total / count as f64) }
    }
}

// ─── Composite feature vector ─────────────────────────────────────────────────

/// All features at one fused-tick instant.
///
/// `None` means "accumulator not yet seeded" — callers should fill with a
/// sensible default (e.g. 0.0 for signed features, 0.001 for volatility).
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub ts_micros: i64,
    pub price:     f64,

    // ── Returns (time-horizon aligned) ───────────────────────────────────────
    pub return_5s:   Option<f64>,
    pub return_30s:  Option<f64>,
    pub return_300s: Option<f64>,

    // ── Volatility ────────────────────────────────────────────────────────────
    pub vol_30s:    Option<f64>,
    pub vol_300s:   Option<f64>,
    pub vol_1800s:  Option<f64>,
    /// vol_30s / vol_1800s — > 1 means volatility expansion.
    pub vol_ratio:  Option<f64>,

    // ── Order flow imbalance ──────────────────────────────────────────────────
    pub ofi_5s:       f64,
    pub ofi_30s:      f64,
    pub ofi_300s:     f64,
    /// ofi_5s − ofi_30s: positive when short-term flow is more aggressive than baseline.
    pub ofi_delta_30s: f64,
    pub buy_ratio_30s:  f64,
    pub buy_ratio_300s: f64,

    // ── VWAP deviation (z-scored by vol_1800s) ───────────────────────────────
    pub vwap_dev_30s:  Option<f64>,
    pub vwap_dev_300s: Option<f64>,

    // ── Volume ────────────────────────────────────────────────────────────────
    /// volume_30s / volume_300s — > 1 indicates unusually active recent window.
    pub volume_ratio: Option<f64>,

    // ── Tick activity ─────────────────────────────────────────────────────────
    pub tick_velocity:    f64,
    pub activity_regime:  f64,

    // ── Cross-exchange spread (percentage) ───────────────────────────────────
    pub spread_pct: f64,

    // ── Order book features ───────────────────────────────────────────────────
    pub book_imbalance_5:    Option<f64>,
    pub book_imbalance_full: Option<f64>,
    pub book_spread_pct:     Option<f64>,
    pub book_pressure:       Option<f64>,

    // ── Regime features ───────────────────────────────────────────────────────
    /// abs(return_300s) / vol_1800s — trend strength, regime-normalised.
    pub trend_strength: Option<f64>,
    /// vol_300s / vol_1800s — > 1 means short-term vol above regime baseline.
    pub vol_regime:     Option<f64>,

    // ── Normalised (z-scored) returns — key for cross-regime generalisation ──
    pub zreturn_30s:  Option<f64>,
    pub zreturn_300s: Option<f64>,

    // ── Retained for outlier-filter feedback path ─────────────────────────────
    /// Variance estimate used by the dedup/outlier stage — kept as a convenience
    /// field so the engine can pass it back without holding a separate handle.
    pub vol_1800s_variance: f64,
}

// ─── Feature state ────────────────────────────────────────────────────────────

/// Owns all incremental accumulators; updated on every accepted fused tick.
pub struct FeatureState {
    // Returns
    ret_5s:   WindowedReturn,
    ret_30s:  WindowedReturn,
    ret_300s: WindowedReturn,
    ret_1800s: WindowedReturn,

    // Realised volatility
    vol_30:   RealisedVol,
    vol_300:  RealisedVol,
    vol_1800: RealisedVol,

    // Order flow imbalance
    ofi_5:   OrderFlowImbalance,
    ofi_30:  OrderFlowImbalance,
    ofi_300: OrderFlowImbalance,

    // Rolling VWAP
    vwap_30:  RollingVwap,
    vwap_300: RollingVwap,

    // Volume accumulators (reuse OFI total_volume(), no extra state needed)
    // ofi_30 and ofi_300 already track volume — volume_ratio derived from them.

    // Tick activity / regime
    activity: ActivityRegime,

    // Order book
    pub book_imbalance: BookImbalanceTracker,
    pub book_pressure:  BookPressureTracker,
}

impl FeatureState {
    pub fn new() -> Self {
        Self {
            ret_5s:    WindowedReturn::new(5),
            ret_30s:   WindowedReturn::new(30),
            ret_300s:  WindowedReturn::new(300),
            ret_1800s: WindowedReturn::new(1800),

            vol_30:   RealisedVol::new(30),
            vol_300:  RealisedVol::new(300),
            vol_1800: RealisedVol::new(1800),

            ofi_5:   OrderFlowImbalance::new(5),
            ofi_30:  OrderFlowImbalance::new(30),
            ofi_300: OrderFlowImbalance::new(300),

            vwap_30:  RollingVwap::new(30),
            vwap_300: RollingVwap::new(300),

            activity: ActivityRegime::new(),

            book_imbalance: BookImbalanceTracker::new(),
            book_pressure:  BookPressureTracker::new(),
        }
    }

    /// Primary hot-path update — called on every [`FusedTick`].
    pub fn update_from_fused(&mut self, fused: &FusedTick) -> FeatureVector {
        let price = fused.price;
        let ts    = fused.ts_micros;
        let vol   = fused.volume;

        // ── Realised vol (price-based, time-windowed) ─────────────────────────
        self.vol_30.update_fused(ts, price);
        self.vol_300.update_fused(ts, price);
        self.vol_1800.update_fused(ts, price);

        // ── Returns — must come after vol so vol_1800s is fresh for z-scoring ─
        let lp = if price > 0.0 { price.ln() } else { 0.0 };
        self.ret_5s.update(ts, price);
        self.ret_30s.update(ts, price);
        self.ret_300s.update(ts, price);
        self.ret_1800s.update(ts, price);

        // ── Order flow imbalance ─────────────────────────────────────────────
        if let Some(buy_ratio) = fused.buy_ratio {
            let buy_vol  = vol * buy_ratio;
            let sell_vol = vol * (1.0 - buy_ratio);
            self.ofi_5.update_fused(ts, buy_vol, sell_vol);
            self.ofi_30.update_fused(ts, buy_vol, sell_vol);
            self.ofi_300.update_fused(ts, buy_vol, sell_vol);
        }

        // ── VWAP ─────────────────────────────────────────────────────────────
        self.vwap_30.update(ts, price, vol);
        self.vwap_300.update(ts, price, vol);

        // ── Tick activity / regime ────────────────────────────────────────────
        self.activity.update(ts);

        // ── Derived scalars ───────────────────────────────────────────────────
        let vol_30s   = self.vol_30.value();
        let vol_300s  = self.vol_300.value();
        let vol_1800s = self.vol_1800.value();

        let vol_ratio = vol_1800s.and_then(|v1800| {
            if v1800 == 0.0 { None } else { vol_30s.map(|v30| v30 / v1800) }
        });
        let vol_regime = vol_1800s.and_then(|v1800| {
            if v1800 == 0.0 { None } else { vol_300s.map(|v300| v300 / v1800) }
        });

        let return_5s   = self.ret_5s.value(lp);
        let return_30s  = self.ret_30s.value(lp);
        let return_300s = self.ret_300s.value(lp);
        let return_1800s = self.ret_1800s.value(lp);

        // z-scored returns: divide by vol_1800s to normalise across price regimes.
        let zreturn_30s = vol_1800s.and_then(|v| {
            if v == 0.0 { None } else { return_30s.map(|r| r / v) }
        });
        let zreturn_300s = vol_1800s.and_then(|v| {
            if v == 0.0 { None } else { return_300s.map(|r| r / v) }
        });

        let trend_strength = vol_1800s.and_then(|v| {
            if v == 0.0 { None }
            else { return_300s.map(|r| r.abs() / v) }
        });

        let ofi_5s  = self.ofi_5.value();
        let ofi_30s = self.ofi_30.value();
        let ofi_delta_30s = ofi_5s - ofi_30s;

        let vwap_dev_30s  = vol_1800s.and_then(|v| self.vwap_30.deviation_z(price, v));
        let vwap_dev_300s = vol_1800s.and_then(|v| self.vwap_300.deviation_z(price, v));

        // Volume ratio: recent / baseline activity.
        let vol_30_total  = self.ofi_30.total_volume();
        let vol_300_total = self.ofi_300.total_volume();
        let volume_ratio = if vol_300_total == 0.0 { None }
            else { Some(vol_30_total / vol_300_total) };

        // Cross-exchange spread as a percentage of mid price.
        let spread_pct = fused.spread_pct();

        let ts_now = ts;
        FeatureVector {
            ts_micros: ts,
            price,

            return_5s,
            return_30s,
            return_300s,

            vol_30s,
            vol_300s,
            vol_1800s,
            vol_ratio,

            ofi_5s,
            ofi_30s,
            ofi_300s:      self.ofi_300.value(),
            ofi_delta_30s,
            buy_ratio_30s:  self.ofi_30.buy_ratio(),
            buy_ratio_300s: self.ofi_300.buy_ratio(),

            vwap_dev_30s,
            vwap_dev_300s,

            volume_ratio,

            tick_velocity:   self.activity.fast_rate(),
            activity_regime: self.activity.ratio(),

            spread_pct,

            book_imbalance_5:    self.book_imbalance.fused_top5(ts_now),
            book_imbalance_full: self.book_imbalance.fused_full(ts_now),
            book_spread_pct:     self.book_pressure.spread_pct(ts_now),
            book_pressure:       self.book_pressure.pressure(ts_now),

            trend_strength,
            vol_regime,

            zreturn_30s,
            zreturn_300s,

            vol_1800s_variance: self.vol_1800.value().map(|v| v * v).unwrap_or(0.0),
        }
    }

    /// Update book-derived features from a [`BookSnapshot`].
    ///
    /// Called by the pipeline book stage; does not emit a `FeatureVector`.
    pub fn update_book(&mut self, snap: &crate::types::BookSnapshot) {
        let ts = snap.ts_micros;

        let imb5    = snap.imbalance(5);
        let imb_full = snap.imbalance(snap.bids.len().max(snap.asks.len()));

        if let (Some(i5), Some(ifull)) = (imb5, imb_full) {
            self.book_imbalance.update(snap.exchange, i5, ifull, ts);
        }

        if let (Some(spread_usd), Some(mid), Some(microprice)) = (
            snap.spread_usd(),
            snap.mid_price(),
            snap.weighted_mid(5),
        ) {
            let mp_dev = if mid > 0.0 { (microprice - mid) / mid } else { 0.0 };
            self.book_pressure.update(snap.exchange, spread_usd, mp_dev, mid, ts);
        }
    }
}

impl Default for FeatureState { fn default() -> Self { Self::new() } }

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windowed_return_fills_then_reads() {
        let mut wr = WindowedReturn::new(30);
        let base_ts = 0_i64;
        // Feed 35 seconds of prices at 1 Hz.
        for i in 0..=35_i64 {
            wr.update(base_ts + i * 1_000_000, 100.0 + i as f64);
        }
        // Should now have a valid 30-second return.
        let lp_now = (135.0_f64).ln();
        assert!(wr.value(lp_now).is_some());
    }

    #[test]
    fn windowed_return_not_ready_during_warmup() {
        let mut wr = WindowedReturn::new(30);
        // Only 10 seconds of history — should not return a value.
        for i in 0..=10_i64 {
            wr.update(i * 1_000_000, 100.0 + i as f64);
        }
        let lp_now = (110.0_f64).ln();
        assert!(wr.value(lp_now).is_none());
    }

    #[test]
    fn realised_vol_seeding() {
        let mut rv = RealisedVol::new(30);
        // < 3 returns → None
        rv.update_fused(0, 100.0);
        rv.update_fused(1_000_000, 101.0);
        assert!(rv.value().is_none());
        rv.update_fused(2_000_000, 102.0);
        assert!(rv.value().is_some());
    }

    #[test]
    fn rolling_vwap_basic() {
        let mut vwap = RollingVwap::new(30);
        vwap.update(0, 100.0, 1.0);
        vwap.update(1_000_000, 200.0, 1.0);
        // VWAP should be (100 + 200) / 2 = 150.
        assert!((vwap.vwap().unwrap() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn ofi_balanced() {
        let mut ofi = OrderFlowImbalance::new(60);
        let mk = |side: TradeSide| Tick {
            ts_micros: 0, price: 50_000.0, quantity: 1.0, side: Some(side),
            exchange: Exchange::Binance, symbol: crate::types::Symbol::BtcUsd, trade_id: "0".into(),
        };
        ofi.update(&mk(TradeSide::Buy));
        ofi.update(&mk(TradeSide::Sell));
        assert!((ofi.value()).abs() < 1e-9);
        assert!((ofi.buy_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn book_imbalance_tracker_staleness() {
        let mut tracker = BookImbalanceTracker::new();
        tracker.update(Exchange::Binance, 0.4, 0.3, 0);
        // Fresh at t=0
        assert!(tracker.fused_top5(0).is_some());
        // Stale after BOOK_STALE_MICROS + 1
        assert!(tracker.fused_top5(BOOK_STALE_MICROS + 1).is_none());
    }

    #[test]
    fn activity_regime_burst() {
        let mut ar = ActivityRegime::new();
        // Seed 1800 s of slow activity (1 tick / 10 s).
        for i in 0..=180_i64 {
            ar.update(i * 10_000_000);
        }
        let slow_ratio = ar.ratio();
        // Now burst 30 s of fast activity (10 ticks / s).
        let base = 1_800_000_000_i64;
        for i in 0..300_i64 {
            ar.update(base + i * 100_000);
        }
        let fast_ratio = ar.ratio();
        // Fast ratio should be substantially > slow ratio.
        assert!(fast_ratio > slow_ratio);
    }
}
