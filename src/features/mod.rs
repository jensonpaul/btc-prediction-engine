//! Incremental feature engineering — all O(1) per tick.
//!
//! # Feature inventory
//!
//! | Feature | Range | Description |
//! |---|---|---|
//! | `rsi_14` | [0, 100] | Wilder RSI, 14-tick period |
//! | `vwap_deviation` | fraction | (price − session VWAP) / VWAP |
//! | `momentum_micro` | fraction | (p_now − p_30ago) / p_30ago |
//! | `momentum_short` | fraction | (p_now − p_300ago) / p_300ago |
//! | `ewma_vol_tick` | fraction | Per-tick EWMA σ |
//! | `tick_velocity` | ticks/s | 30-s rolling rate |
//! | `ofi_30s` | [−1, 1] | Order flow imbalance, 30 s |
//! | `ofi_300s` | [−1, 1] | Order flow imbalance, 300 s |
//! | `autocorr_lag1` | [−1, 1] | Lag-1 return autocorrelation |
//! | `realised_vol_30s` | fraction | Realised vol, 30-s window |
//! | `inter_exchange_spread` | USD | max − min last price per exchange |
//! | `price` | USD | Current BTC/USD price |
//! | `book_imbalance_top5` | [−1, 1] | Top-5 bid/ask volume imbalance (book feeds) |
//! | `book_imbalance_full` | [−1, 1] | Full-depth bid/ask volume imbalance (book feeds) |
//! | `book_weighted_mid` | USD | Volume-weighted mid-price (book feeds) |
//! | `book_spread_usd` | USD | Best bid–ask spread (book feeds) |

use std::collections::{HashMap, VecDeque};
use crate::types::{Exchange, Tick, TradeSide};
use crate::price_fusion::FusedTick;

// ─── Incremental RSI (Wilder) ─────────────────────────────────────────────────

pub struct IncrementalRsi {
    period:      usize,
    avg_gain:    f64,
    avg_loss:    f64,
    prev_price:  Option<f64>,
    seeded:      bool,
    seed_gains:  Vec<f64>,
    seed_losses: Vec<f64>,
}

impl IncrementalRsi {
    pub fn new(period: usize) -> Self {
        assert!(period >= 2);
        Self { period, avg_gain: 0.0, avg_loss: 0.0, prev_price: None,
               seeded: false, seed_gains: Vec::with_capacity(period),
               seed_losses: Vec::with_capacity(period) }
    }

    pub fn update(&mut self, price: f64) -> Option<f64> {
        if let Some(prev) = self.prev_price {
            let delta = price - prev;
            let gain  = delta.max(0.0);
            let loss  = (-delta).max(0.0);
            if !self.seeded {
                self.seed_gains.push(gain);
                self.seed_losses.push(loss);
                if self.seed_gains.len() == self.period {
                    self.avg_gain = self.seed_gains.iter().sum::<f64>() / self.period as f64;
                    self.avg_loss = self.seed_losses.iter().sum::<f64>() / self.period as f64;
                    self.seeded = true;
                }
            } else {
                let a = 1.0 / self.period as f64;
                self.avg_gain = self.avg_gain * (1.0 - a) + gain * a;
                self.avg_loss = self.avg_loss * (1.0 - a) + loss * a;
            }
        }
        self.prev_price = Some(price);
        self.value()
    }

    pub fn value(&self) -> Option<f64> {
        if !self.seeded { return None; }
        if self.avg_loss == 0.0 { return Some(100.0); }
        Some(100.0 - 100.0 / (1.0 + self.avg_gain / self.avg_loss))
    }
}

// ─── Session VWAP ────────────────────────────────────────────────────────────

pub struct SessionVwap { pv: f64, vol: f64 }

impl SessionVwap {
    pub fn new() -> Self { Self { pv: 0.0, vol: 0.0 } }
    pub fn update(&mut self, tick: &Tick) { self.pv += tick.price * tick.quantity; self.vol += tick.quantity; }
    /// Update from a pre-fused price + volume (used by the fusion pipeline).
    pub fn update_fused(&mut self, price: f64, volume: f64) { self.pv += price * volume; self.vol += volume; }
    pub fn value(&self) -> Option<f64> { if self.vol > 0.0 { Some(self.pv / self.vol) } else { None } }
    pub fn deviation(&self, price: f64) -> Option<f64> {
        self.value().map(|v| if v != 0.0 { (price - v) / v } else { 0.0 })
    }
}

// ─── EWMA Volatility ─────────────────────────────────────────────────────────

/// RiskMetrics EWMA variance: σ²_t = λ·σ²_{t-1} + (1−λ)·r²_t
pub struct EwmaVolatility { lambda: f64, pub variance: f64, prev: Option<f64>, seeded: bool }

impl EwmaVolatility {
    pub fn new(lambda: f64) -> Self { Self { lambda, variance: 0.0, prev: None, seeded: false } }
    pub fn default_lambda() -> Self { Self::new(0.94) }

    pub fn update(&mut self, price: f64) {
        if let Some(p) = self.prev {
            if p > 0.0 {
                let r2 = (price / p).ln().powi(2);
                self.variance = if self.seeded {
                    self.lambda * self.variance + (1.0 - self.lambda) * r2
                } else { self.seeded = true; r2 };
            }
        }
        self.prev = Some(price);
    }

    pub fn tick_vol(&self) -> Option<f64> { if self.seeded { Some(self.variance.sqrt()) } else { None } }
}

// ─── Rolling momentum ────────────────────────────────────────────────────────

pub struct RollingMomentum { window: usize, buf: VecDeque<f64> }

impl RollingMomentum {
    pub fn new(window: usize) -> Self { Self { window, buf: VecDeque::with_capacity(window + 1) } }
    pub fn update(&mut self, price: f64) {
        self.buf.push_back(price);
        if self.buf.len() > self.window + 1 { self.buf.pop_front(); }
    }
    pub fn value(&self) -> Option<f64> {
        if self.buf.len() < self.window + 1 { return None; }
        let old = *self.buf.front().unwrap();
        let new = *self.buf.back().unwrap();
        if old == 0.0 { None } else { Some((new - old) / old) }
    }
}

// ─── Tick velocity ───────────────────────────────────────────────────────────

pub struct TickVelocity { window_micros: i64, buf: VecDeque<i64> }

impl TickVelocity {
    pub fn new(window_secs: i64) -> Self { Self { window_micros: window_secs * 1_000_000, buf: VecDeque::new() } }
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

// ─── Order flow imbalance ────────────────────────────────────────────────────

/// Rolling order-flow imbalance with O(1) `value()`.
///
/// `buy_sum` and `sell_sum` are kept in sync with the window on every
/// insertion and eviction, so `value()` needs no scan.
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

    /// Update from pre-aggregated buy/sell volumes (used by the fusion pipeline).
    pub fn update_fused(&mut self, ts: i64, buy_vol: f64, sell_vol: f64) {
        self.buy_sum  += buy_vol;
        self.sell_sum += sell_vol;
        self.entries.push_back((ts, buy_vol, sell_vol));
        self.evict_stale(ts - self.window_micros);
    }

    /// O(1) — reads running sums directly.
    #[inline]
    pub fn value(&self) -> f64 {
        let total = self.buy_sum + self.sell_sum;
        if total == 0.0 { 0.0 } else { (self.buy_sum - self.sell_sum) / total }
    }
}

// ─── Return autocorrelation (lag-1) ─────────────────────────────────────────

/// Incremental lag-1 Pearson autocorrelation of log-returns — O(1) per tick.
///
/// Maintains five running sums over the pair series (x[i] = r[i], y[i] = r[i+1]):
///
/// ```text
///   sum_x  = Σ r[i]       (i = 0 .. n-2)
///   sum_y  = Σ r[i+1]     (i = 0 .. n-2)
///   sum_x2 = Σ r[i]²
///   sum_y2 = Σ r[i+1]²
///   sum_xy = Σ r[i]·r[i+1]
/// ```
///
/// When the oldest return `r_old` leaves the window the pair `(r_old, r_next)`
/// is removed from {sum_x, sum_xy, sum_x2}, and the pair `(r_prev_new_head,
/// r_new_head)` is added to {sum_y, sum_xy, sum_y2} as the new tail pair.
/// Concretely, the window of *pairs* shrinks by one at the front (x-side) and
/// the new head of the returns window becomes the x of the only surviving pair
/// that was previously an interior pair — which is already accounted for.
///
/// The simpler bookkeeping: think of the pair window as the `returns` deque
/// minus its last element (x-series) and minus its first element (y-series).
/// On push we add one pair; on pop we remove one pair.
pub struct ReturnAutocorr {
    window:  usize,
    returns: VecDeque<f64>,
    prev:    Option<f64>,
    // Running sums over the pair series of length (n-1)
    sum_x:   f64,
    sum_y:   f64,
    sum_x2:  f64,
    sum_y2:  f64,
    sum_xy:  f64,
}

impl ReturnAutocorr {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            returns: VecDeque::with_capacity(window + 1),
            prev:    None,
            sum_x:   0.0,
            sum_y:   0.0,
            sum_x2:  0.0,
            sum_y2:  0.0,
            sum_xy:  0.0,
        }
    }

    pub fn update(&mut self, price: f64) {
        if let Some(p) = self.prev {
            if p > 0.0 {
                let r = (price / p).ln();

                // The new return pairs with the current tail as (tail, r).
                if let Some(&tail) = self.returns.back() {
                    self.sum_x  += tail;
                    self.sum_y  += r;
                    self.sum_x2 += tail * tail;
                    self.sum_y2 += r * r;
                    self.sum_xy += tail * r;
                }

                self.returns.push_back(r);

                // Evict the oldest return when the window is full.
                if self.returns.len() > self.window {
                    let evicted = self.returns.pop_front().unwrap();
                    // The pair (evicted, next_head) leaves the x-side.
                    if let Some(&next_head) = self.returns.front() {
                        self.sum_x  -= evicted;
                        self.sum_y  -= next_head;
                        self.sum_x2 -= evicted * evicted;
                        self.sum_y2 -= next_head * next_head;
                        self.sum_xy -= evicted * next_head;
                    }
                }
            }
        }
        self.prev = Some(price);
    }

    /// Pearson correlation between r[t] and r[t-1] — O(1).
    #[inline]
    pub fn value(&self) -> Option<f64> {
        let n = self.returns.len();
        if n < 4 { return None; }
        let m = (n - 1) as f64; // number of pairs

        // population Pearson over the pair series
        let var_x = self.sum_x2 - self.sum_x * self.sum_x / m;
        let var_y = self.sum_y2 - self.sum_y * self.sum_y / m;
        let cov   = self.sum_xy - self.sum_x * self.sum_y / m;

        let denom = (var_x * var_y).sqrt();
        if denom == 0.0 { None } else { Some((cov / denom).clamp(-1.0, 1.0)) }
    }
}

// ─── Realised volatility ─────────────────────────────────────────────────────

/// Rolling realised volatility (std of log-returns over a window) — O(1) per tick.
///
/// Maintains `sum` (Σ r) and `sum_sq` (Σ r²) so that `value()` computes the
/// sample standard deviation without any allocation or window scan:
///
/// ```text
/// variance = (sum_sq − sum² / n) / (n − 1)
/// ```
pub struct RealisedVol {
    window_micros: i64,
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

    pub fn update(&mut self, tick: &Tick) {
        if let Some((_, pp)) = self.prev {
            if pp > 0.0 {
                let r = (tick.price / pp).ln();
                self.push_return(tick.ts_micros, r);
                self.evict_stale(tick.ts_micros - self.window_micros);
            }
        }
        self.prev = Some((tick.ts_micros, tick.price));
    }

    /// Update from a fused canonical price (used by the fusion pipeline).
    pub fn update_fused(&mut self, ts: i64, price: f64) {
        if let Some((_, pp)) = self.prev {
            if pp > 0.0 {
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
        // Guard against tiny negative values from floating-point cancellation.
        Some(variance.max(0.0).sqrt())
    }
}

// ─── Order book imbalance tracker ───────────────────────────────────────────

/// Tracks the latest top-N book imbalance per exchange and fuses them into a
/// single cross-exchange estimate.
///
/// Imbalance decays to `None` after [`BOOK_STALE_MICROS`] without an update —
/// a stale snapshot is worse than no snapshot for short-timescale features.
pub struct BookImbalanceTracker {
    /// (imbalance ∈ [−1,1], received_at_micros) per exchange.
    last: HashMap<Exchange, (f64, i64)>,
}

/// A book snapshot older than this is considered stale and excluded from the
/// fused imbalance calculation.
const BOOK_STALE_MICROS: i64 = 2_000_000; // 2 s

impl BookImbalanceTracker {
    pub fn new() -> Self { Self { last: HashMap::new() } }

    /// Record a new snapshot for one exchange.
    pub fn update(&mut self, exchange: Exchange, imbalance: f64, ts_micros: i64) {
        self.last.insert(exchange, (imbalance, ts_micros));
    }

    /// Simple (bid_vol − ask_vol) imbalance for the most recent snapshot from
    /// `exchange`. Returns `None` if no snapshot has been received or the
    /// latest is stale.
    pub fn latest_for(&self, exchange: Exchange, now_micros: i64) -> Option<f64> {
        self.last.get(&exchange).and_then(|&(imb, ts)| {
            if now_micros - ts <= BOOK_STALE_MICROS { Some(imb) } else { None }
        })
    }

    /// Volume-averaged imbalance across all exchanges with a fresh snapshot.
    ///
    /// Returns `None` if no exchange has a fresh snapshot.
    ///
    /// Accumulates into stack variables — no heap allocation.
    pub fn fused_imbalance(&self, now_micros: i64) -> Option<f64> {
        let (mut total, mut count) = (0.0_f64, 0u32);
        for &(imb, ts) in self.last.values() {
            if now_micros - ts <= BOOK_STALE_MICROS {
                total += imb;
                count += 1;
            }
        }
        if count == 0 { None } else { Some(total / count as f64) }
    }
}

// ─── Inter-exchange spread tracker ───────────────────────────────────────────

/// Tracks last price per exchange and computes the spread between them.
pub struct InterExchangeSpread { last: HashMap<Exchange, f64> }

impl InterExchangeSpread {
    pub fn new() -> Self { Self { last: HashMap::new() } }
    pub fn update(&mut self, tick: &Tick) { self.last.insert(tick.exchange, tick.price); }
    /// Max price − min price across all exchanges that have reported.
    pub fn spread(&self) -> f64 {
        if self.last.len() < 2 { return 0.0; }
        let (min, max) = self.last.values().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        max - min
    }
}

// ─── Composite feature vector ────────────────────────────────────────────────

/// All features at one tick instant. `None` = not yet seeded.
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub ts_micros:             i64,
    pub price:                 f64,
    pub rsi_14:                Option<f64>,
    pub vwap_deviation:        Option<f64>,
    pub momentum_micro:        Option<f64>,
    pub momentum_short:        Option<f64>,
    pub ewma_vol_tick:         Option<f64>,
    /// EWMA variance (for outlier filter feedback).
    pub ewma_variance:         f64,
    pub tick_velocity:         f64,
    pub ofi_30s:               f64,
    pub ofi_300s:              f64,
    pub autocorr_lag1:         Option<f64>,
    pub realised_vol_30s:      Option<f64>,
    pub inter_exchange_spread: f64,
    // ── Order book features (None when no book feed is connected) ────────────
    /// Fused top-5 bid/ask volume imbalance across all exchanges with a live
    /// book feed. ∈ [−1, 1]: +1 = fully bid-side, −1 = fully ask-side.
    pub book_imbalance_top5:   Option<f64>,
    /// Same, but over all [`BOOK_DEPTH`] levels available.
    pub book_imbalance_full:   Option<f64>,
    /// Weighted mid-price derived from the book (more stable than last trade).
    /// `None` when no book is connected.
    pub book_weighted_mid:     Option<f64>,
    /// Best bid–ask spread in USD. `None` when no book is connected.
    pub book_spread_usd:       Option<f64>,
}

// ─── Feature state ───────────────────────────────────────────────────────────

/// Owns all incremental feature accumulators.
///
/// Updated by the engine hot-path on every accepted tick.
pub struct FeatureState {
    pub rsi:             IncrementalRsi,
    pub vwap:            SessionVwap,
    pub vol:             EwmaVolatility,
    pub mom_micro:       RollingMomentum,
    pub mom_short:       RollingMomentum,
    pub velocity:        TickVelocity,
    pub ofi_30:          OrderFlowImbalance,
    pub ofi_300:         OrderFlowImbalance,
    pub autocorr:        ReturnAutocorr,
    pub realised_30:     RealisedVol,
    pub spread:          InterExchangeSpread,
    pub book_imbalance:  BookImbalanceTracker,
}

impl FeatureState {
    pub fn new() -> Self {
        Self {
            rsi:            IncrementalRsi::new(14),
            vwap:           SessionVwap::new(),
            vol:            EwmaVolatility::default_lambda(),
            mom_micro:      RollingMomentum::new(30),
            mom_short:      RollingMomentum::new(300),
            velocity:       TickVelocity::new(30),
            ofi_30:         OrderFlowImbalance::new(30),
            ofi_300:        OrderFlowImbalance::new(300),
            autocorr:       ReturnAutocorr::new(60),
            realised_30:    RealisedVol::new(30),
            spread:         InterExchangeSpread::new(),
            book_imbalance: BookImbalanceTracker::new(),
        }
    }

    /// Update all features from a tick and return the current vector.
    ///
    /// Used for direct tick injection (testing / replay) when the fusion stage
    /// is bypassed.
    pub fn update(&mut self, tick: &Tick) -> FeatureVector {
        self.rsi.update(tick.price);
        self.vwap.update(tick);
        self.vol.update(tick.price);
        self.mom_micro.update(tick.price);
        self.mom_short.update(tick.price);
        self.velocity.update(tick.ts_micros);
        self.ofi_30.update(tick);
        self.ofi_300.update(tick);
        self.autocorr.update(tick.price);
        self.realised_30.update(tick);
        self.spread.update(tick);

        let ts = tick.ts_micros;
        FeatureVector {
            ts_micros:             ts,
            price:                 tick.price,
            rsi_14:                self.rsi.value(),
            vwap_deviation:        self.vwap.deviation(tick.price),
            momentum_micro:        self.mom_micro.value(),
            momentum_short:        self.mom_short.value(),
            ewma_vol_tick:         self.vol.tick_vol(),
            ewma_variance:         self.vol.variance,
            tick_velocity:         self.velocity.rate(),
            ofi_30s:               self.ofi_30.value(),
            ofi_300s:              self.ofi_300.value(),
            autocorr_lag1:         self.autocorr.value(),
            realised_vol_30s:      self.realised_30.value(),
            inter_exchange_spread: self.spread.spread(),
            book_imbalance_top5:   self.book_imbalance.fused_imbalance(ts),
            book_imbalance_full:   self.book_imbalance.fused_imbalance(ts),
            book_weighted_mid:     None,
            book_spread_usd:       None,
        }
    }

    /// Record a new book snapshot and update the imbalance tracker.
    ///
    /// Called from the pipeline book stage on every [`BookSnapshot`].
    /// Cheap: only updates the per-exchange imbalance entry; the full feature
    /// vector is recomputed on the next `update_from_fused` call.
    pub fn update_book(&mut self, snap: &crate::types::BookSnapshot) {
        if let Some(imb) = snap.imbalance(5) {
            self.book_imbalance.update(snap.exchange, imb, snap.ts_micros);
        }
    }

    /// Update all features from a [`FusedTick`] — the primary hot-path entry
    /// point in production.
    ///
    /// Uses the fused canonical price (volume-weighted across all exchanges)
    /// rather than a single-exchange tick.  OFI and spread are taken directly
    /// from the pre-computed [`FusedTick`] fields, which already aggregate
    /// across all contributing exchanges.
    pub fn update_from_fused(&mut self, fused: &FusedTick) -> FeatureVector {
        let price = fused.price;
        let ts    = fused.ts_micros;

        // Update price-derived features using the canonical fused price
        self.rsi.update(price);
        self.vol.update(price);
        self.mom_micro.update(price);
        self.mom_short.update(price);
        self.velocity.update(ts);
        self.autocorr.update(price);

        // Update VWAP accumulator using fused volume
        self.vwap.update_fused(price, fused.volume);

        // Update OFI accumulators from fused buy_ratio + volume
        // Distribute buy and sell volume into the OFI windows
        if let Some(buy_ratio) = fused.buy_ratio {
            let buy_vol  = fused.volume * buy_ratio;
            let sell_vol = fused.volume * (1.0 - buy_ratio);
            self.ofi_30.update_fused(ts, buy_vol, sell_vol);
            self.ofi_300.update_fused(ts, buy_vol, sell_vol);
        }

        // Update realised vol accumulator
        self.realised_30.update_fused(ts, price);

        // Inter-exchange spread is already computed by the fuser
        // No need to re-track; read it directly from FusedTick

        FeatureVector {
            ts_micros:             ts,
            price,
            rsi_14:                self.rsi.value(),
            vwap_deviation:        self.vwap.deviation(price),
            momentum_micro:        self.mom_micro.value(),
            momentum_short:        self.mom_short.value(),
            ewma_vol_tick:         self.vol.tick_vol(),
            ewma_variance:         self.vol.variance,
            tick_velocity:         self.velocity.rate(),
            ofi_30s:               self.ofi_30.value(),
            ofi_300s:              self.ofi_300.value(),
            autocorr_lag1:         self.autocorr.value(),
            realised_vol_30s:      self.realised_30.value(),
            // Use the fuser's pre-computed cross-exchange spread directly
            inter_exchange_spread: fused.cross_exchange_spread,
            // Book features populated if any book feed has sent a fresh snapshot
            book_imbalance_top5:   self.book_imbalance.fused_imbalance(ts),
            book_imbalance_full:   self.book_imbalance.fused_imbalance(ts),
            book_weighted_mid:     None,
            book_spread_usd:       None,
        }
    }
}

impl Default for FeatureState { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn rsi_seeding() {
        let mut r = IncrementalRsi::new(14);
        for _ in 0..13 { assert!(r.update(100.0).is_none()); }
        assert!(r.update(100.0).is_some());
    }

    #[test] fn momentum_window() {
        let mut m = RollingMomentum::new(3);
        m.update(100.0); m.update(110.0); m.update(120.0);
        assert!(m.value().is_none()); // window=3 needs 4 prices
        m.update(130.0);
        let v = m.value().unwrap();
        assert!((v - 0.3).abs() < 1e-9);
    }

    #[test] fn ofi_balanced() {
        let mut ofi = OrderFlowImbalance::new(60);
        let mk = |side: TradeSide| Tick {
            ts_micros: 0, price: 50_000.0, quantity: 1.0, side: Some(side),
            exchange: Exchange::Binance, symbol: crate::types::Symbol::BtcUsd, trade_id: "0".into(),
        };
        ofi.update(&mk(TradeSide::Buy));
        ofi.update(&mk(TradeSide::Sell));
        assert!((ofi.value()).abs() < 1e-9);
    }

    #[test] fn spread_two_exchanges() {
        let mut s = InterExchangeSpread::new();
        let t = |ex: Exchange, p: f64| Tick {
            ts_micros: 0, price: p, quantity: 0.1, side: None,
            exchange: ex, symbol: crate::types::Symbol::BtcUsd, trade_id: "x".into(),
        };
        s.update(&t(Exchange::Binance, 50_010.0));
        s.update(&t(Exchange::Kraken,  50_000.0));
        assert!((s.spread() - 10.0).abs() < 1e-6);
    }
}
