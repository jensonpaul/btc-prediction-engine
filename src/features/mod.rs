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

pub struct OrderFlowImbalance { window_micros: i64, entries: VecDeque<(i64, f64, Option<TradeSide>)> }

impl OrderFlowImbalance {
    pub fn new(window_secs: i64) -> Self { Self { window_micros: window_secs * 1_000_000, entries: VecDeque::new() } }
    pub fn update(&mut self, tick: &Tick) {
        self.entries.push_back((tick.ts_micros, tick.quantity, tick.side));
        let cutoff = tick.ts_micros - self.window_micros;
        while self.entries.front().map_or(false, |e| e.0 < cutoff) { self.entries.pop_front(); }
    }
    /// Update from pre-aggregated buy/sell volumes (used by the fusion pipeline).
    pub fn update_fused(&mut self, ts: i64, buy_vol: f64, sell_vol: f64) {
        // Represent as two synthetic entries: one buy, one sell
        if buy_vol > 0.0  { self.entries.push_back((ts, buy_vol,  Some(TradeSide::Buy)));  }
        if sell_vol > 0.0 { self.entries.push_back((ts, sell_vol, Some(TradeSide::Sell))); }
        let cutoff = ts - self.window_micros;
        while self.entries.front().map_or(false, |e| e.0 < cutoff) { self.entries.pop_front(); }
    }
    pub fn value(&self) -> f64 {
        let (mut buy, mut sell, mut total) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (_, qty, side) in &self.entries {
            total += qty;
            match side { Some(TradeSide::Buy) => buy += qty, Some(TradeSide::Sell) => sell += qty, None => {} }
        }
        if total == 0.0 { 0.0 } else { (buy - sell) / total }
    }
}

// ─── Return autocorrelation (lag-1) ─────────────────────────────────────────

/// Incremental lag-1 autocorrelation of log-returns.
pub struct ReturnAutocorr {
    window:  usize,
    returns: VecDeque<f64>,
    prev:    Option<f64>,
}

impl ReturnAutocorr {
    pub fn new(window: usize) -> Self { Self { window, returns: VecDeque::with_capacity(window + 1), prev: None } }
    pub fn update(&mut self, price: f64) {
        if let Some(p) = self.prev {
            if p > 0.0 {
                let r = (price / p).ln();
                self.returns.push_back(r);
                if self.returns.len() > self.window { self.returns.pop_front(); }
            }
        }
        self.prev = Some(price);
    }
    /// Pearson correlation between r[t] and r[t-1].
    pub fn value(&self) -> Option<f64> {
        let n = self.returns.len();
        if n < 4 { return None; }
        let r: Vec<f64> = self.returns.iter().copied().collect();
        let (x, y): (Vec<f64>, Vec<f64>) = (r[..n-1].to_vec(), r[1..].to_vec());
        let mx = mean(&x);
        let my = mean(&y);
        let cov: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / (n - 1) as f64;
        let sx = std_dev(&x, mx);
        let sy = std_dev(&y, my);
        if sx * sy == 0.0 { None } else { Some((cov / (sx * sy)).clamp(-1.0, 1.0)) }
    }
}

fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn std_dev(v: &[f64], m: f64) -> f64 { (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt() }

// ─── Realised volatility ─────────────────────────────────────────────────────

/// Rolling realised volatility (std of log-returns over a window).
pub struct RealisedVol { window_secs: i64, entries: VecDeque<(i64, f64)>, prev: Option<(i64, f64)> }

impl RealisedVol {
    pub fn new(window_secs: i64) -> Self {
        Self { window_secs, entries: VecDeque::new(), prev: None }
    }
    pub fn update(&mut self, tick: &Tick) {
        if let Some((_, pp)) = self.prev {
            if pp > 0.0 {
                let r = (tick.price / pp).ln();
                self.entries.push_back((tick.ts_micros, r));
                let cutoff = tick.ts_micros - self.window_secs * 1_000_000;
                while self.entries.front().map_or(false, |e| e.0 < cutoff) { self.entries.pop_front(); }
            }
        }
        self.prev = Some((tick.ts_micros, tick.price));
    }
    /// Update from a fused canonical price (used by the fusion pipeline).
    pub fn update_fused(&mut self, ts: i64, price: f64) {
        if let Some((_, pp)) = self.prev {
            if pp > 0.0 {
                let r = (price / pp).ln();
                self.entries.push_back((ts, r));
                let cutoff = ts - self.window_secs * 1_000_000;
                while self.entries.front().map_or(false, |e| e.0 < cutoff) { self.entries.pop_front(); }
            }
        }
        self.prev = Some((ts, price));
    }
    pub fn value(&self) -> Option<f64> {
        let v: Vec<f64> = self.entries.iter().map(|e| e.1).collect();
        if v.len() < 3 { return None; }
        let m = mean(&v);
        Some(std_dev(&v, m))
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
}

impl FeatureState {
    pub fn new() -> Self {
        Self {
            rsi:         IncrementalRsi::new(14),
            vwap:        SessionVwap::new(),
            vol:         EwmaVolatility::default_lambda(),
            mom_micro:   RollingMomentum::new(30),
            mom_short:   RollingMomentum::new(300),
            velocity:    TickVelocity::new(30),
            ofi_30:      OrderFlowImbalance::new(30),
            ofi_300:     OrderFlowImbalance::new(300),
            autocorr:    ReturnAutocorr::new(60),
            realised_30: RealisedVol::new(30),
            spread:      InterExchangeSpread::new(),
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

        FeatureVector {
            ts_micros:             tick.ts_micros,
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
