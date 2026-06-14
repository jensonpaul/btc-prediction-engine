//! Prediction models.
//!
//! # Built-in models
//!
//! | Model | Output | Notes |
//! |---|---|---|
//! | [`MultiScaleTrendModel`] | `[TrendSignal; 4]` | EMA crossover × 4 scales |
//! | [`HeuristicDirectionClassifier`] | `TrendSignal` | RSI + OFI + momentum composite |
//! | [`MomentumExtrapolator`] | `ShortTermForecast` | Drift × √t extrapolation |
//! | [`SignalFuser`] | `(TrendDirection, f64)` | Confidence-weighted vote |
//!
//! # Plugging in trained models
//!
//! Implement [`TrendModelExt`] or [`ForecastModelExt`] and pass via
//! [`crate::engine::EngineConfig`].
//!
//! ## Training feature vector
//!
//! The 28 features in [`crate::features::FeatureVector`], all time-horizon
//! aligned and scale-normalised.  See `btc-model-trainer/src/main.rs`
//! (`FEATURE_NAMES` / `feature_array`) for the exact ordering written to disk.
//!
//! ```text
//!  1  return_5s        – log-return over last 5 s
//!  2  return_30s       – log-return over last 30 s
//!  3  return_300s      – log-return over last 300 s
//!  4  vol_30s          – realised vol (std of log-returns), 30 s
//!  5  vol_300s         – realised vol, 300 s
//!  6  vol_1800s        – realised vol, 1800 s (regime reference)
//!  7  vol_ratio        – vol_30s / vol_1800s
//!  8  ofi_5s           – order-flow imbalance, 5 s
//!  9  ofi_30s          – order-flow imbalance, 30 s
//! 10  ofi_300s         – order-flow imbalance, 300 s
//! 11  ofi_delta_30s    – ofi_5s − ofi_30s
//! 12  buy_ratio_30s    – buy_vol / total_vol, 30 s
//! 13  buy_ratio_300s   – buy_vol / total_vol, 300 s
//! 14  vwap_dev_30s     – (price − VWAP_30s) / VWAP_30s / vol_1800s
//! 15  vwap_dev_300s    – (price − VWAP_300s) / VWAP_300s / vol_1800s
//! 16  volume_ratio     – volume_30s / volume_300s
//! 17  tick_velocity    – rolling tick rate, 30 s window (ticks/s / 20)
//! 18  activity_regime  – tick_rate_30s / tick_rate_1800s
//! 19  spread_pct       – cross-exchange (max−min) / mid
//! 20  book_imb5        – top-5 bid/ask volume imbalance ∈ [−1, 1]
//! 21  book_imb_full    – full-depth bid/ask imbalance ∈ [−1, 1]
//! 22  book_spread_pct  – best bid-ask spread / mid
//! 23  book_pressure    – micro-price deviation from mid
//! 24  trend_strength   – abs(return_300s) / vol_1800s
//! 25  vol_regime       – vol_300s / vol_1800s
//! 26  zreturn_30s      – return_30s / vol_1800s
//! 27  zreturn_300s     – return_300s / vol_1800s
//! ```
//!
//! ## Recommended upgrade path
//!
//! 1. Collect `(feature_array, label)` pairs from a historical replay
//!    by injecting ticks via [`crate::engine::PredictionEngine::inject_tick`].
//! 2. Label: `return_5min > +0.1%` → Bullish(2), `< -0.1%` → Bearish(0), else Sideways(1).
//! 3. Walk-forward train/validate (never random split).
//! 4. Export to ONNX (`torch.onnx.export`) or XGBoost JSON.
//! 5. Load with [`tract-onnx`](https://crates.io/crates/tract-onnx) or
//!    [`xgboost`](https://crates.io/crates/xgboost) and wrap in `TrendModelExt`.

use crate::features::FeatureVector;
use crate::types::{ShortTermForecast, TimeScale, TrendDirection, TrendSignal};

// ─── Extension traits ────────────────────────────────────────────────────────

/// Pluggable trend-direction model.
pub trait TrendModelExt: Send + Sync {
    fn predict(&self, features: &FeatureVector, scale: TimeScale) -> TrendSignal;
}

/// Pluggable short-term price forecast model.
pub trait ForecastModelExt: Send + Sync {
    fn forecast(&self, features: &FeatureVector, step_secs: u32, n_steps: usize) -> ShortTermForecast;
}

// ─── EMA helper ──────────────────────────────────────────────────────────────

pub struct Ema { alpha: f64, value: Option<f64> }

impl Ema {
    pub fn new(period: usize) -> Self { Self { alpha: 2.0 / (period as f64 + 1.0), value: None } }
    pub fn update(&mut self, p: f64) -> f64 {
        let v = self.value.map_or(p, |v| v + self.alpha * (p - v));
        self.value = Some(v); v
    }
    pub fn value(&self) -> Option<f64> { self.value }
}

// ─── Multi-scale trend model ──────────────────────────────────────────────────

/// Maintains four EMA pairs simultaneously.
///
/// | Scale | Fast | Slow | Mom window |
/// |---|---|---|---|
/// | Micro  | 5  | 20  | 10  |
/// | Short  | 12 | 60  | 50  |
/// | Medium | 20 | 200 | 150 |
/// | Broad  | 50 | 500 | 400 |
pub struct MultiScaleTrendModel {
    micro_f: Ema, micro_s: Ema,
    short_f: Ema, short_s: Ema,
    med_f:   Ema, med_s:   Ema,
    broad_f: Ema, broad_s: Ema,
}

impl MultiScaleTrendModel {
    pub fn new() -> Self {
        Self {
            micro_f: Ema::new(5),   micro_s: Ema::new(20),
            short_f: Ema::new(12),  short_s: Ema::new(60),
            med_f:   Ema::new(20),  med_s:   Ema::new(200),
            broad_f: Ema::new(50),  broad_s: Ema::new(500),
        }
    }

    pub fn update(&mut self, f: &FeatureVector) -> [TrendSignal; 4] {
        let p  = f.price;
        let ts = f.ts_micros;
        let mf = self.micro_f.update(p);  let ms = self.micro_s.update(p);
        let sf = self.short_f.update(p);  let ss = self.short_s.update(p);
        let ef = self.med_f.update(p);    let es = self.med_s.update(p);
        let bf = self.broad_f.update(p);  let bs = self.broad_s.update(p);
        [
            Self::crossover(mf, ms, f.return_5s,   TimeScale::Micro,  ts),
            Self::crossover(sf, ss, f.return_30s,  TimeScale::Short,  ts),
            Self::crossover(ef, es, f.return_30s,  TimeScale::Medium, ts),
            Self::crossover(bf, bs, f.return_300s, TimeScale::Broad,  ts),
        ]
    }

    fn crossover(fast: f64, slow: f64, mom: Option<f64>, scale: TimeScale, ts: i64) -> TrendSignal {
        let spread = if slow != 0.0 { (fast - slow) / slow } else { 0.0 };
        let thr = 0.0002;
        let direction = if spread > thr { TrendDirection::Bullish }
                        else if spread < -thr { TrendDirection::Bearish }
                        else { TrendDirection::Sideways };
        let raw_conf = (spread.abs() / 0.005).tanh();
        let conf = match mom {
            Some(m) if (spread > 0.0) == (m > 0.0) => (raw_conf * 1.2).min(1.0),
            Some(_) => raw_conf * 0.7,
            None    => raw_conf,
        };
        TrendSignal { direction, confidence: conf.clamp(0.0, 1.0), scale, computed_at: ts }
    }
}

impl Default for MultiScaleTrendModel { fn default() -> Self { Self::new() } }

// ─── Heuristic direction classifier ─────────────────────────────────────────

/// Built-in baseline — composite of RSI, momentum, OFI, VWAP deviation,
/// autocorrelation, and inter-exchange spread.
///
/// Replace with a trained model via [`TrendModelExt`] for production use.
pub struct HeuristicDirectionClassifier;

impl HeuristicDirectionClassifier {
    pub fn predict(f: &FeatureVector, scale: TimeScale) -> TrendSignal {
        let ts = f.ts_micros;

        // Momentum signal — use time-aligned returns matched to the scale.
        // z-scored returns are preferred when vol_1800s is available because
        // they normalise across price regimes; fall back to raw log-returns.
        let mom_score = match scale {
            TimeScale::Micro => f.zreturn_30s
                .or(f.return_5s)
                .unwrap_or(0.0),
            TimeScale::Short => f.zreturn_30s
                .or(f.return_30s)
                .unwrap_or(0.0),
            _ => f.zreturn_300s
                .or(f.return_300s)
                .unwrap_or(0.0),
        }.clamp(-3.0, 3.0) / 3.0; // normalise z-score to [−1, 1]

        // Order-flow signal.
        let ofi_score = match scale {
            TimeScale::Micro => f.ofi_5s,
            TimeScale::Short => f.ofi_30s,
            _                => f.ofi_300s,
        };

        // VWAP deviation — already z-scored; clamp to ±3σ → [−1, 1].
        let vwap_score = match scale {
            TimeScale::Micro | TimeScale::Short =>
                f.vwap_dev_30s.unwrap_or(0.0).clamp(-3.0, 3.0) / 3.0,
            _ =>
                f.vwap_dev_300s.unwrap_or(0.0).clamp(-3.0, 3.0) / 3.0,
        };

        // Book pressure: positive = bid side dominates → bullish lean.
        let book_score = f.book_pressure.unwrap_or(0.0).clamp(-0.01, 0.01) * 100.0;

        // Volatility expansion dampens confidence — when vol_ratio >> 1 the
        // market is spiking and short signals are less reliable.
        let vol_dampen = f.vol_ratio.map(|vr| {
            // Smoothly reduce to 0.5× at vol_ratio = 3.0.
            (1.0 - (vr - 1.0).max(0.0) / 4.0).max(0.5)
        }).unwrap_or(1.0);

        // High cross-exchange spread → uncertainty, dampen further.
        let spread_dampen = if f.spread_pct > 0.001 { 0.8 } else { 1.0 };

        let composite = (0.35 * mom_score
            + 0.30 * ofi_score
            + 0.20 * vwap_score
            + 0.15 * book_score)
            * vol_dampen
            * spread_dampen;

        let thr = 0.1;
        let direction = if composite > thr { TrendDirection::Bullish }
                        else if composite < -thr { TrendDirection::Bearish }
                        else { TrendDirection::Sideways };

        TrendSignal { direction, confidence: composite.abs().min(1.0), scale, computed_at: ts }
    }
}

// ─── Momentum extrapolator ───────────────────────────────────────────────────

/// Heuristic short-term forecast — replace with LSTM via [`ForecastModelExt`].
///
/// ## LSTM upgrade path
///
/// ```text
/// Training:
///   Input:  last 60 ticks × [price_delta, volume, rsi_14, ofi_30s, ofi_300s]
///   Output: [delta_5s, delta_10s, delta_30s, delta_60s, delta_300s]
///   Architecture: Bi-LSTM(hidden=128, layers=2, dropout=0.2)
///   Export: torch.onnx.export → load with tract-onnx
/// ```
pub struct MomentumExtrapolator;

impl MomentumExtrapolator {
    pub fn forecast(f: &FeatureVector, step_secs: u32, n_steps: usize) -> ShortTermForecast {
        // Drift: use the 5s return as the instantaneous velocity estimate.
        // Divide by 5 to get a per-second rate, then multiply by price for USD/s.
        let drift = f.return_5s.unwrap_or(0.0) * f.price / 5.0; // USD/s

        // Per-step volatility: use vol_30s (closest horizon to typical step sizes);
        // fall back to vol_1800s, then a hard floor.
        let vol_ref = f.vol_30s
            .or(f.vol_1800s)
            .unwrap_or(0.001);
        let ticks_per_step = (step_secs as f64 * f.tick_velocity.max(1.0)).max(1.0);
        let vol_per_step = vol_ref * ticks_per_step.sqrt() * f.price;

        let decay = 60.0_f64;
        let deltas: Vec<f64>     = (1..=n_steps).map(|i| drift * step_secs as f64 * i as f64).collect();
        let confidence: Vec<f64> = (1..=n_steps).map(|i| {
            let t    = step_secs as f64 * i as f64;
            let conf = (-t / decay).exp();
            let snr  = if vol_per_step > 0.0 {
                (drift.abs() * step_secs as f64 / vol_per_step).min(1.0)
            } else { 0.5 };
            (conf * snr).clamp(0.0, 1.0)
        }).collect();

        ShortTermForecast { step_secs, deltas, confidence, generated_at: f.ts_micros }
    }
}

// ─── Signal fuser ────────────────────────────────────────────────────────────

/// Combines four scale signals into a single fused direction.
///
/// Weights: micro=0.15, short=0.35, medium=0.30, broad=0.20.
/// A broad+medium conflict with micro+short dampens the final confidence.
pub struct SignalFuser;

impl SignalFuser {
    const W: [f64; 4] = [0.15, 0.35, 0.30, 0.20];

    pub fn fuse(signals: &[TrendSignal; 4]) -> (TrendDirection, f64) {
        let (mut bull, mut bear, mut side) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (s, &w) in signals.iter().zip(Self::W.iter()) {
            let wc = w * s.confidence;
            match s.direction {
                TrendDirection::Bullish  => bull += wc,
                TrendDirection::Bearish  => bear += wc,
                TrendDirection::Sideways => side += wc,
            }
        }
        let total = bull + bear + side;
        if total == 0.0 { return (TrendDirection::Sideways, 0.0); }

        // Dampen when broad/medium disagree with micro/short
        let broad_short_conflict = signals[3].direction != signals[1].direction
            && signals[3].direction != TrendDirection::Sideways;
        let med_short_conflict   = signals[2].direction != signals[1].direction
            && signals[2].direction != TrendDirection::Sideways;
        let dampen = if broad_short_conflict && med_short_conflict { 0.6 }
                     else if broad_short_conflict || med_short_conflict { 0.85 }
                     else { 1.0 };

        let (dir, score) = if bull >= bear && bull >= side { (TrendDirection::Bullish, bull) }
                           else if bear >= side { (TrendDirection::Bearish, bear) }
                           else { (TrendDirection::Sideways, side) };
        (dir, (score / total * dampen).clamp(0.0, 1.0))
    }
}

// ─── Blend helper ────────────────────────────────────────────────────────────

/// Blend two signals with weight `wa` for `a`, `1 - wa` for `b`.
pub fn blend(a: &TrendSignal, b: &TrendSignal, wa: f64) -> TrendSignal {
    let wb = 1.0 - wa;
    let (mut bull, mut bear, mut side) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (s, w) in [(&a, wa), (&b, wb)] {
        let wc = w * s.confidence;
        match s.direction {
            TrendDirection::Bullish  => bull += wc,
            TrendDirection::Bearish  => bear += wc,
            TrendDirection::Sideways => side += wc,
        }
    }
    let total = bull + bear + side;
    let (dir, sc) = if bull >= bear && bull >= side { (TrendDirection::Bullish,  bull) }
                    else if bear >= side            { (TrendDirection::Bearish,  bear) }
                    else                            { (TrendDirection::Sideways, side) };
    TrendSignal {
        direction:   dir,
        confidence:  if total > 0.0 { (sc / total).clamp(0.0, 1.0) } else { 0.0 },
        scale:       a.scale,
        computed_at: a.computed_at,
    }
}
