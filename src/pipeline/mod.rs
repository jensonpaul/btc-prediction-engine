//! Async non-blocking pipeline.
//!
//! The hot path is split into four independent tokio tasks connected by
//! bounded `mpsc` channels.  Each stage runs concurrently and never blocks
//! any other stage.
//!
//! ```text
//! feeds (N tasks)
//!   │  mpsc::Sender<Tick>         (raw_tx / raw_rx)
//!   ▼
//! [Stage 1  ] Dedup + outlier filter   — drops duplicates and spikes
//!   │  mpsc::Sender<Tick>         (clean_tx / clean_rx)
//!   ▼
//! [Stage 1.5] PriceFuser               — NTP drift correction + 100 ms VWAP fusion
//!   │  mpsc::Sender<FusedTick>    (fused_tx / fused_rx)
//!   ▼
//! [Stage 2  ] Feature engineering      — updates FeatureState from FusedTick
//!   │  mpsc::Sender<FeatureVector> (feat_tx / feat_rx)
//!   ▼
//! [Stage 3  ] Model inference          — runs all models in parallel
//!   │  mpsc::Sender<PredictionSnapshot> (snap_tx / snap_rx)
//!   ▼
//! [Stage 4  ] Fanout                   — PredStore, ArcSwap, broadcast
//! ```
//!
//! # Back-pressure
//!
//! All channels are bounded.  If a downstream stage falls behind, sends will
//! return `Err` and the tick is dropped with an incremented counter rather
//! than blocking the upstream feed.  This keeps feeds non-blocking at all times.
//!
//! # Parallelism
//!
//! Stage 3 (model inference) runs all four [`TimeScale`] predictions and all
//! forecast steps concurrently using `tokio::join!`.  Each model call is a
//! pure synchronous computation wrapped in `tokio::task::spawn_blocking` if it
//! is CPU-heavy (the built-in heuristics are fast enough to run inline).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;

use crate::dedup::{FilterConfig, TickFilter};
use crate::features::{FeatureState, FeatureVector};
use crate::models::{
    blend, ForecastModelExt, HeuristicDirectionClassifier, MomentumExtrapolator,
    MultiScaleTrendModel, SignalFuser, TrendModelExt,
};
use crate::price_fusion::{run_fuser, FusedTick, FusionConfig};
use crate::types::{
    EngineMetrics, Exchange, FeedHealth, PredictionSnapshot, ShortTermForecast,
    TimeScale, Tick, TickStore, TrendDirection, TrendSignal, PredictionStore,
};

// ─── Channel capacities ───────────────────────────────────────────────────────

/// Raw ticks from all feeds → dedup stage.
pub const RAW_CHANNEL_CAP: usize = 65_536;
/// Clean ticks (post-dedup) → price fusion stage.
pub const CLEAN_CHANNEL_CAP: usize = 32_768;
/// Fused ticks (post-fusion) → feature stage.
pub const FUSED_CHANNEL_CAP: usize = 16_384;
/// Feature vectors → model stage.
pub const FEAT_CHANNEL_CAP: usize = 8_192;
/// Snapshots → fanout stage.
pub const SNAP_CHANNEL_CAP: usize = 4_096;

// ─── Pipeline configuration ──────────────────────────────────────────────────

pub struct PipelineConfig {
    pub filter:          FilterConfig,
    /// Price fusion configuration (NTP correction + 100 ms bucket VWAP).
    pub fusion:          FusionConfig,
    pub forecast_steps:  Vec<(u32, usize)>,
    pub ext_trend:       Option<Box<dyn TrendModelExt>>,
    pub ext_forecast:    Option<Box<dyn ForecastModelExt>>,
    pub broadcast_cap:   usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            filter:         FilterConfig::default(),
            fusion:         FusionConfig::default(),
            forecast_steps: vec![(5, 6), (30, 10)],
            ext_trend:      None,
            ext_forecast:   None,
            broadcast_cap:  4_096,
        }
    }
}

// ─── Per-feed health tracker ─────────────────────────────────────────────────

#[derive(Clone)]
pub struct FeedHealthTracker {
    inner: Arc<parking_lot::RwLock<HashMap<Exchange, FeedHealthState>>>,
}

struct FeedHealthState {
    health:         FeedHealth,
    last_tick_at:   Option<Instant>,
    ticks_this_sec: u64,
}

impl FeedHealthTracker {
    pub fn new() -> Self {
        Self { inner: Arc::new(parking_lot::RwLock::new(HashMap::new())) }
    }

    pub fn record_tick(&self, exchange: Exchange) {
        let mut g = self.inner.write();
        let s = g.entry(exchange).or_insert(FeedHealthState {
            health: FeedHealth::Healthy, last_tick_at: None, ticks_this_sec: 0,
        });
        s.last_tick_at   = Some(Instant::now());
        s.health         = FeedHealth::Healthy;
        s.ticks_this_sec += 1;
    }

    pub fn connected_count(&self) -> u32 {
        let g = self.inner.read();
        g.values().filter(|s| s.health == FeedHealth::Healthy).count() as u32
    }

    /// Sweep stale feeds — called periodically by the fanout stage.
    pub fn sweep_stale(&self) {
        let threshold = Duration::from_secs(10);
        let mut g = self.inner.write();
        for s in g.values_mut() {
            if let Some(last) = s.last_tick_at {
                if last.elapsed() > threshold && s.health == FeedHealth::Healthy {
                    s.health = FeedHealth::Stale;
                }
            }
        }
    }
}

// ─── Metrics accumulator ─────────────────────────────────────────────────────

struct MetricsAcc {
    tick_timestamps:   std::collections::VecDeque<Instant>,
    dedup_last_sec:    u64,
    outlier_last_sec:  u64,
    dedup_acc:         u64,
    outlier_acc:       u64,
    last_reset:        Instant,
}

impl MetricsAcc {
    fn new() -> Self {
        Self {
            tick_timestamps:  std::collections::VecDeque::new(),
            dedup_last_sec:   0,
            outlier_last_sec: 0,
            dedup_acc:        0,
            outlier_acc:      0,
            last_reset:       Instant::now(),
        }
    }

    fn record_tick(&mut self) {
        let now = Instant::now();
        self.tick_timestamps.push_back(now);
        // Evict timestamps older than 10 s
        while self.tick_timestamps.front().map_or(false, |t| now.duration_since(*t) > Duration::from_secs(10)) {
            self.tick_timestamps.pop_front();
        }
    }

    fn record_dedup(&mut self)   { self.dedup_acc   += 1; }
    fn record_outlier(&mut self) { self.outlier_acc += 1; }

    fn flush_if_needed(&mut self) {
        if self.last_reset.elapsed() >= Duration::from_secs(1) {
            self.dedup_last_sec   = self.dedup_acc;
            self.outlier_last_sec = self.outlier_acc;
            self.dedup_acc        = 0;
            self.outlier_acc      = 0;
            self.last_reset       = Instant::now();
        }
    }

    fn tick_rate(&self) -> f64 {
        let n = self.tick_timestamps.len();
        if n < 2 { return 0.0; }
        let span = self.tick_timestamps.back().unwrap().duration_since(*self.tick_timestamps.front().unwrap());
        if span.as_secs_f64() == 0.0 { 0.0 } else { (n as f64 - 1.0) / span.as_secs_f64() }
    }
}

// ─── Pipeline ────────────────────────────────────────────────────────────────

/// All task join handles produced by [`Pipeline::spawn`].
pub struct PipelineHandles {
    pub dedup:   JoinHandle<()>,
    /// Stage 1.5 — NTP drift correction + 100 ms VWAP price fusion.
    pub fusion:  JoinHandle<()>,
    pub feature: JoinHandle<()>,
    pub model:   JoinHandle<()>,
    pub fanout:  JoinHandle<()>,
}

/// Channels and handles for the live pipeline.
pub struct Pipeline {
    /// Send raw ticks from feeds into the pipeline.
    pub raw_tx:       mpsc::Sender<Tick>,
    /// Subscribe to prediction snapshots.
    pub broadcast_tx: broadcast::Sender<PredictionSnapshot>,
    pub health:       FeedHealthTracker,
}

impl Pipeline {
    /// Spawn all four pipeline stages as independent tokio tasks.
    /// Returns a [`Pipeline`] handle and the join handles for all stages.
    pub fn spawn(
        config:      PipelineConfig,
        tick_store:  Arc<TickStore>,
        pred_store:  Arc<PredictionStore>,
        latest_snap: Arc<arc_swap::ArcSwap<Option<PredictionSnapshot>>>,
    ) -> (Self, PipelineHandles) {
        let (raw_tx,   raw_rx)   = mpsc::channel::<Tick>(RAW_CHANNEL_CAP);
        let (clean_tx, clean_rx) = mpsc::channel::<Tick>(CLEAN_CHANNEL_CAP);
        let (fused_tx, fused_rx) = mpsc::channel::<FusedTick>(FUSED_CHANNEL_CAP);
        let (feat_tx,  feat_rx)  = mpsc::channel::<FeatureVector>(FEAT_CHANNEL_CAP);
        let (snap_tx,  snap_rx)  = mpsc::channel::<PredictionSnapshot>(SNAP_CHANNEL_CAP);
        let (bcast_tx, _)        = broadcast::channel::<PredictionSnapshot>(config.broadcast_cap);

        let health = FeedHealthTracker::new();

        // ── Stage 1: dedup + outlier ─────────────────────────────────────────
        let dedup_handle = {
            let filter_cfg = config.filter.clone();
            tokio::spawn(async move {
                stage_dedup(raw_rx, clean_tx, filter_cfg).await;
            })
        };

        // ── Stage 1.5: price fusion (NTP + 100 ms VWAP) ─────────────────────
        let fusion_handle = {
            let fusion_cfg = config.fusion.clone();
            tokio::spawn(async move {
                run_fuser(fusion_cfg, clean_rx, fused_tx).await;
            })
        };

        // ── Stage 2: feature engineering ────────────────────────────────────
        let feature_handle = {
            tokio::spawn(async move {
                stage_features(fused_rx, feat_tx).await;
            })
        };

        // ── Stage 3: model inference (parallel) ─────────────────────────────
        let model_handle = {
            let steps        = config.forecast_steps.clone();
            let ext_trend    = config.ext_trend.map(Arc::from);
            let ext_forecast = config.ext_forecast.map(Arc::from);
            let health_clone = health.clone();
            tokio::spawn(async move {
                stage_models(feat_rx, snap_tx, steps, ext_trend, ext_forecast, health_clone).await;
            })
        };

        // ── Stage 4: fanout ──────────────────────────────────────────────────
        let fanout_handle = {
            let tick_store   = Arc::clone(&tick_store);
            let pred_store   = Arc::clone(&pred_store);
            let latest_snap  = Arc::clone(&latest_snap);
            let bcast_tx     = bcast_tx.clone();
            let health_clone = health.clone();
            tokio::spawn(async move {
                stage_fanout(snap_rx, tick_store, pred_store, latest_snap, bcast_tx, health_clone).await;
            })
        };

        (
            Self { raw_tx, broadcast_tx: bcast_tx, health },
            PipelineHandles {
                dedup:   dedup_handle,
                fusion:  fusion_handle,
                feature: feature_handle,
                model:   model_handle,
                fanout:  fanout_handle,
            },
        )
    }
}

// ─── Stage 1: Dedup + outlier filter ─────────────────────────────────────────

async fn stage_dedup(
    mut rx: mpsc::Receiver<Tick>,
    tx:     mpsc::Sender<Tick>,
    cfg:    FilterConfig,
) {
    let mut filter  = TickFilter::new(cfg);
    let mut metrics = MetricsAcc::new();

    while let Some(tick) = rx.recv().await {
        metrics.flush_if_needed();

        if !filter.accept(&tick) {
            if filter.dedup_count > metrics.dedup_acc {
                metrics.record_dedup();
            } else {
                metrics.record_outlier();
            }
            continue;
        }

        metrics.record_tick();
        // Non-blocking send — drop tick on back-pressure rather than blocking
        if tx.try_send(tick).is_err() {
            // Channel full; tick dropped. Increment a counter if metrics enabled.
            #[cfg(feature = "metrics")]
            metrics::counter!("btc_engine_pipeline_drops_total", "stage" => "dedup").increment(1);
        }
    }
}

// ─── Stage 2: Feature engineering ────────────────────────────────────────────

async fn stage_features(
    mut rx: mpsc::Receiver<FusedTick>,
    tx:     mpsc::Sender<FeatureVector>,
) {
    let mut state = FeatureState::new();

    while let Some(fused) = rx.recv().await {
        let fv = state.update_from_fused(&fused);

        if tx.try_send(fv).is_err() {
            #[cfg(feature = "metrics")]
            metrics::counter!("btc_engine_pipeline_drops_total", "stage" => "features").increment(1);
        }
    }
}

// ─── Stage 3: Model inference ────────────────────────────────────────────────

async fn stage_models(
    mut rx:          mpsc::Receiver<FeatureVector>,
    tx:              mpsc::Sender<PredictionSnapshot>,
    forecast_steps:  Vec<(u32, usize)>,
    ext_trend:       Option<Arc<dyn TrendModelExt>>,
    ext_forecast:    Option<Arc<dyn ForecastModelExt>>,
    health:          FeedHealthTracker,
) {
    let mut ms_model = MultiScaleTrendModel::new();

    while let Some(fv) = rx.recv().await {
        // ── EMA multi-scale signals (synchronous, O(1) each) ─────────────────
        let ms_signals = ms_model.update(&fv);

        // ── Direction signals — run per-scale in parallel ─────────────────────
        let signals: [TrendSignal; 4] = if let Some(ext) = &ext_trend {
            let (s0, s1, s2, s3) = tokio::join!(
                async { ext.predict(&fv, TimeScale::Micro) },
                async { ext.predict(&fv, TimeScale::Short) },
                async { ext.predict(&fv, TimeScale::Medium) },
                async { ext.predict(&fv, TimeScale::Broad) },
            );
            [s0, s1, s2, s3]
        } else {
            let heuristic_short = HeuristicDirectionClassifier::predict(&fv, TimeScale::Short);
            let blended         = blend(&ms_signals[1], &heuristic_short, 0.6);
            [ms_signals[0].clone(), blended, ms_signals[2].clone(), ms_signals[3].clone()]
        };

        // ── Fuse ─────────────────────────────────────────────────────────────
        let (fused_direction, fused_confidence) = SignalFuser::fuse(&signals);

        // ── Forecasts — all step sizes in parallel ───────────────────────────
        let forecasts: HashMap<u32, ShortTermForecast> = {
            let fv_ref       = &fv;
            let steps_ref    = &forecast_steps;
            let ext_fc_ref   = &ext_forecast;

            // Build all forecasts concurrently using FuturesUnordered
            use futures_util::stream::{FuturesUnordered, StreamExt};
            let mut futs = FuturesUnordered::new();
            for &(step_secs, n_steps) in steps_ref.iter() {
                let fv_clone  = fv_ref.clone();
                let ext_clone = ext_fc_ref.clone();
                futs.push(async move {
                    let fc = if let Some(ext) = ext_clone {
                        ext.forecast(&fv_clone, step_secs, n_steps)
                    } else {
                        MomentumExtrapolator::forecast(&fv_clone, step_secs, n_steps)
                    };
                    (step_secs, fc)
                });
            }
            let mut map = HashMap::with_capacity(steps_ref.len());
            while let Some((k, v)) = futs.next().await { map.insert(k, v); }
            map
        };

        // ── Engine metrics ───────────────────────────────────────────────────
        let engine_metrics = EngineMetrics {
            tick_rate:             0.0, // updated by fanout stage
            dedup_rate:            0.0,
            outlier_rate:          0.0,
            connected_feeds:       health.connected_count(),
            cross_exchange_spread: fv.inter_exchange_spread,
        };

        let snap = PredictionSnapshot {
            snapshot_at:      fv.ts_micros,
            price:            fv.price,
            micro:            signals[0].clone(),
            short:            signals[1].clone(),
            medium:           signals[2].clone(),
            broad:            signals[3].clone(),
            forecasts,
            fused_direction,
            fused_confidence,
            metrics:          engine_metrics,
        };

        if tx.try_send(snap).is_err() {
            #[cfg(feature = "metrics")]
            metrics::counter!("btc_engine_pipeline_drops_total", "stage" => "models").increment(1);
        }
    }
}

// ─── Stage 4: Fanout ─────────────────────────────────────────────────────────

async fn stage_fanout(
    mut rx:       mpsc::Receiver<PredictionSnapshot>,
    tick_store:   Arc<TickStore>,
    pred_store:   Arc<PredictionStore>,
    latest_snap:  Arc<arc_swap::ArcSwap<Option<PredictionSnapshot>>>,
    bcast_tx:     broadcast::Sender<PredictionSnapshot>,
    health:       FeedHealthTracker,
) {
    let mut sweep_tick = tokio::time::interval(Duration::from_secs(10));
    sweep_tick.tick().await;

    loop {
        tokio::select! {
            snap = rx.recv() => {
                let snap = match snap { Some(s) => s, None => break };

                // 1. Push to pred store (write lock, brief)
                pred_store.push(snap.clone());

                // 2. Atomic swap — zero-cost reads for all subsystems
                latest_snap.store(Arc::new(Some(snap.clone())));

                // 3. Broadcast to all subscribers (non-blocking; lagged receivers
                //    get RecvError::Lagged rather than blocking the pipeline)
                let _ = bcast_tx.send(snap);

                #[cfg(feature = "metrics")]
                metrics::counter!("btc_engine_snapshots_total").increment(1);
            }
            _ = sweep_tick.tick() => {
                health.sweep_stale();
            }
        }
    }
}
