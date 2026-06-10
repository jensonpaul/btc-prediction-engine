//! Core prediction engine — public entry point.
//!
//! [`PredictionEngine`] wires together all subsystems:
//!
//! * Spawns exchange feeds as independent tokio tasks.
//! * Owns the [`Pipeline`] (4 concurrent async stages).
//! * Exposes [`TickStore`] and [`PredictionStore`] for direct queries.
//! * Provides a zero-lock [`latest_snapshot()`] via [`ArcSwap`].
//! * Broadcasts every new [`PredictionSnapshot`] to N concurrent subscribers.
//!
//! # Usage
//!
//! ```rust,no_run
//! use btc_prediction_engine::prelude::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let (engine, _handles) = PredictionEngine::start(EngineConfig::default()).await;
//!
//!     // Attach feeds — synchronous, each spawns a background task
//!     engine.add_feed(FeedConfig::public(Exchange::Binance,  Symbol::BtcUsd));
//!     engine.add_feed(FeedConfig::public(Exchange::Kraken,   Symbol::BtcUsd));
//!     engine.add_feed(FeedConfig::public(Exchange::Bitstamp, Symbol::BtcUsd));
//!
//!     // Subscribe to push updates
//!     let mut rx = engine.subscribe();
//!     tokio::spawn(async move {
//!         while let Ok(snap) = rx.recv().await {
//!             println!("{:?} @ ${:.2} conf={:.2}", snap.fused_direction, snap.price, snap.fused_confidence);
//!         }
//!     });
//!
//!     // Zero-lock latest read from any thread
//!     if let Some(snap) = engine.latest_snapshot() {
//!         println!("current price: ${:.2}", snap.price);
//!     }
//! }
//! ```
//!
//! # Shutdown
//!
//! Drop the [`PredictionEngine`] handle — all feed tasks observe a closed
//! channel and terminate cleanly.  Call [`PredictionEngine::shutdown()`]
//! for a graceful await on all task handles.

use std::sync::Arc;
use arc_swap::ArcSwap;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use crate::feeds::FeedConfig;
use crate::pipeline::{Pipeline, PipelineConfig, PipelineHandles};
use crate::types::{
    Exchange, FeedHealth, PredictionSnapshot, Symbol, Tick, TickStore, PredictionStore,
};
use crate::types::tick_store;
use crate::types::pred_store;

// ─── Engine configuration ─────────────────────────────────────────────────────

/// Full engine configuration.
pub struct EngineConfig {
    /// Tick ring capacity.
    /// Default: [`tick_store::DEFAULT_CAPACITY`] ≈ 28 h at 15 ticks/s.
    pub tick_capacity:  usize,
    /// Prediction ring capacity.
    pub pred_capacity:  usize,
    /// Pipeline configuration (dedup, outlier filter, models, forecasts).
    pub pipeline:       PipelineConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            tick_capacity: tick_store::DEFAULT_CAPACITY,
            pred_capacity: pred_store::DEFAULT_PRED_CAPACITY,
            pipeline:      PipelineConfig::default(),
        }
    }
}

// ─── All join handles ────────────────────────────────────────────────────────

/// All background task handles returned by [`PredictionEngine::start`].
pub struct EngineHandles {
    pub pipeline: PipelineHandles,
    /// Feed task handles, one per [`add_feed`] call.
    pub feeds:    parking_lot::Mutex<Vec<JoinHandle<()>>>,
}

impl EngineHandles {
    /// Await all tasks to completion (call after dropping the engine handle).
    pub async fn join_all(self) {
        let (p, feeds) = (self.pipeline, self.feeds.into_inner());
        let _ = tokio::join!(p.dedup, p.fusion, p.feature, p.model, p.fanout);
        for h in feeds { let _ = h.await; }
    }
}

// ─── Engine handle ───────────────────────────────────────────────────────────

/// The prediction engine handle. `Clone + Send + Sync`.
///
/// Cloning gives a second handle to the **same** engine — no new state is
/// created.
#[derive(Clone)]
pub struct PredictionEngine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    tick_store:   Arc<TickStore>,
    pred_store:   Arc<PredictionStore>,
    /// Zero-lock latest snapshot.
    latest:       Arc<ArcSwap<Option<PredictionSnapshot>>>,
    pipeline:     Pipeline,
}

impl PredictionEngine {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Start the engine. Returns a handle and all background task handles.
    pub async fn start(config: EngineConfig) -> (Self, EngineHandles) {
        let tick_store  = Arc::new(TickStore::with_capacity(config.tick_capacity));
        let pred_store  = Arc::new(PredictionStore::with_capacity(config.pred_capacity));
        let latest      = Arc::new(ArcSwap::from_pointee(None::<PredictionSnapshot>));

        let (pipeline, pipeline_handles) = Pipeline::spawn(
            config.pipeline,
            Arc::clone(&tick_store),
            Arc::clone(&pred_store),
            Arc::clone(&latest),
        );

        let engine = Self {
            inner: Arc::new(EngineInner { tick_store, pred_store, latest, pipeline }),
        };

        let handles = EngineHandles {
            pipeline: pipeline_handles,
            feeds:    parking_lot::Mutex::new(Vec::new()),
        };

        (engine, handles)
    }

    // ── Feed management ───────────────────────────────────────────────────────

    /// Spawn an exchange feed.
    ///
    /// Each feed runs as an independent tokio task with its own reconnect loop.
    /// Feeds communicate with the engine via a bounded `mpsc` channel — they
    /// never block each other or the engine loop.
    pub fn add_feed(&self, config: FeedConfig) -> JoinHandle<()> {
        let tx       = self.inner.pipeline.raw_tx.clone();
        let health   = self.inner.pipeline.health.clone();
        let exchange = config.exchange;

        tokio::spawn(async move {
            health.record_tick(exchange); // mark as known
            let result = match exchange {
                #[cfg(feature = "feeds-binance")]
                Exchange::Binance  => crate::feeds::binance::run(config, tx).await,
                #[cfg(feature = "feeds-coinbase")]
                Exchange::Coinbase => crate::feeds::coinbase::run(config, tx).await,
                #[cfg(feature = "feeds-kraken")]
                Exchange::Kraken   => crate::feeds::kraken::run(config, tx).await,
                #[cfg(feature = "feeds-bitstamp")]
                Exchange::Bitstamp => crate::feeds::bitstamp::run(config, tx).await,
                #[allow(unreachable_patterns)]
                _ => Err(crate::types::EngineError::Other(
                    anyhow::anyhow!("feed not compiled in: {exchange}")
                )),
            };
            if let Err(e) = result {
                crate::feeds::log_error(exchange, &format!("feed terminated: {e}"));
            }
        })
    }

    /// Inject a tick directly (for testing or historical replay).
    ///
    /// Uses `try_send` — returns `false` if the channel is full.
    pub fn inject_tick(&self, tick: Tick) -> bool {
        self.inner.pipeline.raw_tx.try_send(tick).is_ok()
    }

    // ── Subscription ─────────────────────────────────────────────────────────

    /// Subscribe to prediction snapshots.
    ///
    /// Each call returns an independent receiver.  Slow receivers receive
    /// [`tokio::sync::broadcast::error::RecvError::Lagged`] when they fall
    /// behind rather than blocking the pipeline.
    pub fn subscribe(&self) -> broadcast::Receiver<PredictionSnapshot> {
        self.inner.pipeline.broadcast_tx.subscribe()
    }

    // ── Zero-lock reads ───────────────────────────────────────────────────────

    /// Latest prediction snapshot — **single atomic load, no lock**.
    ///
    /// Safe to call from any thread at any frequency.
    #[inline]
    pub fn latest_snapshot(&self) -> Option<PredictionSnapshot> {
        self.inner.latest.load().as_ref().clone()
    }

    /// Latest BTC/USD price — **zero lock**.
    #[inline]
    pub fn latest_price(&self) -> Option<f64> {
        self.inner.tick_store.latest().map(|t| t.price)
    }

    // ── Store access ─────────────────────────────────────────────────────────

    pub fn tick_store(&self) -> &Arc<TickStore>       { &self.inner.tick_store }
    pub fn pred_store(&self) -> &Arc<PredictionStore> { &self.inner.pred_store }

    pub fn tick_count(&self)       -> usize { self.inner.tick_store.len() }
    pub fn prediction_count(&self) -> usize { self.inner.pred_store.len() }

    /// Feed health map (read lock on internal HashMap).
    pub fn connected_feeds(&self) -> u32 {
        self.inner.pipeline.health.connected_count()
    }

    // ── Graceful shutdown ────────────────────────────────────────────────────

    /// Drop the raw_tx sender — all pipeline stages drain and exit cleanly.
    ///
    /// This is called automatically when the last `PredictionEngine` clone is
    /// dropped.  Call explicitly if you need to await full drain.
    pub fn shutdown_signal(&self) -> broadcast::Sender<PredictionSnapshot> {
        self.inner.pipeline.broadcast_tx.clone()
    }
}
