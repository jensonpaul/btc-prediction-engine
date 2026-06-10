//! # `btc_prediction_engine` v0.2
//!
//! Continuous, lock-free BTC/USD prediction engine — multi-exchange,
//! multi-scale, fully async non-blocking.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Exchange feeds — 4 independent tokio tasks                     │
//! │  Binance (btcusdt) · Coinbase (BTC-USD) ·                       │
//! │  Kraken (BTC/USD)  · Bitstamp (btcusd)                          │
//! └──────────────────────┬──────────────────────────────────────────┘
//!                        │ mpsc::Sender<Tick>  (bounded, non-blocking)
//!                        ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Pipeline — 4 concurrent async stages                           │
//! │                                                                 │
//! │  [1] Dedup + Outlier filter                                     │
//! │        cross-exchange dedup · spike rejection                   │
//! │        │                                                        │
//! │  [2] Feature engineering                                        │
//! │        RSI · VWAP · OFI · momentum · autocorr · realised vol   │
//! │        inter-exchange spread · tick velocity (all incremental)  │
//! │        │                                                        │
//! │  [3] Model inference  (parallel via tokio::join! / FuturesUnordered) │
//! │        EMA multi-scale · heuristic classifier · signal fuser   │
//! │        momentum extrapolator (or plug-in LSTM / XGBoost)        │
//! │        │                                                        │
//! │  [4] Fanout                                                     │
//! │        ArcSwap<snapshot> · PredictionStore · broadcast channel  │
//! └──────────────────────┬──────────────────────────────────────────┘
//!                        │
//!        ┌───────────────┼───────────────────┐
//!        │               │                   │
//!   zero-lock        broadcast           QueryEngine
//!   ArcSwap load    Receiver<snap>       window / trend / forecast
//! ```
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use btc_prediction_engine::prelude::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let (engine, _handles) = PredictionEngine::start(EngineConfig::default()).await;
//!
//!     // Attach feeds (all public — no API key needed except Coinbase)
//!     engine.add_feed(FeedConfig::public(Exchange::Binance,  Symbol::BtcUsd));
//!     engine.add_feed(FeedConfig::public(Exchange::Kraken,   Symbol::BtcUsd));
//!     engine.add_feed(FeedConfig::public(Exchange::Bitstamp, Symbol::BtcUsd));
//!     // Coinbase: engine.add_feed(FeedConfig::authenticated(Exchange::Coinbase, Symbol::BtcUsd, key, secret));
//!
//!     // Push subscription
//!     let mut rx = engine.subscribe();
//!     tokio::spawn(async move {
//!         while let Ok(snap) = rx.recv().await {
//!             println!("{:?} @ ${:.2}  conf={:.2}", snap.fused_direction, snap.price, snap.fused_confidence);
//!         }
//!     });
//!
//!     // Query API
//!     let q = QueryEngine::new(engine);
//!     if let Ok(win) = q.window_5m() {
//!         if let Some(ohlcv) = win.ohlcv {
//!             println!("5m return: {:+.4}%", ohlcv.return_pct() * 100.0);
//!         }
//!     }
//!     if let Some(fc) = q.forecast_5s() {
//!         println!("next 30s Δprice steps: {:?}", fc.deltas);
//!     }
//! }
//! ```
//!
//! ## System requirements
//!
//! | Requirement | Notes |
//! |---|---|
//! | Rust ≥ 1.75 | MSRV |
//! | OpenSSL dev headers | `apt install libssl-dev pkg-config` (Linux) |
//! | — or rustls — | Replace `native-tls` with `rustls-tls` in Cargo.toml for pure-Rust TLS |
//!
//! ## Feature flags
//!
//! | Flag | Default | Effect |
//! |---|---|---|
//! | `feeds-binance`  | on | Binance feed |
//! | `feeds-coinbase` | on | Coinbase feed (requires JWT credentials) |
//! | `feeds-kraken`   | on | Kraken feed |
//! | `feeds-bitstamp` | on | Bitstamp feed |
//! | `metrics`        | off | Prometheus `/metrics` endpoint |
//! | `tracing`        | off | `tracing` crate instrumentation |
//! | `persistence`    | off | Snapshot save/load (bincode + zstd) |
//!
//! ## Plugging in trained models
//!
//! Implement [`models::TrendModelExt`] or [`models::ForecastModelExt`] and
//! set them on [`pipeline::PipelineConfig`]:
//!
//! ```rust,ignore
//! let config = EngineConfig {
//!     pipeline: PipelineConfig {
//!         ext_trend:    Some(Box::new(MyXgbModel::load("trend.json"))),
//!         ext_forecast: Some(Box::new(MyLstmModel::load("lstm.onnx"))),
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```
//!
//! Recommended crates: `tract-onnx` (pure Rust ONNX), `xgboost` (C-FFI),
//! `candle-core` (HuggingFace), `burn` (full framework).

// ─── Module declarations ──────────────────────────────────────────────────────

pub mod types;
pub mod dedup;
pub mod price_fusion;
pub mod features;
pub mod models;
pub mod feeds;
pub mod pipeline;
pub mod engine;
pub mod query;

// ─── Prelude ──────────────────────────────────────────────────────────────────

/// Everything you need for typical usage — `use btc_prediction_engine::prelude::*`.
pub mod prelude {
    pub use crate::engine::{EngineConfig, EngineHandles, PredictionEngine};
    pub use crate::feeds::FeedConfig;
    pub use crate::pipeline::PipelineConfig;
    pub use crate::price_fusion::{FusedTick, FusionConfig};
    pub use crate::query::QueryEngine;
    pub use crate::types::{
        EngineError, EngineResult,
        EngineMetrics,
        Exchange, Symbol,
        FeedHealth,
        Ohlcv,
        PredictionSnapshot,
        ShortTermForecast,
        TimeScale,
        Tick, TradeSide,
        TrendDirection, TrendSignal,
        WindowProjection,
    };
}
