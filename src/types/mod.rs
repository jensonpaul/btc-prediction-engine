//! Core domain types.
//!
//! All monetary values are in **USD** (quote currency).
//! All quantities are in **BTC** (base currency).
//! All timestamps are **µs since UNIX epoch** (`i64`).

pub mod tick_store;
pub mod pred_store;

pub use tick_store::TickStore;
pub use pred_store::PredictionStore;

use std::fmt;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─── Trading pair ─────────────────────────────────────────────────────────────

/// The asset pair being tracked.
///
/// Currently BTC/USD only. Add variants here to support ETH/USD, SOL/USD, etc.
/// Each exchange feed maps its native symbol string to this enum at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Symbol {
    /// Bitcoin / US Dollar
    BtcUsd,
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BtcUsd => write!(f, "BTC/USD"),
        }
    }
}

impl Symbol {
    /// Native symbol string used by each exchange for this pair.
    ///
    /// Returns the exact string to pass in WebSocket subscribe messages.
    pub fn for_exchange(self, exchange: Exchange) -> &'static str {
        match (self, exchange) {
            (Self::BtcUsd, Exchange::Binance)  => "btcusdt",
            (Self::BtcUsd, Exchange::Coinbase) => "BTC-USD",
            (Self::BtcUsd, Exchange::Kraken)   => "BTC/USD",
            (Self::BtcUsd, Exchange::Bitstamp) => "btcusd",
        }
    }
}

// ─── Exchange ─────────────────────────────────────────────────────────────────

/// Supported data-source exchanges.
///
/// Add a new variant + feed module to extend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Exchange {
    /// Binance Spot
    /// WS: `wss://stream.binance.com:9443/ws/btcusdt@trade?timeUnit=MICROSECOND`
    /// Docs: <https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md>
    Binance,

    /// Coinbase Advanced Trade
    /// WS: `wss://advanced-trade-ws.coinbase.com`  channel: `market_trades`
    /// Docs: <https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-channels>
    Coinbase,

    /// Kraken Spot WebSocket v2
    /// WS: `wss://ws.kraken.com/v2`  channel: `trade`
    /// Docs: <https://docs.kraken.com/api/docs/websocket-v2/trade>
    Kraken,

    /// Bitstamp WebSocket v2
    /// WS: `wss://ws.bitstamp.net`  channel: `live_trades_btcusd`
    /// Docs: <https://www.bitstamp.net/websocket/v2/>
    Bitstamp,
}

impl fmt::Display for Exchange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binance  => write!(f, "binance"),
            Self::Coinbase => write!(f, "coinbase"),
            Self::Kraken   => write!(f, "kraken"),
            Self::Bitstamp => write!(f, "bitstamp"),
        }
    }
}

// ─── Tick ─────────────────────────────────────────────────────────────────────

/// A single matched trade tick — the atomic unit of the engine.
///
/// All prices and quantities are normalised to BTC/USD before this struct is
/// created. Exchange-specific fields (raw trade IDs, fee info) are discarded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tick {
    /// Trade timestamp in **microseconds** since UNIX epoch.
    ///
    /// Source precision by exchange:
    /// * Binance  — µs (request `?timeUnit=MICROSECOND`)
    /// * Coinbase — parsed from RFC-3339 nanoseconds → µs
    /// * Kraken   — parsed from RFC-3339 µs field
    /// * Bitstamp — `microtimestamp` field (native µs)
    pub ts_micros: i64,

    /// Executed price in USD.
    pub price: f64,

    /// Executed quantity in BTC.
    pub quantity: f64,

    /// Taker side. `None` when not provided by the exchange.
    pub side: Option<TradeSide>,

    /// Source exchange.
    pub exchange: Exchange,

    /// Symbol (always `BtcUsd` in this version).
    pub symbol: Symbol,

    /// Exchange-assigned trade ID. Used by the dedup layer.
    pub trade_id: String,
}

impl Tick {
    #[inline]
    pub fn timestamp(&self) -> DateTime<Utc> {
        let secs  = self.ts_micros / 1_000_000;
        let nanos = ((self.ts_micros % 1_000_000) * 1_000) as u32;
        DateTime::from_timestamp(secs, nanos).unwrap_or(DateTime::<Utc>::MIN_UTC)
    }

    #[inline]
    pub fn ts_millis(&self) -> i64 { self.ts_micros / 1_000 }

    /// Notional value in USD.
    #[inline]
    pub fn notional(&self) -> f64 { self.price * self.quantity }
}

/// Taker-initiated trade side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

// ─── OHLCV ───────────────────────────────────────────────────────────────────

/// OHLCV bar assembled lazily from a tick slice.
///
/// `open`  = price of the **first** tick in the window.  
/// `close` = price of the **last** tick in the window.  
/// No periodic resets — windows are query-time projections.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ohlcv {
    pub open_ts:    i64,
    pub close_ts:   i64,
    pub open:       f64,
    pub high:       f64,
    pub low:        f64,
    pub close:      f64,
    /// Sum of BTC quantity across all ticks.
    pub volume:     f64,
    /// Sum of USD notional across all ticks.
    pub notional:   f64,
    pub tick_count: usize,
    pub vwap:       f64,
    /// Buy volume / total volume ∈ [0, 1]. `None` if no side information.
    pub buy_ratio:  Option<f64>,
}

impl Ohlcv {
    #[inline]
    pub fn return_pct(&self) -> f64 {
        if self.open == 0.0 { 0.0 } else { (self.close - self.open) / self.open }
    }

    pub fn true_range(&self, prev_close: f64) -> f64 {
        self.high.max(prev_close) - self.low.min(prev_close)
    }
}

// ─── Trend signal ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection { Bullish, Bearish, Sideways }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrendSignal {
    pub direction:   TrendDirection,
    /// Calibrated probability ∈ [0, 1].
    pub confidence:  f64,
    pub scale:       TimeScale,
    pub computed_at: i64,
}

/// Named time-scales — all run concurrently on the same tick ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeScale {
    Micro,   // ~30 s
    Short,   // ~5 min
    Medium,  // ~1 h
    Broad,   // ~24 h
}

impl TimeScale {
    pub fn lookback_secs(self) -> i64 {
        match self { Self::Micro => 30, Self::Short => 300, Self::Medium => 3_600, Self::Broad => 86_400 }
    }
    #[inline] pub fn lookback_micros(self) -> i64 { self.lookback_secs() * 1_000_000 }

    pub const ALL: [Self; 4] = [Self::Micro, Self::Short, Self::Medium, Self::Broad];
}

// ─── Short-term forecast ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShortTermForecast {
    /// Step size in seconds (e.g. 5 or 30).
    pub step_secs:    u32,
    /// Δprice for each future step (USD). `delta[0]` = next step.
    pub deltas:       Vec<f64>,
    /// Per-step confidence ∈ [0, 1], same length as `deltas`.
    pub confidence:   Vec<f64>,
    pub generated_at: i64,
}

// ─── Fused metrics ───────────────────────────────────────────────────────────

/// Additional computed metrics included in each snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineMetrics {
    /// Ticks received per second (10-second rolling average).
    pub tick_rate:           f64,
    /// Ticks deduplicated in last second.
    pub dedup_rate:          f64,
    /// Ticks rejected by outlier filter in last second.
    pub outlier_rate:        f64,
    /// Number of feeds currently connected.
    pub connected_feeds:     u32,
    /// Cross-exchange price spread (max - min of last tick per exchange) in USD.
    pub cross_exchange_spread: f64,
}

// ─── Prediction snapshot ──────────────────────────────────────────────────────

/// Complete prediction state at a single µs tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionSnapshot {
    pub snapshot_at:      i64,
    /// Latest BTC/USD price at snapshot time.
    pub price:            f64,
    pub micro:            TrendSignal,
    pub short:            TrendSignal,
    pub medium:           TrendSignal,
    pub broad:            TrendSignal,
    /// Short-term forecasts keyed by step_secs.
    pub forecasts:        std::collections::HashMap<u32, ShortTermForecast>,
    pub fused_direction:  TrendDirection,
    pub fused_confidence: f64,
    pub metrics:          EngineMetrics,
}

// ─── Window projection ───────────────────────────────────────────────────────

/// On-demand projection of global state onto a caller-specified time range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowProjection {
    pub window_start: i64,
    pub window_end:   i64,
    pub ohlcv:        Option<Ohlcv>,
    pub prediction:   Option<PredictionSnapshot>,
    pub tick_count:   usize,
}

// ─── Feed health ─────────────────────────────────────────────────────────────

/// Health state of a single exchange feed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedHealth {
    /// Connected and receiving ticks.
    Healthy,
    /// Connected but no ticks received in the last 10 s.
    Stale,
    /// Reconnecting after a disconnect.
    Reconnecting { attempt: u32 },
    /// Feed has exceeded max reconnect attempts.
    Dead,
}

// ─── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("WebSocket error [{exchange}]: {source}")]
    WebSocket { exchange: Exchange, #[source] source: anyhow::Error },

    #[error("Parse error [{exchange}]: {source}")]
    Parse { exchange: Exchange, #[source] source: serde_json::Error },

    #[error("Auth error [{exchange}]: {detail}")]
    Auth { exchange: Exchange, detail: String },

    #[error("Ring buffer empty — no ticks received yet")]
    EmptyBuffer,

    #[error("No data in range [{start}, {end}]")]
    NoDataInRange { start: i64, end: i64 },

    #[error("Feature error: {0}")]
    Feature(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Snapshot I/O error: {0}")]
    Snapshot(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type EngineResult<T> = Result<T, EngineError>;
