//! Exchange feed abstraction and helpers.

pub mod binance;
pub mod coinbase;
pub mod kraken;
pub mod bitstamp;

// ── Order book feeds (Binance, Kraken, Bitstamp) ─────────────────────────────
#[cfg(feature = "feeds-binance")]
pub mod binance_book;
#[cfg(feature = "feeds-kraken")]
pub mod kraken_book;
#[cfg(feature = "feeds-bitstamp")]
pub mod bitstamp_book;

use std::time::Duration;
use tokio::sync::mpsc;
use crate::types::{BookSnapshot, EngineResult, Exchange, Symbol, Tick};

// ─── Trade feed configuration ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FeedConfig {
    pub exchange:               Exchange,
    pub symbol:                 Symbol,
    pub api_key:                Option<String>,
    pub api_secret:             Option<String>,
    pub max_reconnect_attempts: u32,
    pub reconnect_initial:      Duration,
    pub reconnect_cap:          Duration,
}

impl FeedConfig {
    pub fn public(exchange: Exchange, symbol: Symbol) -> Self {
        Self {
            exchange,
            symbol,
            api_key:                None,
            api_secret:             None,
            max_reconnect_attempts: 0,
            reconnect_initial:      Duration::from_secs(1),
            reconnect_cap:          Duration::from_secs(60),
        }
    }

    pub fn authenticated(
        exchange:   Exchange,
        symbol:     Symbol,
        api_key:    impl Into<String>,
        api_secret: impl Into<String>,
    ) -> Self {
        let mut c = Self::public(exchange, symbol);
        c.api_key    = Some(api_key.into());
        c.api_secret = Some(api_secret.into());
        c
    }

    /// Exponential backoff capped at `reconnect_cap`.
    pub fn backoff(&self, attempt: u32) -> Duration {
        let secs = self.reconnect_initial.as_secs_f64()
            * 2.0_f64.powi(attempt.min(10) as i32);
        Duration::from_secs_f64(secs.min(self.reconnect_cap.as_secs_f64()))
    }

    /// Native exchange symbol string for this (exchange, symbol) pair.
    pub fn native_symbol(&self) -> &'static str {
        self.symbol.for_exchange(self.exchange)
    }
}

// ─── Order book feed configuration ───────────────────────────────────────────

/// Configuration for a Level 2 order book feed.
///
/// Wraps a [`FeedConfig`] (for shared reconnect / symbol logic) and adds any
/// book-specific options.
#[derive(Debug, Clone)]
pub struct BookFeedConfig {
    /// Shared trade/book feed settings (exchange, symbol, reconnect policy).
    pub feed: FeedConfig,
}

impl BookFeedConfig {
    /// Construct a public (unauthenticated) book feed config.
    pub fn public(exchange: Exchange, symbol: Symbol) -> Self {
        Self { feed: FeedConfig::public(exchange, symbol) }
    }
}

// ─── Feed trait ───────────────────────────────────────────────────────────────

/// A feed that can run as an independent async task.
#[async_trait::async_trait]
pub trait Feed: Send + 'static {
    async fn run(self, tx: mpsc::Sender<Tick>) -> EngineResult<()>;
    fn exchange(&self) -> Exchange;
}

// ─── Logging ─────────────────────────────────────────────────────────────────

pub(crate) fn log_info(exchange: Exchange, msg: &str) {
    #[cfg(feature = "tracing")]
    tracing::info!(exchange = %exchange, "{}", msg);
    #[cfg(not(feature = "tracing"))]
    eprintln!("[btc_engine:info ] {exchange}: {msg}");
}

pub(crate) fn log_warn(exchange: Exchange, msg: &str) {
    #[cfg(feature = "tracing")]
    tracing::warn!(exchange = %exchange, "{}", msg);
    #[cfg(not(feature = "tracing"))]
    eprintln!("[btc_engine:warn ] {exchange}: {msg}");
}

pub(crate) fn log_error(exchange: Exchange, msg: &str) {
    #[cfg(feature = "tracing")]
    tracing::error!(exchange = %exchange, "{}", msg);
    #[cfg(not(feature = "tracing"))]
    eprintln!("[btc_engine:error] {exchange}: {msg}");
}

pub(crate) fn log_reconnect(exchange: Exchange, attempt: u32, msg: &str) {
    #[cfg(feature = "tracing")]
    tracing::warn!(exchange = %exchange, attempt, "{}", msg);
    #[cfg(not(feature = "tracing"))]
    eprintln!("[btc_engine:reconn] {exchange} attempt {attempt}: {msg}");
}
