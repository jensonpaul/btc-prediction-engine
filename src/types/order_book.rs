//! Order book snapshot types.
//!
//! A [`BookSnapshot`] represents the state of the top-N levels of a single
//! exchange's order book at a point in time. The engine processes these
//! alongside trade ticks to compute depth-based features.
//!
//! ## Design
//!
//! * Levels are sorted: bids descending by price, asks ascending by price.
//! * Only the top [`BOOK_DEPTH`] levels on each side are kept — enough for the
//!   imbalance features we compute, and cheap to copy.
//! * Timestamps are µs since UNIX epoch, same convention as [`Tick`].

use crate::types::{Exchange, Symbol};
use serde::{Deserialize, Serialize};

/// Number of price levels kept on each side of the book.
///
/// 5 is sufficient for top-5 imbalance; keeping a few extra gives models
/// the option to compute depth falloff without wasting memory.
pub const BOOK_DEPTH: usize = 10;

/// A single price level: (price, quantity) in USD / BTC.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Level {
    pub price: f64,
    pub qty:   f64,
}

/// Top-[`BOOK_DEPTH`] order book snapshot from one exchange.
///
/// Bids are sorted **descending** (best bid first).
/// Asks are sorted **ascending** (best ask first).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookSnapshot {
    /// Snapshot timestamp in µs since UNIX epoch.
    pub ts_micros: i64,
    pub exchange:  Exchange,
    pub symbol:    Symbol,
    /// Top bids, best first (highest price).
    pub bids:      Vec<Level>,
    /// Top asks, best first (lowest price).
    pub asks:      Vec<Level>,
}

impl BookSnapshot {
    /// Best bid price (`None` if book is empty).
    #[inline]
    pub fn best_bid(&self) -> Option<f64> { self.bids.first().map(|l| l.price) }

    /// Best ask price (`None` if book is empty).
    #[inline]
    pub fn best_ask(&self) -> Option<f64> { self.asks.first().map(|l| l.price) }

    /// Mid-price = (best_bid + best_ask) / 2.
    #[inline]
    pub fn mid_price(&self) -> Option<f64> {
        Some((self.best_bid()? + self.best_ask()?) / 2.0)
    }

    /// Bid-ask spread in USD.
    #[inline]
    pub fn spread_usd(&self) -> Option<f64> {
        Some(self.best_ask()? - self.best_bid()?)
    }

    /// Volume-weighted bid-ask imbalance over the top `depth` levels.
    ///
    /// `imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)`
    ///
    /// Returns a value in [−1, 1]: +1 = all bids, −1 = all asks.
    /// Returns `None` if either side has no levels or total volume is zero.
    pub fn imbalance(&self, depth: usize) -> Option<f64> {
        let bid_vol: f64 = self.bids.iter().take(depth).map(|l| l.qty).sum();
        let ask_vol: f64 = self.asks.iter().take(depth).map(|l| l.qty).sum();
        let total = bid_vol + ask_vol;
        if total == 0.0 { return None; }
        Some((bid_vol - ask_vol) / total)
    }

    /// Weighted mid-price using top-`depth` volume as weights.
    ///
    /// Gives a more stable mid than the simple (best_bid + best_ask) / 2
    /// when the spread is wide or one side is thin.
    pub fn weighted_mid(&self, depth: usize) -> Option<f64> {
        let bid_vol: f64 = self.bids.iter().take(depth).map(|l| l.qty).sum();
        let ask_vol: f64 = self.asks.iter().take(depth).map(|l| l.qty).sum();
        let best_bid = self.best_bid()?;
        let best_ask = self.best_ask()?;
        let total = bid_vol + ask_vol;
        if total == 0.0 { return None; }
        // bid_vol-weighted ask + ask_vol-weighted bid (micro-price formula)
        Some((best_bid * ask_vol + best_ask * bid_vol) / total)
    }
}
