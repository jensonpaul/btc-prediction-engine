//! Kraken Spot WebSocket v2 order book feed — top-[`BOOK_DEPTH`] bids and asks.
//!
//! **Endpoint:** `wss://ws.kraken.com/v2`  channel: `book`
//!
//! **Docs:** <https://docs.kraken.com/api/docs/websocket-v2/book>
//!
//! ## Subscribe
//! ```json
//! {"method":"subscribe","params":{"channel":"book","symbol":["BTC/USD"],"depth":10,"snapshot":true}}
//! ```
//!
//! ## Snapshot message (`type: "snapshot"`)
//! ```json
//! {"channel":"book","type":"snapshot","data":[{
//!   "symbol":"BTC/USD","checksum":12345,
//!   "bids":[{"price":34000.0,"qty":0.5}, …],
//!   "asks":[{"price":34001.0,"qty":0.3}, …]
//! }]}
//! ```
//!
//! ## Update message (`type: "update"`)
//! Same structure as snapshot but only the changed levels are included.
//! Changed levels with `qty: 0.0` indicate a level removal.
//!
//! ## Local state management
//!
//! Kraken v2 sends a full snapshot on subscribe, then incremental diffs.
//! We maintain a local `BTreeMap<OrderedF64, f64>` per side, apply diffs, then
//! emit a [`BookSnapshot`] of the top [`BOOK_DEPTH`] levels after each update.
//!
//! ## Notes
//! * Valid depths: 10, 25, 100, 500, 1000.
//! * Server closes idle connections after ~60 s — send JSON ping every 30 s.
//! * Public stream — no API key required.

use std::collections::BTreeMap;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::feeds::{log_error, log_info, log_reconnect, BookFeedConfig};
use crate::types::{BookSnapshot, EngineError, EngineResult, Exchange, Level, Symbol, BOOK_DEPTH};

const WS_URL: &str = "wss://ws.kraken.com/v2";

// ─── Wire types ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct KrakenBookSub<'a> {
    method: &'static str,
    params: KrakenBookSubParams<'a>,
}
#[derive(Serialize)]
struct KrakenBookSubParams<'a> {
    channel:  &'static str,
    symbol:   &'a [&'a str],
    depth:    usize,
    snapshot: bool,
}
#[derive(Serialize)]
struct KrakenPing { method: &'static str }

#[derive(Deserialize)]
struct KrakenBookMsg {
    channel:  Option<String>,
    #[serde(rename = "type")]
    msg_type: Option<String>,
    data:     Option<Vec<KrakenBookData>>,
}
#[derive(Deserialize)]
struct KrakenBookData {
    bids: Option<Vec<KrakenLevel>>,
    asks: Option<Vec<KrakenLevel>>,
}
#[derive(Deserialize)]
struct KrakenLevel {
    price: f64,
    qty:   f64,
}

// ─── Local book state ─────────────────────────────────────────────────────────

/// Newtype so we can use f64 as a `BTreeMap` key.
///
/// Safe here: exchange prices are always finite positive numbers.
#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

struct LocalBook {
    /// price → qty, sorted ascending
    bids: BTreeMap<OrdF64, f64>,
    /// price → qty, sorted ascending
    asks: BTreeMap<OrdF64, f64>,
}

impl LocalBook {
    fn new() -> Self {
        Self { bids: BTreeMap::new(), asks: BTreeMap::new() }
    }

    fn apply(&mut self, levels: &[KrakenLevel], side: Side) {
        let map = match side { Side::Bid => &mut self.bids, Side::Ask => &mut self.asks };
        for l in levels {
            if l.qty == 0.0 {
                map.remove(&OrdF64(l.price));
            } else {
                map.insert(OrdF64(l.price), l.qty);
            }
        }
    }

    fn snapshot(&self, ts_micros: i64) -> BookSnapshot {
        // Bids: highest prices first
        let bids: Vec<Level> = self.bids.iter().rev()
            .take(BOOK_DEPTH)
            .map(|(k, &q)| Level { price: k.0, qty: q })
            .collect();
        // Asks: lowest prices first
        let asks: Vec<Level> = self.asks.iter()
            .take(BOOK_DEPTH)
            .map(|(k, &q)| Level { price: k.0, qty: q })
            .collect();
        BookSnapshot { ts_micros, exchange: Exchange::Kraken, symbol: Symbol::BtcUsd, bids, asks }
    }
}

#[derive(Clone, Copy)]
enum Side { Bid, Ask }

// ─── Feed runner ─────────────────────────────────────────────────────────────

pub async fn run(config: BookFeedConfig, tx: mpsc::Sender<BookSnapshot>) -> EngineResult<()> {
    let sym         = config.feed.native_symbol();
    let mut attempt = 0u32;
    let ping_json   = serde_json::to_string(&KrakenPing { method: "ping" }).unwrap_or_default();

    loop {
        log_info(Exchange::Kraken, &format!("[book] connecting → {WS_URL}"));
        match connect_async(WS_URL).await {
            Ok((mut ws, _)) => {
                attempt = 0;
                log_info(Exchange::Kraken, "[book] connected");

                let sub = serde_json::to_string(&KrakenBookSub {
                    method: "subscribe",
                    params: KrakenBookSubParams {
                        channel:  "book",
                        symbol:   &[sym],
                        depth:    BOOK_DEPTH,
                        snapshot: true,
                    },
                }).map_err(|e| EngineError::Parse { exchange: Exchange::Kraken, source: e })?;

                if ws.send(Message::Text(sub.into())).await.is_err() { continue; }

                let mut book        = LocalBook::new();
                let mut ping_tick   = tokio::time::interval(std::time::Duration::from_secs(30));
                ping_tick.tick().await; // consume immediate first tick

                loop {
                    tokio::select! {
                        msg = ws.next() => match msg {
                            Some(Ok(Message::Text(t))) => {
                                let ts = chrono::Utc::now().timestamp_micros();
                                if let Ok(m) = serde_json::from_str::<KrakenBookMsg>(&t) {
                                    if m.channel.as_deref() == Some("book") {
                                        let is_update = m.msg_type.as_deref() == Some("update");
                                        let is_snap   = m.msg_type.as_deref() == Some("snapshot");
                                        if is_update || is_snap {
                                            for d in m.data.into_iter().flatten() {
                                                if let Some(bids) = d.bids {
                                                    book.apply(&bids, Side::Bid);
                                                }
                                                if let Some(asks) = d.asks {
                                                    book.apply(&asks, Side::Ask);
                                                }
                                            }
                                            // Only emit after we have both sides
                                            if !book.bids.is_empty() && !book.asks.is_empty() {
                                                if tx.send(book.snapshot(ts)).await.is_err() {
                                                    return Ok(());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Some(Ok(Message::Ping(d))) => { let _ = ws.send(Message::Pong(d)).await; }
                            Some(Ok(Message::Close(_))) | None => break,
                            Some(Err(e)) => { log_error(Exchange::Kraken, &format!("[book] {e}")); break; }
                            _ => {}
                        },
                        _ = ping_tick.tick() => {
                            if ws.send(Message::Text(ping_json.clone().into())).await.is_err() { break; }
                        }
                    }
                }
            }
            Err(e) => log_error(Exchange::Kraken, &format!("[book] {e}")),
        }
        attempt += 1;
        if config.feed.max_reconnect_attempts > 0 && attempt > config.feed.max_reconnect_attempts {
            return Err(EngineError::WebSocket {
                exchange: Exchange::Kraken,
                source:   anyhow::anyhow!("book feed: max reconnects exceeded"),
            });
        }
        let d = config.feed.backoff(attempt);
        log_reconnect(Exchange::Kraken, attempt, &format!("[book] retry in {d:?}"));
        tokio::time::sleep(d).await;
    }
}
