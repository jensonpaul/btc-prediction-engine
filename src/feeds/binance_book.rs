//! Binance Spot order book feed — top-[`BOOK_DEPTH`] bids and asks.
//!
//! **Stream:** `<symbol>@depth<N>@100ms`
//!
//! Uses the partial-depth push stream which delivers a **full snapshot** of the
//! top N levels on every update (no local state management / diff application
//! needed). Updates arrive every 100 ms.
//!
//! **Endpoint:** `wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms`
//!
//! **Docs:**
//! <https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#partial-book-depth-streams>
//!
//! ## Payload
//! ```json
//! {
//!   "lastUpdateId": 1027024,
//!   "bids": [["4.00000000","431.00000000"], …],
//!   "asks": [["4.00000200","12.00000000"], …]
//! }
//! ```
//! Each entry is `["price", "qty"]` as strings. Bids are sorted best (highest)
//! first; asks best (lowest) first — matches our [`BookSnapshot`] convention.
//!
//! ## Limits (June 2026)
//! * Valid depths: 5, 10, 20. We use 10 to match [`BOOK_DEPTH`].
//! * 100 ms update interval (also available: `@1000ms`).
//! * Counts as one connection toward the 300 connections/IP limit.

use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::feeds::{log_error, log_info, log_reconnect, BookFeedConfig};
use crate::types::{BookSnapshot, EngineError, EngineResult, Exchange, Level, Symbol, BOOK_DEPTH};

const WS_BASE: &str = "wss://stream.binance.com:9443/ws";

#[derive(Deserialize)]
struct BinanceDepth {
    bids: Vec<[String; 2]>,
    asks: Vec<[String; 2]>,
}

fn parse_levels(raw: &[[String; 2]]) -> Vec<Level> {
    raw.iter()
        .filter_map(|pair| {
            let price: f64 = pair[0].parse().ok()?;
            let qty:   f64 = pair[1].parse().ok()?;
            // Filter zero-qty levels (Binance sends these to signal level removal
            // in diff streams; they shouldn't appear in partial-depth snapshots,
            // but guard anyway).
            if qty == 0.0 { return None; }
            Some(Level { price, qty })
        })
        .take(BOOK_DEPTH)
        .collect()
}

fn parse(text: &str, ts_micros: i64) -> Option<BookSnapshot> {
    let d: BinanceDepth = serde_json::from_str(text).ok()?;
    Some(BookSnapshot {
        ts_micros,
        exchange: Exchange::Binance,
        symbol:   Symbol::BtcUsd,
        bids:     parse_levels(&d.bids),
        asks:     parse_levels(&d.asks),
    })
}

pub async fn run(config: BookFeedConfig, tx: mpsc::Sender<BookSnapshot>) -> EngineResult<()> {
    let sym  = config.feed.native_symbol(); // "btcusdt"
    let url  = format!("{WS_BASE}/{sym}@depth{BOOK_DEPTH}@100ms");
    let mut attempt = 0u32;

    loop {
        log_info(Exchange::Binance, &format!("[book] connecting → {url}"));
        match connect_async(&url).await {
            Ok((mut ws, _)) => {
                attempt = 0;
                log_info(Exchange::Binance, "[book] connected");
                loop {
                    match ws.next().await {
                        Some(Ok(Message::Text(t))) => {
                            let ts = chrono::Utc::now().timestamp_micros();
                            if let Some(snap) = parse(&t, ts) {
                                if tx.send(snap).await.is_err() { return Ok(()); }
                            }
                        }
                        Some(Ok(Message::Ping(d))) => { let _ = ws.send(Message::Pong(d)).await; }
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Err(e)) => { log_error(Exchange::Binance, &format!("[book] {e}")); break; }
                        _ => {}
                    }
                }
            }
            Err(e) => log_error(Exchange::Binance, &format!("[book] {e}")),
        }
        attempt += 1;
        if config.feed.max_reconnect_attempts > 0 && attempt > config.feed.max_reconnect_attempts {
            return Err(EngineError::WebSocket {
                exchange: Exchange::Binance,
                source:   anyhow::anyhow!("book feed: max reconnects exceeded"),
            });
        }
        let d = config.feed.backoff(attempt);
        log_reconnect(Exchange::Binance, attempt, &format!("[book] retry in {d:?}"));
        tokio::time::sleep(d).await;
    }
}
