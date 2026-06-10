//! Bitstamp WebSocket v2 order book feed — top-[`BOOK_DEPTH`] bids and asks.
//!
//! **Endpoint:** `wss://ws.bitstamp.net`  channel: `order_book_btcusd`
//!
//! **Docs:** <https://www.bitstamp.net/websocket/v2/>
//!
//! ## Subscribe
//! ```json
//! {"event":"bts:subscribe","data":{"channel":"order_book_btcusd"}}
//! ```
//!
//! ## Book event (`event: "data"`)
//! ```json
//! {"event":"data","channel":"order_book_btcusd","data":{
//!   "timestamp":"1699000000",
//!   "microtimestamp":"1699000000000000",
//!   "bids":[["34000.00","0.500"], …],
//!   "asks":[["34001.00","0.300"], …]
//! }}
//! ```
//! Each entry is `["price", "qty"]` as strings.
//!
//! ## Full vs diff
//!
//! `order_book_btcusd` delivers **full snapshots** on every message (not
//! diffs), so no local state management is required. The channel sends the
//! complete top-100 book; we take only the top [`BOOK_DEPTH`] levels.
//!
//! For diff-based feeds use `diff_order_book_btcusd` — but that requires
//! local state management and a REST snapshot to initialise. Full snapshots
//! are simpler and sufficient for the imbalance features we compute.
//!
//! ## Notes
//! * `microtimestamp` is already µs — use this, not `timestamp`.
//! * Public stream — no API key required.
//! * `bts:request_reconnect` is handled: reconnects immediately.

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::feeds::{log_error, log_info, log_reconnect, log_warn, BookFeedConfig};
use crate::types::{BookSnapshot, EngineError, EngineResult, Exchange, Level, Symbol, BOOK_DEPTH};

const WS_URL: &str = "wss://ws.bitstamp.net";

// ─── Wire types ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct BitstampSub<'a> {
    event: &'static str,
    data:  BitstampSubData<'a>,
}
#[derive(Serialize)]
struct BitstampSubData<'a> { channel: &'a str }

#[derive(Deserialize)]
struct BitstampEnvelope {
    event:   String,
    data:    Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct BitstampBook {
    microtimestamp: String,
    bids:           Vec<[String; 2]>,
    asks:           Vec<[String; 2]>,
}

// ─── Parsing ─────────────────────────────────────────────────────────────────

fn parse_levels(raw: &[[String; 2]]) -> Vec<Level> {
    raw.iter()
        .filter_map(|pair| {
            let price: f64 = pair[0].parse().ok()?;
            let qty:   f64 = pair[1].parse().ok()?;
            if qty == 0.0 { return None; }
            Some(Level { price, qty })
        })
        .take(BOOK_DEPTH)
        .collect()
}

enum ParseResult { Snapshot(BookSnapshot), Reconnect, Skip }

fn parse(text: &str) -> ParseResult {
    let env: BitstampEnvelope = match serde_json::from_str(text) {
        Ok(e) => e,
        Err(e) => { log_error(Exchange::Bitstamp, &format!("[book] envelope: {e}")); return ParseResult::Skip; }
    };

    match env.event.as_str() {
        "data" => {
            let val = match env.data { Some(v) => v, None => return ParseResult::Skip };
            let b: BitstampBook = match serde_json::from_value(val) {
                Ok(b) => b,
                Err(e) => { log_error(Exchange::Bitstamp, &format!("[book] data: {e}")); return ParseResult::Skip; }
            };
            let ts_micros = b.microtimestamp.parse::<i64>()
                .unwrap_or_else(|_| chrono::Utc::now().timestamp_micros());
            let bids = parse_levels(&b.bids);
            let asks = parse_levels(&b.asks);
            if bids.is_empty() && asks.is_empty() { return ParseResult::Skip; }
            ParseResult::Snapshot(BookSnapshot {
                ts_micros,
                exchange: Exchange::Bitstamp,
                symbol:   Symbol::BtcUsd,
                bids,
                asks,
            })
        }
        "bts:request_reconnect" => {
            log_warn(Exchange::Bitstamp, "[book] server requested reconnect");
            ParseResult::Reconnect
        }
        "bts:heartbeat" | "bts:subscription_succeeded" => ParseResult::Skip,
        _ => ParseResult::Skip,
    }
}

// ─── Feed runner ─────────────────────────────────────────────────────────────

pub async fn run(config: BookFeedConfig, tx: mpsc::Sender<BookSnapshot>) -> EngineResult<()> {
    let channel     = format!("order_book_{}", config.feed.native_symbol()); // order_book_btcusd
    let mut attempt = 0u32;

    loop {
        log_info(Exchange::Bitstamp, &format!("[book] connecting → {WS_URL}"));
        match connect_async(WS_URL).await {
            Ok((mut ws, _)) => {
                attempt = 0;
                log_info(Exchange::Bitstamp, &format!("[book] connected; subscribing {channel}"));

                let sub = serde_json::to_string(&BitstampSub {
                    event: "bts:subscribe",
                    data:  BitstampSubData { channel: &channel },
                }).map_err(|e| EngineError::Parse { exchange: Exchange::Bitstamp, source: e })?;

                if ws.send(Message::Text(sub.into())).await.is_err() { continue; }

                loop {
                    match ws.next().await {
                        Some(Ok(Message::Text(t))) => match parse(&t) {
                            ParseResult::Snapshot(snap) => {
                                if tx.send(snap).await.is_err() { return Ok(()); }
                            }
                            ParseResult::Reconnect => break,
                            ParseResult::Skip => {}
                        },
                        Some(Ok(Message::Ping(d))) => { let _ = ws.send(Message::Pong(d)).await; }
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Err(e)) => { log_error(Exchange::Bitstamp, &format!("[book] {e}")); break; }
                        _ => {}
                    }
                }
            }
            Err(e) => log_error(Exchange::Bitstamp, &format!("[book] {e}")),
        }
        attempt += 1;
        if config.feed.max_reconnect_attempts > 0 && attempt > config.feed.max_reconnect_attempts {
            return Err(EngineError::WebSocket {
                exchange: Exchange::Bitstamp,
                source:   anyhow::anyhow!("book feed: max reconnects exceeded"),
            });
        }
        let d = config.feed.backoff(attempt);
        log_reconnect(Exchange::Bitstamp, attempt, &format!("[book] retry in {d:?}"));
        tokio::time::sleep(d).await;
    }
}
