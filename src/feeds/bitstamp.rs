//! Bitstamp WebSocket v2 trade feed — BTC/USD.
//!
//! **Endpoint:** `wss://ws.bitstamp.net`  channel: `live_trades_btcusd`
//! **Docs:** <https://www.bitstamp.net/websocket/v2/>
//!
//! ## Subscribe
//! ```json
//! {"event":"bts:subscribe","data":{"channel":"live_trades_btcusd"}}
//! ```
//!
//! ## Trade event
//! ```json
//! {"event":"trade","channel":"live_trades_btcusd","data":{
//!   "id":123456,"timestamp":"1699000000","microtimestamp":"1699000000000000",
//!   "amount":0.002,"amount_str":"0.00200000",
//!   "price":34000,"price_str":"34000",
//!   "type":0,"buy_order_id":111,"sell_order_id":222
//! }}
//! ```
//!
//! ## `type` field
//! * `0` = taker is buyer (Buy)
//! * `1` = taker is seller (Sell)
//!
//! ## Special server events
//! * `bts:heartbeat`          — no action needed
//! * `bts:request_reconnect`  — server rolling restart; reconnect immediately
//! * `bts:subscription_succeeded` — confirmation
//!
//! ## Notes
//! * `microtimestamp` is already µs — use this, not `timestamp`.
//! * `price` may arrive as JSON number or quoted string depending on server
//!   version; both forms are handled.
//! * Public stream — no API key required.
//!
//! ## Limits (June 2026)
//! * No documented per-IP WS connection limit.
//! * REST: ~8000 req/10 min (auth), ~400 (public). WS trade streams unmetered.

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::feeds::{FeedConfig, log_error, log_info, log_reconnect, log_warn};
use crate::types::{EngineError, EngineResult, Exchange, Symbol, Tick, TradeSide};

const WS_URL: &str = "wss://ws.bitstamp.net";

// ─── Outbound ─────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct BitstampSub<'a> {
    event: &'static str,
    data:  BitstampSubData<'a>,
}
#[derive(Serialize)]
struct BitstampSubData<'a> { channel: &'a str }

// ─── Inbound ──────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct BitstampEnvelope {
    event:   String,
    data:    Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct BitstampTrade {
    id:              Option<u64>,
    microtimestamp:  String,
    #[serde(deserialize_with = "de_f64_flexible")]
    price:           f64,
    #[serde(deserialize_with = "de_f64_flexible")]
    amount:          f64,
    #[serde(rename = "type")]
    trade_type:      u8,
}

/// Deserialise a field that Bitstamp sends as either a JSON number or a
/// quoted decimal string (inconsistent across server versions).
fn de_f64_flexible<'de, D>(d: D) -> Result<f64, D::Error>
where D: serde::Deserializer<'de> {
    use serde::de::{self, Visitor};
    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = f64;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "number or decimal string")
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<f64, E> { Ok(v) }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<f64, E> { Ok(v as f64) }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<f64, E> { Ok(v as f64) }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<f64, E> {
            v.parse().map_err(de::Error::custom)
        }
    }
    d.deserialize_any(V)
}

// ─── Parse result ────────────────────────────────────────────────────────────

enum ParseResult { Tick(Tick), Reconnect, Skip }

fn parse(text: &str) -> ParseResult {
    let env: BitstampEnvelope = match serde_json::from_str(text) {
        Ok(e) => e,
        Err(e) => { log_error(Exchange::Bitstamp, &format!("envelope: {e}")); return ParseResult::Skip; }
    };

    match env.event.as_str() {
        "trade" => {
            let val = match env.data { Some(v) => v, None => return ParseResult::Skip };
            let t: BitstampTrade = match serde_json::from_value(val) {
                Ok(t) => t,
                Err(e) => { log_error(Exchange::Bitstamp, &format!("trade: {e}")); return ParseResult::Skip; }
            };
            let ts_micros = t.microtimestamp.parse::<i64>().unwrap_or_else(|_| {
                chrono::Utc::now().timestamp_micros()
            });
            let side = match t.trade_type {
                0 => Some(TradeSide::Buy),
                1 => Some(TradeSide::Sell),
                _ => None,
            };
            ParseResult::Tick(Tick {
                ts_micros, price: t.price, quantity: t.amount, side,
                exchange: Exchange::Bitstamp, symbol: Symbol::BtcUsd,
                trade_id: t.id.map(|id| id.to_string()).unwrap_or_default(),
            })
        }
        "bts:request_reconnect" => {
            log_warn(Exchange::Bitstamp, "server requested reconnect");
            ParseResult::Reconnect
        }
        "bts:heartbeat" | "bts:subscription_succeeded" => ParseResult::Skip,
        other => {
            #[cfg(feature = "tracing")]
            tracing::debug!(exchange = "bitstamp", event = other, "unknown event");
            #[cfg(not(feature = "tracing"))]
            let _ = other;
            ParseResult::Skip
        }
    }
}

// ─── Feed runner ─────────────────────────────────────────────────────────────

pub async fn run(config: FeedConfig, tx: mpsc::Sender<Tick>) -> EngineResult<()> {
    let channel     = format!("live_trades_{}", config.native_symbol()); // live_trades_btcusd
    let mut attempt = 0u32;

    loop {
        log_info(Exchange::Bitstamp, &format!("connecting → {WS_URL}"));
        //let url = url::Url::parse(WS_URL).expect("static");
        match connect_async(WS_URL).await {
            Ok((mut ws, _)) => {
                attempt = 0;
                log_info(Exchange::Bitstamp, &format!("connected; subscribing {channel}"));

                let sub = serde_json::to_string(&BitstampSub {
                    event: "bts:subscribe",
                    data:  BitstampSubData { channel: &channel },
                }).map_err(|e| EngineError::Parse { exchange: Exchange::Bitstamp, source: e })?;

                if ws.send(Message::Text(sub.into())).await.is_err() { continue; }

                loop {
                    match ws.next().await {
                        Some(Ok(Message::Text(t))) => match parse(&t) {
                            ParseResult::Tick(tick) => {
                                if tx.send(tick).await.is_err() { return Ok(()); }
                            }
                            ParseResult::Reconnect => break,
                            ParseResult::Skip => {}
                        },
                        Some(Ok(Message::Ping(d))) => { let _ = ws.send(Message::Pong(d)).await; }
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Err(e)) => { log_error(Exchange::Bitstamp, &e.to_string()); break; }
                        _ => {}
                    }
                }
            }
            Err(e) => log_error(Exchange::Bitstamp, &e.to_string()),
        }
        attempt += 1;
        if config.max_reconnect_attempts > 0 && attempt > config.max_reconnect_attempts {
            return Err(EngineError::WebSocket {
                exchange: Exchange::Bitstamp,
                source:   anyhow::anyhow!("max reconnects exceeded"),
            });
        }
        let d = config.backoff(attempt);
        log_reconnect(Exchange::Bitstamp, attempt, &format!("retry in {d:?}"));
        tokio::time::sleep(d).await;
    }
}
