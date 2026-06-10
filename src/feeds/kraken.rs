//! Kraken Spot WebSocket v2 trade feed — BTC/USD.
//!
//! **Endpoint:** `wss://ws.kraken.com/v2`  channel: `trade`
//! **Docs:** <https://docs.kraken.com/api/docs/websocket-v2/trade>
//!
//! ## Subscribe
//! ```json
//! {"method":"subscribe","params":{"channel":"trade","symbol":["BTC/USD"],"snapshot":false}}
//! ```
//!
//! ## Trade event
//! ```json
//! {"channel":"trade","type":"update","data":[
//!   {"symbol":"BTC/USD","side":"buy","price":34000.5,"qty":0.005,
//!    "ord_type":"market","trade_id":4711,"timestamp":"2024-01-01T00:00:00.123456Z"}
//! ]}
//! ```
//!
//! ## Notes
//! * Use `"BTC/USD"` — **not** `"XBT/USD"` (v2 uses the ISO code).
//! * Timestamps are RFC-3339 with µs precision.
//! * Multiple trades may be batched in a single message.
//! * Server closes idle connections after ~60 s — send JSON ping every 30 s.
//!
//! ## Limits (June 2026)
//! * 200 symbols / connection (standard tier)
//! * 200 subscription ops/s standard; 500/s pro
//! * Public streams — no API key required

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::feeds::{FeedConfig, log_error, log_info, log_reconnect};
use crate::types::{EngineError, EngineResult, Exchange, Symbol, Tick, TradeSide};

const WS_URL: &str = "wss://ws.kraken.com/v2";

#[derive(Serialize)]
struct KrakenSub<'a> {
    method: &'static str,
    params: KrakenSubParams<'a>,
}
#[derive(Serialize)]
struct KrakenSubParams<'a> {
    channel:  &'static str,
    symbol:   &'a [&'a str],
    snapshot: bool,
}
#[derive(Serialize)]
struct KrakenPing { method: &'static str }

#[derive(Deserialize)]
struct KrakenMsg {
    channel:  Option<String>,
    #[serde(rename = "type")]
    msg_type: Option<String>,
    data:     Option<Vec<KrakenTrade>>,
}
#[derive(Deserialize)]
struct KrakenTrade {
    trade_id:  Option<u64>,
    price:     f64,
    qty:       f64,
    side:      String,
    timestamp: String,
}

fn parse(text: &str) -> Vec<Tick> {
    let msg: KrakenMsg = match serde_json::from_str(text) { Ok(m) => m, Err(_) => return vec![] };
    if msg.channel.as_deref() != Some("trade") { return vec![]; }
    match msg.msg_type.as_deref() {
        Some("update") | Some("snapshot") => {}
        _ => return vec![],
    }
    msg.data.unwrap_or_default().into_iter().filter_map(|t| {
        let ts = chrono::DateTime::parse_from_rfc3339(&t.timestamp)
            .map(|d| d.timestamp_micros()).ok()?;
        let side = match t.side.as_str() {
            "buy"  => Some(TradeSide::Buy),
            "sell" => Some(TradeSide::Sell),
            _      => None,
        };
        Some(Tick {
            ts_micros: ts, price: t.price, quantity: t.qty, side,
            exchange: Exchange::Kraken, symbol: Symbol::BtcUsd,
            trade_id: t.trade_id.map(|id| id.to_string()).unwrap_or_default(),
        })
    }).collect()
}

pub async fn run(config: FeedConfig, tx: mpsc::Sender<Tick>) -> EngineResult<()> {
    let sym         = config.native_symbol();
    let mut attempt = 0u32;
    let ping_json   = serde_json::to_string(&KrakenPing { method: "ping" }).unwrap_or_default();

    loop {
        log_info(Exchange::Kraken, &format!("connecting → {WS_URL}"));
        //let url = url::Url::parse(WS_URL).expect("static");
        match connect_async(WS_URL).await {
            Ok((mut ws, _)) => {
                attempt = 0;
                log_info(Exchange::Kraken, "connected");

                let sub = serde_json::to_string(&KrakenSub {
                    method: "subscribe",
                    params: KrakenSubParams { channel: "trade", symbol: &[sym], snapshot: false },
                }).map_err(|e| EngineError::Parse { exchange: Exchange::Kraken, source: e })?;

                if ws.send(Message::Text(sub.into())).await.is_err() { continue; }

                let mut ping_tick = tokio::time::interval(std::time::Duration::from_secs(30));
                ping_tick.tick().await; // consume immediate first tick

                loop {
                    tokio::select! {
                        msg = ws.next() => match msg {
                            Some(Ok(Message::Text(t))) => {
                                for tick in parse(&t) {
                                    if tx.send(tick).await.is_err() { return Ok(()); }
                                }
                            }
                            Some(Ok(Message::Ping(d))) => { let _ = ws.send(Message::Pong(d)).await; }
                            Some(Ok(Message::Close(_))) | None => break,
                            Some(Err(e)) => { log_error(Exchange::Kraken, &e.to_string()); break; }
                            _ => {}
                        },
                        _ = ping_tick.tick() => {
                            if ws.send(Message::Text(ping_json.clone().into())).await.is_err() { break; }
                        }
                    }
                }
            }
            Err(e) => log_error(Exchange::Kraken, &e.to_string()),
        }
        attempt += 1;
        if config.max_reconnect_attempts > 0 && attempt > config.max_reconnect_attempts {
            return Err(EngineError::WebSocket {
                exchange: Exchange::Kraken,
                source:   anyhow::anyhow!("max reconnects exceeded"),
            });
        }
        let d = config.backoff(attempt);
        log_reconnect(Exchange::Kraken, attempt, &format!("retry in {d:?}"));
        tokio::time::sleep(d).await;
    }
}
