//! Binance Spot WebSocket trade feed — BTC/USD (symbol: `btcusdt`).
//!
//! **Endpoint:** `wss://stream.binance.com:9443/ws/btcusdt@trade?timeUnit=MICROSECOND`
//! **Alt (market-data mirror):** `wss://data-stream.binance.vision/ws/btcusdt@trade?timeUnit=MICROSECOND`
//! **Docs:** <https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md>
//!
//! ## Payload
//! ```json
//! { "e":"trade","E":1699000000000000,"s":"BTCUSDT","t":123456,
//!   "p":"34000.50","q":"0.001","T":1699000000000000,"m":false }
//! ```
//! `m=false` → taker is buyer; `m=true` → taker is seller.
//!
//! ## Limits (June 2026)
//! * 300 WS connections / IP
//! * Server pings every ~3 min; must pong within 10 min or connection drops
//! * Connection auto-closes at 24 h; this feed reconnects automatically

use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::feeds::{FeedConfig, log_error, log_info, log_reconnect};
use crate::types::{EngineError, EngineResult, Exchange, Symbol, Tick, TradeSide};

const WS_BASE: &str = "wss://stream.binance.com:9443/ws";

#[derive(Deserialize)]
struct BinanceTrade {
    #[serde(rename = "t")] trade_id:        u64,
    #[serde(rename = "p")] price:            String,
    #[serde(rename = "q")] qty:              String,
    #[serde(rename = "T")] trade_time:       i64,
    #[serde(rename = "m")] buyer_is_maker:   bool,
}

fn parse(text: &str) -> Option<Tick> {
    if !text.contains("\"trade\"") { return None; }
    let t: BinanceTrade = serde_json::from_str(text).ok()?;
    let price: f64 = t.price.parse().ok()?;
    let qty:   f64 = t.qty.parse().ok()?;
    // With ?timeUnit=MICROSECOND the field is already µs
    let ts_micros = if t.trade_time < 10_000_000_000_000 {
        t.trade_time * 1_000  // ms → µs fallback
    } else {
        t.trade_time
    };
    Some(Tick {
        ts_micros, price, quantity: qty,
        side:      Some(if t.buyer_is_maker { TradeSide::Sell } else { TradeSide::Buy }),
        exchange:  Exchange::Binance,
        symbol:    Symbol::BtcUsd,
        trade_id:  t.trade_id.to_string(),
    })
}

pub async fn run(config: FeedConfig, tx: mpsc::Sender<Tick>) -> EngineResult<()> {
    let sym = config.native_symbol();
    let url = format!("{WS_BASE}/{sym}@trade?timeUnit=MICROSECOND");
    let mut attempt = 0u32;
    loop {
        log_info(Exchange::Binance, &format!("connecting → {url}"));
        match connect_async(&url).await {
            Ok((mut ws, _)) => {
                attempt = 0;
                log_info(Exchange::Binance, "connected");
                loop {
                    match ws.next().await {
                        Some(Ok(Message::Text(t))) => {
                            if let Some(tick) = parse(&t) {
                                if tx.send(tick).await.is_err() { return Ok(()); }
                            }
                        }
                        Some(Ok(Message::Ping(d))) => { let _ = ws.send(Message::Pong(d)).await; }
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Err(e)) => { log_error(Exchange::Binance, &e.to_string()); break; }
                        _ => {}
                    }
                }
            }
            Err(e) => log_error(Exchange::Binance, &e.to_string()),
        }
        attempt += 1;
        if config.max_reconnect_attempts > 0 && attempt > config.max_reconnect_attempts {
            return Err(EngineError::WebSocket {
                exchange: Exchange::Binance,
                source:   anyhow::anyhow!("max reconnects exceeded"),
            });
        }
        let d = config.backoff(attempt);
        log_reconnect(Exchange::Binance, attempt, &format!("retry in {d:?}"));
        tokio::time::sleep(d).await;
    }
}
