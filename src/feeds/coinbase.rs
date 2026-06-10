//! Coinbase Advanced Trade WebSocket feed — BTC-USD.
//!
//! **Endpoint:** `wss://advanced-trade-ws.coinbase.com`  channel: `market_trades`
//! **Docs:** <https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-channels>
//!
//! ## Auth (required for reliable connections)
//!
//! Obtain credentials from <https://portal.cdp.coinbase.com/>:
//! * `api_key`    = `"name"` field from the downloaded JSON key file
//! * `api_secret` = `"privateKey"` field (PEM-encoded EC P-256 key)
//!
//! JWT is regenerated every 90 s automatically.

use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::feeds::{FeedConfig, log_error, log_info, log_reconnect};
use crate::types::{EngineError, EngineResult, Exchange, Symbol, Tick, TradeSide};

const WS_URL: &str = "wss://advanced-trade-ws.coinbase.com";

#[derive(Serialize)]
struct SubscribeMsg<'a> {
    #[serde(rename = "type")] msg_type: &'static str,
    product_ids: &'a [&'a str],
    channel:     &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    jwt:         Option<String>,
}

#[derive(Deserialize)]
struct CbEnvelope { channel: Option<String>, events: Option<Vec<CbEvent>> }
#[derive(Deserialize)]
struct CbEvent { #[serde(rename = "type")] ev_type: Option<String>, trades: Option<Vec<CbTrade>> }
#[derive(Deserialize)]
struct CbTrade { trade_id: String, price: String, size: String, side: String, time: String }

fn build_jwt(key: &str, pem: &str) -> Option<String> {
    use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
    let now = Utc::now().timestamp();
    let claims = serde_json::json!({ "sub": key, "iss": "cdp", "nbf": now, "exp": now + 120 });
    let mut h = Header::new(Algorithm::ES256);
    h.kid = Some(key.to_string());
    encode(&h, &claims, &EncodingKey::from_ec_pem(pem.as_bytes()).ok()?)
        .map_err(|e| log_error(Exchange::Coinbase, &format!("JWT: {e}"))).ok()
}

fn parse_msgs(text: &str) -> Vec<Tick> {
    let env: CbEnvelope = match serde_json::from_str(text) { Ok(e) => e, Err(_) => return vec![] };
    if env.channel.as_deref() != Some("market_trades") { return vec![]; }
    let mut ticks = vec![];
    for ev in env.events.into_iter().flatten() {
        if ev.ev_type.as_deref() != Some("update") { continue; }
        for t in ev.trades.into_iter().flatten() {
            let price: f64 = match t.price.parse() { Ok(v) => v, Err(_) => continue };
            let qty:   f64 = match t.size.parse()  { Ok(v) => v, Err(_) => continue };
            let side = match t.side.as_str() { "BUY" => Some(TradeSide::Buy), "SELL" => Some(TradeSide::Sell), _ => None };
            let ts_micros = chrono::DateTime::parse_from_rfc3339(&t.time)
                .map(|d| d.timestamp_micros()).unwrap_or_else(|_| Utc::now().timestamp_micros());
            ticks.push(Tick { ts_micros, price, quantity: qty, side, exchange: Exchange::Coinbase,
                              symbol: Symbol::BtcUsd, trade_id: t.trade_id });
        }
    }
    ticks
}

pub async fn run(config: FeedConfig, tx: mpsc::Sender<Tick>) -> EngineResult<()> {
    let sym = config.native_symbol();
    let mut attempt    = 0u32;
    let mut jwt_ts     = 0i64;
    let mut cached_jwt: Option<String> = None;

    loop {
        log_info(Exchange::Coinbase, &format!("connecting → {WS_URL}"));
        //let url = url::Url::parse(WS_URL).expect("static");
        match connect_async(WS_URL).await {
            Ok((mut ws, _)) => {
                attempt = 0;
                log_info(Exchange::Coinbase, "connected");
                let jwt = if let (Some(k), Some(s)) = (&config.api_key, &config.api_secret) {
                    let now = Utc::now().timestamp();
                    if now - jwt_ts > 90 { cached_jwt = build_jwt(k, s); jwt_ts = now; }
                    cached_jwt.clone()
                } else { None };

                let sub = serde_json::to_string(&SubscribeMsg {
                    msg_type: "subscribe", product_ids: &[sym],
                    channel: "market_trades", jwt: jwt.clone(),
                }).map_err(|e| EngineError::Parse { exchange: Exchange::Coinbase, source: e })?;
                if ws.send(Message::Text(sub.into())).await.is_err() { continue; }

                // heartbeats keep the connection alive during low-liquidity periods
                let hb = serde_json::to_string(&SubscribeMsg {
                    msg_type: "subscribe", product_ids: &[sym],
                    channel: "heartbeats", jwt,
                }).map_err(|e| EngineError::Parse { exchange: Exchange::Coinbase, source: e })?;
                let _ = ws.send(Message::Text(hb.into())).await;

                loop {
                    match ws.next().await {
                        Some(Ok(Message::Text(t))) => {
                            for tick in parse_msgs(&t) {
                                if tx.send(tick).await.is_err() { return Ok(()); }
                            }
                        }
                        Some(Ok(Message::Ping(d))) => { let _ = ws.send(Message::Pong(d)).await; }
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Err(e)) => { log_error(Exchange::Coinbase, &e.to_string()); break; }
                        _ => {}
                    }
                }
            }
            Err(e) => log_error(Exchange::Coinbase, &e.to_string()),
        }
        attempt += 1;
        if config.max_reconnect_attempts > 0 && attempt > config.max_reconnect_attempts {
            return Err(EngineError::WebSocket { exchange: Exchange::Coinbase, source: anyhow::anyhow!("max reconnects") });
        }
        let d = config.backoff(attempt);
        log_reconnect(Exchange::Coinbase, attempt, &format!("retry in {d:?}"));
        tokio::time::sleep(d).await;
    }
}
