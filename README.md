# btc_prediction_engine

A continuous, multi-scale BTC price prediction library in Rust.

Ticks flow in from four exchanges. Every tick updates a global state — no window resets, no cold starts. Subsystems query predictions on demand at any time-scale or window range.

---

## Architecture

```
Binance ─┐
Coinbase ─┤─► mpsc::Sender<Tick> ─► PredictionEngine (background task)
Kraken  ─┤                               │
Bitstamp─┘                               ├── TickStore     (ring buffer)
                                         ├── FeatureState  (RSI, VWAP, OFI, …)
                                         ├── Models        (EMA, heuristic, forecast)
                                         └── PredictionStore (snapshot ring)
                                                    │
                              broadcast::Sender<PredictionSnapshot>
                                                    │
                                         QueryEngine (subsystem API)
                                         ├── latest() / at(ts)
                                         ├── trend(scale)  micro/short/medium/broad
                                         ├── forecast_5s() / forecast_30s()
                                         ├── window(start, end) / window_5m / window_1h
                                         └── sub_windows(range, step)
```

---

## System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Rust (stable) | ≥ 1.75 | MSRV driven by `tokio` 1.x |
| OpenSSL dev headers | Any recent | Required for TLS. See below. |

### OpenSSL

The crate uses `tokio-tungstenite` with `native-tls` for WebSocket TLS. Install the system headers:

```bash
# Debian / Ubuntu
sudo apt install libssl-dev pkg-config

# Fedora / RHEL
sudo dnf install openssl-devel

# macOS (Homebrew)
brew install openssl
export OPENSSL_DIR=$(brew --prefix openssl)

# Windows
# Use the rustls-tls feature instead — edit Cargo.toml:
# tokio-tungstenite = { version = "0.24", features = ["rustls-tls"] }
```

To avoid OpenSSL entirely, replace `native-tls` with `rustls-tls` in `Cargo.toml` for both `tokio-tungstenite` and `reqwest`.

---

## Exchange Credentials

### Binance, Kraken, Bitstamp

Public trade streams — **no API key required**.

```rust
engine.add_feed(FeedConfig::public(Exchange::Binance,  "btcusdt")).await;
engine.add_feed(FeedConfig::public(Exchange::Kraken,   "BTC/USD")).await;
engine.add_feed(FeedConfig::public(Exchange::Bitstamp, "btcusd")).await;
```

### Coinbase Advanced Trade

Requires a **CDP (Coinbase Developer Platform)** API key with `view` scope.

1. Visit <https://portal.cdp.coinbase.com/>
2. Create an API key → download the JSON credential file.
3. The file contains:
   - `"name"` → use as `api_key`
   - `"privateKey"` → PEM-encoded EC (P-256) private key → use as `api_secret`

```rust
engine.add_feed(FeedConfig::authenticated(
    Exchange::Coinbase,
    "BTC-USD",
    std::env::var("COINBASE_KEY_NAME").unwrap(),
    std::env::var("COINBASE_PRIVATE_KEY").unwrap(),
)).await;
```

The engine regenerates the JWT every 90 seconds automatically.

---

## Quick Start

```toml
# Cargo.toml
[dependencies]
btc_prediction_engine = { path = "../btc_prediction_engine" }
tokio = { version = "1", features = ["full"] }
```

```rust
use btc_prediction_engine::prelude::*;

#[tokio::main]
async fn main() {
    // Start the engine
    let (engine, _handle) = PredictionEngine::start(EngineConfig::default()).await;

    // Attach feeds
    engine.add_feed(FeedConfig::public(Exchange::Binance,  "btcusdt")).await;
    engine.add_feed(FeedConfig::public(Exchange::Kraken,   "BTC/USD")).await;
    engine.add_feed(FeedConfig::public(Exchange::Bitstamp, "btcusd")).await;

    // Subscribe to live prediction broadcast
    let mut rx = engine.subscribe();
    tokio::spawn(async move {
        while let Ok(snap) = rx.recv().await {
            println!(
                "fused={:?} conf={:.2}  short={:?}  broad={:?}",
                snap.fused_direction, snap.fused_confidence,
                snap.short.direction, snap.broad.direction,
            );
        }
    });

    // Build a query handle for on-demand subsystem queries
    let q = QueryEngine::new(engine);

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        // Latest prediction
        if let Some(snap) = q.latest() {
            println!("price={:.2}  fused={:?}", q.latest_price().unwrap_or(0.0), snap.fused_direction);
        }

        // 5-minute window
        if let Ok(win) = q.window_5m() {
            if let Some(ohlcv) = win.ohlcv {
                println!("5m  O={:.2} H={:.2} L={:.2} C={:.2}  return={:+.4}%",
                    ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close,
                    ohlcv.return_pct() * 100.0);
            }
        }

        // 5s forecasts
        if let Some(fc) = q.forecast_5s() {
            println!("next 30s deltas: {:?}", fc.deltas);
        }
    }
}
```

---

## Plugging in Trained Models

The built-in heuristic models work with no ML setup. For production:

### Trend model (direction classifier)

```rust
use btc_prediction_engine::models::TrendModelExt;

struct MyXgbModel { /* ... */ }

impl TrendModelExt for MyXgbModel {
    fn predict(&self, f: &FeatureVector, scale: TimeScale) -> TrendSignal {
        // Map your model's output to TrendSignal
        // Input features: f.rsi, f.vwap_deviation, f.momentum_micro,
        //   f.momentum_short, f.ofi_30s, f.ofi_300s, f.tick_velocity
        todo!()
    }
}

let config = EngineConfig {
    trend_model: Some(Box::new(MyXgbModel::load("model.json"))),
    ..Default::default()
};
```

### Forecast model (LSTM Δprice sequence)

```rust
use btc_prediction_engine::models::ForecastModelExt;

struct MyLstm { /* tract-onnx session */ }

impl ForecastModelExt for MyLstm {
    fn forecast(&self, f: &FeatureVector, step_secs: u32, n_steps: usize) -> ShortTermForecast {
        // Run LSTM inference, return delta sequence
        todo!()
    }
}
```

### Recommended crates

| Purpose | Crate | Notes |
|---|---|---|
| ONNX inference | [`tract-onnx`](https://crates.io/crates/tract-onnx) | Pure Rust, no system deps, works with PyTorch exports |
| XGBoost | [`xgboost`](https://crates.io/crates/xgboost) | C-FFI, requires libxgboost |
| LightGBM | [`lightgbm`](https://crates.io/crates/lightgbm) | C-FFI, requires liblightgbm |
| Deep learning | [`candle-core`](https://crates.io/crates/candle-core) | HuggingFace pure-Rust |
| Deep learning | [`burn`](https://crates.io/crates/burn) | Full framework, WGPU/CUDA backends |

### Training data

| Source | URL |
|---|---|
| Binance historical tick data | <https://data.binance.vision> |
| Kraken OHLC REST | `GET https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1` |
| Bitstamp transactions REST | `GET https://www.bitstamp.net/api/v2/transactions/btcusd/` |
| Coinbase product candles | `GET https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles` |

---

## Feature Flags

| Flag | Default | Effect |
|---|---|---|
| `feeds-binance`  | on | Compile Binance feed |
| `feeds-coinbase` | on | Compile Coinbase feed |
| `feeds-kraken`   | on | Compile Kraken feed |
| `feeds-bitstamp` | on | Compile Bitstamp feed |
| `tracing`        | off | `tracing::info/warn/error` instead of `eprintln!` |
| `serde-state`    | off | Serde derives on internal state structs |

To enable `tracing`:

```toml
btc_prediction_engine = { path = "…", features = ["tracing"] }
```

And add a subscriber in your binary:

```rust
tracing_subscriber::fmt::init();
```

---

## Memory Budget

| Component | Default capacity | Approx. memory |
|---|---|---|
| TickStore | 1,500,000 ticks | ~120 MB |
| PredictionStore | 100,000 snapshots | ~50 MB |

At ~10 ticks/s (multi-exchange), the TickStore covers ~40 hours. Reduce `EngineConfig::tick_capacity` if memory is constrained.

---

## Running Tests

```bash
cargo test
# or with output
cargo test -- --nocapture
```

All tests are unit tests that inject ticks directly — no live network connections required.
