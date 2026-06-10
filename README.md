# btc_prediction_engine

A continuous, lock-free, multi-scale BTC/USD prediction library in Rust.

Ticks stream in from four exchanges. Every tick updates global state — no window resets, no cold starts. Subsystems query predictions on demand at any time-scale or window range.

---

## Architecture

```
Binance ─┐
Coinbase ─┤─► mpsc::Sender<Tick> ─► Pipeline (5 concurrent async stages)
Kraken  ─┤
Bitstamp─┘
              Stage 1 │ Dedup + outlier filter
              Stage 2 │ Price fusion (NTP drift correction + 100 ms VWAP)
              Stage 3 │ Feature engineering (RSI, VWAP, OFI, momentum, …)
              Stage 4 │ Model inference (parallel per TimeScale + forecasts)
              Stage 5 │ Fanout → TickStore · PredictionStore · ArcSwap · broadcast

                                    │
                    ┌───────────────┼───────────────────┐
                    │               │                   │
              zero-lock        broadcast           QueryEngine
              ArcSwap load    Receiver<snap>       window / trend / forecast
```

---

## System Requirements

| Requirement       | Version      | Notes                          |
|-------------------|--------------|--------------------------------|
| Rust (stable)     | ≥ 1.75       | MSRV driven by `tokio` 1.x     |
| OpenSSL dev headers | Any recent | Required for TLS. See below.   |

### OpenSSL

The crate uses `tokio-tungstenite` with `native-tls`. Install the system headers:

```bash
# Debian / Ubuntu
sudo apt install libssl-dev pkg-config

# Fedora / RHEL
sudo dnf install openssl-devel

# macOS (Homebrew)
brew install openssl
export OPENSSL_DIR=$(brew --prefix openssl)

# Windows — use rustls-tls instead (edit Cargo.toml):
# tokio-tungstenite = { version = "0.24", features = ["rustls-tls"] }
# reqwest           = { version = "0.12", features = ["rustls-tls"], default-features = false }
```

---

## Exchange Credentials

### Binance, Kraken, Bitstamp

Public trade streams — **no API key required**.

```rust
engine.add_feed(FeedConfig::public(Exchange::Binance,  Symbol::BtcUsd));
engine.add_feed(FeedConfig::public(Exchange::Kraken,   Symbol::BtcUsd));
engine.add_feed(FeedConfig::public(Exchange::Bitstamp, Symbol::BtcUsd));
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
    Symbol::BtcUsd,
    std::env::var("COINBASE_KEY_NAME").unwrap(),
    std::env::var("COINBASE_PRIVATE_KEY").unwrap(),
));
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
    let (engine, _handles) = PredictionEngine::start(EngineConfig::default()).await;

    // Attach feeds (add_feed is synchronous — spawns a background task)
    engine.add_feed(FeedConfig::public(Exchange::Binance,  Symbol::BtcUsd));
    engine.add_feed(FeedConfig::public(Exchange::Kraken,   Symbol::BtcUsd));
    engine.add_feed(FeedConfig::public(Exchange::Bitstamp, Symbol::BtcUsd));

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

        // Latest prediction (zero-lock ArcSwap read)
        if let Some(snap) = q.latest() {
            println!("price={:.2}  fused={:?}", snap.price, snap.fused_direction);
        }

        // 5-minute window OHLCV
        if let Ok(win) = q.window_5m() {
            if let Some(ohlcv) = win.ohlcv {
                println!("5m  O={:.2} H={:.2} L={:.2} C={:.2}  return={:+.4}%",
                    ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close,
                    ohlcv.return_pct() * 100.0);
            }
        }

        // Short-term forecasts
        if let Some(fc) = q.forecast_5s() {
            println!("next 30s Δprice steps: {:?}", fc.deltas);
        }
    }
}
```

---

## Plugging in Trained Models

The built-in heuristic models work with no ML setup. For production, implement the extension traits and pass them via `PipelineConfig`.

### Trend model (direction classifier)

```rust
use btc_prediction_engine::{models::TrendModelExt, prelude::*};

struct MyXgbModel { /* ... */ }

impl TrendModelExt for MyXgbModel {
    fn predict(&self, f: &FeatureVector, scale: TimeScale) -> TrendSignal {
        // 13 input features — see Feature Vector section below
        todo!()
    }
}

let config = EngineConfig {
    pipeline: PipelineConfig {
        ext_trend: Some(Box::new(MyXgbModel::load("trend.json"))),
        ..Default::default()
    },
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

let config = EngineConfig {
    pipeline: PipelineConfig {
        ext_forecast: Some(Box::new(MyLstm::load("lstm.onnx"))),
        ..Default::default()
    },
    ..Default::default()
};
```

### Feature vector

All 13 fields available at inference time:

| Field                   | Type          | Range / unit          | Description                         |
|-------------------------|---------------|-----------------------|-------------------------------------|
| `price`                 | `f64`         | USD                   | Current BTC/USD price               |
| `rsi_14`                | `Option<f64>` | [0, 100]              | Wilder RSI, 14-tick period          |
| `vwap_deviation`        | `Option<f64>` | fraction              | (price − session VWAP) / VWAP       |
| `momentum_micro`        | `Option<f64>` | fraction              | (p_now − p_30ago) / p_30ago         |
| `momentum_short`        | `Option<f64>` | fraction              | (p_now − p_300ago) / p_300ago       |
| `ewma_vol_tick`         | `Option<f64>` | fraction              | Per-tick EWMA σ                     |
| `ewma_variance`         | `f64`         | fraction²             | EWMA price variance                 |
| `tick_velocity`         | `f64`         | ticks/s               | 30-s rolling tick rate              |
| `ofi_30s`               | `f64`         | [−1, 1]               | Order flow imbalance, 30 s          |
| `ofi_300s`              | `f64`         | [−1, 1]               | Order flow imbalance, 300 s         |
| `autocorr_lag1`         | `Option<f64>` | [−1, 1]               | Lag-1 return autocorrelation        |
| `realised_vol_30s`      | `Option<f64>` | fraction              | Realised volatility, 30-s window    |
| `inter_exchange_spread` | `f64`         | USD                   | max − min last price per exchange   |

Export helper for training pipelines:

```rust
pub fn feature_array(f: &FeatureVector) -> [f64; 13] {
    [
        f.rsi_14.unwrap_or(50.0) / 100.0,
        f.vwap_deviation.unwrap_or(0.0),
        f.momentum_micro.unwrap_or(0.0),
        f.momentum_short.unwrap_or(0.0),
        f.ewma_vol_tick.unwrap_or(0.001),
        f.tick_velocity / 20.0,
        f.ofi_30s,
        f.ofi_300s,
        f.autocorr_lag1.unwrap_or(0.0),
        f.realised_vol_30s.unwrap_or(0.001),
        f.inter_exchange_spread / 100.0,
        (f.price - 30_000.0) / 70_000.0,   // normalised BTC price
        f.ewma_variance,
    ]
}
```

### Recommended crates

| Purpose        | Crate                                                             | Notes                                    |
|----------------|-------------------------------------------------------------------|------------------------------------------|
| ONNX inference | [`tract-onnx`](https://crates.io/crates/tract-onnx)               | Pure Rust, no system deps, PyTorch exports |
| XGBoost        | [`xgboost`](https://crates.io/crates/xgboost)                     | C-FFI, requires libxgboost               |
| LightGBM       | [`lightgbm`](https://crates.io/crates/lightgbm)                   | C-FFI, requires liblightgbm              |
| Deep learning  | [`candle-core`](https://crates.io/crates/candle-core)             | HuggingFace, pure Rust                   |
| Deep learning  | [`burn`](https://crates.io/crates/burn)                           | Full framework, WGPU/CUDA backends       |

### Training data sources

| Source                    | URL                                                                                    |
|---------------------------|----------------------------------------------------------------------------------------|
| Binance historical ticks  | <https://data.binance.vision>                                                          |
| Kraken OHLC REST          | `GET https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1`                     |
| Bitstamp transactions REST| `GET https://www.bitstamp.net/api/v2/transactions/btcusd/`                             |
| Coinbase product candles  | `GET https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles`        |

---

## Feature Flags

| Flag             | Default | Effect                                              |
|------------------|---------|-----------------------------------------------------|
| `feeds-binance`  | on      | Compile Binance feed                                |
| `feeds-coinbase` | on      | Compile Coinbase feed (requires JWT credentials)    |
| `feeds-kraken`   | on      | Compile Kraken feed                                 |
| `feeds-bitstamp` | on      | Compile Bitstamp feed                               |
| `metrics`        | off     | Prometheus `/metrics` endpoint                      |
| `tracing`        | off     | `tracing` crate instrumentation                     |
| `persistence`    | off     | Snapshot save/load via bincode + zstd               |

To enable `tracing`:

```toml
btc_prediction_engine = { path = "…", features = ["tracing"] }
```

Add a subscriber in your binary:

```rust
tracing_subscriber::fmt::init();
```

---

## Memory Budget

| Component         | Default capacity   | Approx. memory | Coverage at ~15 ticks/s |
|-------------------|--------------------|----------------|-------------------------|
| `TickStore`       | 1,500,000 ticks    | ~150 MB        | ~28 hours               |
| `PredictionStore` | 100,000 snapshots  | ~50 MB         | —                       |

Reduce `EngineConfig::tick_capacity` or `EngineConfig::pred_capacity` if memory is constrained.

---

## Running Tests

```bash
cargo test
# with output
cargo test -- --nocapture
```

All tests inject ticks directly — no live network connections required.
