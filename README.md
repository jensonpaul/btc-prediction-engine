# btc_prediction_engine

A continuous, lock-free, multi-scale BTC/USD prediction library in Rust.

Ticks stream in from four exchanges. Every tick updates global state — no window resets, no cold starts. Subsystems query predictions on demand at any time-scale or window range.

---

## Architecture

```
Binance ─┐                           Binance ─┐
Coinbase ─┤─► mpsc::Sender<Tick>     Kraken  ─┤─► mpsc::Sender<BookSnapshot>
Kraken  ─┤                           Bitstamp─┘   (100 ms depth snapshots)
Bitstamp─┘

              │                                         │
              └──────────────┬──────────────────────────┘
                             ▼
              Pipeline (5 concurrent async stages)

              Stage 1   │ Dedup + outlier filter
              Stage 1.5 │ Price fusion (NTP drift correction + 100 ms VWAP)
              Stage 2   │ Feature engineering — select! over fused ticks + book snapshots
                        │   RSI, VWAP, OFI, momentum, book_imbalance_top5, …
              Stage 3   │ Model inference (parallel per TimeScale + forecasts)
              Stage 4   │ Fanout → TickStore · PredictionStore · ArcSwap · broadcast

                                    │
                    ┌───────────────┼───────────────────┐
                    │               │                   │
              zero-lock        broadcast           QueryEngine
              ArcSwap load    Receiver<snap>       window / trend / forecast
```

---

## System Requirements

| Requirement         | Version      | Notes                         |
|---------------------|--------------|-------------------------------|
| Rust (stable)       | ≥ 1.75       | MSRV driven by `tokio` 1.x    |
| OpenSSL dev headers | Any recent   | Required for TLS. See below.  |

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

## Order Book Feeds

Binance, Kraken, and Bitstamp all expose a Level 2 depth feed on the same WebSocket connection as the trade feed. Attaching them enables the `book_imbalance_top5` and related features — the highest-signal features at sub-minute timescales.

```rust
// All three are public — no API key required
engine.add_book_feed(BookFeedConfig::public(Exchange::Binance,  Symbol::BtcUsd));
engine.add_book_feed(BookFeedConfig::public(Exchange::Kraken,   Symbol::BtcUsd));
engine.add_book_feed(BookFeedConfig::public(Exchange::Bitstamp, Symbol::BtcUsd));
```

Book feeds are optional. When none are connected, `book_imbalance_top5`, `book_imbalance_full`, `book_weighted_mid`, and `book_spread_usd` on `FeatureVector` are `None`. The rest of the engine runs normally.

| Exchange | Stream                    | Update interval | Mechanism                              |
|----------|---------------------------|-----------------|----------------------------------------|
| Binance  | `btcusdt@depth10@100ms`   | 100 ms          | Full snapshot pushed each update       |
| Kraken   | `book` channel, depth 10  | On change       | Initial snapshot + incremental diffs   |
| Bitstamp | `order_book_btcusd`       | On change       | Full snapshot pushed each update       |

Coinbase Advanced Trade does not provide an equivalent public order book WebSocket.

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

    // Order book feeds — top-5 bid/ask imbalance, highest-signal sub-minute feature
    engine.add_book_feed(BookFeedConfig::public(Exchange::Binance,  Symbol::BtcUsd));
    engine.add_book_feed(BookFeedConfig::public(Exchange::Kraken,   Symbol::BtcUsd));
    engine.add_book_feed(BookFeedConfig::public(Exchange::Bitstamp, Symbol::BtcUsd));

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

When an external model is loaded, every `PredictionSnapshot` carries **both** the model's output and the heuristic baseline simultaneously — `model_active` tells you which is which. See [Heuristic Baseline](#heuristic-baseline) below.

### Trend model (direction classifier)

```rust
use btc_prediction_engine::{models::TrendModelExt, prelude::*};

struct MyXgbModel { /* ... */ }

impl TrendModelExt for MyXgbModel {
    fn predict(&self, f: &FeatureVector, scale: TimeScale) -> TrendSignal {
        // 17 input features — see Feature Vector section below
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

### ONNX trend model (btc-onnx-trend-model)

A ready-made [`TrendModelExt`] wrapper for LightGBM (or any sklearn-compatible) classifiers exported with `skl2onnx` is available as a separate crate:

```toml
[dependencies]
btc-onnx-trend-model = { path = "../btc-onnx-trend-model" }
```

```rust
use btc_onnx_trend_model::OnnxTrendModel;
use btc_prediction_engine::pipeline::PipelineConfig;

let onnx_model = OnnxTrendModel::load("models/direction_model.onnx")
    .expect("failed to load direction_model.onnx");

let config = EngineConfig {
    pipeline: PipelineConfig {
        ext_trend: Some(Box::new(onnx_model)),
        ..PipelineConfig::default()
    },
    ..EngineConfig::default()
};
let (engine, _handles) = PredictionEngine::start(config).await;
```

For build-time embedding (no external file dependency at runtime):

```rust
const MODEL_BYTES: &[u8] = include_bytes!("../../models/direction_model.onnx");
let onnx_model = OnnxTrendModel::load_from_bytes(MODEL_BYTES).expect("...");
```

### Heuristic baseline

Regardless of whether `ext_trend` is set, every `PredictionSnapshot` always contains a `heuristic` field with the built-in RSI/OFI/EMA signals:

```rust
pub struct HeuristicSnapshot {
    pub micro:            TrendSignal,
    pub short:            TrendSignal,
    pub medium:           TrendSignal,
    pub broad:            TrendSignal,
    pub fused_direction:  TrendDirection,
    pub fused_confidence: f64,
}
```

`PredictionSnapshot` exposes:

| Field          | Type                | Meaning                                                              |
|----------------|---------------------|----------------------------------------------------------------------|
| `micro`/`short`/`medium`/`broad` | `TrendSignal` | Primary signals — from `ext_trend` when loaded, heuristic otherwise |
| `fused_direction` / `fused_confidence` | — | Fused primary signals                               |
| `heuristic`    | `HeuristicSnapshot` | Built-in baseline, **always populated**                              |
| `model_active` | `bool`              | `true` when primary signals come from an external model              |

When `model_active` is `false`, the top-level signals and `heuristic` are identical. When `true` they differ, enabling side-by-side comparison in the terminal or any downstream consumer:

```rust
let snap = engine.latest_snapshot().unwrap();

if snap.model_active {
    println!("[ONNX]      fused={:?}  conf={:.2}", snap.fused_direction, snap.fused_confidence);
    println!("[Heuristic] fused={:?}  conf={:.2}", snap.heuristic.fused_direction, snap.heuristic.fused_confidence);
} else {
    println!("[Heuristic] fused={:?}  conf={:.2}", snap.fused_direction, snap.fused_confidence);
}
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

All 17 fields available at inference time:

| Field                   | Type          | Range / unit | Description                        |
|-------------------------|---------------|--------------|------------------------------------|
| `price`                 | `f64`         | USD          | Current BTC/USD price              |
| `rsi_14`                | `Option<f64>` | [0, 100]     | Wilder RSI, 14-tick period         |
| `vwap_deviation`        | `Option<f64>` | fraction     | (price − session VWAP) / VWAP      |
| `momentum_micro`        | `Option<f64>` | fraction     | (p_now − p_30ago) / p_30ago        |
| `momentum_short`        | `Option<f64>` | fraction     | (p_now − p_300ago) / p_300ago      |
| `ewma_vol_tick`         | `Option<f64>` | fraction     | Per-tick EWMA σ                    |
| `ewma_variance`         | `f64`         | fraction²    | EWMA price variance                |
| `tick_velocity`         | `f64`         | ticks/s      | 30-s rolling tick rate             |
| `ofi_30s`               | `f64`         | [−1, 1]      | Order flow imbalance, 30 s         |
| `ofi_300s`              | `f64`         | [−1, 1]      | Order flow imbalance, 300 s        |
| `autocorr_lag1`         | `Option<f64>` | [−1, 1]      | Lag-1 return autocorrelation       |
| `realised_vol_30s`      | `Option<f64>` | fraction     | Realised volatility, 30-s window   |
| `inter_exchange_spread` | `f64`         | USD          | max − min last price per exchange  |
| `book_imbalance_top5`   | `Option<f64>` | [−1, 1]      | Top-5 bid/ask volume imbalance     |
| `book_imbalance_full`   | `Option<f64>` | [−1, 1]      | Full-depth bid/ask imbalance       |
| `book_weighted_mid`     | `Option<f64>` | USD          | Volume-weighted mid-price          |
| `book_spread_usd`       | `Option<f64>` | USD          | Best bid–ask spread                |

Export helper for training pipelines:

```rust
pub fn feature_array(f: &FeatureVector) -> [f64; 17] {
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
        // Book features: 0.0 (neutral) when no book feed is connected
        f.book_imbalance_top5.unwrap_or(0.0),
        f.book_imbalance_full.unwrap_or(0.0),
        f.book_weighted_mid.map(|m| (m - 30_000.0) / 70_000.0).unwrap_or(0.0),
        f.book_spread_usd.map(|s| s / 100.0).unwrap_or(0.0),
    ]
}
```

### Recommended crates

| Purpose        | Crate                                                           | Notes                                     |
|----------------|-----------------------------------------------------------------|-------------------------------------------|
| ONNX inference | [`tract-onnx`](https://crates.io/crates/tract-onnx)             | Pure Rust, no system deps, PyTorch exports |
| XGBoost        | [`xgboost`](https://crates.io/crates/xgboost)                   | C-FFI, requires libxgboost                |
| LightGBM       | [`lightgbm`](https://crates.io/crates/lightgbm)                 | C-FFI, requires liblightgbm               |
| Deep learning  | [`candle-core`](https://crates.io/crates/candle-core)           | HuggingFace, pure Rust                    |
| Deep learning  | [`burn`](https://crates.io/crates/burn)                         | Full framework, WGPU/CUDA backends        |

### Training data sources

| Source                     | URL                                                                               |
|----------------------------|-----------------------------------------------------------------------------------|
| Binance historical ticks   | <https://data.binance.vision>                                                     |
| Kraken OHLC REST           | `GET https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1`                |
| Bitstamp transactions REST | `GET https://www.bitstamp.net/api/v2/transactions/btcusd/`                        |
| Coinbase product candles   | `GET https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles`   |

---

## Feature Flags

| Flag             | Default | Effect                                           |
|------------------|---------|--------------------------------------------------|
| `feeds-binance`  | on      | Compile Binance feed                             |
| `feeds-coinbase` | on      | Compile Coinbase feed (requires JWT credentials) |
| `feeds-kraken`   | on      | Compile Kraken feed                              |
| `feeds-bitstamp` | on      | Compile Bitstamp feed                            |
| `metrics`        | off     | Prometheus `/metrics` endpoint                   |
| `tracing`        | off     | `tracing` crate instrumentation                  |
| `persistence`    | off     | Snapshot save/load via bincode + zstd            |

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

| Component         | Default capacity  | Approx. memory | Coverage at ~15 ticks/s |
|-------------------|-------------------|----------------|-------------------------|
| `TickStore`       | 1,500,000 ticks   | ~150 MB        | ~28 hours               |
| `PredictionStore` | 100,000 snapshots | ~50 MB         | —                       |

Reduce `EngineConfig::tick_capacity` or `EngineConfig::pred_capacity` if memory is constrained.

---

## Running Tests

```bash
cargo test
# with output
cargo test -- --nocapture
```

All tests inject ticks directly — no live network connections required.
