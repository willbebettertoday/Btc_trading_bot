# 🤖 BTC Transformer Trading Bot

I trained a Transformer on hourly BTC data to predict short-term price movement, then wired the prediction into a rule-based signal, position-sizing and exit system that posts to Telegram.

[![ci](https://github.com/willbebettertoday/Btc_trading_bot/actions/workflows/ci.yml/badge.svg)](https://github.com/willbebettertoday/Btc_trading_bot/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

## Visual

No screenshot, chart or demo GIF exists for this project. There is no trained model or backtest result to show yet (see Results below), and the live bot's only output is a Telegram text message, which is not worth a screenshot on its own.

## Problem

I wanted to see whether a Transformer trained on hourly BTC OHLCV, plus a handful of on-chain and sentiment series, could predict short-term price movement well enough to trade, packaged as a bot rather than a notebook that stops at a metrics cell. The repository's own history turned into a more interesting problem than the one I started with: it published a Sharpe ratio of 20 to 25 and a 58 to 68% win rate with no test suite and no backtest script anywhere in it, so nobody, including me, could check where those numbers came from. That is exactly why Validation and limitations below carries more weight on this page than Results does.

## Approach

The pipeline runs data to features to Transformer to a predicted percentile to a rule-based signal to a rule-based exit. `src/features.py` builds momentum, volatility, RSI, MACD and multi-timeframe momentum features from BTC OHLCV, optionally enriched with ETH and gold correlation, hash rate, funding rate and Fear & Greed data when those series are available. `src/model.py`'s Transformer encoder predicts the percentile rank of the next hour's return rather than the return itself, and `src/trading.py` turns that percentile into a LONG or SHORT signal with confidence-scaled take-profit and stop-loss levels.

The one choice worth calling out: every statistic fitted on data, the `RobustScaler` in `scripts/train.py` and the outlier-clip bounds in `src/features.py`, is fit on the training split only and saved to disk (`scaler.json`, `clip_bounds.json`) so every later split, including the live bot, reuses those exact numbers instead of refitting on data it has no business seeing yet.

## Results

I removed the Sharpe 20-25 and 58-68% win-rate figures that used to be here. They were computed by a feature pipeline with two look-ahead leaks (see How it works), so they are not merely unverified, they are known to be wrong: past feature rows depended on rows that had not happened yet, and a Sharpe ratio or win rate measured on inputs like that is measuring the leak, not a strategy.

I have not retrained the model since fixing the leaks, so there is no honest number to put in this table yet. `scripts/backtest.py` is the harness that will produce one: it builds expanding-window walk-forward splits so evaluation always happens strictly after training, reports each fold's buy-and-hold baseline, and its `performance_summary` function is ready to score a trained model's trades (Sharpe, win rate, max drawdown, total return) net of explicit fee and slippage inputs, once a trained model is wired into the loop. Until that run happens, this section stays empty rather than guessing.

## How it works

1. `scripts/scraper.py` downloads hourly BTC, ETH and gold OHLCV from Binance, plus funding rates, hash rate and the Fear & Greed index, caching each to CSV. A first run pulls history back to 2019; later runs fetch only new rows.
2. `src/features.py`'s `create_features` turns raw OHLCV into the indicators above. `fit_clip_bounds` and `apply_clip_bounds` keep outlier clipping split between the training split and every later split, so a feature at time `t` never depends on a row after `t`. `tests/test_features.py::test_no_feature_uses_future_data` asserts this directly: it found real look-ahead leakage twice during this cleanup. First, the clip bounds were fitted across the whole frame instead of the training split alone, leaking future rows into nine of eleven feature columns. After that fix, the same test still failed: `resample('4h'|'1D').agg({'close': 'last'})` labelled each bucket at its start while storing the bucket's last value, so the forward fill spread a bucket's closing price backwards across its own earlier hours, letting the model read a price up to 22 hours ahead of when it was actually known. Both are fixed now (`tests/test_timeframe_features.py` covers the second one directly), and the invariant test passes without being edited.
3. `scripts/train.py` fits the `RobustScaler` and the clip bounds on the training split only, trains the Transformer to predict the percentile rank of the next hour's return, and saves the model, scaler, clip bounds and config to `RESULTS_DIR`.
4. `src/trading.py` turns a predicted percentile into a signal above or below configurable thresholds, sizes the position by risk per trade, and evaluates exits. `check_exit` checks the stop loss before the take profit, so a bar that touches both levels books the loss, since one hourly bar cannot tell you which level was actually touched first.
5. `scripts/bot.py` runs this loop live against Binance, refuses to start if `clip_bounds.json` is missing rather than running unclipped, and posts signals and exits to Telegram.
6. `scripts/backtest.py` is the offline harness for checking any of this: the walk-forward splits, buy-and-hold baseline and cost accounting described in Results.

## Reproduce it

```bash
git clone https://github.com/willbebettertoday/Btc_trading_bot.git
cd Btc_trading_bot
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest
pytest -q
```

This runs the 58 tests. They stub the `config` module (`tests/conftest.py`), so they need no real market data, no `config.py` and no trained model.

Training needs data this repository does not ship:

```bash
cp config_example.py config.py    # edit CACHE_DIR, RESULTS_DIR and the Telegram values
python scripts/scraper.py         # downloads BTC/ETH/gold OHLCV, funding, hash rate, Fear & Greed
python scripts/train.py           # trains the Transformer on whatever scraper.py just downloaded
python scripts/backtest.py --data <your CACHE_DIR>/btc_1h.csv
```

The `--data` flag on the last line is required: `scripts/backtest.py` defaults to `<CACHE_DIR>/hourly.csv`, but `scripts/scraper.py` writes `btc_1h.csv`, so the default path never exists. Replace `<your CACHE_DIR>` with the path set in `config.py`.

## Validation and limitations

Read this before the (currently empty) Results section above.

- **The previously published figures are gone, on purpose.** A Sharpe of 20-25 and a 58-68% win rate came out of a pipeline that could see the future twice over. Removing them is not caution, it is correction: they were wrong, not merely unproven.
- **The model has not been retrained since the leaks were fixed.** Every number this repository has ever produced predates the fix, so there is nothing trustworthy to report yet, and I have not invented a replacement.
- **`config_example.py` ships runnable defaults, not tuned values.** A fresh clone will train and run end to end without editing the signal, risk or model parameters, but nothing about the result is expected to reproduce any particular figure. The dates it ships (`START_DATE`, `TRAIN_END`, `VAL_END`) still describe the historical split `scripts/train.py` uses.
- **`scripts/backtest.py` is an evaluation harness, not a completed experiment.** It produces fold boundaries, a buy-and-hold baseline per fold, and a `performance_summary` ready to score trades. The strategy loop, loading a trained model, turning its output into signals via `generate_signal`, walking the test bars with `check_exit`, sizing with `calculate_position_size`, and netting each trade through `apply_costs`, still needs to be wired in and run.
- **Exits are evaluated pessimistically.** `check_exit` checks the stop loss before the take profit, so a bar touching both levels books the loss, never the win.
- **`momentum_daily_7d` and `momentum_daily_30d` are coarse by construction.** They now measure exactly the spans their names claim, seven and thirty calendar days, but they are derived from a daily close series resampled from hourly bars, so intraday moves are invisible to them and their values only change once a day.
- **The look-ahead invariant tests do not cover the optional external-data features.** `test_no_feature_uses_future_data` and the multi-timeframe causality tests in `tests/test_timeframe_features.py` only exercise the BTC-only feature path. The eight feature columns built from five optional external sources (`eth_momentum`, `btc_eth_corr`, `gold_momentum`, `btc_gold_corr`, `hashrate_momentum`, `hashrate_ratio`, `funding`, `fear_greed`) are never constructed in any test, so I can say they are shifted by `external_shift` where the code applies it, but I cannot say the test suite has verified they are causal.
- This was built to learn, not to trade with. Nothing on this page is investment advice, and no live trading has happened since the leak fixes above.

## Tech stack

- **ML:** PyTorch (Transformer encoder), scikit-learn (`RobustScaler`)
- **Data:** ccxt (Binance OHLCV and funding rates), requests (mempool.space hash rate, alternative.me Fear & Greed), pandas, numpy
- **Live bot:** raw Telegram Bot API over `requests`, SQLite for open-trade state
- **Testing / CI:** pytest, ruff, GitHub Actions on Python 3.11 and 3.13

## Project structure

```
Btc_trading_bot/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   └── trading.py
├── scripts/
│   ├── backtest.py
│   ├── bot.py
│   ├── scraper.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_backtest.py
│   ├── test_clip_bounds.py
│   ├── test_features.py
│   ├── test_timeframe_features.py
│   └── test_trading.py
├── config_example.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Danil Sysenko.
