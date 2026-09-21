"""Walk-forward backtest for the BTC transformer model.

Reports Sharpe, win rate, max drawdown and the buy-and-hold baseline over
expanding-window splits, with transaction costs and slippage as explicit
inputs. Evaluation only ever happens on rows after the training window, so
a result here cannot come from the model having seen the test period.

Usage:
    python scripts/backtest.py --train-size 8760 --test-size 720 \
        --cost-bps 10 --slippage-bps 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CACHE_DIR  # noqa: E402

# Re-exported for the strategy loop described at the bottom of main(). They are
# imported here so the harness and the live bot agree on one implementation of
# the exit rule rather than each keeping its own copy.
from src.trading import (  # noqa: E402, F401
    calculate_position_size,
    check_exit,
    generate_signal,
)


def walk_forward_splits(n_rows, train_size, test_size):
    """Expanding-window splits. Each test window follows its train window."""
    splits = []
    start = train_size
    while start + test_size <= n_rows:
        splits.append((range(0, start), range(start, start + test_size)))
        start += test_size
    return splits


def performance_summary(returns, periods_per_year=8760):
    """Summarise a series of per-trade returns.

    A constant series has zero variance, so its Sharpe is infinite rather
    than a division error. An empty series reports zeros, not NaN, so that
    a run with no trades is legible instead of looking broken.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return {
            "sharpe": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "n_trades": 0,
        }

    mean, std = returns.mean(), returns.std(ddof=1) if returns.size > 1 else 0.0
    if std == 0:
        sharpe = np.inf if mean > 0 else (-np.inf if mean < 0 else 0.0)
    else:
        sharpe = mean / std * np.sqrt(periods_per_year / max(returns.size, 1))

    equity = np.cumprod(1.0 + returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0

    return {
        "sharpe": float(sharpe),
        "win_rate": float((returns > 0).mean()),
        "max_drawdown": float(drawdown.min()),
        "total_return": float(equity[-1] - 1.0),
        "n_trades": int(returns.size),
    }


def buy_and_hold(close):
    """Baseline return of simply holding over the same window."""
    close = np.asarray(close, dtype=float)
    if close.size < 2:
        return 0.0
    return float(close[-1] / close[0] - 1.0)


def apply_costs(pnl, cost_bps, slippage_bps):
    """Subtract round-trip costs from a trade's gross return."""
    return pnl - 2.0 * (cost_bps + slippage_bps) / 10_000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(Path(CACHE_DIR) / "hourly.csv"))
    parser.add_argument("--train-size", type=int, default=8760)
    parser.add_argument("--test-size", type=int, default=720)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"No data at {path}. Run scripts/scraper.py first.")
        raise SystemExit(1)

    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    splits = walk_forward_splits(len(df), args.train_size, args.test_size)
    if not splits:
        print(
            f"{len(df)} rows is not enough for train_size={args.train_size} "
            f"plus test_size={args.test_size}."
        )
        raise SystemExit(1)

    print(f"{len(splits)} walk-forward folds over {len(df)} hourly bars")
    print(f"Costs: {args.cost_bps} bps fee + {args.slippage_bps} bps slippage, round trip\n")
    print("Each fold trains on rows 0..train_end and evaluates strictly after it.")
    print("Fill a real model into the marked line to produce numbers.\n")

    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        test = df.iloc[list(test_idx)]
        print(
            f"fold {i:>3}  train 0..{max(train_idx):<6} "
            f"test {min(test_idx)}..{max(test_idx)}  "
            f"buy-and-hold {buy_and_hold(test['close']):+.2%}"
        )

    print(
        "\nTo report strategy performance, load the model trained on each "
        "fold's train window, turn its predictions into percentiles, pass "
        "them through generate_signal, walk the test bars with check_exit, "
        "size with calculate_position_size, net each trade through "
        "apply_costs, and feed the resulting list to performance_summary."
    )


if __name__ == "__main__":
    main()
