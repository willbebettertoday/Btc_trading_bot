# 🤖 BTC Transformer Trading Bot

Bitcoin trading bot using Transformer neural network.

## What it does

- Predicts if BTC will go up or down
- Sends signals to Telegram
- Manages trades automatically

## Results (Backtest)

- Sharpe Ratio: 20-25
- Win Rate: 58-68%

> ⚠️ This is for learning. Don't risk money you can't lose.

## Project Structure

~~~
btc_bot/
├── README.md
├── requirements.txt
├── config_example.py
│
├── src/
│   ├── model.py        # neural network
│   ├── features.py     # calculate indicators
│   ├── trading.py      # trading logic
│   └── data.py         # load data
│
└── scripts/
    ├── train.py        # train the model
    ├── bot.py          # run live bot
    └── scraper.py      # download data
~~~

## How to run

### Option A: Local (just training, no server)

~~~bash
# 1. Install
pip install -r requirements.txt

# 2. Config
cp config_example.py config.py
# edit config.py

# 3. Download historical data (one time, ~10 min)
python scripts/scraper.py

# 4. Train model
python scripts/train.py

# Done! You have trained model in RESULTS_DIR
~~~

### Option B: Server (live trading 24/7)

~~~bash
# 1-4 same as above, then:

# 5. Setup cron to update data every 8 hours
crontab -e
# add: 0 */8 * * * cd /path/to/btc_bot && python scripts/scraper.py >> scraper.log 2>&1

# 6. Run bot
screen -S btc_bot
python scripts/bot.py
# Ctrl+A, D to detach
~~~

### Scraper modes

The scraper is smart:
- **First run**: downloads everything from 2019 (~10 min)
- **Next runs**: only downloads new data (~30 sec)

So you can use same script for initial download AND regular updates.