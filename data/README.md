# 📥 Data Collection Module

This module handles automated downloading and caching of cryptocurrency market data.

## Data Sources

### 1. Hash Rate (mempool.space)
- **What**: Bitcoin network hash rate
- **Update**: Real-time (<1h lag)
- **Importance**: 7% model importance (Top 3 feature!)
- **Format**: Converted to TH/s

### 2. OHLCV Data (Binance API)
- **BTC**: Bitcoin futures hourly candles
- **ETH**: Ethereum futures hourly candles
- **Gold**: PAXG/USDT spot market (safe haven correlation)
- **Update**: Every hour
- **Free**: Yes

### 3. Fear & Greed Index (alternative.me)
- **What**: Market sentiment indicator (0-100)
- **Update**: Daily
- **Use**: Contrarian signal

### 4. Funding Rates (Binance Futures)
- **What**: Long/short market bias
- **Update**: Every 8 hours
- **Use**: Sentiment indicator

## Usage

### Single Run
```bash
python scraper.py
```

### Automated (Cron)
```bash
# Edit crontab
crontab -e

# Add this line (runs every hour at :05)
5 * * * * cd /path/to/project/data && python3 scraper.py >> scraper.log 2>&1
```

### Check Logs
```bash
tail -f scraper.log
```

## Output

Data saved to:
```
../binance_hourly_cache/
   ├── btc_1h.csv       (~3.5 MB)
   ├── eth_1h.csv       (~3.3 MB)
   ├── gold_1h.csv      (~2.6 MB)
   ├── hash_rate.csv    (~5.4 MB)
   └── funding_1h.csv   (~1.6 MB)

../binance_daily_cache/
   └── fear_greed.csv   (~100 KB)
```

## Requirements

```bash
pip install -r requirements.txt
```

Packages:
- pandas (data manipulation)
- requests (API calls)
- ccxt (exchange API wrapper)

## Troubleshooting

**Problem**: Hash rate not updating
```bash
# Check mempool.space API
curl https://mempool.space/api/v1/mining/hashrate/3m

# If fails, fallback to blockchain.info (has lag)
```

**Problem**: Binance API errors
```bash
# Check rate limits (scraper respects them automatically)
# Max 1200 requests per minute (we use ~100)
```

**Problem**: Old data
```bash
# Re-download everything
rm -rf ../binance_hourly_cache/*
python scraper.py
```

## API Limits

- **Mempool.space**: No limit, free
- **Binance**: 1200 req/min (we use ~100)
- **Alternative.me**: No limit, free

---
