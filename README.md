# Crypto Linear Model

An quantitative trading system specifically designed for cryptocurrency markets, combining ARIMA forecasting, machine learning predictions, enhanced sentiment analysis, and portfolio optimization to generate crypto investment recommendations.

## Features

- **Crypto-Optimized Multi-Model Forecasting**
  - ARIMA time series forecasting with MSE evaluation
  - Machine Learning predictions with cross-validation
  - Combined forecast approach for volatile crypto markets

- **Enhanced Crypto Sentiment Analysis**
  - Integration with Google News API (70% weight for crypto)
  - YFinance news integration (30% weight for crypto)
  - Crypto-specific search terms (price, market, blockchain)
  - Increased sentiment impact (1.2x) for crypto assets

- **Specialized Portfolio Optimization**
  - Risk-adjusted return optimization
  - Dynamic weight allocation with crypto constraints
  - Minimum position size: 1%
  - Maximum position size: 40%

- **Crypto-Focused Risk Management**
  - Volatility regime detection
  - Dynamic position sizing
  - Enhanced weekend volatility handling
  - Crypto-specific Kelly Criterion adjustments

## Prerequisites

- Python 3.9+
- Required packages:
  ```
  yfinance
  statsmodels
  scikit-learn
  pandas
  numpy
  pygooglenews
  textblob
  cvxpy
  matplotlib
  ```

## Installation

1. Install all required dependencies:
   ```bash
   pip install numpy pandas yfinance matplotlib statsmodels scikit-learn cvxpy textblob pygooglenews --upgrade
   ```

## Usage

1. Run the main script:
   ```bash
   python linear.py
   ```

2. The system will analyze a predefined set of cryptocurrencies including:
   - BTC-USD (Bitcoin)
   - ETH-USD (Ethereum)
   - SOL-USD (Solana)
   - S-USD (Sonic)
   - BERA-USD (Bera)
   - And other major cryptocurrencies

## Example Output

```
python Linear_model_cryp
to/linear.py
YF.download() has changed argument auto_adjust default to True
[*********************100%***********************]  16 of 16 completed

Analyzing assets...


Analyzing BTC-USD...

Sentiment Analysis Results for BTC-USD:
YFinance Articles: 0
Google News Articles: 20
Search terms: BTC price, BTC crypto market, BTC cryptocurrency news, BTC blockchain
Combined sentiment: 0.056
Interpretation: Neutral
ARIMA forecast: Last price = 95539.55, Forecast = 95577.59, Daily return = 0.04%, Annual = 10.03%, MSE = 1752639.177518
ML forecast: Daily return = 0.27%, Annual = 69.26%, Avg CV MSE = 0.000694

Analyzing ETH-USD...

Sentiment Analysis Results for ETH-USD:
YFinance Articles: 0
Google News Articles: 19
Search terms: ETH price, ETH crypto market, ETH cryptocurrency news, ETH blockchain
Combined sentiment: 0.010
Interpretation: Neutral
ARIMA forecast: Last price = 2669.34, Forecast = 2669.24, Daily return = -0.00%, Annual = -0.94%, MSE = 7027.977601
ML forecast: Daily return = 0.21%, Annual = 51.80%, Avg CV MSE = 0.001069

Analyzing SOL-USD...

Sentiment Analysis Results for SOL-USD:
YFinance Articles: 0
Google News Articles: 18
Search terms: SOL price, SOL crypto market, SOL cryptocurrency news, SOL blockchain
Combined sentiment: 0.009
Interpretation: Neutral
ARIMA forecast: Last price = 169.08, Forecast = 168.02, Daily return = -0.63%, Annual = -95.00%, MSE = 25.209110
ML forecast: Daily return = -0.09%, Annual = -22.14%, Avg CV MSE = 0.002632

Analyzing HYPE32196-USD...

Sentiment Analysis Results for HYPE32196-USD:
YFinance Articles: 0
Google News Articles: 20
Search terms: HYPE price, HYPE crypto market, HYPE cryptocurrency news, HYPE blockchain
Combined sentiment: 0.113
Interpretation: Positive
ARIMA forecast: Last price = 23.95, Forecast = 24.01, Daily return = 0.25%, Annual = 63.21%, MSE = 3.054756
ML forecast: Daily return = 2.12%, Annual = 533.60%, Avg CV MSE = 0.007715

Analyzing ALGO-USD...

Sentiment Analysis Results for ALGO-USD:
YFinance Articles: 0
Google News Articles: 9
Search terms: ALGO price, ALGO crypto market, ALGO cryptocurrency news, ALGO blockchain
Combined sentiment: 0.029
Interpretation: Neutral
ARIMA forecast: Last price = 0.26, Forecast = 0.26, Daily return = -0.58%, Annual = -95.00%, MSE = 0.000382
ML forecast: Daily return = 0.46%, Annual = 116.07%, Avg CV MSE = 0.002361

Analyzing FET-USD...

Sentiment Analysis Results for FET-USD:
YFinance Articles: 0
Google News Articles: 7
Search terms: FET price, FET crypto market, FET cryptocurrency news, FET blockchain
Combined sentiment: 0.000
Interpretation: Neutral
ARIMA forecast: Last price = 0.73, Forecast = 0.73, Daily return = 0.21%, Annual = 52.57%, MSE = 0.004388
ML forecast: Daily return = 0.78%, Annual = 197.73%, Avg CV MSE = 0.004228

Analyzing AVAX-USD...

Sentiment Analysis Results for AVAX-USD:
YFinance Articles: 0
Google News Articles: 18
Search terms: AVAX price, AVAX crypto market, AVAX cryptocurrency news, AVAX blockchain
Combined sentiment: 0.075
Interpretation: Neutral
ARIMA forecast: Last price = 23.49, Forecast = 23.38, Daily return = -0.47%, Annual = -95.00%, MSE = 3.891894
ML forecast: Daily return = 0.29%, Annual = 72.96%, Avg CV MSE = 0.002237

Analyzing LINK-USD...
^[[C
Sentiment Analysis Results for LINK-USD:
YFinance Articles: 0
Google News Articles: 19
Search terms: LINK price, LINK crypto market, LINK cryptocurrency news, LINK blockchain
Combined sentiment: 0.051
Interpretation: Neutral
ARIMA forecast: Last price = 17.83, Forecast = 17.76, Daily return = -0.42%, Annual = -95.00%, MSE = 0.403379
ML forecast: Daily return = 0.25%, Annual = 64.24%, Avg CV MSE = 0.002068

Analyzing NEAR-USD...

Sentiment Analysis Results for NEAR-USD:
YFinance Articles: 0
Google News Articles: 19
Search terms: NEAR price, NEAR crypto market, NEAR cryptocurrency news, NEAR blockchain
Combined sentiment: 0.080
Interpretation: Neutral
ARIMA forecast: Last price = 3.13, Forecast = 3.14, Daily return = 0.24%, Annual = 61.35%, MSE = 0.154513
ML forecast: Daily return = 0.15%, Annual = 38.36%, Avg CV MSE = 0.002780

Analyzing DOGE-USD...

Sentiment Analysis Results for DOGE-USD:
YFinance Articles: 0
Google News Articles: 20
Search terms: DOGE price, DOGE crypto market, DOGE cryptocurrency news, DOGE blockchain
Combined sentiment: 0.130
Interpretation: Positive
ARIMA forecast: Last price = 0.25, Forecast = 0.25, Daily return = -0.58%, Annual = -95.00%, MSE = 0.000062
ML forecast: Daily return = -0.02%, Annual = -4.01%, Avg CV MSE = 0.002407

Analyzing TAO22974-USD...

Sentiment Analysis Results for TAO22974-USD:
YFinance Articles: 0
Google News Articles: 11
Search terms: TAO price, TAO crypto market, TAO cryptocurrency news, TAO blockchain
Combined sentiment: 0.060
Interpretation: Neutral
ARIMA forecast: Last price = 375.83, Forecast = 375.89, Daily return = 0.01%, Annual = 3.70%, MSE = 479.326288
ML forecast: Daily return = 0.21%, Annual = 53.96%, Avg CV MSE = 0.004100

Analyzing GRT6719-USD...
Warning: Insufficient news data for GRT6719-USD (YF: 0, GN: 2 articles)

Sentiment Analysis Results for GRT6719-USD:
YFinance Articles: 0
Google News Articles: 2
Search terms: GRT price, GRT crypto market, GRT cryptocurrency news, GRT blockchain
Combined sentiment: 0.000
Interpretation: Neutral
ARIMA forecast: Last price = 0.13, Forecast = 0.13, Daily return = 0.33%, Annual = 83.96%, MSE = 0.000168
ML forecast: Daily return = 0.56%, Annual = 140.27%, Avg CV MSE = 0.003213

Analyzing MKR-USD...

Sentiment Analysis Results for MKR-USD:
YFinance Articles: 0
Google News Articles: 20
Search terms: MKR price, MKR crypto market, MKR cryptocurrency news, MKR blockchain
Combined sentiment: 0.037
Interpretation: Neutral
ARIMA forecast: Last price = 1115.21, Forecast = 1108.22, Daily return = -0.63%, Annual = -95.00%, MSE = 5679.506893
ML forecast: Daily return = -0.09%, Annual = -23.50%, Avg CV MSE = 0.002047

Analyzing S32684-USD...

Sentiment Analysis Results for S32684-USD:
YFinance Articles: 0
Google News Articles: 6
Search terms: S price, S crypto market, S cryptocurrency news, S blockchain
Combined sentiment: -0.002
Interpretation: Neutral
ARIMA forecast: Last price = 0.60, Forecast = 0.60, Daily return = 0.10%, Annual = 25.94%, MSE = 0.002158
ML forecast: Daily return = -3.07%, Annual = -95.00%, Avg CV MSE = 0.025764

Analyzing JTO-USD...

Sentiment Analysis Results for JTO-USD:
YFinance Articles: 0
Google News Articles: 16
Search terms: JTO price, JTO crypto market, JTO cryptocurrency news, JTO blockchain
Combined sentiment: 0.031
Interpretation: Neutral
ARIMA forecast: Last price = 2.64, Forecast = 2.62, Daily return = -0.48%, Annual = -95.00%, MSE = 0.035039
ML forecast: Daily return = -1.04%, Annual = -95.00%, Avg CV MSE = 0.004842

Analyzing BERA-USD...

Sentiment Analysis Results for BERA-USD:
YFinance Articles: 0
Google News Articles: 20
Search terms: BERA price, BERA crypto market, BERA cryptocurrency news, BERA blockchain
Combined sentiment: 0.111
Interpretation: Positive
ARIMA forecast: Last price = 6.26, Forecast = 5.73, Daily return = -8.59%, Annual = -95.00%, MSE = 0.047920
Warning: Not enough data for cross-validation
Combined return for BTC-USD: 70.43% (Sentiment impact: 1.7%)
Combined return for ETH-USD: 51.95% (Sentiment impact: 0.3%)
Combined return for SOL-USD: -22.20% (Sentiment impact: 0.3%)
Combined return for HYPE32196-USD: 550.40% (Sentiment impact: 3.4%)
Combined return for ALGO-USD: -66.14% (Sentiment impact: 0.9%)
Combined return for FET-USD: 126.50% (Sentiment impact: 0.0%)
Combined return for AVAX-USD: 74.52% (Sentiment impact: 2.3%)
Combined return for LINK-USD: 64.39% (Sentiment impact: 1.5%)
Combined return for NEAR-USD: 39.70% (Sentiment impact: 2.4%)
Combined return for DOGE-USD: -96.34% (Sentiment impact: 3.9%)
Combined return for TAO22974-USD: 54.93% (Sentiment impact: 1.8%)
Combined return for GRT6719-USD: 86.76% (Sentiment impact: 0.0%)
Combined return for MKR-USD: -23.76% (Sentiment impact: 1.1%)
Combined return for S32684-USD: 16.58% (Sentiment impact: -0.1%)
Combined return for JTO-USD: -95.89% (Sentiment impact: 0.9%)
Volatility regime: normal, Adjustment: 1.00x

=== Analysis Results ===

Optimized Portfolio Weights:
  ALGO-USD: 1.00%
  AVAX-USD: 1.00%
  BTC-USD: 40.00%
  DOGE-USD: 1.00%
  ETH-USD: 1.00%
  FET-USD: 8.00%
  GRT6719-USD: 1.00%
  HYPE32196-USD: 40.00%
  JTO-USD: 1.00%
  LINK-USD: 1.00%
  MKR-USD: 1.00%
  NEAR-USD: 1.00%
  S32684-USD: 1.00%
  SOL-USD: 1.00%
  TAO22974-USD: 1.00%

Portfolio Metrics:
Expected Return: 259.30%
Annual Variance: 34.24%
Risk-Managed Allocation: 10.00% ($10,000.00)

Sentiment Analysis Results:
BTC-USD: 0.056 (Neutral)
ETH-USD: 0.010 (Neutral)
SOL-USD: 0.009 (Neutral)
HYPE32196-USD: 0.113 (Positive)
ALGO-USD: 0.029 (Neutral)
FET-USD: 0.000 (Neutral)
AVAX-USD: 0.075 (Neutral)
LINK-USD: 0.051 (Neutral)
NEAR-USD: 0.080 (Neutral)
DOGE-USD: 0.130 (Positive)
TAO22974-USD: 0.060 (Neutral)
GRT6719-USD: 0.000 (Neutral)
MKR-USD: 0.037 (Neutral)
S32684-USD: -0.002 (Neutral)
JTO-USD: 0.031 (Neutral)
BERA-USD: 0.111 (Positive)
```

## Analysis Results

Optimized Portfolio Weights:
- BTC-USD: 40.00%
- HYPE32196-USD: 40.00%
- FET-USD: 8.00%
- Other assets: 1.00% each

Portfolio Metrics:
- Expected Return: 259.30%
- Annual Variance: 34.24%
- Risk-Managed Allocation: 10.00% ($10,000.00)

Detailed Asset Analysis:

1. BTC-USD
   - Combined return: 70.43%
   - Sentiment impact: 1.7%
   - Sentiment score: 0.056 (Neutral)

2. ETH-USD
   - Combined return: 51.95%
   - Sentiment impact: 0.3%
   - Sentiment score: 0.010 (Neutral)

3. SOL-USD
   - Combined return: -22.20%
   - Sentiment impact: 0.3%
   - Sentiment score: 0.009 (Neutral)

4. HYPE32196-USD
   - Combined return: 550.40%
   - Sentiment impact: 3.4%
   - Sentiment score: 0.113 (Positive)

5. AVAX-USD
   - Combined return: 74.52%
   - Sentiment impact: 2.3%
   - Sentiment score: 0.075 (Neutral)

## Visualization

The system generates three key visualizations:

1. **Price Performance with Technical Indicators**
   - Normalized price evolution
   - 20 and 50-day EMAs
   - Bollinger Bands

2. **Trading Volume Analysis**
   - Volume bars for each asset
   - Volume trend analysis

3. **Risk Metrics**
   - Rolling volatility (21-day window)
   - Rolling Sharpe ratio (63-day window)

## Crypto-Specific Features

1. **Enhanced Weekend Handling**
   - Uses 252 trading days to avoid low weekend volatility
   - Implements special weekend data handling

2. **Crypto News Sentiment**
   - Higher weight (70%) to Google News for crypto assets
   - Crypto-specific search terms
   - Increased sentiment impact multiplier

3. **Volatility Regime Detection**
   - Low: Increases allocation by 20%
   - Normal: Standard allocation
   - High: Reduces allocation by 50%

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
