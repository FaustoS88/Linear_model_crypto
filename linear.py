import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import cvxpy as cp
import warnings
from textblob import TextBlob
import datetime
from pygooglenews import GoogleNews
from textblob import TextBlob
import re

# Suppress specific warnings while maintaining important ones
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Global parameters
TRADING_DAYS = 252
MAX_FORECAST = float('inf')  # Allow unlimited upside potential
MIN_FORECAST = -0.95  # Minimum allowed forecast return (-95%)

# -------------------------------
# Enhanced ARIMA Module
# -------------------------------
def arima_forecast_with_error(prices, order=(5, 1, 0)):
    """
    Enhanced ARIMA forecast with error estimation and validation.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if len(prices) < max(order) + 1:
        raise ValueError("Not enough data points for the specified ARIMA order")
    
    # Handle NaN values
    prices = prices.fillna(method='ffill').fillna(method='bfill')
    
    if prices.index.freq is None:
        inferred_freq = pd.infer_freq(prices.index)
        prices = prices.asfreq(inferred_freq) if inferred_freq else prices.asfreq("B")
        prices = prices.fillna(method='ffill').fillna(method='bfill')  # Fill any new NaNs from resampling
    
    try:
        model = ARIMA(prices, order=order,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        model_fit = model.fit()
        
        # Calculate in-sample error
        fitted_values = model_fit.fittedvalues
        mse = mean_squared_error(prices[model_fit.loglikelihood_burn:], 
                               fitted_values[-len(prices[model_fit.loglikelihood_burn:]):])
        
        forecast_price = model_fit.forecast(steps=1)[0]
        last_price = prices.iloc[-1]
        
        if not np.isfinite(forecast_price) or forecast_price <= 0:
            print("Warning: Invalid ARIMA forecast price. Using last price.")
            return 0.0, np.inf
        
        daily_return = (forecast_price / last_price) - 1
        annual_return = daily_return * TRADING_DAYS
        annual_return = np.clip(annual_return, MIN_FORECAST, MAX_FORECAST)
        
        print(f"ARIMA forecast: Last price = {last_price:.2f}, Forecast = {forecast_price:.2f}, "
              f"Daily return = {daily_return:.2%}, Annual = {annual_return:.2%}, MSE = {mse:.6f}")
        return annual_return, mse
    
    except Exception as e:
        print(f"Error in ARIMA forecast: {e}")
        return 0.0, np.inf

# -------------------------------
# Enhanced ML Forecast Module
# -------------------------------
def ml_forecast_with_cv(prices, lag=10, n_splits=5):
    """
    Enhanced ML forecast with cross-validation and error estimation.
    """
    try:
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        if len(prices) < lag + 2:
            print(f"Warning: Insufficient data for ML forecast (need {lag + 2}, got {len(prices)})")
            return np.nan, np.inf
        
        returns = prices.pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(returns.mean())
        
        X, y = [], []
        for i in range(lag, len(returns)):
            X.append(returns.iloc[i-lag:i].values)
            y.append(returns.iloc[i])
        X = np.array(X)
        y = np.array(y)
        
        # Determine number of folds based on data size
        n_splits = min(n_splits, len(X) - 1)  # Ensure we have at least 1 sample per fold
        if n_splits < 2:
            print("Warning: Not enough data for cross-validation")
            return np.nan, np.inf
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores.append(mean_squared_error(y_test, y_pred))
        
        # Final prediction
        final_model = LinearRegression()
        final_model.fit(X, y)
        X_pred = returns.iloc[-lag:].values.reshape(1, -1)
        daily_return_pred = final_model.predict(X_pred)[0]
        
        if not np.isfinite(daily_return_pred):
            print("Warning: Invalid ML prediction. Using zero forecast.")
            return np.nan, np.inf
        
        annual_return = daily_return_pred * TRADING_DAYS
        annual_return = np.clip(annual_return, MIN_FORECAST, MAX_FORECAST)
        avg_mse = np.mean(cv_scores)
        
        print(f"ML forecast: Daily return = {daily_return_pred:.2%}, "
              f"Annual = {annual_return:.2%}, Avg CV MSE = {avg_mse:.6f}")
        return annual_return, avg_mse
    
    except Exception as e:
        print(f"Error in ML forecast: {e}")
        return np.nan, np.inf

# -------------------------------
# Enhanced Visualization Module
# -------------------------------
def calculate_technical_indicators(data):
    """Calculate technical indicators for visualization."""
    result = data.copy()
    # Calculate moving averages
    result['EMA20'] = data.ewm(span=20, adjust=False).mean()
    result['EMA50'] = data.ewm(span=50, adjust=False).mean()
    
    # Calculate Bollinger Bands
    result['MA20'] = data.rolling(window=20).mean()
    result['20dSTD'] = data.rolling(window=20).std()
    result['Upper'] = result['MA20'] + (result['20dSTD'] * 2)
    result['Lower'] = result['MA20'] - (result['20dSTD'] * 2)
    
    # Calculate daily returns and rolling Sharpe ratio
    result['Returns'] = data.pct_change()
    result['Rolling_Vol'] = result['Returns'].rolling(window=21).std() * np.sqrt(252)
    risk_free_rate = 0.03  # Assuming 3% annual risk-free rate
    excess_returns = result['Returns'] - (risk_free_rate / 252)
    result['Rolling_Sharpe'] = (excess_returns.rolling(window=63).mean() * 252) / \
                            (result['Returns'].rolling(window=63).std() * np.sqrt(252))
    
    return result

def plot_enhanced_analysis(close_data, volume_data=None):
    """Create enhanced visualization with technical indicators."""
    # Set up the plot style with improved aesthetics
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.3
    
    # Define a color palette for better distinction
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create figure and subplots with adjusted size and spacing
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # Plot 1: Price action with technical indicators
    ax1 = fig.add_subplot(gs[0])
    
    # Normalize prices for better comparison
    normalized_data = close_data.div(close_data.iloc[0]) * 100
    
    # Calculate and plot technical indicators for each asset
    for idx, column in enumerate(normalized_data.columns):
        color = colors[idx % len(colors)]
        data = pd.DataFrame(normalized_data[column])
        data = calculate_technical_indicators(data)
        
        # Plot main price line with increased visibility
        ax1.plot(data.index, data[column], label=f'{column} Price', 
                 color=color, linewidth=2)
        
        # Plot EMAs with distinct styles
        ax1.plot(data.index, data['EMA20'], '--', 
                 label=f'{column} 20-day EMA', color=color, alpha=0.4, linewidth=1)
        ax1.plot(data.index, data['EMA50'], ':', 
                 label=f'{column} 50-day EMA', color=color, alpha=0.4, linewidth=1)
        
        # Plot Bollinger Bands with improved transparency
        ax1.fill_between(data.index, data['Upper'], data['Lower'], 
                        color=color, alpha=0.1)
    
    ax1.set_title('Normalized Price Performance with Technical Indicators',
                  fontsize=12, pad=20)
    ax1.set_xlabel('')
    ax1.set_ylabel('Normalized Price (Base=100)', fontsize=10)
    
    # Improve legend readability
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Plot 2: Volume with improved visibility
    if volume_data is not None:
        ax2 = fig.add_subplot(gs[1])
        for idx, column in enumerate(volume_data.columns):
            ax2.bar(volume_data.index, volume_data[column],
                    color=colors[idx % len(colors)], alpha=0.5,
                    label=f'{column} Volume')
        ax2.set_title('Trading Volume', fontsize=12, pad=20)
        ax2.set_xlabel('')
        ax2.set_ylabel('Volume', fontsize=10)
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                   fontsize=9, frameon=True)
        ax2.tick_params(axis='y', labelsize=8)
    
    # Plot 3: Risk Metrics with separate y-axes
    ax3 = fig.add_subplot(gs[2])
    ax3_twin = ax3.twinx()
    
    # Plot risk metrics for each asset
    for idx, column in enumerate(normalized_data.columns):
        color = colors[idx % len(colors)]
        data = pd.DataFrame(normalized_data[column])
        data = calculate_technical_indicators(data)
        
        # Plot volatility on left axis
        ax3.plot(data.index, data['Rolling_Vol'],
                 label=f'{column} Volatility', color=color,
                 alpha=0.7, linewidth=1.5)
        
        # Plot Sharpe ratio on right axis with different style
        ax3_twin.plot(data.index, data['Rolling_Sharpe'], '--',
                      label=f'{column} Sharpe', color=color,
                      alpha=0.5, linewidth=1)
    
    ax3.set_title('Risk Metrics: Rolling Volatility and Sharpe Ratio',
                  fontsize=12, pad=20)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Rolling Volatility', fontsize=10)
    ax3_twin.set_ylabel('Rolling Sharpe Ratio', fontsize=10)
    
    # Combine legends from both axes with improved formatting
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3_twin.legend(lines1 + lines2, labels1 + labels2,
                    loc='center left', bbox_to_anchor=(1.02, 0.5),
                    fontsize=9, frameon=True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

# -------------------------------
# Enhanced Sentiment Analysis Module
# -------------------------------
def get_enhanced_sentiment_score(ticker, search_term=None, days_lookback=7):
    """Get sentiment score using custom search terms from multiple news sources.
    Returns a score between -1 (very negative) and 1 (very positive)."""
    # Clean up ticker by removing numerical suffix and special characters
    clean_ticker = re.sub(r'[0-9-]+', '', ticker).replace('USD', '').strip()
    is_crypto = '-USD' in ticker
    
    # Generate default search terms based on asset type
    if not search_term or search_term.isspace():
        if is_crypto:
            search_term = f"{clean_ticker} price, {clean_ticker} crypto market, {clean_ticker} cryptocurrency news, {clean_ticker} blockchain"
        else:
            search_term = f"{clean_ticker} stock market, {clean_ticker} company news, {clean_ticker} analysis"
    
    try:
        # Initialize sentiment collectors with minimum thresholds
        yf_sentiments = []
        gn_sentiments = []
        min_articles = 3  # Minimum number of articles needed for reliable sentiment
        
        # 1. Get yfinance news sentiment with error handling
        try:
            stock = yf.Ticker(ticker)
            yf_news = stock.news
            
            if yf_news:
                for article in yf_news:
                    text = article.get('title', '')
                    if article.get('summary'):
                        text += ' ' + article['summary']
                    
                    if text:
                        blob = TextBlob(text)
                        sentiment = blob.sentiment.polarity
                        try:
                            pub_date = datetime.datetime.fromtimestamp(article.get('providerPublishTime', 0))
                            days_old = (datetime.datetime.now() - pub_date).days
                            if days_old <= days_lookback:
                                weight = 1.0 - (days_old / days_lookback)
                                yf_sentiments.append(sentiment * weight)
                        except (ValueError, TypeError):
                            print(f"Warning: Invalid date format in YF article for {ticker}")
        except Exception as yf_error:
            print(f"Warning: Error fetching YF news for {ticker}: {yf_error}")
        
        # 2. Get Google News sentiment with custom search term
        try:
            gn = GoogleNews(lang='en', country='US')
            
            # Split search terms and create multiple searches
            search_terms = [term.strip() for term in search_term.split(',')]
            for term in search_terms:
                # Add clean ticker to each search term for better relevance
                full_term = f"{clean_ticker} {term}"
                try:
                    search = gn.search(full_term)
                    
                    if search and 'entries' in search:
                        for item in search['entries'][:5]:  # Process top 5 news items per term
                            text = item.get('title', '')
                            if item.get('summary'):
                                text += ' ' + item['summary']
                            
                            if text:
                                blob = TextBlob(text)
                                sentiment = blob.sentiment.polarity
                                try:
                                    pub_date = datetime.datetime.strptime(item['published'], '%a, %d %b %Y %H:%M:%S %Z')
                                    days_old = (datetime.datetime.now() - pub_date).days
                                    if days_old <= days_lookback:
                                        weight = 1.0 - (days_old / days_lookback)
                                        gn_sentiments.append(sentiment * weight)
                                except (ValueError, KeyError):
                                    # If date parsing fails, include sentiment with reduced weight
                                    gn_sentiments.append(sentiment * 0.3)
                except Exception as search_error:
                    print(f"Warning: Error searching term '{full_term}': {search_error}")
        except Exception as gn_error:
            print(f"Warning: Error in Google News processing for {ticker}: {gn_error}")
        
        # Combine sentiments with dynamic source weighting and reliability check
        weighted_sentiment = 0.0
        if len(yf_sentiments) >= min_articles or len(gn_sentiments) >= min_articles:
            # Adjust weights based on number of articles and asset type
            if is_crypto:
                yf_weight = 0.3  # Less weight to yfinance for crypto
                gn_weight = 0.7  # More weight to Google News for crypto
            else:
                yf_weight = 0.6  # More weight to yfinance for stocks
                gn_weight = 0.4  # Less weight to Google News for stocks
            
            # Further adjust weights based on article count reliability
            if len(yf_sentiments) < min_articles:
                yf_weight *= 0.5
                gn_weight = 1.0 - yf_weight
            elif len(gn_sentiments) < min_articles:
                gn_weight *= 0.5
                yf_weight = 1.0 - gn_weight
            
            if yf_sentiments and gn_sentiments:
                weighted_sentiment = (yf_weight * np.mean(yf_sentiments) +
                                    gn_weight * np.mean(gn_sentiments))
            elif yf_sentiments:
                weighted_sentiment = np.mean(yf_sentiments)
            elif gn_sentiments:
                weighted_sentiment = np.mean(gn_sentiments)
                
            # Apply stronger sentiment impact for crypto assets
            if is_crypto:
                weighted_sentiment *= 1.2  # Increase sentiment impact for crypto
        else:
            print(f"Warning: Insufficient news data for {ticker} (YF: {len(yf_sentiments)}, GN: {len(gn_sentiments)} articles)")
        
        # Print detailed analysis results
        print(f"\nSentiment Analysis Results for {ticker}:")
        print(f"YFinance Articles: {len(yf_sentiments)}")
        print(f"Google News Articles: {len(gn_sentiments)}")
        print(f"Search terms: {', '.join(search_terms)}")
        print(f"Combined sentiment: {weighted_sentiment:.3f}")
        print(f"Interpretation: {'Positive' if weighted_sentiment > 0.1 else 'Negative' if weighted_sentiment < -0.1 else 'Neutral'}")
        
        return weighted_sentiment
    
    except Exception as e:
        print(f"Critical error in sentiment analysis for {ticker}: {e}")
        return 0.0

# -------------------------------
# Portfolio Optimization Module
# -------------------------------
def portfolio_optimization(mu, covariance, risk_aversion, max_weight=0.4):
    """Enhanced portfolio optimization with position limits and improved error handling."""
    # Input validation and preprocessing
    if not isinstance(mu, np.ndarray) or not isinstance(covariance, np.ndarray):
        raise TypeError("mu and covariance must be numpy arrays")
    if len(mu) != len(covariance):
        raise ValueError("Dimension mismatch between mu and covariance")
    
    # Handle NaN values
    if np.any(np.isnan(mu)) or np.any(np.isnan(covariance)):
        print("Warning: NaN values detected in inputs. Removing affected assets...")
        valid_idx = ~np.isnan(mu)
        mu = mu[valid_idx]
        covariance = covariance[valid_idx][:, valid_idx]
        if len(mu) == 0:
            print("Error: No valid assets remaining after NaN removal")
            return None
    
    n = len(mu)
    w = cp.Variable(n)
    
    # Scale inputs to improve numerical stability
    scale_factor = np.max(np.abs(mu))
    if scale_factor > 0:
        mu_scaled = mu / scale_factor
    else:
        mu_scaled = mu
    
    objective = cp.Maximize(mu_scaled.T @ w - (risk_aversion / 2) * cp.quad_form(w, covariance))
    
    # Enhanced constraints with minimum weight threshold
    min_weight = 0.01  # 1% minimum weight threshold
    constraints = [
        cp.sum(w) == 1,  # Full investment
        w >= min_weight,  # Minimum position size
        w <= max_weight  # Maximum position size
    ]
    
    try:
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.SCS, eps=1e-6, max_iters=10000)
        
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            print(f"Warning: Optimization status: {prob.status}; using modified equal weights")
            # Modified equal weights with constraints
            n_assets = len(mu)
            base_weight = min(max_weight, 1.0 / n_assets)
            weights = np.full(n_assets, base_weight)
            weights = weights / np.sum(weights)
            return weights
        
        if w.value is None:
            print("Warning: Optimization failed to find solution; using modified equal weights")
            n_assets = len(mu)
            base_weight = min(max_weight, 1.0 / n_assets)
            weights = np.full(n_assets, base_weight)
            weights = weights / np.sum(weights)
            return weights
        
        # Process optimization results
        weights = np.array(w.value).flatten()
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / np.sum(weights)
        
        # Verify results
        if not np.isclose(np.sum(weights), 1.0, rtol=1e-5):
            print("Warning: Weights don't sum to 1; normalizing...")
            weights = weights / np.sum(weights)
        
        return weights
    
    except Exception as e:
        print(f"Error in portfolio optimization: {e}")
        # Fallback to modified equal weights
        n_assets = len(mu)
        base_weight = min(max_weight, 1.0 / n_assets)
        weights = np.full(n_assets, base_weight)
        weights = weights / np.sum(weights)
        return weights

# -------------------------------
# Risk Management Module
# -------------------------------
def calculate_volatility_regime(returns, lookback=63):
    """Determine the current volatility regime using rolling volatility."""
    if len(returns) < lookback:
        return "normal"
    
    current_vol = returns.tail(lookback).std() * np.sqrt(TRADING_DAYS)
    historical_vol = returns.std() * np.sqrt(TRADING_DAYS)
    
    if current_vol > 1.5 * historical_vol:
        return "high"
    elif current_vol < 0.5 * historical_vol:
        return "low"
    return "normal"

def dynamic_risk_management(portfolio_return, risk_free_rate, portfolio_variance,
                          capital, returns, target_allocation=0.10):
    """Enhanced risk management with dynamic position sizing based on volatility regime."""
    try:
        # Handle NaN inputs
        if np.isnan(portfolio_return) or np.isnan(portfolio_variance):
            print("Warning: Invalid portfolio metrics detected")
            return 0.0, 0.0

        # Basic Kelly calculation with safeguards
        if portfolio_variance <= 0:
            print("Warning: Invalid portfolio variance")
            return 0.0, 0.0

        kelly_fraction = (portfolio_return - risk_free_rate) / portfolio_variance
        kelly_fraction = np.clip(kelly_fraction, 0.0, 1.0)
        
        # Adjust based on volatility regime
        vol_regime = calculate_volatility_regime(returns)
        regime_adjustments = {
            "low": 1.2,    # Increase allocation in low vol
            "normal": 1.0,  # Normal allocation
            "high": 0.5     # Reduce allocation in high vol
        }
        
        adjusted_kelly = kelly_fraction * regime_adjustments[vol_regime]
        allocation = min(adjusted_kelly, target_allocation)
        dollar_allocation = allocation * capital
        
        # Verify final values
        if not np.isfinite(allocation) or not np.isfinite(dollar_allocation):
            print("Warning: Invalid allocation calculated")
            return 0.0, 0.0

        print(f"Volatility regime: {vol_regime}, Adjustment: {regime_adjustments[vol_regime]:.2f}x")
        return allocation, dollar_allocation
    
    except Exception as e:
        print(f"Risk management error: {e}")
        return 0.0, 0.0

# -------------------------------
# Main Execution Block
# -------------------------------
# Main execution block update
if __name__ == "__main__":
    # Define tickers including high-growth tech stocks
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'HYPE32196-USD', 'ALGO-USD', 'FET-USD', 'AVAX-USD', 'LINK-USD', 'NEAR-USD', 'DOGE-USD', 'TAO22974-USD', 'GRT6719-USD', 'MKR-USD', 'S32684-USD', 'JTO-USD', 'BERA-USD']
    start_date = '2022-01-01'
    end_date = '2025-02-19'
    capital = 100000  # $100,000 capital
    risk_aversion = 0.5
    
    try:
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date)
        
        # Extract close and volume data
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data.xs('Close', axis=1, level='Price')
            volume_data = data.xs('Volume', axis=1, level='Price')
        else:
            close_data = data['Close']
            volume_data = data['Volume']
        
        # Initialize storage for results
        arima_returns = {}
        ml_returns = {}
        forecast_errors = {}
        sentiment_scores = {}
        
        print("\nAnalyzing assets...\n")
        
        # Analysis per ticker
        for ticker in tickers:
            print(f"\nAnalyzing {ticker}...")
            if ticker not in close_data.columns:
                print(f"Warning: {ticker} not found in data")
                continue
            
            prices = close_data[ticker].dropna()
            
            # Get sentiment score
            sentiment = get_enhanced_sentiment_score(ticker)
            sentiment_scores[ticker] = sentiment
            
            # Get forecasts with errors
            arima_ret, arima_error = arima_forecast_with_error(prices)
            ml_ret, ml_error = ml_forecast_with_cv(prices)
            
            if np.isfinite(arima_ret) and np.isfinite(ml_ret):
                arima_returns[ticker] = arima_ret
                ml_returns[ticker] = ml_ret
                forecast_errors[ticker] = {
                    'arima': arima_error,
                    'ml': ml_error
                }
        
        # Combine forecasts with sentiment adjustment
        combined_returns = {}
        for ticker in arima_returns:
            if ticker in ml_returns:
                total_error = forecast_errors[ticker]['arima'] + forecast_errors[ticker]['ml']
                if total_error > 0:
                    arima_weight = forecast_errors[ticker]['ml'] / total_error
                    ml_weight = forecast_errors[ticker]['arima'] / total_error
                else:
                    arima_weight = ml_weight = 0.5
                
                base_return = (arima_weight * arima_returns[ticker] + 
                              ml_weight * ml_returns[ticker])
                
                # Apply sentiment adjustment
                sentiment_adj = 1.0 + (0.3 * sentiment_scores[ticker])  # Adjust returns by up to ±30%
                combined_returns[ticker] = base_return * sentiment_adj
                print(f"Combined return for {ticker}: {combined_returns[ticker]:.2%} (Sentiment impact: {(sentiment_adj-1)*100:.1f}%)")
        
        # Portfolio optimization
        tickers_used = sorted(combined_returns.keys())
        mu = np.array([combined_returns[tkr] for tkr in tickers_used])
        returns_df = close_data[tickers_used].pct_change().dropna()
        covariance_daily = returns_df.cov().values
        covariance_annual = covariance_daily * TRADING_DAYS
        
        weights = portfolio_optimization(mu, covariance_annual, risk_aversion)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, mu)
        portfolio_variance = np.dot(weights, np.dot(covariance_annual, weights))
        
        # Risk management
        risk_free_rate = 0.03
        allocation, dollar_allocation = dynamic_risk_management(
            portfolio_return, risk_free_rate, portfolio_variance,
            capital, returns_df.mean(axis=1)
        )
        
        # Print results
        print("\n=== Analysis Results ===")
        print("\nOptimized Portfolio Weights:")
        for tkr, wt in zip(tickers_used, weights):
            print(f"  {tkr}: {wt:.2%}")
        
        print(f"\nPortfolio Metrics:")
        print(f"Expected Return: {portfolio_return:.2%}")
        print(f"Annual Variance: {portfolio_variance:.2%}")
        print(f"Risk-Managed Allocation: {allocation:.2%} (${dollar_allocation:,.2f})")
        
        print("\nSentiment Analysis Results:")
        for ticker in sentiment_scores:
            sentiment = sentiment_scores[ticker]
            print(f"{ticker}: {sentiment:.3f} ({('Positive' if sentiment > 0.1 else 'Negative' if sentiment < -0.1 else 'Neutral')})")
        
        # Create and display visualization
        fig = plot_enhanced_analysis(close_data, volume_data)
        plt.show()
        
    except Exception as e:
        print(f"Error in analysis: {e}")