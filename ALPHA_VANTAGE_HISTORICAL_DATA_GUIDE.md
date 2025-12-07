# Alpha Vantage Historical Market & Macroeconomic Data Guide

## Overview
Alpha Vantage provides extensive historical data across multiple asset classes and economic indicators. This guide focuses on **historical data** that can be integrated into your RL agent for enhanced market context.

---

## 1. **Economic Indicators (Macroeconomic Data)**

Alpha Vantage integrates with **FRED (Federal Reserve Economic Data)** to provide historical economic indicators:

### **Available Historical Economic Indicators:**

#### **GDP & Economic Growth:**
- `REAL_GDP` - Real GDP (billions of chained 2012 dollars), **historical data back to 1947**
- `REAL_GDP_PER_CAPITA` - Real GDP per capita, **historical data available**
- `GDP` - Nominal GDP, **historical data back to 1929**
- `GDPC1` - Real GDP (quarterly), **decades of history**

#### **Inflation Metrics:**
- `CPI` - Consumer Price Index for All Urban Consumers, **historical data back to 1913**
- `CPIAUCSL` - CPI All Urban Consumers (seasonally adjusted), **extensive history**
- `CPILFESL` - Core CPI (excluding food & energy), **long historical series**
- `INFLATION` - Inflation rate (YoY %), **historical data available**

#### **Employment & Labor:**
- `UNEMPLOYMENT` - Unemployment rate (%), **historical data back to 1948**
- `UNRATE` - Civilian unemployment rate, **extensive history**
- `PAYEMS` - Total nonfarm payrolls, **historical data back to 1939**
- `ICSA` - Initial jobless claims, **weekly historical data**

#### **Interest Rates:**
- `FEDERAL_FUNDS_RATE` - Federal funds effective rate, **historical data back to 1954**
- `FEDFUNDS` - Federal funds rate, **long historical series**
- `T10Y2Y` - 10-Year Treasury minus 2-Year Treasury (yield curve), **decades of data**
- `T10Y3M` - 10-Year Treasury minus 3-Month Treasury, **historical data**
- `DGS10` - 10-Year Treasury constant maturity rate, **extensive history**
- `DGS2` - 2-Year Treasury constant maturity rate, **historical data**

#### **Money Supply:**
- `M2` - Money supply (M2), **historical data back to 1959**
- `M2SL` - M2 money stock, **long historical series**
- `M1` - Money supply (M1), **historical data available**

#### **Consumer Metrics:**
- `CONSUMER_SENTIMENT` - Consumer sentiment index (University of Michigan), **historical data back to 1978**
- `UMCSENT` - Consumer sentiment index, **monthly historical data**
- `RETAIL_SALES` - Total retail sales, **historical data available**

#### **Producer Metrics:**
- `PPI` - Producer Price Index, **historical data back to 1913**
- `PPIFIS` - PPI for finished goods, **extensive history**

#### **Housing:**
- `HOUSING_STARTS` - New housing starts, **historical data back to 1959**
- `HOUST` - Housing starts, **monthly historical data**

#### **International Trade:**
- `TRADE_BALANCE` - U.S. trade balance, **historical data available**
- `EXPORTS` - Total exports, **historical data**
- `IMPORTS` - Total imports, **historical data**

### **Usage in RL Agent:**
These indicators can serve as **macroeconomic context features** for your RL agent:
- Help identify market regimes (expansion vs. recession)
- Provide leading indicators for market direction
- Add context for risk-on vs. risk-off periods

---

## 2. **Commodities (Historical Pricing)**

### **Available Commodities:**
- `WTI` - West Texas Intermediate (crude oil), **historical data back to 1983**
- `BRENT` - Brent crude oil, **extensive historical data**
- `NATURAL_GAS` - Natural gas futures, **historical data available**
- `COPPER` - Copper futures, **historical pricing data**
- `ALUMINUM` - Aluminum futures, **historical data**
- `WHEAT` - Wheat futures, **historical data**
- `CORN` - Corn futures, **historical data**
- `COTTON` - Cotton futures, **historical data**
- `SUGAR` - Sugar futures, **historical data**
- `COFFEE` - Coffee futures, **historical data**
- `GOLD` - Gold spot price, **historical data back to 1970s**
- `SILVER` - Silver spot price, **extensive historical data**
- `PLATINUM` - Platinum spot price, **historical data**
- `PALLADIUM` - Palladium spot price, **historical data**

### **Usage in RL Agent:**
- **Inflation hedging**: Gold/silver during inflationary periods
- **Energy correlation**: Oil prices correlate with energy sector stocks
- **Economic activity**: Copper ("Dr. Copper") as economic indicator
- **Sector rotation**: Commodity prices signal economic cycles

---

## 3. **Foreign Exchange (Forex) - Historical Rates**

### **Major Currency Pairs:**
- `EUR/USD` - Euro to U.S. Dollar, **historical data back to 1999**
- `GBP/USD` - British Pound to U.S. Dollar, **extensive history**
- `USD/JPY` - U.S. Dollar to Japanese Yen, **historical data**
- `USD/CHF` - U.S. Dollar to Swiss Franc, **historical data**
- `AUD/USD` - Australian Dollar to U.S. Dollar, **historical data**
- `USD/CAD` - U.S. Dollar to Canadian Dollar, **historical data**
- `NZD/USD` - New Zealand Dollar to U.S. Dollar, **historical data**
- Many other pairs available with **historical data**

### **Usage in RL Agent:**
- **Risk sentiment**: USD strength signals risk-off, weakness signals risk-on
- **Sector impacts**: Strong USD hurts multinationals, weak USD helps exports
- **Cross-asset correlation**: Currency movements affect international stocks

---

## 4. **Cryptocurrency (Historical Data)**

### **Available Cryptocurrencies:**
- `BTC` - Bitcoin, **historical data back to 2013**
- `ETH` - Ethereum, **historical data back to 2015**
- `BNB` - Binance Coin, **historical data**
- `XRP` - Ripple, **extensive historical data**
- Many other cryptocurrencies with **historical pricing**

### **Usage in RL Agent:**
- **Risk-on indicator**: Crypto rallies often signal risk appetite
- **Tech sector correlation**: Crypto and tech stocks often move together
- **Volatility proxy**: Crypto volatility can signal market stress

---

## 5. **Sector ETFs & Market Indices (Historical Data)**

Alpha Vantage provides data for major ETFs and indices:

### **Sector ETFs:**
- `XLK` - Technology Select Sector SPDR, **historical data**
- `XLF` - Financial Select Sector SPDR, **historical data**
- `XLE` - Energy Select Sector SPDR, **historical data**
- `XLV` - Health Care Select Sector SPDR, **historical data**
- `XLI` - Industrial Select Sector SPDR, **historical data**
- `XLP` - Consumer Staples Select Sector SPDR, **historical data**
- `XLY` - Consumer Discretionary Select Sector SPDR, **historical data**
- `XLB` - Materials Select Sector SPDR, **historical data**
- `XLU` - Utilities Select Sector SPDR, **historical data**
- `XLRE` - Real Estate Select Sector SPDR, **historical data**
- `XLC` - Communication Services Select Sector SPDR, **historical data**

### **Market Indices:**
- `SPY` - S&P 500 ETF, **extensive historical data**
- `QQQ` - Nasdaq-100 ETF, **historical data**
- `DIA` - Dow Jones Industrial Average ETF, **historical data**
- `IWM` - Russell 2000 (small caps), **historical data**
- `VIX` - CBOE Volatility Index (via ticker `^VIX`), **historical data**

### **Usage in RL Agent:**
- **Sector rotation**: Identify which sectors are outperforming
- **Market regime**: SPY/QQQ ratios signal growth vs. value
- **Volatility regime**: VIX signals market stress/calm
- **Correlation features**: Stock performance relative to sector/benchmark

---

## 6. **Options Data (Premium - Historical)**

Alpha Vantage Premium provides **historical options data**:
- **Options chains**: Historical option prices and volumes
- **Greeks**: Delta, gamma, theta, vega (if available historically)
- **Implied volatility**: Historical IV data
- **Put/Call ratios**: Historical sentiment indicators

### **Usage in RL Agent:**
- **Sentiment**: Put/call ratios signal market sentiment
- **Volatility expectations**: IV signals expected volatility
- **Hedging activity**: Options activity signals institutional positioning

---

## 7. **Technical Indicators (50+ Indicators - Historical)**

Alpha Vantage provides **50+ technical indicators** calculated from historical price data:

### **Trend Indicators:**
- `SMA` - Simple Moving Average (multiple periods), **historical**
- `EMA` - Exponential Moving Average (multiple periods), **historical**
- `WMA` - Weighted Moving Average, **historical**
- `DEMA` - Double Exponential Moving Average, **historical**
- `TEMA` - Triple Exponential Moving Average, **historical**
- `TRIMA` - Triangular Moving Average, **historical**
- `KAMA` - Kaufman Adaptive Moving Average, **historical**
- `MAMA` - MESA Adaptive Moving Average, **historical**
- `VWAP` - Volume Weighted Average Price, **historical**

### **Momentum Indicators:**
- `RSI` - Relative Strength Index, **historical**
- `STOCH` - Stochastic oscillator, **historical**
- `STOCHF` - Fast stochastic, **historical**
- `STOCHRSI` - Stochastic RSI, **historical**
- `MACD` - Moving Average Convergence Divergence, **historical**
- `MOM` - Momentum, **historical**
- `ROC` - Rate of Change, **historical**
- `CCI` - Commodity Channel Index, **historical**
- `CMO` - Chande Momentum Oscillator, **historical**
- `AROON` - Aroon oscillator, **historical**
- `AROONOSC` - Aroon oscillator, **historical**
- `PPO` - Percentage Price Oscillator, **historical**

### **Volatility Indicators:**
- `BBANDS` - Bollinger Bands, **historical**
- `ATR` - Average True Range, **historical**
- `NATR` - Normalized Average True Range, **historical**
- `ADX` - Average Directional Index, **historical**
- `ADXR` - Average Directional Movement Rating, **historical**
- `APO` - Absolute Price Oscillator, **historical**

### **Volume Indicators:**
- `OBV` - On-Balance Volume, **historical**
- `AD` - Chaikin A/D Line, **historical**
- `ADOSC` - Chaikin A/D Oscillator, **historical**

### **Price Indicators:**
- `TYPPRICE` - Typical Price, **historical**
- `WCLPRICE` - Weighted Close Price, **historical**
- `MIDPRICE` - Midpoint Price, **historical**
- `MIDPOINT` - Midpoint, **historical**
- `SAR` - Parabolic SAR, **historical**
- `TRANGE` - True Range, **historical**

### **Usage in RL Agent:**
- **Feature engineering**: Use as input features for RL agent
- **Signal generation**: Multiple indicators provide diverse signals
- **Confirmation**: Multiple indicators confirm trends

---

## 8. **Fundamental Data (Historical Financial Statements)**

### **Available Fundamental Data:**
- **Income Statement**: Quarterly and annual, **historical data back to company IPO/listing**
- **Balance Sheet**: Quarterly and annual, **historical data**
- **Cash Flow Statement**: Quarterly and annual, **historical data**
- **Company Overview**: Key metrics (P/E, P/B, EPS, etc.), **current snapshot**
- **Earnings Calendar**: Historical earnings dates and surprises, **years of history**

### **Usage in RL Agent:**
- **Fundamental features**: P/E, P/B, debt ratios as state features
- **Earnings momentum**: Track earnings surprises over time
- **Financial health**: Balance sheet metrics signal company strength

---

## 9. **News & Sentiment (Historical)**

### **Available News Data:**
- **News Articles**: Historical news articles, **limited history** (typically last 1000 articles per symbol)
- **Sentiment Scores**: Historical sentiment scores, **matched to news dates**
- **News Categories**: Categorized news (earnings, guidance, M&A, etc.)

### **Usage in RL Agent:**
- **Sentiment features**: Historical sentiment as state input
- **News momentum**: Track sentiment trends over time
- **Event correlation**: Correlate news events with price movements

---

## Implementation Recommendations

### **Priority 1: High-Impact Historical Data for RL Agent**

1. **Economic Indicators** (FRED integration):
   - `REAL_GDP` (quarterly, back to 1947)
   - `CPI` (monthly, back to 1913)
   - `UNEMPLOYMENT` (monthly, back to 1948)
   - `FEDERAL_FUNDS_RATE` (monthly, back to 1954)
   - `T10Y2Y` (yield curve - recession predictor)

2. **Commodities**:
   - `WTI` (crude oil - energy sector correlation)
   - `GOLD` (safe haven / inflation hedge)
   - `COPPER` (economic activity indicator)

3. **Market Indices**:
   - `SPY` (market benchmark)
   - `VIX` (volatility/risk indicator)
   - Sector ETFs (`XLK`, `XLF`, `XLE`, etc.)

4. **Technical Indicators**:
   - `MACD`, `RSI`, `BBANDS`, `ATR` (already calculated in your pipeline)

### **Priority 2: Medium-Impact Historical Data**

5. **Forex**:
   - `EUR/USD`, `USD/JPY` (major currency pairs)
   - `DXY` (U.S. Dollar Index - if available)

6. **Cryptocurrency**:
   - `BTC` (risk-on/risk-off indicator)

7. **Fundamental Data**:
   - Historical P/E ratios
   - Historical earnings surprises

### **How to Integrate:**

```python
# Example: Add economic indicators as features
from alpha_vantage.alphavantage import AlphaVantage

av = AlphaVantage(api_key=API_KEY)

# Get historical GDP data
gdp_data = av.get_economic_indicator('REAL_GDP', interval='quarterly')

# Get historical CPI
cpi_data = av.get_economic_indicator('CPI', interval='monthly')

# Get historical unemployment
unemployment_data = av.get_economic_indicator('UNEMPLOYMENT', interval='monthly')

# Get historical commodities
wti_data = av.get_commodity('WTI', interval='daily')
gold_data = av.get_commodity('GOLD', interval='daily')

# Align dates and merge with stock features
# Use as additional state features in RL agent
```

### **API Endpoints for Historical Data:**

```python
# Economic Indicators
GET https://www.alphavantage.co/query?function=REAL_GDP&interval=quarterly&apikey=YOUR_KEY
GET https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey=YOUR_KEY
GET https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey=YOUR_KEY

# Commodities
GET https://www.alphavantage.co/query?function=WTI&interval=monthly&apikey=YOUR_KEY
GET https://www.alphavantage.co/query?function=BRENT&interval=monthly&apikey=YOUR_KEY
GET https://www.alphavantage.co/query?function=NATURAL_GAS&interval=monthly&apikey=YOUR_KEY

# Forex
GET https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD&apikey=YOUR_KEY

# Cryptocurrency
GET https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey=YOUR_KEY
```

---

## Notes

1. **Rate Limits**: Premium subscription = 75 calls/minute (vs. 5 calls/minute free tier)
2. **Historical Coverage**: Most economic indicators have **decades of history** (1940s-present), commodities have **30-40 years**, forex has **20+ years**
3. **Data Frequency**: Economic indicators are typically **monthly or quarterly**, commodities/forex/crypto are **daily**
4. **Alignment**: You'll need to align different frequencies (monthly macro, daily prices) when merging with stock features
5. **Alpha Vantage Python SDK**: The `alpha_vantage` Python package supports most of these endpoints

---

## Next Steps

1. **Implement historical data collection** for top-priority indicators
2. **Feature engineering** to align frequencies and create meaningful features
3. **Integrate into RL state space** as additional context features
4. **Test impact** on model performance (A/B test with/without macro features)

