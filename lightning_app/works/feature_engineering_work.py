from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from . import LightningWork, HAS_LIGHTNING


class FeatureEngineeringWork(LightningWork):
    """
    Creates comprehensive features including:
    - Price & Returns (Head 1)
    - Technical Indicators (Head 6)
    - Merges with Macro/Market data (Heads 2-5, 7-8)
    
    Features are organized into 8 groups for multi-head attention.
    """

    def __init__(self, cache_dir: str) -> None:
        if HAS_LIGHTNING:
            super().__init__(parallel=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature groups for multi-head attention (8 heads)
        self.feature_groups = {
            "group_1_price_returns": [
                "return", "volatility", "sma_20", "sma_50", "rsi_14",
                "log_return", "price_change", "sma_5", "sma_10"
            ],
            "group_2_macro": [],  # Filled from macro data
            "group_3_commodities": [],  # Filled from commodities data
            "group_4_market_indices": [],  # Filled from market indices
            "group_5_forex": [],  # Filled from forex data
            "group_6_technical": [
                "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle", "bb_lower",
                "atr", "adx", "momentum", "roc", "williams_r"
            ],
            "group_7_sentiment": [
                "earnings_reported_eps", "earnings_estimated_eps", "earnings_surprise",
                "earnings_surprise_pct", "earnings_eps_growth", "days_since_earnings",
                "approaching_earnings", "sentiment_score", "sentiment_momentum",
                "earnings_call_score"  # OpenAI-generated score from earnings call transcript
            ],
            "group_8_crypto": [],  # Filled from crypto data
        }

    def run(
        self, 
        price_path: str, 
        ticker: str, 
        macro_market_data: Optional[Dict[str, str]] = None,
        enable_sentiment: bool = True,
        sentiment_payload: Optional[Dict] = None,
        fundamental_payload: Optional[Dict] = None
    ) -> str:
        """
        Create comprehensive features including stock, macro, and market data.
        
        Args:
            price_path: Path to price data parquet file
            ticker: Stock ticker symbol
            macro_market_data: Dictionary mapping group names to parquet file paths
        
        Returns:
            Path to saved feature parquet file
        """
        print(f"[1] Feature Engineering: Processing {ticker}")
        print(f"[2] Reading price data from: {price_path}")
        
        df = pd.read_parquet(price_path)
        print(f"[3] Loaded {len(df)} rows of price data")
        
        # Ensure date index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
        
        # Rename columns if needed
        if "5. adjusted close" in df.columns:
            df = df.rename(columns={"5. adjusted close": "close"})
        elif "adjusted close" in df.columns:
            df = df.rename(columns={"adjusted close": "close"})
        elif "Close" in df.columns:
            df = df.rename(columns={"Close": "close"})
        
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        
        # GROUP 1: Price & Returns Features
        print(f"[4] Calculating Group 1: Price & Returns features...")
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["price_change"] = df["close"].diff()
        df["volatility"] = df["return"].rolling(window=20, min_periods=5).std()
        df["sma_5"] = df["close"].rolling(window=5, min_periods=3).mean()
        df["sma_10"] = df["close"].rolling(window=10, min_periods=5).mean()
        df["sma_20"] = df["close"].rolling(window=20, min_periods=5).mean()
        df["sma_50"] = df["close"].rolling(window=50, min_periods=10).mean()
        df["rsi_14"] = self._relative_strength_index(df["close"], 14)
        
        # GROUP 6: Advanced Technical Indicators
        print(f"[5] Calculating Group 6: Technical indicators...")
        df = self._add_technical_indicators(df)
        
        # GROUP 7: Earnings Call Data (Fundamental Features)
        print(f"[6] Adding Group 7: Earnings features...")
        df = self._add_earnings_features(df, ticker)
        
        # GROUP 7: Sentiment Features (if enabled)
        if enable_sentiment and sentiment_payload:
            print(f"[7] Adding sentiment features to Group 7...")
            df = self._add_sentiment_features(df, sentiment_payload)
        else:
            print(f"[INFO] Sentiment features disabled or not available")
        
        # GROUP 7: Earnings Call Score (from OpenAI analysis of transcript)
        if fundamental_payload:
            print(f"[7b] Adding earnings call score to Group 7...")
            df = self._add_earnings_call_score(df, fundamental_payload)
        
        # Merge with macro/market data if provided
        if macro_market_data:
            print(f"[8] Merging macro/market data...")
            df = self._merge_macro_market_data(df, macro_market_data)
        else:
            print(f"[INFO] No macro/market data provided, using stock features only")
        
        # Fill missing values (forward fill for macro data, then backward fill)
        before_fill = df.isna().sum().sum()
        df = df.ffill().bfill()
        after_fill = df.isna().sum().sum()
        print(f"[9] Filled {before_fill - after_fill} missing values")
        
        # Drop rows where critical features are still NaN
        critical_features = ["close", "return"]
        before_dropna = len(df)
        df = df.dropna(subset=critical_features)
        after_dropna = len(df)
        
        print(f"[10] After dropna: {before_dropna} -> {after_dropna} rows")
        print(f"[11] Final feature count: {len(df.columns)} columns")
        
        if after_dropna == 0:
            raise ValueError(
                f"Feature engineering failed: All {before_dropna} rows were dropped. "
                "Need more price data or valid prices."
            )
        
        # Add feature group metadata (for reference, not used in training)
        df.attrs['feature_groups'] = self._get_feature_group_mapping(df.columns.tolist())
        
        feature_path = self.cache_dir / f"{ticker.lower()}_features.parquet"
        df.to_parquet(feature_path)
        
        print(f"[SUCCESS] Features saved to: {feature_path.absolute()}")
        print(f"[RESULT] Final dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"[INFO] Feature groups: {len(df.attrs.get('feature_groups', {}))} groups identified")
        
        return str(feature_path)

    @staticmethod
    def _relative_strength_index(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
        rs = gain / loss.replace(to_replace=0, value=pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators (GROUP 6)."""
        close = df["close"]
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        df["bb_middle"] = close.rolling(window=bb_window).mean()
        bb_std_dev = close.rolling(window=bb_window).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * bb_std_dev)
        df["bb_lower"] = df["bb_middle"] - (bb_std * bb_std_dev)
        
        # ATR (Average True Range)
        high = df.get("high", close)
        low = df.get("low", close)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()
        
        # ADX (Average Directional Index) - simplified
        plus_dm = np.where((high - high.shift()) > (low.shift() - low), 
                          np.maximum(high - high.shift(), 0), 0)
        minus_dm = np.where((low.shift() - low) > (high - high.shift()), 
                           np.maximum(low.shift() - low, 0), 0)
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / df["atr"]
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / df["atr"]
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df["adx"] = dx.rolling(14).mean()
        
        # Momentum
        df["momentum"] = close.pct_change(periods=10)
        
        # ROC (Rate of Change)
        df["roc"] = close.pct_change(periods=10) * 100
        
        # Williams %R
        highest_high = high.rolling(window=14).max()
        lowest_low = low.rolling(window=14).min()
        df["williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)
        
        return df
    
    def _add_earnings_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Add historical earnings call data as features (GROUP 7: Sentiment/Fundamental)."""
        from ..config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_URL
        import requests
        
        try:
            # Fetch historical earnings data from Alpha Vantage
            params = {
                "function": "EARNINGS",
                "symbol": ticker.upper(),
                "apikey": ALPHA_VANTAGE_API_KEY,
            }
            # Retry logic with exponential backoff for timeout issues
            import time
            max_retries = 3
            timeout_seconds = 60  # Increased timeout to 60 seconds
            
            response = None
            for attempt in range(max_retries):
                try:
                    response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=timeout_seconds)
                    response.raise_for_status()
                    break  # Success, exit retry loop
                except requests.exceptions.Timeout as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                        print(f"[WARNING] Alpha Vantage earnings timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] Alpha Vantage earnings timeout after {max_retries} attempts")
                        raise
                except requests.exceptions.RequestException as e:
                    print(f"[ERROR] Alpha Vantage earnings request error: {e}")
                    raise
            
            if response is None:
                raise requests.exceptions.RequestException("Failed to get response after retries")
            response.raise_for_status()
            payload = response.json()
            
            if "Error Message" in payload or "Note" in payload:
                print(f"[WARNING] Earnings data not available: {payload.get('Error Message', payload.get('Note', 'Unknown error'))}")
                return df
            
            if "quarterlyEarnings" not in payload:
                return df
            
            # Convert to DataFrame
            earnings_list = []
            for earning in payload["quarterlyEarnings"]:
                fiscal_date = earning.get("fiscalDateEnding", "")
                if not fiscal_date:
                    continue
                
                try:
                    earnings_list.append({
                        "date": pd.to_datetime(fiscal_date),
                        "reportedEPS": float(earning.get("reportedEPS", 0)) if earning.get("reportedEPS") and earning.get("reportedEPS") != "None" else None,
                        "estimatedEPS": float(earning.get("estimatedEPS", 0)) if earning.get("estimatedEPS") and earning.get("estimatedEPS") != "None" else None,
                        "surprise": float(earning.get("surprise", 0)) if earning.get("surprise") and earning.get("surprise") != "None" else None,
                        "surprisePercentage": float(earning.get("surprisePercentage", 0)) if earning.get("surprisePercentage") and earning.get("surprisePercentage") != "None" else None,
                    })
                except (ValueError, TypeError):
                    continue
            
            if not earnings_list:
                return df
            
            earnings_df = pd.DataFrame(earnings_list)
            earnings_df.set_index("date", inplace=True)
            earnings_df.sort_index(inplace=True)
            
            # Forward fill earnings data to daily frequency (earnings are quarterly)
            earnings_df = earnings_df.reindex(df.index)
            earnings_df = earnings_df.ffill()
            
            # Add earnings features
            df["earnings_reported_eps"] = earnings_df["reportedEPS"]
            df["earnings_estimated_eps"] = earnings_df["estimatedEPS"]
            df["earnings_surprise"] = earnings_df["surprise"]
            df["earnings_surprise_pct"] = earnings_df["surprisePercentage"]
            
            # Calculate earnings momentum features
            df["earnings_eps_growth"] = earnings_df["reportedEPS"].pct_change()
            df["earnings_surprise_momentum"] = earnings_df["surprisePercentage"].rolling(window=4, min_periods=1).mean()
            
            # Days since last earnings (useful for predicting next earnings)
            # Get original earnings dates before reindexing
            earnings_df_original = pd.DataFrame(earnings_list).set_index("date").sort_index()
            last_earnings_dates = earnings_df_original.index.unique()
            
            if len(last_earnings_dates) > 0:
                # For each date in df, calculate days since most recent earnings
                days_since_list = []
                for date in df.index:
                    # Find most recent earnings date before or on this date
                    recent_earnings = last_earnings_dates[last_earnings_dates <= date]
                    if len(recent_earnings) > 0:
                        days_since = (date - recent_earnings[-1]).days
                    else:
                        days_since = None
                    days_since_list.append(days_since)
                
                df["days_since_earnings"] = days_since_list
                
                # Feature: approaching earnings date (0-90 days window before next earnings)
                # Calculate days until next earnings
                days_until_list = []
                for date in df.index:
                    future_earnings = last_earnings_dates[last_earnings_dates > date]
                    if len(future_earnings) > 0:
                        days_until = (future_earnings[0] - date).days
                    else:
                        days_until = None
                    days_until_list.append(days_until)
                
                df["days_until_earnings"] = days_until_list
                df["approaching_earnings"] = (df["days_until_earnings"] <= 90).fillna(0).astype(int)
            
            print(f"[INFO] Added {len(earnings_df)} earnings dates as features")
            
        except Exception as e:
            print(f"[WARNING] Failed to add earnings features: {e}")
            # Add empty columns to maintain consistency
            df["earnings_reported_eps"] = None
            df["earnings_estimated_eps"] = None
            df["earnings_surprise"] = None
            df["earnings_surprise_pct"] = None
            df["earnings_eps_growth"] = None
            df["earnings_surprise_momentum"] = None
            df["days_since_earnings"] = None
            df["approaching_earnings"] = 0
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame, sentiment_payload: Dict) -> pd.DataFrame:
        """Add sentiment features to Group 7 (if sentiment is enabled)."""
        try:
            # Extract sentiment scores
            combined_score = sentiment_payload.get("combined_score", 0.0)
            alpha_vantage_score = sentiment_payload.get("alpha_vantage_score", 0.0)
            openai_score = sentiment_payload.get("openai_score", 0.0)
            
            # Add sentiment as constant features (can be forward-filled if we have historical sentiment)
            # For now, use the current sentiment score across all dates (can be enhanced later)
            df["sentiment_score"] = combined_score
            df["sentiment_alpha_vantage"] = alpha_vantage_score
            df["sentiment_openai"] = openai_score if openai_score is not None else 0.0
            
            # Calculate sentiment momentum (rolling average if we had historical data)
            # For now, use current score (can be enhanced with historical sentiment data)
            df["sentiment_momentum"] = combined_score
            
            print(f"[INFO] Added sentiment features: combined_score={combined_score:.3f}")
            
        except Exception as e:
            print(f"[WARNING] Failed to add sentiment features: {e}")
            df["sentiment_score"] = 0.0
            df["sentiment_momentum"] = 0.0
        
        return df
    
    def _add_earnings_call_score(self, df: pd.DataFrame, fundamental_payload: Dict) -> pd.DataFrame:
        """Add earnings call sentiment score from OpenAI transcript analysis to Group 7."""
        try:
            earnings_data = fundamental_payload.get("earnings_data")
            if earnings_data and "earnings_call_score" in earnings_data:
                earnings_call_score = earnings_data.get("earnings_call_score")
                
                if earnings_call_score is not None:
                    # Add as a constant feature (can be forward-filled if we have historical scores)
                    # For now, use the current score across all dates
                    # In future, can be enhanced with historical earnings call scores
                    df["earnings_call_score"] = float(earnings_call_score)
                    
                    # Calculate momentum based on earnings date proximity
                    # If we're close to earnings date, use the score; otherwise decay it
                    if "days_since_earnings" in df.columns:
                        # Decay the score over time (90-day half-life)
                        decay_factor = np.exp(-df["days_since_earnings"].fillna(90) / 90.0)
                        df["earnings_call_score_decayed"] = df["earnings_call_score"] * decay_factor
                    else:
                        df["earnings_call_score_decayed"] = df["earnings_call_score"]
                    
                    print(f"[INFO] Added earnings call score: {earnings_call_score:.3f}")
                else:
                    df["earnings_call_score"] = 0.0
                    df["earnings_call_score_decayed"] = 0.0
            else:
                df["earnings_call_score"] = 0.0
                df["earnings_call_score_decayed"] = 0.0
                
        except Exception as e:
            print(f"[WARNING] Failed to add earnings call score: {e}")
            df["earnings_call_score"] = 0.0
            df["earnings_call_score_decayed"] = 0.0
        
        return df
    
    def _merge_macro_market_data(
        self, 
        df: pd.DataFrame, 
        macro_market_data: Dict[str, str]
    ) -> pd.DataFrame:
        """Merge macro and market data with stock features."""
        result = df.copy()
        
        # Map macro/market data groups to feature groups
        group_mapping = {
            "macro": "group_2_macro",
            "commodities": "group_3_commodities",
            "market_indices": "group_4_market_indices",
            "forex": "group_5_forex",
            "crypto": "group_8_crypto",
        }
        
        for data_group, file_path in macro_market_data.items():
            if not file_path or file_path == "":
                continue
            
            try:
                macro_df = pd.read_parquet(file_path)
                if macro_df.empty:
                    continue
                
                # Ensure date index
                if not isinstance(macro_df.index, pd.DatetimeIndex):
                    if 'date' in macro_df.columns:
                        macro_df.set_index('date', inplace=True)
                    else:
                        macro_df.index = pd.to_datetime(macro_df.index)
                
                # Reindex to match stock data frequency (daily)
                # Forward fill for monthly/quarterly macro data
                macro_df = macro_df.reindex(result.index)
                macro_df = macro_df.ffill()
                
                # Merge with stock features
                result = result.join(macro_df, how="left")
                
                # Update feature group mapping
                feature_group = group_mapping.get(data_group)
                if feature_group:
                    self.feature_groups[feature_group].extend(macro_df.columns.tolist())
                
                print(f"[INFO] Merged {data_group}: {len(macro_df.columns)} features")
                
            except Exception as e:
                print(f"[WARNING] Failed to merge {data_group}: {e}")
                continue
        
        return result
    
    def _get_feature_group_mapping(self, columns: list) -> Dict[str, list]:
        """Create mapping of feature groups to column names."""
        mapping = {}
        
        for group_name, expected_features in self.feature_groups.items():
            group_cols = [col for col in columns if any(
                feat in col.lower() or col.lower() in feat.lower()
                for feat in expected_features
            )]
            
            # Also check for explicit group prefixes
            if group_name == "group_2_macro":
                group_cols.extend([col for col in columns if "macro_" in col.lower()])
            elif group_name == "group_3_commodities":
                group_cols.extend([col for col in columns if "commodities_" in col.lower()])
            elif group_name == "group_4_market_indices":
                group_cols.extend([col for col in columns if "market_indices_" in col.lower()])
            elif group_name == "group_5_forex":
                group_cols.extend([col for col in columns if "forex_" in col.lower()])
            elif group_name == "group_7_sentiment":
                group_cols.extend([col for col in columns if "earnings_" in col.lower() or "days_since_earnings" in col.lower() or "approaching_earnings" in col.lower() or "earnings_call_score" in col.lower() or "sentiment_" in col.lower()])
            elif group_name == "group_8_crypto":
                group_cols.extend([col for col in columns if "crypto_" in col.lower()])
            
            # Remove duplicates
            group_cols = list(set(group_cols))
            mapping[group_name] = group_cols
        
        return mapping
