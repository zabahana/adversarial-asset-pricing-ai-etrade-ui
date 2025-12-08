from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from . import LightningWork, HAS_LIGHTNING
from .model_training_work import MHADQN

MetricDict = Dict[str, float]
PlotList = Dict[str, str]
ResultPayload = Dict[str, Dict[str, Dict[str, float]]]


class ModelInferenceWork(LightningWork):
    """Runs portfolio backtests for DQN and MHA-DQN variants."""

    def __init__(self, model_dir: str, results_dir: str) -> None:
        if HAS_LIGHTNING:
            super().__init__(parallel=True)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self, ticker: str, feature_path: str, risk_level: str = "Medium", enable_sentiment: bool = True) -> str:
        print(f"[1] Running model evaluation for {ticker}...")
        print(f"[CONFIG] Risk Level: {risk_level}, Sentiment: {'ENABLED' if enable_sentiment else 'DISABLED'}")
        
        features = pd.read_parquet(feature_path)
        print(f"[2] Loaded {len(features)} rows of features for evaluation")

        # Get input dimension from features for model loading
        input_dim = features.select_dtypes(include=[float, int]).shape[1]
        sequence_length = 20
        print(f"[3] Input dimension: {input_dim}, Sequence length: {sequence_length}")

        # Try to load models, fall back to mock data if not available
        model_configs = [
            ("baseline_dqn", self.model_dir / "dqn" / "latest.ckpt"),
            ("mha_dqn_clean", self.model_dir / "mha_dqn" / "clean.ckpt"),
            ("mha_dqn_robust", self.model_dir / "mha_dqn" / "adversarial.ckpt"),
        ]

        results: ResultPayload = {}
        model_num = 4
        for name, model_path in model_configs:
            print(f"\n[{model_num}] Evaluating {name}...")
            model = self._load_model(model_path, input_dim=input_dim, sequence_length=sequence_length)
            if model is None:
                print(f"[{model_num+1}] ⚠️  Model not found for {name} at {model_path}")
                print(f"[{model_num+1}] ⚠️  Skipping evaluation - model checkpoint required for backtest")
                continue
            else:
                print(f"[{model_num+1}] ✅ Model loaded successfully, running ACTUAL backtest evaluation...")
                metrics, plots = self._evaluate_model(model, features, name, input_dim, sequence_length, risk_level=risk_level)
                print(f"[{model_num+2}] ✅ Evaluation complete for {name} - Results from ACTUAL model run")
                model_num += 3
                
                results[name] = {
                    "metrics": metrics,
                    "plots": plots,
                }

        # Generate forecasts for next-day prediction (only for robust model)
        print(f"\n[FORECAST] Generating next-day forecasts...")
        try:
            # Get current price from features
            current_price = float(features["close"].iloc[-1]) if "close" in features.columns else 0.0
            
            # Generate forecast for robust model
            forecast_result = self.forecast_all_models(
                ticker=ticker,
                feature_path=feature_path,
                current_price=current_price,
                risk_level=risk_level,
                enable_sentiment=enable_sentiment
            )
            
            # Add forecast data to results (ensure structure matches UI expectations)
            if forecast_result.get("success") and "mha_dqn_robust" in forecast_result.get("model_forecasts", {}):
                robust_forecast = forecast_result["model_forecasts"]["mha_dqn_robust"]
                
                # Ensure robust model entry exists in results
                if "mha_dqn_robust" not in results:
                    results["mha_dqn_robust"] = {"metrics": {}, "plots": {}}
                
                # Update with forecast data - ensure 'available' flag is ALWAYS set to True when we have forecast data
                # Check if we have actual forecast data
                has_forecast_fields = any([
                    robust_forecast.get("recommendation") is not None,
                    robust_forecast.get("price_change_pct") is not None,
                    robust_forecast.get("price_diff_pct") is not None,
                    robust_forecast.get("forecasted_price") is not None,
                    robust_forecast.get("confidence") is not None
                ])
                # If we have forecast fields, force available to True
                available_flag = True if has_forecast_fields else robust_forecast.get("available", False)
                
                results["mha_dqn_robust"].update({
                    "available": available_flag,  # Always True if forecast fields exist
                    "recommendation": robust_forecast.get("recommendation", "HOLD"),
                    "price_change_pct": robust_forecast.get("price_change_pct", 0.0),
                    "price_diff_pct": robust_forecast.get("price_diff_pct", 0.0),
                    "confidence": robust_forecast.get("confidence", 0.5),
                    "forecasted_price": robust_forecast.get("forecasted_price", current_price),
                    "last_actual_price": robust_forecast.get("last_actual_price", current_price),
                    "forecast_date": robust_forecast.get("forecast_date", forecast_result.get("forecast_date", "")),
                    "q_values": robust_forecast.get("q_values", []),
                    "explainability": robust_forecast.get("explainability", {}),
                })
                
                # Add top-level forecast metadata (UI reads these directly)
                results["last_data_date"] = forecast_result.get("last_data_date", "")
                results["forecast_date"] = forecast_result.get("forecast_date", "")
                results["current_price"] = forecast_result.get("last_actual_price", current_price)
                
                print(f"[FORECAST] ✅ Forecast generated: {robust_forecast.get('recommendation', 'HOLD')} with {robust_forecast.get('price_change_pct', 0):+.2f}% change")
                print(f"[FORECAST] Available flag set to: {results['mha_dqn_robust'].get('available', False)}")
                print(f"[FORECAST] Forecast keys in results: {list(results['mha_dqn_robust'].keys())}")
            else:
                error_msg = forecast_result.get("error", "Unknown error")
                print(f"[WARNING] Forecast generation failed: {error_msg}")
                print(f"[WARNING] Forecast result keys: {list(forecast_result.keys())}")
                if "model_forecasts" in forecast_result:
                    print(f"[WARNING] Available models in forecast: {list(forecast_result['model_forecasts'].keys())}")
                # Ensure available flag is False if forecast failed
                if "mha_dqn_robust" in results:
                    results["mha_dqn_robust"]["available"] = False
                    print(f"[WARNING] Set available flag to False for mha_dqn_robust")
        except Exception as e:
            print(f"[WARNING] Error generating forecast: {e}")
            import traceback
            traceback.print_exc()

        # Evaluate adversarial attacks on robust model
        print(f"\n[ADVERSARIAL] Starting adversarial attack evaluation...")
        try:
            robust_model_path = self.model_dir / "mha_dqn" / "adversarial.ckpt"
            if robust_model_path.exists():
                robust_model = self._load_model(robust_model_path, input_dim=input_dim, sequence_length=sequence_length)
                if robust_model is not None:
                    attack_results = self._evaluate_adversarial_attacks(
                        robust_model, features, input_dim, sequence_length, ticker
                    )
                    if attack_results:
                        results["adversarial_attack_results"] = attack_results
                        print(f"[ADVERSARIAL] ✅ Attack evaluation complete with {len(attack_results)} attack types")
                    else:
                        print(f"[ADVERSARIAL] ⚠️ Attack evaluation returned no results")
                else:
                    print(f"[ADVERSARIAL] ⚠️ Robust model could not be loaded for attack evaluation")
            else:
                print(f"[ADVERSARIAL] ⚠️ Robust model not found at {robust_model_path}, skipping attack evaluation")
        except Exception as e:
            print(f"[ADVERSARIAL] ⚠️ Error during adversarial attack evaluation: {e}")
            import traceback
            traceback.print_exc()

        output_path = self.results_dir / f"{ticker.lower()}_model_results.json"
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)

        print(f"\n[COMPLETE] Model evaluation complete. Results saved to: {output_path}")
        return str(output_path)

    def _load_model(self, model_path: Path, input_dim: int = None, sequence_length: int = 20):
        """Load model checkpoint, return None if not found."""
        if not model_path.exists():
            print(f"[ERROR] Model file does not exist: {model_path}")
            return None
        try:
            print(f"[INFO] Loading model checkpoint from {model_path}...")
            
            # Try to load as state_dict (new format from training)
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
                
                # Check if it's a state_dict or a full model
                if isinstance(state_dict, dict) and any(k.startswith('input_projection') or k.startswith('attention_layers') for k in state_dict.keys()):
                    # It's a state_dict, need to instantiate model first
                    if input_dim is None:
                        # Try to infer from state_dict
                        # Look for input_projection weight shape
                        if 'input_projection.weight' in state_dict:
                            input_dim = state_dict['input_projection.weight'].shape[1]
                        else:
                            print(f"[WARNING] Cannot infer input_dim from state_dict, using default 13")
                            input_dim = 13
                    
                    print(f"[INFO] Instantiating MHA-DQN model with input_dim={input_dim}, sequence_length={sequence_length}")
                    model = MHADQN(
                        input_dim=input_dim,
                        sequence_length=sequence_length,
                        d_model=128,
                        num_heads=8,
                        num_layers=3,
                        hidden_sizes=[256, 128],
                        dropout_rate=0.1,
                        output_size=3,
                    )
                    model.load_state_dict(state_dict)
                    model.eval()
                    print(f"[SUCCESS] Model loaded from state_dict and set to eval mode")
                    return model
                else:
                    # It's a full model object (old format)
                    model = state_dict
                    if hasattr(model, "eval"):
                        model.eval()
                        print(f"[SUCCESS] Model loaded (full object) and set to eval mode")
                    else:
                        print(f"[SUCCESS] Model loaded (full object, no eval method)")
                    return model
            except TypeError as e:
                # Fallback for older PyTorch versions
                try:
                    state_dict = torch.load(model_path, map_location="cpu")
                    if isinstance(state_dict, dict) and any(k.startswith('input_projection') for k in state_dict.keys()):
                        if input_dim is None:
                            input_dim = state_dict.get('input_projection.weight', torch.zeros(128, 13)).shape[1] if 'input_projection.weight' in state_dict else 13
                        model = MHADQN(input_dim=input_dim, sequence_length=sequence_length)
                        model.load_state_dict(state_dict)
                        model.eval()
                        return model
                    return state_dict if hasattr(state_dict, "eval") else None
                except Exception:
                    return None
            
        except Exception as e:
            print(f"[ERROR] Failed to load model from {model_path}: {e}")
            print(f"[INFO] Model checkpoint may be corrupted or incompatible with current PyTorch version")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _validate_forecasted_price(forecasted_price: float, last_actual_price: float, max_change_pct: float = 50.0) -> float:
        """
        Validate and adjust forecasted price to prevent anomalies.
        
        Args:
            forecasted_price: The predicted price
            last_actual_price: The last known actual price
            max_change_pct: Maximum allowed percentage change (default 50%)
        
        Returns:
            Validated/adjusted forecasted price
        """
        import numpy as np
        
        # If forecasted price is negative, set to last actual price
        if forecasted_price <= 0:
            print(f"[WARNING] Forecasted price is negative ({forecasted_price:.2f}), setting to last actual price ({last_actual_price:.2f})")
            return last_actual_price
        
        # Calculate percentage change
        price_change_pct = ((forecasted_price - last_actual_price) / last_actual_price) * 100
        
        # Check if change is anomalous (>100% or more than max_change_pct)
        if abs(price_change_pct) > 100.0:
            print(f"[WARNING] Forecasted price change is anomalous ({price_change_pct:.2f}%), capping to ±{max_change_pct}%")
            # Cap to max_change_pct
            if price_change_pct > 0:
                adjusted_price = last_actual_price * (1 + max_change_pct / 100)
            else:
                adjusted_price = last_actual_price * (1 - max_change_pct / 100)
            return adjusted_price
        
        # If change is between 50% and 100%, apply conservative adjustment
        if abs(price_change_pct) > max_change_pct:
            print(f"[WARNING] Forecasted price change is high ({price_change_pct:.2f}%), applying conservative adjustment")
            # Reduce the change by 50% (so a 80% change becomes 40%)
            adjustment_factor = max_change_pct / abs(price_change_pct)
            adjusted_change_pct = price_change_pct * adjustment_factor
            adjusted_price = last_actual_price * (1 + adjusted_change_pct / 100)
            return adjusted_price
        
        return forecasted_price
    

    @staticmethod
    def _evaluate_model(model, features: pd.DataFrame, model_name: str, input_dim: int, sequence_length: int, risk_level: str = "Medium") -> Tuple[MetricDict, PlotList]:
        """Run backtest evaluation on model using features data."""
        import numpy as np
        from collections import deque
        
        print(f"[INFO] Starting backtest simulation...")
        
        # Prepare features for model input
        numeric_features = features.select_dtypes(include=[float, int]).values
        
        # Normalize features
        feature_mean = np.nanmean(numeric_features, axis=0, keepdims=True)
        feature_std = np.nanstd(numeric_features, axis=0, keepdims=True) + 1e-8
        normalized_features = (numeric_features - feature_mean) / feature_std
        normalized_features = np.nan_to_num(normalized_features, nan=0.0)
        
        # Get price data (for portfolio value calculation)
        if "return" in features.columns:
            returns = features["return"].fillna(0).values
        elif "close" in features.columns:
            prices = features["close"].fillna(method='ffill').fillna(method='bfill')
            returns = prices.pct_change().fillna(0).values
        else:
            # Use first numeric column as proxy
            prices = numeric_features[:, 0]
            returns = np.diff(prices, prepend=prices[0]) / (prices + 1e-8)
        
        # Initialize portfolio
        initial_capital = 10000.0
        portfolio_value = initial_capital
        cash = initial_capital
        shares = 0.0
        positions = deque(maxlen=1000)  # Track positions
        
        # Portfolio value history
        portfolio_values = [initial_capital]
        actions_taken = []
        actual_prices = []     # Store actual prices that occurred next day (for comparison plots only)
        attention_weights_snapshot = None  # Store attention weights for explainability
        
        # Verify we have required price data before proceeding
        if "close" not in features.columns:
            print(f"[ERROR] Close price column required for backtesting with actual data. No approximations allowed.")
            raise ValueError("Close price data required for backtesting. Cannot use approximations.")
        
        model.eval()
        with torch.no_grad():
            # Run backtest: process sequences through model
            for i in range(sequence_length, len(normalized_features) - 1):  # -1 to ensure we can get next day's price
                # Get sequence
                sequence = normalized_features[i - sequence_length:i]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # (1, seq_len, features)
                
                # Get model prediction with attention weights (only for last iteration to save memory)
                try:
                    capture_attention = (i == len(normalized_features) - 2)  # Only capture for last prediction
                    if capture_attention:
                        # Check if model supports return_attention parameter
                        try:
                            q_values, attention_weights = model(sequence_tensor, return_attention=True)
                            # Convert attention weights to serializable format
                            attention_weights_snapshot = {}
                            for layer_name, attn in attention_weights.items():
                                # attn shape: (batch_size, num_heads, seq_len, seq_len)
                                # Average across batch and heads for visualization
                                attn_avg = np.mean(attn, axis=(0, 1))  # (seq_len, seq_len)
                                attention_weights_snapshot[layer_name] = attn_avg.tolist()
                        except (TypeError, AttributeError):
                            # Model doesn't support return_attention, use regular forward
                            q_values = model(sequence_tensor)
                    else:
                        q_values = model(sequence_tensor)
                    q_values_flat = q_values.squeeze(0)  # (3,)
                    
                    # Select action (0=SELL, 1=HOLD, 2=BUY)
                    action_idx = q_values_flat.argmax().item()
                    action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
                    action = action_map.get(action_idx, "HOLD")
                    actions_taken.append(action)
                    
                    # Get current price from features - MUST use actual close price only (no approximations)
                    if "close" not in features.columns:
                        print(f"[ERROR] Close price column not found in features. Cannot compute backtest with actual data.")
                        raise ValueError("Close price data required for backtesting. No approximations allowed.")
                    
                    current_price = float(features.iloc[i]["close"])
                    if pd.isna(current_price) or current_price <= 0:
                        print(f"[ERROR] Invalid close price at index {i}: {current_price}. Skipping this step.")
                        continue
                    
                    # Do NOT generate synthetic predicted prices - model outputs actions, not price predictions
                    # Predicted prices would be synthetic/approximate, so we skip them
                    # Portfolio performance metrics use actual prices and actual actions only
                    
                    # Store actual price that occurred next day - MUST use actual close price only (no approximations)
                    if i + 1 < len(features) and "close" in features.columns:
                        actual_next_price = float(features.iloc[i + 1]["close"])
                        if not pd.isna(actual_next_price) and actual_next_price > 0:
                            actual_prices.append(actual_next_price)
                        else:
                            actual_prices.append(None)  # Invalid price, skip comparison
                    else:
                        actual_prices.append(None)  # No next day data available
                    
                    # Execute action with realistic trading
                    transaction_cost = 0.001  # 0.1% transaction cost
                    
                    # Position sizing based on risk level
                    risk_sizing = {
                        "Low": 0.10,      # 10% - conservative
                        "Medium": 0.30,   # 30% - balanced (default)
                        "High": 0.50,     # 50% - aggressive
                    }
                    position_size_pct = risk_sizing.get(risk_level, 0.30)
                    
                    if action == "BUY" and cash > 100:  # Need at least $100 to buy
                        # Buy with risk-adjusted % of available cash
                        buy_amount = min(cash * position_size_pct, cash - 100)  # Keep $100 buffer
                        if buy_amount > 100:
                            shares_to_buy = buy_amount / (current_price + 1e-8)
                            cost = buy_amount * (1 + transaction_cost)
                            
                            if cost <= cash:
                                shares += shares_to_buy
                                cash -= cost
                    
                    elif action == "SELL" and shares > 0:
                        # Sell risk-adjusted % of shares
                        shares_to_sell = shares * position_size_pct
                        proceeds = shares_to_sell * current_price * (1 - transaction_cost)
                        if proceeds > 0:
                            shares -= shares_to_sell
                            cash += proceeds
                    
                    # Update portfolio value (cash + shares * current price)
                    portfolio_value = cash + shares * current_price
                    portfolio_values.append(portfolio_value)
                    
                except Exception as e:
                    print(f"[WARNING] Error in backtest step {i}: {e}")
                    # Continue with HOLD action
                    portfolio_value = portfolio_values[-1] if len(portfolio_values) > 0 else initial_capital
                    portfolio_values.append(portfolio_value)
                    actions_taken.append("HOLD")
        
        # Calculate metrics from backtest results - ONLY use actual data, NO mock fallbacks
        if len(portfolio_values) < 2:
            print(f"[ERROR] Insufficient backtest data ({len(portfolio_values)} values). Cannot generate metrics from actual data.")
            print(f"[ERROR] Returning empty metrics - mock data is NOT allowed for performance metrics.")
            # Return empty metrics with flag indicating insufficient data
            return {
                "from_actual_backtest": False,
                "is_mock_data": False,
                "insufficient_data": True,
                "error": "Insufficient backtest data - need at least 2 portfolio values"
            }, {}
        
        portfolio_values_array = np.array(portfolio_values)
        
        # Calculate returns
        portfolio_returns = np.diff(portfolio_values_array) / (portfolio_values_array[:-1] + 1e-8)
        portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]
        
        if len(portfolio_returns) == 0:
            print(f"[ERROR] No valid returns calculated. Cannot generate metrics from actual data.")
            print(f"[ERROR] Returning empty metrics - mock data is NOT allowed for performance metrics.")
            # Return empty metrics with flag indicating insufficient data
            return {
                "from_actual_backtest": False,
                "is_mock_data": False,
                "insufficient_data": True,
                "error": "No valid returns calculated"
            }, {}
        
        # Annualized metrics
        total_return = (portfolio_values[-1] / initial_capital) - 1
        num_years = len(portfolio_values) / 252  # Approximate trading days per year
        cagr = ((1 + total_return) ** (1 / num_years) - 1) if num_years > 0 else 0.0
        
        # Sharpe ratio
        annual_return = np.mean(portfolio_returns) * 252 if len(portfolio_returns) > 0 else 0.0
        annual_vol = np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 1 else 0.0
        sharpe = (annual_return / annual_vol) if annual_vol > 0 else 0.0
        
        # Max drawdown
        running_max = np.maximum.accumulate(portfolio_values_array)
        drawdowns = (portfolio_values_array - running_max) / (running_max + 1e-8)
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Robustness score (based on consistency and Sharpe ratio)
        win_rate = sum(1 for r in portfolio_returns if r > 0) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0.5
        robustness_score = (win_rate * 0.4) + (min(max(sharpe / 2.0, 0), 1) * 0.6)  # Combine win rate and Sharpe
        
        # NO ARTIFICIAL ADJUSTMENTS - Use actual computed metrics only
        # Metrics are computed directly from backtest results without modification
        
        metrics: MetricDict = {
            "sharpe": round(float(sharpe), 2),
            "cagr": round(float(cagr), 2),
            "max_drawdown": round(float(max_drawdown), 2),
            "robustness_score": round(float(robustness_score), 2),
            "total_return": round(float(total_return), 2),
            "final_portfolio_value": round(float(portfolio_values[-1]), 2),
            "win_rate": round(float(win_rate), 2),
            "num_trades": len([a for a in actions_taken if a != "HOLD"]),
            # Flag to indicate these are actual backtest results (not mock)
            "from_actual_backtest": True,
            "is_mock_data": False,
            "backtest_period_days": len(portfolio_values),
            # Store portfolio values and dates for plotting
            "portfolio_values": [round(float(v), 2) for v in portfolio_values],
            "portfolio_returns": [round(float(r), 4) for r in portfolio_returns] if len(portfolio_returns) > 0 else [],
            "drawdowns": [round(float(d), 4) for d in drawdowns.tolist()] if len(drawdowns) > 0 else [],
            # Do NOT include synthetic predicted prices - they are approximations, not actual model predictions
            # Portfolio performance metrics are calculated from actual prices and actual actions only
            "predicted_prices": [],  # Removed synthetic predictions - model outputs actions, not price predictions
            "actual_prices": [round(float(a), 2) if a is not None else None for a in actual_prices] if len(actual_prices) > 0 else [],
            "attention_weights": attention_weights_snapshot if attention_weights_snapshot else None,  # For explainability
        }
        
        # Get dates if available - for portfolio values timeline (actual dates from historical data)
        if isinstance(features.index, pd.DatetimeIndex):
            # Portfolio values start from initial capital (before first trade), so we need sequence_length-1 to sequence_length+len(portfolio_values)-1
            start_idx = max(0, sequence_length - 1)
            end_idx = min(len(features.index), start_idx + len(portfolio_values))
            dates = features.index[start_idx:end_idx].strftime("%Y-%m-%d").tolist()
            # Pad or trim to match portfolio_values length
            if len(dates) < len(portfolio_values):
                dates = [features.index[0].strftime("%Y-%m-%d")] + dates + [features.index[-1].strftime("%Y-%m-%d")] * (len(portfolio_values) - len(dates) - 1)
            metrics["dates"] = dates[:len(portfolio_values)]
        else:
            metrics["dates"] = [f"Day {i}" for i in range(len(portfolio_values))]
        
        plots: PlotList = {}
        
        print(f"[RESULT] ✅ ACTUAL BACKTEST results from model run:")
        print(f"        Sharpe={sharpe:.2f}, CAGR={cagr:.2%}, MaxDD={max_drawdown:.2%}")
        print(f"        Portfolio values: {len(portfolio_values)} days, {len(actions_taken)} actions")
        print(f"        Metrics computed from actual model predictions and portfolio simulation")
        
        return metrics, plots
    
    def _evaluate_adversarial_attacks(
        self, model, features: pd.DataFrame, input_dim: int, sequence_length: int, ticker: str
    ) -> Dict:
        """Evaluate model robustness against adversarial attacks (FGSM, PGD, C&W, BIM, DeepFool)."""
        import numpy as np
        import torch.nn.functional as F
        from sklearn.metrics import mean_squared_error
        
        print(f"[ADVERSARIAL] Preparing test data for attack evaluation...")
        
        # Create sequences from features
        numeric_cols = features.select_dtypes(include=[float, int]).columns.tolist()
        if "return" not in numeric_cols and "close" in features.columns:
            features["return"] = features["close"].pct_change()
            numeric_cols.append("return")
        
        feature_data = features[numeric_cols].values
        feature_mean = np.nanmean(feature_data, axis=0, keepdims=True)
        feature_std = np.nanstd(feature_data, axis=0, keepdims=True) + 1e-8
        feature_data = (feature_data - feature_mean) / feature_std
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create sequences
        sequences = []
        targets = []
        for i in range(sequence_length, len(feature_data)):
            seq = feature_data[i-sequence_length:i]
            # Use next return as target
            if i < len(feature_data) - 1:
                target = feature_data[i+1, numeric_cols.index("return")] if "return" in numeric_cols else 0.0
            else:
                target = feature_data[i, numeric_cols.index("return")] if "return" in numeric_cols else 0.0
            sequences.append(seq)
            targets.append(target)
        
        if len(sequences) < 10:
            print(f"[ADVERSARIAL] ⚠️ Insufficient sequences for attack evaluation ({len(sequences)}), skipping")
            return None
        
        # Use subset for faster evaluation (last 100 sequences)
        test_size = min(100, len(sequences))
        X_test = np.array(sequences[-test_size:])
        y_test = np.array(targets[-test_size:])
        
        print(f"[ADVERSARIAL] Testing with {test_size} sequences...")
        
        model.eval()
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        
        # Get clean predictions
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            clean_preds = model(X_tensor).squeeze().cpu().numpy()
            # Convert Q-values to price predictions (simplified)
            if len(clean_preds.shape) > 1 and clean_preds.shape[-1] == 3:
                # Q-values for BUY/HOLD/SELL - use weighted average
                clean_preds = (clean_preds[:, 2] - clean_preds[:, 0]) * 0.01  # Simplified conversion
            clean_mse = mean_squared_error(y_test, clean_preds[:len(y_test)])
        
        attack_results = {}
        
        # FGSM Attack
        try:
            print(f"[ADVERSARIAL] Testing FGSM attack...")
            epsilon = 0.01
            X_tensor.requires_grad_(True)
            pred = model(X_tensor).squeeze()
            if len(pred.shape) > 1 and pred.shape[-1] == 3:
                loss = F.mse_loss(pred.mean(dim=-1), torch.FloatTensor(y_test).to(device))
            else:
                loss = F.mse_loss(pred, torch.FloatTensor(y_test).to(device))
            model.zero_grad()
            loss.backward()
            X_adv = X_tensor + epsilon * X_tensor.grad.sign()
            X_adv = X_adv.detach()
            
            with torch.no_grad():
                adv_preds = model(X_adv).squeeze().cpu().numpy()
                if len(adv_preds.shape) > 1 and adv_preds.shape[-1] == 3:
                    adv_preds = (adv_preds[:, 2] - adv_preds[:, 0]) * 0.01
            adv_mse = mean_squared_error(y_test, adv_preds[:len(y_test)])
            mse_increase_pct = ((adv_mse - clean_mse) / clean_mse) * 100 if clean_mse > 0 else 0
            
            attack_results["FGSM"] = {
                "resistance": f"{max(0, 100 - abs(mse_increase_pct)):.1f}%",
                "improvement": f"{max(0, -mse_increase_pct):.2f}%",
                "robustness_score": f"{max(0, min(1.0, 1.0 - abs(mse_increase_pct)/100)):.2f}",
                "status": "Good" if abs(mse_increase_pct) < 5 else "Moderate" if abs(mse_increase_pct) < 15 else "Needs Improvement"
            }
            print(f"[ADVERSARIAL] FGSM: MSE increase = {mse_increase_pct:.2f}%")
        except Exception as e:
            print(f"[ADVERSARIAL] FGSM attack failed: {e}")
            attack_results["FGSM"] = {"resistance": "N/A", "improvement": "N/A", "robustness_score": "0.50", "status": "Error"}
        
        # PGD Attack
        try:
            print(f"[ADVERSARIAL] Testing PGD attack...")
            epsilon = 0.01
            alpha = 0.001
            num_iter = 10
            X_adv = X_tensor.clone().detach()
            for _ in range(num_iter):
                X_adv.requires_grad_(True)
                pred = model(X_adv).squeeze()
                if len(pred.shape) > 1 and pred.shape[-1] == 3:
                    loss = F.mse_loss(pred.mean(dim=-1), torch.FloatTensor(y_test).to(device))
                else:
                    loss = F.mse_loss(pred, torch.FloatTensor(y_test).to(device))
                model.zero_grad()
                loss.backward()
                X_adv = X_adv + alpha * X_adv.grad.sign()
                X_adv = torch.clamp(X_adv, X_tensor - epsilon, X_tensor + epsilon).detach()
            
            with torch.no_grad():
                adv_preds = model(X_adv).squeeze().cpu().numpy()
                if len(adv_preds.shape) > 1 and adv_preds.shape[-1] == 3:
                    adv_preds = (adv_preds[:, 2] - adv_preds[:, 0]) * 0.01
            adv_mse = mean_squared_error(y_test, adv_preds[:len(y_test)])
            mse_increase_pct = ((adv_mse - clean_mse) / clean_mse) * 100 if clean_mse > 0 else 0
            
            attack_results["PGD"] = {
                "resistance": f"{max(0, 100 - abs(mse_increase_pct)):.1f}%",
                "improvement": f"{max(0, -mse_increase_pct):.2f}%",
                "robustness_score": f"{max(0, min(1.0, 1.0 - abs(mse_increase_pct)/100)):.2f}",
                "status": "Good" if abs(mse_increase_pct) < 5 else "Moderate" if abs(mse_increase_pct) < 15 else "Needs Improvement"
            }
            print(f"[ADVERSARIAL] PGD: MSE increase = {mse_increase_pct:.2f}%")
        except Exception as e:
            print(f"[ADVERSARIAL] PGD attack failed: {e}")
            attack_results["PGD"] = {"resistance": "N/A", "improvement": "N/A", "robustness_score": "0.50", "status": "Error"}
        
        # BIM Attack (similar to PGD)
        try:
            print(f"[ADVERSARIAL] Testing BIM attack...")
            epsilon = 0.01
            alpha = 0.001
            num_iter = 10
            X_adv = X_tensor.clone().detach()
            for _ in range(num_iter):
                X_adv.requires_grad_(True)
                pred = model(X_adv).squeeze()
                if len(pred.shape) > 1 and pred.shape[-1] == 3:
                    loss = F.mse_loss(pred.mean(dim=-1), torch.FloatTensor(y_test).to(device))
                else:
                    loss = F.mse_loss(pred, torch.FloatTensor(y_test).to(device))
                model.zero_grad()
                loss.backward()
                X_adv = X_adv + alpha * X_adv.grad.sign()
                X_adv = torch.clamp(X_adv, X_tensor - epsilon, X_tensor + epsilon).detach()
            
            with torch.no_grad():
                adv_preds = model(X_adv).squeeze().cpu().numpy()
                if len(adv_preds.shape) > 1 and adv_preds.shape[-1] == 3:
                    adv_preds = (adv_preds[:, 2] - adv_preds[:, 0]) * 0.01
            adv_mse = mean_squared_error(y_test, adv_preds[:len(y_test)])
            mse_increase_pct = ((adv_mse - clean_mse) / clean_mse) * 100 if clean_mse > 0 else 0
            
            attack_results["BIM"] = {
                "resistance": f"{max(0, 100 - abs(mse_increase_pct)):.1f}%",
                "improvement": f"{max(0, -mse_increase_pct):.2f}%",
                "robustness_score": f"{max(0, min(1.0, 1.0 - abs(mse_increase_pct)/100)):.2f}",
                "status": "Good" if abs(mse_increase_pct) < 5 else "Moderate" if abs(mse_increase_pct) < 15 else "Needs Improvement"
            }
            print(f"[ADVERSARIAL] BIM: MSE increase = {mse_increase_pct:.2f}%")
        except Exception as e:
            print(f"[ADVERSARIAL] BIM attack failed: {e}")
            attack_results["BIM"] = {"resistance": "N/A", "improvement": "N/A", "robustness_score": "0.50", "status": "Error"}
        
        # C&W Attack (simplified)
        try:
            print(f"[ADVERSARIAL] Testing C&W attack...")
            c = 1.0
            max_iter = 50
            X_adv = X_tensor.clone().detach().requires_grad_(True)
            for _ in range(max_iter):
                pred = model(X_adv).squeeze()
                if len(pred.shape) > 1 and pred.shape[-1] == 3:
                    loss = F.mse_loss(pred.mean(dim=-1), torch.FloatTensor(y_test).to(device))
                else:
                    loss = F.mse_loss(pred, torch.FloatTensor(y_test).to(device))
                perturbation = (X_adv - X_tensor)
                total_loss = loss + c * perturbation.norm()
                model.zero_grad()
                total_loss.backward()
                X_adv = X_adv - 0.01 * X_adv.grad
                X_adv = X_adv.detach().requires_grad_(True)
            
            with torch.no_grad():
                adv_preds = model(X_adv).squeeze().cpu().numpy()
                if len(adv_preds.shape) > 1 and adv_preds.shape[-1] == 3:
                    adv_preds = (adv_preds[:, 2] - adv_preds[:, 0]) * 0.01
            adv_mse = mean_squared_error(y_test, adv_preds[:len(y_test)])
            mse_increase_pct = ((adv_mse - clean_mse) / clean_mse) * 100 if clean_mse > 0 else 0
            
            attack_results["C&W"] = {
                "resistance": f"{max(0, 100 - abs(mse_increase_pct)):.1f}%",
                "improvement": f"{max(0, -mse_increase_pct):.2f}%",
                "robustness_score": f"{max(0, min(1.0, 1.0 - abs(mse_increase_pct)/100)):.2f}",
                "status": "Good" if abs(mse_increase_pct) < 5 else "Moderate" if abs(mse_increase_pct) < 15 else "Needs Improvement"
            }
            print(f"[ADVERSARIAL] C&W: MSE increase = {mse_increase_pct:.2f}%")
        except Exception as e:
            print(f"[ADVERSARIAL] C&W attack failed: {e}")
            attack_results["C&W"] = {"resistance": "N/A", "improvement": "N/A", "robustness_score": "0.50", "status": "Error"}
        
        # DeepFool Attack
        try:
            print(f"[ADVERSARIAL] Testing DeepFool attack...")
            max_iter = 50
            overshoot = 0.02
            X_adv = X_tensor.clone().detach()
            y_initial = None
            with torch.no_grad():
                y_pred_initial = model(X_adv).squeeze()
                # For Q-values (3 actions), use max Q-value as the prediction
                if len(y_pred_initial.shape) > 1 and y_pred_initial.shape[-1] == 3:
                    y_initial = y_pred_initial.max(dim=-1)[0]  # Max Q-value across actions
                else:
                    y_initial = y_pred_initial
            
            y_test_tensor = torch.FloatTensor(y_test).to(device)
            
            for i in range(max_iter):
                X_adv.requires_grad_(True)
                y_pred = model(X_adv).squeeze()
                
                # Handle Q-values: use max Q-value for loss computation
                if len(y_pred.shape) > 1 and y_pred.shape[-1] == 3:
                    y_pred_scalar = y_pred.max(dim=-1)[0]  # Max Q-value
                    loss = F.mse_loss(y_pred_scalar, y_test_tensor)
                else:
                    y_pred_scalar = y_pred
                    loss = F.mse_loss(y_pred_scalar, y_test_tensor)
                
                model.zero_grad()
                loss.backward()
                
                if X_adv.grad is not None:
                    # Compute gradient norm: flatten to (batch, -1) then compute L2 norm per sample
                    grad_flat = X_adv.grad.view(X_adv.size(0), -1)  # (batch, seq_len * features)
                    grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True) + 1e-8  # (batch, 1)
                    
                    # Compute prediction difference (scalar per sample)
                    pred_diff = torch.abs(y_pred_scalar - y_test_tensor)  # (batch,)
                    
                    # Compute perturbation: (pred_diff / grad_norm) * gradient
                    # Reshape to match gradient dimensions
                    scale = (pred_diff / grad_norm.squeeze(-1)).view(-1, 1, 1)  # (batch, 1, 1)
                    perturbation = scale * X_adv.grad
                    
                    X_adv = X_adv + (1 + overshoot) * perturbation
                    X_adv = torch.clamp(X_adv, X_tensor.min(), X_tensor.max()).detach()
                else:
                    break
                
                # Check convergence
                with torch.no_grad():
                    y_new = model(X_adv).squeeze()
                    if len(y_new.shape) > 1 and y_new.shape[-1] == 3:
                        y_new = y_new.max(dim=-1)[0]
                    if torch.mean(torch.abs(y_new - y_initial)) > 0.1:
                        break
            
            with torch.no_grad():
                adv_preds = model(X_adv).squeeze().cpu().numpy()
                if len(adv_preds.shape) > 1 and adv_preds.shape[-1] == 3:
                    # Convert Q-values to scalar prediction: use max Q-value
                    adv_preds = adv_preds.max(axis=-1) * 0.01  # Scale appropriately
            adv_mse = mean_squared_error(y_test, adv_preds[:len(y_test)])
            mse_increase_pct = ((adv_mse - clean_mse) / clean_mse) * 100 if clean_mse > 0 else 0
            
            attack_results["DeepFool"] = {
                "resistance": f"{max(0, 100 - abs(mse_increase_pct)):.1f}%",
                "improvement": f"{max(0, -mse_increase_pct):.2f}%",
                "robustness_score": f"{max(0, min(1.0, 1.0 - abs(mse_increase_pct)/100)):.2f}",
                "status": "Good" if abs(mse_increase_pct) < 5 else "Moderate" if abs(mse_increase_pct) < 15 else "Needs Improvement"
            }
            print(f"[ADVERSARIAL] DeepFool: MSE increase = {mse_increase_pct:.2f}%")
        except Exception as e:
            print(f"[ADVERSARIAL] DeepFool attack failed: {e}")
            import traceback
            traceback.print_exc()
            attack_results["DeepFool"] = {"resistance": "N/A", "improvement": "N/A", "robustness_score": "0.50", "status": "Error"}
        
        return attack_results
    
    def forecast_all_models(self, ticker: str, feature_path: str, current_price: float, risk_level: str = "Medium", enable_sentiment: bool = True) -> Dict:
        """Generate next-day forecasts for all models (baseline_dqn, mha_dqn_clean, mha_dqn_robust) and compare with last actual price."""
        import numpy as np
        from datetime import datetime, timedelta
        
        model_forecasts = {}
        models_to_forecast = [
            ("baseline_dqn", self.model_dir / "dqn" / "latest.ckpt"),
            ("mha_dqn_clean", self.model_dir / "mha_dqn" / "clean.ckpt"),
            ("mha_dqn_robust", self.model_dir / "mha_dqn" / "adversarial.ckpt"),
        ]
        
        # Load features once
        try:
            from pathlib import Path
            feature_path_obj = Path(feature_path)
            if not feature_path_obj.exists():
                return {"success": False, "error": "Feature file not found", "model_forecasts": {}}
            
            features = pd.read_parquet(feature_path)
            if len(features) == 0:
                return {"success": False, "error": "No features available", "model_forecasts": {}}
            
            # Get last actual price from features (if available)
            if "close" in features.columns:
                last_actual_price = float(features["close"].iloc[-1])
            else:
                last_actual_price = current_price
            
            sequence_length = 20
            if len(features) < sequence_length:
                return {"success": False, "error": f"Insufficient data: need {sequence_length} days, have {len(features)}", "model_forecasts": {}}
            
            # Prepare normalized sequence for all models
            all_features_array = features.select_dtypes(include=[float, int]).values
            feature_means = np.mean(all_features_array, axis=0)
            feature_stds = np.std(all_features_array, axis=0) + 1e-8
            latest_sequence = features.iloc[-sequence_length:].select_dtypes(include=[float, int]).values
            latest_sequence_normalized = (latest_sequence - feature_means) / feature_stds
            latest_sequence_normalized = np.nan_to_num(latest_sequence_normalized, nan=0.0)
            
            # Get date info
            if isinstance(features.index, pd.DatetimeIndex):
                last_date = features.index[-1]
                last_data_date = last_date.strftime("%Y-%m-%d")
                forecast_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                last_data_date = None
                forecast_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Forecast for each model
            input_dim = len(features.select_dtypes(include=[float, int]).columns)
            sequence_length = 20
            print(f"[FORECAST] Using input_dim={input_dim}, sequence_length={sequence_length}")
            
            for model_name, model_path in models_to_forecast:
                try:
                    print(f"[FORECAST] Loading model {model_name} from {model_path}")
                    # Load model with proper input_dim and sequence_length
                    model = self._load_model(model_path, input_dim=input_dim, sequence_length=sequence_length)
                    if model is None:
                        # Model not available
                        model_forecasts[model_name] = {
                            "available": False,
                            "error": "Model not found or failed to load"
                        }
                        continue
                    
                    model.eval()
                    torch.manual_seed(42)
                    
                    # Prepare input
                    sequence_tensor = torch.FloatTensor(latest_sequence_normalized).unsqueeze(0)
                    
                    # Get Q-values
                    with torch.no_grad():
                        if hasattr(model, "q_network"):
                            q_values, attention_weights = model.q_network(sequence_tensor)
                        elif hasattr(model, "forward"):
                            output = model(sequence_tensor)
                            if isinstance(output, tuple):
                                q_values, attention_weights = output
                            else:
                                q_values = output
                                attention_weights = None
                        else:
                            model_forecasts[model_name] = {"available": False, "error": "Invalid model structure"}
                            continue
                    
                    # Extract Q-values
                    if isinstance(q_values, torch.Tensor):
                        q_values_np = q_values.cpu().numpy().flatten()
                    else:
                        q_values_np = np.array(q_values).flatten()
                    
                    # Get predicted action
                    predicted_action_idx = np.argmax(q_values_np)
                    actions = ["SELL", "HOLD", "BUY"]
                    predicted_action = actions[predicted_action_idx] if predicted_action_idx < len(actions) else "HOLD"
                    
                    # Normalize Q-values for confidence
                    q_values_normalized = np.exp(q_values_np - np.max(q_values_np))
                    q_values_normalized = q_values_normalized / q_values_normalized.sum()
                    
                    # Estimate price change from Q-values
                    base_daily_return = (q_values_normalized[2] * 0.02) + (q_values_normalized[1] * 0.0) - (q_values_normalized[0] * 0.02)
                    forecasted_price = last_actual_price * (1 + base_daily_return)
                    
                    # Validate and adjust forecasted price
                    forecasted_price = ModelInferenceWork._validate_forecasted_price(forecasted_price, last_actual_price)
                    price_change_pct = ((forecasted_price - last_actual_price) / last_actual_price) * 100
                    
                    # Compare with last actual price to determine recommendation
                    # React to ANY change, even very small margins (threshold = 0)
                    price_diff_pct = ((forecasted_price - last_actual_price) / last_actual_price) * 100
                    
                    if price_diff_pct > 0:  # Any positive change = BUY
                        recommendation = "BUY"
                        confidence = float(q_values_normalized[2])
                    elif price_diff_pct < 0:  # Any negative change = SELL
                        recommendation = "SELL"
                        confidence = float(q_values_normalized[0])
                    else:  # Exactly 0% change = HOLD
                        recommendation = "HOLD"
                        confidence = float(q_values_normalized[1])
                    
                    # Adjust confidence based on risk level (aggressive = higher confidence needed)
                    if risk_level == "Low":
                        confidence *= 0.9  # Lower confidence threshold for conservative
                    elif risk_level == "High":
                        confidence *= 1.1  # Higher confidence for aggressive (but cap at 1.0)
                        confidence = min(confidence, 1.0)
                    
                    # Get explainability (only if actual model)
                    explainability = {
                        "q_values": q_values_np.tolist(),
                        "action_confidence": {
                            "BUY": float(q_values_normalized[2]) if len(q_values_normalized) > 2 else 0.0,
                            "HOLD": float(q_values_normalized[1]) if len(q_values_normalized) > 1 else 0.0,
                            "SELL": float(q_values_normalized[0]) if len(q_values_normalized) > 0 else 0.0,
                        },
                        "predicted_action": predicted_action,
                    }
                    
                    # Generate explainability text only for actual models
                    # Get date ranges for explainability
                    if isinstance(features.index, pd.DatetimeIndex):
                        first_date_str = features.index[0].strftime('%Y-%m-%d')
                        last_date_str = features.index[-1].strftime('%Y-%m-%d')
                        sequence_length = 20  # Default sequence length
                        sequence_start_str = features.index[-sequence_length].strftime('%Y-%m-%d') if len(features) >= sequence_length else first_date_str
                        sequence_end_str = last_date_str
                        full_range_str = f"{first_date_str} to {last_date_str} ({len(features)} days, ~{len(features)/252:.1f} years)"
                        sequence_range_str = f"{sequence_start_str} to {sequence_end_str} ({sequence_length} days)"
                        last_data_date_str = last_date_str
                    else:
                        full_range_str = f"{len(features)} days (~{len(features)/252:.1f} years)"
                        sequence_range_str = f"Last 20 days"
                        last_data_date_str = last_data_date if last_data_date else None
                    
                    forecast_date_str = forecast_date if forecast_date else "next trading day"
                    
                    try:
                        from lightning_app.utils.llm_summarizer import ModelResultsSummarizer
                        from lightning_app.config import OPENAI_API_KEY
                        if ModelResultsSummarizer and OPENAI_API_KEY:
                            summarizer = ModelResultsSummarizer(api_key=OPENAI_API_KEY)
                            forecast_data_for_llm = {
                                "q_values": q_values_np.tolist(),
                                "predicted_action": predicted_action,
                                "confidence_score": float(confidence),
                                "forecasted_price": float(forecasted_price),
                                "current_price": float(last_actual_price),
                                "expected_return_pct": float(price_change_pct / 100),
                                "total_data_points": len(features),
                                "full_data_range": full_range_str,
                                "data_range": sequence_range_str,
                                "forecast_date": forecast_date_str,
                                "last_data_date": last_data_date_str,
                            }
                            explainability["explainability_text"] = summarizer.generate_forecast_explainability(
                                ticker=ticker,
                                forecast_data=forecast_data_for_llm,
                                model_type=model_name.replace("_", " ").title()
                            )
                    except Exception as e:
                        # Fallback explainability
                        date_info = f" for {forecast_date_str} (next trading day after {last_data_date_str if last_data_date_str else 'last data point'})" if forecast_date_str and last_data_date_str else ""
                        explainability["explainability_text"] = f"The {model_name.replace('_', ' ').title()} model predicted {recommendation}{date_info} with {confidence:.1%} confidence. Forecasted price: ${forecasted_price:.2f} ({price_change_pct:+.2f}% change from last actual price of ${last_actual_price:.2f})."
                    
                    model_forecasts[model_name] = {
                        "available": True,
                        "forecasted_price": float(forecasted_price),
                        "last_actual_price": float(last_actual_price),
                        "price_change_pct": float(price_change_pct),
                        "price_diff_pct": float(price_diff_pct),
                        "recommendation": recommendation,
                        "confidence": float(confidence),
                        "forecast_date": forecast_date,
                        "q_values": q_values_np.tolist(),
                        "explainability": explainability,
                        "model_type": model_name.replace("_", " ").title(),
                    }
                    
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Error forecasting with {model_name}: {e}")
                    traceback.print_exc()
                    model_forecasts[model_name] = {
                        "available": False,
                        "error": str(e)
                    }
            
            return {
                "success": True,
                "last_actual_price": float(last_actual_price),
                "last_data_date": last_data_date if 'last_data_date' in locals() else None,
                "forecast_date": forecast_date,
                "model_forecasts": model_forecasts,
                "total_data_points": len(features),
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "model_forecasts": {}}
    
    def forecast_multiple_horizons(self, ticker: str, feature_path: str, current_price: float, horizons: list = [1, 5, 10], risk_level: str = "Medium", enable_sentiment: bool = True) -> Dict:
        """Forecast prices for multiple horizons (1, 5, 10 days) using MHA-DQN robust model and provide Buy/Sell/Hold recommendations."""
        import numpy as np
        from datetime import datetime, timedelta
        
        try:
            print(f"📂 Loading features from: {feature_path}")
            print(f"   → Input feature_path parameter: {repr(feature_path)}")
            print(f"   → Type: {type(feature_path)}")
            
            # Check if file exists
            from pathlib import Path
            feature_path_obj = Path(feature_path)
            print(f"   → Path object: {feature_path_obj}")
            print(f"   → Absolute path: {feature_path_obj.absolute()}")
            print(f"   → File exists: {feature_path_obj.exists()}")
            
            if not feature_path_obj.exists():
                print(f"⚠️  ERROR: Feature file does not exist at {feature_path_obj.absolute()}")
                print(f"   → This means features were not generated. Check feature engineering step.")
                print(f"   → Searching for similar files...")
                # Try to find if file exists with different case or in parent directory
                parent_dir = feature_path_obj.parent
                if parent_dir.exists():
                    similar_files = list(parent_dir.glob(f"*{feature_path_obj.stem}*"))
                    print(f"   → Found similar files in {parent_dir}: {[f.name for f in similar_files]}")
                features = pd.DataFrame()
            else:
                print(f"✅ Feature file exists at: {feature_path_obj.absolute()}")
                file_size = feature_path_obj.stat().st_size
                print(f"   → File size: {file_size} bytes")
                
                try:
                    features = pd.read_parquet(feature_path)
                    print(f"   → Successfully read parquet file")
                    
                    if len(features) == 0:
                        print(f"⚠️  WARNING: Feature file is empty (0 rows) at {feature_path}")
                        print(f"   → File exists but has no data!")
                        print(f"   → This means all rows were dropped during feature engineering")
                    else:
                        print(f"🔥 Features loaded successfully: {len(features)} rows, {len(features.columns)} columns")
                        print(f"   → Column names: {list(features.columns)[:10]}...")  # Show first 10 columns
                        print(f"   → Index type: {type(features.index)}")
                        if isinstance(features.index, pd.DatetimeIndex):
                            print(f"   → Date range: {features.index.min()} to {features.index.max()}")
                except Exception as read_error:
                    print(f"⚠️  ERROR reading feature file: {read_error}")
                    import traceback
                    traceback.print_exc()
                    features = pd.DataFrame()
            
            print(f"📊 Total data available after loading: {len(features)} days")
            
            # CRITICAL: Store features count for later verification
            features_row_count = len(features)
            
            # CRITICAL: Verify features are not empty before proceeding
            if features_row_count == 0:
                print(f"⚠️  CRITICAL: Features DataFrame is EMPTY after loading!")
                print(f"   → Re-attempting to load from: {feature_path_obj.absolute()}")
                try:
                    features = pd.read_parquet(feature_path)
                    features_row_count = len(features)
                    print(f"   → Successfully reloaded features: {features_row_count} rows")
                except Exception as reload_err:
                    print(f"   → Reload failed: {reload_err}")
                    print(f"   → Will proceed with empty DataFrame - forecast will show 'No data available'")
            else:
                print(f"   → ⚡ Features loaded: {features_row_count} rows - proceeding with forecast")
            
            # Ensure features are sorted by date (if date index exists)
            if len(features) > 0:
                if isinstance(features.index, pd.DatetimeIndex):
                    features = features.sort_index()
                    last_date = features.index[-1]
                elif 'date' in features.columns:
                    features['date'] = pd.to_datetime(features['date'], errors='coerce')
                    features = features.sort_values('date')
                    last_date = features['date'].iloc[-1]
                elif 'Date' in features.columns:
                    features['Date'] = pd.to_datetime(features['Date'], errors='coerce')
                    features = features.sort_values('Date')
                    last_date = features['Date'].iloc[-1]
                else:
                    last_date = None
            else:
                last_date = None
                print(f"⚠️  Cannot sort features - DataFrame is empty")
            
            # Log total date range (only if features are available)
            if len(features) > 0:
                if isinstance(features.index, pd.DatetimeIndex):
                    first_date = features.index[0]
                    last_date = features.index[-1]
                    print(f"📅 Full dataset range: {first_date.date()} to {last_date.date()} ({len(features)} days = ~{len(features)/252:.1f} years)")
                elif last_date is not None:
                    first_date_col = 'date' if 'date' in features.columns else 'Date'
                    first_date = pd.to_datetime(features[first_date_col].iloc[0])
                    print(f"📅 Full dataset range: {first_date.date()} to {last_date.date()} ({len(features)} days = ~{len(features)/252:.1f} years)")
            else:
                print(f"⚠️  Cannot log date range - features DataFrame is empty")
            
            # Get robust model
            robust_model_path = self.model_dir / "mha_dqn" / "adversarial.ckpt"
            print(f"🔍 Looking for model at: {robust_model_path}")
            print(f"   → Absolute path: {robust_model_path.absolute()}")
            print(f"   → Model directory exists: {self.model_dir.exists()}")
            print(f"   → MHA-DQN directory exists: {(self.model_dir / 'mha_dqn').exists()}")
            
            # Check if model directory exists and list files
            if (self.model_dir / "mha_dqn").exists():
                model_files = list((self.model_dir / "mha_dqn").glob("*.ckpt"))
                print(f"   → Available model files: {[f.name for f in model_files]}")
            else:
                print(f"   → MHA-DQN directory does not exist")
            
            model = self._load_model(robust_model_path)
            
            if model is None:
                print(f"⚠️  MODEL NOT FOUND or FAILED TO LOAD at {robust_model_path.absolute()}")
                print(f"   → Current features DataFrame has: {len(features)} rows")
                print(f"   → Using MOCK forecast (NOT real model)")
                print(f"   → NOTE: Model checkpoints appear to be placeholders (DummyModel)")
                print(f"   → To use real model, train and save actual model checkpoints")
                print(f"   → Model type: MOCK (deterministic based on historical trends)")
                
                # CRITICAL: Ensure features are available before mock forecast
                # Features should be available, but reload if somehow lost
                if len(features) == 0:
                    print(f"⚠️  CRITICAL: Features DataFrame is EMPTY when falling back to mock!")
                    print(f"   → This should NOT happen - features were verified to have {features_row_count} rows earlier")
                    print(f"   → Attempting emergency reload from: {feature_path}")
                    try:
                        features_reload = pd.read_parquet(feature_path)
                        print(f"   → ⚡ Emergency reload successful: {len(features_reload)} rows")
                        features = features_reload
                    except Exception as reload_error:
                        print(f"   → 🔴 Emergency reload failed: {reload_error}")
                        print(f"   → Will use empty DataFrame (forecast will show 'No data available')")
                else:
                    print(f"   → 🔥 Features available: {len(features)} rows - will use for mock forecast")
                
                # Generate mock forecast if model not available - uses historical data if available
                print(f"   → Calling mock forecast with {len(features)} rows of features...")
                return ModelInferenceWork._generate_mock_forecast_multiple_horizons(features, current_price, ticker, horizons)
            
            print(f"⚡ REAL MODEL loaded successfully from {robust_model_path}")
            print(f"   → Model type: REAL MHA-DQN (Adversarially Robust)")
            print(f"   → Using REAL model for deterministic predictions")
            
            # Set model to evaluation mode for deterministic inference
            model.eval()
            # Set random seeds for reproducibility (if using dropout or other stochastic layers)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            
            # Prepare latest sequence for forecasting
            sequence_length = 20  # Standard sequence length for MHA-DQN
            if len(features) < sequence_length:
                print(f"⚠️  Warning: Only {len(features)} days of data available, need at least {sequence_length} for forecasting")
                return ModelInferenceWork._generate_mock_forecast_multiple_horizons(features, current_price, ticker, horizons)
            
            # Use ALL 5-year data for normalization context (important for proper scaling)
            # This ensures the model sees features normalized consistently with training data
            all_features_array = features.select_dtypes(include=[float, int]).values
            
            # Calculate normalization statistics from ALL 5 years of data
            feature_means = np.mean(all_features_array, axis=0)
            feature_stds = np.std(all_features_array, axis=0) + 1e-8  # Avoid division by zero
            
            print(f"[29] Normalization statistics calculated from {len(features)} days of historical data")
            print(f"[30] Feature means range: [{feature_means.min():.2f}, {feature_means.max():.2f}]")
            print(f"[31] Feature stds range: [{feature_stds.min():.2f}, {feature_stds.max():.2f}]")
            
            # Get last N days (sequence_length) of features for model input
            latest_sequence = features.iloc[-sequence_length:].select_dtypes(include=[float, int]).values
            
            # Normalize using 5-year statistics (not just last 20 days)
            latest_sequence_normalized = (latest_sequence - feature_means) / feature_stds
            
            # Log what dates are being used for the sequence input
            if isinstance(features.index, pd.DatetimeIndex):
                date_range = features.index[-sequence_length:]
                sequence_date_range = f"{date_range[0].date()} to {date_range[-1].date()}"
                forecast_start_date = (pd.Timestamp(date_range[-1]) + pd.Timedelta(days=1)).date()
                print(f"[32] Sequence input: {sequence_date_range} ({len(date_range)} days)")
                print(f"[33] Forecasting from: {date_range[-1].date()} -> Next trading day")
            elif 'date' in features.columns or 'Date' in features.columns:
                date_col = 'date' if 'date' in features.columns else 'Date'
                date_range = features.iloc[-sequence_length:][date_col]
                sequence_date_range = f"{pd.to_datetime(date_range.iloc[0]).date()} to {pd.to_datetime(date_range.iloc[-1]).date()}"
                last_date_ts = pd.to_datetime(date_range.iloc[-1])
                forecast_start_date = (last_date_ts + pd.Timedelta(days=1)).date()
                print(f"[32] Sequence input: {sequence_date_range} ({len(date_range)} days)")
                print(f"[33] Forecasting from: {pd.to_datetime(date_range.iloc[-1]).date()} -> Next trading day")
            else:
                sequence_date_range = f"Last {sequence_length} rows"
                forecast_start_date = (datetime.now() + timedelta(days=1)).date()
                print(f"[32] Sequence input: {sequence_date_range} (assuming daily data)")
            
            print(f"[34] Input sequence shape: {latest_sequence_normalized.shape} ({latest_sequence_normalized.shape[0]} days x {latest_sequence_normalized.shape[1]} features)")
            print(f"[35] Model will analyze last {sequence_length} days, informed by {len(features)} days of historical patterns")
            
            # Prepare input tensor (using normalized sequence)
            sequence_tensor = torch.FloatTensor(latest_sequence_normalized).unsqueeze(0)
            
            # Get model prediction
            years = len(features) / 252.0
            print(f"[36] Running model inference on {latest_sequence_normalized.shape[0]} days of data...")
            print(f"[37] Model informed by {len(features)} days (~{years:.2f} years) of historical patterns")
            with torch.no_grad():
                if hasattr(model, "q_network"):
                    print("[38] Using model.q_network() for prediction")
                    q_values, attention_weights = model.q_network(sequence_tensor)
                    print(f"[39] Model returned Q-values shape: {q_values.shape if isinstance(q_values, torch.Tensor) else len(q_values)}")
                elif hasattr(model, "forward"):
                    print("[38] Using model.forward() for prediction")
                    output = model(sequence_tensor)
                    if isinstance(output, tuple):
                        q_values, attention_weights = output
                        print(f"[39] Model returned Q-values shape: {q_values.shape if isinstance(q_values, torch.Tensor) else len(q_values)}")
                    else:
                        q_values = output
                        attention_weights = None
                        print(f"[39] Model returned Q-values shape: {q_values.shape if isinstance(q_values, torch.Tensor) else len(q_values)}")
                else:
                    print("[WARNING] Model doesn't have expected structure (q_network or forward method)")
                    return ModelInferenceWork._generate_mock_forecast_multiple_horizons(features, current_price, ticker, horizons)
            
            # Extract Q-values for actions (0: Sell, 1: Hold, 2: Buy)
            if isinstance(q_values, torch.Tensor):
                q_values_np = q_values.cpu().numpy().flatten()
            else:
                q_values_np = np.array(q_values).flatten()
            
            # Get predicted action (highest Q-value)
            predicted_action_idx = np.argmax(q_values_np)
            actions = ["SELL", "HOLD", "BUY"]
            predicted_action = actions[predicted_action_idx] if predicted_action_idx < len(actions) else "HOLD"
            
            # Calculate forecasted price change based on Q-values
            # Normalize Q-values to get probability-like scores
            q_values_normalized = np.exp(q_values_np - np.max(q_values_np))
            q_values_normalized = q_values_normalized / q_values_normalized.sum()
            
            # Estimate base 1-day price change: Buy (positive), Hold (neutral), Sell (negative)
            base_daily_return = (q_values_normalized[2] * 0.02) + (q_values_normalized[1] * 0.0) - (q_values_normalized[0] * 0.02)
            
            # Generate forecasts for all horizons
            horizon_forecasts = {}
            
            for horizon_days in sorted(horizons):
                # For longer horizons, apply uncertainty discounting (uncertainty increases with time)
                # Use compound return with decreasing confidence: (1 + r)^n * confidence_factor
                uncertainty_factor = 1.0 / (1.0 + (horizon_days - 1) * 0.15)  # 15% discount per day
                horizon_return = base_daily_return * horizon_days * uncertainty_factor
                
                # Compound return for longer horizons (but discounted for uncertainty)
                if horizon_days > 1:
                    # Use compound interest formula: (1 + daily_return)^days
                    # But apply uncertainty discount
                    daily_return_discounted = base_daily_return * uncertainty_factor
                    horizon_return = ((1 + daily_return_discounted) ** horizon_days) - 1
                
                forecasted_price_horizon = current_price * (1 + horizon_return)
                
                # Validate and adjust forecasted price
                forecasted_price_horizon = ModelInferenceWork._validate_forecasted_price(forecasted_price_horizon, current_price)
                price_change_pct_horizon = ((forecasted_price_horizon - current_price) / current_price) * 100
                
                # Determine recommendation for this horizon
                # React to ANY change, even very small margins (no threshold)
                price_diff_pct_horizon = price_change_pct_horizon
                
                if price_diff_pct_horizon > 0:  # Any positive change = BUY
                    recommendation_horizon = "BUY"
                    confidence_horizon = float(q_values_normalized[2]) * uncertainty_factor
                elif price_diff_pct_horizon < 0:  # Any negative change = SELL
                    recommendation_horizon = "SELL"
                    confidence_horizon = float(q_values_normalized[0]) * uncertainty_factor
                else:  # Exactly 0% change = HOLD
                    recommendation_horizon = "HOLD"
                    confidence_horizon = float(q_values_normalized[1]) * uncertainty_factor
                
                # Calculate forecast date for this horizon
                if 'forecast_start_date' in locals():
                    forecast_date_horizon = (pd.Timestamp(forecast_start_date) + pd.Timedelta(days=horizon_days-1)).date().strftime("%Y-%m-%d")
                else:
                    forecast_date_horizon = (datetime.now() + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
                
                horizon_forecasts[horizon_days] = {
                    "forecasted_price": round(float(forecasted_price_horizon), 2),
                    "current_price": round(float(current_price), 2),
                    "price_change_pct": round(float(price_change_pct_horizon), 2),
                    "recommendation": recommendation_horizon,
                    "confidence": round(float(confidence_horizon), 2),
                    "forecast_date": forecast_date_horizon,
                    "horizon_days": horizon_days,
                    "expected_return": round(float(horizon_return), 2),
                }
            
            # For backward compatibility, keep the 1-day forecast at root level
            forecast_1day = horizon_forecasts.get(1, {})
            forecasted_price = forecast_1day.get("forecasted_price", current_price)
            price_change_pct = forecast_1day.get("price_change_pct", 0)
            recommendation = forecast_1day.get("recommendation", "HOLD")
            confidence = forecast_1day.get("confidence", 0.5)
            
            # Extract attention weights if available
            attention_info = None
            if attention_weights is not None:
                if isinstance(attention_weights, torch.Tensor):
                    attention_info = attention_weights.cpu().numpy().tolist()
                else:
                    attention_info = attention_weights.tolist() if hasattr(attention_weights, 'tolist') else None
            
            # Determine recommendation based on forecast
            # React to ANY change, even very small margins (no threshold)
            price_diff_pct = ((forecasted_price - current_price) / current_price) * 100
            
            if price_diff_pct > 0:  # Any positive change = BUY
                recommendation = "BUY"
                confidence = float(q_values_normalized[2])
            elif price_diff_pct < 0:  # Any negative change = SELL
                recommendation = "SELL"
                confidence = float(q_values_normalized[0])
            else:  # Exactly 0% change = HOLD
                recommendation = "HOLD"
                confidence = float(q_values_normalized[1])
            
            # Adjust confidence based on risk level
            if risk_level == "Low":
                confidence *= 0.9  # Lower confidence threshold for conservative
            elif risk_level == "High":
                confidence *= 1.1  # Higher confidence for aggressive (but cap at 1.0)
                confidence = min(confidence, 1.0)
            
            # Get feature importance from attention weights if available
            feature_importance = None
            if attention_info and len(attention_info) > 0:
                # Average attention across heads if multi-head
                if isinstance(attention_info[0], list):
                    feature_importance = np.mean(attention_info, axis=0).tolist()
                else:
                    feature_importance = attention_info
            
            # Calculate model explainability metrics (with text explanation)
            # Get actual date ranges for explainability
            if isinstance(features.index, pd.DatetimeIndex):
                first_date_str = features.index[0].strftime('%Y-%m-%d')
                last_date_str = features.index[-1].strftime('%Y-%m-%d')
                sequence_start_str = features.index[-sequence_length].strftime('%Y-%m-%d') if len(features) >= sequence_length else first_date_str
                sequence_end_str = last_date_str
                forecast_date_str = forecast_start_date.strftime('%Y-%m-%d') if 'forecast_start_date' in locals() else (pd.Timestamp(last_date_str) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                full_range_str = f"{first_date_str} to {last_date_str} ({len(features)} days, ~{len(features)/252:.1f} years)"
                sequence_range_str = f"{sequence_start_str} to {sequence_end_str} ({sequence_length} days)"
            else:
                full_range_str = f"{len(features)} days (~{len(features)/252:.1f} years)"
                sequence_range_str = f"Last {sequence_length} days"
                forecast_date_str = forecast_start_date.strftime('%Y-%m-%d') if 'forecast_start_date' in locals() else 'next trading day'
            
            # Get Q-values for explainability (format: [SELL, HOLD, BUY])
            q_vals_str = f"SELL: {q_values_np[0]:.2f}, HOLD: {q_values_np[1]:.2f}, BUY: {q_values_np[2]:.2f}" if len(q_values_np) >= 3 else "N/A"
            
            # Prepare forecast data for OpenAI explainability generation
            years = len(features) / 252.0
            forecast_data_for_llm = {
                "q_values": [round(float(v), 2) for v in q_values_np.tolist()],
                "predicted_action": predicted_action,
                "confidence_score": round(float(confidence), 2),
                "forecasted_price": round(float(forecasted_price), 2),
                "current_price": round(float(current_price), 2),
                "expected_return_pct": round(float(price_change_pct / 100), 2),
                "full_data_range": full_range_str,
                "data_range": sequence_range_str,
                "total_data_points": len(features),
                "forecast_date": forecast_date_str,
                "last_data_date": last_date_str if 'last_date_str' in locals() else None,
                "horizon_forecasts": horizon_forecasts,  # Include all horizon forecasts
            }
            
            # Generate explainability text using OpenAI (will fallback to template if unavailable)
            try:
                from ..utils.llm_summarizer import ModelResultsSummarizer
                from ..config import OPENAI_API_KEY
                
                summarizer = ModelResultsSummarizer(api_key=OPENAI_API_KEY)
                explainability_text = summarizer.generate_forecast_explainability(
                    ticker=ticker,
                    forecast_data=forecast_data_for_llm,
                    model_type="MHA-DQN (Adversarially Robust)"
                )
                print(f"[40] Generated explainability text using OpenAI API")
            except Exception as e:
                print(f"[WARNING] OpenAI explainability generation failed: {e}, using template")
                # Fallback to template-based explainability
                explainability_text = f"The MHA-DQN (Adversarially Robust) model utilized {full_range_str} of historical data to learn market patterns and contextualize recent trends. The model analyzed the sequence from {sequence_range_str} as input, informed by the full {len(features)} days (~{years:.2f} years) dataset. The model predicted {predicted_action} for {forecast_date_str} (next trading day after {last_date_str if 'last_date_str' in locals() else 'last data point'}) with {round(confidence*100, 2):.2f}% confidence. Q-values: [{q_vals_str}]. The model's Q-values suggest that {predicted_action} is the optimal action based on expected future returns. The forecasted price of ${round(forecasted_price, 2):.2f} represents a {round(price_change_pct, 2):+.2f}% change from the current price of ${round(current_price, 2):.2f}."
            
            explainability = {
                "explainability_text": explainability_text,  # Added text explanation
                "action_confidence": {
                    "BUY": round(float(q_values_normalized[2]), 2) if len(q_values_normalized) > 2 else 0.0,
                    "HOLD": round(float(q_values_normalized[1]), 2) if len(q_values_normalized) > 1 else 0.0,
                    "SELL": round(float(q_values_normalized[0]), 2) if len(q_values_normalized) > 0 else 0.0,
                },
                "predicted_action": predicted_action,
                "q_values": [round(float(v), 2) for v in q_values_np.tolist()],
                "expected_return_pct": round(float(price_change_pct), 2),
                "confidence_score": round(float(confidence), 2),
                "feature_importance": feature_importance,
            }
            
            # Get date range info for explainability
            sequence_range_info = f"Last {sequence_length} days"
            full_range_info = f"{len(features)} days (~{len(features)/252:.1f} years)"
            last_date = None
            
            if isinstance(features.index, pd.DatetimeIndex):
                date_range = features.index[-sequence_length:]
                sequence_range_info = f"{date_range[0].strftime('%Y-%m-%d')} to {date_range[-1].strftime('%Y-%m-%d')}"
                full_range_info = f"{features.index[0].strftime('%Y-%m-%d')} to {features.index[-1].strftime('%Y-%m-%d')} ({len(features)} days)"
                last_date = features.index[-1]
            elif 'date' in features.columns or 'Date' in features.columns:
                date_col = 'date' if 'date' in features.columns else 'Date'
                date_range = features.iloc[-sequence_length:][date_col]
                sequence_range_info = f"{pd.to_datetime(date_range.iloc[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(date_range.iloc[-1]).strftime('%Y-%m-%d')}"
                first_date = pd.to_datetime(features[date_col].iloc[0])
                last_date = pd.to_datetime(features[date_col].iloc[-1])
                full_range_info = f"{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')} ({len(features)} days)"
            
            # Use the actual last date + 1 day for forecast date (not today + 1)
            if 'forecast_start_date' in locals():
                forecast_date_str = forecast_start_date.strftime("%Y-%m-%d")
            elif last_date is not None:
                forecast_date_str = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                forecast_date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Get last_data_date string for return
            if last_date is not None:
                if isinstance(last_date, pd.Timestamp):
                    last_data_date_str = last_date.strftime("%Y-%m-%d")
                else:
                    last_data_date_str = pd.to_datetime(last_date).strftime("%Y-%m-%d")
            else:
                last_data_date_str = None
            
            print(f"[COMPLETE] Forecast complete: Predicting for {forecast_date_str}")
            
            return {
                "success": True,
                "forecasted_price": round(float(forecasted_price), 2),  # 1-day forecast (backward compatibility)
                "current_price": round(float(current_price), 2),
                "price_change_pct": round(float(price_change_pct), 2),  # 1-day change (backward compatibility)
                "recommendation": recommendation,  # 1-day recommendation (backward compatibility)
                "confidence": round(float(confidence), 2),  # 1-day confidence (backward compatibility)
                "explainability": explainability,
                "forecast_date": forecast_date_str,  # 1-day forecast date (backward compatibility)
                "model_type": "MHA-DQN (Adversarially Robust)",
                "data_range": sequence_range_info,
                "full_data_range": full_range_info,
                "total_data_points": len(features),
                "sequence_length": sequence_length,
                "last_data_date": last_data_date_str,
                "horizon_forecasts": horizon_forecasts,  # All horizon forecasts
            }
            
        except Exception as e:
            print(f"⚠️  FORECAST ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Try to get features if not in scope
            try:
                print(f"📂 Attempting to reload features from: {feature_path}")
                from pathlib import Path
                feature_path_obj = Path(feature_path)
                if feature_path_obj.exists():
                    features = pd.read_parquet(feature_path)
                    print(f"📊 Successfully reloaded features from exception handler: {len(features)} rows, {len(features.columns)} columns")
                else:
                    print(f"⚠️  Feature file does not exist at {feature_path_obj.absolute()}")
                    features = pd.DataFrame()
            except Exception as load_error:
                print(f"⚠️  ERROR loading features from {feature_path}: {load_error}")
                import pandas as pd
                features = pd.DataFrame()
            
            print(f"⚠️  Falling back to MOCK forecast due to exception above")
            print(f"   → Features DataFrame has {len(features)} rows (should be > 0 for real data)")
            return ModelInferenceWork._generate_mock_forecast_multiple_horizons(features, current_price, ticker, horizons)
    
    @staticmethod
    def _generate_mock_forecast_multiple_horizons(features: pd.DataFrame, current_price: float, ticker: str, horizons: list = [1, 5, 10]) -> Dict:
        """Generate mock forecasts for multiple horizons when model is not available - uses 5-year historical patterns."""
        import numpy as np
        from datetime import datetime, timedelta
        
        # Use 5-year historical data for better forecast
        total_days = len(features)
        if total_days == 0:
            print(f"⚠️  ERROR: Features DataFrame is empty (0 rows).")
            print(f"   → This means the feature file has no data!")
            print(f"   → Possible causes:")
            print(f"     1. Feature file doesn't exist or is empty")
            print(f"     2. All rows were dropped during feature engineering (dropna removed all)")
            print(f"     3. Not enough price data (need at least 50 days for SMA_50)")
            print(f"     4. Price data is invalid/all NaN")
            print(f"   → Check the feature engineering step output above")
            print(f"   → Using neutral forecast (0% return) as fallback")
            # Use a default dataset range when features are empty
            total_days = 0
            full_range_str = "No data available"
            sequence_range_str = "Last 20 days (mock)"
            last_date_str = None
        else:
            print(f"📊 Generating mock forecast using {total_days} days (~{total_days/252:.1f} years) of historical data")
            
            # Get actual date ranges for explainability
            if isinstance(features.index, pd.DatetimeIndex):
                first_date_str = features.index[0].strftime('%Y-%m-%d')
                last_date_str = features.index[-1].strftime('%Y-%m-%d')
                sequence_length_mock = min(20, len(features))
                sequence_start_str = features.index[-sequence_length_mock].strftime('%Y-%m-%d') if len(features) >= sequence_length_mock else first_date_str
                sequence_end_str = last_date_str
                full_range_str = f"{first_date_str} to {last_date_str} ({total_days} days, ~{total_days/252:.1f} years)"
                sequence_range_str = f"{sequence_start_str} to {sequence_end_str} ({sequence_length_mock} days)"
            elif 'date' in features.columns or 'Date' in features.columns:
                date_col = 'date' if 'date' in features.columns else 'Date'
                first_date = pd.to_datetime(features[date_col].iloc[0])
                last_date = pd.to_datetime(features[date_col].iloc[-1])
                first_date_str = first_date.strftime('%Y-%m-%d')
                last_date_str = last_date.strftime('%Y-%m-%d')
                sequence_length_mock = min(20, len(features))
                sequence_start = pd.to_datetime(features[date_col].iloc[-sequence_length_mock]) if len(features) >= sequence_length_mock else first_date
                sequence_end = last_date
                full_range_str = f"{first_date_str} to {last_date_str} ({total_days} days, ~{total_days/252:.1f} years)"
                sequence_range_str = f"{sequence_start.strftime('%Y-%m-%d')} to {sequence_end.strftime('%Y-%m-%d')} ({sequence_length_mock} days)"
            else:
                full_range_str = f"{total_days} days (~{total_days/252:.1f} years)"
                sequence_range_str = f"Last {min(20, total_days)} days"
                last_date_str = None
        
        # Calculate trend from different time horizons using ONLY actual data (no random values)
        expected_return = 0.0  # Default to neutral if no data available
        
        # Try to use return column first (most reliable)
        if "return" in features.columns and len(features) > 5:
            # Get actual returns, dropping NaN values
            returns = features["return"].dropna()
            if len(returns) > 0:
                # Short-term momentum (last 5 days) - use whatever data is available
                if len(returns) >= 5:
                    short_term = returns.tail(5).mean()
                else:
                    short_term = returns.mean()
                
                # Medium-term trend (last 20 days) - use available data
                if len(returns) >= 20:
                    medium_term = returns.tail(20).mean()
                else:
                    medium_term = returns.mean()
                
                # Long-term trend (last 252 days = 1 year) - use available data
                if len(returns) >= 252:
                    long_term = returns.tail(252).mean()
                elif len(returns) > 0:
                    long_term = returns.mean()
                else:
                    long_term = 0.0
                
                # Weighted average: more weight to short-term, but consider long-term context
                # Only use this if we have actual return data
                expected_return = (short_term * 0.5) + (medium_term * 0.3) + (long_term * 0.2)
                
                print(f"   → Short-term trend (last {min(5, len(returns))}d): {short_term*100:.2f}%")
                print(f"   → Medium-term trend (last {min(20, len(returns))}d): {medium_term*100:.2f}%")
                print(f"   → Long-term trend (last {min(252, len(returns))}d): {long_term*100:.2f}%")
                print(f"   → Combined expected return (from actual data): {expected_return*100:.2f}%")
        else:
            # No return column - try to calculate from price changes using actual numeric columns
            numeric_cols = features.select_dtypes(include=[float, int]).columns
            
            if len(numeric_cols) > 0 and len(features) > 1:
                # Try to find a price-like column (close, adjusted close, etc.)
                price_col = None
                for col in ['close', '5. adjusted close', 'adjusted close', 'Close', 'close_price']:
                    if col in features.columns:
                        price_col = col
                        break
                
                # If no price column found, use first numeric column as proxy
                if price_col is None:
                    price_col = numeric_cols[0]
                
                if price_col and len(features) > 1:
                    prices = features[price_col].dropna()
                    if len(prices) > 1:
                        # Calculate actual returns from price changes
                        price_changes = prices.pct_change().dropna()
                        if len(price_changes) > 0:
                            # Use last 5, 20, or all available for trends
                            if len(price_changes) >= 5:
                                short_term = price_changes.tail(5).mean()
                            else:
                                short_term = price_changes.mean()
                            
                            if len(price_changes) >= 20:
                                medium_term = price_changes.tail(20).mean()
                            else:
                                medium_term = price_changes.mean()
                            
                            expected_return = (short_term * 0.5) + (medium_term * 0.5)
                            print(f"   → Calculated from actual price changes in '{price_col}' column")
                            print(f"   → Expected return (from actual data): {expected_return*100:.2f}%")
                        else:
                            print(f"   ⚠️  No valid price changes calculated, using neutral (0.0%)")
                    else:
                        print(f"   ⚠️  Insufficient price data, using neutral (0.0%)")
                else:
                    print(f"   ⚠️  No price column found, using neutral (0.0%)")
            else:
                print(f"   ⚠️  No numeric data available, using neutral (0.0%)")
        
        # Base daily return (calculated from historical data)
        base_daily_return = expected_return
        
        # Get last_date if available
        last_date = None
        if isinstance(features.index, pd.DatetimeIndex):
            last_date = features.index[-1]
        elif 'date' in features.columns:
            last_date = pd.to_datetime(features['date'].iloc[-1])
        elif 'Date' in features.columns:
            last_date = pd.to_datetime(features['Date'].iloc[-1])
        
        # Generate forecasts for all horizons
        horizon_forecasts = {}
        
        for horizon_days in sorted(horizons):
            # For longer horizons, apply uncertainty discounting
            uncertainty_factor = 1.0 / (1.0 + (horizon_days - 1) * 0.15)  # 15% discount per day
            
            # Compound return for longer horizons (but discounted for uncertainty)
            if horizon_days > 1:
                daily_return_discounted = base_daily_return * uncertainty_factor
                horizon_return = ((1 + daily_return_discounted) ** horizon_days) - 1
            else:
                horizon_return = base_daily_return
            
            forecasted_price_horizon = current_price * (1 + horizon_return)
            
            # Validate and adjust forecasted price
            forecasted_price_horizon = ModelInferenceWork._validate_forecasted_price(forecasted_price_horizon, current_price)
            price_change_pct_horizon = ((forecasted_price_horizon - current_price) / current_price) * 100
            
            # Determine recommendation for this horizon
            # React to ANY change, even very small margins (no threshold)
            price_diff_pct_horizon_mock = price_change_pct_horizon
            
            if price_diff_pct_horizon_mock > 0:  # Any positive change = BUY
                recommendation_horizon = "BUY"
                confidence_horizon = 0.65 * uncertainty_factor
            elif price_diff_pct_horizon_mock < 0:  # Any negative change = SELL
                recommendation_horizon = "SELL"
                confidence_horizon = 0.65 * uncertainty_factor
            else:  # Exactly 0% change = HOLD
                recommendation_horizon = "HOLD"
                confidence_horizon = 0.70 * uncertainty_factor
            
            # Calculate forecast date for this horizon
            forecast_date_horizon = (datetime.now() + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
            
            horizon_forecasts[horizon_days] = {
                "forecasted_price": float(forecasted_price_horizon),
                "current_price": float(current_price),
                "price_change_pct": float(price_change_pct_horizon),
                "recommendation": recommendation_horizon,
                "confidence": float(confidence_horizon),
                "forecast_date": forecast_date_horizon,
                "horizon_days": horizon_days,
                "expected_return": float(horizon_return),
            }
        
        # For backward compatibility, keep the 1-day forecast at root level
        forecast_1day = horizon_forecasts.get(1, {})
        forecasted_price = forecast_1day.get("forecasted_price", current_price)
        price_change_pct = forecast_1day.get("price_change_pct", 0)
        recommendation = forecast_1day.get("recommendation", "HOLD")
        confidence = forecast_1day.get("confidence", 0.5)
        
        # Mock Q-values (based on recommendation)
        q_values = np.array([0.3, 0.4, 0.5]) if recommendation == "BUY" else \
                  (np.array([0.5, 0.4, 0.3]) if recommendation == "SELL" else np.array([0.3, 0.5, 0.3]))
        
        # Generate explainability text with actual data ranges using OpenAI if available
        forecast_data_for_llm_mock = {
            "q_values": q_values.tolist(),
            "predicted_action": recommendation,
            "confidence_score": float(confidence),
            "forecasted_price": float(forecasted_price),
            "current_price": float(current_price),
            "expected_return_pct": float(price_change_pct / 100),
            "horizon_forecasts": horizon_forecasts,
        }
        
        if total_days > 0:
            # Calculate forecast date
            if last_date_str:
                forecast_date_mock = (pd.Timestamp(last_date_str) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                forecast_date_mock = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Update data_range and full_data_range with actual dates
            data_range_mock = sequence_range_str
            full_data_range_mock = full_range_str
            
            forecast_data_for_llm_mock.update({
                "full_data_range": full_range_str,
                "data_range": sequence_range_str,
                "total_data_points": total_days,
                "forecast_date": forecast_date_mock,
                "last_data_date": last_date_str if last_date_str else None,
            })
            
            # Try to generate OpenAI explainability text
            try:
                from ..utils.llm_summarizer import ModelResultsSummarizer
                from ..config import OPENAI_API_KEY
                
                summarizer = ModelResultsSummarizer(api_key=OPENAI_API_KEY)
                explainability_text_mock = summarizer.generate_forecast_explainability(
                    ticker=ticker,
                    forecast_data=forecast_data_for_llm_mock,
                    model_type="MHA-DQN (Adversarially Robust) - Mock"
                )
                print(f"   → Generated mock explainability text using OpenAI API")
            except Exception as e:
                print(f"   → ⚠️  OpenAI explainability generation failed for mock: {e}, using template")
                q_vals_str = f"SELL: {q_values[0]:.3f}, HOLD: {q_values[1]:.3f}, BUY: {q_values[2]:.3f}"
                explainability_text_mock = f"The MHA-DQN (Adversarially Robust) - Mock model utilized {full_range_str} of historical data to learn market patterns and contextualize recent trends. The model analyzed the sequence from {sequence_range_str} as input, informed by the full {total_days} days (~{total_days/252:.1f} years) dataset. The model predicted {recommendation} for {forecast_date_mock} (next trading day after {last_date_str if last_date_str else 'last data point'}) with {confidence*100:.1f}% confidence. Q-values: [{q_vals_str}]. The model's Q-values suggest that {recommendation} is the optimal action based on expected future returns. The forecasted price of ${forecasted_price:.2f} represents a {price_change_pct:+.2f}% change from the current price of ${current_price:.2f}."
        else:
            data_range_mock = "Last 20 days (mock)"
            full_data_range_mock = "No data available (mock)"
            forecast_date_mock = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            forecast_data_for_llm_mock.update({
                "full_data_range": full_data_range_mock,
                "data_range": data_range_mock,
                "total_data_points": 0,
                "forecast_date": forecast_date_mock,
                "last_data_date": None,
            })
            
            # Try to generate OpenAI explainability text even with no data
            try:
                from ..utils.llm_summarizer import ModelResultsSummarizer
                from ..config import OPENAI_API_KEY
                
                summarizer = ModelResultsSummarizer(api_key=OPENAI_API_KEY)
                explainability_text_mock = summarizer.generate_forecast_explainability(
                    ticker=ticker,
                    forecast_data=forecast_data_for_llm_mock,
                    model_type="MHA-DQN (Adversarially Robust) - Mock (No Data)"
                )
                print(f"   → Generated mock explainability text using OpenAI API (no data case)")
            except Exception as e:
                print(f"   → ⚠️  OpenAI explainability generation failed for mock (no data): {e}, using template")
                explainability_text_mock = f"The MHA-DQN (Adversarially Robust) - Mock model could not utilize historical data (0 trading days) as no data is available. The model analyzed the sequence from Last 20 days (mock) as input. The model predicted {recommendation} for {forecast_date_mock} (next trading day) with {confidence*100:.1f}% confidence. Q-values: [SELL: {q_values[0]:.3f}, HOLD: {q_values[1]:.3f}, BUY: {q_values[2]:.3f}]. The forecasted price of ${forecasted_price:.2f} represents a {price_change_pct:+.2f}% change from the current price of ${current_price:.2f}."
        
        return {
            "success": True,
            "forecasted_price": float(forecasted_price),  # 1-day forecast (backward compatibility)
            "current_price": float(current_price),
            "price_change_pct": float(price_change_pct),  # 1-day change (backward compatibility)
            "recommendation": recommendation,  # 1-day recommendation (backward compatibility)
            "confidence": float(confidence),  # 1-day confidence (backward compatibility)
            "explainability": {
                "explainability_text": explainability_text_mock,  # Add explainability text with actual data
                "action_confidence": {
                    "BUY": float(q_values[2]),
                    "HOLD": float(q_values[1]),
                    "SELL": float(q_values[0]),
                },
                "predicted_action": recommendation,
                "q_values": q_values.tolist(),
                "expected_return_pct": float(price_change_pct),
                "confidence_score": float(confidence),
                "feature_importance": None,
                "note": "Mock forecast (model not available)" if total_days > 0 else "Mock forecast (no data available)",
            },
            "forecast_date": forecast_date_mock,  # 1-day forecast date (backward compatibility)
            "model_type": "MHA-DQN (Adversarially Robust) - Mock" if total_days > 0 else "MHA-DQN (Adversarially Robust) - Mock (No Data)",
            "data_range": data_range_mock,
            "full_data_range": full_data_range_mock,
            "total_data_points": total_days,
            "sequence_length": 20,
            "last_data_date": last_date_str if total_days > 0 else None,
            "horizon_forecasts": horizon_forecasts,  # All horizon forecasts
        }
